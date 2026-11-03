"""Timestamp checkpoint and duplicate-boundary helpers for database sync."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

import config


def normalize_timestamp(value: Any) -> datetime | None:
	if value is None or pd.isna(value):
		return None
	if isinstance(value, pd.Timestamp):
		return value.to_pydatetime()
	return value


def timestamp_text(value: datetime | None) -> str | None:
	return value.isoformat() if value is not None else None


def parse_timestamp(value: str | None) -> datetime | None:
	return datetime.fromisoformat(value) if value else None


def source_fingerprint(record: dict[str, Any]) -> str:
	values = []
	for column in (*config.DB_COLUMNS.keys(), config.DB_TIME_COLUMN):
		value = record.get(column)
		if value is None or pd.isna(value):
			values.append(None)
		elif isinstance(value, (datetime, pd.Timestamp)):
			values.append(normalize_timestamp(value).isoformat())
		else:
			values.append(str(value))
	serialized = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
	return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def boundary_fingerprints(dataframe: pd.DataFrame, watermark: datetime | None) -> dict[str, int]:
	if watermark is None or dataframe.empty:
		return {}
	counts: Counter[str] = Counter()
	for record in dataframe.to_dict(orient="records"):
		if normalize_timestamp(record.get(config.DB_TIME_COLUMN)) == watermark:
			counts[source_fingerprint(record)] += 1
	return dict(counts)


def select_unseen_rows(
	dataframe: pd.DataFrame,
	watermark: datetime,
	previous_boundary: dict[str, int],
) -> tuple[pd.DataFrame, datetime, dict[str, int]]:
	"""Select newer rows and only excess duplicates at the inclusive boundary."""
	if dataframe.empty:
		return dataframe.copy(), watermark, previous_boundary

	selected_indices: list[Any] = []
	boundary_seen: Counter[str] = Counter()
	timestamps: list[datetime] = []
	for index, record in dataframe.iterrows():
		timestamp = normalize_timestamp(record[config.DB_TIME_COLUMN])
		if timestamp is None:
			continue
		timestamps.append(timestamp)
		if timestamp > watermark:
			selected_indices.append(index)
		elif timestamp == watermark:
			fingerprint = source_fingerprint(record.to_dict())
			boundary_seen[fingerprint] += 1
			if boundary_seen[fingerprint] > previous_boundary.get(fingerprint, 0):
				selected_indices.append(index)

	new_watermark = max(timestamps, default=watermark)
	new_boundary = boundary_fingerprints(dataframe, new_watermark)
	return dataframe.loc[selected_indices].copy(), new_watermark, new_boundary