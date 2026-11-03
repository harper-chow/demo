"""Cleaning, normalization, aggregation, and JSONL serialization."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

import config


LOGGER = logging.getLogger(__name__)
DEDUPLICATION_FIELDS = ("company_code", "category_name", "price", "category_id", "category")
_VALID_CODE_PATTERN = re.compile(r"^[^\s]+$")
_VALID_PRICE_PATTERN = re.compile(r"^\+?(?:\d+(?:\.\d+)?|\.\d+)$")


def _is_missing(value: Any) -> bool:
	if value is None:
		return True
	try:
		return bool(pd.isna(value))
	except (TypeError, ValueError):
		return False


def clean_text(value: Any, *, uppercase: bool = False) -> str | None:
	"""Normalize a text value shared by ingestion and future query handling."""

	if _is_missing(value):
		return None
	text = str(value).strip()
	if not text:
		return None
	text = "".join(
		chr(ord(char) - 0xFEE0)
		if ("０" <= char <= "９") or ("Ａ" <= char <= "Ｚ") or ("ａ" <= char <= "ｚ")
		else char
		for char in text
	)
	text = re.sub(r"\s+", " ", text)
	return text.upper() if uppercase else text


def clean_category_name(value: Any) -> str | None:
	return clean_text(value, uppercase=True)


def clean_company_code(value: Any) -> str | None:
	if _is_missing(value):
		return None
	text = str(value).strip()
	if not text or not _VALID_CODE_PATTERN.fullmatch(text):
		return None
	return text


def clean_price(value: Any) -> float | None:
	if _is_missing(value):
		return None
	text = str(value).strip()
	if not _VALID_PRICE_PATTERN.fullmatch(text):
		return None
	try:
		number = float(text)
	except (TypeError, ValueError, OverflowError):
		return None
	return number if math.isfinite(number) and number >= 0 else None


def clean_record(raw_record: dict[str, Any]) -> dict[str, Any]:
	"""Normalize one source record using the shared light text cleaning rules."""

	cleaned_category_name = clean_category_name(raw_record.get("category_name"))
	initial_category_name = None if _is_missing(raw_record.get("category_name")) else str(raw_record.get("category_name"))

	return {
		"company_code": clean_company_code(raw_record.get("company_code")),
		"category_name": cleaned_category_name,
		"price": clean_price(raw_record.get("price")),
		"category": clean_text(raw_record.get("category")),
		"category_id": clean_text(raw_record.get("category_id")),
		"currency": clean_text(raw_record.get("currency")),
		"init_text": initial_category_name,
	}


def process_dataframe(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
	"""Clean, filter, and five-field aggregate an input dataframe."""

	missing_columns = [column for column in config.DB_COLUMNS if column not in dataframe.columns]
	if missing_columns:
		raise ValueError(f"Missing configured business columns: {missing_columns}")

	cleaned_records: list[dict[str, Any]] = []
	dropped_empty_reference = 0
	for raw_record in dataframe.to_dict(orient="records"):
		cleaned = clean_record(raw_record)
		if cleaned["category"] is None or cleaned["category_id"] is None:
			dropped_empty_reference += 1
			continue
		cleaned_records.append(cleaned)

	grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
	for record in cleaned_records:
		key = tuple(record[field] for field in DEDUPLICATION_FIELDS)
		if key not in grouped:
			grouped[key] = {**record, "count": 1}
		else:
			grouped[key]["count"] += 1

	LOGGER.info(
		"Processed rows: input=%d, retained=%d, dropped_empty_category_or_id=%d, unique=%d",
		len(dataframe), len(cleaned_records), dropped_empty_reference, len(grouped),
	)
	return list(grouped.values())


def write_jsonl(records: list[dict[str, Any]], output_path: Path | str) -> Path:
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("w", encoding="utf-8") as file:
		for record in records:
			file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
	LOGGER.info("Wrote cleaned JSONL: path=%s, records=%d", output, len(records))
	return output
