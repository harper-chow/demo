"""Incrementally index database rows created or updated since the last sync."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.data_loader import get_latest_createtime, load_data
from services.data_process import DEDUPLICATION_FIELDS, process_dataframe, write_jsonl
from services.sync_checkpoint import (
	boundary_fingerprints,
	normalize_timestamp,
	parse_timestamp,
	select_unseen_rows,
	timestamp_text,
)
from services.vector_store import ChromaVectorStore
from services.index_builder import IndexBuilder
from services.embedder import LocalEmbedder
import config


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncResult:
	changed: bool
	source_rows: int = 0
	indexed_rows: int = 0
	version: str | None = None
	last_synced_at: str | None = None


def _group_key(record: dict[str, Any]) -> str:
	values = [record.get(field) for field in DEDUPLICATION_FIELDS]
	return json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str)


def _group_id(key: str) -> str:
	return f"inc-{hashlib.sha256(key.encode('utf-8')).hexdigest()}"


def _existing_groups(store: ChromaVectorStore) -> dict[str, str]:
	groups: dict[str, str] = {}
	for record_id, entry in store.bm25.documents.items():
		metadata = entry.get("metadata", {})
		record = {
			"company_code": metadata.get("company_code"),
			"category_name": entry.get("document"),
			"price": metadata.get("price"),
			"category_id": metadata.get("category_id"),
			"category": metadata.get("category"),
		}
		groups.setdefault(_group_key(record), record_id)
	return groups


def _all_indexed_records(store: ChromaVectorStore) -> list[dict[str, Any]]:
	records = []
	for entry in store.bm25.documents.values():
		records.append({"category_name": entry["document"], **entry.get("metadata", {})})
	return records


class IncrementalSyncService:
	"""Keep the active dense and sparse indexes current without rebuilding them."""

	def initialize_checkpoint(self, bm25: Any, index_path: Path | str | None = None) -> bool:
		state = getattr(bm25, "sync_state", {}) or {}
		if state.get("last_synced_at"):
			return False

		watermark = get_latest_createtime()
		if watermark is None:
			watermark = datetime.now(timezone.utc)
			boundary = {}
		else:
			boundary_data = load_data(since=watermark)
			boundary = boundary_fingerprints(boundary_data, watermark)
		bm25.sync_state = {
			"last_synced_at": timestamp_text(watermark),
			"boundary_fingerprints": boundary,
		}
		bm25.save(index_path if index_path is not None else config.get_bm25_index_path())
		LOGGER.info("Initialized database sync checkpoint at %s", watermark.isoformat())
		return True

	def sync(
		self,
		embedder: LocalEmbedder | None = None,
		*,
		index_directory: Path | str | None = None,
	) -> SyncResult:
		index_root = Path(index_directory) if index_directory is not None else None
		store = ChromaVectorStore(
			embedder,
			allow_missing=False,
			persist_directory=index_root / "chromadb" if index_root else None,
			bm25_index_path=index_root / "bm25.pkl" if index_root else None,
		)
		try:
			if store.dense_collection is None or not store.bm25_index_path.is_file():
				raise RuntimeError("Active indexes are missing; run a manual full rebuild first")
			state = getattr(store.bm25, "sync_state", {}) or {}
			watermark = parse_timestamp(state.get("last_synced_at"))
			if watermark is None:
				raise RuntimeError("Database sync checkpoint is missing; restart the API or run a full rebuild")

			dataframe = load_data(since=watermark)
			new_rows, next_watermark, next_boundary = select_unseen_rows(
				dataframe,
				watermark,
				state.get("boundary_fingerprints", {}),
			)
			if dataframe.empty:
				return SyncResult(changed=False, last_synced_at=timestamp_text(watermark))

			records = process_dataframe(new_rows) if not new_rows.empty else []
			groups = _existing_groups(store)
			metadata_updates: list[tuple[str, dict[str, Any]]] = []
			new_records: list[tuple[str, dict[str, Any]]] = []
			for record in records:
				key = _group_key(record)
				record_id = groups.get(key)
				if record_id is None:
					record_id = _group_id(key)
					metadata = IndexBuilder._metadata(record)
					new_records.append((record_id, {**record, "metadata": metadata}))
					groups[key] = record_id
				else:
					metadata = dict(store.bm25.documents[record_id].get("metadata", {}))
					metadata["count"] = int(metadata.get("count", 1)) + int(record["count"])
					metadata_updates.append((record_id, metadata))

			if metadata_updates:
				ids, metadatas = zip(*metadata_updates)
				store.update_metadata(list(ids), list(metadatas))
				for record_id, metadata in metadata_updates:
					store.bm25.update_metadata(record_id, metadata)

			if new_records:
				if store.embedder is None:
					store.embedder = embedder or LocalEmbedder()
				ids = [record_id for record_id, _ in new_records]
				documents = [record["category_name"] for _, record in new_records]
				embeddings = store.embedder.encode(documents)
				metadatas = [record["metadata"] for _, record in new_records]
				store.upsert(
					ids=ids,
					documents=documents,
					embeddings=embeddings,
					metadatas=metadatas,
				)
				for record_id, record in new_records:
					store.bm25.add(record_id, record["category_name"], record["metadata"])

			if new_rows.empty:
				return SyncResult(changed=False, last_synced_at=timestamp_text(next_watermark))

			if records:
				cleaned_path = index_root / "cleaned.jsonl" if index_root else config.get_cleaned_jsonl_path()
				write_jsonl(_all_indexed_records(store), cleaned_path)
			store.validate_consistency()
			store.bm25.sync_state = {
				"last_synced_at": timestamp_text(next_watermark),
				"boundary_fingerprints": next_boundary,
			}
			store.persist_bm25()
			LOGGER.info(
				"Incremental database sync complete: source_rows=%d, indexed_groups=%d, watermark=%s",
				len(new_rows), len(records), next_watermark.isoformat(),
			)
			return SyncResult(
				changed=True,
				source_rows=len(new_rows),
				indexed_rows=sum(int(record["count"]) for record in records),
				last_synced_at=timestamp_text(next_watermark),
			)
		finally:
			store.close()