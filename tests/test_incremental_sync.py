from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import config
from services.incremental_sync import IncrementalSyncService, SyncResult
from services.sync_checkpoint import source_fingerprint


class FakeBM25:
	def __init__(
		self,
		watermark: datetime,
		boundary: dict[str, int],
		documents: dict[str, dict] | None = None,
	) -> None:
		self.documents = documents or {
			"rec-1": {
				"id": "rec-1",
				"document": "ITEM",
				"metadata": {
					"company_code": "9200",
					"price": 1.0,
					"category_id": "A1",
					"category": "BB",
					"currency": "RMB",
					"count": 2,
					"init_text": "Item",
				},
			}
		}
		self.sync_state = {
			"last_synced_at": watermark.isoformat(),
			"boundary_fingerprints": boundary,
		}

	def update_metadata(self, record_id: str, metadata: dict) -> None:
		self.documents[record_id]["metadata"] = metadata

	def add(self, record_id: str, document: str, metadata: dict) -> None:
		self.documents[record_id] = {"id": record_id, "document": document, "metadata": metadata}


class FakeEmbedder:
	def encode(self, documents: list[str]) -> list[list[float]]:
		return [[0.1] for _ in documents]


class FakeStore:
	def __init__(self, bm25: FakeBM25) -> None:
		self.dense_collection = object()
		self.bm25 = bm25
		self.bm25_index_path = Path("fake-bm25.pkl")
		self.embedder = None
		self.upserted_ids: list[str] = []
		self.updated_ids: list[str] = []
		self.dense_ids = set(bm25.documents)

	def update_metadata(self, ids: list[str], metadatas: list[dict]) -> None:
		self.updated_ids.extend(ids)

	def upsert(self, **kwargs) -> None:
		self.upserted_ids.extend(kwargs["ids"])
		self.dense_ids.update(kwargs["ids"])

	def validate_consistency(self) -> None:
		pass

	def persist_bm25(self) -> None:
		pass

	def close(self) -> None:
		pass


class IncrementalSyncTests(unittest.TestCase):
	def test_updates_aggregate_count_and_upserts_new_group_once(self) -> None:
		watermark = datetime(2026, 10, 29, tzinfo=timezone.utc)
		existing_row = {
			"company_code": "9200",
			"category_name": "Item",
			"price": "1.0",
			"category": "BB",
			"category_id": "A1",
			"currency": "RMB",
			"createtime": watermark,
		}
		new_group = {**existing_row, "category_id": "A2"}
		dataframe = pd.DataFrame([existing_row, existing_row, new_group])
		bm25 = FakeBM25(watermark, {source_fingerprint(existing_row): 1})
		store = FakeStore(bm25)

		with TemporaryDirectory() as directory:
			bm25_path = Path(directory) / "bm25.pkl"
			bm25_path.touch()
			with (
				patch("services.incremental_sync.ChromaVectorStore", return_value=store),
				patch("services.incremental_sync.load_data", return_value=dataframe),
				patch("services.incremental_sync.write_jsonl"),
			):
				store.bm25_index_path = bm25_path
				indexed_rows = IncrementalSyncService().sync(FakeEmbedder())

		self.assertEqual(
			indexed_rows,
			SyncResult(
				changed=True,
				source_rows=2,
				indexed_rows=2,
				last_synced_at=watermark.isoformat(),
			),
		)
		self.assertEqual(bm25.documents["rec-1"]["metadata"]["count"], 3)
		self.assertEqual(store.updated_ids, ["rec-1"])
		self.assertEqual(len(store.upserted_ids), 1)
		self.assertEqual(len(bm25.documents), 2)

	def test_last_five_cleaned_rows_are_incrementally_indexed(self) -> None:
		if not config.CLEANED_JSONL_PATH.is_file():
			self.skipTest("Requires the current cleaned.jsonl snapshot")
		with config.CLEANED_JSONL_PATH.open(encoding="utf-8") as file:
			records = [json.loads(line) for line in file if line.strip()]
		if len(records) < 6:
			self.skipTest("Requires at least six cleaned records")

		watermark = datetime(2026, 10, 29, tzinfo=timezone.utc)
		existing_records = records[:-5]
		new_records = records[-5:]
		existing_documents = {
			f"rec-{index}": {
				"id": f"rec-{index}",
				"document": record["category_name"],
				"metadata": {
					key: record[key]
					for key in ("company_code", "price", "currency", "category", "category_id", "count", "init_text")
					if key in record and record[key] is not None
				},
			}
			for index, record in enumerate(existing_records, start=1)
		}
		bm25 = FakeBM25(watermark, {}, existing_documents)
		store = FakeStore(bm25)
		new_timestamp = datetime(2026, 10, 30, tzinfo=timezone.utc)
		new_dataframe = pd.DataFrame([
			{
				"company_code": str(record["company_code"]),
				"category_name": record["category_name"],
				"price": str(record["price"]),
				"category": record["category"],
				"category_id": str(record["category_id"]),
				"currency": record["currency"],
				"createtime": new_timestamp,
			}
			for record in new_records
		])

		with TemporaryDirectory() as directory:
			bm25_path = Path(directory) / "bm25.pkl"
			bm25_path.touch()
			with (
				patch("services.incremental_sync.ChromaVectorStore", return_value=store),
				patch("services.incremental_sync.load_data", return_value=new_dataframe),
				patch("services.incremental_sync.write_jsonl"),
			):
				store.bm25_index_path = bm25_path
				indexed_rows = IncrementalSyncService().sync(FakeEmbedder())

		self.assertEqual(
			indexed_rows,
			SyncResult(
				changed=True,
				source_rows=5,
				indexed_rows=5,
				last_synced_at=new_timestamp.isoformat(),
			),
		)
		self.assertEqual(len(store.upserted_ids), 5)
		self.assertEqual(len(store.updated_ids), 0)
		self.assertEqual(len(store.dense_ids), len(existing_records) + 5)


if __name__ == "__main__":
	unittest.main()