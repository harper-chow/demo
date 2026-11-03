"""Index construction orchestration."""

from __future__ import annotations

import json
import gc
import logging
import shutil
from pathlib import Path
from typing import Any

import config
from .bm25_index import BM25Index
from .chroma_store import ChromaStore
from .embedder import LocalEmbedder
from .index_validator import IndexValidator


LOGGER = logging.getLogger(__name__)
METADATA_FIELDS = ("company_code", "price", "currency", "category", "category_id", "count", "init_text")


class IndexBuilder:
	"""Build the dense and sparse indexes from cleaned JSONL records."""

	def __init__(
		self,
		chroma_store: ChromaStore,
		bm25: BM25Index,
		embedder: LocalEmbedder | None,
		bm25_index_path: Path | str | None = None,
	) -> None:
		self.chroma_store = chroma_store
		self.bm25 = bm25
		self.embedder = embedder
		self.bm25_index_path = Path(
			bm25_index_path if bm25_index_path is not None else config.get_bm25_index_path()
		)

	@staticmethod
	def _metadata(record: dict[str, Any]) -> dict[str, Any]:
		metadata: dict[str, Any] = {}
		for field in METADATA_FIELDS:
			value = record.get(field)
			if value is not None:
				metadata[field] = value
		return metadata

	def build_from_jsonl(self, jsonl_path: Path | str | None = None, batch_size: int = 256) -> int:
		if self.embedder is None:
			self.embedder = LocalEmbedder()
		jsonl = Path(jsonl_path if jsonl_path is not None else config.get_cleaned_jsonl_path())
		if not jsonl.is_file():
			raise FileNotFoundError(f"JSONL file does not exist: {jsonl}")
		self.chroma_store.close()
		gc.collect()
		if self.chroma_store.persist_directory.exists():
			shutil.rmtree(self.chroma_store.persist_directory)
		if self.chroma_store.persist_directory.exists():
			raise RuntimeError(
				f"Failed to clear Chroma persistence directory: {self.chroma_store.persist_directory}"
			)
		self.chroma_store.persist_directory.mkdir(parents=True, exist_ok=True)
		self.chroma_store.open_collection()
		self.bm25 = BM25Index()
		records: list[tuple[int, dict[str, Any]]] = []
		with jsonl.open(encoding="utf-8") as file:
			for line_number, line in enumerate(file, start=1):
				if line.strip():
					records.append((line_number, json.loads(line)))
		if not records:
			raise ValueError("Cleaned JSONL is empty; refusing to build an empty index")

		indexed_count = 0
		for start in range(0, len(records), batch_size):
			batch_with_lines = [(line, record) for line, record in records[start:start + batch_size] if record.get("category_name")]
			if not batch_with_lines:
				continue
			batch = [record for _, record in batch_with_lines]
			ids = [f"rec-{line_number}" for line_number, _ in batch_with_lines]
			documents = [record["category_name"] for record in batch]
			embeddings = self.embedder.encode(documents)
			metadatas = [self._metadata(record) for record in batch]
			self.chroma_store.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
			for record_id, document, metadata in zip(ids, documents, metadatas):
				self.bm25.add(record_id, document, metadata)
			indexed_count += len(batch)
		IndexValidator.validate(self.chroma_store, self.bm25)
		if indexed_count == 0:
			raise ValueError("No valid category_name records; refusing to build an empty index")
		self._save_bm25()
		LOGGER.info("Rebuilt indexes: dense=%d, bm25=%d", indexed_count, len(self.bm25.documents))
		return indexed_count

	def _save_bm25(self) -> None:
		self.bm25.save(self.bm25_index_path)
