"""Compatibility facade for hybrid dense/BM25 retrieval."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import config
from .bm25_index import BM25Index
from .chroma_store import ChromaStore
from .embedder import LocalEmbedder
from .hybrid_search import HybridSearch
from .index_builder import IndexBuilder
from .index_validator import IndexValidator


LOGGER = logging.getLogger(__name__)
COLLECTION_NAME = f"{config.CHROMA_COLLECTION_NAME}_dense"
LocalBM25 = BM25Index


class ChromaVectorStore:
	"""Backward-compatible facade over the split index components."""

	def __init__(
		self,
		embedder: LocalEmbedder | None = None,
		allow_missing: bool = False,
		persist_directory: Path | str | None = None,
		bm25_index_path: Path | str | None = None,
	) -> None:
		self.embedder = embedder
		self.persist_directory = Path(
			persist_directory if persist_directory is not None else config.get_chroma_persist_directory()
		)
		self.bm25_index_path = Path(
			bm25_index_path if bm25_index_path is not None else config.get_bm25_index_path()
		)
		config.ensure_runtime_directories()
		self._chroma = ChromaStore(allow_missing=allow_missing, persist_directory=self.persist_directory)
		self.client = self._chroma.client
		self.dense_collection = self._chroma.dense_collection
		self.bm25 = self._load_bm25() if self.bm25_index_path.is_file() else LocalBM25()
		self._hybrid_search = HybridSearch(self._chroma, self.bm25)
		if self.dense_collection is not None:
			if not self.bm25_index_path.is_file() and not allow_missing:
				raise RuntimeError(f"BM25 index is missing: {self.bm25_index_path}")
			if not allow_missing:
				self.validate_consistency()

	def build_from_jsonl(
		self,
		jsonl_path: Path | str | None = None,
		batch_size: int = 256,
	) -> int:
		if jsonl_path is None:
			jsonl_path = config.get_cleaned_jsonl_path()
		builder = IndexBuilder(
			chroma_store=self._chroma,
			bm25=self.bm25,
			embedder=self.embedder,
			bm25_index_path=self.bm25_index_path,
		)
		indexed_count = builder.build_from_jsonl(jsonl_path, batch_size)
		self.embedder = builder.embedder
		self.bm25 = builder.bm25
		self.client = self._chroma.client
		self.dense_collection = self._chroma.dense_collection
		self._hybrid_search = HybridSearch(self._chroma, self.bm25)
		return indexed_count

	def upsert(
		self,
		*,
		ids: list[str],
		documents: list[str],
		embeddings: list[list[float]],
		metadatas: list[dict[str, Any]],
	) -> None:
		self._chroma.upsert(
			ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
		)

	def update_metadata(self, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
		self._chroma.update_metadata(ids, metadatas)

	def persist_bm25(self) -> None:
		self.bm25.save(self.bm25_index_path)

	def _load_bm25(self) -> LocalBM25:
		class _LegacyUnpickler(pickle.Unpickler):
			def find_class(self, module: str, name: str) -> Any:
				if module in {"src.vector_store", "src.bm25_index"} and name in {"LocalBM25", "BM25Index"}:
					return LocalBM25
				return super().find_class(module, name)

		try:
			with self.bm25_index_path.open("rb") as file:
				bm25 = _LegacyUnpickler(file).load()
		except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError, ValueError) as exc:
			raise RuntimeError(f"Failed to load BM25 index: {self.bm25_index_path}") from exc
		if not isinstance(bm25, LocalBM25):
			raise RuntimeError(f"Invalid BM25 index file: {self.bm25_index_path}")
		LOGGER.info("Loaded persisted BM25 index: documents=%d", len(bm25.documents))
		return bm25

	def validate_consistency(self) -> None:
		IndexValidator.validate(self._chroma, self.bm25)

	def close(self) -> None:
		self._chroma.close()

	def hybrid_search(
		self,
		query_text: str,
		query_embedding: list[float],
		top_k: int,
		vector_weight: float | None = None,
	) -> list[dict[str, Any]]:
		return self._hybrid_search.search(
			query_text, query_embedding, top_k, vector_weight=vector_weight
		)
