"""Hybrid dense and BM25 ranking."""

from __future__ import annotations

import logging
from typing import Any

import config
from .bm25_index import BM25Index
from .chroma_store import ChromaStore


LOGGER = logging.getLogger(__name__)


class HybridSearch:
	"""Fuse dense and BM25 rankings using the existing RRF formula."""

	def __init__(self, chroma_store: ChromaStore, bm25: BM25Index) -> None:
		self.chroma_store = chroma_store
		self.bm25 = bm25

	@staticmethod
	def _rows(response: dict[str, Any]) -> list[dict[str, Any]]:
		ids = (response.get("ids") or [[]])[0]
		documents = (response.get("documents") or [[]])[0]
		metadatas = (response.get("metadatas") or [[]])[0]
		distances = (response.get("distances") or [[]])[0]
		return [
			{
				"id": record_id,
				"document": documents[index] if index < len(documents) else "",
				"metadata": metadatas[index] if index < len(metadatas) else {},
				"distance": distances[index] if index < len(distances) else None,
			}
			for index, record_id in enumerate(ids)
		]

	def search(
		self,
		query_text: str,
		query_embedding: list[float],
		top_k: int,
		vector_weight: float | None = None,
	) -> list[dict[str, Any]]:
		dense_weight = config.CHROMA_DENSE_WEIGHT if vector_weight is None else vector_weight
		sparse_weight = 1.0 - dense_weight
		dense_count = self.chroma_store.count()
		bm25_count = len(self.bm25.documents)
		if dense_weight > 0 and dense_count == 0:
			LOGGER.warning("Dense collection is empty; falling back to BM25-only search")
			dense_weight, sparse_weight = 0.0, 1.0
		if sparse_weight > 0 and bm25_count == 0:
			LOGGER.warning("BM25 index is empty; falling back to dense-only search")
			dense_weight, sparse_weight = 1.0, 0.0
		if dense_weight == 0.0 and sparse_weight == 0.0:
			return []
		dense_response = self.chroma_store.query(query_embedding, top_k) if dense_weight > 0 else {}
		bm25_response = self.bm25.search(query_text, top_k) if sparse_weight > 0 else []
		fused: dict[str, dict[str, Any]] = {}
		if dense_weight > 0:
			for rank, row in enumerate(self._rows(dense_response), start=1):
				fused.setdefault(row["id"], {**row, "rrf_score": 0.0})["rrf_score"] += dense_weight / (60 + rank)
		if sparse_weight > 0:
			for rank, row in enumerate(bm25_response, start=1):
				entry = fused.setdefault(row["id"], {**row, "rrf_score": 0.0})
				entry["rrf_score"] += sparse_weight / (60 + rank)
		return sorted(fused.values(), key=lambda row: row["rrf_score"], reverse=True)[:top_k]
