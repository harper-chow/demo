"""Hybrid retrieval and business-layer ranking."""

from __future__ import annotations

import logging
from typing import Any

import config
from .data_process import clean_category_name
from .embedder import LocalEmbedder
from .vector_store import ChromaVectorStore


LOGGER = logging.getLogger(__name__)


def normalize_query(query: str) -> str | None:
	return clean_category_name(query)


class Retriever:
	def __init__(self, store: ChromaVectorStore | None = None, embedder: LocalEmbedder | None = None) -> None:
		self.embedder = embedder or LocalEmbedder()
		self.store = store or ChromaVectorStore(self.embedder)

	def search(
		self,
		query: str,
		top_k: int = config.DEFAULT_TOP_K,
		top_n: int = config.DEFAULT_TOP_N,
		vector_weight: float | None = None,
		normalized_query: str | None = None,
	) -> list[dict[str, Any]]:
		if not query or not query.strip():
			return []
		if not 1 <= top_k <= config.MAX_TOP_K:
			raise ValueError(f"top_k must be between 1 and {config.MAX_TOP_K}")
		if not 1 <= top_n <= config.MAX_TOP_N:
			raise ValueError(f"top_n must be between 1 and {config.MAX_TOP_N}")
		if vector_weight is not None and not 0.0 <= vector_weight <= 1.0:
			raise ValueError("vector_weight must be between 0 and 1")

		query_text = normalize_query(query) if normalized_query is None else normalized_query
		if query_text is None:
			return []
		query_embedding = self.embedder.encode(query_text)
		candidates = self.store.hybrid_search(query_text, query_embedding, top_k, vector_weight=vector_weight)
		if not candidates:
			return []

		results = []
		for candidate in candidates:
			metadata = candidate.get("metadata") or {}
			try:
				count = max(1, int(metadata.get("count", 1)))
			except (TypeError, ValueError):
				count = 1
			rrf_score = float(candidate.get("rrf_score", 0.0))
			results.append({
				"category_name": candidate.get("document"),
				"rrf_score": rrf_score,
				"count": count,
				"company_code": metadata.get("company_code"),
				"price": metadata.get("price"),
				"currency": metadata.get("currency"),
				"category": metadata.get("category"),
				"category_id": metadata.get("category_id"),
				"init_text": metadata.get("init_text"),
			})
		category_scores: dict[str | None, float] = {}
		for item in results:
			category_name = item["category_name"]
			category_scores[category_name] = max(category_scores.get(category_name, 0.0), item["rrf_score"])
		results.sort(key=lambda item: (category_scores[item["category_name"]], item["count"]), reverse=True)
		return results[:top_n]
