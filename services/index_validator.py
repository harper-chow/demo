"""Consistency validation for dense and sparse indexes."""

from __future__ import annotations

from typing import Any

from .bm25_index import BM25Index
from .chroma_store import ChromaStore


class IndexValidator:
	"""Validate that Chroma and BM25 contain the same documents."""

	@staticmethod
	def validate(chroma_store: ChromaStore, bm25: BM25Index) -> None:
		if chroma_store.dense_collection is None:
			raise RuntimeError("Dense collection is unavailable")
		dense_documents = chroma_store.documents()
		bm25_ids = set(bm25.documents)
		dense_ids = set(dense_documents)
		if dense_ids != bm25_ids:
			raise RuntimeError(
				f"Dense/BM25 index mismatch: dense={len(dense_ids)}, bm25={len(bm25_ids)}"
			)
		mismatched_documents = [
			record_id
			for record_id in dense_ids
			if dense_documents[record_id] != bm25.documents[record_id]["document"]
		]
		if mismatched_documents:
			raise RuntimeError(
				f"Dense/BM25 document mismatch for {len(mismatched_documents)} records"
			)
