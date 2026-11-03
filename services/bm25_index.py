"""BM25 sparse index implementation."""

from __future__ import annotations

import logging
import math
import pickle
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jieba

from .data_process import clean_category_name


LOGGER = logging.getLogger(__name__)


class BM25Index:
	"""BM25 index that can be persisted to the configured local index file."""

	def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
		self.k1 = k1
		self.b = b
		self.documents: dict[str, dict[str, Any]] = {}
		self.term_frequencies: dict[str, Counter[str]] = {}
		self.postings: dict[str, dict[str, int]] = defaultdict(dict)
		self.document_frequency: Counter[str] = Counter()
		self.average_document_length = 0.0
		self.sync_state: dict[str, Any] = {}

	_JUNK_TOKEN_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)
	_JUNK_TOKENS = {"NANA", "无"}

	@staticmethod
	def _tokens(text: Any) -> list[str]:
		cleaned = clean_category_name(text)
		if not cleaned:
			return []
		return [
			token for token in jieba.lcut(cleaned)
			if token not in BM25Index._JUNK_TOKENS and not BM25Index._JUNK_TOKEN_PATTERN.fullmatch(token)
		]

	def add(self, record_id: str, document: str, metadata: dict[str, Any]) -> None:
		tokens = self._tokens(document)
		if not tokens:
			return
		frequencies = Counter(tokens)
		self.documents[record_id] = {"id": record_id, "document": document, "metadata": metadata}
		self.term_frequencies[record_id] = frequencies
		for token in frequencies:
			self.postings[token][record_id] = frequencies[token]
		self.document_frequency.update(frequencies.keys())
		document_count = len(self.term_frequencies)
		self.average_document_length += (len(tokens) - self.average_document_length) / document_count

	def update_metadata(self, record_id: str, metadata: dict[str, Any]) -> None:
		if record_id not in self.documents:
			raise KeyError(f"BM25 record does not exist: {record_id}")
		self.documents[record_id]["metadata"] = metadata

	def save(self, index_path: Path | str) -> None:
		path = Path(index_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as temporary:
			temporary_path = Path(temporary.name)
			pickle.dump(self, temporary, protocol=pickle.HIGHEST_PROTOCOL)
			temporary.flush()
		try:
			temporary_path.replace(path)
		finally:
			temporary_path.unlink(missing_ok=True)

	def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
		query_tokens = set(self._tokens(query))
		if not query_tokens or not self.documents:
			return []
		scores: dict[str, float] = defaultdict(float)
		document_count = len(self.documents)
		for token in query_tokens:
			document_frequency = self.document_frequency.get(token, 0)
			if not document_frequency:
				continue
			idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
			for record_id, term_frequency in self.postings[token].items():
				document_length = sum(self.term_frequencies[record_id].values())
				normalizer = term_frequency + self.k1 * (1 - self.b + self.b * document_length / self.average_document_length)
				scores[record_id] += idf * term_frequency * (self.k1 + 1) / normalizer
		ordered_ids = sorted(
			(record_id for record_id, score in scores.items() if score > 0),
			key=lambda record_id: scores[record_id],
			reverse=True,
		)
		return [{**self.documents[record_id], "bm25_score": scores[record_id]} for record_id in ordered_ids[:top_k]]
