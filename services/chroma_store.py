"""Chroma persistence adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import config


LOGGER = logging.getLogger(__name__)
COLLECTION_NAME = f"{config.CHROMA_COLLECTION_NAME}_dense"


def _load_chromadb() -> Any:
	try:
		import chromadb
	except ImportError as exc:
		raise RuntimeError("chromadb is not installed in the active Retrieval environment") from exc
	return chromadb


class ChromaStore:
	"""Persistent Chroma collection adapter."""

	def __init__(
		self,
		allow_missing: bool = False,
		persist_directory: Path | str | None = None,
	) -> None:
		self.persist_directory = Path(
			persist_directory if persist_directory is not None else config.get_chroma_persist_directory()
		)
		if allow_missing:
			self.persist_directory.mkdir(parents=True, exist_ok=True)
		elif not self.persist_directory.is_dir():
			raise RuntimeError(f"Chroma persistence directory is missing: {self.persist_directory}")
		self.client = None
		self.dense_collection = None
		self._connect(allow_missing)

	def _connect(self, allow_missing: bool) -> None:
		self.client = _load_chromadb().PersistentClient(path=str(self.persist_directory))
		try:
			self.dense_collection = self.client.get_collection(name=COLLECTION_NAME)
		except Exception as exc:
			if not allow_missing:
				raise RuntimeError("Active Chroma dense collection is missing") from exc
			self.dense_collection = None

	def close(self) -> None:
		old_client = self.client
		self.client = None
		self.dense_collection = None
		if old_client is not None:
			old_client.close()

	def open_collection(self) -> None:
		self.client = _load_chromadb().PersistentClient(path=str(self.persist_directory))
		self.dense_collection = self.client.get_or_create_collection(
			name=COLLECTION_NAME,
			configuration={"hnsw": {"space": "cosine"}},
		)

	def add(
		self,
		*,
		ids: list[str],
		documents: list[str],
		embeddings: list[list[float]],
		metadatas: list[dict[str, Any]],
	) -> None:
		self.dense_collection.add(
			ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
		)

	def upsert(
		self,
		*,
		ids: list[str],
		documents: list[str],
		embeddings: list[list[float]],
		metadatas: list[dict[str, Any]],
	) -> None:
		self.dense_collection.upsert(
			ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
		)

	def update_metadata(self, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
		self.dense_collection.update(ids=ids, metadatas=metadatas)

	def count(self) -> int:
		return self.dense_collection.count()

	def query(self, query_embedding: list[float], top_k: int) -> dict[str, Any]:
		return self.dense_collection.query(
			query_embeddings=[query_embedding],
			n_results=min(top_k, self.count()),
			include=["documents", "metadatas", "distances"],
		)

	def documents(self, page_size: int = 1000) -> dict[str, str]:
		dense_documents: dict[str, str] = {}
		offset = 0
		while True:
			page = self.dense_collection.get(limit=page_size, offset=offset, include=["documents"])
			page_ids = page.get("ids", [])
			page_documents = page.get("documents", [])
			for index, record_id in enumerate(page_ids):
				dense_documents[record_id] = page_documents[index] if index < len(page_documents) else ""
			if len(page_ids) < page_size:
				break
			offset += page_size
		return dense_documents
