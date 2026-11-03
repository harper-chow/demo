"""Local sentence-transformer wrapper for dense embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import config
from .downloader import model_path


LOGGER = logging.getLogger(__name__)


class LocalEmbedder:
	def __init__(self, model_directory: Path | str = config.MODEL_DIRECTORY) -> None:
		local_path = model_path(model_directory)
		if not local_path.is_dir():
			raise FileNotFoundError(
				f"Local embedding model is missing: {local_path}. Run python -m services.downloader first."
			)
		from sentence_transformers import SentenceTransformer

		LOGGER.info("Loading local embedding model: %s", local_path)
		self.model = SentenceTransformer(str(local_path), local_files_only=True)

	def encode(self, texts: str | Sequence[str]) -> list[float] | list[list[float]]:
		single = isinstance(texts, str)
		values = [texts] if single else list(texts)
		if not values:
			return []
		vectors = self.model.encode(
			values,
			normalize_embeddings=True,
			convert_to_numpy=True,
			show_progress_bar=False,
		)
		result = vectors.tolist()
		return result[0] if single else result
