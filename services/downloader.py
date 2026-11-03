"""Download the embedding model into the project-local models directory."""

from __future__ import annotations

import logging
from pathlib import Path

import config


LOGGER = logging.getLogger(__name__)


def model_path(model_directory: Path | str = config.MODEL_DIRECTORY) -> Path:
	return Path(model_directory) / config.MODEL_NAME.rsplit("/", 1)[-1]


def download_model(force: bool = False) -> Path:
	"""Download and persist the configured model locally.

	This function is intentionally manual. It is not called by API startup.
	"""

	target = model_path()
	if target.exists() and any(target.iterdir()) and not force:
		LOGGER.info("Local embedding model already exists: %s", target)
		return target

	from sentence_transformers import SentenceTransformer

	config.MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
	LOGGER.info("Downloading embedding model %s to %s", config.MODEL_NAME, target)
	model = SentenceTransformer(config.MODEL_NAME)
	model.save(str(target))
	LOGGER.info("Embedding model saved: %s", target)
	return target


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
	download_model()
