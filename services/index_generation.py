"""Immutable index generations with an atomically switched active symlink."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

import config


LOGGER = logging.getLogger(__name__)


class IndexGenerationManager:
	def __init__(self) -> None:
		self.versions_directory = config.INDEX_VERSIONS_DIRECTORY
		self.active_link = config.ACTIVE_INDEX_SYMLINK

	def active_directory(self) -> Path | None:
		if self.active_link.is_symlink():
			return self.active_link.resolve(strict=True)
		if self.active_link.exists():
			return self.active_link
		return None

	def active_version(self) -> str | None:
		active = self.active_directory()
		return active.name if active is not None else None

	def create_staging(self, *, clone_active: bool) -> Path:
		self.versions_directory.mkdir(parents=True, exist_ok=True)
		staging = self.versions_directory / f".staging-{uuid.uuid4().hex}"
		if clone_active:
			active = self.active_directory()
			if active is None:
				raise RuntimeError("No active index exists to clone for incremental update")
			shutil.copytree(active, staging)
		else:
			staging.mkdir(parents=True, exist_ok=False)
		return staging

	def ensure_active(self, before_switch=None) -> Path | None:
		"""Seed a versioned active link from legacy artifacts once, if present."""
		active = self.active_directory()
		if active is not None:
			return active
		legacy_chroma = config.LEGACY_CHROMA_PERSIST_DIRECTORY
		legacy_bm25 = config.LEGACY_BM25_INDEX_PATH
		legacy_cleaned = config.LEGACY_CLEANED_JSONL_PATH
		if not legacy_chroma.is_dir() or not legacy_bm25.is_file():
			return None

		staging = self.create_staging(clone_active=False)
		try:
			shutil.copytree(legacy_chroma, staging / "chromadb")
			shutil.copy2(legacy_bm25, staging / "bm25.pkl")
			if legacy_cleaned.is_file():
				shutil.copy2(legacy_cleaned, staging / "cleaned.jsonl")
			else:
				(staging / "cleaned.jsonl").write_text("", encoding="utf-8")
			return self.publish(staging, before_switch=before_switch)
		except Exception:
			shutil.rmtree(staging, ignore_errors=True)
			raise

	def publish(self, staging: Path, *, before_switch=None) -> Path:
		for artifact in ("chromadb", "bm25.pkl", "cleaned.jsonl"):
			if not (staging / artifact).exists():
				raise RuntimeError(f"Staged index artifact is missing: {staging / artifact}")

		version = f"generation-{uuid.uuid4().hex}"
		final_directory = self.versions_directory / version
		staging.rename(final_directory)
		temporary_link = self.active_link.with_name(f".{self.active_link.name}-{uuid.uuid4().hex}")
		self.active_link.parent.mkdir(parents=True, exist_ok=True)
		try:
			if before_switch is not None:
				before_switch()
			link_target = os.path.relpath(final_directory, self.active_link.parent)
			os.symlink(link_target, temporary_link, target_is_directory=True)
			os.replace(temporary_link, self.active_link)
		except Exception as exc:
			temporary_link.unlink(missing_ok=True)
			if isinstance(exc, OSError) and os.name == "nt":
				raise RuntimeError(
					"Atomic index symlink switching requires Windows Developer Mode or symlink privileges"
				) from exc
			raise
		LOGGER.info("Published index generation %s", version)
		return final_directory