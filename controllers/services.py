"""Application services used by the HTTP layer."""

from __future__ import annotations

from pathlib import Path

from services.ingest_new_data import build_index
from services.index_generation import IndexGenerationManager


class IndexArtifactInstaller:
	"""Publish a complete immutable index generation via active-link replacement."""

	def __init__(self, generations: IndexGenerationManager | None = None) -> None:
		self.generations = generations or IndexGenerationManager()

	def install(self, staging: Path, before_switch=None) -> Path:
		return self.generations.publish(staging, before_switch=before_switch)


class RebuildService:
	"""Build a complete index in staging and install it after validation."""

	def __init__(self, installer: IndexArtifactInstaller | None = None) -> None:
		self.installer = installer or IndexArtifactInstaller()

	def rebuild(self, before_switch=None) -> tuple[int, str]:
		staging = self.installer.generations.create_staging(clone_active=False)
		try:
			indexed_count = build_index(staging)
			version = self.installer.install(staging, before_switch=before_switch)
			return indexed_count, version.name
		finally:
			if staging.exists():
				import shutil
				shutil.rmtree(staging, ignore_errors=True)
