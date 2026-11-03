import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import config
from services.index_generation import IndexGenerationManager


class IndexGenerationTests(unittest.TestCase):
	def _create_staging(self, manager: IndexGenerationManager) -> Path:
		staging = manager.create_staging(clone_active=False)
		(staging / "chromadb").mkdir()
		(staging / "bm25.pkl").write_bytes(b"bm25")
		(staging / "cleaned.jsonl").write_text("{}\n", encoding="utf-8")
		return staging

	def test_publish_switches_active_pointer_after_guard(self) -> None:
		with TemporaryDirectory() as directory:
			root = Path(directory)
			active = root / "active-index"
			versions = root / "versions"
			with (
				patch.object(config, "ACTIVE_INDEX_SYMLINK", active),
				patch.object(config, "INDEX_VERSIONS_DIRECTORY", versions),
			):
				manager = IndexGenerationManager()
				staging = self._create_staging(manager)
				guard_called = False

				def before_switch() -> None:
					nonlocal guard_called
					guard_called = True

				def emulate_symlink(target: str, link: Path, *, target_is_directory: bool) -> None:
					link.write_text(target, encoding="utf-8")

				with patch("services.index_generation.os.symlink", side_effect=emulate_symlink):
					generation = manager.publish(staging, before_switch=before_switch)

			self.assertTrue(guard_called)
			self.assertEqual(os.path.normpath(active.read_text(encoding="utf-8")), os.path.normpath(os.path.relpath(generation, active.parent)))
			self.assertTrue((generation / "chromadb").is_dir())

	def test_failed_guard_does_not_replace_active_pointer(self) -> None:
		with TemporaryDirectory() as directory:
			root = Path(directory)
			active = root / "active-index"
			versions = root / "versions"
			active.write_text("old-generation", encoding="utf-8")
			with (
				patch.object(config, "ACTIVE_INDEX_SYMLINK", active),
				patch.object(config, "INDEX_VERSIONS_DIRECTORY", versions),
			):
				manager = IndexGenerationManager()
				staging = self._create_staging(manager)

				def reject_switch() -> None:
					raise RuntimeError("writer lease lost")

				with self.assertRaisesRegex(RuntimeError, "writer lease lost"):
					manager.publish(staging, before_switch=reject_switch)

			self.assertEqual(active.read_text(encoding="utf-8"), "old-generation")


if __name__ == "__main__":
	unittest.main()