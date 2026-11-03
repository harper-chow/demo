"""Single-writer service for scheduled incremental and manual full syncs."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

import config
from controllers.services import RebuildService
from services.incremental_sync import IncrementalSyncService
from services.index_generation import IndexGenerationManager
from services.ingest_new_data import configure_logging
from services.redis_coordination import (
	RedisTaskBroker,
	RedisWriterLease,
	create_redis_client,
)
from services.vector_store import ChromaVectorStore


LOGGER = logging.getLogger(__name__)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class SyncWorker:
	def __init__(self) -> None:
		self.redis = create_redis_client()
		self.broker = RedisTaskBroker(self.redis)
		self.lease = RedisWriterLease(self.redis)
		self.generations = IndexGenerationManager()
		self.rebuild_service = RebuildService()
		self.incremental_service = IncrementalSyncService()
		self._rebuild_lock = threading.Lock()
		self._stop = threading.Event()
		self._scheduler: BackgroundScheduler | None = None

	def stop(self) -> None:
		self._stop.set()

	def _status(self, task_type: str, status: str, **values: Any) -> None:
		self.broker.set_status(task_type, {"status": status, "updated_at": _now(), **values})

	def _announce_generation(self, generation: str) -> None:
		self.broker.publish_index_version(generation)

	def _prepare_active_generation(self) -> None:
		self.lease.assert_owned()
		previous_version = self.generations.active_version()
		active = self.generations.ensure_active(before_switch=self.lease.assert_owned)
		if active is None:
			LOGGER.warning("No active index exists; waiting for a manual full rebuild")
			return
		current_version = self.generations.active_version()
		if current_version != previous_version:
			self._announce_generation(current_version or active.name)

		staging = self.generations.create_staging(clone_active=True)
		store = None
		try:
			store = ChromaVectorStore(
				allow_missing=False,
				persist_directory=staging / "chromadb",
				bm25_index_path=staging / "bm25.pkl",
			)
			if self.incremental_service.initialize_checkpoint(
				store.bm25, index_path=staging / "bm25.pkl"
			):
				store.validate_consistency()
				store.close()
				store = None
				self.lease.assert_owned()
				version = self.generations.publish(staging, before_switch=self.lease.assert_owned)
				self._announce_generation(version.name)
				LOGGER.info("Published initial incremental-sync checkpoint")
		finally:
			if store is not None:
				store.close()
			if staging.exists():
				shutil.rmtree(staging, ignore_errors=True)

	def _enqueue_daily_sync(self) -> None:
		try:
			self.lease.assert_owned()
			self.broker.enqueue("sync")
		except Exception:
			LOGGER.exception("Could not enqueue scheduled incremental sync")

	def _start_scheduler(self) -> None:
		self._scheduler = BackgroundScheduler(timezone=config.DAILY_SYNC_TIMEZONE)
		self._scheduler.add_job(
			self._enqueue_daily_sync,
			trigger="cron",
			hour=0,
			minute=0,
			id="daily-database-incremental-sync",
			coalesce=True,
			max_instances=1,
			misfire_grace_time=24 * 60 * 60,
			replace_existing=True,
		)
		self._scheduler.start()
		LOGGER.info("Sync worker schedule enabled at 00:00 %s", config.DAILY_SYNC_TIMEZONE)

	def _stop_scheduler(self) -> None:
		if self._scheduler is not None and self._scheduler.running:
			self._scheduler.shutdown(wait=False)
		self._scheduler = None

	def _run_rebuild(self, task: dict[str, Any]) -> None:
		self.lease.assert_owned()
		indexed_count, generation = self.rebuild_service.rebuild(
			before_switch=self.lease.assert_owned
		)
		self._announce_generation(generation)
		self._status(
			"rebuild",
			"completed",
			task_id=task.get("task_id"),
			indexed_count=indexed_count,
			generation=generation,
			completed_at=_now(),
		)

	def _run_incremental_sync(self, task: dict[str, Any]) -> None:
		self.lease.assert_owned()
		staging = self.generations.create_staging(clone_active=True)
		try:
			result = self.incremental_service.sync(index_directory=staging)
			if not result.changed:
				self._status(
					"sync",
					"completed",
					task_id=task.get("task_id"),
					indexed_rows=0,
					generation=self.generations.active_version(),
					last_synced_at=result.last_synced_at,
					completed_at=_now(),
				)
				return
			self.lease.assert_owned()
			version = self.generations.publish(staging, before_switch=self.lease.assert_owned)
			self._announce_generation(version.name)
			self._status(
				"sync",
				"completed",
				task_id=task.get("task_id"),
				indexed_rows=result.indexed_rows,
				source_rows=result.source_rows,
				generation=version.name,
				last_synced_at=result.last_synced_at,
				completed_at=_now(),
			)
		finally:
			if staging.exists():
				shutil.rmtree(staging, ignore_errors=True)

	def _run_task(self, task: dict[str, Any]) -> None:
		task_type = task.get("task_type")
		if task_type not in {"sync", "rebuild"}:
			LOGGER.error("Ignoring unknown task type: %s", task_type)
			return
		if not self._rebuild_lock.acquire(blocking=False):
			LOGGER.warning("Skipping duplicate queued %s task", task_type)
			return
		self._status(task_type, "running", task_id=task.get("task_id"), started_at=_now())
		try:
			if task_type == "rebuild":
				self._run_rebuild(task)
			else:
				self._run_incremental_sync(task)
		except Exception as exc:
			LOGGER.exception("%s task failed", task_type)
			self._status(
				task_type,
				"failed",
				task_id=task.get("task_id"),
				error=str(exc),
				completed_at=_now(),
			)
		finally:
			self._rebuild_lock.release()

	def _run_as_leader(self) -> None:
		self.lease.start_renewal()
		try:
			self._recover_inflight_tasks()
			self._prepare_active_generation()
			self.lease.assert_owned()
			self._start_scheduler()
			while not self._stop.is_set() and not self.lease.lost:
				try:
					self.lease.assert_owned()
					queued = self.redis.brpoplpush(
						config.REDIS_TASK_QUEUE,
						config.REDIS_PROCESSING_QUEUE,
						timeout=1,
					)
					if queued is not None:
						try:
							self._run_task(json.loads(queued))
						finally:
							self.redis.lrem(config.REDIS_PROCESSING_QUEUE, 1, queued)
				except Exception:
					if self.lease.lost:
						break
					LOGGER.exception("Sync worker queue loop failed")
					time.sleep(1)
		finally:
			self._stop_scheduler()
			self.lease.release()
			LOGGER.warning("Sync worker stopped leader tasks")

	def _recover_inflight_tasks(self) -> None:
		while True:
			self.lease.assert_owned()
			recovered = self.redis.rpoplpush(
				config.REDIS_PROCESSING_QUEUE,
				config.REDIS_TASK_QUEUE,
			)
			if recovered is None:
				return
			LOGGER.warning("Recovered an unfinished Redis index task")

	def run(self) -> None:
		self.redis.ping()
		self.generations.versions_directory.mkdir(parents=True, exist_ok=True)
		while not self._stop.is_set():
			try:
				if self.lease.acquire():
					LOGGER.info("Acquired Redis writer lease")
					self._run_as_leader()
				else:
					LOGGER.info("Another sync worker owns the writer lease")
					self._stop.wait(5)
			except Exception:
				LOGGER.exception("Sync worker loop failed; retrying")
				self.lease.release()
				self._stop.wait(5)


def main() -> None:
	configure_logging()
	worker = SyncWorker()
	try:
		worker.run()
	finally:
		worker.stop()
		worker.lease.release()
		worker.redis.close()


if __name__ == "__main__":
	main()