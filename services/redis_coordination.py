"""Redis primitives shared by the sync worker and read-only API processes."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import redis

import config


LOGGER = logging.getLogger(__name__)
LOCK_TTL_SECONDS = 60
LOCK_RENEW_INTERVAL_SECONDS = 20

_RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


def create_redis_client() -> redis.Redis:
	return redis.Redis.from_url(
		config.REDIS_URL,
		decode_responses=True,
		socket_connect_timeout=5,
		socket_timeout=5,
	)


class RedisWriterLease:
	"""A renewable leader lease; any uncertainty is treated as lease loss."""

	def __init__(self, client: redis.Redis, lock_key: str = config.REDIS_LOCK_KEY) -> None:
		self.client = client
		self.lock_key = lock_key
		self.token = uuid.uuid4().hex
		self._lost = threading.Event()
		self._stop_renewal = threading.Event()
		self._renewal_thread: threading.Thread | None = None

	@property
	def lost(self) -> bool:
		return self._lost.is_set()

	def acquire(self) -> bool:
		acquired = self.client.set(
			self.lock_key, self.token, nx=True, ex=LOCK_TTL_SECONDS
		)
		if acquired:
			self._lost.clear()
			self._stop_renewal.clear()
		return bool(acquired)

	def start_renewal(self) -> None:
		if self._renewal_thread is not None and self._renewal_thread.is_alive():
			return
		self._stop_renewal.clear()
		self._renewal_thread = threading.Thread(
			target=self._renew_loop,
			name="redis-writer-lease-renewal",
			daemon=True,
		)
		self._renewal_thread.start()

	def _renew_loop(self) -> None:
		while not self._stop_renewal.wait(LOCK_RENEW_INTERVAL_SECONDS):
			try:
				renewed = self.client.eval(
					_RENEW_SCRIPT,
					1,
					self.lock_key,
					self.token,
					LOCK_TTL_SECONDS,
				)
				if not renewed:
					self._lost.set()
					LOGGER.error("Redis writer lease was lost")
					return
			except Exception:
				self._lost.set()
				LOGGER.exception("Redis writer lease renewal failed; stopping writes")
				return

	def assert_owned(self) -> None:
		if self._lost.is_set() or self.client.get(self.lock_key) != self.token:
			self._lost.set()
			raise RuntimeError("Redis writer lease is not held")

	def stop_renewal(self) -> None:
		self._stop_renewal.set()
		if self._renewal_thread is not None:
			self._renewal_thread.join(timeout=LOCK_RENEW_INTERVAL_SECONDS + 1)
		self._renewal_thread = None

	def release(self) -> None:
		self.stop_renewal()
		try:
			self.client.eval(_RELEASE_SCRIPT, 1, self.lock_key, self.token)
		except Exception:
			LOGGER.exception("Failed to release Redis writer lease; it will expire")


class RedisTaskBroker:
	"""Queue manual writes and store operation status outside API memory."""

	def __init__(self, client: redis.Redis) -> None:
		self.client = client

	def status_key(self, task_type: str) -> str:
		return f"{config.REDIS_STATUS_PREFIX}:{task_type}"

	def set_status(self, task_type: str, status: dict[str, Any]) -> None:
		self.client.set(self.status_key(task_type), json.dumps(status, ensure_ascii=False))

	def get_status(self, task_type: str) -> dict[str, Any] | None:
		value = self.client.get(self.status_key(task_type))
		return json.loads(value) if value else None

	def enqueue(self, task_type: str) -> str:
		task_id = uuid.uuid4().hex
		now = datetime.now(timezone.utc).isoformat()
		self.set_status(task_type, {
			"status": "queued",
			"task_id": task_id,
			"requested_at": now,
		})
		self.client.lpush(
			config.REDIS_TASK_QUEUE,
			json.dumps({"task_type": task_type, "task_id": task_id, "requested_at": now}),
		)
		return task_id

	def publish_index_version(self, version: str) -> None:
		self.client.publish(
			config.REDIS_INDEX_CHANNEL,
			json.dumps({"version": version, "published_at": datetime.now(timezone.utc).isoformat()}),
		)