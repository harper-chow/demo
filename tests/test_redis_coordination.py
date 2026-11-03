import json
import unittest

from services.redis_coordination import RedisTaskBroker, RedisWriterLease


class FakeRedis:
	def __init__(self) -> None:
		self.values: dict[str, str] = {}
		self.queue: list[tuple[str, str]] = []

	def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None):
		if nx and key in self.values:
			return False
		self.values[key] = value
		return True

	def get(self, key: str):
		return self.values.get(key)

	def eval(self, script: str, _numkeys: int, key: str, token: str, *_args):
		if self.values.get(key) != token:
			return 0
		if "DEL" in script:
			self.values.pop(key, None)
		return 1

	def lpush(self, key: str, value: str) -> int:
		self.queue.insert(0, (key, value))
		return len(self.queue)


class RedisCoordinationTests(unittest.TestCase):
	def test_writer_lease_is_owned_and_release_is_token_checked(self) -> None:
		client = FakeRedis()
		lease = RedisWriterLease(client, "test-lock")

		self.assertTrue(lease.acquire())
		lease.assert_owned()
		client.values["test-lock"] = "another-owner"
		with self.assertRaises(RuntimeError):
			lease.assert_owned()
		lease.release()
		self.assertEqual(client.values["test-lock"], "another-owner")

	def test_task_queue_and_status_are_shared_through_redis(self) -> None:
		client = FakeRedis()
		broker = RedisTaskBroker(client)

		task_id = broker.enqueue("rebuild")

		queued = json.loads(client.queue[0][1])
		status = broker.get_status("rebuild")
		self.assertEqual(queued["task_id"], task_id)
		self.assertEqual(queued["task_type"], "rebuild")
		self.assertEqual(status["status"], "queued")
		self.assertEqual(status["task_id"], task_id)


if __name__ == "__main__":
	unittest.main()