import unittest
from unittest.mock import Mock, patch

from controllers.main import incremental_sync_status, trigger_incremental_sync
from controllers.schemas import IncrementalSyncStatusResponse


class IncrementalSyncAdminTests(unittest.TestCase):
	@patch("controllers.main.create_redis_client")
	@patch("controllers.main.RedisTaskBroker")
	def test_trigger_queues_incremental_sync(self, broker_type: Mock, create_client: Mock) -> None:
		client = create_client.return_value
		broker = broker_type.return_value

		response = trigger_incremental_sync()

		broker.enqueue.assert_called_once_with("sync")
		client.close.assert_called_once_with()
		self.assertEqual(response.status, "queued")

	@patch("controllers.main.create_redis_client")
	@patch("controllers.main.RedisTaskBroker")
	def test_status_exposes_worker_sync_fields(self, broker_type: Mock, create_client: Mock) -> None:
		client = create_client.return_value
		broker_type.return_value.get_status.return_value = {
			"status": "completed",
			"task_id": "task-1",
			"source_rows": 4,
			"indexed_rows": 3,
			"generation": "generation-1",
			"last_synced_at": "2026-11-03T00:00:00+08:00",
		}

		response = incremental_sync_status()

		self.assertIsInstance(response, IncrementalSyncStatusResponse)
		self.assertEqual(response.status, "completed")
		self.assertEqual(response.task_id, "task-1")
		self.assertEqual(response.source_rows, 4)
		self.assertEqual(response.indexed_rows, 3)
		self.assertEqual(response.generation, "generation-1")
		self.assertEqual(response.last_synced_at, "2026-11-03T00:00:00+08:00")
		client.close.assert_called_once_with()


if __name__ == "__main__":
	unittest.main()