from datetime import datetime, timezone
import unittest

import pandas as pd

from services.sync_checkpoint import select_unseen_rows, source_fingerprint


class SyncCheckpointTests(unittest.TestCase):
	def test_selects_new_rows_without_replaying_boundary_rows(self) -> None:
		watermark = datetime(2026, 10, 29, tzinfo=timezone.utc)
		base_record = {
			"company_code": "9200",
			"category_name": "ITEM",
			"price": "1.00",
			"category": "BB",
			"category_id": "A1",
			"currency": "RMB",
		}
		old_row = {**base_record, "createtime": watermark}
		new_boundary_row = {**base_record, "category_id": "A2", "createtime": watermark}
		newer_row = {
			**base_record,
			"category_id": "A3",
			"createtime": datetime(2026, 10, 30, tzinfo=timezone.utc),
		}
		dataframe = pd.DataFrame([old_row, old_row, new_boundary_row, newer_row])

		selected, next_watermark, next_boundary = select_unseen_rows(
			dataframe,
			watermark,
			{source_fingerprint(old_row): 1},
		)

		self.assertEqual(len(selected), 3)
		self.assertEqual(next_watermark, newer_row["createtime"])
		self.assertEqual(next_boundary, {source_fingerprint(newer_row): 1})

	def test_repeated_boundary_query_returns_no_rows(self) -> None:
		watermark = datetime(2026, 10, 29, tzinfo=timezone.utc)
		row = {
			"company_code": "9200",
			"category_name": "ITEM",
			"price": "1.00",
			"category": "BB",
			"category_id": "A1",
			"currency": "RMB",
			"createtime": watermark,
		}
		dataframe = pd.DataFrame([row])

		selected, next_watermark, next_boundary = select_unseen_rows(
			dataframe,
			watermark,
			{source_fingerprint(row): 1},
		)

		self.assertTrue(selected.empty)
		self.assertEqual(next_watermark, watermark)
		self.assertEqual(next_boundary, {source_fingerprint(row): 1})


if __name__ == "__main__":
	unittest.main()