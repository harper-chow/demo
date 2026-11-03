"""Read and validate records from the configured PostgreSQL view."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import psycopg
from psycopg import sql

import config


LOGGER = logging.getLogger(__name__)


def validate_columns(dataframe: pd.DataFrame) -> None:
	expected = list(config.DB_COLUMNS.values())
	missing = [column for column in expected if column not in dataframe.columns]
	if missing:
		raise ValueError(f"Database view is missing configured columns: {missing}")


def load_data(since: datetime | None = None) -> pd.DataFrame:
	"""Read business fields and createtime, optionally from a saved watermark."""
	if not config.DB_USER or not config.DB_PASSWORD:
		raise ValueError("Set DB_USER and DB_PASSWORD before reading the database")

	source_columns = list(config.DB_COLUMNS.values())
	selected_columns = sql.SQL(", ").join(
		sql.SQL("{}::text AS {}").format(sql.Identifier(column), sql.Identifier(column))
		for column in source_columns
	)
	selected_columns = sql.SQL("{}, {}").format(
		selected_columns,
		sql.Identifier(config.DB_TIME_COLUMN),
	)
	query = sql.SQL("SELECT {} FROM {}").format(
		selected_columns, sql.Identifier(config.DB_SCHEMA, config.DB_VIEW)
	)
	if since is not None:
		query += sql.SQL(" WHERE {} >= %s ORDER BY {} ASC").format(
			sql.Identifier(config.DB_TIME_COLUMN),
			sql.Identifier(config.DB_TIME_COLUMN),
		)
	LOGGER.info("Reading PostgreSQL view: %s.%s", config.DB_SCHEMA, config.DB_VIEW)
	with psycopg.connect(
		host=config.DB_HOST,
		port=config.DB_PORT,
		dbname=config.DB_NAME,
		user=config.DB_USER,
		password=config.DB_PASSWORD,
	) as connection:
		with connection.cursor() as cursor:
			if since is None:
				cursor.execute(query)
			else:
				cursor.execute(query, (since,))
			dataframe = pd.DataFrame(
				cursor.fetchall(),
				columns=[*source_columns, config.DB_TIME_COLUMN],
			)
	for column in source_columns:
		dataframe[column] = dataframe[column].astype("string")
	validate_columns(dataframe)
	dataframe = dataframe.rename(columns={source: field for field, source in config.DB_COLUMNS.items()})
	LOGGER.info("Read database rows=%d, columns=%d", len(dataframe), len(dataframe.columns))
	return dataframe


def get_latest_createtime() -> datetime | None:
	"""Return the greatest source timestamp without reading the full view."""
	if not config.DB_USER or not config.DB_PASSWORD:
		raise ValueError("Set DB_USER and DB_PASSWORD before reading the database")
	query = sql.SQL("SELECT MAX({}) FROM {}").format(
		sql.Identifier(config.DB_TIME_COLUMN),
		sql.Identifier(config.DB_SCHEMA, config.DB_VIEW),
	)
	with psycopg.connect(
		host=config.DB_HOST,
		port=config.DB_PORT,
		dbname=config.DB_NAME,
		user=config.DB_USER,
		password=config.DB_PASSWORD,
	) as connection:
		with connection.cursor() as cursor:
			cursor.execute(query)
			return cursor.fetchone()[0]
