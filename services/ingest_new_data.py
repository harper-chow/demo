"""Load database records and build a full index set."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from services.data_loader import load_data
from services.data_process import process_dataframe, write_jsonl
from services.embedder import LocalEmbedder
from services.sync_checkpoint import boundary_fingerprints, normalize_timestamp, timestamp_text
from services.vector_store import ChromaVectorStore


def configure_logging() -> None:
    config.ensure_runtime_directories()
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_DIR / "ingest.log", encoding="utf-8"),
        ],
    )


def build_index(staging_directory: Path | str) -> int:
    """Build and validate a complete index set in an isolated staging directory."""
    configure_logging()
    staging = Path(staging_directory)
    dataframe = load_data()
    records = process_dataframe(dataframe)
    cleaned_path = write_jsonl(records, staging / "cleaned.jsonl")
    store = ChromaVectorStore(
        LocalEmbedder(),
        allow_missing=True,
        persist_directory=staging / "chromadb",
        bm25_index_path=staging / "bm25.pkl",
    )
    try:
        indexed_count = store.build_from_jsonl(cleaned_path)
        timestamps = dataframe[config.DB_TIME_COLUMN].dropna()
        watermark = (
            normalize_timestamp(timestamps.max())
            if not timestamps.empty
            else datetime.now(timezone.utc)
        )
        store.bm25.sync_state = {
            "last_synced_at": timestamp_text(watermark),
            "boundary_fingerprints": boundary_fingerprints(dataframe, watermark),
        }
        store.persist_bm25()
    finally:
        store.close()
    if indexed_count <= 0:
        raise ValueError("No records were indexed")
    return indexed_count