"""Application configuration for the data ingestion stage."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: Path) -> Path:
	value = os.getenv(name)
	if not value:
		return default
	path = Path(value)
	return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LEGACY_CLEANED_JSONL_PATH = _path_from_env("CLEANED_JSONL_PATH", PROCESSED_DIR / "cleaned.jsonl")
LEGACY_BM25_INDEX_PATH = _path_from_env("BM25_INDEX_PATH", PROCESSED_DIR / "bm25.pkl")
LEGACY_CHROMA_PERSIST_DIRECTORY = _path_from_env("CHROMA_PERSIST_DIRECTORY", DATA_DIR / "chromadb")
INDEX_VERSIONS_DIRECTORY = _path_from_env("INDEX_VERSIONS_DIRECTORY", DATA_DIR / "index-versions")
ACTIVE_INDEX_SYMLINK = _path_from_env("ACTIVE_INDEX_SYMLINK", DATA_DIR / "active-index")

DB_HOST = os.getenv("DB_HOST", "172.20.170.149")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ai_budget")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
DB_VIEW = "v_mro4ai"
DB_TIME_COLUMN = "createtime"
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_COLUMNS = {
	"company_code": "company_code",
	"category_name": "product_name",
	"price": "price",
	"category": "actual_mro_category",
	"category_id": "account_code",
	"currency": "currency",
}

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "category_retrieval")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MODEL_DIRECTORY = _path_from_env("MODEL_DIRECTORY", PROJECT_ROOT / "models")

DEFAULT_TOP_K = 50
MAX_TOP_K = 200
DEFAULT_TOP_N = 5
MAX_TOP_N = 20
CHROMA_DENSE_WEIGHT = float(os.getenv("CHROMA_DENSE_WEIGHT", "0.4"))

LOG_DIR = _path_from_env("LOG_DIR", DATA_DIR / "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
INDEX_STAGING_DIRECTORY = _path_from_env("INDEX_STAGING_DIRECTORY", DATA_DIR / ".index-staging")
DAILY_SYNC_TIMEZONE = os.getenv("DAILY_SYNC_TIMEZONE", "Asia/Shanghai")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_LOCK_KEY = os.getenv("REDIS_LOCK_KEY", "retrieval:index-writer:lock")
REDIS_STATUS_PREFIX = os.getenv("REDIS_STATUS_PREFIX", "retrieval:status")
REDIS_INDEX_CHANNEL = os.getenv("REDIS_INDEX_CHANNEL", "retrieval:index:published")
REDIS_TASK_QUEUE = os.getenv("REDIS_TASK_QUEUE", "retrieval:index:tasks")
REDIS_PROCESSING_QUEUE = os.getenv("REDIS_PROCESSING_QUEUE", "retrieval:index:processing")


def active_index_path(relative_path: str) -> Path:
	if ACTIVE_INDEX_SYMLINK.is_symlink() or ACTIVE_INDEX_SYMLINK.exists():
		return ACTIVE_INDEX_SYMLINK / relative_path
	legacy_paths = {
		"chromadb": LEGACY_CHROMA_PERSIST_DIRECTORY,
		"bm25.pkl": LEGACY_BM25_INDEX_PATH,
		"cleaned.jsonl": LEGACY_CLEANED_JSONL_PATH,
	}
	return legacy_paths[relative_path]


def get_chroma_persist_directory() -> Path:
	return active_index_path("chromadb")


def get_bm25_index_path() -> Path:
	return active_index_path("bm25.pkl")


def get_cleaned_jsonl_path() -> Path:
	return active_index_path("cleaned.jsonl")


def get_active_index_version() -> str | None:
	if ACTIVE_INDEX_SYMLINK.is_symlink():
		return ACTIVE_INDEX_SYMLINK.resolve(strict=True).name
	if ACTIVE_INDEX_SYMLINK.exists():
		return ACTIVE_INDEX_SYMLINK.name
	return None


# Compatibility aliases; new runtime code should use the dynamic getters.
CHROMA_PERSIST_DIRECTORY = LEGACY_CHROMA_PERSIST_DIRECTORY
BM25_INDEX_PATH = LEGACY_BM25_INDEX_PATH
CLEANED_JSONL_PATH = LEGACY_CLEANED_JSONL_PATH


def ensure_runtime_directories() -> None:
	"""Create directories needed by ingestion without creating an index."""

	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
	MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
	LOG_DIR.mkdir(parents=True, exist_ok=True)
