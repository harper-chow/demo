"""HTTP routes for category retrieval."""

from __future__ import annotations

import logging
import threading
import json
from contextlib import contextmanager
from functools import lru_cache
from time import perf_counter

from fastapi import APIRouter, HTTPException

import config
from services.downloader import model_path
from services.redis_coordination import RedisTaskBroker, create_redis_client
from services.retriever import Retriever, normalize_query

from .schemas import (
	HealthResponse,
	IncrementalSyncStatusResponse,
	RebuildResponse,
	RebuildStatusResponse,
	SearchRequest,
	SearchResponse,
)
LOGGER = logging.getLogger(__name__)
_initialization_error: str | None = None
_retriever_lock = threading.Condition()
_active_searches = 0
_reloading_retriever = False
_loaded_generation: str | None = None
_index_listener_stop = threading.Event()
_index_listener_thread: threading.Thread | None = None


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
	global _loaded_generation
	retriever = Retriever()
	_loaded_generation = _current_generation()
	return retriever


@contextmanager
def _retriever_read():
	global _active_searches
	with _retriever_lock:
		while _reloading_retriever:
			_retriever_lock.wait()
		_active_searches += 1
	try:
		yield get_retriever()
	finally:
		with _retriever_lock:
			_active_searches -= 1
			if _active_searches == 0:
				_retriever_lock.notify_all()


def _current_generation() -> str:
	return config.get_active_index_version() or "legacy"


def reload_retriever(generation: str | None = None) -> None:
	global _active_searches, _initialization_error, _loaded_generation, _reloading_retriever
	with _retriever_lock:
		_reloading_retriever = True
		while _active_searches:
			_retriever_lock.wait()
	try:
		if get_retriever.cache_info().currsize:
			get_retriever().store.close()
		get_retriever.cache_clear()
		get_retriever()
		_initialization_error = None
		_loaded_generation = generation or _current_generation()
		LOGGER.info("Reloaded read-only index generation %s", _loaded_generation)
	except Exception as exc:
		_initialization_error = str(exc)
		LOGGER.exception("Failed to reload index generation")
	finally:
		with _retriever_lock:
			_reloading_retriever = False
			_retriever_lock.notify_all()


def _listen_for_index_publications() -> None:
	while not _index_listener_stop.is_set():
		client = create_redis_client()
		pubsub = None
		try:
			client.ping()
			pubsub = client.pubsub()
			pubsub.subscribe(config.REDIS_INDEX_CHANNEL)
			if _initialization_error or _current_generation() != _loaded_generation:
				reload_retriever()
			while not _index_listener_stop.is_set():
				message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
				if not message:
					continue
				payload = json.loads(message["data"])
				generation = payload.get("version")
				if generation and generation != _loaded_generation:
					reload_retriever(generation)
		except Exception:
			if not _index_listener_stop.is_set():
				LOGGER.exception("Index publication listener disconnected; retrying")
				_index_listener_stop.wait(2)
		finally:
			if pubsub is not None:
				pubsub.close()
			client.close()


def start_index_listener() -> None:
	global _index_listener_thread
	if _index_listener_thread is not None and _index_listener_thread.is_alive():
		return
	_index_listener_stop.clear()
	_index_listener_thread = threading.Thread(
		target=_listen_for_index_publications,
		name="index-generation-listener",
		daemon=True,
	)
	_index_listener_thread.start()


def stop_index_listener() -> None:
	global _index_listener_thread
	_index_listener_stop.set()
	if _index_listener_thread is not None:
		_index_listener_thread.join(timeout=3)
	_index_listener_thread = None


router = APIRouter()


@router.post("/admin/rebuild", response_model=RebuildResponse, status_code=202)
def rebuild_indexes() -> RebuildResponse:
	client = create_redis_client()
	try:
		RedisTaskBroker(client).enqueue("rebuild")
	except Exception as exc:
		LOGGER.exception("Could not enqueue full rebuild")
		raise HTTPException(status_code=503, detail="Sync worker or Redis is unavailable") from exc
	finally:
		client.close()
	return RebuildResponse(status="queued", message="Full database rebuild queued")


@router.post("/admin/sync", response_model=RebuildResponse, status_code=202)
def trigger_incremental_sync() -> RebuildResponse:
	client = create_redis_client()
	try:
		RedisTaskBroker(client).enqueue("sync")
	except Exception as exc:
		LOGGER.exception("Could not enqueue incremental sync")
		raise HTTPException(status_code=503, detail="Sync worker or Redis is unavailable") from exc
	finally:
		client.close()
	return RebuildResponse(status="queued", message="Incremental database sync queued")


@router.get("/admin/rebuild/status", response_model=RebuildStatusResponse)
def rebuild_status() -> RebuildStatusResponse:
	client = create_redis_client()
	try:
		status = RedisTaskBroker(client).get_status("rebuild")
	except Exception as exc:
		LOGGER.exception("Could not read rebuild status")
		raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
	finally:
		client.close()
	return RebuildStatusResponse(**(status or {"status": "idle"}))


@router.get("/admin/sync/status", response_model=IncrementalSyncStatusResponse)
def incremental_sync_status() -> IncrementalSyncStatusResponse:
	client = create_redis_client()
	try:
		status = RedisTaskBroker(client).get_status("sync")
	except Exception as exc:
		LOGGER.exception("Could not read incremental sync status")
		raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
	finally:
		client.close()
	return IncrementalSyncStatusResponse(**(status or {"status": "idle"}))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
	model_available = model_path(config.MODEL_DIRECTORY).is_dir()
	index_available = config.get_chroma_persist_directory().is_dir()
	if _initialization_error:
		return HealthResponse(
			status="degraded",
			index_available=index_available,
			model_available=model_available,
			initialized=False,
			chroma_collection_available=False,
			bm25_loaded=False,
			indexes_consistent=False,
			detail=_initialization_error,
		)
	try:
		with _retriever_read() as retriever:
			store = retriever.store
			store.validate_consistency()
			collection_available = store.dense_collection.count() > 0
			bm25_loaded = bool(store.bm25.documents)
		indexes_consistent = collection_available and bm25_loaded
		return HealthResponse(
			status="ok" if model_available and indexes_consistent else "degraded",
			index_available=index_available,
			model_available=model_available,
			initialized=True,
			chroma_collection_available=collection_available,
			bm25_loaded=bm25_loaded,
			indexes_consistent=indexes_consistent,
			detail=None if indexes_consistent else "Chroma collection or BM25 index is empty",
		)
	except Exception as exc:
		LOGGER.error("Health check failed: %s", exc)
		return HealthResponse(
			status="degraded",
			index_available=index_available,
			model_available=model_available,
			initialized=False,
			chroma_collection_available=False,
			bm25_loaded=False,
			indexes_consistent=False,
			detail=str(exc),
		)
@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
	query_text = normalize_query(request.query)
	if query_text is None:
		return SearchResponse(query=request.query, query_text=None, results=[], returned_count=0)
	if _initialization_error:
		raise HTTPException(status_code=503, detail="Retrieval service is not initialized")

	start_time = perf_counter()
	try:
		with _retriever_read() as retriever:
			results = retriever.search(
				query=request.query,
				top_k=request.top_k,
				top_n=request.top_n,
				vector_weight=request.vector_weight,
				normalized_query=query_text,
			)
	except ValueError as exc:
		LOGGER.warning("Invalid search parameters: %s", exc)
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except (FileNotFoundError, RuntimeError) as exc:
		LOGGER.error("Retrieval service is unavailable: %s", exc)
		raise HTTPException(status_code=503, detail="Retrieval service is unavailable") from exc
	except Exception as exc:
		LOGGER.exception("Search failed: %s", exc)
		raise HTTPException(status_code=500, detail="Search failed") from exc

	LOGGER.info(
		"Search completed: query=%r, normalized=%r, results=%d, elapsed=%.3fs",
		request.query,
		query_text,
		len(results),
		perf_counter() - start_time,
	)
	return SearchResponse(
		query=request.query,
		query_text=query_text,
		results=results,
		returned_count=len(results),
	)
