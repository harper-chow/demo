"""FastAPI request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

import config


class SearchRequest(BaseModel):
	model_config = ConfigDict(extra="forbid")

	query: str = Field(..., min_length=1, description="Natural-language category name query")
	top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K)
	top_n: int = Field(default=config.DEFAULT_TOP_N, ge=1, le=config.MAX_TOP_N)
	vector_weight: float | None = Field(default=None, ge=0, le=1, description="RRF fusion weight: 0 = pure BM25, 1 = pure vector")


class SearchResult(BaseModel):
	category_name: str | None = None
	company_code: str | None = None
	price: float | None = None
	currency: str | None = None
	category: str | None = None
	category_id: str | None = None
	init_text: str | None = None
	rrf_score: float | None = None
	count: int


class SearchResponse(BaseModel):
	query: str
	query_text: str | None
	results: list[SearchResult]
	returned_count: int


class HealthResponse(BaseModel):
	status: str
	index_available: bool
	model_available: bool
	initialized: bool
	chroma_collection_available: bool
	bm25_loaded: bool
	indexes_consistent: bool
	detail: str | None = None


class RebuildResponse(BaseModel):
	status: str
	message: str


class RebuildStatusResponse(BaseModel):
	status: str
	indexed_count: int | None = None
	error: str | None = None


class IncrementalSyncStatusResponse(BaseModel):
	status: str
	task_id: str | None = None
	requested_at: str | None = None
	started_at: str | None = None
	completed_at: str | None = None
	source_rows: int | None = None
	generation: str | None = None
	indexed_rows: int | None = None
	checked_at: str | None = None
	last_synced_at: str | None = None
	error: str | None = None
