"""Application startup entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config
from controllers import main as controller


def configure_logging() -> None:
	config.ensure_runtime_directories()
	root_logger = logging.getLogger()
	if root_logger.handlers:
		return
	root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
	formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	file_handler = logging.FileHandler(config.LOG_DIR / "api.log", encoding="utf-8")
	file_handler.setFormatter(formatter)
	root_logger.addHandler(console_handler)
	root_logger.addHandler(file_handler)


configure_logging()
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
	controller._initialization_error = None
	start_time = perf_counter()
	try:
		controller.get_retriever()
		LOGGER.info("Retriever ready in %.3fs", perf_counter() - start_time)
	except Exception as exc:
		controller._initialization_error = str(exc)
		LOGGER.exception("Retriever initialization failed")
	controller.start_index_listener()
	try:
		yield
	finally:
		controller.stop_index_listener()


app = FastAPI(title="Retrieval API", version="1.0.0", lifespan=lifespan)
app.include_router(controller.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
	LOGGER.exception("Unhandled API error: path=%s, error=%s", request.url.path, exc)
	return JSONResponse(status_code=500, content={"detail": "Internal server error"})