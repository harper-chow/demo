"""Gradio UI for category retrieval — thin client over the FastAPI backend.

Start the backend first (model/index loading happens there):
    uvicorn app:app --host 127.0.0.1 --port 8000
Then start the UI:
    python ui.py
"""

from __future__ import annotations

import html
import os
from typing import Any

import gradio as gr
import httpx

import config


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
RESULT_HEADERS = ["init_text", "company_code", "price", "category_id", "category", "rrf_score"]

UI_CSS = """
:root {
    --fluent-blue: #0f6cbd;
    --fluent-blue-dark: #115ea3;
    --fluent-ink: #1f2937;
    --fluent-muted: #5f6b7a;
    --fluent-border: #d6e2f0;
    --fluent-surface: #ffffff;
    --fluent-tint: #f3f8fd;
}

body {
    background: #ffffff;
}

.gradio-container {
    width: min(1180px, calc(100% - 48px)) !important;
    max-width: 1180px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 32px !important;
    color: var(--fluent-ink);
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}

.gradio-container h1 {
    color: #12344d;
    letter-spacing: 0;
    font-weight: 650;
}

.gradio-container p,
.gradio-container label {
    color: var(--fluent-muted);
}

.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap {
    border-color: var(--fluent-border) !important;
    background: var(--fluent-surface) !important;
    box-shadow: 0 1px 2px rgba(31, 52, 73, 0.05);
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: var(--fluent-blue) !important;
    box-shadow: 0 0 0 2px rgba(15, 108, 189, 0.16) !important;
}

.gradio-container button.primary {
    background: var(--fluent-blue) !important;
    border-color: var(--fluent-blue) !important;
    box-shadow: 0 2px 4px rgba(15, 108, 189, 0.2);
}

.gradio-container button.primary:hover {
    background: var(--fluent-blue-dark) !important;
}

.gradio-container input[type="range"] {
    accent-color: var(--fluent-blue) !important;
}

.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    background: #c8def1 !important;
}

.gradio-container input[type="range"]::-webkit-slider-thumb {
    background: var(--fluent-blue) !important;
    border-color: var(--fluent-blue) !important;
}

.gradio-container input[type="range"]::-moz-range-track {
    background: #c8def1 !important;
}

.gradio-container input[type="range"]::-moz-range-progress {
    background: var(--fluent-blue) !important;
}

.gradio-container input[type="range"]::-moz-range-thumb {
    background: var(--fluent-blue) !important;
    border-color: var(--fluent-blue) !important;
}

.results-table {
    border: 1px solid var(--fluent-border);
    border-radius: 6px;
    background: var(--fluent-surface);
    overflow: hidden;
}

.results-table {
    overflow: visible !important;
}

.results-table-scroll {
    max-height: 360px;
    overflow-y: auto;
    scrollbar-color: #9bbbd8 #edf4fa;
}

.results-table-scroll table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}

.results-table-scroll th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--fluent-tint) !important;
    color: #244761;
    font-weight: 600;
    text-align: left;
}

.results-table-scroll th,
.results-table-scroll td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--fluent-border);
    overflow-wrap: anywhere;
    vertical-align: top;
}

.results-table-scroll tbody tr:nth-child(even) {
    background: #f8fbfe;
}

.results-table-scroll th:nth-child(1),
.results-table-scroll td:nth-child(1) { width: 42%; }
.results-table-scroll th:nth-child(2),
.results-table-scroll td:nth-child(2) { width: 13%; }
.results-table-scroll th:nth-child(3),
.results-table-scroll td:nth-child(3) { width: 10%; }
.results-table-scroll th:nth-child(4),
.results-table-scroll td:nth-child(4) { width: 15%; }
.results-table-scroll th:nth-child(5),
.results-table-scroll td:nth-child(5) { width: 14%; }
.results-table-scroll th:nth-child(6),
.results-table-scroll td:nth-child(6) { width: 6%; }
"""


def render_results(rows: list[dict[str, Any]]) -> str:
    """Render backend-ranked results as a read-only semantic table."""
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in RESULT_HEADERS)
    row_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                row.get("init_text") or "",
                row.get("company_code") or "",
                row.get("price") if row.get("price") is not None else "",
                row.get("category_id") or "",
                row.get("category") or "",
                format_score(row.get("rrf_score")),
            )
        )
        + "</tr>"
        for row in rows
    )
    return (
        f'<div class="results-table-scroll"><table><thead><tr>{header_html}</tr>'
        f"</thead><tbody>{row_html}</tbody></table></div>"
    )


def format_score(score: Any) -> str | float:
    """Format a result score for table display."""
    return round(score, 4) if score is not None else ""


def search(query: str, top_n: float, vector_weight: float) -> str:
    """Query the backend /search API and return the ranked result table."""
    if not query or not query.strip():
        raise gr.Error("请输入检索内容")
    payload = {
        "query": query.strip(),
        "top_k": config.DEFAULT_TOP_K,
        "top_n": int(top_n),
        "vector_weight": float(vector_weight),
    }
    try:
        response = httpx.post(f"{API_BASE_URL}/search", json=payload, timeout=300.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise gr.Error(f"请求后端失败（{API_BASE_URL}）：{exc}，请确认后端已启动") from exc
    return render_results(response.json().get("results", []))


def trigger_incremental_sync() -> dict[str, Any]:
    """Queue a manual incremental database sync and show its status."""
    try:
        response = httpx.post(f"{API_BASE_URL}/admin/sync", timeout=30.0)
        response.raise_for_status()
        status_response = httpx.get(f"{API_BASE_URL}/admin/sync/status", timeout=30.0)
        status_response.raise_for_status()
    except httpx.HTTPError as exc:
        raise gr.Error(f"手动增量同步请求失败：{exc}") from exc
    return status_response.json()


def get_incremental_sync_status() -> dict[str, Any]:
    """Read the current manual or scheduled incremental-sync status."""
    try:
        response = httpx.get(f"{API_BASE_URL}/admin/sync/status", timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise gr.Error(f"读取增量同步状态失败：{exc}") from exc
    return response.json()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ID检索") as demo:
        gr.Markdown("# ID检索")
        gr.Markdown(
            "输入检索内容（关键词/描述），按语义与关键词混合检索，返回对应的ID和关联信息。"
        )
        query = gr.Textbox(
            label=None,
            placeholder="例如: 以category_name为检索主体内容",
        )
        with gr.Row():
            top_n = gr.Slider(
                1, config.MAX_TOP_N, value=config.DEFAULT_TOP_N, step=1, label="返回数量",
            )
            vector_weight = gr.Slider(
                0, 1, value=config.CHROMA_DENSE_WEIGHT, step=0.1, label="语义权重",
            )
            search_button = gr.Button("搜索", variant="primary")
        with gr.Row():
            sync_button = gr.Button("手动同步新增数据")
            sync_status_button = gr.Button("刷新同步状态")
        sync_status = gr.JSON(label="增量同步状态", value={"status": "idle"})
        results = gr.HTML(
            value=render_results([]),
            label="检索结果",
            elem_id="results-table",
            elem_classes=["results-table"],
        )
        search_button.click(search, inputs=[query, top_n, vector_weight], outputs=results)
        query.submit(search, inputs=[query, top_n, vector_weight], outputs=results)
        sync_button.click(trigger_incremental_sync, outputs=sync_status)
        sync_status_button.click(get_incremental_sync_status, outputs=sync_status)
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=UI_CSS,
    )
