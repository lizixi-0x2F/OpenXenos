"""
OpenXenos — Multi-model deliberation without a judge.

A thin Anthropic Messages API compatible server.
Claude Code points its Anthropic base_url here, we intercept the request,
run multi-model deliberation, and return an Anthropic-format response.

Usage:
    uv run uvicorn openxenos.server:app --host 0.0.0.0 --port 8787
    uv run openxenos

Claude Code config:
    base_url: http://localhost:8787/v1
    api_key:  anything (real key comes from ANTHROPIC_AUTH_TOKEN env var)
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openxenos")

from .client import LLMClient
from .config import PANEL_SIZE, MODEL, SERVER_HOST, SERVER_PORT
from .deliberation import deliberate
from .schemas import AnthropicRequest, AnthropicResponse, ContentBlock

# Global client, lazily initialized
_llm_client: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing to do on startup — client is lazily initialized
    yield
    # Cleanup on shutdown
    global _llm_client
    if _llm_client:
        await _llm_client.close()
        _llm_client = None


app = FastAPI(
    title="OpenXenos",
    description="Multi-model deliberation without a judge — Anthropic Messages API compatible",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def normalize_double_v1(request: Request, call_next):
    """Claude Code appends /v1/messages to base_url; if base_url ends with /v1
    we get /v1/v1/messages. Normalize it."""
    path = request.url.path
    if path.startswith("/v1/v1/"):
        request.scope["path"] = path[3:]  # strip the leading /v1
        request.scope["raw_path"] = request.scope["raw_path"][3:]
    response = await call_next(request)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request so we can see what Claude Code sends."""
    body = None
    if request.method == "POST":
        try:
            body = await request.body()
            body = body[:2000]  # truncate
        except Exception:
            pass
    logger.info(f"← {request.method} {request.url.path} body={body}")
    response = await call_next(request)
    logger.info(f"→ {response.status_code}")
    return response


def _get_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ═══════════════════════════════════════════════════════════════
# Anthropic Messages API compatible endpoint
# ═══════════════════════════════════════════════════════════════

@app.post("/v1/messages")
async def create_message(request: AnthropicRequest):
    """
    Anthropic Messages API compatible endpoint.

    Tool-using requests (Claude Code agent loop): pass through directly.
    Pure reasoning requests (no tools): run multi-model deliberation.
    """
    client = _get_client()

    messages_dicts = [msg.model_dump() for msg in request.messages]

    # ── Tool request → pass through, no deliberation ──
    if request.tools:
        logger.info("→ tools detected, pass-through (no deliberation)")
        try:
            direct = await client.messages(
                model=request.model,
                messages=messages_dicts,
                system=request.system,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
                stop_sequences=request.stop_sequences,
                thinking=request.thinking,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"API error: {e}")

        return JSONResponse(content={
            **direct.model_dump(),
            "model": request.model,
            "_openxenos": {"mode": "passthrough"},
        })

    # ── No tools → full deliberation ──
    logger.info("→ no tools, running deliberation")
    try:
        result = await deliberate(
            client=client,
            model=request.model,
            messages=messages_dicts,
            system=request.system,
            max_tokens=request.max_tokens,
            thinking=request.thinking,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deliberation failed: {e}")

    response = AnthropicResponse(
        id=f"msg_xenos_{uuid.uuid4().hex[:24]}",
        content=[ContentBlock(type="text", text=result.final_answer)],
        model=f"openxenos-{PANEL_SIZE}x-{client.default_model}",
        stop_reason="end_turn",
        usage=result.total_usage,
    )

    def _preview(text: str, maxlen: int = 300) -> str:
        return text[:maxlen] + "..." if len(text) > maxlen else text

    return JSONResponse(content={
        **response.model_dump(),
        "_openxenos": {
            "phase_1": [
                {
                    "model": pr.model,
                    "index": pr.index,
                    "preview": _preview(pr.content),
                }
                for pr in result.phase_1_responses
            ],
            "refinement_trace": [_preview(t) for t in result.refinement_trace],
            "failed_indices": result.failed_indices,
        },
    })


# ═══════════════════════════════════════════════════════════════
# Model list — so Claude Code knows this model exists
# ═══════════════════════════════════════════════════════════════

@app.get("/v1/models")
async def list_models():
    """
    Model-agnostic: accept whatever model the client sends.
    Return nothing — every model is valid.
    """
    return {"data": [], "has_more": False, "first_id": None}


# ═══════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "panel_size": PANEL_SIZE,
        "model": MODEL,
    }


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    """Entry point for `uv run openxenos`."""
    import uvicorn
    reload = os.getenv("OPENXENOS_RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run("openxenos.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=reload)


if __name__ == "__main__":
    main()
