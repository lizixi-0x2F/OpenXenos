"""
Data models — Anthropic Messages API compatible request/response,
plus internal deliberation protocol structures.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════
# Anthropic Messages API models
# ═══════════════════════════════════════════════════════════════

class ContentBlock(BaseModel):
    """Anthropic content block. Supports text / tool_use / thinking types."""
    type: str
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    # tool_result fields
    tool_use_id: str | None = None
    content: str | list[dict[str, Any]] | None = None
    # image fields
    source: dict[str, Any] | None = None


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str | list[ContentBlock | dict[str, Any]]


class AnthropicRequest(BaseModel):
    """Anthropic Messages API request body."""
    model_config = {"extra": "allow"}  # accept any extra fields CC sends
    model: str
    messages: list[Message]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 4096
    temperature: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    metadata: dict[str, Any] | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    top_p: float | None = None
    top_k: int | None = None
    thinking: dict[str, Any] | None = None


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    """Anthropic Messages API response body."""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: UsageInfo


# ═══════════════════════════════════════════════════════════════
# Internal deliberation protocol
# ═══════════════════════════════════════════════════════════════

class PeerResponse(BaseModel):
    """A single model's Phase 1 answer — the communication unit between models."""
    model: str
    index: int
    content: str


class DeliberationResult(BaseModel):
    """Complete deliberation result — single final answer from the refinement chain."""
    phase_1_responses: list[PeerResponse]
    final_answer: str
    total_usage: UsageInfo
    failed_indices: list[int]
    refinement_trace: list[str] = []
