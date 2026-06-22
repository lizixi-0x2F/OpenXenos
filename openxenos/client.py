"""
Async LLM client — thin wrapper around the Anthropic Messages API.

Zero format conversion. Requests go in as Anthropic, come out as Anthropic.
Works with any Anthropic-compatible endpoint (Anthropic, DeepSeek, etc.).

Supports both batch (non-streaming) and SSE streaming modes.
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

import httpx

from .config import API_KEY, BASE_URL, MODEL
from .schemas import AnthropicResponse, ContentBlock, UsageInfo


def _make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


class LLMClient:
    """
    Thin Anthropic Messages API client.

    Usage:
        client = LLMClient()
        resp = await client.messages(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
            max_tokens=4096,
        )

        # Streaming
        async for chunk in client.messages_stream(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=4096,
        ):
            ...
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        base_url: str = BASE_URL,
        default_model: str = MODEL,
        timeout: float = 600.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_body(
        self,
        model: str | None,
        messages: list[dict],
        system: str | list[dict] | None,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None,
        tool_choice: dict | str | None,
        stop_sequences: list[str] | None,
        top_p: float | None,
        top_k: int | None,
        thinking: dict | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request body shared by batch and streaming calls."""
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            body["stream"] = True

        if system is not None:
            body["system"] = system
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stop_sequences is not None:
            body["stop_sequences"] = stop_sequences
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if thinking is not None:
            body["thinking"] = thinking

        return body

    async def messages(
        self,
        *,
        model: str | None = None,
        messages: list[dict],
        system: str | list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
        stop_sequences: list[str] | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        thinking: dict | None = None,
    ) -> AnthropicResponse:
        """
        Send an Anthropic Messages API request (batch), return an AnthropicResponse.
        """
        client = await self._get_client()

        body = self._build_body(
            model=model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stop_sequences=stop_sequences,
            top_p=top_p,
            top_k=top_k,
            thinking=thinking,
            stream=False,
        )

        resp = await client.post("/v1/messages", json=body)
        resp.raise_for_status()
        data = resp.json()

        # Parse into AnthropicResponse
        content_blocks = [ContentBlock(**block) for block in data.get("content", [])]

        usage_raw = data.get("usage", {})
        usage = UsageInfo(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
        )

        return AnthropicResponse(
            id=data.get("id", _make_msg_id()),
            type=data.get("type", "message"),
            role=data.get("role", "assistant"),
            content=content_blocks,
            model=data.get("model", model or self.default_model),
            stop_reason=data.get("stop_reason", "end_turn"),
            stop_sequence=data.get("stop_sequence"),
            usage=usage,
        )

    async def messages_stream(
        self,
        *,
        model: str | None = None,
        messages: list[dict],
        system: str | list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
        stop_sequences: list[str] | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        thinking: dict | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream SSE bytes from the Anthropic-compatible endpoint.

        Yields raw bytes chunks that can be forwarded directly in a StreamingResponse.
        The upstream API must support `stream: true` and return SSE events.
        """
        client = await self._get_client()

        body = self._build_body(
            model=model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stop_sequences=stop_sequences,
            top_p=top_p,
            top_k=top_k,
            thinking=thinking,
            stream=True,
        )

        async with client.stream("POST", "/v1/messages", json=body) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk
