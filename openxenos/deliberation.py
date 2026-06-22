"""
Dense multi-model deliberation with explicit DONE consensus.

Round 1: N models answer simultaneously — answers stream into a shared
         message pool via as_completed(). Dense: every model's output
         is immediately visible to the group.

Round 2+: All N models see the full accumulated discussion pool AND the
          original conversation context. Each decides:
            DONE → consensus reached, here's the final answer
            REVIEW → still disagree, here's my critique/improvement

          REVIEW means "keep discussing" — critiques are added to the pool.
          The first DONE wins, output immediately.
          Iterates until DONE or MAX_ROUNDS reached.
"""

from __future__ import annotations

import asyncio

from .client import LLMClient
from .config import (
    PANEL_SIZE,
    PHASE1_TEMPERATURE,
    PHASE2_TEMPERATURE,
    MAX_ROUNDS,
)
from .schemas import (
    AnthropicResponse,
    PeerResponse,
    DeliberationResult,
    UsageInfo,
)


def _extract_text(response: AnthropicResponse) -> str:
    parts = []
    for block in response.content:
        if block.type == "text" and block.text:
            parts.append(block.text)
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Discussion pool formatting
# ═══════════════════════════════════════════════════════════════

def _make_usage(input_tokens: int, output_tokens: int) -> UsageInfo:
    """Build a UsageInfo from separately-tracked input and output token counts."""
    return UsageInfo(input_tokens=input_tokens, output_tokens=output_tokens)


def _build_refinement_trace(all_verdicts: list[list[dict]]) -> list[str]:
    """Build the refinement trace from accumulated verdicts."""
    trace: list[str] = []
    for r, vs in enumerate(all_verdicts, start=2):
        for v in vs:
            trace.append(f"Round {r} [{v['signal']}]: {v['body'][:200]}")
    return trace


def _format_discussion(pool: list[dict]) -> str:
    """Format the accumulated discussion pool for model consumption."""
    sections = []
    for entry in pool:
        sections.append(f"### {entry['label']}\n\n{entry['content']}")
    return "\n\n---\n\n".join(sections)


def _format_original_context(messages: list[dict]) -> str:
    """Format the original conversation messages as readable context.

    Preserves role structure so models understand the conversation flow,
    including tool calls and tool results.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            blocks = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type", "")
                if bt == "text":
                    blocks.append(block.get("text", ""))
                elif bt == "tool_use":
                    blocks.append(
                        f"[Tool Call: {block.get('name', '?')}"
                        f"({_summarize_input(block.get('input', {}))})]"
                    )
                elif bt == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            r.get("text", "") if isinstance(r, dict) else str(r)
                            for r in result_content
                        )
                    blocks.append(
                        f"[Tool Result: {str(result_content)[:300]}]"
                    )
                elif bt == "image":
                    blocks.append("[Image]")
                elif bt == "thinking":
                    blocks.append("[Thinking block]")
                else:
                    blocks.append(f"[{bt}]")
            text = "\n".join(blocks)
        else:
            text = str(content)

        if text.strip():
            parts.append(f"**{role}**: {text}")

    return "\n\n".join(parts)


def _summarize_input(input_dict: dict, max_len: int = 120) -> str:
    """Brief summary of tool input for context display."""
    if not input_dict:
        return ""
    parts = []
    for k, v in input_dict.items():
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "..."
        parts.append(f"{k}={v_str}")
    result = ", ".join(parts)
    if len(result) > max_len:
        result = result[:max_len - 3] + "..."
    return result


# ═══════════════════════════════════════════════════════════════
# Round 1: async message passing — dense broadcast
# ═══════════════════════════════════════════════════════════════

async def _run_round1(
    client: LLMClient,
    model: str,
    messages: list[dict],
    system: str | list[dict] | None,
    max_tokens: int,
    thinking: dict | None = None,
) -> tuple[list[PeerResponse], list[int], int, int]:
    """
    Round 1 — all N models fire simultaneously.
    as_completed(): answers pool up as they arrive. Dense broadcast.

    Returns:
        (pool, failed_indices, total_input_tokens, total_output_tokens)
    """

    async def call_one(index: int):
        try:
            response = await client.messages(
                model=model,
                messages=messages,
                system=system,
                temperature=PHASE1_TEMPERATURE,
                max_tokens=max_tokens,
                thinking=thinking,
            )
            text = _extract_text(response)
            return index, PeerResponse(
                model=response.model,
                index=index,
                content=text,
            ), response.usage
        except Exception:
            return index, None, None

    tasks = [asyncio.create_task(call_one(i)) for i in range(PANEL_SIZE)]

    pool: list[PeerResponse] = []
    failed: list[int] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for coro in asyncio.as_completed(tasks):
        index, resp, usage = await coro
        if resp is not None:
            pool.append(resp)
            if usage:
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens
        else:
            failed.append(index)

    return pool, failed, total_input_tokens, total_output_tokens


# ═══════════════════════════════════════════════════════════════
# Convergence rounds — iterative DONE/REVIEW until consensus
# ═══════════════════════════════════════════════════════════════

CONVERGENCE_SYSTEM_SUFFIX = """

---

[DELIBERATION CONVERGENCE MODE]

You are now in a multi-model deliberation convergence round. The original
conversation (shown above) provides the full context of what the user needs.
Below that is the multi-model discussion that has taken place so far.

Your response MUST start with exactly one of these two signals on its own line:

DONE
<your final answer>

— use this if the group has clearly converged. The models substantially agree,
and you can write the definitive answer. This ends the deliberation.

REVIEW
<your critique and improved answer>

— use this if there are still substantive disagreements, gaps, or blind spots.
Explain what you disagree with, then write an improved answer.
Your critique will be shared with the group in the next round.

Do NOT use both signals. Pick one."""


def _build_convergence_prompt(
    messages: list[dict],
    discussion_pool: list[dict],
    round_num: int,
) -> str:
    """Build the convergence round prompt, preserving original conversation context."""
    original_context = _format_original_context(messages)

    return f"""## Original Conversation

{original_context}

---

## Multi-Model Deliberation (Rounds 1–{round_num - 1})

{_format_discussion(discussion_pool)}

---

## Your Task (Round {round_num})

Read the original conversation and the deliberation above. Decide:

- **DONE**: If the models have substantially converged (same core answer,
  minor phrasing differences don't count as disagreement), signal DONE
  and write the final answer.

- **REVIEW**: If there are still meaningful disagreements, blind spots, or
  unresolved issues, signal REVIEW. Explain the issue, then write an
  improved answer. Your critique will go back to the group.

Your response must start with `DONE` or `REVIEW` on its own line."""


async def _run_convergence_round(
    client: LLMClient,
    model: str,
    messages: list[dict],
    system: str | list[dict] | None,
    discussion_pool: list[dict],
    max_tokens: int,
    thinking: dict | None = None,
    round_num: int = 2,
) -> tuple[str | None, list[dict], int, int]:
    """
    One convergence round — all N models see the full accumulated discussion pool
    AND the original conversation context. Each signals DONE or REVIEW.

    Returns:
        (final_answer, verdicts, input_tokens, output_tokens)
        final_answer is None if no model said DONE (all REVIEW → keep discussing).
    """

    prompt = _build_convergence_prompt(messages, discussion_pool, round_num)

    # Merge original system with convergence instructions
    if system is None:
        merged_system = CONVERGENCE_SYSTEM_SUFFIX.strip()
    elif isinstance(system, str):
        merged_system = system + CONVERGENCE_SYSTEM_SUFFIX
    else:
        # system is a list of content blocks (Anthropic format)
        merged_system = list(system) + [
            {"type": "text", "text": CONVERGENCE_SYSTEM_SUFFIX}
        ]

    async def call_one():
        try:
            response = await client.messages(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                system=merged_system,
                temperature=PHASE2_TEMPERATURE,
                max_tokens=max_tokens,
                thinking=thinking,
            )
            text = _extract_text(response)
            return text, response.usage
        except Exception:
            return None, None

    results = await asyncio.gather(*[call_one() for _ in range(PANEL_SIZE)])

    verdicts: list[dict] = []
    round_input_tokens = 0
    round_output_tokens = 0

    for text, usage in results:
        if text and usage:
            round_input_tokens += usage.input_tokens
            round_output_tokens += usage.output_tokens
            signal = "REVIEW"  # default
            body = text
            if text.strip().upper().startswith("DONE"):
                signal = "DONE"
                body = text.strip()[4:].strip()
            elif text.strip().upper().startswith("REVIEW"):
                signal = "REVIEW"
                body = text.strip()[6:].strip()
            verdicts.append({"signal": signal, "body": body})

    # DONE consensus: first DONE wins, output immediately
    for v in verdicts:
        if v["signal"] == "DONE":
            return v["body"], verdicts, round_input_tokens, round_output_tokens

    # All REVIEW — no consensus yet, return None to keep discussing
    return None, verdicts, round_input_tokens, round_output_tokens


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def deliberate(
    client: LLMClient,
    model: str,
    messages: list[dict],
    system: str | list[dict] | None = None,
    max_tokens: int = 4096,
    thinking: dict | None = None,
) -> DeliberationResult:
    """
    Dense iterative deliberation with DONE consensus.

    Round 1 — N models answer simultaneously. Dense broadcast via as_completed().
              Each model sees the full original conversation context.
    Round 2+ — N models see full accumulated discussion AND original context,
               then signal DONE or REVIEW.
               REVIEW = keep discussing. DONE = output immediately.
               Iterates until DONE or MAX_ROUNDS reached.
    """

    # Round 1: dense broadcast (full original context)
    pool, failed, r1_input, r1_output = await _run_round1(
        client, model, messages, system, max_tokens, thinking,
    )

    if not pool:
        return DeliberationResult(
            phase_1_responses=[],
            final_answer="[All models failed]",
            total_usage=UsageInfo(),
            failed_indices=failed,
            refinement_trace=[],
        )

    if len(pool) == 1:
        return DeliberationResult(
            phase_1_responses=pool,
            final_answer=pool[0].content,
            total_usage=_make_usage(r1_input, r1_output),
            failed_indices=failed,
            refinement_trace=[],
        )

    # Build initial discussion pool from Round 1 answers
    discussion_pool: list[dict] = [
        {"label": f"Round 1 — Model {r.index + 1}", "content": r.content}
        for r in pool
    ]

    total_input_tokens = r1_input
    total_output_tokens = r1_output
    all_verdicts: list[list[dict]] = []

    # Iterative convergence — keep discussing until DONE or max rounds
    for round_num in range(2, MAX_ROUNDS + 1):
        final_answer, verdicts, r_input, r_output = await _run_convergence_round(
            client, model, messages, system, discussion_pool,
            max_tokens, thinking, round_num,
        )

        total_input_tokens += r_input
        total_output_tokens += r_output
        all_verdicts.append(verdicts)

        if final_answer is not None:
            # Someone said DONE — output immediately
            return DeliberationResult(
                phase_1_responses=pool,
                final_answer=final_answer,
                total_usage=_make_usage(total_input_tokens, total_output_tokens),
                failed_indices=failed,
                refinement_trace=_build_refinement_trace(all_verdicts),
            )

        # All REVIEW — add their critiques to the discussion pool for next round
        for i, v in enumerate(verdicts):
            discussion_pool.append({
                "label": f"Round {round_num} — Model {i + 1} REVIEW",
                "content": v["body"],
            })

    # MAX_ROUNDS reached without DONE — fallback to Round 1's first answer
    trace = _build_refinement_trace(all_verdicts)
    trace.append(f"MAX_ROUNDS ({MAX_ROUNDS}) reached, falling back to Round 1 answer")

    return DeliberationResult(
        phase_1_responses=pool,
        final_answer=pool[0].content,
        total_usage=_make_usage(total_input_tokens, total_output_tokens),
        failed_indices=failed,
        refinement_trace=trace,
    )
