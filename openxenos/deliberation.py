"""
Dense multi-model deliberation with explicit DONE consensus.

Round 1: N models answer simultaneously — answers stream into a shared
         message pool via as_completed(). Dense: every model's output
         is immediately visible to the group.

Round 2+: All N models see the full accumulated discussion pool.
          Each decides:
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


def _extract_question_text(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


# ═══════════════════════════════════════════════════════════════
# Discussion pool formatting
# ═══════════════════════════════════════════════════════════════

def _make_usage(total: int) -> UsageInfo:
    """Build a UsageInfo from a combined input+output token count."""
    return UsageInfo(input_tokens=total, output_tokens=total)


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
) -> tuple[list[PeerResponse], list[int], int]:
    """
    Round 1 — all N models fire simultaneously.
    as_completed(): answers pool up as they arrive. Dense broadcast.
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
    total_tokens = 0

    for coro in asyncio.as_completed(tasks):
        index, resp, usage = await coro
        if resp is not None:
            pool.append(resp)
            if usage:
                total_tokens += usage.input_tokens + usage.output_tokens
        else:
            failed.append(index)

    return pool, failed, total_tokens


# ═══════════════════════════════════════════════════════════════
# Convergence rounds — iterative DONE/REVIEW until consensus
# ═══════════════════════════════════════════════════════════════

CONVERGENCE_SYSTEM = """You are in a multi-model deliberation that may span several rounds.

You will see a discussion history: the original question, Round 1 answers from
multiple models, and possibly REVIEW critiques from earlier rounds.

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


def _build_convergence_prompt(question: str, discussion_pool: list[dict], round_num: int) -> str:
    return f"""## Original Question

{question}

---

## Discussion So Far (Round 1–{round_num - 1})

{_format_discussion(discussion_pool)}

---

## Your Task (Round {round_num})

Read the discussion above. Decide:

- **DONE**: If the models have substantially converged (same core answer, minor phrasing differences don't count as disagreement), signal DONE and write the final answer.

- **REVIEW**: If there are still meaningful disagreements, blind spots, or unresolved issues, signal REVIEW. Explain the issue, then write an improved answer. Your critique will go back to the group.

Your response must start with `DONE` or `REVIEW` on its own line."""


async def _run_convergence_round(
    client: LLMClient,
    model: str,
    question: str,
    discussion_pool: list[dict],
    max_tokens: int,
    thinking: dict | None = None,
    round_num: int = 2,
) -> tuple[str | None, list[dict], int]:
    """
    One convergence round — all N models see the full accumulated discussion pool.
    Each signals DONE or REVIEW.

    Returns:
        (final_answer, verdicts, tokens)
        final_answer is None if no model said DONE (all REVIEW → keep discussing).
    """

    prompt = _build_convergence_prompt(question, discussion_pool, round_num)

    async def call_one():
        try:
            response = await client.messages(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                system=CONVERGENCE_SYSTEM,
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
    total_tokens = 0

    for text, usage in results:
        if text and usage:
            total_tokens += usage.input_tokens + usage.output_tokens
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
            return v["body"], verdicts, total_tokens

    # All REVIEW — no consensus yet, return None to keep discussing
    return None, verdicts, total_tokens


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
    Round 2+ — N models see full accumulated discussion, signal DONE or REVIEW.
               REVIEW = keep discussing. DONE = output immediately.
               Iterates until DONE or MAX_ROUNDS reached.
    """

    # Round 1: dense broadcast
    pool, failed, r1_tokens = await _run_round1(
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
            total_usage=_make_usage(r1_tokens),
            failed_indices=failed,
            refinement_trace=[],
        )

    question = _extract_question_text(messages)

    # Build initial discussion pool from Round 1 answers
    discussion_pool: list[dict] = [
        {"label": f"Round 1 — Model {r.index + 1}", "content": r.content}
        for r in pool
    ]

    total_tokens = r1_tokens
    all_verdicts: list[list[dict]] = []

    # Iterative convergence — keep discussing until DONE or max rounds
    for round_num in range(2, MAX_ROUNDS + 1):
        final_answer, verdicts, round_tokens = await _run_convergence_round(
            client, model, question, discussion_pool, max_tokens, thinking, round_num,
        )

        total_tokens += round_tokens
        all_verdicts.append(verdicts)

        if final_answer is not None:
            # Someone said DONE — output immediately
            return DeliberationResult(
                phase_1_responses=pool,
                final_answer=final_answer,
                total_usage=_make_usage(total_tokens),
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
        total_usage=_make_usage(total_tokens),
        failed_indices=failed,
        refinement_trace=trace,
    )
