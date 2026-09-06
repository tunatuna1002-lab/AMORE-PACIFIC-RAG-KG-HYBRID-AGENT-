"""
D10: ResponsePipeline mixes confidence scales and skips hallucination checks
============================================================================
``generate()`` did ``max(calculated (0-10), decision.confidence (0-1))`` so the
API-facing ``confidence_score`` could be e.g. 7.5, and HIGH-shortcut decisions
(confidence=0.9) bypassed the hallucination gate (``decision.confidence < 0.8``)
even when the underlying context was weak.

Fix contract:
- ``Response.confidence_score`` is always within [0.0, 1.0]
- it equals ``max(calculated / 10, decision.confidence)`` (times any penalty)
- ``confidence_level`` tier label is unchanged
- the hallucination check runs whenever the *normalized context score* is below
  the gate, even for a HIGH-confidence shortcut decision
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import ConfidenceLevel, Context, Decision, SystemState
from src.core.response_pipeline import ResponsePipeline


@pytest.fixture
def pipeline() -> ResponsePipeline:
    return ResponsePipeline(openai_client=MagicMock(), model="gpt-4o-mini")


def rich_context() -> Context:
    """3 rag docs + 3 kg facts + fresh + kg_initialized -> raw score 7.5 (of 10)."""
    return Context(
        query="LANEIGE 순위",
        entities={"brands": ["LANEIGE"]},
        rag_docs=[{"content": f"doc {i}"} for i in range(3)],
        kg_facts=[{"s": "LANEIGE", "p": "rank", "o": str(i)} for i in range(3)],
        kg_inferences=[],
        system_state=SystemState(data_freshness="fresh", kg_initialized=True, kg_triple_count=5),
        summary="LANEIGE는 Lip Care 1위입니다.",
    )


def weak_context() -> Context:
    """No signals at all -> raw score 0.0."""
    return Context(
        query="무슨 일이 있었나요",
        entities={},
        rag_docs=[],
        kg_facts=[],
        kg_inferences=[],
        system_state=None,
        summary="",
    )


def high_shortcut_decision() -> Decision:
    """Mirrors how brain.py / query_graph.py build the HIGH fast-path decision."""
    return Decision(
        tool="direct_answer",
        tool_params={},
        reason="HIGH confidence (high) - direct context answer",
        confidence=0.9,
        key_points=[],
    )


def _patched(pipeline: ResponsePipeline, text: str = "응답입니다."):
    """Patch both LLM entry points and the hallucination detector."""
    llm = patch.object(pipeline, "_call_llm", new_callable=AsyncMock, return_value=text)
    fast = patch.object(pipeline, "_call_llm_fast", new_callable=AsyncMock, return_value=text)
    check = patch.object(
        pipeline._hallucination_detector,
        "check",
        new_callable=AsyncMock,
        return_value=MagicMock(is_grounded=True, score=0.9),
    )
    return llm, fast, check


class TestUnitScale:
    @pytest.mark.asyncio
    async def test_confidence_without_decision_is_normalized(self, pipeline):
        llm, fast, check = _patched(pipeline)
        with llm, fast, check:
            result = await pipeline.generate("LANEIGE 순위", rich_context())

        assert 0.0 <= result.confidence_score <= 1.0
        assert result.confidence_score == pytest.approx(0.75)
        assert result.confidence_level == ConfidenceLevel.HIGH  # tier label untouched

    @pytest.mark.asyncio
    async def test_confidence_with_decision_is_max_on_unit_scale(self, pipeline):
        decision = Decision(tool="direct_answer", reason="llm", confidence=0.9)
        llm, fast, check = _patched(pipeline)
        with llm, fast, check:
            result = await pipeline.generate("LANEIGE 순위", rich_context(), decision=decision)

        assert result.confidence_score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_calculated_wins_when_higher_than_decision(self, pipeline):
        decision = Decision(tool="direct_answer", reason="llm", confidence=0.3)
        llm, fast, check = _patched(pipeline)
        with llm, fast, check:
            result = await pipeline.generate("LANEIGE 순위", rich_context(), decision=decision)

        assert result.confidence_score == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_out_of_range_decision_confidence_is_clamped(self, pipeline):
        decision = Decision(tool="direct_answer", reason="llm", confidence=8.0)
        llm, fast, check = _patched(pipeline)
        with llm, fast, check:
            result = await pipeline.generate("LANEIGE 순위", rich_context(), decision=decision)

        assert 0.0 <= result.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_penalty_keeps_score_in_range(self, pipeline):
        decision = Decision(tool="direct_answer", reason="llm", confidence=0.5)
        llm, fast, check = _patched(pipeline)
        check.new_callable = None
        with (
            llm,
            fast,
            patch.object(
                pipeline._hallucination_detector,
                "check",
                new_callable=AsyncMock,
                return_value=MagicMock(is_grounded=False, score=0.2),
            ),
        ):
            result = await pipeline.generate("LANEIGE 순위", weak_context(), decision=decision)

        assert result.grounding_warning is True
        assert 0.0 <= result.confidence_score <= 1.0


class TestHallucinationGate:
    @pytest.mark.asyncio
    async def test_high_shortcut_with_weak_context_still_checks(self, pipeline):
        llm, fast, check = _patched(pipeline)
        with llm, fast, check as mock_check:
            await pipeline.generate("질문", weak_context(), decision=high_shortcut_decision())

        mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_high_shortcut_with_strong_context_skips_check(self, pipeline):
        """Fast path preserved: strong context (0.75 >= gate? no) -> use a saturated context."""
        ctx = rich_context()
        ctx.kg_inferences = [{"rule": "r1"}, {"rule": "r2"}]  # +3.0 -> raw 10.0 -> 1.0
        llm, fast, check = _patched(pipeline)
        with llm, fast, check as mock_check:
            await pipeline.generate("질문", ctx, decision=high_shortcut_decision())

        mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_decision_confidence_still_checks(self, pipeline):
        decision = Decision(tool="direct_answer", reason="llm", confidence=0.5)
        llm, fast, check = _patched(pipeline)
        with llm, fast, check as mock_check:
            await pipeline.generate("질문", rich_context(), decision=decision)

        mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_shortcut_high_decision_with_weak_context_checks(self, pipeline):
        decision = Decision(tool="direct_answer", reason="llm", confidence=0.95)
        llm, fast, check = _patched(pipeline)
        with llm, fast, check as mock_check:
            await pipeline.generate("질문", weak_context(), decision=decision)

        mock_check.assert_called_once()
