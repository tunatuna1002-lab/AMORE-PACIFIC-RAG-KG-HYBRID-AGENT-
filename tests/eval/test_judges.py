"""Tests for judge implementations (LLMJudge, NLIJudge, StubJudge, create_judge factory)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval.judge.stub import StubJudge

# =============================================================================
# StubJudge Tests
# =============================================================================


class TestStubJudge:
    """Tests for StubJudge."""

    @pytest.fixture
    def judge(self):
        return StubJudge()

    @pytest.mark.asyncio
    async def test_groundedness_with_overlap(self, judge):
        """High word overlap answer gets higher score."""
        answer = "LANEIGE Lip Sleeping Mask is ranked #1 in Lip Care"
        context = "LANEIGE Lip Sleeping Mask holds the #1 position in Lip Care category"
        score = await judge.score_groundedness(answer, context)
        assert 0.3 <= score <= 0.7
        # High overlap should push score toward upper range
        assert score > 0.4

    @pytest.mark.asyncio
    async def test_groundedness_no_overlap(self, judge):
        """No word overlap gets low score."""
        answer = "The weather is sunny today"
        context = "LANEIGE products dominate lip care rankings"
        score = await judge.score_groundedness(answer, context)
        assert 0.3 <= score <= 0.7

    @pytest.mark.asyncio
    async def test_relevance_with_keywords(self, judge):
        """Answer containing question keywords scores higher."""
        question = "What is LANEIGE market share in Lip Care?"
        answer = "LANEIGE has a 5.2% market share in the Lip Care category"
        score = await judge.score_relevance(answer, question)
        assert 0.3 <= score <= 0.7
        assert score > 0.4

    @pytest.mark.asyncio
    async def test_relevance_no_keywords(self, judge):
        """Answer without question keywords scores lower."""
        question = "What is LANEIGE market share in Lip Care?"
        answer = "The sun rises in the east and sets in the west"
        score = await judge.score_relevance(answer, question)
        assert 0.3 <= score <= 0.7

    @pytest.mark.asyncio
    async def test_factuality_with_matching_facts(self, judge):
        """Facts matching answer text score well."""
        answer = "LANEIGE SoS is 5.2% in Lip Care"
        facts = ["LANEIGE SoS is 5.2%", "Lip Care category"]
        score, errors = await judge.score_factuality(answer, facts)
        assert score >= 0.3
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_factuality_no_facts(self, judge):
        """No facts returns default score."""
        score, errors = await judge.score_factuality("some answer", [])
        assert score == judge.default_score
        assert errors == []

    @pytest.mark.asyncio
    async def test_empty_answer(self, judge):
        """Empty answer returns low score."""
        score = await judge.score_groundedness("", "some context")
        assert score == 0.3

    @pytest.mark.asyncio
    async def test_empty_context(self, judge):
        """Empty context returns low score."""
        score = await judge.score_groundedness("some answer", "")
        assert score == 0.3

    @pytest.mark.asyncio
    async def test_empty_question(self, judge):
        """Empty question returns low score."""
        score = await judge.score_relevance("some answer", "")
        assert score == 0.3

    def test_stats_tracking(self):
        """Stats track call count."""
        judge = StubJudge()
        assert judge.get_stats()["call_count"] == 0

        import asyncio

        asyncio.get_event_loop().run_until_complete(judge.score_groundedness("a", "b"))
        asyncio.get_event_loop().run_until_complete(judge.score_relevance("a", "b"))
        assert judge.get_stats()["call_count"] == 2

    def test_custom_default_score(self):
        """Custom default score is respected."""
        judge = StubJudge(default_score=0.7)
        assert judge.default_score == 0.7


# =============================================================================
# LLMJudge Tests
# =============================================================================


class TestLLMJudge:
    """Tests for LLMJudge with mocked API calls."""

    @pytest.fixture
    def mock_acompletion(self):
        """Mock litellm.acompletion."""
        with patch("eval.judge.llm.acompletion", new_callable=AsyncMock) as mock:
            yield mock

    @pytest.fixture
    def mock_litellm_available(self):
        """Ensure LITELLM_AVAILABLE is True for tests."""
        with patch("eval.judge.llm.LITELLM_AVAILABLE", True):
            yield

    def _make_response(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        """Create mock LLM response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.get.return_value = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return response

    @pytest.mark.asyncio
    async def test_groundedness_high_score(self, mock_acompletion, mock_litellm_available):
        """Grounded answer gets high score from LLM."""
        from eval.judge.llm import LLMJudge

        mock_acompletion.return_value = self._make_response(
            json.dumps(
                {
                    "claims": [{"claim": "test", "supported": True, "evidence": "found"}],
                    "supported_count": 1,
                    "total_claims": 1,
                    "score": 0.95,
                    "reasoning": "All claims supported",
                }
            )
        )

        judge = LLMJudge(model="gpt-4.1-mini")
        score = await judge.score_groundedness(
            answer="LANEIGE is #1 in Lip Care",
            context="LANEIGE holds the #1 position in Lip Care category",
        )
        assert score == pytest.approx(0.95)
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_groundedness_low_score(self, mock_acompletion, mock_litellm_available):
        """Hallucinated answer gets low score."""
        from eval.judge.llm import LLMJudge

        mock_acompletion.return_value = self._make_response(
            json.dumps(
                {
                    "claims": [{"claim": "test", "supported": False, "evidence": None}],
                    "supported_count": 0,
                    "total_claims": 1,
                    "score": 0.1,
                    "reasoning": "Claim not supported",
                }
            )
        )

        judge = LLMJudge(model="gpt-4.1-mini")
        score = await judge.score_groundedness(
            answer="LANEIGE has 50% market share",
            context="LANEIGE holds 5.2% share of shelf",
        )
        assert score == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, mock_acompletion, mock_litellm_available):
        """Relevance scoring works with mock."""
        from eval.judge.llm import LLMJudge

        mock_acompletion.return_value = self._make_response(
            json.dumps(
                {
                    "question_intent": "market share",
                    "answer_addresses_intent": True,
                    "completeness": "full",
                    "score": 0.9,
                    "reasoning": "Answer fully addresses question",
                }
            )
        )

        judge = LLMJudge(model="gpt-4.1-mini")
        score = await judge.score_relevance(
            answer="LANEIGE SoS is 5.2%",
            question="What is LANEIGE market share?",
        )
        assert score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_factuality_scoring(self, mock_acompletion, mock_litellm_available):
        """Factuality scoring returns score and errors."""
        from eval.judge.llm import LLMJudge

        mock_acompletion.return_value = self._make_response(
            json.dumps(
                {
                    "claims_checked": [
                        {"claim": "SoS is 5.2%", "matches_facts": True, "contradiction": None}
                    ],
                    "correct_claims": 1,
                    "total_claims": 1,
                    "score": 1.0,
                    "errors": [],
                }
            )
        )

        judge = LLMJudge(model="gpt-4.1-mini")
        score, errors = await judge.score_factuality(
            answer="LANEIGE SoS is 5.2%",
            facts=["LANEIGE SoS is 5.2%"],
        )
        assert score == pytest.approx(1.0)
        assert errors == []

    @pytest.mark.asyncio
    async def test_api_error_fallback(self, mock_acompletion, mock_litellm_available):
        """Falls back to word overlap on API error."""
        from eval.judge.llm import LLMJudge

        mock_acompletion.side_effect = Exception("API rate limit")

        judge = LLMJudge(model="gpt-4.1-mini", max_retries=1)
        # Should not raise, should fall back
        with pytest.raises(RuntimeError):
            await judge.score_groundedness(
                answer="LANEIGE is great",
                context="LANEIGE products are popular",
            )

    @pytest.mark.asyncio
    async def test_empty_inputs(self, mock_litellm_available):
        """Empty answer/context returns 0.0 without API call."""
        from eval.judge.llm import LLMJudge

        judge = LLMJudge(model="gpt-4.1-mini")
        score = await judge.score_groundedness("", "some context")
        assert score == 0.0

        score = await judge.score_groundedness("some answer", "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_relevance_inputs(self, mock_litellm_available):
        """Empty inputs for relevance return 0.0."""
        from eval.judge.llm import LLMJudge

        judge = LLMJudge(model="gpt-4.1-mini")
        score = await judge.score_relevance("", "question")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_factuality_no_facts(self, mock_litellm_available):
        """No facts returns 1.0 (assume correct)."""
        from eval.judge.llm import LLMJudge

        judge = LLMJudge(model="gpt-4.1-mini")
        score, errors = await judge.score_factuality("some answer", [])
        assert score == 1.0
        assert errors == []

    def test_usage_tracking(self, mock_litellm_available):
        """Usage statistics are tracked correctly."""
        from eval.judge.llm import LLMJudge

        judge = LLMJudge(model="gpt-4.1-mini")
        usage = judge.get_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["calls"] == 0
        assert "estimated_cost_usd" in usage

    def test_reset_usage(self, mock_litellm_available):
        """Usage can be reset."""
        from eval.judge.llm import LLMJudge

        judge = LLMJudge(model="gpt-4.1-mini")
        judge._usage["calls"] = 5
        judge._usage["prompt_tokens"] = 1000
        judge.reset_usage()
        assert judge._usage["calls"] == 0
        assert judge._usage["prompt_tokens"] == 0

    def test_repr(self, mock_litellm_available):
        """Repr includes model and calls."""
        from eval.judge.llm import LLMJudge

        judge = LLMJudge(model="gpt-4.1-mini")
        assert "gpt-4.1-mini" in repr(judge)

    @pytest.mark.asyncio
    async def test_json_parse_failure_fallback(self, mock_acompletion, mock_litellm_available):
        """Invalid JSON response falls back to word overlap."""
        from eval.judge.llm import LLMJudge

        mock_acompletion.return_value = self._make_response("not valid json {{{")

        judge = LLMJudge(model="gpt-4.1-mini")
        score = await judge.score_groundedness(
            answer="LANEIGE is popular in Lip Care",
            context="LANEIGE dominates the Lip Care market with popular products",
        )
        # Should use word overlap fallback, getting a non-zero score
        assert 0.0 <= score <= 1.0


# =============================================================================
# NLIJudge Tests
# =============================================================================


class TestNLIJudge:
    """Tests for NLIJudge (using fallback mode since transformers may not be installed)."""

    @pytest.fixture
    def judge(self):
        """Create NLI judge (will use fallback if transformers unavailable)."""
        from eval.judge.nli import NLIJudge

        return NLIJudge()

    @pytest.mark.asyncio
    async def test_groundedness_fallback_mode(self, judge):
        """Groundedness works in fallback mode."""
        score = await judge.score_groundedness(
            answer="LANEIGE Lip Sleeping Mask is the top product in Lip Care",
            context="LANEIGE Lip Sleeping Mask holds the #1 position in Lip Care",
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_relevance_fallback_mode(self, judge):
        """Relevance works in fallback mode."""
        score = await judge.score_relevance(
            answer="LANEIGE has 5.2% SoS",
            question="What is LANEIGE market share?",
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_factuality_fallback_mode(self, judge):
        """Factuality works in fallback mode."""
        score, contradictions = await judge.score_factuality(
            answer="LANEIGE SoS is 5.2%",
            facts=["LANEIGE SoS is 5.2%", "Lip Care category"],
        )
        assert 0.0 <= score <= 1.0
        assert isinstance(contradictions, list)

    @pytest.mark.asyncio
    async def test_empty_answer(self, judge):
        """Empty answer returns 0.0."""
        score = await judge.score_groundedness("", "some context")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_context(self, judge):
        """Empty context returns 0.0."""
        score = await judge.score_groundedness("some answer", "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_question(self, judge):
        """Empty question returns 0.0."""
        score = await judge.score_relevance("some answer", "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_factuality_no_facts(self, judge):
        """No facts returns 1.0."""
        score, contradictions = await judge.score_factuality("answer", [])
        assert score == 1.0
        assert contradictions == []

    def test_split_sentences(self, judge):
        """Sentence splitting works correctly."""
        sentences = judge._split_sentences("First sentence. Second sentence! Third?")
        assert len(sentences) == 3
        assert sentences[0] == "First sentence"
        assert sentences[1] == "Second sentence"
        assert sentences[2] == "Third"

    def test_split_sentences_empty(self, judge):
        """Empty text returns empty list."""
        sentences = judge._split_sentences("")
        assert sentences == []

    def test_question_to_premise_what_is(self, judge):
        """'What is X?' converted correctly."""
        premise = judge._question_to_premise("What is LANEIGE market share?")
        assert "explains" in premise.lower() or "laneige" in premise.lower()

    def test_question_to_premise_what_are(self, judge):
        """'What are X?' converted correctly."""
        premise = judge._question_to_premise("What are the top brands?")
        assert "explains" in premise.lower() or "brands" in premise.lower()

    def test_question_to_premise_how(self, judge):
        """'How...' converted to method/process premise."""
        premise = judge._question_to_premise("How does LANEIGE compete?")
        assert "method" in premise.lower() or "process" in premise.lower()

    def test_question_to_premise_why(self, judge):
        """'Why...' converted to reasons premise."""
        premise = judge._question_to_premise("Why is LANEIGE popular?")
        assert "reason" in premise.lower() or "cause" in premise.lower()

    def test_question_to_premise_when(self, judge):
        """'When...' converted to time premise."""
        premise = judge._question_to_premise("When did LANEIGE launch?")
        assert "time" in premise.lower() or "date" in premise.lower()

    def test_question_to_premise_where(self, judge):
        """'Where...' converted to location premise."""
        premise = judge._question_to_premise("Where is LANEIGE sold?")
        assert "location" in premise.lower()

    def test_question_to_premise_who(self, judge):
        """'Who...' converted to people premise."""
        premise = judge._question_to_premise("Who makes LANEIGE?")
        assert "people" in premise.lower() or "entities" in premise.lower()

    def test_question_to_premise_generic(self, judge):
        """Generic question converted to topic premise."""
        premise = judge._question_to_premise("Tell me about LANEIGE")
        assert "topic" in premise.lower() or "addresses" in premise.lower()

    def test_word_overlap_score(self, judge):
        """Word overlap heuristic computes correctly."""
        score = judge._word_overlap_score(
            "LANEIGE lip care products",
            "LANEIGE lip care category ranking",
        )
        assert 0.0 < score < 1.0
        # Should have overlap on "LANEIGE", "lip", "care"
        assert score > 0.3

    def test_word_overlap_empty(self, judge):
        """Empty inputs return 0.0."""
        assert judge._word_overlap_score("", "something") == 0.0
        assert judge._word_overlap_score("something", "") == 0.0
        assert judge._word_overlap_score("", "") == 0.0

    def test_usage_tracking(self, judge):
        """Usage stats are tracked."""
        usage = judge.get_usage()
        assert "pairs_evaluated" in usage
        assert "chunks_processed" in usage
        assert usage["pairs_evaluated"] == 0

    def test_reset_usage(self, judge):
        """Usage can be reset."""
        judge._usage["pairs_evaluated"] = 10
        judge.reset_usage()
        assert judge._usage["pairs_evaluated"] == 0

    def test_repr(self, judge):
        """Repr includes model name and status."""
        r = repr(judge)
        assert "NLIJudge" in r
        assert "nli-deberta" in r or "fallback" in r


# =============================================================================
# Judge Factory Tests
# =============================================================================


class TestJudgeFactory:
    """Tests for create_judge factory function."""

    def test_create_stub(self):
        """create_judge('stub') returns StubJudge."""
        from eval.judge import create_judge

        judge = create_judge("stub")
        assert isinstance(judge, StubJudge)

    def test_create_default_is_stub(self):
        """Default judge type is stub."""
        from eval.judge import create_judge

        judge = create_judge()
        assert isinstance(judge, StubJudge)

    def test_create_stub_with_kwargs(self):
        """Stub judge accepts kwargs."""
        from eval.judge import create_judge

        judge = create_judge("stub", default_score=0.7)
        assert isinstance(judge, StubJudge)
        assert judge.default_score == 0.7

    def test_invalid_type_raises(self):
        """Invalid judge type raises ValueError."""
        from eval.judge import create_judge

        with pytest.raises(ValueError, match="Unknown judge_type"):
            create_judge("invalid_type")

    def test_create_nli_returns_judge(self):
        """NLI creation returns a judge (NLIJudge or StubJudge fallback)."""
        from eval.judge import create_judge

        judge = create_judge("nli")
        # Should return either NLIJudge or StubJudge (fallback)
        assert hasattr(judge, "score_groundedness")
        assert hasattr(judge, "score_relevance")
        assert hasattr(judge, "score_factuality")

    def test_create_llm_returns_judge(self):
        """LLM creation returns a judge (LLMJudge or StubJudge fallback)."""
        from eval.judge import create_judge

        # Without API key, may fall back to stub
        judge = create_judge("llm")
        assert hasattr(judge, "score_groundedness")
        assert hasattr(judge, "score_relevance")
        assert hasattr(judge, "score_factuality")


# =============================================================================
# JudgeInterface Protocol Tests
# =============================================================================


class TestJudgeInterface:
    """Tests for JudgeInterface protocol compliance."""

    def test_stub_implements_protocol(self):
        """StubJudge implements JudgeInterface."""
        from eval.judge.interface import JudgeInterface

        judge = StubJudge()
        assert isinstance(judge, JudgeInterface)

    def test_llm_implements_protocol(self):
        """LLMJudge implements JudgeInterface."""
        from eval.judge.interface import JudgeInterface

        with patch("eval.judge.llm.LITELLM_AVAILABLE", True):
            from eval.judge.llm import LLMJudge

            judge = LLMJudge(model="gpt-4.1-mini")
            assert isinstance(judge, JudgeInterface)

    def test_nli_implements_protocol(self):
        """NLIJudge implements JudgeInterface."""
        from eval.judge.interface import JudgeInterface
        from eval.judge.nli import NLIJudge

        judge = NLIJudge()
        assert isinstance(judge, JudgeInterface)
