"""
LLM-as-a-Judge
==============
Real LLM-based judge implementation using OpenAI GPT-4 or similar models.

This implementation follows the RAGAS pattern for groundedness/relevance scoring,
using structured prompts that elicit chain-of-thought reasoning followed by
a numeric score.

Usage:
    judge = LLMJudge(model="gpt-4.1-mini")
    score = await judge.score_groundedness(answer, context)

Cost tracking is built-in - call judge.get_usage() for token counts.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Try to import litellm for LLM calls
try:
    import litellm  # noqa: F401
    from litellm import acompletion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("litellm not installed. LLMJudge will not work.")


# =============================================================================
# Prompt Templates (RAGAS-style)
# =============================================================================

GROUNDEDNESS_PROMPT = """You are evaluating the groundedness of an answer based on the provided context.

Groundedness measures whether the claims in the answer can be attributed to and inferred from the given context.

**Context:**
{context}

**Answer:**
{answer}

**Instructions:**
1. Identify each distinct claim/statement in the answer
2. For each claim, determine if it can be verified from the context
3. Calculate the ratio of supported claims to total claims

**Response Format (JSON):**
{{
    "claims": [
        {{"claim": "claim text", "supported": true/false, "evidence": "quote from context or null"}}
    ],
    "supported_count": <number>,
    "total_claims": <number>,
    "score": <0.0 to 1.0>,
    "reasoning": "brief explanation"
}}

Respond with valid JSON only."""

RELEVANCE_PROMPT = """You are evaluating the relevance of an answer to a question.

Relevance measures how well the answer addresses what the user actually asked.

**Question:**
{question}

**Answer:**
{answer}

**Instructions:**
1. Identify the main intent/requirement of the question
2. Assess whether the answer addresses this intent
3. Check if the answer provides the specific information requested
4. Consider completeness - does it fully answer or only partially?

**Response Format (JSON):**
{{
    "question_intent": "what the user is asking for",
    "answer_addresses_intent": true/false,
    "completeness": "full/partial/minimal/none",
    "score": <0.0 to 1.0>,
    "reasoning": "brief explanation"
}}

Respond with valid JSON only."""

FACTUALITY_PROMPT = """You are checking the factual accuracy of an answer against known facts.

**Known Facts:**
{facts}

**Answer:**
{answer}

**Instructions:**
1. Extract factual claims from the answer
2. Check each claim against the known facts
3. Identify any claims that contradict the known facts

**Response Format (JSON):**
{{
    "claims_checked": [
        {{"claim": "claim text", "matches_facts": true/false, "contradiction": "description if false, null if true"}}
    ],
    "correct_claims": <number>,
    "total_claims": <number>,
    "score": <0.0 to 1.0>,
    "errors": ["list of factual errors found"]
}}

Respond with valid JSON only."""


class LLMJudge:
    """
    LLM-as-a-judge implementation using OpenAI GPT-4 or similar models.

    Features:
    - Structured JSON outputs for reliable parsing
    - Chain-of-thought reasoning for explainability
    - Token tracking for cost estimation
    - Retry logic for API failures
    """

    # Pricing per 1M tokens (input/output) - approximate for 2026
    MODEL_PRICING = {
        "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    }

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        api_key: str | None = None,
    ):
        """
        Initialize LLM Judge.

        Args:
            model: LLM model to use (via litellm)
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Number of retries for API failures
            api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError(
                "litellm is required for LLMJudge. Install with: pip install litellm"
            )

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Token usage tracking
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    async def score_groundedness(self, answer: str, context: str) -> float:
        """
        Score how well the answer is grounded in the context.

        Uses claim extraction and verification pattern from RAGAS.
        """
        if not answer.strip() or not context.strip():
            return 0.0

        prompt = GROUNDEDNESS_PROMPT.format(context=context, answer=answer)
        result = await self._call_llm(prompt)

        try:
            parsed = json.loads(result)
            score = float(parsed.get("score", 0.0))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse groundedness response: {e}")
            # Fallback: word overlap heuristic
            return self._word_overlap_score(answer, context)

    async def score_relevance(self, answer: str, question: str) -> float:
        """
        Score how relevant the answer is to the question.
        """
        if not answer.strip() or not question.strip():
            return 0.0

        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        result = await self._call_llm(prompt)

        try:
            parsed = json.loads(result)
            score = float(parsed.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse relevance response: {e}")
            # Fallback: word overlap heuristic
            return self._word_overlap_score(answer, question)

    async def score_factuality(self, answer: str, facts: list[str]) -> tuple[float, list[str]]:
        """
        Score factual accuracy of the answer against known facts.

        Returns:
            Tuple of (score, list of factual errors found)
        """
        if not answer.strip() or not facts:
            return 1.0, []  # No facts to check = assume correct

        facts_str = "\n".join(f"- {fact}" for fact in facts)
        prompt = FACTUALITY_PROMPT.format(facts=facts_str, answer=answer)
        result = await self._call_llm(prompt)

        try:
            parsed = json.loads(result)
            score = float(parsed.get("score", 1.0))
            errors = parsed.get("errors", [])
            return max(0.0, min(1.0, score)), errors
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse factuality response: {e}")
            return 0.5, []  # Neutral score on parse failure

    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with retry logic.

        Args:
            prompt: User prompt to send

        Returns:
            Model response text

        Raises:
            RuntimeError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                # Track usage
                usage = response.get("usage", {})
                self._usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self._usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self._usage["total_tokens"] += usage.get("total_tokens", 0)
                self._usage["calls"] += 1

                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    import asyncio

                    await asyncio.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def _word_overlap_score(self, text1: str, text2: str) -> float:
        """Fallback word overlap heuristic when LLM parsing fails."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_usage(self) -> dict[str, Any]:
        """Get token usage statistics."""
        usage = self._usage.copy()

        # Calculate estimated cost
        pricing = self.MODEL_PRICING.get(self.model, {"input": 0.15, "output": 0.60})
        input_cost = (usage["prompt_tokens"] / 1_000_000) * pricing["input"]
        output_cost = (usage["completion_tokens"] / 1_000_000) * pricing["output"]
        usage["estimated_cost_usd"] = input_cost + output_cost

        return usage

    def reset_usage(self) -> None:
        """Reset usage tracking."""
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    def __repr__(self) -> str:
        return f"LLMJudge(model={self.model!r}, calls={self._usage['calls']})"
