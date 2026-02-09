"""
Stub Judge
==========
Placeholder judge implementation for offline evaluation without network calls.

Returns neutral scores (0.5) for all judge-based metrics.
"""

import logging

logger = logging.getLogger(__name__)


class StubJudge:
    """
    Stub implementation of JudgeInterface for offline evaluation.

    Returns placeholder scores without making any API calls.
    Useful for:
    - Running evaluation without LLM costs
    - Testing the evaluation pipeline
    - CI/CD environments without API access
    """

    def __init__(self, default_score: float = 0.5):
        """
        Initialize stub judge.

        Args:
            default_score: Default score to return (0.5 = neutral)
        """
        self.default_score = default_score
        self._call_count = 0

    async def score_groundedness(self, answer: str, context: str) -> float:
        """
        Return placeholder groundedness score.

        The stub uses heuristics to provide slightly better than random scores:
        - Returns higher score if answer words appear in context
        - Returns lower score for very short answers
        """
        self._call_count += 1

        if not answer or not context:
            return 0.3

        # Simple heuristic: check word overlap
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.3

        overlap = len(answer_words & context_words)
        overlap_ratio = overlap / len(answer_words)

        # Scale to 0.3-0.7 range (neutral zone)
        return 0.3 + (overlap_ratio * 0.4)

    async def score_relevance(self, answer: str, question: str) -> float:
        """
        Return placeholder relevance score.

        Uses simple heuristics:
        - Returns higher score if question keywords appear in answer
        - Returns lower score for very short answers
        """
        self._call_count += 1

        if not answer or not question:
            return 0.3

        # Extract question keywords (filter common words)
        stopwords = {"is", "the", "a", "an", "what", "how", "why", "when", "where", "who"}
        question_words = {
            w.lower().strip("?.,!") for w in question.split() if w.lower() not in stopwords
        }
        answer_words = set(answer.lower().split())

        if not question_words:
            return self.default_score

        overlap = len(question_words & answer_words)
        overlap_ratio = overlap / len(question_words)

        # Scale to 0.3-0.7 range (neutral zone)
        return 0.3 + (overlap_ratio * 0.4)

    async def score_factuality(self, answer: str, facts: list[str]) -> tuple[float, list[str]]:
        """
        Return placeholder factuality score.

        Stub always returns neutral score with no errors detected.
        """
        self._call_count += 1

        if not answer or not facts:
            return self.default_score, []

        # Simple heuristic: check if fact keywords appear in answer
        answer_lower = answer.lower()
        matched_facts = 0

        for fact in facts:
            fact_words = fact.lower().split()
            if any(word in answer_lower for word in fact_words):
                matched_facts += 1

        if not facts:
            return self.default_score, []

        score = matched_facts / len(facts)
        return max(0.3, score), []

    def get_stats(self) -> dict[str, int | float]:
        """Get stub usage statistics."""
        return {
            "call_count": self._call_count,
            "default_score": self.default_score,
        }
