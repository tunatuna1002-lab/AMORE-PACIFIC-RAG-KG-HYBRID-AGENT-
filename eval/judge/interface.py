"""
Judge Interface
===============
Protocol for LLM-as-a-judge evaluation.

The judge provides subjective quality scores for:
- Groundedness: Is the answer supported by the provided context?
- Relevance: Does the answer address the question?
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class JudgeInterface(Protocol):
    """
    Protocol for LLM-as-a-judge implementations.

    Implementations should be async to support concurrent evaluation.
    """

    async def score_groundedness(self, answer: str, context: str) -> float:
        """
        Score how well the answer is grounded in the context.

        Args:
            answer: Generated answer text
            context: Retrieved context used for generation

        Returns:
            Score between 0.0 (not grounded) and 1.0 (fully grounded)
        """
        ...

    async def score_relevance(self, answer: str, question: str) -> float:
        """
        Score how relevant the answer is to the question.

        Args:
            answer: Generated answer text
            question: Original user question

        Returns:
            Score between 0.0 (not relevant) and 1.0 (fully relevant)
        """
        ...

    async def score_factuality(self, answer: str, facts: list[str]) -> tuple[float, list[str]]:
        """
        Score factual accuracy of the answer against known facts.

        Args:
            answer: Generated answer text
            facts: List of known facts to check against

        Returns:
            Tuple of (score, list of factual errors found)
        """
        ...
