"""
L5 Answer Metrics
=================
Final answer quality metrics.

Measures answer quality:
- Exact Match: Normalized exact match against gold answer
- Token F1: Token-level F1 score
- Semantic Similarity: Embedding-based similarity score (optional)
- Groundedness Score: LLM-as-judge score for context grounding (optional)
- Answer Relevance Score: LLM-as-judge score for question relevance (optional)
"""

import re

from eval.judge.interface import JudgeInterface
from eval.judge.stub import StubJudge
from eval.metrics.base import MetricCalculator
from eval.schemas import AnswerTrace, GoldEvidence, L5Metrics


class L5AnswerMetrics(MetricCalculator):
    """
    L5 metrics for final answer quality.

    Combines label-based metrics (exact match, F1) with optional
    semantic similarity and judge-based metrics (groundedness, relevance).
    """

    def __init__(
        self,
        judge: JudgeInterface | None = None,
        use_semantic_similarity: bool = False,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize L5 metrics calculator.

        Args:
            judge: JudgeInterface implementation (uses StubJudge if not provided)
            use_semantic_similarity: Whether to compute semantic similarity
            semantic_model: Model name for semantic similarity
        """
        self.judge = judge or StubJudge()
        self.use_semantic_similarity = use_semantic_similarity
        self._semantic_calc = None
        self._semantic_model = semantic_model

    def _get_semantic_calculator(self):
        """Lazy load semantic similarity calculator."""
        if self._semantic_calc is None:
            from eval.metrics.semantic import SemanticSimilarity

            self._semantic_calc = SemanticSimilarity(self._semantic_model)
        return self._semantic_calc

    async def compute(
        self,
        trace: AnswerTrace,
        gold: GoldEvidence,
        question: str,
        context: str,
        use_judge: bool = False,
    ) -> L5Metrics:
        """
        Compute L5 metrics.

        Args:
            trace: Answer trace from evaluation
            gold: Gold standard evidence
            question: Original question (for relevance scoring)
            context: Retrieved context (for groundedness scoring)
            use_judge: Whether to use LLM judge for scoring

        Returns:
            L5Metrics with all answer quality scores
        """
        exact_match = self._compute_exact_match(trace, gold)
        token_f1 = self._compute_token_f1(trace, gold)

        # Semantic similarity (optional)
        semantic_sim = None
        if self.use_semantic_similarity and gold.answer:
            calc = self._get_semantic_calculator()
            semantic_sim = calc.compute(trace.final_answer, gold.answer)

        groundedness = None
        relevance = None

        if use_judge:
            groundedness = await self.judge.score_groundedness(trace.final_answer, context)
            relevance = await self.judge.score_relevance(trace.final_answer, question)

        return L5Metrics(
            answer_exact_match=exact_match,
            answer_f1=token_f1,
            semantic_similarity=semantic_sim,
            groundedness_score=groundedness,
            answer_relevance_score=relevance,
        )

    def compute_sync(
        self,
        trace: AnswerTrace,
        gold: GoldEvidence,
    ) -> L5Metrics:
        """
        Compute L5 metrics synchronously (without judge).

        Args:
            trace: Answer trace from evaluation
            gold: Gold standard evidence

        Returns:
            L5Metrics with label-based scores only
        """
        exact_match = self._compute_exact_match(trace, gold)
        token_f1 = self._compute_token_f1(trace, gold)

        # Semantic similarity (optional)
        semantic_sim = None
        if self.use_semantic_similarity and gold.answer:
            calc = self._get_semantic_calculator()
            semantic_sim = calc.compute(trace.final_answer, gold.answer)

        return L5Metrics(
            answer_exact_match=exact_match,
            answer_f1=token_f1,
            semantic_similarity=semantic_sim,
            groundedness_score=None,
            answer_relevance_score=None,
        )

    def _compute_exact_match(self, trace: AnswerTrace, gold: GoldEvidence) -> float:
        """
        Compute exact match score.

        Normalizes both answers and checks for exact equality.
        """
        if not gold.answer:
            return 1.0  # No gold answer = automatic pass

        predicted = self._normalize_answer(trace.final_answer)
        expected = self._normalize_answer(gold.answer)

        return 1.0 if predicted == expected else 0.0

    def _compute_token_f1(self, trace: AnswerTrace, gold: GoldEvidence) -> float:
        """
        Compute token-level F1 score.

        Tokenizes both answers and computes F1 over tokens.
        """
        if not gold.answer:
            return 1.0  # No gold answer = automatic pass

        predicted_tokens = set(self._tokenize(trace.final_answer))
        gold_tokens = set(self._tokenize(gold.answer))

        return self.set_f1(predicted_tokens, gold_tokens)

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """
        Normalize answer for comparison.

        - Lowercase
        - Remove punctuation
        - Collapse whitespace
        - Strip
        """
        if not answer:
            return ""

        # Lowercase
        normalized = answer.lower()

        # Remove punctuation
        normalized = re.sub(r"[^\w\s]", " ", normalized)

        # Collapse whitespace and strip
        normalized = " ".join(normalized.split())

        return normalized

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text into words.

        Simple whitespace tokenization after normalization.
        """
        if not text:
            return []

        # Normalize and split
        normalized = L5AnswerMetrics._normalize_answer(text)
        return normalized.split()


def answer_exact_match(trace: AnswerTrace, gold: GoldEvidence) -> float:
    """
    Convenience function for exact match.

    Args:
        trace: Answer trace
        gold: Gold evidence

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    calc = L5AnswerMetrics()
    return calc._compute_exact_match(trace, gold)


def answer_token_f1(trace: AnswerTrace, gold: GoldEvidence) -> float:
    """
    Convenience function for token F1.

    Args:
        trace: Answer trace
        gold: Gold evidence

    Returns:
        Token-level F1 score
    """
    calc = L5AnswerMetrics()
    return calc._compute_token_f1(trace, gold)


async def groundedness_score(
    answer: str,
    context: str,
    judge: JudgeInterface | None = None,
) -> float:
    """
    Compute groundedness score using judge.

    Args:
        answer: Generated answer
        context: Retrieved context
        judge: Judge implementation (uses StubJudge if not provided)

    Returns:
        Groundedness score (0.0-1.0)
    """
    judge = judge or StubJudge()
    return await judge.score_groundedness(answer, context)


async def answer_relevance_score(
    answer: str,
    question: str,
    judge: JudgeInterface | None = None,
) -> float:
    """
    Compute answer relevance score using judge.

    Args:
        answer: Generated answer
        question: Original question
        judge: Judge implementation (uses StubJudge if not provided)

    Returns:
        Relevance score (0.0-1.0)
    """
    judge = judge or StubJudge()
    return await judge.score_relevance(answer, question)


def compute_answer_quality(
    answer: str,
    gold_answer: str | None,
    context: str | None = None,
    use_semantic_similarity: bool = False,
) -> dict[str, float]:
    """
    Compute all answer quality metrics without judge.

    Args:
        answer: Generated answer
        gold_answer: Gold standard answer (optional)
        context: Retrieved context (optional, for token overlap)
        use_semantic_similarity: Whether to compute semantic similarity

    Returns:
        Dict of metric_name -> score
    """
    trace = AnswerTrace(
        final_answer=answer,
        citations=[],
        confidence=None,
    )
    gold = GoldEvidence(answer=gold_answer)

    calc = L5AnswerMetrics(use_semantic_similarity=use_semantic_similarity)

    metrics = {
        "exact_match": calc._compute_exact_match(trace, gold),
        "token_f1": calc._compute_token_f1(trace, gold),
    }

    if use_semantic_similarity and gold_answer:
        from eval.metrics.semantic import SemanticSimilarity

        sim_calc = SemanticSimilarity()
        metrics["semantic_similarity"] = sim_calc.compute(answer, gold_answer)

    if context:
        metrics["context_overlap"] = MetricCalculator.token_overlap(answer, context)

    return metrics
