"""LLM-as-a-judge interfaces for evaluation."""

import logging

from eval.judge.interface import JudgeInterface
from eval.judge.llm import LLMJudge
from eval.judge.nli import NLIJudge
from eval.judge.stub import StubJudge

logger = logging.getLogger(__name__)

__all__ = ["JudgeInterface", "LLMJudge", "NLIJudge", "StubJudge", "create_judge"]


def create_judge(judge_type: str = "stub", **kwargs) -> StubJudge:
    """
    Factory function to create a judge instance.

    Args:
        judge_type: One of "stub", "llm", or "nli".
        **kwargs: Passed to the judge constructor.

    Returns:
        A judge instance (StubJudge as fallback if optional deps missing).

    Raises:
        ValueError: If judge_type is not recognized.
    """
    if judge_type == "stub":
        return StubJudge(**kwargs)

    if judge_type == "llm":
        try:
            from eval.judge.llm import LLMJudge as _LLMJudge

            return _LLMJudge(**kwargs)
        except (ImportError, RuntimeError) as e:
            logger.warning(f"LLMJudge unavailable ({e}), falling back to StubJudge")
            return StubJudge()

    if judge_type == "nli":
        try:
            from eval.judge.nli import NLIJudge as _NLIJudge

            return _NLIJudge(**kwargs)
        except (ImportError, RuntimeError) as e:
            logger.warning(f"NLIJudge unavailable ({e}), falling back to StubJudge")
            return StubJudge()

    raise ValueError(f"Unknown judge_type: {judge_type!r}. Must be 'stub', 'llm', or 'nli'.")
