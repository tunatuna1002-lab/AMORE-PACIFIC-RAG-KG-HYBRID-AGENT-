"""LLM-as-a-judge interfaces for evaluation."""

from eval.judge.interface import JudgeInterface
from eval.judge.llm import LLMJudge
from eval.judge.nli import NLIJudge
from eval.judge.stub import StubJudge

__all__ = ["JudgeInterface", "LLMJudge", "NLIJudge", "StubJudge"]
