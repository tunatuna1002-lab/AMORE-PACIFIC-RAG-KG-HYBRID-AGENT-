"""
v4 Brain 평가 어댑터
====================
평가 하니스는 v1 `HybridChatbotAgent`(`/api/chat`)를 측정해 왔지만 대시보드는 v4 Brain
경로(`/api/v4/chat/stream`)를 쓴다(docs/portfolio/amore_architecture_evidence.md §7).
이 어댑터는 `UnifiedBrain.process_query`(비스트림, QueryGraph)를 호출하고, 러너가 v1에서
읽던 것과 같은 스키마(`hybrid_context`, `response`, `llm_usage`)로 결과를 돌려준다.

한계 (결정 D3)
--------------
- 대시보드의 `process_query_stream`은 같은 분기 규칙을 QueryGraph가 아니라 메서드 안에
  따로 구현한다. 이 어댑터가 재는 것은 QueryGraph 쪽이다.
- 캐시는 끈다(`skip_cache=True`). 같은 질문의 반복 측정이 캐시 응답이 되지 않게.

요청별 트레이스 (동시 실행 오염 방지, 5b64536과 같은 원칙)
-------------------------------------------------------------
검색기·ReAct는 문항 간에 공유된다. 공유 속성에 마지막 결과를 저장하면 동시 실행 시 다른
문항의 트레이스로 덮인다. 그래서 문항마다 ContextVar에 보관함을 두고, 검색기·ReAct 메서드를
감싸 그 보관함에 적는다. asyncio 태스크는 생성 시점의 컨텍스트를 복사하므로 보관함 객체는
그 문항의 호출 사슬 안에서만 보인다.

LLM 사용량
----------
litellm 성공 콜백으로 문항 보관함에 토큰을 적립한다(답변·결정·환각 점검·ReAct 전부).
검색기의 질의 확장(`DocumentRetriever.expand_query`)과 임베딩은 litellm이 아니라 openai
클라이언트를 직접 써서 집계되지 않는다 — v1 측정도 같다.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from src.domain.entities.evidence import Evidence
from src.rag.evidence_adapters import EvidenceAdapter

logger = logging.getLogger(__name__)

_CURRENT_REQUEST: ContextVar[dict[str, Any] | None] = ContextVar(
    "brain_eval_current_request", default=None
)


@dataclass
class V4RetrievalTrace:
    """러너의 `hybrid_context` 자리에 들어가는 v4 검색 트레이스 (HybridContext와 같은 속성)."""

    query: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    ontology_facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[Any] = field(default_factory=list)
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    metric_facts: list[dict[str, Any]] = field(default_factory=list)
    retriever_type: str = "none"
    # 검색 오류 가시화(F3) — HybridRetriever.retrieve()가 HybridContext.metadata에
    # 남기는 retrieval_error/degraded와 같은 계약. 러너의 EvalTrace가 여기서 읽는다.
    metadata: dict[str, Any] = field(default_factory=dict)
    # 증거 카드 (설계 E1, 트랙 2-C). 검색이 만든 전체 카드와, 답변 프롬프트에 실제로
    # 실린 카드(선별 후)를 구분한다 — judge 컨텍스트는 후자만 봐야 한다.
    evidence: list[Evidence] = field(default_factory=list)
    prompt_evidence: list[Evidence] = field(default_factory=list)


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    holder = _CURRENT_REQUEST.get()
    if holder is not None:
        usage = holder["usage"]
        usage["prompt_tokens"] += int(prompt_tokens or 0)
        usage["completion_tokens"] += int(completion_tokens or 0)
        usage["calls"] += 1


_usage_logger_installed = False


def _install_usage_logger() -> None:
    """litellm 성공 콜백을 한 번만 등록한다."""
    global _usage_logger_installed
    if _usage_logger_installed:
        return

    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _UsageLogger(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            usage = getattr(response_obj, "usage", None)
            if usage is not None:
                _record_usage(
                    getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)
                )

    litellm.callbacks = [*(litellm.callbacks or []), _UsageLogger()]
    _usage_logger_installed = True


class BrainEvalAdapter:
    """`EvalRunner`가 호출하는 v4 Brain 래퍼 (`chat(question) -> dict`)."""

    target = "v4"

    def __init__(self, brain: Any | None = None):
        from src.core.brain import UnifiedBrain

        self.brain = brain or UnifiedBrain()
        self.model = getattr(self.brain, "model", None)
        self._instrumented = False

    async def initialize(self) -> None:
        if not self.brain._initialized:
            await self.brain.initialize()
        self._instrument()
        _install_usage_logger()

    # ── 계측 ─────────────────────────────────────────────────────────

    def _instrument(self) -> None:
        if self._instrumented:
            return
        gatherer = self.brain._context_gatherer
        retriever = getattr(gatherer, "retriever", None)
        if retriever is None:
            raise RuntimeError("Brain에 검색기가 없어 v4 트레이스를 수집할 수 없다")

        original_retrieve = retriever.retrieve
        original_unified = retriever.retrieve_unified

        async def retrieve(*args: Any, **kwargs: Any) -> Any:
            ctx = await original_retrieve(*args, **kwargs)
            holder = _CURRENT_REQUEST.get()
            if holder is not None:
                holder["hybrid_context"] = ctx
            return ctx

        async def retrieve_unified(*args: Any, **kwargs: Any) -> Any:
            result = await original_unified(*args, **kwargs)
            holder = _CURRENT_REQUEST.get()
            if holder is not None:
                holder["unified"] = result
            return result

        # 인스턴스 속성으로 감싼다: retrieve_unified 내부의 self.retrieve 호출도 감싼 쪽을 탄다
        retriever.retrieve = retrieve
        retriever.retrieve_unified = retrieve_unified

        # 도구 실행(DecisionMaker 경로): 결과 카드가 답변 프롬프트에 실리므로 트레이스에도 싣는다
        coordinator = self.brain.tool_coordinator
        original_execute = coordinator.execute

        async def execute(*args: Any, **kwargs: Any) -> Any:
            result = await original_execute(*args, **kwargs)
            holder = _CURRENT_REQUEST.get()
            if holder is not None:
                holder.setdefault("tool_results", []).append(result)
            return result

        coordinator.execute = execute

        react = getattr(self.brain, "_react_agent", None)
        if react is not None:
            original_run = react.run

            async def run(*args: Any, **kwargs: Any) -> Any:
                result = await original_run(*args, **kwargs)
                holder = _CURRENT_REQUEST.get()
                if holder is not None:
                    holder["react_steps"] = [
                        {
                            "action": s.action,
                            "action_input": s.action_input,
                            "observation": s.observation,
                        }
                        for s in result.steps
                    ]
                return result

            react.run = run

        self._instrumented = True

    # ── 호출 ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_trace(
        query: str, holder: dict[str, Any], *, react_answered: bool = True
    ) -> V4RetrievalTrace:
        ctx = holder.get("hybrid_context")
        unified = holder.get("unified")
        if ctx is not None:
            ctx_metadata = getattr(ctx, "metadata", None)
            trace = V4RetrievalTrace(
                query=query,
                entities=ctx.entities or {},
                ontology_facts=list(ctx.ontology_facts or []),
                inferences=list(ctx.inferences or []),
                rag_chunks=list(ctx.rag_chunks or []),
                metric_facts=list(getattr(ctx, "metric_facts", None) or []),
                retriever_type="legacy",
                metadata=dict(ctx_metadata) if isinstance(ctx_metadata, dict) else {},
                evidence=list(getattr(ctx, "evidence", None) or []),
                prompt_evidence=list(getattr(ctx, "prompt_evidence", None) or []),
            )
        elif unified is not None:
            # OWL 전략 또는 Self-RAG 생략: HybridContext가 없다
            unified_metadata = getattr(unified, "metadata", None)
            trace = V4RetrievalTrace(
                query=query,
                entities=unified.entities or {},
                ontology_facts=list(unified.ontology_facts or []),
                inferences=list(unified.inferences or []),
                rag_chunks=list(unified.rag_chunks or []),
                retriever_type=unified.retriever_type,
                metadata=dict(unified_metadata) if isinstance(unified_metadata, dict) else {},
                evidence=list(getattr(unified, "evidence", None) or []),
                prompt_evidence=list(getattr(unified, "prompt_evidence", None) or []),
            )
        else:
            trace = V4RetrievalTrace(query=query)

        # DecisionMaker 경로의 도구 결과 카드 (레지스트리 도구는 결과가 전부 카드다, 트랙 4-A).
        # ResponsePipeline이 이 카드를 답변 프롬프트에 싣는다 → judge 근거성 컨텍스트도 같게.
        from src.core.tool_registry import tool_evidence

        seen_ids = {card.id for card in trace.prompt_evidence}
        for tool_result in holder.get("tool_results") or []:
            for card in tool_evidence(tool_result):
                if card.id not in seen_ids:
                    seen_ids.add(card.id)
                    trace.prompt_evidence.append(card)

        # ReAct 도구 관찰도 답변의 근거다 — 단, ReAct가 실제로 답을 낸 경우에만.
        # 섀도 모드(agents.react_shadow_mode)는 답변은 파이프라인이 내고 ReAct는 기록만
        # 남기므로, 여기서 관찰 카드를 얹으면 답변이 보지 않은 근거가 judge 컨텍스트에
        # 섞인다 — react_answered(=route_trace.route == "react")일 때만 싣는다.
        if react_answered and holder.get("react_steps"):
            adapter = EvidenceAdapter()
            for step in holder["react_steps"]:
                if not step.get("observation") or not step.get("action"):
                    continue
                if step["action"] == "final_answer":
                    continue
                card = adapter.from_tool_observation(
                    tool_name=step["action"],
                    tool_input=step.get("action_input"),
                    observation=step["observation"][:1500],
                )
                trace.prompt_evidence.append(card)
        return trace

    async def chat(self, question: str) -> dict[str, Any]:
        if not self._instrumented:
            await self.initialize()

        holder: dict[str, Any] = {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        }
        token = _CURRENT_REQUEST.set(holder)
        try:
            response = await self.brain.process_query(question, skip_cache=True)
            # litellm은 성공 콜백을 별도 태스크로 띄운다. 보관함을 읽기 전에 실행될 틈을 준다.
            await asyncio.sleep(0.05)
        finally:
            _CURRENT_REQUEST.reset(token)

        # QueryGraph._finalize_route_trace(31040bf)가 남기는 문항별 경로 관측.
        # v1(HybridChatbotAgent) 경로는 이 키를 채우지 않는다 — v4 전용.
        route_trace = (response.metadata or {}).get("route_trace")
        # ReAct가 실제로 답을 냈는지 — 섀도 모드는 ReAct가 돌아도 route가 "react"가 아니다
        # (트랙 5-C). react_steps 존재만으로 판단하면 섀도 기록이 답변으로 둔갑한다.
        react_answered = bool(route_trace) and route_trace.get("route") == "react"
        return {
            "response": response.text,
            "confidence": response.confidence_score,
            "query_type": "react" if react_answered else response.query_type,
            "sources": [],
            "citations": [],
            "tools_called": list(response.tools_called or []),
            "is_fallback": response.is_fallback,
            "hybrid_context": self._build_trace(question, holder, react_answered=react_answered),
            "llm_usage": dict(holder["usage"]),
            "route_trace": route_trace,
            # ResponsePipeline(트랙 2-D)이 남기는 답변 수치 검증 결과. 플래그 off면 없음.
            "numeric_verification": (response.metadata or {}).get("numeric_verification"),
        }


async def create_eval_agent(target: str) -> Any:
    """평가 대상 에이전트 생성. v1 = HybridChatbotAgent, v4 = UnifiedBrain 어댑터."""
    if target == "v1":
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        return HybridChatbotAgent()
    if target == "v4":
        adapter = BrainEvalAdapter()
        await adapter.initialize()
        return adapter
    raise ValueError(f"Unknown eval target: {target} (v1|v4)")
