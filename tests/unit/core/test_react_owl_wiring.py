"""
ReAct 서비스 경로 배선 검증 (R1, 2026-09 사후 수리)

배경: brain.py가 존재하지 않는 `..agents.react_agent`를 import해 컴포넌트가 초기화 시
예외로 None이 되었다. 예외는 debug/info 로그로 삼켜졌고, 기존
테스트(test_react_integration.py)는 hasattr만 확인해 이를 잡지 못했다.
(OWL 검색 전략은 2026-09 트랙 4-C에서 삭제됐다.)

원칙: LLM 호출(litellm.acompletion)과 문서 색인 I/O(HybridRetriever.initialize)만 가짜로
둔다. import 경로·생성자·도구 등록·KG·QueryGraph 라우팅은 실제 객체로 검증한다.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.models import Context
from src.infrastructure.feature_flags import FeatureFlags


def _llm_reply(payload: dict | str) -> SimpleNamespace:
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def flags_env(monkeypatch, tmp_path):
    """피처 플래그를 환경변수로 제어하고 싱글톤을 격리한다."""
    data_path = tmp_path / "dashboard_data.json"
    data_path.write_text(json.dumps({"brand": {"competitors": []}}), encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_DATA_PATH", str(data_path))

    def _set(react: bool) -> None:
        monkeypatch.setenv("FF_AGENTS_USE_REACT_AGENT", "true" if react else "false")
        FeatureFlags.reset_instance()

    yield _set
    FeatureFlags.reset_instance()


async def _initialized_brain():
    from src.core.brain import UnifiedBrain
    from src.rag.hybrid_retriever import HybridRetriever

    brain = UnifiedBrain()
    # 문서 색인(ChromaDB + 임베딩 API)만 차단한다. 배선은 실제로 수행된다.
    with patch.object(HybridRetriever, "initialize", AsyncMock()):
        await brain.initialize()
    return brain


class TestBrainInitializationWiring:
    @pytest.mark.asyncio
    async def test_react_agent_is_created_when_flag_on(self, flags_env):
        from src.core.react_agent import ReActAgent

        flags_env(react=True)
        brain = await _initialized_brain()

        assert isinstance(brain._react_agent, ReActAgent)
        assert brain._react_agent.tool_executor is not None
        status = brain.get_component_status()
        assert status["react_agent"] == {
            "enabled": True,
            "active": True,
            "error": None,
            "mode": "on",
        }

    @pytest.mark.asyncio
    async def test_react_tools_cover_allowed_actions(self, flags_env):
        from src.core.react_agent import ALLOWED_ACTIONS

        flags_env(react=True)
        brain = await _initialized_brain()

        registered = set(brain._react_agent.tool_executor.get_available_tools())
        # final_answer는 루프 종료, refine_search는 search_docs로 실행된다
        assert ALLOWED_ACTIONS - {"final_answer", "refine_search"} <= registered

    @pytest.mark.asyncio
    async def test_react_and_decision_maker_share_one_registry(self, flags_env):
        """ReAct와 DecisionMaker는 같은 도구 레지스트리를 본다 (트랙 4-A).

        이전에는 ReAct 전용 도구 3종과 대시보드 도구 5종이 따로 있어 경로마다 도구가 달랐다.
        """
        from src.core.tool_registry import TOOL_NAMES

        flags_env(react=True)
        brain = await _initialized_brain()

        decision_tools = brain.tool_coordinator.get_available_tools()
        react_tools = brain._react_agent.tool_executor.get_available_tools()
        assert decision_tools == react_tools == list(TOOL_NAMES)
        assert brain._react_agent.tool_executor.registry is brain.tool_coordinator.tool_executor

    @pytest.mark.asyncio
    async def test_flags_off_leaves_components_inactive_without_error(self, flags_env):
        flags_env(react=False)
        brain = await _initialized_brain()

        assert brain._react_agent is None
        status = brain.get_component_status()
        assert status["react_agent"] == {
            "enabled": False,
            "active": False,
            "error": None,
            "mode": "off",
        }

    def test_default_config_keeps_react_off(self, monkeypatch):
        """수리 직후 기본값은 비활성 (결정 D1: 평가 전 검증 없이 켜지지 않게)."""
        monkeypatch.delenv("FF_AGENTS_USE_REACT_AGENT", raising=False)
        FeatureFlags.reset_instance()
        try:
            assert FeatureFlags.get_instance().use_react_agent() is False
        finally:
            FeatureFlags.reset_instance()


class TestReActServicePath:
    """복잡 질의가 실제로 ReAct 경로를 타고, 도구 관찰을 거쳐 비어 있지 않은 답을 낸다."""

    QUERY = "LANEIGE 경쟁사와 비교해서 점유율이 왜 달라졌는지 분석해줘"

    def _scripted_llm(self):
        return AsyncMock(
            side_effect=[
                _llm_reply(
                    {
                        "thought": "경쟁 관계부터 확인",
                        "action": "kg_neighbors",
                        "action_input": {"entity": "LANEIGE", "relation": "competitors"},
                    }
                ),
                _llm_reply(
                    {
                        "thought": "충분함",
                        "action": "final_answer",
                        "action_input": {"answer": "LANEIGE의 경쟁 구도 분석 결과입니다."},
                    }
                ),
                _llm_reply({"quality_score": 0.8, "needs_improvement": False}),
            ]
        )

    def _thin_context(self) -> Context:
        return Context(query=self.QUERY, entities={"brands": ["laneige"]})

    @staticmethod
    def _forbid_non_react_llm(brain):
        """ReAct 경로를 벗어나면 실제 LLM을 부르기 전에 실패시킨다."""
        leaked = AssertionError("ReAct 경로를 타지 않았다")
        return (
            patch.object(brain.decision_maker, "decide", AsyncMock(side_effect=leaked)),
            patch.object(brain._response_pipeline, "generate", AsyncMock(side_effect=leaked)),
        )

    @pytest.mark.asyncio
    async def test_process_query_routes_complex_query_to_react(self, flags_env):
        flags_env(react=True)
        brain = await _initialized_brain()
        llm = self._scripted_llm()

        no_decide, no_generate = self._forbid_non_react_llm(brain)
        with (
            patch.object(
                brain._context_gatherer, "gather", AsyncMock(return_value=self._thin_context())
            ),
            patch("src.core.react_agent.acompletion", llm),
            no_decide,
            no_generate,
        ):
            response = await brain.process_query(self.QUERY, skip_cache=True)

        assert response.text == "LANEIGE의 경쟁 구도 분석 결과입니다."
        assert response.tools_called == ["kg_neighbors", "final_answer"]
        assert llm.await_count == 3

    @pytest.mark.asyncio
    async def test_stream_routes_complex_query_to_react(self, flags_env):
        flags_env(react=True)
        brain = await _initialized_brain()

        no_decide, no_generate = self._forbid_non_react_llm(brain)
        with (
            patch.object(
                brain._context_gatherer, "gather", AsyncMock(return_value=self._thin_context())
            ),
            patch("src.core.react_agent.acompletion", self._scripted_llm()),
            no_decide,
            no_generate,
        ):
            chunks = [c async for c in brain.process_query_stream(self.QUERY)]

        text = "".join(c["content"] for c in chunks if c["type"] == "text")
        done = next(c for c in chunks if c["type"] == "done")
        assert text == "LANEIGE의 경쟁 구도 분석 결과입니다."
        assert done["content"]["mode"] == "react"


class TestBrainStatusRoute:
    @pytest.mark.asyncio
    async def test_status_route_exposes_component_status(self, flags_env):
        from src.api.routes import brain as brain_routes

        flags_env(react=True)
        brain = await _initialized_brain()

        with patch.object(brain_routes, "get_initialized_brain", AsyncMock(return_value=brain)):
            payload = await brain_routes.get_brain_status.__wrapped__(request=None)

        assert payload["components"]["react_agent"]["active"] is True
        assert "owl_strategy" not in payload["components"]


class TestSharedDocRetriever:
    @pytest.mark.asyncio
    async def test_document_retriever_initialize_is_idempotent(self):
        from src.rag.retriever import DocumentRetriever

        retriever = DocumentRetriever()
        with (
            patch.object(retriever, "_check_vector_search", return_value=True),
            patch.object(retriever, "_initialize_vector_search", AsyncMock()) as vector_init,
        ):
            retriever.collection = object()
            await retriever.initialize()
            chunk_count = len(retriever.chunks)
            await retriever.initialize()

        assert chunk_count > 0
        assert len(retriever.chunks) == chunk_count
        assert vector_init.await_count == 1
