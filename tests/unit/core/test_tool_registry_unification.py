"""
도구 경로 단일화 게이트 (트랙 4-A, 설계 E4)

이 프로젝트에는 도구 목록이 세 갈래였다: 대시보드 JSON 도구 5종(brain), ReAct 전용 3종
(react_tools), 레거시 AGENT_TOOLS(llm_orchestrator). 답변의 근거가 어느 도구에서 왔는지
추적할 수 없었고, 경로에 따라 같은 질문이 다른 근거로 답해졌다.

여기서는 배선이 실제로 하나인지 본다: DecisionMaker(=ToolCoordinator)·ReAct 실행기가
같은 레지스트리 객체를 보고, 대시보드 도구 이름이 서비스 코드에 남아 있지 않다.

가짜는 문서 색인 초기화(HybridRetriever.initialize)뿐이다.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.core.react_agent import ALLOWED_ACTIONS
from src.core.tool_registry import TOOL_NAMES, ToolRegistry, function_schemas
from src.infrastructure.feature_flags import FeatureFlags

REMOVED_DASHBOARD_TOOLS = (
    "get_brand_status",
    "get_product_info",
    "get_competitor_analysis",
    "get_category_info",
    "get_action_items",
)

SRC = Path(__file__).resolve().parents[3] / "src"


@pytest.fixture
def brain_env(monkeypatch, tmp_path):
    data_path = tmp_path / "dashboard_data.json"
    data_path.write_text(json.dumps({"brand": {"competitors": []}}), encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_DATA_PATH", str(data_path))
    monkeypatch.setenv("FF_AGENTS_USE_REACT_AGENT", "true")
    monkeypatch.setenv("FF_RETRIEVER_USE_OWL_STRATEGY", "false")
    FeatureFlags.reset_instance()
    yield
    FeatureFlags.reset_instance()


async def _brain():
    from src.core.brain import UnifiedBrain
    from src.rag.hybrid_retriever import HybridRetriever

    brain = UnifiedBrain()
    with patch.object(HybridRetriever, "initialize", AsyncMock()):
        await brain.initialize()
    return brain


@pytest.mark.asyncio
async def test_every_path_sees_the_same_five_tools(brain_env):
    brain = await _brain()

    registry = brain.tool_executor
    assert isinstance(registry, ToolRegistry)
    assert registry is brain.tool_coordinator.tool_executor
    assert registry is brain._react_agent.tool_executor.registry

    coordinator_tools = brain.tool_coordinator.get_available_tools()
    react_tools = brain._react_agent.tool_executor.get_available_tools()
    decision_schema_names = [
        s["function"]["name"]
        for s in function_schemas(brain._get_system_state()["available_tools"])
    ]

    assert coordinator_tools == react_tools == decision_schema_names == list(TOOL_NAMES)
    assert ALLOWED_ACTIONS - {"final_answer", "refine_search"} == set(TOOL_NAMES)


@pytest.mark.asyncio
async def test_registry_is_bound_to_the_retrieval_backend(brain_env):
    brain = await _brain()

    assert brain.tool_executor.retriever is brain._context_gatherer.retriever


@pytest.mark.asyncio
async def test_dashboard_tool_registration_is_gone(brain_env):
    brain = await _brain()

    assert not hasattr(brain, "_register_dashboard_tools")
    tools = brain._get_system_state()["available_tools"]
    assert not set(tools) & set(REMOVED_DASHBOARD_TOOLS)


def _code_strings_and_identifiers(source: str) -> set[str]:
    """문서화 문자열·주석을 뺀 코드에 나오는 문자열 리터럴과 식별자.

    "옛 도구 5종을 제거했다"는 설명은 남겨도 되지만, 코드가 그 이름을 다루면 안 된다.
    """
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
        elif isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def test_removed_dashboard_tool_names_are_not_referenced_in_src():
    """소스 코드에 옛 도구 이름이 남아 있으면 배선이 덜 옮겨진 것이다.

    PromptGuard의 누출 탐지처럼 이름을 하드코딩했던 곳도 여기서 잡힌다.
    """
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if " 2.py" in path.name:  # 편집기 사본 (다른 트랙 소유)
            continue
        names = _code_strings_and_identifiers(path.read_text(encoding="utf-8"))
        offenders += [
            f"{path.relative_to(SRC)}: {name}" for name in REMOVED_DASHBOARD_TOOLS if name in names
        ]
    assert offenders == []
