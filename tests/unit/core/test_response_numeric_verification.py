"""ResponsePipeline이 생성 직후 답변 수치를 검증하고 메타데이터에 남긴다 (설계 E8 뒷부분).

파이프라인·검증기·카드·렌더러는 실제 객체이고, 가짜는 LLM 응답 문자열
(``litellm.acompletion`` 대역)뿐이다. 모드는 ENV 플래그로 바꾼다(ENV > JSON > 기본).
"""

from types import SimpleNamespace

import pytest

from src.core.models import Context, Decision, ToolResult
from src.core.response_pipeline import ResponsePipeline
from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit
from src.infrastructure.feature_flags import FeatureFlags
from src.rag.evidence_renderer import render_for_prompt

MODE_ENV = "FF_RESPONSE_NUMERIC_VERIFICATION_MODE"

HHI = Evidence.create(
    kind=EvidenceKind.METRIC,
    subject="lip_care",
    predicate="hhi",
    value=0.0681,
    unit=EvidenceUnit.INDEX_0_1,
    as_of="2026-08-31",
    source="sqlite:market_metrics",
    confidence=1.0,
    text="lip_care HHI 0.0681",
)
SOS = Evidence.create(
    kind=EvidenceKind.METRIC,
    subject="laneige",
    predicate="sos",
    object="lip_care",
    value=0.02,
    unit=EvidenceUnit.RATIO,
    as_of="2026-08-31",
    source="sqlite:brand_metrics",
    confidence=1.0,
    text="LANEIGE lip_care SoS 2%",
)
CARDS = [HHI, SOS]


class _FakeLLM:
    """litellm.acompletion 대역: 고정 답변 문자열만 돌려준다."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    async def __call__(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    FeatureFlags.reset_instance()
    monkeypatch.delenv(MODE_ENV, raising=False)
    yield
    FeatureFlags.reset_instance()


def _context(cards=CARDS) -> Context:
    return Context(
        query="LANEIGE Lip Care 현황",
        summary=render_for_prompt(cards),
        evidence=list(cards),
        prompt_evidence=list(cards),
    )


def _install(monkeypatch, answer: str) -> _FakeLLM:
    import litellm

    fake = _FakeLLM(answer)
    monkeypatch.setattr(litellm, "acompletion", fake)
    return fake


WRONG_ANSWER = f"HHI는 0.12[{HHI.id}]이고 SoS는 2%입니다 [{SOS.id}]."


async def test_default_mode_annotates_without_changing_text(monkeypatch):
    fake = _install(monkeypatch, WRONG_ANSWER)

    response = await ResponsePipeline(openai_client=object()).generate("q", _context())

    assert fake.calls == 1
    assert response.text == WRONG_ANSWER
    meta = response.metadata["numeric_verification"]
    assert meta["mode"] == "annotate"
    assert meta["skipped"] is None
    assert (meta["checked"], meta["verified"], meta["mismatch"], meta["replaced"]) == (2, 1, 1, 0)
    assert meta["details"][0] == {
        "status": "mismatch",
        "number": "0.12",
        "value": 0.12,
        "kind": "number",
        "sentence": WRONG_ANSWER,
        "cited_ids": [HHI.id],
        "unknown_ids": [],
        "matched_ids": [],
        "found_in_ids": [],
    }


async def test_enforce_mode_replaces_mismatch(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "enforce")
    _install(monkeypatch, WRONG_ANSWER)

    response = await ResponsePipeline(openai_client=object()).generate("q", _context())

    assert response.text == f"HHI는 확인되지 않음[{HHI.id}]이고 SoS는 2%입니다 [{SOS.id}]."
    meta = response.metadata["numeric_verification"]
    assert (meta["mode"], meta["mismatch"], meta["verified"], meta["replaced"]) == (
        "enforce",
        1,
        1,
        1,
    )


async def test_off_mode_leaves_no_metadata(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "off")
    _install(monkeypatch, WRONG_ANSWER)

    response = await ResponsePipeline(openai_client=object()).generate("q", _context())

    assert response.text == WRONG_ANSWER
    assert "numeric_verification" not in response.metadata


async def test_fast_path_is_verified(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "enforce")
    fake = _install(monkeypatch, WRONG_ANSWER)
    decision = Decision(tool="direct_answer", confidence=0.9, reason="HIGH confidence: 카드 충분")

    response = await ResponsePipeline(openai_client=object()).generate(
        "q", _context(), decision=decision
    )

    assert fake.calls == 1
    assert "확인되지 않음" in response.text
    assert response.metadata["numeric_verification"]["mismatch"] == 1


async def test_tool_result_path_is_verified(monkeypatch):
    _install(monkeypatch, WRONG_ANSWER)

    response = await ResponsePipeline(openai_client=object()).generate_with_tool_result(
        "q", _context(), ToolResult(tool_name="query_data", success=True, data={"x": 1})
    )

    assert response.tools_called == ["query_data"]
    assert response.metadata["numeric_verification"]["mismatch"] == 1


async def test_no_prompt_cards_is_skipped(monkeypatch):
    _install(monkeypatch, "SoS는 5%입니다.")

    response = await ResponsePipeline(openai_client=object()).generate(
        "q", Context(query="q", summary="카드 없음")
    )

    assert response.metadata["numeric_verification"]["skipped"] == "no_evidence"


async def test_fallback_answer_without_llm_is_verified_too(monkeypatch):
    """LLM이 없을 때의 기본 응답(문서 발췌)도 같은 검증을 지난다. 인용이 없으니 치환은 없다."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(MODE_ENV, "enforce")
    context = _context()
    context.rag_docs = [{"content": "HHI가 0.15 미만이면 분산된 시장으로 본다."}]

    response = await ResponsePipeline(openai_client=None).generate("q", context)

    assert response.text == "HHI가 0.15 미만이면 분산된 시장으로 본다."
    meta = response.metadata["numeric_verification"]
    assert (meta["skipped"], meta["checked"], meta["no_citation"], meta["replaced"]) == (
        None,
        1,
        1,
        0,
    )


async def test_verifier_error_keeps_answer_and_is_recorded(monkeypatch, caplog):
    import src.core.response_pipeline as pipeline_module

    def _boom(*args, **kwargs):
        raise RuntimeError("verifier bug")

    monkeypatch.setenv(MODE_ENV, "enforce")
    monkeypatch.setattr(pipeline_module, "apply_numeric_verification", _boom)
    _install(monkeypatch, WRONG_ANSWER)

    response = await ResponsePipeline(openai_client=object()).generate("q", _context())

    assert response.text == WRONG_ANSWER
    assert response.metadata["numeric_verification"]["skipped"] == "error"
    assert "Numeric verification failed" in caplog.text


async def test_invalid_mode_falls_back_to_annotate(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "strict")
    _install(monkeypatch, WRONG_ANSWER)

    response = await ResponsePipeline(openai_client=object()).generate("q", _context())

    assert response.text == WRONG_ANSWER
    assert response.metadata["numeric_verification"]["mode"] == "annotate"


async def test_brain_stream_emits_enforced_text(monkeypatch, tmp_path):
    """대시보드 스트림(``/api/v4/chat/stream``)은 ``generate``가 끝난 답을 내보낸다.

    그래서 enforce 치환이 스트림 텍스트에도 반영된다. 가짜는 LLM 두 곳(판단·답변)과
    I/O(문서 색인 초기화·컨텍스트 수집)뿐이고, Brain·QueryGraph 배선·파이프라인은 실제다.
    """
    from unittest.mock import AsyncMock, patch

    import src.core.decision_maker as decision_module
    from src.core.brain import UnifiedBrain
    from src.rag.hybrid_retriever import HybridRetriever

    data_path = tmp_path / "dashboard_data.json"
    data_path.write_text('{"brand": {"competitors": []}}', encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_DATA_PATH", str(data_path))
    monkeypatch.setenv("FF_AGENTS_USE_REACT_AGENT", "false")
    monkeypatch.setenv(MODE_ENV, "enforce")
    FeatureFlags.reset_instance()

    decision_json = '{"tool": "direct_answer", "confidence": 0.9, "reason": "카드로 답변"}'
    monkeypatch.setattr(decision_module, "acompletion", _FakeLLM(decision_json))
    answer_llm = _install(monkeypatch, WRONG_ANSWER)

    brain = UnifiedBrain()
    with patch.object(HybridRetriever, "initialize", AsyncMock()):
        await brain.initialize()

    with patch.object(brain._context_gatherer, "gather", AsyncMock(return_value=_context())):
        chunks = [c async for c in brain.process_query_stream("LANEIGE Lip Care 현황")]

    text = "".join(c["content"] for c in chunks if c["type"] == "text")
    # 답변 1회 + 환각 점검 1회. function calling에는 모델 자기보고 신뢰도가 없어
    # Decision.confidence가 0.0(미보고)이 됐고, 그래서 환각 점검(< 0.8 조건)이 돈다 (트랙 4-A).
    assert answer_llm.calls == 2
    assert text == f"HHI는 확인되지 않음[{HHI.id}]이고 SoS는 2%입니다 [{SOS.id}]."
