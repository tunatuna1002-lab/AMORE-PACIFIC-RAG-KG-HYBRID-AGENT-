"""[2026-09 사후] OE10: `/api/v4/brain/status`의 `ontology` 필드."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes import brain as brain_routes
from src.infrastructure.feature_flags import FeatureFlags
from src.ontology.ontology import get_ontology

EXPECTED_KEYS = {
    "version",
    "as_of",
    "class_count",
    "brand_count",
    "use_class_reasoning",
    "kg_write_validation",
}


@pytest.fixture(autouse=True)
def _reset_flags(monkeypatch: pytest.MonkeyPatch):
    FeatureFlags.reset_instance()
    monkeypatch.delenv("FF_ONTOLOGY_USE_CLASS_REASONING", raising=False)
    monkeypatch.delenv("FF_KG_WRITE_VALIDATION", raising=False)
    yield
    FeatureFlags.reset_instance()


def _fake_brain() -> MagicMock:
    brain = MagicMock()
    brain.mode.value = "idle"
    brain.scheduler = None
    brain.get_stats.return_value = {}
    brain.get_component_status.return_value = {}
    brain.context_gatherer = None
    return brain


async def _status() -> dict:
    with patch.object(brain_routes, "get_initialized_brain", AsyncMock(return_value=_fake_brain())):
        return await brain_routes.get_brain_status.__wrapped__(request=None)


@pytest.mark.asyncio
async def test_status_exposes_ontology_from_source(monkeypatch: pytest.MonkeyPatch) -> None:
    # OA-10: 저장소 JSON 기본값이 ON으로 바뀌었으므로 OFF 기대치는 env로 명시한다.
    monkeypatch.setenv("FF_ONTOLOGY_USE_CLASS_REASONING", "false")
    payload = await _status()
    onto = get_ontology()

    assert set(payload["ontology"]) == EXPECTED_KEYS
    assert payload["ontology"]["version"] == onto.version
    assert payload["ontology"]["as_of"] == onto.as_of
    assert payload["ontology"]["class_count"] == onto.class_count > 0
    assert payload["ontology"]["brand_count"] == onto.brand_count > 0
    # KG 쓰기 검증 저장소 기본값: warn
    assert payload["ontology"]["use_class_reasoning"] is False
    assert payload["ontology"]["kg_write_validation"] == "warn"


@pytest.mark.asyncio
async def test_status_reflects_repository_default_on() -> None:
    """OA-10: env·직접 오버라이드가 없으면 저장소 JSON 기본값(ON)을 그대로 노출한다."""
    payload = await _status()

    assert payload["ontology"]["use_class_reasoning"] is True


@pytest.mark.asyncio
async def test_status_reflects_flag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FF_ONTOLOGY_USE_CLASS_REASONING", "true")
    monkeypatch.setenv("FF_KG_WRITE_VALIDATION", "enforce")

    payload = await _status()

    assert payload["ontology"]["use_class_reasoning"] is True
    assert payload["ontology"]["kg_write_validation"] == "enforce"


@pytest.mark.asyncio
async def test_status_reports_load_error_without_failing() -> None:
    with patch("src.ontology.ontology.get_ontology", side_effect=ValueError("bad schema")):
        payload = await _status()

    assert payload["initialized"] is True
    assert payload["ontology"]["class_count"] is None
    assert "bad schema" in payload["ontology"]["error"]


@pytest.mark.asyncio
async def test_uninitialized_brain_still_reports_ontology() -> None:
    with patch.object(
        brain_routes, "get_initialized_brain", AsyncMock(side_effect=RuntimeError("no brain"))
    ):
        payload = await brain_routes.get_brain_status.__wrapped__(request=None)

    assert payload["initialized"] is False
    assert set(payload["ontology"]) >= EXPECTED_KEYS
