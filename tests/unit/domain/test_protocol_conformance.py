"""
D16: Protocol ↔ 구현 정렬 (Phase 5)
====================================
``src/domain/interfaces`` 의 Protocol 은 배치 파이프라인이 주입받는 계약이다. 이전에는
구현에 없는 메서드(``crawl`` / ``save`` / ``calculate`` / ``generate``)를 선언하고 있어서
``isinstance`` 가 항상 False 였고, 프로토콜을 믿고 호출하면 TypeError 가 났다.

이 테스트는 두 가지를 고정한다.
1. 실제 에이전트가 해당 Protocol 의 ``isinstance`` 를 통과한다 (메서드 존재).
2. 선언된 시그니처가 구현과 일치한다 (``inspect.signature`` 의 파라미터 이름·순서).

``runtime_checkable`` Protocol 의 ``isinstance`` 는 메서드 이름만 보므로 2번을 따로 둔다.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol

import pytest

from src.agents.alert_agent import AlertAgent
from src.agents.crawler_agent import CrawlerAgent
from src.agents.hybrid_insight_agent import HybridInsightAgent
from src.agents.metrics_agent import MetricsAgent
from src.agents.storage_agent import StorageAgent
from src.domain.interfaces.agent import (
    CrawlerAgentProtocol,
    InsightAgentProtocol,
    MetricsAgentProtocol,
    StorageAgentProtocol,
)
from src.domain.interfaces.alert import AlertAgentProtocol
from src.domain.interfaces.insight import InsightAgentProtocol as InsightProtocolFromInsightModule

PAIRS: list[tuple[type, type[Protocol]]] = [
    (CrawlerAgent, CrawlerAgentProtocol),
    (StorageAgent, StorageAgentProtocol),
    (MetricsAgent, MetricsAgentProtocol),
    (HybridInsightAgent, InsightAgentProtocol),
    (AlertAgent, AlertAgentProtocol),
]


def _protocol_methods(protocol: type) -> list[str]:
    """Protocol 이 요구하는 공개 멤버 이름 (3.11: typing._get_protocol_attrs, 3.12+: __protocol_attrs__)."""
    attrs = getattr(protocol, "__protocol_attrs__", None)
    if attrs is None:
        from typing import _get_protocol_attrs  # noqa: PLC0415 - 버전별 비공개 API

        attrs = _get_protocol_attrs(protocol)
    names = sorted(name for name in attrs if not name.startswith("_"))
    assert names, f"{protocol.__name__} declares no members"
    return names


def test_insight_protocol_has_a_single_definition() -> None:
    """``interfaces.agent`` 는 ``interfaces.insight`` 의 정의를 재수출만 한다."""
    assert InsightAgentProtocol is InsightProtocolFromInsightModule


@pytest.mark.parametrize(("impl", "protocol"), PAIRS, ids=lambda x: getattr(x, "__name__", str(x)))
def test_implementation_declares_every_protocol_method(impl: type, protocol: type) -> None:
    missing = [name for name in _protocol_methods(protocol) if not hasattr(impl, name)]
    assert missing == [], f"{impl.__name__} is missing {missing} required by {protocol.__name__}"


@pytest.mark.parametrize(("impl", "protocol"), PAIRS, ids=lambda x: getattr(x, "__name__", str(x)))
def test_protocol_and_implementation_signatures_agree(impl: type, protocol: type) -> None:
    """파라미터 이름과 순서가 같아야 한다 (키워드 호출이 깨지지 않도록)."""
    mismatches: list[str] = []
    for name in _protocol_methods(protocol):
        proto_params = list(inspect.signature(getattr(protocol, name)).parameters)
        impl_params = list(inspect.signature(getattr(impl, name)).parameters)
        if proto_params != impl_params:
            mismatches.append(f"{name}: protocol{proto_params} != impl{impl_params}")
    assert mismatches == [], f"{impl.__name__} vs {protocol.__name__}:\n" + "\n".join(mismatches)


@pytest.mark.parametrize(("impl", "protocol"), PAIRS, ids=lambda x: getattr(x, "__name__", str(x)))
def test_isinstance_passes_for_a_duck_typed_stand_in(impl: type, protocol: type) -> None:
    """
    Protocol 은 runtime_checkable 이어야 하고, 프로토콜 메서드만 갖춘 대역이
    ``isinstance`` 를 통과해야 한다 (배치 파이프라인이 fake 를 주입할 수 있는 근거).
    """
    namespace: dict[str, Any] = {}
    for name in _protocol_methods(protocol):
        namespace[name] = getattr(impl, name)
    stand_in = type(f"StandIn{protocol.__name__}", (), namespace)()
    assert isinstance(stand_in, protocol)
