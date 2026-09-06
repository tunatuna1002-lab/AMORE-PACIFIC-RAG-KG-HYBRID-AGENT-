"""Regression tests for defect D24.

``HybridChatbotAgent._normalize_response_brands`` mapped the bare word "Beauty" to
"Beauty of Joseon", corrupting the category name "Beauty & Personal Care" (and any other
full brand name containing the truncated token, e.g. "Rare Beauty").
"""

from __future__ import annotations

import pytest

from src.agents.hybrid_chatbot_agent import HybridChatbotAgent


@pytest.fixture
def agent() -> HybridChatbotAgent:
    return HybridChatbotAgent()


def test_beauty_personal_care_category_is_preserved(agent: HybridChatbotAgent) -> None:
    text = "Beauty & Personal Care 카테고리에서 Beauty of Joseon이 상승"
    assert agent._normalize_response_brands(text) == text


def test_beauty_personal_care_preserved_without_full_brand_present(
    agent: HybridChatbotAgent,
) -> None:
    text = "Beauty & Personal Care > Skin Care > Lip Care 순위"
    assert agent._normalize_response_brands(text) == text
    assert "Beauty of Joseon" not in agent._normalize_response_brands(text)


def test_bare_beauty_still_normalized(agent: HybridChatbotAgent) -> None:
    text = "Beauty 브랜드가 3위입니다."
    assert agent._normalize_response_brands(text) == "Beauty of Joseon 브랜드가 3위입니다."


def test_other_full_brand_names_are_not_corrupted(agent: HybridChatbotAgent) -> None:
    text = "Rare Beauty와 Fenty Beauty, First Aid Beauty가 Top 10에 있습니다."
    assert agent._normalize_response_brands(text) == text


def test_legit_normalizations_still_apply(agent: HybridChatbotAgent) -> None:
    text = "Beauty & Personal Care에서 Burt's 제품과 Beauty 제품이 인기입니다."
    out = agent._normalize_response_brands(text)
    assert out.startswith("Beauty & Personal Care에서 Burt's Bees 제품과 ")
    assert "Beauty of Joseon 제품이 인기입니다." in out
    assert out.count("Beauty & Personal Care") == 1
