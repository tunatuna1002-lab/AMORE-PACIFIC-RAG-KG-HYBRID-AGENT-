"""
F15: 실제 OpenAI API 호출 차단 가드 검증 테스트
====================================================
tests/conftest.py의 pytest_configure가 OPENAI_API_KEY/OPENAI_BASE_URL/
OPENAI_API_BASE를 더미 값과 닫힌 로컬 주소로 강제 override하고 있는지,
그리고 그 override가 실제로 litellm의 acompletion 호출을 네트워크 없이
빠르게 실패시키는지를 검증한다.

이 테스트는 외부 네트워크 요청을 0건 발생시켜야 한다.
"""

import os
import time

import pytest
from litellm import acompletion
from openai import OpenAIError


def test_llm_api_guard_env_is_dummy_and_closed():
    """세션 전체에서 OPENAI 관련 env가 더미 키 + 닫힌 로컬 주소로 강제되어 있어야 한다"""
    assert os.environ.get("OPENAI_API_KEY", "").startswith("sk-test-dummy")
    assert os.environ.get("OPENAI_BASE_URL") == "http://127.0.0.1:9/v1"
    assert os.environ.get("OPENAI_API_BASE") == "http://127.0.0.1:9/v1"


@pytest.mark.asyncio
async def test_llm_api_guard_blocks_real_acompletion_call():
    """가드가 걸려 있으면 실제 acompletion 호출도 네트워크 없이 즉시 실패해야 한다

    litellm이 실제로 사용하는 값(OPENAI_BASE_URL)을 읽어 닫힌 로컬 포트(127.0.0.1:9)로
    연결을 시도하므로, DNS/TLS 핸드셰이크 없이 connection refused로 즉시 예외가 난다.
    이 테스트가 통과한다는 것은 (1) 외부로 나가는 요청이 전혀 없었고, (2) 수십 초짜리
    네트워크 타임아웃을 기다리지 않았다는 뜻이다.
    """
    start = time.monotonic()
    with pytest.raises(OpenAIError):
        await acompletion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
    elapsed = time.monotonic() - start

    # 실제 OpenAI 엔드포인트로 나갔다면 정상 응답이든 인증 오류든 수백ms~수초가
    # 걸리고, 네트워크가 막힌 환경이라면 수십 초 타임아웃까지 기다리게 된다.
    # 가드가 제대로 동작하면 로컬 포트의 connection refused로 즉시(수 초 이내) 실패한다.
    assert elapsed < 5.0, (
        f"acompletion이 {elapsed:.1f}초나 걸렸다 — 가드가 뚫려 실제 네트워크로 "
        "나갔을 가능성이 있다"
    )
