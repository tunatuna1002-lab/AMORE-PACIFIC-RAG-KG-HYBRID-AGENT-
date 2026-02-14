"""
LLM Completion Retry Helper
============================
공통 LLM 호출 재시도 로직 (지수 백오프 + 타임아웃)
"""

import asyncio
import logging

from litellm import acompletion

logger = logging.getLogger(__name__)


async def llm_completion_with_retry(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int = 500,
    temperature: float = 0.2,
    max_retries: int = 3,
    timeout: float = 30.0,
) -> dict:
    """
    LLM completion with exponential backoff retry.

    Args:
        model: LLM 모델명
        messages: 메시지 리스트
        max_tokens: 최대 토큰 수
        temperature: 온도
        max_retries: 최대 재시도 횟수
        timeout: 단일 호출 타임아웃 (초)

    Returns:
        LLM 응답 객체

    Raises:
        Exception: 모든 재시도 실패 시
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                acompletion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            last_error = TimeoutError(f"LLM call timed out after {timeout}s")
            logger.warning(f"LLM timeout (attempt {attempt + 1}/{max_retries}): {model}")
        except Exception as e:
            last_error = e
            logger.warning(f"LLM error (attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            wait_time = 2**attempt
            await asyncio.sleep(wait_time)

    raise last_error
