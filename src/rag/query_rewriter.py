"""
Query Rewriter
대화 맥락 기반 질문 재구성 모듈

Flow:
1. needs_rewrite() - 지시어 감지 (regex, LLM 없이)
2. rewrite() - LLM으로 독립 질문 생성
3. validate() - 재구성 결과 검증 (6자 미만 → fallback)

Usage:
    from src.rag.query_rewriter import QueryRewriter, RewriteResult

    rewriter = QueryRewriter()

    # 지시어 감지 (LLM 호출 없이)
    if rewriter.needs_rewrite("그 제품 가격은?"):
        result = await rewriter.rewrite(
            query="그 제품 가격은?",
            conversation_history=[
                {"role": "user", "content": "LANEIGE Lip Sleeping Mask 분석해줘"},
                {"role": "assistant", "content": "LANEIGE Lip Sleeping Mask는..."}
            ]
        )
        print(result.rewritten_query)  # "LANEIGE Lip Sleeping Mask의 가격은?"
"""

import re
from dataclasses import dataclass

from litellm import acompletion


@dataclass
class RewriteResult:
    """재구성 결과"""

    original_query: str  # 원본 질문
    rewritten_query: str  # 재구성된 질문
    was_rewritten: bool  # 재구성 여부
    needs_clarification: bool  # 명확화 필요 여부
    clarification_message: str  # 명확화 요청 메시지
    confidence: float  # 신뢰도 (0-1)
    resolved_entities: list[str]  # 해소된 지시어 목록


class QueryRewriter:
    """
    대화 히스토리 기반 질문 재구성

    후속 질문에서 지시어(그것, 그 제품, 해당 등)를 이전 대화 맥락을 참조하여
    구체적인 대상으로 치환합니다.

    최적화:
    - needs_rewrite()로 먼저 지시어 존재 여부 확인 (regex, LLM 호출 없이)
    - 캐싱으로 동일 쿼리+히스토리 조합 중복 호출 방지
    - 최근 3턴만 참조하여 토큰 절약

    Example:
        [대화]
        사용자: LANEIGE Lip Sleeping Mask 분석해줘
        어시스턴트: LANEIGE Lip Sleeping Mask는...

        [후속 질문] "그 제품의 가격은?"
        [재구성] "LANEIGE Lip Sleeping Mask의 가격은?"
    """

    # 지시어 패턴 (LLM 호출 전 빠른 감지)
    DEMONSTRATIVE_PATTERNS = [
        # 한국어 지시어 (단독)
        r"(그것|그건|이것|이건|저것|저건)",
        # 한국어 지시어 + 명사
        r"(그|이|저|해당|위|아래|앞|뒤)\s*(제품|브랜드|카테고리|지표|수치|결과|분석|데이터|항목)",
        # 영어 지시어
        r"\b(it|this|that|these|those)\b",
        r"\bthe\s+(same|product|brand|category|metric|result|item)\b",
        # 생략 패턴 (주어 없는 질문)
        r"^(왜|어떻게|언제|얼마나)\s",
        r"^(비교|분석|설명|요약)해",
        # 대명사적 사용
        r"(거기|여기|어디)",
    ]

    # 재구성 프롬프트
    REWRITE_PROMPT = """당신은 대화형 검색 시스템의 질문 재구성 전문가입니다.

## 작업
후속 질문을 이전 대화 맥락을 반영하여 **독립적인 검색 질문**으로 변환하세요.

## 규칙
1. 지시어(그것, 그 제품, 이 브랜드, 해당, 위의 등)를 구체적인 대상으로 치환
2. 생략된 주어/목적어를 맥락에서 복원
3. 1문장, 간결하게 작성
4. 고유명사(브랜드명, 제품명, 지표명)는 정확히 보존
5. 질문 의도를 변경하지 않음
6. 맥락에서 대상을 특정할 수 없으면 원본 그대로 반환

## 예시
[대화]
사용자: LANEIGE Lip Sleeping Mask 분석해줘
어시스턴트: LANEIGE Lip Sleeping Mask는 Lip Care 카테고리에서...

[후속 질문] 그 제품의 가격은?
[재구성] LANEIGE Lip Sleeping Mask의 가격은?

[대화]
사용자: COSRX와 LANEIGE 비교해줘
어시스턴트: 두 브랜드를 비교하면...

[후속 질문] 그럼 SoS는?
[재구성] COSRX와 LANEIGE의 SoS 비교는?

[대화]
사용자: Skin Care 카테고리 분석해줘
어시스턴트: Skin Care 카테고리에서는...

[후속 질문] 왜 떨어졌어?
[재구성] Skin Care 카테고리 SoS가 왜 떨어졌어?

---

[대화 히스토리]
{history}

[후속 질문]
{query}

[재구성 결과]
"""

    # 최소 유효 길이 (이보다 짧으면 모호한 재구성)
    MIN_VALID_LENGTH = 6

    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Args:
            model: LLM 모델명 (기본: gpt-4.1-mini)
        """
        self.model = model
        self._cache: dict[str, str] = {}
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DEMONSTRATIVE_PATTERNS
        ]

    def needs_rewrite(self, query: str) -> bool:
        """
        재구성이 필요한지 빠르게 판단 (LLM 호출 없이)

        Args:
            query: 사용자 질문

        Returns:
            True if 지시어/생략 패턴 감지됨
        """
        for pattern in self._compiled_patterns:
            if pattern.search(query):
                return True
        return False

    async def rewrite(
        self, query: str, conversation_history: list[dict], max_history_turns: int = 3
    ) -> RewriteResult:
        """
        후속 질문을 독립적인 검색 쿼리로 재구성

        Args:
            query: 현재 사용자 질문
            conversation_history: 이전 대화 [{role, content, timestamp?}, ...]
            max_history_turns: 참조할 최대 대화 턴 수 (user+assistant 쌍)

        Returns:
            RewriteResult 객체
        """
        # 히스토리 없으면 원본 반환
        if not conversation_history:
            return self._no_rewrite(query)

        # 캐시 확인
        cache_key = self._make_cache_key(query, conversation_history, max_history_turns)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return RewriteResult(
                original_query=query,
                rewritten_query=cached,
                was_rewritten=(cached != query),
                needs_clarification=False,
                clarification_message="",
                confidence=1.0,
                resolved_entities=[],
            )

        # 히스토리 포맷팅 (최근 N턴만)
        history_text = self._format_history(conversation_history[-(max_history_turns * 2) :])

        # LLM 호출
        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.REWRITE_PROMPT.format(history=history_text, query=query),
                    }
                ],
                temperature=0.1,
                max_tokens=150,
            )

            rewritten = response.choices[0].message.content.strip()

            # 후처리: 불필요한 접두어 제거
            rewritten = self._clean_response(rewritten)

            # 검증: 너무 짧으면 명확화 요청
            if len(rewritten) < self.MIN_VALID_LENGTH:
                return self._needs_clarification(query, rewritten)

            # 캐시 저장
            self._cache[cache_key] = rewritten

            # 캐시 크기 제한 (최대 100개)
            if len(self._cache) > 100:
                # 가장 오래된 항목 제거 (간단한 FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                was_rewritten=(rewritten.lower() != query.lower()),
                needs_clarification=False,
                clarification_message="",
                confidence=0.9,
                resolved_entities=self._extract_resolved(query, rewritten),
            )

        except Exception:
            # 오류 시 원본 반환 (graceful degradation)
            return self._no_rewrite(query)

    def _make_cache_key(self, query: str, history: list[dict], max_turns: int) -> str:
        """캐시 키 생성"""
        # 최근 히스토리의 content만 추출하여 해시
        recent = history[-(max_turns * 2) :]
        history_str = "|".join(h.get("content", "")[:50] for h in recent)
        return f"{query}:{hash(history_str)}"

    def _format_history(self, history: list[dict]) -> str:
        """
        대화 히스토리를 텍스트로 포맷

        Args:
            history: [{role, content}, ...]

        Returns:
            포맷된 문자열
        """
        lines = []
        for turn in history:
            role = "사용자" if turn.get("role") == "user" else "어시스턴트"
            content = turn.get("content", "")
            # 토큰 절약: 긴 응답은 앞부분만
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _clean_response(self, response: str) -> str:
        """
        LLM 응답 후처리

        Args:
            response: LLM 응답

        Returns:
            정제된 질문
        """
        # 불필요한 접두어 제거
        prefixes_to_remove = [
            "[재구성]",
            "[재구성 결과]",
            "재구성:",
            "재구성 결과:",
            "독립 질문:",
        ]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()

        # 따옴표 제거
        response = response.strip("\"'")

        return response

    def _no_rewrite(self, query: str) -> RewriteResult:
        """재구성 불필요 결과 반환"""
        return RewriteResult(
            original_query=query,
            rewritten_query=query,
            was_rewritten=False,
            needs_clarification=False,
            clarification_message="",
            confidence=1.0,
            resolved_entities=[],
        )

    def _needs_clarification(self, original: str, rewritten: str) -> RewriteResult:
        """명확화 필요 결과 반환"""
        return RewriteResult(
            original_query=original,
            rewritten_query=original,  # 원본 유지
            was_rewritten=False,
            needs_clarification=True,
            clarification_message="조금 더 구체적으로 말씀해 주세요. 어떤 제품/브랜드/지표를 의미하시나요?",
            confidence=0.3,
            resolved_entities=[],
        )

    def _extract_resolved(self, original: str, rewritten: str) -> list[str]:
        """
        해소된 지시어 추출 (로깅용)

        Args:
            original: 원본 질문
            rewritten: 재구성된 질문

        Returns:
            해소된 지시어 목록
        """
        resolved = []
        for pattern in self._compiled_patterns[:4]:  # 지시어 패턴만
            matches = pattern.findall(original)
            if matches:
                # findall은 그룹이 있으면 그룹 내용 반환
                for match in matches:
                    if isinstance(match, tuple):
                        resolved.extend(m for m in match if m)
                    else:
                        resolved.append(match)
        return list(set(resolved))

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache.clear()


# 편의 함수
def create_rewrite_result_no_change(query: str) -> RewriteResult:
    """변경 없는 RewriteResult 생성 (편의 함수)"""
    return RewriteResult(
        original_query=query,
        rewritten_query=query,
        was_rewritten=False,
        needs_clarification=False,
        clarification_message="",
        confidence=1.0,
        resolved_entities=[],
    )
