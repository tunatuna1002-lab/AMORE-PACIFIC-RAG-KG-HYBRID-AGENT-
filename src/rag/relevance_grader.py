"""
Relevance Grader
================
검색 결과의 관련성을 이진 판정하는 모듈

LangGraph Agentic RAG 핵심 패턴:
검색 후 관련성 판정 → 미달 시 쿼리 재작성 → 재검색

비용 최적화: 최대 5개 문서를 1회 LLM 호출로 일괄 판정
모델: gpt-4.1-mini (저비용)
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RelevanceGrader:
    """
    검색 결과 관련성 판정기

    검색된 문서가 쿼리에 실제로 관련이 있는지 이진 판정합니다.
    관련 문서가 부족하면 쿼리 재작성을 트리거합니다.

    Usage:
        grader = RelevanceGrader()
        relevant, irrelevant = await grader.grade_documents(query, documents)
        if len(relevant) < min_relevant:
            # 쿼리 재작성 필요
    """

    GRADING_PROMPT = """당신은 검색 결과의 관련성을 판단하는 전문가입니다.

## 사용자 질문
{query}

## 검색된 문서들
{documents}

## 지시사항
각 문서가 사용자 질문에 답하는 데 실제로 도움이 되는지 판단하세요.

다음 JSON 형식으로만 응답하세요:
```json
{{
    "grades": [
        {{"doc_index": 0, "relevant": true, "reason": "판단 이유"}},
        {{"doc_index": 1, "relevant": false, "reason": "판단 이유"}}
    ]
}}
```

판단 기준:
- relevant=true: 질문에 직접 또는 간접적으로 답변에 도움
- relevant=false: 질문과 무관하거나 너무 일반적"""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_docs_per_call: int = 5,
        min_relevant_threshold: int = 2,
    ):
        """
        Args:
            model: 판정에 사용할 LLM 모델
            temperature: LLM temperature (결정적 판정을 위해 0)
            max_docs_per_call: 1회 호출당 최대 문서 수
            min_relevant_threshold: 관련 문서 최소 수 (미달 시 재검색)
        """
        self.model = model
        self.temperature = temperature
        self.max_docs_per_call = max_docs_per_call
        self.min_relevant_threshold = min_relevant_threshold
        self._stats = {"total_graded": 0, "relevant": 0, "irrelevant": 0}

    async def grade_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        문서 관련성 일괄 판정

        Args:
            query: 사용자 질문
            documents: 검색된 문서 리스트 (각 문서에 "content" 키 필요)

        Returns:
            (관련 문서 리스트, 비관련 문서 리스트)
        """
        if not documents:
            return [], []

        # 문서가 적으면 모두 관련으로 간주 (LLM 호출 절약)
        if len(documents) <= 1:
            self._stats["total_graded"] += len(documents)
            self._stats["relevant"] += len(documents)
            return documents, []

        try:
            grades = await self._call_llm_grading(query, documents[: self.max_docs_per_call])

            relevant = []
            irrelevant = []

            for i, doc in enumerate(documents[: self.max_docs_per_call]):
                is_relevant = grades.get(i, True)  # 기본값: 관련
                if is_relevant:
                    relevant.append(doc)
                    self._stats["relevant"] += 1
                else:
                    irrelevant.append(doc)
                    self._stats["irrelevant"] += 1

            self._stats["total_graded"] += len(documents[: self.max_docs_per_call])

            # max_docs_per_call 이후 문서는 관련으로 간주
            if len(documents) > self.max_docs_per_call:
                remaining = documents[self.max_docs_per_call :]
                relevant.extend(remaining)

            logger.info(
                f"Relevance grading: {len(relevant)} relevant, "
                f"{len(irrelevant)} irrelevant out of {len(documents)}"
            )
            return relevant, irrelevant

        except Exception as e:
            logger.warning(f"Relevance grading failed: {e}, passing all documents")
            return documents, []

    async def _call_llm_grading(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> dict[int, bool]:
        """
        LLM으로 관련성 판정

        Returns:
            {문서인덱스: 관련여부} 딕셔너리
        """
        from litellm import acompletion

        # 문서 포맷
        doc_texts = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            doc_texts.append(f"[문서 {i}] {content}")

        prompt = self.GRADING_PROMPT.format(
            query=query,
            documents="\n\n".join(doc_texts),
        )

        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=self.temperature,
        )

        return self._parse_grades(response.choices[0].message.content)

    def _parse_grades(self, response_text: str) -> dict[int, bool]:
        """LLM 응답 파싱"""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                grades = {}
                for item in data.get("grades", []):
                    idx = item.get("doc_index", 0)
                    grades[idx] = item.get("relevant", True)
                return grades
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse grading response: {e}")

        return {}  # 파싱 실패 시 빈 딕셔너리 (기본값 True 적용)

    def needs_rewrite(self, relevant_count: int) -> bool:
        """관련 문서 수가 임계값 미만인지 확인"""
        return relevant_count < self.min_relevant_threshold

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {**self._stats, "model": self.model}
