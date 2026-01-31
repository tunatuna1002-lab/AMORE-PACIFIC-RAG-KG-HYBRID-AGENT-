"""
RAG-to-KG 추출기
================
RAG 문서(특히 intelligence/playbook 타입)에서 구조화된 지식을 추출하여
Knowledge Graph에 저장하는 모듈

추출 대상:
- 트렌드 키워드 (HAS_TREND)
- 액션 아이템 (REQUIRES_ACTION)
- 상태/특성 (HAS_STATE)

Usage:
    extractor = RAGKGExtractor(knowledge_graph)
    await extractor.extract_from_chunks(rag_chunks, brand="LANEIGE")
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.domain.entities.relations import Relation, RelationType

if TYPE_CHECKING:
    from src.ontology.knowledge_graph import KnowledgeGraph


class RAGKGExtractor:
    """
    RAG 문서에서 KG 관계를 추출하는 추출기

    추출 규칙:
    1. playbook/intelligence 문서에서 트렌드 키워드 추출
    2. 체크리스트/액션 아이템 추출
    3. 브랜드/제품 상태 특성 추출
    """

    # 트렌드 키워드 패턴 (K-뷰티, 글래스 스킨 등)
    TREND_PATTERNS = [
        r"(?:트렌드|trend)[:\s]*([가-힣A-Za-z\s]+)",
        r"(?:키워드|keyword)[:\s]*([가-힣A-Za-z\s,]+)",
        r"#([A-Za-z가-힣]+skin|[A-Za-z가-힣]+beauty)",  # 해시태그
        r"['\"](glass\s*skin|dewy|glow|hydrat\w*|moisturiz\w*)['\"]",  # 영어 트렌드
        r"['\"](K-?beauty|K-?뷰티)['\"]",
    ]

    # 액션 아이템 패턴
    ACTION_PATTERNS = [
        r"(?:□|☐|▢|\[\s?\])\s*(.+?)(?=\n|$)",  # 체크박스
        r"(?:권장|recommend)[:\s]*(.+?)(?=\n|$)",
        r"(?:액션|action)[:\s]*(.+?)(?=\n|$)",
        r"(?:조치|대응)[:\s]*(.+?)(?=\n|$)",
        r"(?:\d+\.\s*)(.+?(?:하세요|해야|필요|권장))",  # 번호 + 지시문
    ]

    # 상태/특성 패턴
    STATE_PATTERNS = [
        r"(?:상태|status)[:\s]*(급등|급락|안정|상승|하락|유지)",
        r"(?:특성|특징)[:\s]*(.+?)(?=\n|$)",
        r"(?:시장|market)[:\s]*(성장|축소|포화|신흥)",
    ]

    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Args:
            knowledge_graph: 관계를 저장할 KnowledgeGraph 인스턴스
        """
        self._kg = knowledge_graph

    def extract_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        brand: str = "LANEIGE",
        source_context: str | None = None,
    ) -> dict[str, Any]:
        """
        RAG 청크들에서 KG 관계 추출

        Args:
            chunks: RAG 검색 결과 청크 리스트
            brand: 연결할 브랜드 (기본: LANEIGE)
            source_context: 추출 컨텍스트 (예: "daily_insight_2026-01-31")

        Returns:
            {
                "trends_added": int,
                "actions_added": int,
                "states_added": int,
                "total": int,
                "details": [...]
            }
        """
        result = {
            "trends_added": 0,
            "actions_added": 0,
            "states_added": 0,
            "total": 0,
            "details": [],
        }

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")
            doc_type = metadata.get("doc_type", "")
            doc_id = metadata.get("doc_id", "")
            chunk_id = metadata.get("chunk_id", str(uuid.uuid4())[:8])

            # playbook, intelligence, response_guide 문서 우선 처리
            if doc_type not in ["playbook", "intelligence", "response_guide", "metric_guide"]:
                continue

            source_info = {
                "source_doc": doc_id,
                "source_chunk": chunk_id,
                "doc_type": doc_type,
                "extracted_at": datetime.now().isoformat(),
            }
            if source_context:
                source_info["context"] = source_context

            # 1. 트렌드 추출
            trends = self._extract_trends(content)
            for trend in trends:
                rel = Relation(
                    subject=brand,
                    predicate=RelationType.HAS_TREND,
                    object=trend,
                    confidence=0.7,
                    properties={
                        **source_info,
                        "trend_keyword": trend,
                    },
                )
                if self._kg.add_relation(rel):
                    result["trends_added"] += 1
                    result["details"].append({"type": "trend", "value": trend, "source": doc_id})

            # 2. 액션 아이템 추출
            actions = self._extract_actions(content)
            for action in actions:
                rel = Relation(
                    subject=brand,
                    predicate=RelationType.REQUIRES_ACTION,
                    object=action[:100],  # 최대 100자
                    confidence=0.6,
                    properties={
                        **source_info,
                        "action_text": action,
                        "priority": self._infer_action_priority(action),
                    },
                )
                if self._kg.add_relation(rel):
                    result["actions_added"] += 1
                    result["details"].append(
                        {"type": "action", "value": action[:50], "source": doc_id}
                    )

            # 3. 상태/특성 추출
            states = self._extract_states(content)
            for state in states:
                rel = Relation(
                    subject=brand,
                    predicate=RelationType.HAS_STATE,
                    object=state,
                    confidence=0.65,
                    properties={
                        **source_info,
                        "state_value": state,
                    },
                )
                if self._kg.add_relation(rel):
                    result["states_added"] += 1
                    result["details"].append({"type": "state", "value": state, "source": doc_id})

        result["total"] = result["trends_added"] + result["actions_added"] + result["states_added"]
        return result

    def _extract_trends(self, content: str) -> list[str]:
        """트렌드 키워드 추출"""
        trends = set()
        for pattern in self.TREND_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # 정규화: 공백 제거, 소문자
                trend = match.strip().lower()
                if len(trend) >= 3 and len(trend) <= 30:  # 유효한 길이
                    trends.add(trend)

        # 명시적 트렌드 키워드 (자주 등장하는 뷰티 트렌드)
        known_trends = [
            "glass skin",
            "glazed donut",
            "clean girl",
            "k-beauty",
            "skinimalism",
            "skip care",
            "slugging",
            "skin barrier",
            "hydration",
        ]
        content_lower = content.lower()
        for trend in known_trends:
            if trend in content_lower:
                trends.add(trend)

        return list(trends)[:10]  # 최대 10개

    def _extract_actions(self, content: str) -> list[str]:
        """액션 아이템 추출"""
        actions = []
        for pattern in self.ACTION_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                action = match.strip()
                if len(action) >= 10 and len(action) <= 200:  # 유효한 길이
                    actions.append(action)

        return list(set(actions))[:5]  # 최대 5개 (중복 제거)

    def _extract_states(self, content: str) -> list[str]:
        """상태/특성 추출"""
        states = []
        for pattern in self.STATE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                state = match.strip()
                if len(state) >= 2:
                    states.append(state)

        return list(set(states))[:3]  # 최대 3개

    def _infer_action_priority(self, action: str) -> str:
        """액션 우선순위 추론"""
        action_lower = action.lower()

        high_keywords = ["즉시", "긴급", "반드시", "critical", "urgent", "immediate"]
        medium_keywords = ["권장", "필요", "recommend", "should", "consider"]

        for kw in high_keywords:
            if kw in action_lower:
                return "high"

        for kw in medium_keywords:
            if kw in action_lower:
                return "medium"

        return "low"

    def extract_from_signal(
        self,
        signal: dict[str, Any],
        brand: str = "LANEIGE",
    ) -> dict[str, Any]:
        """
        External Signal에서 KG 관계 추출

        Args:
            signal: External Signal 데이터
            brand: 연결할 브랜드

        Returns:
            추출 결과
        """
        result = {
            "trends_added": 0,
            "actions_added": 0,
            "total": 0,
            "details": [],
        }

        content = signal.get("summary", "") + " " + signal.get("title", "")
        source_type = signal.get("source", "unknown")
        url = signal.get("url", "")
        collected_at = signal.get("collected_at", datetime.now().isoformat())

        source_info = {
            "source_signal": source_type,
            "signal_url": url,
            "collected_at": collected_at,
        }

        # 트렌드 추출 (시그널에서는 트렌드 위주)
        trends = self._extract_trends(content)
        for trend in trends:
            rel = Relation(
                subject=brand,
                predicate=RelationType.HAS_TREND,
                object=trend,
                confidence=0.6,  # 시그널은 신뢰도 약간 낮음
                properties={
                    **source_info,
                    "trend_keyword": trend,
                },
            )
            if self._kg.add_relation(rel):
                result["trends_added"] += 1
                result["details"].append({"type": "trend", "value": trend, "source": source_type})

        result["total"] = result["trends_added"] + result["actions_added"]
        return result
