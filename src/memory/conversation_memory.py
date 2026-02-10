"""
Conversation Memory
===================
LangGraph add_messages reducer 패턴 적용.

세션별 대화 이력 관리:
- 최근 N턴 전문 보존
- 이전 턴 요약 (비용 절감)
- 엔티티 추적 (브랜드, 카테고리 자동 추출/유지)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """대화 턴"""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    entities: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "entities": self.entities,
        }


@dataclass
class ConversationContext:
    """대화 컨텍스트 (LLM 프롬프트용)"""

    summary: str = ""  # 이전 대화 요약
    recent_turns: list[dict] = field(default_factory=list)  # 최근 N턴
    total_turns: int = 0
    tracked_entities: dict[str, list[str]] = field(default_factory=dict)
    session_id: str = ""

    def to_prompt_text(self) -> str:
        """LLM 프롬프트용 텍스트"""
        parts = []
        if self.summary:
            parts.append(f"## 이전 대화 요약\n{self.summary}\n")
        if self.recent_turns:
            parts.append("## 최근 대화")
            for turn in self.recent_turns:
                role_label = "사용자" if turn["role"] == "user" else "어시스턴트"
                content = turn["content"]
                if len(content) > 300:
                    content = content[:300] + "..."
                parts.append(f"**{role_label}**: {content}")
            parts.append("")
        if self.tracked_entities:
            entities_str = ", ".join(
                f"{k}: {', '.join(v)}" for k, v in self.tracked_entities.items() if v
            )
            if entities_str:
                parts.append(f"## 대화에서 언급된 엔티티\n{entities_str}\n")
        return "\n".join(parts)


class ConversationMemory:
    """
    세션별 대화 메모리

    Usage:
        memory = ConversationMemory()
        memory.add_turn("session1", "user", "LANEIGE 점유율 알려줘")
        memory.add_turn("session1", "assistant", "현재 LANEIGE의 SoS는...")
        ctx = memory.get_context("session1")
    """

    def __init__(
        self,
        max_recent_turns: int = 6,
        max_sessions: int = 100,
        max_turns_per_session: int = 50,
    ):
        """
        Args:
            max_recent_turns: LLM에 전달할 최근 턴 수
            max_sessions: 최대 세션 수 (LRU eviction)
            max_turns_per_session: 세션당 최대 턴 수
        """
        self.max_recent_turns = max_recent_turns
        self.max_sessions = max_sessions
        self.max_turns_per_session = max_turns_per_session

        # 세션별 대화 이력
        self._sessions: dict[str, list[ConversationTurn]] = defaultdict(list)
        # 세션별 요약
        self._summaries: dict[str, str] = {}
        # 세션별 추적 엔티티 (누적)
        self._tracked_entities: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # 세션 접근 순서 (LRU)
        self._access_order: list[str] = []

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        entities: dict[str, list[str]] | None = None,
    ) -> None:
        """
        대화 턴 추가

        Args:
            session_id: 세션 ID
            role: "user" 또는 "assistant"
            content: 메시지 내용
            entities: 추출된 엔티티 (없으면 자동 추출)
        """
        # LRU 업데이트
        self._touch_session(session_id)

        # 엔티티 자동 추출
        if entities is None:
            entities = self._extract_entities(content)

        turn = ConversationTurn(
            role=role,
            content=content,
            entities=entities,
        )

        self._sessions[session_id].append(turn)

        # 엔티티 누적
        for key, values in entities.items():
            for v in values:
                self._tracked_entities[session_id][key].add(v)

        # 턴 수 제한
        if len(self._sessions[session_id]) > self.max_turns_per_session:
            # 오래된 턴 요약 후 제거
            excess = len(self._sessions[session_id]) - self.max_turns_per_session
            old_turns = self._sessions[session_id][:excess]
            self._summarize_turns(session_id, old_turns)
            self._sessions[session_id] = self._sessions[session_id][excess:]

        logger.debug(
            f"Turn added: session={session_id}, role={role}, "
            f"total_turns={len(self._sessions[session_id])}"
        )

    def get_context(self, session_id: str) -> ConversationContext:
        """
        세션 대화 컨텍스트 반환

        Args:
            session_id: 세션 ID

        Returns:
            ConversationContext
        """
        self._touch_session(session_id)

        turns = self._sessions.get(session_id, [])
        recent = turns[-self.max_recent_turns :] if turns else []

        tracked = {}
        for key, values in self._tracked_entities.get(session_id, {}).items():
            tracked[key] = sorted(values)

        return ConversationContext(
            summary=self._summaries.get(session_id, ""),
            recent_turns=[t.to_dict() for t in recent],
            total_turns=len(turns),
            tracked_entities=tracked,
            session_id=session_id,
        )

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """세션 전체 대화 이력"""
        return [t.to_dict() for t in self._sessions.get(session_id, [])]

    def clear_session(self, session_id: str) -> None:
        """세션 초기화"""
        self._sessions.pop(session_id, None)
        self._summaries.pop(session_id, None)
        self._tracked_entities.pop(session_id, None)
        if session_id in self._access_order:
            self._access_order.remove(session_id)

    def _touch_session(self, session_id: str) -> None:
        """세션 접근 기록 (LRU)"""
        if session_id in self._access_order:
            self._access_order.remove(session_id)
        self._access_order.append(session_id)

        # 최대 세션 수 초과 시 LRU eviction
        while len(self._access_order) > self.max_sessions:
            oldest = self._access_order.pop(0)
            self.clear_session(oldest)
            logger.info(f"Session evicted (LRU): {oldest}")

    def _summarize_turns(self, session_id: str, turns: list[ConversationTurn]) -> None:
        """오래된 턴 요약 (단순 텍스트 압축)"""
        existing = self._summaries.get(session_id, "")
        new_summary_parts = []

        for turn in turns:
            role = "사용자" if turn.role == "user" else "시스템"
            content = turn.content[:100]
            new_summary_parts.append(f"{role}: {content}")

        new_summary = " | ".join(new_summary_parts)

        if existing:
            self._summaries[session_id] = f"{existing} | {new_summary}"
        else:
            self._summaries[session_id] = new_summary

        # 요약 길이 제한 (2000자)
        if len(self._summaries[session_id]) > 2000:
            self._summaries[session_id] = self._summaries[session_id][-2000:]

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """텍스트에서 간단한 엔티티 추출"""
        entities: dict[str, list[str]] = {"brands": [], "indicators": []}
        text_lower = text.lower()

        # 브랜드
        brand_keywords = {
            "laneige": "LANEIGE",
            "라네즈": "LANEIGE",
            "cosrx": "COSRX",
            "코스알엑스": "COSRX",
            "tirtir": "TIRTIR",
            "티르티르": "TIRTIR",
            "rare beauty": "Rare Beauty",
            "innisfree": "Innisfree",
            "sulwhasoo": "Sulwhasoo",
            "설화수": "Sulwhasoo",
        }
        for keyword, brand in brand_keywords.items():
            if keyword in text_lower and brand not in entities["brands"]:
                entities["brands"].append(brand)

        # 지표
        indicator_keywords = {
            "sos": "SoS",
            "점유율": "SoS",
            "share of shelf": "SoS",
            "hhi": "HHI",
            "집중도": "HHI",
            "cpi": "CPI",
            "가격지수": "CPI",
        }
        for keyword, indicator in indicator_keywords.items():
            if keyword in text_lower and indicator not in entities["indicators"]:
                entities["indicators"].append(indicator)

        return entities

    def get_stats(self) -> dict[str, Any]:
        """통계"""
        return {
            "active_sessions": len(self._sessions),
            "total_turns": sum(len(t) for t in self._sessions.values()),
            "sessions_with_summary": len(self._summaries),
        }
