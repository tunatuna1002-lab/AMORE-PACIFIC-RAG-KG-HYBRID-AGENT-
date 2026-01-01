"""
Context Manager
대화 컨텍스트 및 워크플로우 상태 관리
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ConversationTurn:
    """대화 턴"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """워크플로우 실행 컨텍스트"""
    current_step: str = ""
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DataContext:
    """데이터 컨텍스트"""
    last_crawl_date: Optional[str] = None
    categories_crawled: List[str] = field(default_factory=list)
    products_count: int = 0
    laneige_products: List[Dict[str, Any]] = field(default_factory=list)
    metrics_calculated: bool = False
    insights_generated: bool = False


class ContextManager:
    """컨텍스트 관리자"""

    def __init__(self, context_dir: str = "./data/context"):
        """
        Args:
            context_dir: 컨텍스트 저장 디렉토리
        """
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)

        # 컨텍스트 상태
        self._conversation: List[ConversationTurn] = []
        self._workflow: WorkflowContext = WorkflowContext()
        self._data: DataContext = DataContext()
        self._variables: Dict[str, Any] = {}

        # 최대 대화 기록
        self._max_conversation_turns = 100

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """사용자 메시지 추가"""
        turn = ConversationTurn(
            role="user",
            content=content,
            metadata=metadata or {}
        )
        self._conversation.append(turn)
        self._trim_conversation()

    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """어시스턴트 메시지 추가"""
        turn = ConversationTurn(
            role="assistant",
            content=content,
            metadata=metadata or {}
        )
        self._conversation.append(turn)
        self._trim_conversation()

    def _trim_conversation(self) -> None:
        """대화 기록 정리 (최대 개수 유지)"""
        if len(self._conversation) > self._max_conversation_turns:
            self._conversation = self._conversation[-self._max_conversation_turns:]

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """최근 대화 기록 조회"""
        recent = self._conversation[-limit:]
        return [asdict(turn) for turn in recent]

    def get_conversation_summary(self) -> str:
        """대화 요약 생성 (LLM 프롬프트용)"""
        if not self._conversation:
            return "이전 대화 없음"

        recent = self._conversation[-5:]
        lines = []
        for turn in recent:
            role = "사용자" if turn.role == "user" else "어시스턴트"
            content = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
            lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    # 워크플로우 컨텍스트
    def start_workflow(self, steps: List[str]) -> None:
        """워크플로우 시작"""
        self._workflow = WorkflowContext(
            pending_steps=steps.copy(),
            current_step=steps[0] if steps else ""
        )

    def advance_workflow(self, result: Optional[Any] = None) -> Optional[str]:
        """
        다음 스텝으로 진행

        Returns:
            다음 스텝 이름 또는 None (완료 시)
        """
        if self._workflow.current_step:
            self._workflow.completed_steps.append(self._workflow.current_step)
            if result is not None:
                self._workflow.step_results[self._workflow.current_step] = result

        if self._workflow.pending_steps:
            self._workflow.current_step = self._workflow.pending_steps.pop(0)
            return self._workflow.current_step

        self._workflow.current_step = ""
        return None

    def record_workflow_error(self, step: str, error: str) -> None:
        """워크플로우 에러 기록"""
        self._workflow.errors.append({
            "step": step,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_workflow_status(self) -> Dict[str, Any]:
        """워크플로우 상태 조회"""
        total = (
            len(self._workflow.completed_steps) +
            len(self._workflow.pending_steps) +
            (1 if self._workflow.current_step else 0)
        )
        completed = len(self._workflow.completed_steps)

        return {
            "current_step": self._workflow.current_step,
            "completed_steps": self._workflow.completed_steps,
            "pending_steps": self._workflow.pending_steps,
            "progress": f"{completed}/{total}" if total > 0 else "0/0",
            "progress_percent": round(completed / total * 100, 1) if total > 0 else 0,
            "has_errors": len(self._workflow.errors) > 0,
            "error_count": len(self._workflow.errors)
        }

    # 데이터 컨텍스트
    def update_crawl_data(
        self,
        categories: List[str],
        products_count: int,
        laneige_products: List[Dict]
    ) -> None:
        """크롤링 데이터 업데이트"""
        self._data.last_crawl_date = datetime.now().isoformat()
        self._data.categories_crawled = categories
        self._data.products_count = products_count
        self._data.laneige_products = laneige_products

    def set_metrics_calculated(self, calculated: bool = True) -> None:
        """지표 계산 완료 표시"""
        self._data.metrics_calculated = calculated

    def set_insights_generated(self, generated: bool = True) -> None:
        """인사이트 생성 완료 표시"""
        self._data.insights_generated = generated

    def get_data_status(self) -> Dict[str, Any]:
        """데이터 상태 조회"""
        return {
            "last_crawl": self._data.last_crawl_date,
            "categories": self._data.categories_crawled,
            "total_products": self._data.products_count,
            "laneige_count": len(self._data.laneige_products),
            "metrics_ready": self._data.metrics_calculated,
            "insights_ready": self._data.insights_generated
        }

    def get_laneige_products(self) -> List[Dict]:
        """LANEIGE 제품 목록 반환"""
        return self._data.laneige_products

    # 변수 관리
    def set_variable(self, key: str, value: Any) -> None:
        """변수 설정"""
        self._variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """변수 조회"""
        return self._variables.get(key, default)

    def clear_variables(self) -> None:
        """변수 초기화"""
        self._variables.clear()

    # 전체 컨텍스트
    def get_full_context(self) -> Dict[str, Any]:
        """전체 컨텍스트 조회"""
        return {
            "conversation": self.get_conversation_history(limit=10),
            "workflow": self.get_workflow_status(),
            "data": self.get_data_status(),
            "variables": self._variables.copy()
        }

    def build_llm_context(self) -> str:
        """LLM 프롬프트용 컨텍스트 문자열 생성"""
        parts = []

        # 데이터 상태
        data = self.get_data_status()
        if data["last_crawl"]:
            parts.append(f"[데이터 현황]")
            parts.append(f"- 마지막 크롤링: {data['last_crawl'][:10]}")
            parts.append(f"- 수집 카테고리: {', '.join(data['categories'])}")
            parts.append(f"- 전체 제품 수: {data['total_products']}")
            parts.append(f"- LANEIGE 제품 수: {data['laneige_count']}")
            parts.append("")

        # 워크플로우 상태
        workflow = self.get_workflow_status()
        if workflow["current_step"]:
            parts.append(f"[워크플로우 진행]")
            parts.append(f"- 현재 단계: {workflow['current_step']}")
            parts.append(f"- 진행률: {workflow['progress']} ({workflow['progress_percent']}%)")
            parts.append("")

        # 최근 대화
        conversation_summary = self.get_conversation_summary()
        if conversation_summary != "이전 대화 없음":
            parts.append(f"[최근 대화]")
            parts.append(conversation_summary)

        return "\n".join(parts) if parts else "컨텍스트 없음"

    # 저장/로드
    def save_context(self, session_id: str) -> None:
        """컨텍스트 저장"""
        filepath = self.context_dir / f"{session_id}_context.json"

        data = {
            "conversation": [asdict(t) for t in self._conversation],
            "workflow": asdict(self._workflow),
            "data": asdict(self._data),
            "variables": self._variables,
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_context(self, session_id: str) -> bool:
        """컨텍스트 로드"""
        filepath = self.context_dir / f"{session_id}_context.json"

        if not filepath.exists():
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._conversation = [
                ConversationTurn(**t) for t in data.get("conversation", [])
            ]
            self._workflow = WorkflowContext(**data.get("workflow", {}))
            self._data = DataContext(**data.get("data", {}))
            self._variables = data.get("variables", {})

            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def reset(self) -> None:
        """컨텍스트 초기화"""
        self._conversation = []
        self._workflow = WorkflowContext()
        self._data = DataContext()
        self._variables = {}
