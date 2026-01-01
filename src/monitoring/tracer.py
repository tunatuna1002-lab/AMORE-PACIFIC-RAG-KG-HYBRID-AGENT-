"""
Execution Tracer
에이전트 실행 추적 및 디버깅
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager


@dataclass
class Span:
    """실행 스팬 (추적 단위)"""
    span_id: str
    name: str
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


@dataclass
class Trace:
    """실행 추적"""
    trace_id: str
    session_id: str
    spans: List[Span] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionTracer:
    """실행 추적기"""

    def __init__(self, trace_dir: str = "./data/traces"):
        """
        Args:
            trace_dir: 추적 파일 저장 디렉토리
        """
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self._current_trace: Optional[Trace] = None
        self._span_stack: List[Span] = []

    def start_trace(self, session_id: str, metadata: Optional[Dict] = None) -> str:
        """
        새 추적 시작

        Args:
            session_id: 세션 ID
            metadata: 추가 메타데이터

        Returns:
            trace_id
        """
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._current_trace = Trace(
            trace_id=trace_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        self._span_stack = []
        return trace_id

    def end_trace(self) -> Optional[Dict]:
        """
        추적 종료 및 저장

        Returns:
            추적 요약
        """
        if not self._current_trace:
            return None

        # 미완료 스팬 정리
        for span in self._span_stack:
            span.end_time = time.time()
            span.status = "incomplete"

        # 저장
        self._save_trace()

        summary = self._get_trace_summary()
        self._current_trace = None
        self._span_stack = []

        return summary

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict] = None):
        """
        스팬 컨텍스트 매니저

        Usage:
            with tracer.span("crawl_category", {"category": "lip_care"}):
                # do work
        """
        span = self.start_span(name, attributes)
        try:
            yield span
            self.end_span(status="completed")
        except Exception as e:
            self.end_span(status="failed", error=str(e))
            raise

    def start_span(self, name: str, attributes: Optional[Dict] = None) -> Span:
        """
        스팬 시작

        Args:
            name: 스팬 이름
            attributes: 속성

        Returns:
            생성된 스팬
        """
        parent_id = self._span_stack[-1].span_id if self._span_stack else None

        span = Span(
            span_id=f"span_{uuid.uuid4().hex[:12]}",
            name=name,
            parent_id=parent_id,
            attributes=attributes or {}
        )

        if self._current_trace:
            self._current_trace.spans.append(span)

        self._span_stack.append(span)
        return span

    def end_span(self, status: str = "completed", error: Optional[str] = None) -> Optional[Span]:
        """
        현재 스팬 종료

        Args:
            status: 상태 (completed, failed)
            error: 에러 메시지

        Returns:
            종료된 스팬
        """
        if not self._span_stack:
            return None

        span = self._span_stack.pop()
        span.end_time = time.time()
        span.status = status

        if error:
            span.attributes["error"] = error

        return span

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        """
        현재 스팬에 이벤트 추가

        Args:
            name: 이벤트 이름
            attributes: 이벤트 속성
        """
        if not self._span_stack:
            return

        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        self._span_stack[-1].events.append(event)

    def set_attribute(self, key: str, value: Any) -> None:
        """현재 스팬에 속성 추가"""
        if self._span_stack:
            self._span_stack[-1].attributes[key] = value

    def _save_trace(self) -> None:
        """추적 저장"""
        if not self._current_trace:
            return

        filepath = self.trace_dir / f"{self._current_trace.trace_id}.json"

        # Span을 dict로 변환
        trace_data = {
            "trace_id": self._current_trace.trace_id,
            "session_id": self._current_trace.session_id,
            "created_at": self._current_trace.created_at,
            "metadata": self._current_trace.metadata,
            "spans": [asdict(s) for s in self._current_trace.spans]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2, default=str)

    def _get_trace_summary(self) -> Dict[str, Any]:
        """추적 요약 생성"""
        if not self._current_trace:
            return {}

        spans = self._current_trace.spans
        total_spans = len(spans)
        completed = sum(1 for s in spans if s.status == "completed")
        failed = sum(1 for s in spans if s.status == "failed")

        # 총 소요 시간
        if spans:
            start = min(s.start_time for s in spans)
            end = max(s.end_time or time.time() for s in spans)
            total_duration = (end - start) * 1000
        else:
            total_duration = 0

        return {
            "trace_id": self._current_trace.trace_id,
            "session_id": self._current_trace.session_id,
            "total_spans": total_spans,
            "completed": completed,
            "failed": failed,
            "total_duration_ms": round(total_duration, 2)
        }

    def get_current_trace_id(self) -> Optional[str]:
        """현재 추적 ID 반환"""
        return self._current_trace.trace_id if self._current_trace else None

    def get_current_span(self) -> Optional[Span]:
        """현재 스팬 반환"""
        return self._span_stack[-1] if self._span_stack else None

    def get_span_tree(self) -> List[Dict]:
        """스팬 트리 구조 반환 (디버깅용)"""
        if not self._current_trace:
            return []

        spans = self._current_trace.spans
        root_spans = [s for s in spans if s.parent_id is None]

        def build_tree(span: Span, depth: int = 0) -> Dict:
            children = [s for s in spans if s.parent_id == span.span_id]
            return {
                "name": span.name,
                "status": span.status,
                "duration_ms": span.duration_ms,
                "depth": depth,
                "children": [build_tree(c, depth + 1) for c in children]
            }

        return [build_tree(s) for s in root_spans]

    def format_trace_tree(self) -> str:
        """추적 트리를 문자열로 포맷팅"""
        tree = self.get_span_tree()

        lines = []

        def format_node(node: Dict, indent: str = "") -> None:
            status_icon = {"completed": "✓", "failed": "✗", "running": "◐"}.get(node["status"], "?")
            duration = f" ({node['duration_ms']:.1f}ms)" if node["duration_ms"] else ""
            lines.append(f"{indent}{status_icon} {node['name']}{duration}")

            for i, child in enumerate(node["children"]):
                is_last = i == len(node["children"]) - 1
                child_indent = indent + ("  └─ " if is_last else "  ├─ ")
                next_indent = indent + ("     " if is_last else "  │  ")
                format_node(child, child_indent if indent else "  ")

        for node in tree:
            format_node(node)

        return "\n".join(lines)

    def load_trace(self, trace_id: str) -> Optional[Dict]:
        """저장된 추적 로드"""
        filepath = self.trace_dir / f"{trace_id}.json"

        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_traces(self, limit: int = 20) -> List[Dict]:
        """최근 추적 목록"""
        trace_files = sorted(
            self.trace_dir.glob("trace_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:limit]

        traces = []
        for filepath in trace_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    traces.append({
                        "trace_id": data["trace_id"],
                        "session_id": data["session_id"],
                        "created_at": data["created_at"],
                        "span_count": len(data.get("spans", []))
                    })
            except:
                continue

        return traces
