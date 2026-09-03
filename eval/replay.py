"""
Golden-set record/replay
========================
리팩토링 회귀 게이트를 오프라인으로 돌리기 위한 기록·재생 에이전트.

- RecordingAgent: 실제 에이전트(chat())를 감싸 결과를 JSONL로 기록한다 (LLM 키 필요, 1회).
- ReplayAgent: 기록된 JSONL을 읽어 같은 질문에 같은 결과를 돌려준다 (키·네트워크 불필요).

EvalRunner는 result["hybrid_context"]의 .entities/.inferences/.rag_chunks 속성만 읽으므로
재생 시에는 그 세 속성만 가진 경량 객체로 복원한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReplayHybridContext:
    """EvalRunner가 접근하는 속성만 가진 HybridContext 대체물."""

    entities: dict[str, list[str]] = field(default_factory=dict)
    inferences: list[dict[str, Any]] = field(default_factory=list)
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    ontology_facts: list[dict[str, Any]] = field(default_factory=list)
    combined_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _serialize(obj: Any) -> Any:
    """dataclass / to_dict 지원 객체를 JSON 직렬화 가능한 형태로 변환."""
    if hasattr(obj, "to_dict"):
        return _serialize(obj.to_dict())
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _deserialize_result(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    hc = result.get("hybrid_context")
    if isinstance(hc, dict):
        result["hybrid_context"] = ReplayHybridContext(
            entities=hc.get("entities", {}) or {},
            inferences=hc.get("inferences", []) or [],
            rag_chunks=hc.get("rag_chunks", []) or [],
            ontology_facts=hc.get("ontology_facts", []) or [],
            combined_context=hc.get("combined_context", "") or "",
            metadata=hc.get("metadata", {}) or {},
        )
    return result


class RecordingAgent:
    """실제 에이전트를 감싸 chat() 결과를 JSONL에 기록한다."""

    def __init__(self, agent: Any, output_path: str | Path):
        self.agent = agent
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")

    async def chat(self, question: str, **kwargs: Any) -> dict[str, Any]:
        result = await self.agent.chat(question, **kwargs)
        record = {"question": question, "result": _serialize(result)}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        return result

    def close(self) -> None:
        self._fh.close()


class ReplayAgent:
    """기록된 JSONL에서 질문→결과를 재생한다."""

    def __init__(self, recording_path: str | Path):
        self.recording_path = Path(recording_path)
        self._by_question: dict[str, dict[str, Any]] = {}
        with self.recording_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._by_question[rec["question"]] = rec["result"]

    def __len__(self) -> int:
        return len(self._by_question)

    async def chat(self, question: str, **kwargs: Any) -> dict[str, Any]:
        if question not in self._by_question:
            raise KeyError(f"No recording for question: {question!r}")
        return _deserialize_result(self._by_question[question])
