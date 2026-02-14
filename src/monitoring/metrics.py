"""
Quality Metrics
에이전트 품질 및 성능 메트릭
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MetricPoint:
    """메트릭 포인트"""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class QualityMetrics:
    """품질 및 성능 메트릭 수집기"""

    def __init__(self, metrics_dir: str = "./data/metrics"):
        """
        Args:
            metrics_dir: 메트릭 저장 디렉토리
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # 메트릭 저장소
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._timers: dict[str, float] = {}  # 진행 중인 타이머

        # 세션 메트릭
        self._session_start: float | None = None
        self._agent_metrics: dict[str, dict] = {}

    # 기본 메트릭 메서드
    def increment(self, name: str, value: float = 1.0, labels: dict | None = None) -> None:
        """카운터 증가"""
        key = self._make_key(name, labels)
        self._counters[key] += value

    def gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        """게이지 설정"""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def histogram(self, name: str, value: float, labels: dict | None = None) -> None:
        """히스토그램에 값 추가"""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)

        # 최대 1000개 유지
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

    def _make_key(self, name: str, labels: dict | None = None) -> str:
        """라벨이 포함된 메트릭 키 생성"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # 타이머 메서드
    def start_timer(self, name: str) -> None:
        """타이머 시작"""
        self._timers[name] = time.time()

    def stop_timer(self, name: str, labels: dict | None = None) -> float | None:
        """타이머 종료 및 히스토그램에 기록"""
        if name not in self._timers:
            return None

        duration = time.time() - self._timers.pop(name)
        self.histogram(f"{name}_seconds", duration, labels)
        return duration

    # 세션 메트릭
    def start_session(self) -> None:
        """세션 시작"""
        self._session_start = time.time()
        self._agent_metrics = {}
        self.increment("sessions_total")

    def end_session(self, status: str = "completed") -> dict[str, Any]:
        """세션 종료"""
        if not self._session_start:
            return {}

        duration = time.time() - self._session_start
        self.histogram("session_duration_seconds", duration, {"status": status})
        self.increment(f"sessions_{status}")

        summary = {
            "duration_seconds": round(duration, 2),
            "status": status,
            "agents": self._agent_metrics.copy(),
        }

        self._session_start = None
        return summary

    # 에이전트 메트릭
    def record_agent_start(self, agent_name: str) -> None:
        """에이전트 시작 기록"""
        self._agent_metrics[agent_name] = {"start_time": time.time(), "status": "running"}
        self.increment("agent_executions_total", labels={"agent": agent_name})

    def record_agent_complete(self, agent_name: str, result: dict | None = None) -> None:
        """에이전트 완료 기록"""
        if agent_name in self._agent_metrics:
            start = self._agent_metrics[agent_name].get("start_time", time.time())
            duration = time.time() - start

            self._agent_metrics[agent_name].update(
                {"status": "completed", "duration_seconds": round(duration, 2), "result": result}
            )

            self.histogram("agent_duration_seconds", duration, {"agent": agent_name})
            self.increment("agent_success_total", labels={"agent": agent_name})

    def record_agent_error(self, agent_name: str, error: str) -> None:
        """에이전트 에러 기록"""
        if agent_name in self._agent_metrics:
            start = self._agent_metrics[agent_name].get("start_time", time.time())
            duration = time.time() - start

            self._agent_metrics[agent_name].update(
                {"status": "failed", "duration_seconds": round(duration, 2), "error": error}
            )

            self.histogram(
                "agent_duration_seconds", duration, {"agent": agent_name, "status": "error"}
            )
            self.increment("agent_errors_total", labels={"agent": agent_name})

    # LLM 메트릭
    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost: float | None = None,
    ) -> None:
        """LLM 호출 기록"""
        labels = {"model": model}

        self.increment("llm_calls_total", labels=labels)
        self.increment("llm_prompt_tokens_total", prompt_tokens, labels)
        self.increment("llm_completion_tokens_total", completion_tokens, labels)
        self.histogram("llm_latency_ms", latency_ms, labels)

        if cost:
            self.increment("llm_cost_usd_total", cost, labels)

    # 크롤링 메트릭
    def record_crawl(
        self, category: str, products_count: int, duration_seconds: float, success: bool = True
    ) -> None:
        """크롤링 결과 기록"""
        labels = {"category": category}

        self.increment("crawls_total", labels=labels)
        self.gauge("crawl_products_count", products_count, labels)
        self.histogram("crawl_duration_seconds", duration_seconds, labels)

        if success:
            self.increment("crawl_success_total", labels=labels)
        else:
            self.increment("crawl_errors_total", labels=labels)

    # 통계 조회
    def get_counter(self, name: str, labels: dict | None = None) -> float:
        """카운터 값 조회"""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: dict | None = None) -> float | None:
        """게이지 값 조회"""
        key = self._make_key(name, labels)
        return self._gauges.get(key)

    def get_histogram_stats(self, name: str, labels: dict | None = None) -> dict[str, float]:
        """히스토그램 통계 조회"""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p90": sorted_values[int(n * 0.9)],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """모든 메트릭 조회"""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: self.get_histogram_stats(k.split("{")[0], None) for k in self._histograms
            },
        }

    # 보고서 생성
    def generate_report(self) -> dict[str, Any]:
        """성능 보고서 생성"""
        report = {"generated_at": datetime.now().isoformat(), "summary": {}}

        # 세션 통계
        total_sessions = self.get_counter("sessions_total")
        completed = self.get_counter("sessions_completed")
        failed = self.get_counter("sessions_failed")

        report["summary"]["sessions"] = {
            "total": total_sessions,
            "completed": completed,
            "failed": failed,
            "success_rate": round(completed / total_sessions, 3) if total_sessions else 0,
        }

        # 세션 지속시간 통계
        duration_stats = self.get_histogram_stats("session_duration_seconds")
        if duration_stats:
            report["summary"]["session_duration"] = duration_stats

        # 에이전트 통계
        agent_stats = {}
        for key, value in self._counters.items():
            if key.startswith("agent_executions_total{agent="):
                agent = key.split("agent=")[1].rstrip("}")
                if agent not in agent_stats:
                    agent_stats[agent] = {}
                agent_stats[agent]["executions"] = value

        for key, value in self._counters.items():
            if key.startswith("agent_success_total{agent="):
                agent = key.split("agent=")[1].rstrip("}")
                if agent in agent_stats:
                    agent_stats[agent]["success"] = value

        for agent, stats in agent_stats.items():
            total = stats.get("executions", 0)
            success = stats.get("success", 0)
            stats["success_rate"] = round(success / total, 3) if total else 0

        report["summary"]["agents"] = agent_stats

        # LLM 통계
        llm_calls = self.get_counter("llm_calls_total")
        prompt_tokens = self.get_counter("llm_prompt_tokens_total")
        completion_tokens = self.get_counter("llm_completion_tokens_total")
        total_cost = self.get_counter("llm_cost_usd_total")

        report["summary"]["llm"] = {
            "total_calls": llm_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": round(total_cost, 4),
        }

        return report

    # 저장/로드
    def save(self) -> None:
        """메트릭 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = self.metrics_dir / f"metrics_{today}.json"

        data = {
            "saved_at": datetime.now().isoformat(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: v[-100:] for k, v in self._histograms.items()},  # 최근 100개만
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, date: str | None = None) -> bool:
        """메트릭 로드"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        filepath = self.metrics_dir / f"metrics_{date}.json"

        if not filepath.exists():
            return False

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            self._counters = defaultdict(float, data.get("counters", {}))
            self._gauges = data.get("gauges", {})
            self._histograms = defaultdict(list, data.get("histograms", {}))

            return True
        except json.JSONDecodeError:
            return False

    def reset(self) -> None:
        """메트릭 초기화"""
        self._counters = defaultdict(float)
        self._gauges = {}
        self._histograms = defaultdict(list)
        self._timers = {}
        self._session_start = None
        self._agent_metrics = {}
