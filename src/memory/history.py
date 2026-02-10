"""
History Manager
과거 실행 히스토리 관리
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class HistoryManager:
    """실행 히스토리 관리자"""

    def __init__(self, history_dir: str = "./data/history"):
        """
        Args:
            history_dir: 히스토리 저장 디렉토리
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self.history_dir / "execution_history.json"
        self._history: list[dict] = []
        self._load_history()

    def _load_history(self) -> None:
        """히스토리 파일 로드"""
        if self._history_file.exists():
            try:
                with open(self._history_file, encoding="utf-8") as f:
                    self._history = json.load(f)
            except json.JSONDecodeError:
                self._history = []

    def _save_history(self) -> None:
        """히스토리 파일 저장"""
        with open(self._history_file, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2, default=str)

    def add_execution(self, session_summary: dict[str, Any]) -> None:
        """
        실행 기록 추가

        Args:
            session_summary: 세션 요약 정보
        """
        record = {"timestamp": datetime.now().isoformat(), **session_summary}
        self._history.append(record)

        # 최대 1000개 유지
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        self._save_history()

    def get_recent_executions(self, days: int = 7, limit: int = 50) -> list[dict]:
        """
        최근 실행 기록 조회

        Args:
            days: 조회 기간 (일)
            limit: 최대 개수

        Returns:
            실행 기록 리스트
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        recent = [h for h in self._history if h.get("timestamp", "") >= cutoff_str]

        return recent[-limit:]

    def get_success_rate(self, days: int = 7) -> dict[str, float]:
        """
        성공률 계산

        Args:
            days: 계산 기간

        Returns:
            {
                "overall": 0.95,
                "crawler_agent": 0.98,
                ...
            }
        """
        recent = self.get_recent_executions(days=days, limit=1000)

        if not recent:
            return {"overall": 0.0}

        # 전체 성공률
        successful = sum(1 for h in recent if h.get("status") == "completed")
        overall_rate = successful / len(recent)

        # 에이전트별 성공률
        agent_stats = {}
        for record in recent:
            agents = record.get("agents", {})
            for agent_name, agent_data in agents.items():
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {"total": 0, "success": 0}

                agent_stats[agent_name]["total"] += 1
                if agent_data.get("status") == "completed":
                    agent_stats[agent_name]["success"] += 1

        rates = {"overall": round(overall_rate, 3)}
        for agent_name, stats in agent_stats.items():
            if stats["total"] > 0:
                rates[agent_name] = round(stats["success"] / stats["total"], 3)

        return rates

    def get_average_duration(self, days: int = 7) -> dict[str, float]:
        """
        평균 실행 시간 계산

        Args:
            days: 계산 기간

        Returns:
            에이전트별 평균 실행 시간 (초)
        """
        recent = self.get_recent_executions(days=days, limit=1000)

        durations = {}
        for record in recent:
            # 전체 세션
            total = record.get("total_duration_seconds")
            if total:
                if "overall" not in durations:
                    durations["overall"] = []
                durations["overall"].append(total)

            # 에이전트별
            agents = record.get("agents", {})
            for agent_name, agent_data in agents.items():
                duration = agent_data.get("duration_seconds")
                if duration:
                    if agent_name not in durations:
                        durations[agent_name] = []
                    durations[agent_name].append(duration)

        # 평균 계산
        averages = {}
        for name, values in durations.items():
            if values:
                averages[name] = round(sum(values) / len(values), 2)

        return averages

    def get_error_summary(self, days: int = 7) -> list[dict]:
        """
        에러 요약 조회

        Args:
            days: 조회 기간

        Returns:
            에러 기록 리스트
        """
        recent = self.get_recent_executions(days=days, limit=1000)
        errors = []

        for record in recent:
            if record.get("status") == "failed":
                errors.append(
                    {
                        "timestamp": record.get("timestamp"),
                        "session_id": record.get("session_id"),
                        "error": record.get("error"),
                    }
                )

            # 에이전트별 에러
            agents = record.get("agents", {})
            for agent_name, agent_data in agents.items():
                if agent_data.get("status") == "failed":
                    errors.append(
                        {
                            "timestamp": record.get("timestamp"),
                            "session_id": record.get("session_id"),
                            "agent": agent_name,
                            "error": agent_data.get("error"),
                        }
                    )

        return errors

    def get_daily_stats(self, days: int = 30) -> list[dict]:
        """
        일별 통계 조회

        Args:
            days: 조회 기간

        Returns:
            일별 통계 리스트
        """
        recent = self.get_recent_executions(days=days, limit=10000)

        # 일별 그룹핑
        daily = {}
        for record in recent:
            date = record.get("timestamp", "")[:10]  # YYYY-MM-DD
            if date not in daily:
                daily[date] = {"total": 0, "success": 0, "failed": 0}

            daily[date]["total"] += 1
            if record.get("status") == "completed":
                daily[date]["success"] += 1
            else:
                daily[date]["failed"] += 1

        # 리스트 변환
        stats = []
        for date, data in sorted(daily.items()):
            data["date"] = date
            data["success_rate"] = (
                round(data["success"] / data["total"], 3) if data["total"] > 0 else 0
            )
            stats.append(data)

        return stats
