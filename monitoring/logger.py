"""
Agent Logger
에이전트 실행 로깅 시스템
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict
from functools import wraps


class AgentLogger:
    """에이전트 로거"""

    _instances: Dict[str, "AgentLogger"] = {}

    def __new__(cls, name: str = "agent", log_dir: str = "./logs"):
        """싱글톤 패턴 (이름별)"""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name: str = "agent", log_dir: str = "./logs"):
        if hasattr(self, "_initialized"):
            return

        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logger()
        self._initialized = True

    def _setup_logger(self) -> None:
        """로거 설정"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)

        # 기존 핸들러 제거
        self.logger.handlers = []

        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 파일 핸들러 (일별)
        today = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{today}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def _format_extra(self, extra: Optional[Dict]) -> str:
        """추가 데이터 포맷팅"""
        if not extra:
            return ""
        try:
            return f" | {json.dumps(extra, ensure_ascii=False, default=str)}"
        except:
            return f" | {str(extra)}"

    def debug(self, message: str, extra: Optional[Dict] = None) -> None:
        """디버그 로그"""
        self.logger.debug(f"{message}{self._format_extra(extra)}")

    def info(self, message: str, extra: Optional[Dict] = None) -> None:
        """정보 로그"""
        self.logger.info(f"{message}{self._format_extra(extra)}")

    def warning(self, message: str, extra: Optional[Dict] = None) -> None:
        """경고 로그"""
        self.logger.warning(f"{message}{self._format_extra(extra)}")

    def error(self, message: str, extra: Optional[Dict] = None, exc_info: bool = False) -> None:
        """에러 로그"""
        self.logger.error(f"{message}{self._format_extra(extra)}", exc_info=exc_info)

    def critical(self, message: str, extra: Optional[Dict] = None) -> None:
        """치명적 에러 로그"""
        self.logger.critical(f"{message}{self._format_extra(extra)}")

    # 에이전트 전용 로그 메서드
    def agent_start(self, agent_name: str, task: Optional[str] = None) -> None:
        """에이전트 시작 로그"""
        self.info(f"🚀 Agent Started: {agent_name}", {"task": task})

    def agent_complete(self, agent_name: str, duration: float, result: Optional[str] = None) -> None:
        """에이전트 완료 로그"""
        self.info(
            f"✅ Agent Completed: {agent_name}",
            {"duration_seconds": round(duration, 2), "result": result}
        )

    def agent_error(self, agent_name: str, error: str, duration: Optional[float] = None) -> None:
        """에이전트 에러 로그"""
        self.error(
            f"❌ Agent Failed: {agent_name}",
            {"error": error, "duration_seconds": round(duration, 2) if duration else None}
        )

    def tool_call(self, tool_name: str, params: Optional[Dict] = None) -> None:
        """도구 호출 로그"""
        self.debug(f"🔧 Tool Call: {tool_name}", {"params": params})

    def tool_result(self, tool_name: str, success: bool, result_summary: Optional[str] = None) -> None:
        """도구 결과 로그"""
        status = "✓" if success else "✗"
        self.debug(f"   {status} Tool Result: {tool_name}", {"success": success, "summary": result_summary})

    def llm_request(self, model: str, prompt_tokens: Optional[int] = None) -> None:
        """LLM 요청 로그"""
        self.debug(f"🤖 LLM Request: {model}", {"prompt_tokens": prompt_tokens})

    def llm_response(self, model: str, completion_tokens: Optional[int] = None, latency_ms: Optional[float] = None) -> None:
        """LLM 응답 로그"""
        self.debug(
            f"   LLM Response: {model}",
            {"completion_tokens": completion_tokens, "latency_ms": round(latency_ms, 1) if latency_ms else None}
        )

    def workflow_step(self, step: str, status: str, details: Optional[Dict] = None) -> None:
        """워크플로우 스텝 로그"""
        emoji = {"start": "▶", "complete": "✓", "skip": "⏭", "error": "✗"}.get(status, "•")
        self.info(f"{emoji} Workflow: {step} [{status}]", details)

    def metric(self, name: str, value: Any, unit: Optional[str] = None) -> None:
        """메트릭 로그"""
        self.debug(f"📊 Metric: {name} = {value}{' ' + unit if unit else ''}")


def log_execution(logger: Optional[AgentLogger] = None):
    """함수 실행 로깅 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _logger = logger or AgentLogger()
            func_name = func.__name__
            start = datetime.now()

            _logger.debug(f"Executing: {func_name}")
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start).total_seconds()
                _logger.debug(f"Completed: {func_name}", {"duration": round(duration, 3)})
                return result
            except Exception as e:
                duration = (datetime.now() - start).total_seconds()
                _logger.error(f"Failed: {func_name}", {"error": str(e), "duration": round(duration, 3)})
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            _logger = logger or AgentLogger()
            func_name = func.__name__
            start = datetime.now()

            _logger.debug(f"Executing: {func_name}")
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start).total_seconds()
                _logger.debug(f"Completed: {func_name}", {"duration": round(duration, 3)})
                return result
            except Exception as e:
                duration = (datetime.now() - start).total_seconds()
                _logger.error(f"Failed: {func_name}", {"error": str(e), "duration": round(duration, 3)})
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
