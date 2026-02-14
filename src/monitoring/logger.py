"""
Agent Logger
에이전트 실행 로깅 시스템
"""

import json
import logging
import re
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any


class SensitiveDataFilter(logging.Filter):
    """
    API 키 및 민감 정보를 마스킹하는 로깅 필터

    마스킹 대상:
    - OpenAI API Key (sk-...)
    - Apify API Key (apify_api_...)
    - Tavily API Key (tvly-...)
    - 일반 API 키/토큰/비밀번호 패턴
    """

    PATTERNS = [
        # OpenAI API Key
        (r"sk-[a-zA-Z0-9]{20,}", "sk-****"),
        # Apify API Key
        (r"apify_api_[a-zA-Z0-9]{20,}", "apify_api_****"),
        # Tavily API Key
        (r"tvly-[a-zA-Z0-9]{20,}", "tvly-****"),
        # Generic API key/token/secret patterns
        # Matches: api_key=xxx, apiKey: "xxx", token='xxx', password=xxx
        (
            r'(?i)(api[_-]?key|token|secret|password)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?',
            r"\1=****",
        ),
        # Bearer tokens
        (r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", "Bearer ****"),
        # Generic long alphanumeric strings that look like keys (conservative)
        (r"\b[a-zA-Z0-9_\-]{40,}\b", "****"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        로그 레코드의 메시지에서 민감 정보 마스킹

        Args:
            record: 로깅 레코드

        Returns:
            True (항상 로그 통과, 메시지만 수정)
        """
        # 메시지 마스킹
        if record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                msg = re.sub(pattern, replacement, msg)
            record.msg = msg

        # args 마스킹 (포맷팅 인자)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._mask_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._mask_value(arg) for arg in record.args)

        return True

    def _mask_value(self, value: Any) -> Any:
        """개별 값 마스킹"""
        if isinstance(value, str):
            for pattern, replacement in self.PATTERNS:
                value = re.sub(pattern, replacement, value)
        elif isinstance(value, dict):
            return {k: self._mask_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(self._mask_value(item) for item in value)
        return value


class ErrorDeduplicationFilter(logging.Filter):
    """
    동일 에러 메시지 중복 제거 필터

    동일한 에러 메시지가 짧은 시간 내 반복될 때 로그 폭주를 방지합니다.
    window_seconds 이내에 max_count 이상 동일 메시지가 발생하면
    이후 메시지를 억제하고, 억제 종료 시 요약 로그를 출력합니다.

    Usage:
        dedup_filter = ErrorDeduplicationFilter(window_seconds=60, max_count=3)
        logger.addFilter(dedup_filter)
    """

    def __init__(
        self,
        window_seconds: int = 60,
        max_count: int = 3,
        name: str = "",
    ):
        super().__init__(name)
        self.window_seconds = window_seconds
        self.max_count = max_count
        # {message_key: {"count": int, "first_seen": float, "suppressed": int}}
        self._seen: dict[str, dict[str, Any]] = {}

    def _message_key(self, record: logging.LogRecord) -> str:
        """로그 레코드에서 중복 판단 키 생성"""
        # 에러/경고만 중복 제거 대상
        msg = str(record.msg)
        # 숫자와 타임스탬프를 제거하여 유사 메시지 그룹화
        import re

        normalized = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TIMESTAMP>", msg)
        normalized = re.sub(r"\b\d+\.\d+\b", "<NUM>", normalized)
        return f"{record.levelno}:{normalized[:200]}"

    def filter(self, record: logging.LogRecord) -> bool:
        """
        중복 에러 필터링

        Returns:
            True: 로그 통과
            False: 로그 억제
        """
        # DEBUG/INFO는 필터링하지 않음
        if record.levelno < logging.WARNING:
            return True

        import time

        now = time.time()
        key = self._message_key(record)

        # 만료된 항목 정리
        self._cleanup(now)

        if key not in self._seen:
            self._seen[key] = {
                "count": 1,
                "first_seen": now,
                "suppressed": 0,
            }
            return True

        entry = self._seen[key]
        entry["count"] += 1

        if entry["count"] <= self.max_count:
            return True

        # 억제
        entry["suppressed"] += 1

        # 첫 억제 시 또는 10건마다 요약 로그
        if entry["suppressed"] == 1 or entry["suppressed"] % 10 == 0:
            record.msg = (
                f"[Dedup] {entry['suppressed']}건 동일 에러 억제됨 (원본: {str(record.msg)[:100]})"
            )
            return True

        return False

    def _cleanup(self, now: float) -> None:
        """만료된 항목 정리"""
        expired = [
            key
            for key, entry in self._seen.items()
            if now - entry["first_seen"] > self.window_seconds
        ]
        for key in expired:
            entry = self._seen.pop(key)
            if entry["suppressed"] > 0:
                # 만료 시 최종 요약은 logging 모듈로 직접 출력
                logging.getLogger(__name__).info(
                    f"[Dedup Summary] {entry['suppressed']}건 동일 에러가 "
                    f"{self.window_seconds}초 내 억제되었습니다"
                )

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        total_suppressed = sum(entry["suppressed"] for entry in self._seen.values())
        return {
            "tracked_messages": len(self._seen),
            "total_suppressed": total_suppressed,
            "window_seconds": self.window_seconds,
            "max_count": self.max_count,
        }


class AgentLogger:
    """에이전트 로거"""

    _instances: dict[str, "AgentLogger"] = {}

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

        # 민감 정보 마스킹 필터 생성
        sensitive_filter = SensitiveDataFilter()
        dedup_filter = ErrorDeduplicationFilter(window_seconds=60, max_count=3)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        console_handler.addFilter(sensitive_filter)  # 필터 적용
        console_handler.addFilter(dedup_filter)
        self.logger.addHandler(console_handler)

        # 파일 핸들러 (일별)
        today = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{today}.log", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        file_handler.setFormatter(file_format)
        file_handler.addFilter(sensitive_filter)  # 필터 적용
        file_handler.addFilter(dedup_filter)
        self.logger.addHandler(file_handler)

    def _format_extra(self, extra: dict | None) -> str:
        """추가 데이터 포맷팅"""
        if not extra:
            return ""
        try:
            return f" | {json.dumps(extra, ensure_ascii=False, default=str)}"
        except Exception:
            # JSON 직렬화 실패 시 문자열 변환 사용
            return f" | {str(extra)}"

    def debug(self, message: str, extra: dict | None = None) -> None:
        """디버그 로그"""
        self.logger.debug(f"{message}{self._format_extra(extra)}")

    def info(self, message: str, extra: dict | None = None) -> None:
        """정보 로그"""
        self.logger.info(f"{message}{self._format_extra(extra)}")

    def warning(self, message: str, extra: dict | None = None) -> None:
        """경고 로그"""
        self.logger.warning(f"{message}{self._format_extra(extra)}")

    def error(self, message: str, extra: dict | None = None, exc_info: bool = False) -> None:
        """에러 로그"""
        self.logger.error(f"{message}{self._format_extra(extra)}", exc_info=exc_info)

    def critical(self, message: str, extra: dict | None = None) -> None:
        """치명적 에러 로그"""
        self.logger.critical(f"{message}{self._format_extra(extra)}")

    # 에이전트 전용 로그 메서드
    def agent_start(self, agent_name: str, task: str | None = None) -> None:
        """에이전트 시작 로그"""
        self.info(f"🚀 Agent Started: {agent_name}", {"task": task})

    def agent_complete(self, agent_name: str, duration: float, result: str | None = None) -> None:
        """에이전트 완료 로그"""
        self.info(
            f"✅ Agent Completed: {agent_name}",
            {"duration_seconds": round(duration, 2), "result": result},
        )

    def agent_error(self, agent_name: str, error: str, duration: float | None = None) -> None:
        """에이전트 에러 로그"""
        self.error(
            f"❌ Agent Failed: {agent_name}",
            {"error": error, "duration_seconds": round(duration, 2) if duration else None},
        )

    def tool_call(self, tool_name: str, params: dict | None = None) -> None:
        """도구 호출 로그"""
        self.debug(f"🔧 Tool Call: {tool_name}", {"params": params})

    def tool_result(self, tool_name: str, success: bool, result_summary: str | None = None) -> None:
        """도구 결과 로그"""
        status = "✓" if success else "✗"
        self.debug(
            f"   {status} Tool Result: {tool_name}", {"success": success, "summary": result_summary}
        )

    def llm_request(self, model: str, prompt_tokens: int | None = None) -> None:
        """LLM 요청 로그"""
        self.debug(f"🤖 LLM Request: {model}", {"prompt_tokens": prompt_tokens})

    def llm_response(
        self, model: str, completion_tokens: int | None = None, latency_ms: float | None = None
    ) -> None:
        """LLM 응답 로그"""
        self.debug(
            f"   LLM Response: {model}",
            {
                "completion_tokens": completion_tokens,
                "latency_ms": round(latency_ms, 1) if latency_ms else None,
            },
        )

    def workflow_step(self, step: str, status: str, details: dict | None = None) -> None:
        """워크플로우 스텝 로그"""
        emoji = {"start": "▶", "complete": "✓", "skip": "⏭", "error": "✗"}.get(status, "•")
        self.info(f"{emoji} Workflow: {step} [{status}]", details)

    def metric(self, name: str, value: Any, unit: str | None = None) -> None:
        """메트릭 로그"""
        self.debug(f"📊 Metric: {name} = {value}{' ' + unit if unit else ''}")

    # =========================================================================
    # 챗봇 감사 로깅 (Audit Report 요구사항 반영)
    # =========================================================================

    def chat_request(
        self, query: str, session_id: str | None = None, user_id: str | None = None
    ) -> dict[str, Any]:
        """
        챗봇 요청 시작 로깅

        Returns:
            request_context: chat_response에 전달할 컨텍스트
        """
        import time

        context = {
            "request_id": f"chat_{int(time.time() * 1000)}",
            "session_id": session_id,
            "user_id": user_id,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "start_time": time.time(),
            "timestamp": datetime.now().isoformat(),
        }
        self.info("💬 Chat Request", context)
        return context

    def chat_response(
        self,
        request_context: dict[str, Any],
        response: str,
        model: str = "gpt-4.1-mini",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        entities_extracted: dict | None = None,
        intent_detected: str | None = None,
        kg_facts_count: int = 0,
        rag_chunks_count: int = 0,
        inferences_count: int = 0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """
        챗봇 응답 완료 로깅 (상세 메트릭 포함)

        Args:
            request_context: chat_request에서 반환된 컨텍스트
            response: 응답 텍스트
            model: 사용된 LLM 모델
            prompt_tokens: 프롬프트 토큰 수
            completion_tokens: 완료 토큰 수
            total_tokens: 총 토큰 수
            entities_extracted: 추출된 엔티티
            intent_detected: 감지된 의도
            kg_facts_count: KG에서 조회한 사실 수
            rag_chunks_count: RAG 검색 청크 수
            inferences_count: 추론 결과 수
            success: 성공 여부
            error: 에러 메시지
        """
        import time

        start_time = request_context.get("start_time", time.time())
        latency_ms = (time.time() - start_time) * 1000

        audit_record = {
            "request_id": request_context.get("request_id"),
            "session_id": request_context.get("session_id"),
            "timestamp": datetime.now().isoformat(),
            # 성능 메트릭
            "latency_ms": round(latency_ms, 1),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or ((prompt_tokens or 0) + (completion_tokens or 0)),
            # 처리 메트릭
            "intent": intent_detected,
            "entities": entities_extracted,
            "kg_facts": kg_facts_count,
            "rag_chunks": rag_chunks_count,
            "inferences": inferences_count,
            # 결과
            "success": success,
            "error": error,
            "response_length": len(response) if response else 0,
        }

        if success:
            self.info(
                f"✅ Chat Response | {latency_ms:.0f}ms | {total_tokens or 0} tokens | "
                f"KG:{kg_facts_count} RAG:{rag_chunks_count} INF:{inferences_count}",
                audit_record,
            )
        else:
            self.error(f"❌ Chat Failed | {error}", audit_record)

        # 감사 로그 파일에 별도 기록
        self._write_audit_log(audit_record)

    def _write_audit_log(self, record: dict[str, Any]) -> None:
        """감사 로그 파일에 JSON Lines 형식으로 기록"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            audit_file = self.log_dir / f"chatbot_audit_{today}.jsonl"

            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            self.warning(f"Failed to write audit log: {e}")


def log_execution(logger: AgentLogger | None = None):
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
                _logger.error(
                    f"Failed: {func_name}", {"error": str(e), "duration": round(duration, 3)}
                )
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
                _logger.error(
                    f"Failed: {func_name}", {"error": str(e), "duration": round(duration, 3)}
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
