"""
Telegram Admin Bot
==================
관리자 전용 Telegram Bot - 로그 조회 및 시스템 모니터링

보안:
- TELEGRAM_ADMIN_CHAT_ID에 등록된 사용자만 명령 실행 가능
- 웹 엔드포인트 노출 없음

환경변수:
- TELEGRAM_BOT_TOKEN: BotFather에서 발급받은 토큰
- TELEGRAM_ADMIN_CHAT_ID: 관리자 Chat ID (쉼표로 복수 가능)

명령어:
- /start - 봇 시작 및 Chat ID 확인
- /help - 명령어 목록
- /logs [type] [lines] - 로그 조회 (crawler, insight, chatbot, error)
- /status - 시스템 상태
- /jobs - 백그라운드 작업 현황
- /crawl - 크롤링 상태
- /kg - Knowledge Graph 상태

Usage:
    # FastAPI에 webhook 연동
    from src.tools.telegram_bot import TelegramAdminBot, telegram_router
    app.include_router(telegram_router)

    # 또는 직접 메시지 발송
    bot = TelegramAdminBot()
    await bot.send_alert("에러 발생!", level="error")
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

# 한국 시간대
KST = timezone(timedelta(hours=9))

# Router for webhook
telegram_router = APIRouter(prefix="/api/telegram", tags=["telegram"])


class TelegramAdminBot:
    """Telegram 관리자 봇"""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.admin_chat_ids = self._parse_admin_ids()
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.logs_dir = Path(os.getenv("LOGS_DIR", "logs"))

        # Railway 환경 감지
        self.is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
        if self.is_railway:
            self.logs_dir = Path("/data/logs")

    def _parse_admin_ids(self) -> set[int]:
        """관리자 Chat ID 파싱"""
        raw = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        if not raw:
            return set()
        try:
            return {int(cid.strip()) for cid in raw.split(",") if cid.strip()}
        except ValueError:
            logger.error("Invalid TELEGRAM_ADMIN_CHAT_ID format")
            return set()

    def is_enabled(self) -> bool:
        """봇 활성화 여부"""
        return bool(self.token and self.admin_chat_ids)

    def is_admin(self, chat_id: int) -> bool:
        """관리자 여부 확인"""
        return chat_id in self.admin_chat_ids

    # =========================================================================
    # Telegram API
    # =========================================================================

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> dict[str, Any] | None:
        """메시지 전송"""
        if not self.token:
            logger.warning("Telegram bot token not configured")
            return None

        # Telegram 메시지 길이 제한 (4096자)
        if len(text) > 4000:
            text = text[:4000] + "\n\n... (truncated)"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_notification": disable_notification,
                    },
                )
                return response.json()
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return None

    async def send_to_admins(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> None:
        """모든 관리자에게 메시지 전송"""
        for chat_id in self.admin_chat_ids:
            await self.send_message(chat_id, text, parse_mode, disable_notification)

    async def send_alert(
        self,
        message: str,
        level: str = "info",
        details: str | None = None,
    ) -> None:
        """알림 전송 (에러, 경고 등)"""
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
            "success": "✅",
            "critical": "🔥",
        }
        emoji = emoji_map.get(level, "📢")

        now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        text = f"{emoji} <b>{level.upper()}</b>\n\n{message}\n\n<i>{now} KST</i>"

        if details:
            # 코드 블록으로 상세 정보
            text += f"\n\n<pre>{details[:1000]}</pre>"

        await self.send_to_admins(text)

    # =========================================================================
    # 명령어 핸들러
    # =========================================================================

    async def handle_command(self, chat_id: int, text: str) -> str:
        """명령어 처리"""
        # 관리자 확인
        if not self.is_admin(chat_id):
            return (
                "⛔ 권한이 없습니다.\n\n"
                f"Your Chat ID: <code>{chat_id}</code>\n\n"
                "관리자에게 이 ID를 전달하여 등록을 요청하세요."
            )

        # 명령어 파싱
        parts = text.strip().split()
        command = parts[0].lower().replace("/", "").split("@")[0]  # @botname 제거
        args = parts[1:] if len(parts) > 1 else []

        handlers = {
            "start": self._cmd_start,
            "help": self._cmd_help,
            "logs": self._cmd_logs,
            "errors": self._cmd_errors,
            "status": self._cmd_status,
            "jobs": self._cmd_jobs,
            "crawl": self._cmd_crawl,
            "kg": self._cmd_kg,
            "db": self._cmd_db,
        }

        handler = handlers.get(command)
        if handler:
            return await handler(args)
        else:
            return f"❓ 알 수 없는 명령어: {command}\n\n/help 로 명령어 목록을 확인하세요."

    async def _cmd_start(self, args: list[str]) -> str:
        """시작 명령"""
        return (
            "👋 <b>AMORE Admin Bot</b>\n\n"
            "관리자 전용 모니터링 봇입니다.\n\n"
            "/help 로 명령어 목록을 확인하세요."
        )

    async def _cmd_help(self, args: list[str]) -> str:
        """도움말"""
        return """📖 <b>명령어 목록</b>

<b>로그 조회</b>
/logs [type] [lines] - 로그 조회
  • type: crawler, insight, chatbot, period, error
  • lines: 줄 수 (기본 30)
  예: /logs crawler 50

/errors [lines] - 에러 로그만 조회

<b>시스템 상태</b>
/status - 시스템 리소스 현황
/jobs - 백그라운드 작업 목록
/crawl - 크롤링 상태
/kg - Knowledge Graph 상태
/db - 데이터베이스 통계

<b>기타</b>
/help - 이 도움말
/start - 봇 시작"""

    async def _cmd_logs(self, args: list[str]) -> str:
        """로그 조회"""
        log_type = args[0] if args else "crawler"
        lines = int(args[1]) if len(args) > 1 else 30
        lines = min(lines, 100)  # 최대 100줄

        # 로그 파일 패턴 매핑
        patterns = {
            "crawler": "crawler_*.log",
            "insight": "hybrid_insight_*.log",
            "chatbot": "hybrid_chatbot_*.log",
            "period": "src.agents.period_insight_agent_*.log",
            "audit": "chatbot_audit_*.log",
        }

        pattern = patterns.get(log_type, f"*{log_type}*.log")

        # 최신 로그 파일 찾기
        log_files = sorted(self.logs_dir.glob(pattern), reverse=True)

        if not log_files:
            return f"📂 로그 파일을 찾을 수 없습니다: {pattern}"

        latest_log = log_files[0]

        try:
            # 마지막 N줄 읽기
            with open(latest_log, encoding="utf-8", errors="ignore") as f:
                all_lines = f.readlines()
                recent = all_lines[-lines:] if len(all_lines) > lines else all_lines

            content = "".join(recent)

            # HTML 이스케이프
            content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            return (
                f"📋 <b>{latest_log.name}</b> (최근 {len(recent)}줄)\n\n"
                f"<pre>{content[:3500]}</pre>"
            )
        except Exception as e:
            return f"❌ 로그 읽기 실패: {e}"

    async def _cmd_errors(self, args: list[str]) -> str:
        """에러 로그만 필터링"""
        lines = int(args[0]) if args else 20
        lines = min(lines, 50)

        errors = []

        # 모든 로그 파일에서 에러 검색
        for log_file in sorted(self.logs_dir.glob("*.log"), reverse=True)[:5]:
            try:
                with open(log_file, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if any(
                            kw in line.upper()
                            for kw in ["ERROR", "EXCEPTION", "TRACEBACK", "FAILED"]
                        ):
                            errors.append(f"[{log_file.stem}] {line.strip()}")
            except Exception:
                continue

        if not errors:
            return "✅ 최근 에러가 없습니다!"

        recent_errors = errors[-lines:]
        content = "\n".join(recent_errors)
        content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return f"🚨 <b>최근 에러</b> ({len(recent_errors)}건)\n\n<pre>{content[:3500]}</pre>"

    async def _cmd_status(self, args: list[str]) -> str:
        """시스템 상태"""
        status_lines = []

        # 메모리
        try:
            import psutil

            mem = psutil.virtual_memory()
            status_lines.append(
                f"💾 메모리: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)"
            )

            # CPU
            cpu = psutil.cpu_percent(interval=1)
            status_lines.append(f"🖥️ CPU: {cpu}%")

            # 디스크
            disk = psutil.disk_usage("/")
            status_lines.append(
                f"💿 디스크: {disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB ({disk.percent}%)"
            )
        except ImportError:
            status_lines.append("⚠️ psutil 미설치 - 리소스 모니터링 불가")

        # 현재 시간
        now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        status_lines.append(f"\n🕐 현재 시간: {now} KST")

        # Railway 환경
        if self.is_railway:
            status_lines.append("🚂 환경: Railway Production")
        else:
            status_lines.append("🏠 환경: Local Development")

        return "📊 <b>시스템 상태</b>\n\n" + "\n".join(status_lines)

    async def _cmd_jobs(self, args: list[str]) -> str:
        """백그라운드 작업 현황"""
        try:
            from src.tools.job_queue import JobQueue

            queue = JobQueue()
            jobs = queue.get_all_jobs()

            if not jobs:
                return "📭 현재 진행 중인 작업이 없습니다."

            lines = ["📋 <b>백그라운드 작업</b>\n"]
            for job in jobs[:10]:
                status_emoji = {
                    "pending": "⏳",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }.get(job.get("status", ""), "❓")

                lines.append(
                    f"{status_emoji} {job.get('job_type', 'unknown')} - {job.get('progress', 0)}%"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"❌ 작업 조회 실패: {e}"

    async def _cmd_crawl(self, args: list[str]) -> str:
        """크롤링 상태"""
        try:
            # 최근 크롤링 로그에서 상태 추출
            log_files = sorted(self.logs_dir.glob("crawler_*.log"), reverse=True)
            if not log_files:
                return "📂 크롤링 로그가 없습니다."

            with open(log_files[0], encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 통계 추출
            success_count = content.count("Successfully crawled")
            error_count = content.count("ERROR") + content.count("Failed")
            blocked_count = content.count("blocked") + content.count("WAF")

            # 최근 크롤링 시간
            dates = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content)
            last_crawl = dates[-1] if dates else "알 수 없음"

            return f"""🕷️ <b>크롤링 상태</b>

📅 마지막 크롤링: {last_crawl}
✅ 성공: {success_count}건
❌ 에러: {error_count}건
🚫 차단: {blocked_count}건

📁 로그 파일: {log_files[0].name}"""

        except Exception as e:
            return f"❌ 크롤링 상태 조회 실패: {e}"

    async def _cmd_kg(self, args: list[str]) -> str:
        """Knowledge Graph 상태"""
        try:
            from src.ontology.knowledge_graph import KnowledgeGraph

            kg = KnowledgeGraph()
            stats = kg.get_stats() if hasattr(kg, "get_stats") else {}

            return f"""🧠 <b>Knowledge Graph</b>

📊 트리플 수: {stats.get('triple_count', 'N/A')}
🏷️ 엔티티 수: {stats.get('entity_count', 'N/A')}
🔗 관계 수: {stats.get('relation_count', 'N/A')}
💾 파일 크기: {stats.get('file_size', 'N/A')}"""

        except Exception as e:
            return f"❌ KG 상태 조회 실패: {e}"

    async def _cmd_db(self, args: list[str]) -> str:
        """데이터베이스 통계"""
        try:
            from src.tools.sqlite_storage import SQLiteStorage

            storage = SQLiteStorage()

            # 테이블별 레코드 수
            tables = ["rankings", "products", "brands", "metrics"]
            stats = []

            for table in tables:
                try:
                    count = storage.execute_query(f"SELECT COUNT(*) FROM {table}")[0][0]
                    stats.append(f"• {table}: {count:,}건")
                except Exception:
                    pass

            return "🗄️ <b>데이터베이스 통계</b>\n\n" + "\n".join(stats)

        except Exception as e:
            return f"❌ DB 통계 조회 실패: {e}"


# =========================================================================
# Webhook Endpoint
# =========================================================================

# 전역 봇 인스턴스
_bot: TelegramAdminBot | None = None


def get_bot() -> TelegramAdminBot:
    """봇 인스턴스 반환"""
    global _bot
    if _bot is None:
        _bot = TelegramAdminBot()
    return _bot


@telegram_router.post("/webhook")
async def telegram_webhook(request: Request):
    """Telegram Webhook 엔드포인트"""
    logger.info("=== Telegram webhook called ===")
    bot = get_bot()
    logger.info(f"Bot token exists: {bool(bot.token)}, admin_ids: {bot.admin_chat_ids}")

    if not bot.is_enabled():
        logger.warning("Bot not enabled - returning early")
        return {"ok": False, "error": "Bot not configured"}

    try:
        data = await request.json()
        logger.info(f"Telegram webhook received: {data}")

        message = data.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")

        logger.info(f"Parsed: chat_id={chat_id}, text={text!r}")
        logger.info(f"Bot enabled: {bot.is_enabled()}, admin_ids: {bot.admin_chat_ids}")

        if chat_id and text and text.startswith("/"):
            logger.info(f"Processing command: {text}")
            response = await bot.handle_command(chat_id, text)
            logger.info(f"Command response: {response[:100]}...")
            result = await bot.send_message(chat_id, response)
            logger.info(f"Send result: {result}")
        else:
            logger.info(f"Skipping: no command (text={text!r})")

        return {"ok": True}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"ok": False, "error": str(e)}


# =========================================================================
# 에러 알림 헬퍼
# =========================================================================


async def notify_error(
    error: Exception,
    context: str = "",
    include_traceback: bool = True,
) -> None:
    """에러 발생 시 관리자에게 알림"""
    import traceback

    bot = get_bot()
    if not bot.is_enabled():
        return

    message = f"<b>{context}</b>\n\n" if context else ""
    message += f"<code>{type(error).__name__}: {str(error)[:200]}</code>"

    details = None
    if include_traceback:
        details = traceback.format_exc()

    await bot.send_alert(message, level="error", details=details)


async def notify_crawl_complete(
    category: str,
    product_count: int,
    duration_sec: float,
) -> None:
    """크롤링 완료 알림"""
    bot = get_bot()
    if not bot.is_enabled():
        return

    await bot.send_alert(
        f"🕷️ <b>크롤링 완료</b>\n\n"
        f"카테고리: {category}\n"
        f"제품 수: {product_count}개\n"
        f"소요 시간: {duration_sec:.1f}초",
        level="success",
    )
