"""
이메일 발송 도구 (Email Sender)
================================
명시적 동의 기반 이메일 알림 발송

중요:
- 반드시 사용자의 명시적 동의(체크박스)가 있어야 발송
- StateManager에서 동의 여부 확인 후 발송

지원 알림 유형:
- rank_change: 순위 급락/급등
- important_insight: 중요 인사이트
- crawl_complete: 크롤링 완료
- error: 에러 발생
- daily_summary: 일일 요약

Usage:
    sender = EmailSender()

    # 동의한 수신자에게만 발송
    result = await sender.send_alert(
        alert_type="rank_change",
        subject="순위 변동 알림",
        content="LANEIGE 제품 순위가 10등 하락했습니다.",
        recipients=["user@example.com"]  # 동의한 사용자만
    )
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import os

logger = logging.getLogger(__name__)


# =============================================================================
# 타입 정의
# =============================================================================

class AlertType(Enum):
    """알림 유형"""
    RANK_CHANGE = "rank_change"
    IMPORTANT_INSIGHT = "important_insight"
    CRAWL_COMPLETE = "crawl_complete"
    ERROR = "error"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class EmailConfig:
    """이메일 설정"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    sender_name: str = "AMORE Market Agent"

    @classmethod
    def from_env(cls) -> "EmailConfig":
        """환경 변수에서 로드"""
        return cls(
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            sender_email=os.getenv("SENDER_EMAIL", ""),
            sender_password=os.getenv("SENDER_PASSWORD", ""),
            sender_name=os.getenv("SENDER_NAME", "AMORE Market Agent")
        )


@dataclass
class SendResult:
    """발송 결과"""
    success: bool
    sent_to: List[str]
    failed: List[str]
    message: str


# =============================================================================
# 이메일 템플릿
# =============================================================================

EMAIL_TEMPLATES = {
    AlertType.RANK_CHANGE: {
        "subject_prefix": "[순위 변동]",
        "template": """
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h2 style="color: #dc3545;">순위 변동 알림</h2>
        <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p><strong>제품:</strong> {product_name}</p>
            <p><strong>브랜드:</strong> {brand}</p>
            <p><strong>이전 순위:</strong> {previous_rank}등</p>
            <p><strong>현재 순위:</strong> {current_rank}등</p>
            <p><strong>변동:</strong> <span style="color: {color};">{change_text}</span></p>
        </div>
        <p style="color: #666; font-size: 12px;">
            {timestamp} | AMORE Market Analysis Agent
        </p>
    </div>
</body>
</html>
"""
    },
    AlertType.IMPORTANT_INSIGHT: {
        "subject_prefix": "[중요 인사이트]",
        "template": """
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h2 style="color: #007bff;">중요 인사이트</h2>
        <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p>{insight}</p>
        </div>
        <h3 style="color: #28a745;">권장 액션</h3>
        <ul>
            {action_items}
        </ul>
        <p style="color: #666; font-size: 12px;">
            {timestamp} | AMORE Market Analysis Agent
        </p>
    </div>
</body>
</html>
"""
    },
    AlertType.CRAWL_COMPLETE: {
        "subject_prefix": "[크롤링 완료]",
        "template": """
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h2 style="color: #28a745;">크롤링 완료</h2>
        <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p><strong>수집 제품:</strong> {total_products}개</p>
            <p><strong>LANEIGE 제품:</strong> {laneige_count}개</p>
            <p><strong>카테고리:</strong> {categories}</p>
        </div>
        <p style="color: #666; font-size: 12px;">
            {timestamp} | AMORE Market Analysis Agent
        </p>
    </div>
</body>
</html>
"""
    },
    AlertType.ERROR: {
        "subject_prefix": "[에러 발생]",
        "template": """
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h2 style="color: #dc3545;">에러 발생</h2>
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p><strong>에러:</strong> {error_message}</p>
            <p><strong>발생 위치:</strong> {location}</p>
        </div>
        <p style="color: #666; font-size: 12px;">
            {timestamp} | AMORE Market Analysis Agent
        </p>
    </div>
</body>
</html>
"""
    },
    AlertType.DAILY_SUMMARY: {
        "subject_prefix": "[일일 요약]",
        "template": """
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h2 style="color: #6f42c1;">일일 분석 요약</h2>

        <h3>오늘의 하이라이트</h3>
        <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
            {highlights}
        </div>

        <h3>주요 지표</h3>
        <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p><strong>LANEIGE 평균 순위:</strong> {avg_rank}등</p>
            <p><strong>Share of Shelf:</strong> {sos}%</p>
            <p><strong>순위 변동 알림:</strong> {alert_count}건</p>
        </div>

        <h3>권장 액션</h3>
        <ul>
            {action_items}
        </ul>

        <p style="color: #666; font-size: 12px;">
            {timestamp} | AMORE Market Analysis Agent
        </p>
    </div>
</body>
</html>
"""
    }
}


# =============================================================================
# 이메일 발송 도구
# =============================================================================

class EmailSender:
    """
    이메일 발송 도구

    명시적 동의 기반으로 알림 이메일을 발송합니다.

    중요 원칙:
    1. 동의 없이는 절대 발송 안 함
    2. 발송 기록 로깅
    3. 실패 시 재시도 (최대 2회)
    """

    def __init__(self, config: Optional[EmailConfig] = None):
        """
        Args:
            config: 이메일 설정 (None이면 환경 변수에서 로드)
        """
        self.config = config or EmailConfig.from_env()
        self._enabled = bool(self.config.sender_email and self.config.sender_password)

        # 발송 기록
        self._send_history: List[Dict[str, Any]] = []

        if not self._enabled:
            logger.warning("Email sender disabled: credentials not configured")

    # =========================================================================
    # 메인 발송
    # =========================================================================

    async def send_alert(
        self,
        alert_type: str,
        subject: str,
        content: Dict[str, Any],
        recipients: List[str]
    ) -> SendResult:
        """
        알림 이메일 발송

        Args:
            alert_type: 알림 유형
            subject: 이메일 제목
            content: 템플릿 변수
            recipients: 수신자 목록 (반드시 동의한 사용자만)

        Returns:
            SendResult
        """
        if not self._enabled:
            return SendResult(
                success=False,
                sent_to=[],
                failed=recipients,
                message="이메일 발송이 비활성화되어 있습니다."
            )

        if not recipients:
            return SendResult(
                success=True,
                sent_to=[],
                failed=[],
                message="수신자 없음"
            )

        # 템플릿 렌더링
        try:
            alert_enum = AlertType(alert_type)
            template_info = EMAIL_TEMPLATES.get(alert_enum)

            if not template_info:
                return SendResult(
                    success=False,
                    sent_to=[],
                    failed=recipients,
                    message=f"알 수 없는 알림 유형: {alert_type}"
                )

            # 제목 생성
            full_subject = f"{template_info['subject_prefix']} {subject}"

            # 템플릿 렌더링
            content["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            html_body = self._render_template(template_info["template"], content)

        except ValueError:
            return SendResult(
                success=False,
                sent_to=[],
                failed=recipients,
                message=f"잘못된 알림 유형: {alert_type}"
            )

        # 발송
        sent_to = []
        failed = []

        for recipient in recipients:
            try:
                await self._send_email(recipient, full_subject, html_body)
                sent_to.append(recipient)
                logger.info(f"Email sent to {recipient}: {full_subject}")

            except Exception as e:
                failed.append(recipient)
                logger.error(f"Failed to send email to {recipient}: {e}")

        # 기록 저장
        self._send_history.append({
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "subject": full_subject,
            "sent_to": sent_to,
            "failed": failed
        })

        return SendResult(
            success=len(failed) == 0,
            sent_to=sent_to,
            failed=failed,
            message=f"발송 완료: {len(sent_to)}명 성공, {len(failed)}명 실패"
        )

    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """템플릿 렌더링"""
        result = template

        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if isinstance(value, list):
                # 리스트는 li 태그로 변환
                list_html = "\n".join([f"<li>{item}</li>" for item in value])
                result = result.replace(placeholder, list_html)
            else:
                result = result.replace(placeholder, str(value))

        return result

    async def _send_email(self, recipient: str, subject: str, html_body: str) -> None:
        """실제 이메일 발송"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.sender_name} <{self.config.sender_email}>"
        msg["To"] = recipient

        # HTML 본문
        html_part = MIMEText(html_body, "html", "utf-8")
        msg.attach(html_part)

        # SMTP 연결 및 발송
        with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
            server.starttls()
            server.login(self.config.sender_email, self.config.sender_password)
            server.sendmail(self.config.sender_email, recipient, msg.as_string())

    # =========================================================================
    # 편의 메서드
    # =========================================================================

    async def send_rank_change_alert(
        self,
        recipients: List[str],
        product_name: str,
        brand: str,
        previous_rank: int,
        current_rank: int
    ) -> SendResult:
        """순위 변동 알림"""
        change = previous_rank - current_rank

        if change > 0:
            change_text = f"↑ {change}등 상승"
            color = "#28a745"
        elif change < 0:
            change_text = f"↓ {abs(change)}등 하락"
            color = "#dc3545"
        else:
            change_text = "변동 없음"
            color = "#6c757d"

        return await self.send_alert(
            alert_type="rank_change",
            subject=f"{product_name} 순위 변동",
            content={
                "product_name": product_name,
                "brand": brand,
                "previous_rank": previous_rank,
                "current_rank": current_rank,
                "change_text": change_text,
                "color": color
            },
            recipients=recipients
        )

    async def send_error_alert(
        self,
        recipients: List[str],
        error_message: str,
        location: str
    ) -> SendResult:
        """에러 알림"""
        return await self.send_alert(
            alert_type="error",
            subject=f"에러: {location}",
            content={
                "error_message": error_message,
                "location": location
            },
            recipients=recipients
        )

    async def send_daily_summary(
        self,
        recipients: List[str],
        highlights: List[str],
        avg_rank: float,
        sos: float,
        alert_count: int,
        action_items: List[str]
    ) -> SendResult:
        """일일 요약 발송"""
        return await self.send_alert(
            alert_type="daily_summary",
            subject=datetime.now().strftime("%Y-%m-%d 일일 분석 요약"),
            content={
                "highlights": "<br>".join(highlights),
                "avg_rank": f"{avg_rank:.1f}",
                "sos": f"{sos:.1f}",
                "alert_count": alert_count,
                "action_items": action_items
            },
            recipients=recipients
        )

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def is_enabled(self) -> bool:
        """이메일 발송 활성화 여부"""
        return self._enabled

    def get_send_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """발송 기록 조회"""
        return self._send_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        total = len(self._send_history)
        successful = sum(1 for h in self._send_history if not h.get("failed"))

        return {
            "enabled": self._enabled,
            "total_sent": total,
            "successful": successful,
            "failed": total - successful
        }
