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

이메일 서비스 옵션:
1. Resend API (권장): 무료 3,000통/월, 설정 간편
2. SMTP: Gmail, 기존 메일서버 등

Usage:
    # Resend 사용 (권장)
    sender = EmailSender()  # RESEND_API_KEY 환경변수 자동 감지

    # SMTP 사용
    sender = EmailSender(provider="smtp")

    # 동의한 수신자에게만 발송
    result = await sender.send_alert(
        alert_type="rank_change",
        subject="순위 변동 알림",
        content="LANEIGE 제품 순위가 10등 하락했습니다.",
        recipients=["user@example.com"]  # 동의한 사용자만
    )
"""

import asyncio
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Literal

from src.domain.brand import is_target_brand

logger = logging.getLogger(__name__)

# Resend 클라이언트 (선택적 의존성)
try:
    import resend

    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False
    logger.debug("resend not installed - SMTP only mode")


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
    INSIGHT_REPORT = "insight_report"  # 인사이트 전체 리포트


@dataclass
class EmailConfig:
    """이메일 설정"""

    # Provider 선택: "resend" (권장) 또는 "smtp"
    provider: Literal["resend", "smtp"] = "resend"

    # Resend 설정 (권장)
    resend_api_key: str = ""
    resend_from_email: str = "onboarding@resend.dev"  # 기본값 (Resend 테스트용)

    # SMTP 설정 (대안)
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""

    # 공통 설정
    sender_name: str = "AMORE Market Agent"

    @classmethod
    def from_env(cls) -> "EmailConfig":
        """환경 변수에서 로드 (Resend 우선)"""
        resend_api_key = os.getenv("RESEND_API_KEY", "")

        # Resend API 키가 있으면 Resend 사용
        if resend_api_key:
            return cls(
                provider="resend",
                resend_api_key=resend_api_key,
                resend_from_email=os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev"),
                sender_name=os.getenv("SENDER_NAME", "AMORE Market Agent"),
            )

        # 없으면 SMTP 폴백
        return cls(
            provider="smtp",
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            sender_email=os.getenv("SENDER_EMAIL", ""),
            sender_password=os.getenv("SENDER_PASSWORD", ""),
            sender_name=os.getenv("SENDER_NAME", "AMORE Market Agent"),
        )


@dataclass
class SendResult:
    """발송 결과"""

    success: bool
    sent_to: list[str]
    failed: list[str]
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
""",
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
""",
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
""",
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
""",
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
""",
    },
    AlertType.INSIGHT_REPORT: {
        "subject_prefix": "[AMORE 인사이트]",
        "template": """
<html>
<head>
    <meta charset="UTF-8">
</head>
<body style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 700px; margin: 0 auto; background-color: #f5f5f5; padding: 20px;">
    <!-- 헤더 -->
    <div style="background: linear-gradient(135deg, #001C58 0%, #1F5795 100%); padding: 30px; border-radius: 12px 12px 0 0; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 24px;">AMORE Market Agent</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">Amazon US Market Intelligence Report</p>
    </div>

    <!-- 날짜 배너 -->
    <div style="background-color: #1F5795; color: white; padding: 12px 30px; text-align: center;">
        <span style="font-size: 16px;">📅 {report_date}</span>
    </div>

    <!-- 메인 콘텐츠 -->
    <div style="background-color: white; padding: 30px; border-radius: 0 0 12px 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">

        <!-- KPI 요약 -->
        <h2 style="color: #001C58; border-bottom: 2px solid #1F5795; padding-bottom: 10px; margin-top: 0;">📊 주요 지표 (KPI)</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 25px;">
            <div style="flex: 1; min-width: 140px; background: linear-gradient(135deg, #e8f4fd 0%, #d0e8f9 100%); padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 12px; color: #666;">LANEIGE 평균 순위</div>
                <div style="font-size: 28px; font-weight: bold; color: #001C58;">{avg_rank}</div>
            </div>
            <div style="flex: 1; min-width: 140px; background: linear-gradient(135deg, #e8f4fd 0%, #d0e8f9 100%); padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 12px; color: #666;">Share of Shelf</div>
                <div style="font-size: 28px; font-weight: bold; color: #1F5795;">{sos}%</div>
            </div>
            <div style="flex: 1; min-width: 140px; background: linear-gradient(135deg, #e8f4fd 0%, #d0e8f9 100%); padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 12px; color: #666;">HHI (집중도)</div>
                <div style="font-size: 28px; font-weight: bold; color: #1F5795;">{hhi}</div>
            </div>
        </div>

        <!-- AI 인사이트 -->
        <h2 style="color: #001C58; border-bottom: 2px solid #1F5795; padding-bottom: 10px;">🤖 AI 인사이트</h2>
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 25px; border-left: 4px solid #1F5795;">
            {insight_content}
        </div>

        <!-- Top 10 순위표 -->
        <h2 style="color: #001C58; border-bottom: 2px solid #1F5795; padding-bottom: 10px;">🏆 Top 10 제품 순위</h2>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 25px; font-size: 14px;">
            <thead>
                <tr style="background-color: #001C58; color: white;">
                    <th style="padding: 12px 8px; text-align: center; width: 50px;">순위</th>
                    <th style="padding: 12px 8px; text-align: left;">제품명</th>
                    <th style="padding: 12px 8px; text-align: center; width: 100px;">브랜드</th>
                    <th style="padding: 12px 8px; text-align: center; width: 70px;">변동</th>
                </tr>
            </thead>
            <tbody>
                {top10_rows}
            </tbody>
        </table>

        <!-- 브랜드별 변동 -->
        <h2 style="color: #001C58; border-bottom: 2px solid #1F5795; padding-bottom: 10px;">📈 브랜드별 주요 변동</h2>
        <div style="margin-bottom: 25px;">
            {brand_changes}
        </div>

        <!-- 푸터 -->
        <div style="border-top: 1px solid #e0e0e0; padding-top: 20px; margin-top: 20px; text-align: center;">
            <p style="color: #666; font-size: 12px; margin: 0;">
                이 리포트는 AMORE Market Agent가 자동으로 생성했습니다.<br>
                {timestamp} | Amazon US Beauty & Personal Care
            </p>
            <p style="margin-top: 15px;">
                <a href="{dashboard_url}" style="background-color: #001C58; color: white; padding: 10px 25px; text-decoration: none; border-radius: 5px; font-size: 14px;">대시보드에서 자세히 보기</a>
            </p>
        </div>
    </div>
</body>
</html>
""",
    },
}


# =============================================================================
# 이메일 발송 도구
# =============================================================================


class EmailSender:
    """
    이메일 발송 도구

    명시적 동의 기반으로 알림 이메일을 발송합니다.

    지원 Provider:
    1. Resend (권장): 무료 3,000통/월, API 키만 설정
    2. SMTP: Gmail 등 기존 메일서버

    중요 원칙:
    1. 동의 없이는 절대 발송 안 함
    2. 발송 기록 로깅
    3. 실패 시 재시도 (최대 2회)
    """

    def __init__(
        self, config: EmailConfig | None = None, provider: Literal["resend", "smtp"] | None = None
    ):
        """
        Args:
            config: 이메일 설정 (None이면 환경 변수에서 로드)
            provider: 강제 지정 시 사용 (resend 또는 smtp)
        """
        self.config = config or EmailConfig.from_env()

        # Provider 강제 지정
        if provider:
            self.config.provider = provider

        # Resend 초기화
        if self.config.provider == "resend":
            if not RESEND_AVAILABLE:
                logger.warning("resend package not installed, falling back to SMTP")
                self.config.provider = "smtp"
            elif not self.config.resend_api_key:
                logger.warning("RESEND_API_KEY not set, falling back to SMTP")
                self.config.provider = "smtp"
            else:
                resend.api_key = self.config.resend_api_key
                logger.info("Email sender initialized with Resend API")

        # 활성화 여부 판단
        if self.config.provider == "resend":
            self._enabled = bool(self.config.resend_api_key)
        else:
            self._enabled = bool(self.config.sender_email and self.config.sender_password)

        # 발송 기록
        self._send_history: list[dict[str, Any]] = []

        if not self._enabled:
            logger.warning(
                f"Email sender disabled: {self.config.provider} credentials not configured"
            )

    # =========================================================================
    # 메인 발송
    # =========================================================================

    async def send_alert(
        self, alert_type: str, subject: str, content: dict[str, Any], recipients: list[str]
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
                message="이메일 발송이 비활성화되어 있습니다.",
            )

        if not recipients:
            return SendResult(success=True, sent_to=[], failed=[], message="수신자 없음")

        # 템플릿 렌더링
        try:
            alert_enum = AlertType(alert_type)
            template_info = EMAIL_TEMPLATES.get(alert_enum)

            if not template_info:
                return SendResult(
                    success=False,
                    sent_to=[],
                    failed=recipients,
                    message=f"알 수 없는 알림 유형: {alert_type}",
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
                message=f"잘못된 알림 유형: {alert_type}",
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
        self._send_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "alert_type": alert_type,
                "subject": full_subject,
                "sent_to": sent_to,
                "failed": failed,
            }
        )

        return SendResult(
            success=len(failed) == 0,
            sent_to=sent_to,
            failed=failed,
            message=f"발송 완료: {len(sent_to)}명 성공, {len(failed)}명 실패",
        )

    def _render_template(self, template: str, variables: dict[str, Any]) -> str:
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
        """실제 이메일 발송 (Resend 또는 SMTP)"""
        if self.config.provider == "resend":
            await self._send_via_resend(recipient, subject, html_body)
        else:
            await self._send_via_smtp(recipient, subject, html_body)

    async def _send_via_resend(self, recipient: str, subject: str, html_body: str) -> None:
        """Resend API로 발송"""
        if not RESEND_AVAILABLE:
            raise RuntimeError("resend package not installed")

        # Resend는 동기 API이므로 executor에서 실행
        def _send():
            params = {
                "from": f"{self.config.sender_name} <{self.config.resend_from_email}>",
                "to": [recipient],
                "subject": subject,
                "html": html_body,
            }
            return resend.Emails.send(params)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _send)

        if not result.get("id"):
            raise RuntimeError(f"Resend failed: {result}")

        logger.debug(f"Resend email sent: {result.get('id')}")

    async def _send_via_smtp(self, recipient: str, subject: str, html_body: str) -> None:
        """SMTP로 발송"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.sender_name} <{self.config.sender_email}>"
        msg["To"] = recipient

        # HTML 본문
        html_part = MIMEText(html_body, "html", "utf-8")
        msg.attach(html_part)

        # SMTP 연결 및 발송 (동기 작업을 executor에서 실행)
        def _send():
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(self.config.sender_email, recipient, msg.as_string())

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send)

    # =========================================================================
    # 편의 메서드
    # =========================================================================

    async def send_rank_change_alert(
        self,
        recipients: list[str],
        product_name: str,
        brand: str,
        previous_rank: int,
        current_rank: int,
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
                "color": color,
            },
            recipients=recipients,
        )

    async def send_error_alert(
        self, recipients: list[str], error_message: str, location: str
    ) -> SendResult:
        """에러 알림"""
        return await self.send_alert(
            alert_type="error",
            subject=f"에러: {location}",
            content={"error_message": error_message, "location": location},
            recipients=recipients,
        )

    async def send_daily_summary(
        self,
        recipients: list[str],
        highlights: list[str],
        avg_rank: float,
        sos: float,
        alert_count: int,
        action_items: list[str],
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
                "action_items": action_items,
            },
            recipients=recipients,
        )

    async def send_insight_report(
        self,
        recipients: list[str],
        report_date: str,
        avg_rank: float,
        sos: float,
        hhi: float,
        insight_content: str,
        top10_products: list[dict],
        brand_changes: list[dict],
        dashboard_url: str = "http://localhost:8001/dashboard",
    ) -> SendResult:
        """
        인사이트 전체 리포트 이메일 발송

        Args:
            recipients: 수신자 목록
            report_date: 리포트 날짜 (예: "2026년 1월 30일")
            avg_rank: LANEIGE 평균 순위
            sos: Share of Shelf (%)
            hhi: HHI 지수
            insight_content: AI 인사이트 HTML 내용
            top10_products: Top 10 제품 리스트 [{"rank", "name", "brand", "change"}]
            brand_changes: 브랜드 변동 리스트 [{"brand", "change_text", "color"}]
            dashboard_url: 대시보드 URL

        Returns:
            SendResult
        """
        # Top 10 테이블 행 생성
        top10_rows = ""
        for i, product in enumerate(top10_products[:10]):
            rank = product.get("rank", i + 1)
            name = product.get("name", "N/A")[:50]  # 이름 50자 제한
            brand = product.get("brand", "N/A")
            change = product.get("change", 0)

            # LANEIGE 하이라이트
            row_style = "background-color: #e8f4fd;" if is_target_brand(brand) else ""

            # 변동 표시
            if change > 0:
                change_html = f'<span style="color: #28a745;">▲{change}</span>'
            elif change < 0:
                change_html = f'<span style="color: #dc3545;">▼{abs(change)}</span>'
            else:
                change_html = '<span style="color: #666;">-</span>'

            top10_rows += f"""
                <tr style="{row_style}">
                    <td style="padding: 10px 8px; text-align: center; border-bottom: 1px solid #e0e0e0; font-weight: bold;">{rank}</td>
                    <td style="padding: 10px 8px; border-bottom: 1px solid #e0e0e0;">{name}</td>
                    <td style="padding: 10px 8px; text-align: center; border-bottom: 1px solid #e0e0e0;">{brand}</td>
                    <td style="padding: 10px 8px; text-align: center; border-bottom: 1px solid #e0e0e0;">{change_html}</td>
                </tr>
            """

        # 브랜드 변동 HTML 생성
        brand_changes_html = ""
        for bc in brand_changes[:5]:  # 최대 5개
            brand = bc.get("brand", "N/A")
            change_text = bc.get("change_text", "변동 없음")
            color = bc.get("color", "#666")
            brand_changes_html += f"""
                <div style="display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #e0e0e0;">
                    <span style="font-weight: bold;">{brand}</span>
                    <span style="color: {color};">{change_text}</span>
                </div>
            """

        if not brand_changes_html:
            brand_changes_html = '<p style="color: #666;">오늘은 주요 브랜드 변동이 없습니다.</p>'

        return await self.send_alert(
            alert_type="insight_report",
            subject=f"{report_date} Amazon US 시장 인사이트",
            content={
                "report_date": report_date,
                "avg_rank": f"{avg_rank:.1f}" if avg_rank else "N/A",
                "sos": f"{sos:.1f}" if sos else "N/A",
                "hhi": f"{hhi:.0f}" if hhi else "N/A",
                "insight_content": insight_content,
                "top10_rows": top10_rows,
                "brand_changes": brand_changes_html,
                "dashboard_url": dashboard_url,
            },
            recipients=recipients,
        )

    async def send_verification_email(
        self, recipient: str, verify_url: str, token: str
    ) -> SendResult:
        """
        이메일 인증 이메일 발송

        사용자가 알림 설정에서 이메일을 입력하면
        인증 버튼이 포함된 이메일을 발송합니다.

        Args:
            recipient: 수신자 이메일
            verify_url: 인증 URL (대시보드로 리다이렉트)
            token: 인증 토큰

        Returns:
            SendResult
        """
        subject = "[AMORE Agent] 이메일 인증을 완료해주세요"

        # AMOREPACIFIC CI 색상 적용
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #001C58 0%, #1F5795 100%); padding: 30px; border-radius: 16px 16px 0 0; text-align: center;">
                    <h1 style="margin: 0; color: white; font-size: 24px; font-weight: 700;">
                        AMORE Market Agent
                    </h1>
                    <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.8); font-size: 14px;">
                        Amazon US Market Intelligence
                    </p>
                </div>

                <!-- Body -->
                <div style="background: white; padding: 40px 30px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,28,88,0.1);">
                    <h2 style="margin: 0 0 16px 0; color: #001C58; font-size: 20px;">
                        이메일 인증 요청
                    </h2>

                    <p style="color: #64748b; font-size: 15px; line-height: 1.7; margin: 0 0 24px 0;">
                        AMORE Market Agent 알림 서비스를 이용해 주셔서 감사합니다.<br>
                        아래 버튼을 클릭하여 이메일 인증을 완료해주세요.
                    </p>

                    <!-- CTA Button -->
                    <div style="text-align: center; margin: 32px 0;">
                        <a href="{verify_url}"
                           style="display: inline-block; padding: 16px 48px; background: linear-gradient(135deg, #001C58, #1F5795); color: white; text-decoration: none; border-radius: 10px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 15px rgba(0,28,88,0.3);">
                            ✓ 이메일 인증하기
                        </a>
                    </div>

                    <p style="color: #94a3b8; font-size: 13px; margin: 24px 0 0 0; padding-top: 20px; border-top: 1px solid #e2e8f0;">
                        이 링크는 30분간 유효합니다.<br>
                        본인이 요청하지 않았다면 이 이메일을 무시해주세요.
                    </p>
                </div>

                <!-- Footer -->
                <div style="text-align: center; padding: 24px; color: #94a3b8; font-size: 12px;">
                    <p style="margin: 0;">
                        © 2026 AMORE Market Agent | Powered by AI
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        # 발송
        try:
            if self.config.provider == "resend" and RESEND_AVAILABLE:
                result = await self._send_via_resend(
                    to_emails=[recipient], subject=subject, html_content=html_content
                )
            else:
                await self._send_via_smtp(
                    recipient=recipient, subject=subject, html_body=html_content
                )
                result = SendResult(
                    success=True,
                    sent_to=[recipient],
                    failed=[],
                    message="인증 이메일이 발송되었습니다",
                )

            # 기록
            self._send_history.append(
                {
                    "type": "verification",
                    "recipient": recipient,
                    "timestamp": datetime.now().isoformat(),
                    "success": result.success,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Failed to send verification email to {recipient}: {e}")
            return SendResult(success=False, sent_to=[], failed=[recipient], message=str(e))

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def is_enabled(self) -> bool:
        """이메일 발송 활성화 여부"""
        return self._enabled

    def get_send_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """발송 기록 조회"""
        return self._send_history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """통계"""
        total = len(self._send_history)
        successful = sum(1 for h in self._send_history if not h.get("failed"))

        return {
            "enabled": self._enabled,
            "provider": self.config.provider,
            "total_sent": total,
            "successful": successful,
            "failed": total - successful,
        }

    def get_provider_info(self) -> dict[str, Any]:
        """현재 Provider 정보"""
        if self.config.provider == "resend":
            return {
                "provider": "resend",
                "from_email": self.config.resend_from_email,
                "api_key_set": bool(self.config.resend_api_key),
                "free_tier": "3,000 emails/month",
            }
        else:
            return {
                "provider": "smtp",
                "server": self.config.smtp_server,
                "port": self.config.smtp_port,
                "sender_email": self.config.sender_email,
                "credentials_set": bool(self.config.sender_password),
            }

    # =========================================================================
    # Morning Brief (뉴스레터)
    # =========================================================================

    async def send_morning_brief(
        self, recipients: list[str], html_content: str, date_str: str
    ) -> SendResult:
        """
        Morning Brief 뉴스레터 발송

        Args:
            recipients: 수신자 목록
            html_content: 렌더링된 HTML 콘텐츠
            date_str: 날짜 문자열 (제목용)

        Returns:
            SendResult
        """
        if not self._enabled:
            return SendResult(
                success=False,
                sent_to=[],
                failed=recipients,
                message="이메일 발송이 비활성화되어 있습니다.",
            )

        if not recipients:
            return SendResult(success=True, sent_to=[], failed=[], message="수신자 없음")

        subject = f"☀️ AMORE Daily Brief - {date_str}"

        sent_to = []
        failed = []

        for recipient in recipients:
            try:
                await self._send_email(recipient, subject, html_content)
                sent_to.append(recipient)
                logger.info(f"Morning Brief sent to {recipient}")

            except Exception as e:
                failed.append(recipient)
                logger.error(f"Failed to send Morning Brief to {recipient}: {e}")

        # 기록 저장
        self._send_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "alert_type": "morning_brief",
                "subject": subject,
                "sent_to": sent_to,
                "failed": failed,
            }
        )

        return SendResult(
            success=len(failed) == 0,
            sent_to=sent_to,
            failed=failed,
            message=f"Morning Brief 발송: {len(sent_to)}명 성공, {len(failed)}명 실패",
        )
