"""
API Pydantic Models
===================
dashboard_api.py에서 사용하는 모든 Pydantic 모델 정의
"""

from typing import Any

from pydantic import BaseModel, Field

# ============= Chat Models =============


class ChatRequest(BaseModel):
    """챗봇 요청"""

    message: str = Field(..., max_length=10000, description="최대 10,000자")
    session_id: str | None = Field(default="default", max_length=100)
    context: dict | None = None


class ChatResponse(BaseModel):
    """챗봇 응답"""

    response: str
    query_type: str
    confidence: float
    sources: list[str]
    suggestions: list[str]
    entities: dict[str, Any]


class BrainChatRequest(BaseModel):
    """Brain 챗봇 요청"""

    message: str
    session_id: str | None = "default"
    skip_cache: bool = False


class BrainChatResponse(BaseModel):
    """Brain 챗봇 응답"""

    text: str
    confidence: float
    sources: list[str]
    reasoning: str | None = None
    tools_used: list[str]
    processing_time_ms: float
    from_cache: bool
    brain_mode: str
    suggestions: list[str] = []
    query_type: str = "unknown"


# ============= Export Models =============


class ExportRequest(BaseModel):
    """내보내기 요청"""

    start_date: str | None = None
    end_date: str | None = None
    include_strategy: bool = True


class AnalystReportRequest(BaseModel):
    """애널리스트 리포트 요청"""

    start_date: str  # Required: YYYY-MM-DD
    end_date: str  # Required: YYYY-MM-DD
    include_charts: bool = True
    include_external_signals: bool = True


# ============= Alert Models =============


class AlertSettingsRequest(BaseModel):
    """알림 설정 요청"""

    email: str
    consent: bool
    alert_types: list[str] = []


class AlertSettingsResponse(BaseModel):
    """알림 설정 응답"""

    email: str
    consent: bool
    alert_types: list[str]
    consent_date: str | None = None


class SubscribeRequest(BaseModel):
    """구독 시작 요청"""

    email: str
    alert_types: list[str] = []


class UpdateAlertSettingsRequest(BaseModel):
    """알림 설정 수정 요청"""

    email: str
    alert_types: list[str]


class UnsubscribeRequest(BaseModel):
    """구독 해지 요청"""

    email: str


class AlertSendRequest(BaseModel):
    """알림 발송 요청"""

    alert_ids: list[int] | None = None  # 발송할 알림 ID (없으면 미발송 전체)


# ============= Deals Models =============


class DealsRequest(BaseModel):
    """Deals 크롤링 요청"""

    max_items: int = 50
    beauty_only: bool = True


class DealsResponse(BaseModel):
    """Deals 응답"""

    success: bool
    count: int
    lightning_count: int
    competitor_count: int
    snapshot_datetime: str
    deals: list[dict[str, Any]]
    competitor_deals: list[dict[str, Any]]
    error: str | None = None


# ============= Market Intelligence Models =============


class MarketIntelligenceStatusResponse(BaseModel):
    """Market Intelligence 상태 응답"""

    initialized: bool
    layers_collected: list[int]
    last_collection: str | None = None
    stats: dict[str, Any]


class LayerDataResponse(BaseModel):
    """레이어 데이터 응답"""

    layer: int
    layer_name: str
    collected_at: str
    data: dict[str, Any]
    sources: list[dict[str, Any]]
