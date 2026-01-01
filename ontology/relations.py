"""
Ontology Relations
엔티티 간 관계 타입 정의 및 트리플(Triple) 구조

Triple 구조: (Subject, Predicate, Object)
예: (LANEIGE, hasProduct, "Lip Sleeping Mask")
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


class RelationType(str, Enum):
    """
    온톨로지 관계 유형

    카테고리:
    1. 엔티티 관계 - 브랜드, 제품, 카테고리 간 구조적 관계
    2. 지표 관계 - 메트릭과 인사이트 간 의미적 관계
    3. 시간 관계 - 스냅샷과 트렌드 간 시계열 관계
    4. 경쟁 관계 - 브랜드/제품 간 경쟁 구도
    """

    # =========================================================================
    # 1. 엔티티 관계 (Entity Relations)
    # =========================================================================

    # Brand → Product: 브랜드가 제품을 보유
    HAS_PRODUCT = "hasProduct"

    # Product → Category: 제품이 카테고리에 속함
    BELONGS_TO_CATEGORY = "belongsToCategory"

    # Product → Brand: 제품의 브랜드 (역관계)
    OWNED_BY = "ownedBy"

    # Category → Category: 상위/하위 카테고리
    HAS_SUBCATEGORY = "hasSubcategory"
    PARENT_CATEGORY = "parentCategory"

    # =========================================================================
    # 2. 순위/성과 관계 (Ranking Relations)
    # =========================================================================

    # Product → Snapshot: 제품이 특정 시점에 순위 보유
    RANKED_IN = "rankedIn"

    # Product → Rank: 현재 순위 보유
    HAS_RANK = "hasRank"

    # Product → Product: 순위 상 앞/뒤 제품
    RANKS_ABOVE = "ranksAbove"
    RANKS_BELOW = "ranksBelow"

    # =========================================================================
    # 3. 경쟁 관계 (Competitive Relations)
    # =========================================================================

    # Brand ↔ Brand: 같은 카테고리 내 경쟁
    COMPETES_WITH = "competesWith"

    # Product ↔ Product: 유사 제품 경쟁
    COMPETES_WITH_PRODUCT = "competesWithProduct"

    # Brand → Brand: 직접 경쟁자 (Top 5 내)
    DIRECT_COMPETITOR = "directCompetitor"

    # Brand → Brand: 간접 경쟁자 (Top 20 내)
    INDIRECT_COMPETITOR = "indirectCompetitor"

    # =========================================================================
    # 4. 지표 관계 (Metric Relations)
    # =========================================================================

    # Metric → Insight: 지표가 인사이트를 나타냄
    INDICATES = "indicates"

    # Metric ↔ Metric: 지표 간 상관관계
    CORRELATES_WITH = "correlatesWith"

    # Metric → Metric: 지표가 다른 지표에 영향
    INFLUENCES = "influences"

    # Insight → Action: 인사이트가 액션을 요구
    REQUIRES_ACTION = "requiresAction"

    # =========================================================================
    # 5. 시간 관계 (Temporal Relations)
    # =========================================================================

    # Snapshot → Snapshot: 시간 순서
    FOLLOWS = "follows"
    PRECEDES = "precedes"

    # Entity → Trend: 트렌드 보유
    HAS_TREND = "hasTrend"

    # Product → History: 히스토리 보유
    HAS_HISTORY = "hasHistory"

    # =========================================================================
    # 6. 상태 관계 (State Relations)
    # =========================================================================

    # Entity → State: 현재 상태
    HAS_STATE = "hasState"

    # Product → Alert: 알림 발생
    HAS_ALERT = "hasAlert"

    # Brand → Position: 시장 포지션
    HAS_POSITION = "hasPosition"


class InsightType(str, Enum):
    """추론된 인사이트 유형"""

    # 시장 포지션 관련
    MARKET_POSITION = "market_position"
    MARKET_DOMINANCE = "market_dominance"
    MARKET_SHARE = "market_share"

    # 경쟁 관련
    COMPETITIVE_THREAT = "competitive_threat"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"
    COMPETITOR_MOVEMENT = "competitor_movement"

    # 성장/기회 관련
    GROWTH_OPPORTUNITY = "growth_opportunity"
    GROWTH_MOMENTUM = "growth_momentum"
    ENTRY_OPPORTUNITY = "entry_opportunity"

    # 위험/경고 관련
    RISK_ALERT = "risk_alert"
    RANK_SHOCK = "rank_shock"
    RATING_DECLINE = "rating_decline"

    # 가격 관련
    PRICE_POSITION = "price_position"
    PRICE_QUALITY_GAP = "price_quality_gap"

    # 안정성 관련
    STABILITY = "stability"
    VOLATILITY = "volatility"


class MarketPosition(str, Enum):
    """시장 포지션 유형"""

    # 지배적 포지션
    DOMINANT = "dominant"                      # 시장 지배자
    DOMINANT_IN_FRAGMENTED = "dominant_in_fragmented"  # 분산 시장 지배자

    # 도전자 포지션
    CHALLENGER = "challenger"                  # 도전자
    STRONG_CHALLENGER = "strong_challenger"    # 강력한 도전자

    # 추종자 포지션
    FOLLOWER = "follower"                      # 추종자
    NICHE = "niche"                           # 니치 플레이어

    # 신규/이탈
    NEW_ENTRANT = "new_entrant"               # 신규 진입자
    DECLINING = "declining"                    # 하락세
    EXITING = "exiting"                       # 이탈 중


@dataclass
class Relation:
    """
    온톨로지 관계 (Triple)

    구조: (Subject, Predicate, Object)

    Attributes:
        subject: 주체 엔티티 (예: "LANEIGE", "B08XYZ1234")
        predicate: 관계 유형 (예: RelationType.HAS_PRODUCT)
        object: 객체 엔티티 (예: "Lip Sleeping Mask")
        properties: 관계의 추가 속성 (메타데이터)
        confidence: 관계의 신뢰도 (0.0 ~ 1.0)
        source: 관계 출처 (crawl, inference, manual)
        created_at: 생성 시간
        valid_from: 유효 시작 시간 (시간적 관계용)
        valid_to: 유효 종료 시간 (시간적 관계용)
    """
    subject: str
    predicate: RelationType
    object: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    def __post_init__(self):
        """유효성 검증"""
        if not self.subject:
            raise ValueError("Subject cannot be empty")
        if not self.object:
            raise ValueError("Object cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """딕셔너리에서 생성"""
        return cls(
            subject=data["subject"],
            predicate=RelationType(data["predicate"]),
            object=data["object"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "system"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_to=datetime.fromisoformat(data["valid_to"]) if data.get("valid_to") else None
        )

    def is_valid_at(self, timestamp: datetime) -> bool:
        """특정 시점에 유효한지 확인"""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_to and timestamp > self.valid_to:
            return False
        return True

    def __hash__(self):
        """해시 (중복 제거용)"""
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        """동등성 비교"""
        if not isinstance(other, Relation):
            return False
        return (
            self.subject == other.subject and
            self.predicate == other.predicate and
            self.object == other.object
        )

    def __repr__(self):
        return f"({self.subject} --[{self.predicate.value}]--> {self.object})"


@dataclass
class InferenceResult:
    """
    추론 결과

    Attributes:
        rule_name: 적용된 규칙 이름
        insight_type: 인사이트 유형
        insight: 추론된 인사이트 문장
        confidence: 추론 신뢰도
        evidence: 근거 데이터
        recommendation: 권장 액션
        related_entities: 관련 엔티티들
        metadata: 추가 메타데이터
    """
    rule_name: str
    insight_type: InsightType
    insight: str
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "rule_name": self.rule_name,
            "insight_type": self.insight_type.value,
            "insight": self.insight,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "related_entities": self.related_entities,
            "metadata": self.metadata
        }

    def __repr__(self):
        return f"[{self.insight_type.value}] {self.insight} (confidence: {self.confidence:.2f})"


# =========================================================================
# 관계 생성 헬퍼 함수
# =========================================================================

def create_brand_product_relation(
    brand: str,
    product_asin: str,
    product_name: str,
    category: str = None,
    **properties
) -> Relation:
    """브랜드-제품 관계 생성"""
    props = {
        "product_name": product_name,
        "category": category,
        **properties
    }
    return Relation(
        subject=brand,
        predicate=RelationType.HAS_PRODUCT,
        object=product_asin,
        properties=props,
        source="crawl"
    )


def create_product_category_relation(
    product_asin: str,
    category_id: str,
    rank: int = None,
    **properties
) -> Relation:
    """제품-카테고리 관계 생성"""
    props = {
        "rank": rank,
        **properties
    }
    return Relation(
        subject=product_asin,
        predicate=RelationType.BELONGS_TO_CATEGORY,
        object=category_id,
        properties=props,
        source="crawl"
    )


def create_competition_relation(
    brand1: str,
    brand2: str,
    category: str,
    competition_type: str = "direct",
    **properties
) -> Relation:
    """경쟁 관계 생성"""
    predicate = (
        RelationType.DIRECT_COMPETITOR
        if competition_type == "direct"
        else RelationType.INDIRECT_COMPETITOR
    )
    props = {
        "category": category,
        "competition_type": competition_type,
        **properties
    }
    return Relation(
        subject=brand1,
        predicate=predicate,
        object=brand2,
        properties=props,
        source="inference"
    )


def create_metric_insight_relation(
    metric_name: str,
    insight_type: InsightType,
    insight_text: str,
    **properties
) -> Relation:
    """지표-인사이트 관계 생성"""
    props = {
        "insight_type": insight_type.value,
        "insight_text": insight_text,
        **properties
    }
    return Relation(
        subject=metric_name,
        predicate=RelationType.INDICATES,
        object=insight_type.value,
        properties=props,
        source="inference"
    )
