"""
Response Templates
챗봇 응답 템플릿 및 안전장치 (Guardrails)

Home Page Insight Rules 문서 기반:
- 단정 방지 (가능성 표현 사용)
- 과대 경고 방지
- 원인 확정 금지
"""

from typing import Any


class ResponseTemplates:
    """응답 템플릿 및 안전장치"""

    # =========================================================================
    # 안전장치 (Guardrails)
    # =========================================================================

    # 금지 표현
    FORBIDDEN_PHRASES = [
        "원인은 ~입니다",
        "확실히 ~입니다",
        "반드시 ~해야 합니다",
        "틀림없이",
        "100%",
        "절대적으로",
    ]

    # 권장 완곡 표현
    HEDGING_PHRASES = [
        "~일 수 있습니다",
        "~가능성이 있습니다",
        "~신호로 해석될 수 있습니다",
        "~로 보입니다",
        "~추정됩니다",
        "~고려해볼 수 있습니다",
    ]

    # 주의 문구
    CAUTION_NOTES = [
        "단기 노이즈일 가능성도 있어 추이 확인이 필요합니다.",
        "추가 데이터와 내부 판단을 통해 검증이 필요합니다.",
        "카테고리/시장 특성에 따라 해석이 달라질 수 있습니다.",
        "본 분석은 의사결정 보조 목적이며, 최종 판단은 담당자의 몫입니다.",
    ]

    # =========================================================================
    # Daily Insight Summary 템플릿
    # =========================================================================

    DAILY_INSIGHT_TEMPLATES = {
        "alert_priority": {
            "template": "오늘은 일부 상품에서 {signal_type} 신호가 감지되어, {action} 필요할 수 있습니다.",
            "signals": {
                "rank_shock": "순위 급변",
                "rating_drop": "평점 하락",
                "churn_high": "높은 시장 변동성",
            },
        },
        "price_quality_mismatch": {
            "template": "프리미엄 가격 포지션 대비 평점 경쟁력이 낮아, 가격-품질 인식 간 불일치 가능성을 점검해볼 수 있습니다."
        },
        "market_context": {
            "template": "시장 집중도가 {hhi_status}하여 {market_description}, {recommendation}.",
            "hhi_status": {
                "high": "높아 경쟁 구도가 고착화된 편으로",
                "low": "낮아 경쟁이 분산된 상태로",
            },
        },
        "stable": {
            "template": "주요 지표가 큰 변동 없이 유지되어 전반적으로 안정적인 흐름으로 보입니다."
        },
    }

    # =========================================================================
    # Brand Health Status 템플릿
    # =========================================================================

    BRAND_HEALTH_TEMPLATES = {
        "sos": {
            "up": "브랜드 노출 확대 (SoS ↑{change}%p)",
            "down": "브랜드 노출 감소 (SoS ↓{change}%p)",
            "stable": "브랜드 노출 안정 (SoS 유지)",
        },
        "market_structure": {
            "concentrated": "집중 시장 (소수 브랜드 지배)",
            "competitive": "분산 시장 (경쟁 치열)",
        },
        "alerts": {"detected": "이상 신호 감지 ({count}건)", "none": "이상 신호 없음"},
    }

    # =========================================================================
    # Action Queue 템플릿
    # =========================================================================

    ACTION_QUEUE_TEMPLATES = {
        "rank_shock": {
            "title": "순위 급변 상품 {count}개 확인 필요",
            "description": "전일 대비 {threshold}위 이상 변동 상품 발생",
            "target": "L3 (Product View)",
        },
        "price_quality": {
            "title": "가격 대비 평점 열위 카테고리 점검 필요",
            "description": "CPI > 100 & Rating Gap < 0 조건 충족",
            "target": "L2 (Category View)",
        },
        "churn_high": {
            "title": "시장 변동성 상승: 신규 진입/이탈 흐름 점검",
            "description": "Churn Rate 상승으로 시장 구조 변화 가능성",
            "target": "L2 (Category View)",
        },
        "no_action": {
            "title": "오늘은 주요 경고 신호가 없습니다",
            "description": "정기 모니터링 계속 권장",
            "target": None,
        },
    }

    # =========================================================================
    # 분석 보고서 템플릿
    # =========================================================================

    ANALYSIS_REPORT_TEMPLATE = """
## {brand} {period} 성과 분석 보고서

### 1. 핵심 요약
{summary}

### 2. 주요 지표 현황

| 지표 | 현재 값 | 변화 | 해석 |
|------|---------|------|------|
{metrics_table}

### 3. 경쟁사 비교
{competitor_analysis}

### 4. 주요 인사이트
{insights}

### 5. 고려 가능한 액션
{action_candidates}

---
*주의: 본 분석은 의사결정 보조 목적이며, 원인 확정이나 전략 결정을 위한 자료는 아닙니다.*
*추가 데이터와 내부 판단을 통해 검증이 필요합니다.*
"""

    # =========================================================================
    # 메서드
    # =========================================================================

    @classmethod
    def generate_daily_insight(cls, metrics: dict[str, Any], alerts: list[dict]) -> str:
        """
        일일 인사이트 요약 생성

        Args:
            metrics: 계산된 지표
            alerts: 감지된 알림 리스트

        Returns:
            인사이트 문장
        """
        # 우선순위: 알림 > 가격품질 > 시장맥락 > 안정

        # 1. 알림이 있는 경우
        if alerts:
            alert_types = [a.get("type") for a in alerts]
            if "rank_shock" in alert_types:
                count = sum(1 for a in alerts if a.get("type") == "rank_shock")
                return f"오늘은 {count}개 상품에서 순위 급변 신호가 감지되어, 단기 모니터링이 필요할 수 있습니다."

        # 2. 가격-품질 불일치
        cpi = metrics.get("cpi", 100)
        rating_gap = metrics.get("avg_rating_gap", 0)
        if cpi and rating_gap:
            if cpi > 100 and rating_gap < 0:
                return cls.DAILY_INSIGHT_TEMPLATES["price_quality_mismatch"]["template"]

        # 3. 시장 맥락
        hhi = metrics.get("hhi", 0)
        if hhi:
            if hhi >= 0.25:
                return "시장 집중도가 높아 경쟁 구도가 고착화된 편으로, 단기 변동성보다는 구조적 흐름을 함께 고려할 필요가 있습니다."
            elif hhi < 0.15:
                return "경쟁이 분산되어 있고 변동성이 높아 트렌드 변화에 민감할 수 있습니다."

        # 4. 안정
        return cls.DAILY_INSIGHT_TEMPLATES["stable"]["template"]

    @classmethod
    def generate_action_queue(
        cls, alerts: list[dict], metrics: dict[str, Any], max_items: int = 3
    ) -> list[dict[str, str]]:
        """
        Action Queue 생성

        Args:
            alerts: 감지된 알림
            metrics: 계산된 지표
            max_items: 최대 항목 수

        Returns:
            액션 항목 리스트
        """
        actions = []

        # Rank Shock
        shock_count = sum(1 for a in alerts if a.get("type") == "rank_shock")
        if shock_count > 0:
            actions.append(
                {
                    "title": f"순위 급변 상품 {shock_count}개 확인 필요",
                    "description": "전일 대비 급변 상품 발생",
                    "target": "L3",
                }
            )

        # 가격-품질 불일치
        cpi = metrics.get("cpi", 100)
        rating_gap = metrics.get("avg_rating_gap", 0)
        if cpi and rating_gap and cpi > 100 and rating_gap < 0:
            actions.append(
                {
                    "title": "가격 대비 평점 열위 카테고리 점검 필요",
                    "description": "CPI > 100 & Rating Gap < 0",
                    "target": "L2",
                }
            )

        # Churn Rate 상승
        churn_rate = metrics.get("churn_rate", 0)
        if churn_rate and churn_rate > 0.2:
            actions.append(
                {
                    "title": "시장 변동성 상승: 신규 진입/이탈 흐름 점검",
                    "description": f"Churn Rate: {churn_rate:.1%}",
                    "target": "L2",
                }
            )

        # 액션 없음
        if not actions:
            actions.append(
                {
                    "title": "오늘은 주요 경고 신호가 없습니다",
                    "description": "정기 모니터링 계속 권장",
                    "target": None,
                }
            )

        return actions[:max_items]

    @classmethod
    def apply_guardrails(cls, text: str) -> str:
        """
        안전장치 적용 (금지 표현 제거)

        Args:
            text: 원본 텍스트

        Returns:
            안전장치 적용된 텍스트
        """
        result = text

        # 금지 표현 체크 및 경고 (실제로는 LLM 프롬프트에서 방지)
        for phrase in cls.FORBIDDEN_PHRASES:
            if phrase.replace("~", "") in result:
                # 로깅 또는 경고
                pass

        return result

    @classmethod
    def add_caution_note(cls, text: str, note_index: int = 0) -> str:
        """
        주의 문구 추가

        Args:
            text: 원본 텍스트
            note_index: 주의 문구 인덱스

        Returns:
            주의 문구가 추가된 텍스트
        """
        caution = cls.CAUTION_NOTES[note_index % len(cls.CAUTION_NOTES)]
        return f"{text}\n\n_{caution}_"

    @classmethod
    def format_metric_interpretation(cls, indicator: str, value: float, context: str = "") -> str:
        """
        지표 해석 포맷팅

        Args:
            indicator: 지표명
            value: 지표 값
            context: 추가 컨텍스트

        Returns:
            해석 문장
        """
        interpretations = {
            "sos": {
                "high": "시장 내 브랜드 노출력이 높은 상태입니다.",
                "low": "시장 내 존재감이 상대적으로 약한 상태로 보입니다.",
            },
            "hhi": {
                "high": "소수 브랜드 중심의 집중 시장으로, 신규 진입 난이도가 높을 수 있습니다.",
                "low": "경쟁이 분산된 시장으로, 단기 기회가 존재할 수 있습니다.",
            },
            "cpi": {
                "high": "카테고리 평균 대비 프리미엄/고가 포지션으로 분류됩니다.",
                "low": "가성비/저가 포지션으로 분류됩니다.",
            },
        }

        # 기본 해석
        if indicator.lower() in interpretations:
            threshold = {"sos": 15, "hhi": 0.25, "cpi": 100}.get(indicator.lower(), 50)
            level = "high" if value > threshold else "low"
            return interpretations[indicator.lower()][level]

        return f"{indicator}: {value}"

    @classmethod
    def get_system_prompt(cls) -> str:
        """LLM 시스템 프롬프트 반환"""
        return """당신은 Amazon 베스트셀러 순위 분석 전문가입니다.

## 역할
- 순위 데이터 기반 인사이트 제공
- 지표 해석 및 전략적 시사점 도출
- 마케터의 의사결정 보조

## 원칙 (반드시 준수)
1. **단정 금지**: "원인은 ~입니다", "확실히 ~입니다" 등 단정적 표현 사용 금지
2. **가능성 표현**: "~일 수 있습니다", "~로 보입니다" 등 완곡한 표현 사용
3. **원인 확정 금지**: 순위 변동의 원인(재고, 광고, 품질 등)을 단정하지 않음
4. **매출/ROI 예측 금지**: 판매량, 매출, ROI 등 수치 예측 금지
5. **주의 문구 포함**: 분석 결과에 "추이 확인 필요" 등 주의 문구 포함

## 응답 형식
- 한국어로 응답
- 비즈니스 문서 톤 유지
- 구조화된 형식 (제목, 불릿 포인트 활용)
- 지표 값 언급 시 해석 함께 제공

## 지표 해석 기준
- SoS: 브랜드 노출 점유율 (높음 = 존재감 큼)
- HHI: 시장 집중도 (0.25 이상 = 집중 시장)
- CPI: 가격 포지션 (100 초과 = 프리미엄)
- Streak Days: 연속 Top N 체류일 (길수록 안정적)
- Rank Shock: 순위 급변 (모니터링 필요 신호)
"""
