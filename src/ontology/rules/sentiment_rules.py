"""
Sentiment Analysis Based Rules
감성 분석 기반 규칙
"""

from ..reasoner import InferenceRule, RuleCondition
from ..relations import InsightType

RULE_SENTIMENT_STRENGTH_HYDRATION = InferenceRule(
    name="sentiment_strength_hydration",
    description="Hydration 클러스터에서 강한 감성 태그는 보습력 강점을 나타냄",
    conditions=[
        RuleCondition(
            name="has_hydration_sentiment",
            check=lambda ctx: any(
                cluster == "Hydration" for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Hydration 클러스터 감성 보유",
        ),
        RuleCondition(
            name="hydration_tags_count",
            check=lambda ctx: len(ctx.get("sentiment_clusters", {}).get("Hydration", [])) >= 2,
            description="Hydration 감성 태그 2개 이상",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"고객들이 보습력({', '.join(ctx.get('sentiment_clusters', {}).get('Hydration', []))})을 "
        f"주요 강점으로 인식하고 있습니다.",
        "strength": "hydration",
        "recommendation": "보습력을 핵심 마케팅 메시지로 강조하세요. "
        "관련 키워드를 제품 설명과 광고에 활용하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_cluster": "Hydration",
            "tags": ctx.get("sentiment_clusters", {}).get("Hydration", []),
        },
    },
    insight_type=InsightType.SENTIMENT_STRENGTH,
    priority=7,
    confidence=0.85,
    tags=["sentiment", "strength", "hydration", "positive"],
)


RULE_SENTIMENT_VALUE_ADVANTAGE = InferenceRule(
    name="sentiment_value_advantage",
    description="가성비 감성이 있고 경쟁사에 없으면 가성비 경쟁 우위",
    conditions=[
        RuleCondition(
            name="has_value_sentiment",
            check=lambda ctx: any(
                tag.lower() in ["value for money", "good value", "affordable", "worth the price"]
                for tag in ctx.get("sentiment_tags", [])
            ),
            description="가성비 관련 감성 태그 보유",
        ),
        RuleCondition(
            name="competitor_lacks_value",
            check=lambda ctx: not any(
                tag.lower() in ["value for money", "good value", "affordable"]
                for tag in ctx.get("competitor_sentiment_tags", [])
            ),
            description="경쟁사에 가성비 감성 없음",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "경쟁사 대비 '가성비' 인식에서 우위를 보이고 있습니다. "
        "고객들이 가격 대비 만족도를 높게 평가합니다.",
        "advantage": "value_perception",
        "recommendation": "가성비 메시지를 마케팅에 적극 활용하세요. "
        "비교 광고나 '가격 대비 효과' 강조 콘텐츠가 효과적일 수 있습니다.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {"sentiment_type": "value", "has_competitor_value": False},
    },
    insight_type=InsightType.SENTIMENT_ADVANTAGE,
    priority=8,
    confidence=0.8,
    tags=["sentiment", "value", "competitive", "positive"],
)


RULE_SENTIMENT_WEAKNESS_PACKAGING = InferenceRule(
    name="sentiment_weakness_packaging",
    description="패키징 관련 부정 감성이 있으면 패키징 개선 필요",
    conditions=[
        RuleCondition(
            name="no_packaging_positive",
            check=lambda ctx: not any(
                cluster == "Packaging" for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="패키징 긍정 감성 없음",
        ),
        RuleCondition(
            name="competitor_has_packaging",
            check=lambda ctx: any(
                cluster == "Packaging"
                for cluster in ctx.get("competitor_sentiment_clusters", {}).keys()
            ),
            description="경쟁사에 패키징 긍정 감성 있음",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "경쟁사 대비 패키징 관련 고객 인식이 약합니다. "
        "패키징 개선 또는 패키징 장점 커뮤니케이션이 필요할 수 있습니다.",
        "weakness": "packaging",
        "recommendation": "패키징 디자인/기능 개선을 검토하거나, "
        "기존 패키징의 장점을 고객에게 더 잘 전달하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {"sentiment_gap": "packaging", "competitor_has": True},
    },
    insight_type=InsightType.SENTIMENT_WEAKNESS,
    priority=6,
    confidence=0.75,
    tags=["sentiment", "weakness", "packaging", "risk"],
)


RULE_SENTIMENT_USABILITY_STRENGTH = InferenceRule(
    name="sentiment_usability_strength",
    description="사용 편의성 감성이 강하면 사용성 강점",
    conditions=[
        RuleCondition(
            name="has_usability_sentiment",
            check=lambda ctx: any(
                cluster == "Usability" for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Usability 클러스터 감성 보유",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"고객들이 사용 편의성({', '.join(ctx.get('sentiment_clusters', {}).get('Usability', []))})을 "
        f"긍정적으로 평가하고 있습니다.",
        "strength": "usability",
        "recommendation": "'간편한 사용법', '휴대성' 등을 마케팅 포인트로 활용하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_cluster": "Usability",
            "tags": ctx.get("sentiment_clusters", {}).get("Usability", []),
        },
    },
    insight_type=InsightType.SENTIMENT_STRENGTH,
    priority=6,
    confidence=0.8,
    tags=["sentiment", "strength", "usability", "positive"],
)


RULE_SENTIMENT_EFFECTIVENESS = InferenceRule(
    name="sentiment_effectiveness_strong",
    description="효과성 관련 감성이 강하면 제품 효과 강점",
    conditions=[
        RuleCondition(
            name="has_effectiveness_sentiment",
            check=lambda ctx: any(
                cluster == "Effectiveness" for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Effectiveness 클러스터 감성 보유",
        ),
        RuleCondition(
            name="effectiveness_tags_count",
            check=lambda ctx: len(ctx.get("sentiment_clusters", {}).get("Effectiveness", [])) >= 1,
            description="효과성 감성 태그 1개 이상",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"고객들이 제품 효과({', '.join(ctx.get('sentiment_clusters', {}).get('Effectiveness', []))})를 "
        f"높게 평가하고 있습니다. 실제 효과가 구매 결정에 중요한 요소입니다.",
        "strength": "effectiveness",
        "recommendation": "Before/After 콘텐츠, 고객 리뷰 기반 효과 증명 마케팅을 강화하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_cluster": "Effectiveness",
            "tags": ctx.get("sentiment_clusters", {}).get("Effectiveness", []),
        },
    },
    insight_type=InsightType.SENTIMENT_STRENGTH,
    priority=7,
    confidence=0.85,
    tags=["sentiment", "strength", "effectiveness", "positive"],
)


RULE_SENTIMENT_GAP_SENSORY = InferenceRule(
    name="sentiment_gap_sensory",
    description="경쟁사에 Sensory 감성이 있고 자사에 없으면 감각적 경험 격차",
    conditions=[
        RuleCondition(
            name="no_sensory_sentiment",
            check=lambda ctx: not any(
                cluster == "Sensory" for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Sensory 감성 없음",
        ),
        RuleCondition(
            name="competitor_has_sensory",
            check=lambda ctx: any(
                cluster == "Sensory"
                for cluster in ctx.get("competitor_sentiment_clusters", {}).keys()
            ),
            description="경쟁사에 Sensory 감성 있음",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "경쟁사 대비 '향', '텍스처' 등 감각적 경험에 대한 고객 인식이 부족합니다.",
        "gap": "sensory_experience",
        "recommendation": "제품의 향, 질감 등 감각적 특성을 마케팅에서 강조하거나 "
        "제품 포뮬러 개선을 검토하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {"sentiment_gap": "sensory", "competitor_has": True},
    },
    insight_type=InsightType.SENTIMENT_GAP,
    priority=6,
    confidence=0.75,
    tags=["sentiment", "gap", "sensory", "risk"],
)


RULE_CUSTOMER_PERCEPTION_POSITIVE = InferenceRule(
    name="customer_perception_positive",
    description="AI 요약에 긍정 키워드가 있으면 전반적 고객 인식 긍정",
    conditions=[
        RuleCondition(
            name="has_ai_summary",
            check=lambda ctx: bool(ctx.get("ai_summary")),
            description="AI 요약 데이터 존재",
        ),
        RuleCondition(
            name="positive_keywords_in_summary",
            check=lambda ctx: any(
                keyword in (ctx.get("ai_summary", "") or "").lower()
                for keyword in [
                    "love",
                    "great",
                    "excellent",
                    "amazing",
                    "best",
                    "recommend",
                    "perfect",
                ]
            ),
            description="AI 요약에 긍정 키워드 포함",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "Amazon AI 리뷰 요약에서 전반적으로 긍정적인 고객 인식이 확인됩니다.",
        "perception": "positive",
        "recommendation": "긍정적 리뷰를 마케팅에 활용하고, 리뷰 수를 늘려 신뢰도를 강화하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {"has_ai_summary": True, "perception_type": "positive"},
    },
    insight_type=InsightType.CUSTOMER_PERCEPTION,
    priority=7,
    confidence=0.8,
    tags=["sentiment", "perception", "positive"],
)


RULE_CUSTOMER_PERCEPTION_MIXED = InferenceRule(
    name="customer_perception_mixed",
    description="AI 요약에 긍정과 부정이 섞여 있으면 혼합된 인식",
    conditions=[
        RuleCondition(
            name="has_ai_summary",
            check=lambda ctx: bool(ctx.get("ai_summary")),
            description="AI 요약 데이터 존재",
        ),
        RuleCondition(
            name="mixed_keywords_in_summary",
            check=lambda ctx: (
                any(
                    pos in (ctx.get("ai_summary", "") or "").lower()
                    for pos in ["like", "good", "nice"]
                )
                and any(
                    neg in (ctx.get("ai_summary", "") or "").lower()
                    for neg in ["but", "however", "although", "wish", "could be better"]
                )
            ),
            description="AI 요약에 긍정+부정 키워드 혼재",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "고객 인식이 긍정과 부정이 혼재되어 있습니다. "
        "특정 측면에서 개선 기회가 있을 수 있습니다.",
        "perception": "mixed",
        "recommendation": "부정적 피드백의 원인을 분석하고 해당 영역의 개선을 검토하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {"has_ai_summary": True, "perception_type": "mixed"},
    },
    insight_type=InsightType.CUSTOMER_PERCEPTION,
    priority=6,
    confidence=0.75,
    tags=["sentiment", "perception", "mixed", "neutral"],
)


SENTIMENT_RULES: list[InferenceRule] = [
    RULE_SENTIMENT_STRENGTH_HYDRATION,
    RULE_SENTIMENT_VALUE_ADVANTAGE,
    RULE_SENTIMENT_WEAKNESS_PACKAGING,
    RULE_SENTIMENT_USABILITY_STRENGTH,
    RULE_SENTIMENT_EFFECTIVENESS,
    RULE_SENTIMENT_GAP_SENSORY,
    RULE_CUSTOMER_PERCEPTION_POSITIVE,
    RULE_CUSTOMER_PERCEPTION_MIXED,
]
