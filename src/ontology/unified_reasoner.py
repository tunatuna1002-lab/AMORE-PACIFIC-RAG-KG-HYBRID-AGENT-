"""
UnifiedReasoner - OWL + Business Rules 통합 추론
================================================
두 추론 엔진의 결과를 통합하여 일관된 인사이트를 제공.

추론 계층:
1. OWL Formal Reasoning (T-Box 기반, 논리적 일관성 보장)
2. Business Rules (Python 람다 기반, 도메인 지식 반영)
3. Result Fusion (중복 제거, 신뢰도 기반 랭킹)

Usage:
    unified = UnifiedReasoner(ontology_reasoner, owl_reasoner)
    results = unified.infer(context, query="LANEIGE 경쟁력 분석")
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Optional OWL import
try:
    from .owl_reasoner import OWLREADY2_AVAILABLE, OWLReasoner
except ImportError:
    OWLREADY2_AVAILABLE = False
    OWLReasoner = None


@dataclass
class UnifiedInferenceResult:
    """통합 추론 결과"""

    insight: str
    recommendation: str = ""
    confidence: float = 0.0
    source: str = "unknown"  # "owl", "business_rule", "merged"
    rule_name: str = ""
    category: str = ""  # insight_type or category
    supporting_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight": self.insight,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "source": self.source,
            "rule_name": self.rule_name,
            "category": self.category,
            "supporting_facts": self.supporting_facts,
        }


class UnifiedReasoner:
    """
    OWL + Business Rules 통합 추론기

    추론 파이프라인:
    1. OWL 추론 (형식 논리) -> 고신뢰도 결과
    2. Business Rule 추론 (도메인 규칙) -> 실용적 인사이트
    3. 결과 융합 -> 중복 제거 + 신뢰도 랭킹
    """

    def __init__(
        self,
        ontology_reasoner: Any = None,
        owl_reasoner: Any = None,
        owl_weight: float = 0.6,
        rule_weight: float = 0.4,
    ):
        """
        Args:
            ontology_reasoner: OntologyReasoner 인스턴스 (비즈니스 규칙)
            owl_reasoner: OWLReasoner 인스턴스 (형식 추론)
            owl_weight: OWL 결과 가중치
            rule_weight: 규칙 결과 가중치
        """
        self.rule_reasoner = ontology_reasoner
        self.owl_reasoner = owl_reasoner
        self.owl_weight = owl_weight
        self.rule_weight = rule_weight

        self._inference_count = 0

    def infer(
        self,
        context: dict[str, Any],
        query: str = "",
        max_results: int = 10,
    ) -> list[UnifiedInferenceResult]:
        """
        통합 추론 실행

        Args:
            context: 추론 컨텍스트 (brand, sos, rank, hhi, etc.)
            query: 사용자 쿼리 (의도 기반 필터링용)
            max_results: 최대 결과 수

        Returns:
            통합 추론 결과 리스트
        """
        self._inference_count += 1
        results: list[UnifiedInferenceResult] = []

        # Step 1: OWL 형식 추론
        owl_results = self._run_owl_reasoning(context)
        results.extend(owl_results)

        # Step 2: Business Rule 추론
        rule_results = self._run_rule_reasoning(context, query)
        results.extend(rule_results)

        # Step 3: 결과 융합 (중복 제거 + 랭킹)
        merged = self._merge_results(results)

        # 신뢰도 순 정렬
        merged.sort(key=lambda r: r.confidence, reverse=True)

        return merged[:max_results]

    def _run_owl_reasoning(self, context: dict[str, Any]) -> list[UnifiedInferenceResult]:
        """OWL 형식 추론 - get_brand_info, infer_market_positions, get_inferred_facts 활용"""
        results = []

        if not self.owl_reasoner:
            return results

        try:
            brand = context.get("brand", "")

            # Brand-specific reasoning via get_brand_info
            if brand and hasattr(self.owl_reasoner, "get_brand_info"):
                brand_info = self.owl_reasoner.get_brand_info(brand)
                if brand_info:
                    position = brand_info.get("market_position")
                    sos = brand_info.get("sos", 0.0)
                    competitors = brand_info.get("competitors", [])

                    if position:
                        position_labels = {
                            "DominantBrand": "시장 지배적 브랜드",
                            "StrongBrand": "강력한 시장 포지션",
                            "NicheBrand": "니치 브랜드",
                        }
                        label = position_labels.get(position, position)
                        results.append(
                            UnifiedInferenceResult(
                                insight=f"{brand}은(는) {label}으로 분류됩니다 (SoS: {sos:.1%})",
                                recommendation=self._position_recommendation(position),
                                confidence=0.8 * self.owl_weight,
                                source="owl",
                                rule_name="owl_market_position",
                                category="market_position",
                                supporting_facts=[f"SoS: {sos:.1%}", f"Position: {position}"],
                            )
                        )

                    if competitors:
                        results.append(
                            UnifiedInferenceResult(
                                insight=f"{brand}의 OWL 경쟁 관계: {', '.join(competitors[:5])}",
                                confidence=0.7 * self.owl_weight,
                                source="owl",
                                rule_name="owl_competition",
                                category="competition",
                                supporting_facts=[f"경쟁사 수: {len(competitors)}"],
                            )
                        )

            # Inferred facts (market positions, competition relations)
            if hasattr(self.owl_reasoner, "get_inferred_facts"):
                facts = self.owl_reasoner.get_inferred_facts()
                for fact in facts or []:
                    fact_type = fact.get("type", "")

                    if fact_type == "market_position" and fact.get("subject") != brand:
                        # Other brands' positions for market context
                        results.append(
                            UnifiedInferenceResult(
                                insight=f"{fact['subject']}은(는) {fact.get('position', 'unknown')} (SoS: {fact.get('sos', 0):.1%})",
                                confidence=0.55 * self.owl_weight,
                                source="owl",
                                rule_name="owl_market_context",
                                category="market_structure",
                            )
                        )

        except Exception as e:
            logger.warning(f"OWL reasoning failed: {e}")

        return results

    def _position_recommendation(self, position: str) -> str:
        """시장 포지션별 권고사항"""
        recommendations = {
            "DominantBrand": "시장 지배력 유지를 위해 혁신과 고객 충성도 강화에 집중하세요.",
            "StrongBrand": "시장 점유율 확대를 위한 공격적 마케팅 및 제품 라인업 확장을 검토하세요.",
            "NicheBrand": "차별화된 포지셔닝을 강화하고 목표 세그먼트 내 점유율 확대에 주력하세요.",
        }
        return recommendations.get(position, "")

    def _run_rule_reasoning(
        self, context: dict[str, Any], query: str
    ) -> list[UnifiedInferenceResult]:
        """Business Rule 추론 - OntologyReasoner.infer / infer_with_intent 활용"""
        results = []

        if not self.rule_reasoner:
            return results

        try:
            # 쿼리 의도 기반 추론 (가능하면)
            if query and hasattr(self.rule_reasoner, "infer_with_intent"):
                inferences = self.rule_reasoner.infer_with_intent(context=context, query=query)
            elif hasattr(self.rule_reasoner, "infer"):
                inferences = self.rule_reasoner.infer(context=context)
            else:
                return results

            for inf in inferences or []:
                # InferenceResult dataclass 처리
                if hasattr(inf, "insight"):
                    insight_type_val = ""
                    if hasattr(inf, "insight_type"):
                        insight_type_val = (
                            inf.insight_type.value
                            if hasattr(inf.insight_type, "value")
                            else str(inf.insight_type)
                        )

                    results.append(
                        UnifiedInferenceResult(
                            insight=inf.insight,
                            recommendation=getattr(inf, "recommendation", "") or "",
                            confidence=getattr(inf, "confidence", 0.5) * self.rule_weight,
                            source="business_rule",
                            rule_name=getattr(inf, "rule_name", ""),
                            category=insight_type_val,
                        )
                    )
                elif isinstance(inf, dict):
                    results.append(
                        UnifiedInferenceResult(
                            insight=inf.get("insight", ""),
                            recommendation=inf.get("recommendation", ""),
                            confidence=inf.get("confidence", 0.5) * self.rule_weight,
                            source="business_rule",
                            rule_name=inf.get("rule_name", ""),
                            category=inf.get("insight_type", inf.get("category", "")),
                        )
                    )

        except Exception as e:
            logger.warning(f"Rule reasoning failed: {e}")

        return results

    def _merge_results(self, results: list[UnifiedInferenceResult]) -> list[UnifiedInferenceResult]:
        """
        결과 융합 - 유사 인사이트 병합, 중복 제거
        """
        if not results:
            return []

        merged: list[UnifiedInferenceResult] = []
        seen_insights: set[str] = set()

        for result in results:
            # 단순 중복 제거 (정규화된 인사이트 텍스트 기준)
            normalized = result.insight.strip().lower()[:50]

            if normalized not in seen_insights:
                seen_insights.add(normalized)
                merged.append(result)
            else:
                # 중복이면 기존 결과의 신뢰도를 높임
                for existing in merged:
                    if existing.insight.strip().lower()[:50] == normalized:
                        # 양쪽 소스에서 확인되면 신뢰도 부스트
                        existing.confidence = min(
                            1.0, existing.confidence + result.confidence * 0.5
                        )
                        existing.source = "merged"
                        if result.supporting_facts:
                            existing.supporting_facts.extend(result.supporting_facts)
                        break

        return merged

    def explain(self, results: list[UnifiedInferenceResult]) -> str:
        """추론 결과 설명 생성"""
        if not results:
            return "추론 결과가 없습니다."

        parts = [f"총 {len(results)}개의 인사이트를 도출했습니다.\n"]

        for i, r in enumerate(results, 1):
            source_label = {
                "owl": "형식 추론",
                "business_rule": "비즈니스 규칙",
                "merged": "복합 추론",
            }.get(r.source, r.source)
            parts.append(f"{i}. [{source_label}] {r.insight}")
            if r.recommendation:
                parts.append(f"   -> 권고: {r.recommendation}")
            parts.append(f"   (신뢰도: {r.confidence:.0%})")
            parts.append("")

        return "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """통계"""
        return {
            "inference_count": self._inference_count,
            "has_owl": self.owl_reasoner is not None,
            "has_rules": self.rule_reasoner is not None,
            "owl_weight": self.owl_weight,
            "rule_weight": self.rule_weight,
        }
