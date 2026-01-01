"""
Ontology Reasoner
규칙 기반 추론 엔진

기능:
1. 조건-결론 규칙 정의 및 실행
2. 순방향 추론 (Forward Chaining)
3. 추론 과정 설명 (Explainability)
4. 규칙 우선순위 관리
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

from .relations import (
    InsightType,
    MarketPosition,
    InferenceResult,
    RelationType
)
from .knowledge_graph import KnowledgeGraph


# 로거 설정
logger = logging.getLogger(__name__)


@dataclass
class RuleCondition:
    """
    규칙 조건

    Attributes:
        name: 조건 이름
        check: 조건 검사 함수 (context -> bool)
        description: 조건 설명
    """
    name: str
    check: Callable[[Dict[str, Any]], bool]
    description: str

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """조건 평가"""
        try:
            return self.check(context)
        except Exception as e:
            logger.warning(f"Condition {self.name} evaluation failed: {e}")
            return False


@dataclass
class InferenceRule:
    """
    추론 규칙

    구조: IF conditions THEN conclusion

    Attributes:
        name: 규칙 이름 (고유)
        description: 규칙 설명
        conditions: 조건 리스트 (모두 만족해야 함 - AND)
        conclusion: 결론 생성 함수
        insight_type: 생성되는 인사이트 유형
        priority: 우선순위 (높을수록 먼저 평가)
        confidence: 규칙의 기본 신뢰도
        tags: 규칙 태그 (필터링용)
    """
    name: str
    description: str
    conditions: List[RuleCondition]
    conclusion: Callable[[Dict[str, Any]], Dict[str, Any]]
    insight_type: InsightType
    priority: int = 0
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)

    def evaluate_conditions(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        모든 조건 평가

        Returns:
            (모든 조건 만족 여부, 만족한 조건 리스트)
        """
        satisfied = []
        for condition in self.conditions:
            if condition.evaluate(context):
                satisfied.append(condition.name)
            else:
                return False, satisfied
        return True, satisfied

    def apply(self, context: Dict[str, Any]) -> Optional[InferenceResult]:
        """
        규칙 적용

        Args:
            context: 추론 컨텍스트

        Returns:
            추론 결과 (조건 불만족 시 None)
        """
        all_satisfied, satisfied_conditions = self.evaluate_conditions(context)

        if not all_satisfied:
            return None

        try:
            conclusion_data = self.conclusion(context)

            return InferenceResult(
                rule_name=self.name,
                insight_type=self.insight_type,
                insight=conclusion_data.get("insight", ""),
                confidence=self.confidence * conclusion_data.get("confidence_modifier", 1.0),
                evidence={
                    "satisfied_conditions": satisfied_conditions,
                    "context_snapshot": {
                        k: v for k, v in context.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                },
                recommendation=conclusion_data.get("recommendation"),
                related_entities=conclusion_data.get("related_entities", []),
                metadata={
                    "rule_description": self.description,
                    "priority": self.priority,
                    "tags": self.tags,
                    **conclusion_data.get("metadata", {})
                }
            )
        except Exception as e:
            logger.error(f"Rule {self.name} conclusion failed: {e}")
            return None


class OntologyReasoner:
    """
    온톨로지 추론 엔진

    기능:
    1. 규칙 등록 및 관리
    2. 순방향 추론 실행
    3. 추론 과정 추적 및 설명
    4. 지식 그래프 연동

    사용 예:
        reasoner = OntologyReasoner(knowledge_graph)
        reasoner.register_rule(some_rule)
        results = reasoner.infer(context)
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        max_iterations: int = 10
    ):
        """
        Args:
            knowledge_graph: 연동할 지식 그래프
            max_iterations: 최대 추론 반복 횟수 (순환 방지)
        """
        self.kg = knowledge_graph
        self.max_iterations = max_iterations

        # 규칙 저장소
        self.rules: Dict[str, InferenceRule] = {}
        self.rules_by_priority: List[InferenceRule] = []

        # 추론 히스토리
        self.inference_history: List[Dict[str, Any]] = []

    # =========================================================================
    # 규칙 관리
    # =========================================================================

    def register_rule(self, rule: InferenceRule) -> None:
        """
        규칙 등록

        Args:
            rule: 등록할 규칙
        """
        if rule.name in self.rules:
            logger.warning(f"Rule {rule.name} already exists, overwriting")

        self.rules[rule.name] = rule
        self._rebuild_priority_list()
        logger.info(f"Registered rule: {rule.name}")

    def register_rules(self, rules: List[InferenceRule]) -> None:
        """여러 규칙 일괄 등록"""
        for rule in rules:
            self.rules[rule.name] = rule
        self._rebuild_priority_list()
        logger.info(f"Registered {len(rules)} rules")

    def unregister_rule(self, rule_name: str) -> bool:
        """규칙 삭제"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self._rebuild_priority_list()
            return True
        return False

    def get_rule(self, rule_name: str) -> Optional[InferenceRule]:
        """규칙 조회"""
        return self.rules.get(rule_name)

    def get_rules_by_tag(self, tag: str) -> List[InferenceRule]:
        """태그로 규칙 필터링"""
        return [r for r in self.rules.values() if tag in r.tags]

    def get_rules_by_insight_type(self, insight_type: InsightType) -> List[InferenceRule]:
        """인사이트 유형으로 규칙 필터링"""
        return [r for r in self.rules.values() if r.insight_type == insight_type]

    def _rebuild_priority_list(self) -> None:
        """우선순위 리스트 재구성"""
        self.rules_by_priority = sorted(
            self.rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )

    # =========================================================================
    # 추론 실행
    # =========================================================================

    def infer(
        self,
        context: Dict[str, Any],
        rule_filter: Optional[Callable[[InferenceRule], bool]] = None,
        max_results: Optional[int] = None
    ) -> List[InferenceResult]:
        """
        순방향 추론 실행

        Args:
            context: 추론 컨텍스트 (지표, 엔티티 정보 등)
            rule_filter: 규칙 필터 함수
            max_results: 최대 결과 수

        Returns:
            추론 결과 리스트
        """
        results: List[InferenceResult] = []
        applied_rules: Set[str] = set()

        # 지식 그래프에서 컨텍스트 보강
        enriched_context = self._enrich_context(context)

        # 규칙 적용
        for rule in self.rules_by_priority:
            # 필터 적용
            if rule_filter and not rule_filter(rule):
                continue

            # 이미 적용된 규칙은 스킵
            if rule.name in applied_rules:
                continue

            # 규칙 적용
            result = rule.apply(enriched_context)

            if result:
                results.append(result)
                applied_rules.add(rule.name)

                # 최대 결과 수 체크
                if max_results and len(results) >= max_results:
                    break

        # 히스토리 기록
        self._record_inference(context, results)

        return results

    def infer_for_entity(
        self,
        entity: str,
        entity_type: str = "brand"
    ) -> List[InferenceResult]:
        """
        특정 엔티티에 대한 추론

        Args:
            entity: 엔티티 ID (브랜드명, ASIN 등)
            entity_type: 엔티티 유형

        Returns:
            추론 결과 리스트
        """
        # 지식 그래프에서 엔티티 정보 수집
        context = self._build_entity_context(entity, entity_type)

        # 추론 실행
        return self.infer(context)

    def infer_by_insight_type(
        self,
        context: Dict[str, Any],
        insight_types: List[InsightType]
    ) -> List[InferenceResult]:
        """
        특정 인사이트 유형에 대해서만 추론

        Args:
            context: 추론 컨텍스트
            insight_types: 원하는 인사이트 유형들

        Returns:
            추론 결과 리스트
        """
        rule_filter = lambda r: r.insight_type in insight_types
        return self.infer(context, rule_filter=rule_filter)

    # =========================================================================
    # 컨텍스트 처리
    # =========================================================================

    def _enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        지식 그래프를 활용한 컨텍스트 보강

        Args:
            context: 원본 컨텍스트

        Returns:
            보강된 컨텍스트
        """
        enriched = context.copy()

        if not self.kg:
            return enriched

        # 브랜드 정보 보강
        brand = context.get("brand")
        if brand:
            # 브랜드 메타데이터
            brand_meta = self.kg.get_entity_metadata(brand)
            if brand_meta:
                enriched["brand_metadata"] = brand_meta

            # 경쟁사 정보
            competitors = self.kg.get_competitors(brand)
            enriched["competitors"] = competitors
            enriched["competitor_count"] = len(competitors)

            # 제품 정보
            products = self.kg.get_brand_products(brand)
            enriched["products"] = products
            enriched["product_count"] = len(products)

        # 카테고리 정보 보강
        category = context.get("category")
        if category:
            category_brands = self.kg.get_category_brands(category)
            enriched["category_brands"] = category_brands
            enriched["category_brand_count"] = len(category_brands)

        return enriched

    def _build_entity_context(
        self,
        entity: str,
        entity_type: str
    ) -> Dict[str, Any]:
        """
        엔티티 기반 컨텍스트 구성

        Args:
            entity: 엔티티 ID
            entity_type: 엔티티 유형

        Returns:
            추론용 컨텍스트
        """
        context = {
            "entity": entity,
            "entity_type": entity_type
        }

        if not self.kg:
            return context

        # 엔티티 메타데이터
        metadata = self.kg.get_entity_metadata(entity)
        context.update(metadata)

        if entity_type == "brand":
            context["brand"] = entity
            # SoS, avg_rank 등은 메타데이터에서 가져옴
            context["sos"] = metadata.get("sos", 0)
            context["avg_rank"] = metadata.get("avg_rank")

        elif entity_type == "product":
            context["asin"] = entity
            context["current_rank"] = metadata.get("current_rank")
            context["rank_change_1d"] = metadata.get("rank_change_1d")
            context["streak_days"] = metadata.get("streak_days")

        # 관련 관계 정보
        entity_context = self.kg.get_entity_context(entity, depth=1)
        context["relations"] = entity_context.get("relations", {})

        return context

    # =========================================================================
    # 설명 가능성 (Explainability)
    # =========================================================================

    def explain_inference(self, result: InferenceResult) -> str:
        """
        추론 과정 설명 생성

        Args:
            result: 추론 결과

        Returns:
            설명 문자열
        """
        rule = self.rules.get(result.rule_name)
        if not rule:
            return f"[알 수 없는 규칙] {result.insight}"

        explanation_parts = []

        # 규칙 정보
        explanation_parts.append(f"## 추론 규칙: {rule.name}")
        explanation_parts.append(f"**설명**: {rule.description}")
        explanation_parts.append("")

        # 조건
        explanation_parts.append("### 적용된 조건")
        for condition in rule.conditions:
            satisfied = condition.name in result.evidence.get("satisfied_conditions", [])
            status = "O" if satisfied else "X"
            explanation_parts.append(f"- [{status}] {condition.description}")
        explanation_parts.append("")

        # 근거 데이터
        explanation_parts.append("### 근거 데이터")
        context_snapshot = result.evidence.get("context_snapshot", {})
        for key, value in context_snapshot.items():
            explanation_parts.append(f"- {key}: {value}")
        explanation_parts.append("")

        # 결론
        explanation_parts.append("### 결론")
        explanation_parts.append(f"**인사이트**: {result.insight}")
        if result.recommendation:
            explanation_parts.append(f"**권장 액션**: {result.recommendation}")
        explanation_parts.append(f"**신뢰도**: {result.confidence:.2f}")

        return "\n".join(explanation_parts)

    def explain_all(self, results: List[InferenceResult]) -> str:
        """모든 추론 결과 설명"""
        if not results:
            return "추론된 인사이트가 없습니다."

        explanations = []
        for i, result in enumerate(results, 1):
            explanations.append(f"---\n## 인사이트 {i}\n")
            explanations.append(self.explain_inference(result))

        return "\n\n".join(explanations)

    # =========================================================================
    # 히스토리 관리
    # =========================================================================

    def _record_inference(
        self,
        context: Dict[str, Any],
        results: List[InferenceResult]
    ) -> None:
        """추론 히스토리 기록"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "context_keys": list(context.keys()),
            "results_count": len(results),
            "applied_rules": [r.rule_name for r in results],
            "insight_types": [r.insight_type.value for r in results]
        }
        self.inference_history.append(record)

        # 최대 100개 유지
        if len(self.inference_history) > 100:
            self.inference_history = self.inference_history[-100:]

    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계"""
        if not self.inference_history:
            return {"total_inferences": 0}

        rule_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}

        for record in self.inference_history:
            for rule in record.get("applied_rules", []):
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            for itype in record.get("insight_types", []):
                type_counts[itype] = type_counts.get(itype, 0) + 1

        return {
            "total_inferences": len(self.inference_history),
            "rules_applied": rule_counts,
            "insight_types": type_counts,
            "registered_rules": len(self.rules)
        }

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def validate_rules(self) -> List[str]:
        """
        등록된 규칙 유효성 검사

        Returns:
            경고 메시지 리스트
        """
        warnings = []

        for name, rule in self.rules.items():
            # 조건 없음
            if not rule.conditions:
                warnings.append(f"Rule '{name}' has no conditions")

            # 중복 우선순위
            same_priority = [
                r.name for r in self.rules.values()
                if r.priority == rule.priority and r.name != name
            ]
            if same_priority:
                warnings.append(
                    f"Rule '{name}' has same priority as: {same_priority}"
                )

        return warnings

    def list_rules(self) -> List[Dict[str, Any]]:
        """등록된 규칙 목록"""
        return [
            {
                "name": r.name,
                "description": r.description,
                "insight_type": r.insight_type.value,
                "priority": r.priority,
                "conditions_count": len(r.conditions),
                "tags": r.tags
            }
            for r in self.rules_by_priority
        ]

    def __repr__(self):
        return f"OntologyReasoner(rules={len(self.rules)}, kg={'connected' if self.kg else 'none'})"


# =========================================================================
# 조건 빌더 헬퍼
# =========================================================================

def condition(
    name: str,
    description: str
) -> Callable[[Callable], RuleCondition]:
    """
    조건 데코레이터

    사용 예:
        @condition("high_sos", "SoS가 15% 이상")
        def check_high_sos(ctx):
            return ctx.get("sos", 0) >= 0.15
    """
    def decorator(func: Callable[[Dict[str, Any]], bool]) -> RuleCondition:
        return RuleCondition(
            name=name,
            check=func,
            description=description
        )
    return decorator


# =========================================================================
# 표준 조건 라이브러리
# =========================================================================

class StandardConditions:
    """자주 사용되는 표준 조건들"""

    @staticmethod
    def sos_above(threshold: float) -> RuleCondition:
        """SoS가 임계값 이상"""
        return RuleCondition(
            name=f"sos_above_{threshold}",
            check=lambda ctx: ctx.get("sos", 0) >= threshold,
            description=f"SoS >= {threshold*100:.0f}%"
        )

    @staticmethod
    def sos_below(threshold: float) -> RuleCondition:
        """SoS가 임계값 이하"""
        return RuleCondition(
            name=f"sos_below_{threshold}",
            check=lambda ctx: ctx.get("sos", 0) < threshold,
            description=f"SoS < {threshold*100:.0f}%"
        )

    @staticmethod
    def hhi_above(threshold: float) -> RuleCondition:
        """HHI가 임계값 이상 (집중 시장)"""
        return RuleCondition(
            name=f"hhi_above_{threshold}",
            check=lambda ctx: ctx.get("hhi", 0) >= threshold,
            description=f"HHI >= {threshold} (집중 시장)"
        )

    @staticmethod
    def hhi_below(threshold: float) -> RuleCondition:
        """HHI가 임계값 이하 (분산 시장)"""
        return RuleCondition(
            name=f"hhi_below_{threshold}",
            check=lambda ctx: ctx.get("hhi", 0) < threshold,
            description=f"HHI < {threshold} (분산 시장)"
        )

    @staticmethod
    def cpi_above(threshold: float) -> RuleCondition:
        """CPI가 임계값 이상 (프리미엄)"""
        return RuleCondition(
            name=f"cpi_above_{threshold}",
            check=lambda ctx: ctx.get("cpi", 100) > threshold,
            description=f"CPI > {threshold} (프리미엄 포지션)"
        )

    @staticmethod
    def cpi_below(threshold: float) -> RuleCondition:
        """CPI가 임계값 이하 (가성비)"""
        return RuleCondition(
            name=f"cpi_below_{threshold}",
            check=lambda ctx: ctx.get("cpi", 100) < threshold,
            description=f"CPI < {threshold} (가성비 포지션)"
        )

    @staticmethod
    def rating_gap_negative() -> RuleCondition:
        """평점 갭이 음수 (경쟁 열위)"""
        return RuleCondition(
            name="rating_gap_negative",
            check=lambda ctx: ctx.get("rating_gap", 0) < 0,
            description="평점 갭 < 0 (경쟁사 대비 열위)"
        )

    @staticmethod
    def rating_gap_positive() -> RuleCondition:
        """평점 갭이 양수 (경쟁 우위)"""
        return RuleCondition(
            name="rating_gap_positive",
            check=lambda ctx: ctx.get("rating_gap", 0) > 0,
            description="평점 갭 > 0 (경쟁사 대비 우위)"
        )

    @staticmethod
    def has_rank_shock() -> RuleCondition:
        """순위 급변 발생"""
        return RuleCondition(
            name="has_rank_shock",
            check=lambda ctx: ctx.get("has_rank_shock", False),
            description="순위 급변 발생"
        )

    @staticmethod
    def churn_rate_high(threshold: float = 0.2) -> RuleCondition:
        """Churn Rate가 높음"""
        return RuleCondition(
            name=f"churn_rate_high_{threshold}",
            check=lambda ctx: ctx.get("churn_rate", 0) > threshold,
            description=f"Churn Rate > {threshold*100:.0f}%"
        )

    @staticmethod
    def streak_days_above(days: int) -> RuleCondition:
        """연속 체류일이 N일 이상"""
        return RuleCondition(
            name=f"streak_above_{days}",
            check=lambda ctx: ctx.get("streak_days", 0) >= days,
            description=f"Top N 연속 체류 >= {days}일"
        )

    @staticmethod
    def rank_improving() -> RuleCondition:
        """순위 상승 추세"""
        return RuleCondition(
            name="rank_improving",
            check=lambda ctx: (ctx.get("rank_change_7d") or 0) < 0,
            description="7일간 순위 상승 (음수 = 상승)"
        )

    @staticmethod
    def rank_declining() -> RuleCondition:
        """순위 하락 추세"""
        return RuleCondition(
            name="rank_declining",
            check=lambda ctx: (ctx.get("rank_change_7d") or 0) > 0,
            description="7일간 순위 하락 (양수 = 하락)"
        )

    @staticmethod
    def is_target_brand() -> RuleCondition:
        """타겟 브랜드 (LANEIGE)"""
        return RuleCondition(
            name="is_target_brand",
            check=lambda ctx: ctx.get("is_target", False) or
                             str(ctx.get("brand", "")).lower() == "laneige",
            description="타겟 브랜드 (LANEIGE)"
        )

    @staticmethod
    def has_competitors(min_count: int = 1) -> RuleCondition:
        """경쟁사가 N개 이상"""
        return RuleCondition(
            name=f"has_competitors_{min_count}",
            check=lambda ctx: ctx.get("competitor_count", 0) >= min_count,
            description=f"경쟁사 >= {min_count}개"
        )

    @staticmethod
    def in_top_n(n: int) -> RuleCondition:
        """Top N 이내"""
        return RuleCondition(
            name=f"in_top_{n}",
            check=lambda ctx: (ctx.get("current_rank") or 100) <= n,
            description=f"현재 순위 Top {n} 이내"
        )
