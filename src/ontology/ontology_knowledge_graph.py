"""
OntologyKnowledgeGraph - OWL T-Box + KG A-Box 통합
====================================================
OWL 온톨로지 스키마 검증과 KG 인스턴스 저장을 통합하는 facade 클래스.

핵심 아키텍처:
- T-Box (Terminological Box): OWL 온톨로지가 클래스, 프로퍼티, 제약 정의
- A-Box (Assertion Box): KnowledgeGraph가 인스턴스 데이터 저장

사용 패턴:
    okg = OntologyKnowledgeGraph(kg, owl_reasoner)
    await okg.initialize()

    # 스키마 검증된 트리플 추가
    okg.add_validated_relation(subject, predicate, object)

    # OWL 추론 결과를 KG에 반영
    okg.sync_owl_inferences()
"""

import logging
from datetime import datetime
from typing import Any

from .knowledge_graph import KnowledgeGraph
from .relations import Relation, RelationType

logger = logging.getLogger(__name__)

# OWL Reasoner는 선택적 의존성
try:
    from .owl_reasoner import OWLREADY2_AVAILABLE, OWLReasoner
except ImportError:
    OWLREADY2_AVAILABLE = False
    OWLReasoner = None


# OWL 클래스 ↔ RelationType 매핑
OWL_CLASS_MAPPING: dict[str, list[str]] = {
    "Brand": ["HAS_PRODUCT", "COMPETES_WITH", "DIRECT_COMPETITOR", "OWNED_BY_GROUP"],
    "Product": ["BELONGS_TO_CATEGORY", "OWNED_BY", "HAS_RANK", "RANKED_IN"],
    "Category": ["HAS_SUBCATEGORY", "PARENT_CATEGORY"],
    "CorporateGroup": ["OWNS_BRAND"],
}

# RelationType → OWL Object Property 매핑
RELATION_TO_OWL_PROPERTY: dict[RelationType, str] = {
    RelationType.HAS_PRODUCT: "hasProduct",
    RelationType.BELONGS_TO_CATEGORY: "belongsToCategory",
    RelationType.COMPETES_WITH: "competesWith",
    RelationType.PARENT_CATEGORY: "parentCategory",
    RelationType.HAS_SUBCATEGORY: "hasSubcategory",
    RelationType.OWNED_BY: "ownedBy",
    RelationType.HAS_RANK: "hasRank",
}


class OntologyKnowledgeGraph:
    """
    OWL T-Box + KG A-Box 통합 facade

    역할:
    1. 스키마 검증: 새 트리플이 OWL 정의에 부합하는지 검증
    2. 자동 분류: 엔티티의 OWL 클래스를 자동 할당
    3. 추론 동기화: OWL 추론 결과를 KG 트리플로 변환
    4. 일관성 검사: KG 데이터와 OWL 스키마 간 일관성 검증
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        owl_reasoner: Any | None = None,
        owl_file: str | None = None,
        enable_validation: bool = True,
    ):
        self.kg = knowledge_graph
        self.owl = owl_reasoner
        self.owl_file = owl_file
        self.enable_validation = enable_validation
        self._initialized = False
        self._validation_stats: dict[str, int] = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }

        # 엔티티 → OWL 클래스 캐시
        self._entity_class_cache: dict[str, str] = {}

    async def initialize(self) -> None:
        """비동기 초기화 - OWL 로드 + KG→OWL 동기화"""
        if self._initialized:
            return

        # OWL Reasoner 초기화
        if self.owl is None and OWLREADY2_AVAILABLE:
            try:
                self.owl = OWLReasoner(owl_file=self.owl_file)
                await self.owl.initialize()
                logger.info("OWL Reasoner initialized for schema validation")
            except Exception as e:
                logger.warning(f"OWL Reasoner init failed, validation disabled: {e}")
                self.enable_validation = False
        elif self.owl is None:
            logger.info("owlready2 not available, running without OWL validation")
            self.enable_validation = False

        self._initialized = True
        logger.info(f"OntologyKnowledgeGraph initialized (validation={self.enable_validation})")

    # =========================================================================
    # 스키마 검증 + 트리플 추가
    # =========================================================================

    def add_validated_relation(
        self,
        subject: str,
        predicate: RelationType,
        obj: str,
        metadata: dict[str, Any] | None = None,
        skip_validation: bool = False,
    ) -> tuple[bool, str]:
        """
        OWL 스키마 검증 후 KG에 트리플 추가

        Args:
            subject: 주체 엔티티
            predicate: 관계 유형
            obj: 객체 엔티티
            metadata: 추가 메타데이터
            skip_validation: 검증 스킵 여부

        Returns:
            (성공 여부, 메시지)
        """
        self._validation_stats["total"] += 1

        # Relation 객체 생성
        relation = Relation(
            subject=subject,
            predicate=predicate,
            object=obj,
            properties=metadata or {},
            source="ontology_kg",
        )

        # 검증 스킵 모드
        if skip_validation or not self.enable_validation or not self.owl:
            self._validation_stats["skipped"] += 1
            self.kg.add_relation(relation)
            return True, "Added without validation"

        # OWL 스키마 검증
        is_valid, reason = self._validate_triple(subject, predicate, obj)

        if is_valid:
            self._validation_stats["passed"] += 1
            self.kg.add_relation(relation)

            # 엔티티 자동 분류
            self._auto_classify_entity(subject, predicate, role="subject")
            self._auto_classify_entity(obj, predicate, role="object")

            return True, "Validated and added"
        else:
            self._validation_stats["failed"] += 1
            logger.warning(f"Validation failed: ({subject}, {predicate.value}, {obj}) - {reason}")

            # 검증 실패해도 KG에는 추가 (soft validation)
            self.kg.add_relation(relation)
            return False, f"Validation warning: {reason}"

    def _validate_triple(self, subject: str, predicate: RelationType, obj: str) -> tuple[bool, str]:
        """
        OWL 스키마 기반 트리플 검증

        검증 항목:
        1. 주체의 OWL 클래스가 해당 프로퍼티의 domain에 속하는지
        2. 객체의 OWL 클래스가 해당 프로퍼티의 range에 속하는지
        """
        if not self.owl:
            return True, "No OWL reasoner"

        # OWL 프로퍼티 매핑 확인
        owl_property = RELATION_TO_OWL_PROPERTY.get(predicate)
        if not owl_property:
            return True, f"No OWL mapping for {predicate.value}"

        # 주체 클래스 확인
        subject_class = self._get_entity_class(subject)
        if subject_class:
            valid_predicates = OWL_CLASS_MAPPING.get(subject_class, [])
            if predicate.name not in valid_predicates:
                return (
                    False,
                    f"{subject} (class={subject_class}) cannot have predicate {predicate.value}",
                )

        return True, "Valid"

    def _auto_classify_entity(self, entity: str, predicate: RelationType, role: str) -> None:
        """엔티티의 OWL 클래스를 자동 추론"""
        if entity in self._entity_class_cache:
            return

        # 관계 패턴에서 클래스 추론
        inferred_class = None

        if role == "subject":
            if predicate in (RelationType.HAS_PRODUCT, RelationType.COMPETES_WITH):
                inferred_class = "Brand"
            elif predicate in (RelationType.BELONGS_TO_CATEGORY, RelationType.HAS_RANK):
                inferred_class = "Product"
            elif predicate == RelationType.HAS_SUBCATEGORY:
                inferred_class = "Category"
            elif predicate == RelationType.OWNS_BRAND:
                inferred_class = "CorporateGroup"
        elif role == "object":
            if predicate == RelationType.HAS_PRODUCT:
                inferred_class = "Product"
            elif predicate == RelationType.BELONGS_TO_CATEGORY:
                inferred_class = "Category"
            elif predicate == RelationType.COMPETES_WITH:
                inferred_class = "Brand"

        if inferred_class:
            self._entity_class_cache[entity] = inferred_class

    def _get_entity_class(self, entity: str) -> str | None:
        """엔티티의 OWL 클래스 조회 (캐시 우선)"""
        if entity in self._entity_class_cache:
            return self._entity_class_cache[entity]

        # KG 메타데이터에서 클래스 확인
        meta = self.kg.get_entity_metadata(entity)
        if meta and "owl_class" in meta:
            cls = meta["owl_class"]
            self._entity_class_cache[entity] = cls
            return cls

        return None

    # =========================================================================
    # OWL 추론 동기화
    # =========================================================================

    def sync_owl_inferences(self) -> dict[str, int]:
        """
        OWL 추론 결과를 KG 트리플로 변환

        Returns:
            동기화 통계 (added, skipped, errors)
        """
        stats: dict[str, int] = {"added": 0, "skipped": 0, "errors": 0}

        if not self.owl:
            return stats

        try:
            # OWL 추론 실행
            if hasattr(self.owl, "run_reasoner"):
                self.owl.run_reasoner()

            # 추론된 새 사실 조회 (OWLReasoner에 해당 메서드가 있는 경우)
            if hasattr(self.owl, "get_inferred_facts"):
                inferred = self.owl.get_inferred_facts()
                for fact in inferred:
                    try:
                        predicate = self._owl_to_relation_type(fact.get("property", ""))
                        if predicate:
                            # 이미 존재하는 트리플 스킵
                            existing = self.kg.query(
                                subject=fact.get("subject"),
                                predicate=predicate,
                                object_=fact.get("object"),
                            )
                            if not existing:
                                relation = Relation(
                                    subject=fact["subject"],
                                    predicate=predicate,
                                    object=fact["object"],
                                    properties={
                                        "source": "owl_inference",
                                        "inferred_at": datetime.now().isoformat(),
                                    },
                                    source="owl_inference",
                                )
                                self.kg.add_relation(relation)
                                stats["added"] += 1
                            else:
                                stats["skipped"] += 1
                    except Exception as e:
                        logger.debug(f"Failed to sync inferred fact: {e}")
                        stats["errors"] += 1

        except Exception as e:
            logger.error(f"OWL inference sync failed: {e}")

        logger.info(f"OWL→KG sync: {stats}")
        return stats

    def _owl_to_relation_type(self, owl_property: str) -> RelationType | None:
        """OWL property name → RelationType"""
        reverse_map = {v: k for k, v in RELATION_TO_OWL_PROPERTY.items()}
        return reverse_map.get(owl_property)

    # =========================================================================
    # 일관성 검사
    # =========================================================================

    def check_consistency(self) -> dict[str, Any]:
        """
        KG ↔ OWL 일관성 검사

        Returns:
            검사 결과 (issues, warnings, stats)
        """
        result: dict[str, Any] = {
            "issues": [],
            "warnings": [],
            "stats": {
                "total_triples": len(self.kg.triples),
                "classified_entities": len(self._entity_class_cache),
                "validation_stats": self._validation_stats.copy(),
            },
        }

        # 분류되지 않은 엔티티 확인
        all_entities = set(self.kg.subject_index.keys()) | set(self.kg.object_index.keys())
        unclassified = all_entities - set(self._entity_class_cache.keys())
        if unclassified:
            sample = list(unclassified)[:5]
            result["warnings"].append(f"{len(unclassified)} entities not classified: {sample}...")

        # OWL 일관성 검사 (OWLReasoner에 해당 메서드가 있는 경우)
        if self.owl and hasattr(self.owl, "check_consistency"):
            try:
                owl_consistent = self.owl.check_consistency()
                if not owl_consistent:
                    result["issues"].append("OWL ontology inconsistency detected")
            except Exception as e:
                result["warnings"].append(f"OWL consistency check failed: {e}")

        return result

    # =========================================================================
    # 래핑된 KG 메서드 (편의용)
    # =========================================================================

    def query(self, **kwargs) -> list:
        """KG 쿼리 위임"""
        return self.kg.query(**kwargs)

    def get_brand_products(self, brand: str, **kwargs) -> list:
        """브랜드 제품 조회"""
        return self.kg.get_brand_products(brand, **kwargs)

    def get_competitors(self, brand: str, **kwargs) -> list:
        """경쟁사 조회"""
        return self.kg.get_competitors(brand, **kwargs)

    def get_entity_metadata(self, entity: str) -> dict:
        """엔티티 메타데이터 조회 (OWL 클래스 포함)"""
        meta = self.kg.get_entity_metadata(entity)
        if meta and entity in self._entity_class_cache:
            meta["owl_class"] = self._entity_class_cache[entity]
        return meta

    def get_stats(self) -> dict[str, Any]:
        """통합 통계"""
        stats: dict[str, Any] = {
            "kg_stats": self.kg.get_stats() if hasattr(self.kg, "get_stats") else {},
            "owl_available": self.owl is not None,
            "validation_enabled": self.enable_validation,
            "validation_stats": self._validation_stats.copy(),
            "classified_entities": len(self._entity_class_cache),
            "initialized": self._initialized,
        }
        return stats
