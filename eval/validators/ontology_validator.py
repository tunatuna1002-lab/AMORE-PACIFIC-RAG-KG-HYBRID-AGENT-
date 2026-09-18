"""
Ontology Validator
==================
Validates ontology constraints including:
- Type consistency: (subject_type, predicate, object_type) matrix
- Cardinality constraints
- Required properties
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# [2026-09 사후] O0-A: 타입 등록부·술어 시그니처·규칙 발화 제약 검사
# =============================================================================
# 골든셋에는 엔티티 타입이 없다(233문항 전부). 그래서 L4 type_consistency_rate는
# expected_types 없이 호출돼 항상 1.0이었다. 아래 등록부는 지금 쓸 수 있는 원본
# (config/brands.json·config/category_hierarchy.json, 읽기 전용)에서 타입을 끌어온다.
# O1의 온톨로지 로더(src.ontology.ontology.get_ontology)가 들어오면 그쪽이 우선한다
# (entity_type(name) -> str | None 을 제공할 때만. 없으면 등록부로 떨어진다).

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 가짜 브랜드 — kg_enricher가 브랜드 추출에 실패한 제품을 묶거나(unknown), 제품 제목의
# 일반 단어를 브랜드로 오인한 것(fresh: "Fresh & Cozy", chi: "KimChiChic"). 검토 보고서 §3.2.
PLACEHOLDER_BRANDS: frozenset[str] = frozenset({"unknown", "fresh", "chi"})

# 런타임 별칭 → 정식 술어. 검색기는 KG의 ownedByGroup을 ownedBy로 바꿔 방출한다
# (hybrid_retriever.py·evidence_adapters.py). 시그니처 검사는 정식 이름 기준이다.
PREDICATE_ALIASES: dict[str, str] = {"ownedBy": "ownedByGroup"}

# 정식 술어 → (도메인 타입 집합, 범위 타입 집합 | None). None = 리터럴(검사 안 함).
# 계획서 §6 O1의 술어 정의를 따른다(rankedIn 도메인 Brand, belongsToCategory 도메인 Product).
# 기존 ALLOWED_RELATIONS(레거시)와 다르다: 레거시는 (product, ownedBy, brand)처럼 런타임
# 사용법과 맞지 않는 항목이 있어 그대로 두고 새 검사에는 쓰지 않는다.
PREDICATE_SIGNATURES: dict[str, tuple[frozenset[str], frozenset[str] | None]] = {
    "hasProduct": (frozenset({"brand"}), frozenset({"product"})),
    "belongsToCategory": (frozenset({"product"}), frozenset({"category"})),
    "ownedByGroup": (frozenset({"brand"}), frozenset({"corporate_group"})),
    "ownsBrand": (frozenset({"corporate_group"}), frozenset({"brand"})),
    "siblingBrand": (frozenset({"brand"}), frozenset({"brand"})),
    "hasSegment": (frozenset({"brand"}), frozenset({"segment"})),
    "originatesFrom": (frozenset({"brand"}), frozenset({"country"})),
    "acquiredIn": (frozenset({"brand"}), None),
    "rankedIn": (frozenset({"brand", "product"}), frozenset({"category"})),
    "hasSoS": (frozenset({"brand"}), frozenset({"category"})),
    "hasHHI": (frozenset({"category"}), None),
    "hasPosition": (frozenset({"brand"}), None),
    "competesWith": (frozenset({"brand"}), frozenset({"brand"})),
    "hasSubcategory": (frozenset({"category"}), frozenset({"category"})),
    "parentCategory": (frozenset({"category"}), frozenset({"category"})),
}

_EDGE_RE = re.compile(r"^\s*(.+?)\s+-([A-Za-z_][A-Za-z0-9_]*)->\s+(.+?)\s*$")
_ASIN_RE = re.compile(r"^B0[A-Z0-9]{8}$", re.IGNORECASE)
_METRIC_NAMES: frozenset[str] = frozenset(
    {"sos", "hhi", "cpi", "churn_rate", "rank_volatility", "rank_shock", "streak_days"}
)


def canonical_predicate(predicate: str) -> str:
    """런타임 별칭을 정식 술어 이름으로 바꾼다 (ownedBy → ownedByGroup)."""
    return PREDICATE_ALIASES.get(predicate, predicate)


def parse_edge(edge: Any) -> tuple[str, str, str] | None:
    """'subject -predicate-> object' 문자열을 (subject, predicate, object)로 나눈다."""
    match = _EDGE_RE.match(str(edge))
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def normalize_entity_key(name: Any) -> str:
    """대소문자·공백·하이픈 차이를 지운 비교용 키 ('Beauty of Joseon' → 'beauty_of_joseon')."""
    return re.sub(r"[\s\-_]+", "_", str(name).strip().lower()).strip("_")


def _key_variants(name: Any) -> list[str]:
    key = normalize_entity_key(name)
    variants = [key, key.replace(".", ""), key.replace("'", ""), key.replace("_", "")]
    return list(dict.fromkeys(v for v in variants if v))


class EntityTypeRegistry:
    """엔티티 이름 → 타입(brand·corporate_group·category·product·metric·segment·country·placeholder).

    원본 우선순위: (1) 온톨로지 로더(있을 때) (2) config 등록부 (3) 모양 규칙(ASIN·지표명).
    ``type_of``는 (타입, 출처) 쌍을 돌려주고, 모르면 (None, None)이다 — 모르는 엔티티는
    위반이 아니라 "검사 불가"로 센다.
    """

    def __init__(
        self,
        types: dict[str, str],
        groups: dict[str, str] | None = None,
        ontology: Any | None = None,
    ):
        self._types = dict(types)
        self._groups = dict(groups or {})
        self._ontology = ontology

    @classmethod
    def from_config(cls, config_dir: Path | str | None = None) -> "EntityTypeRegistry":
        """config/brands.json·config/category_hierarchy.json에서 등록부를 만든다 (읽기 전용)."""
        config_dir = Path(config_dir) if config_dir else _REPO_ROOT / "config"
        types: dict[str, str] = {}
        groups: dict[str, str] = {}

        def put(name: Any, entity_type: str) -> None:
            for variant in _key_variants(name):
                types.setdefault(variant, entity_type)

        try:
            brands = json.loads((config_dir / "brands.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("brands.json을 읽지 못해 브랜드 타입 없이 진행", exc_info=True)
            brands = {}

        target = brands.get("target_brand") or {}
        group_names: set[str] = set()
        if target.get("parent"):
            group_names.add(str(target["parent"]))
        for info in brands.get("brand_ownership", {}).values():
            if isinstance(info, dict) and info.get("owner"):
                group_names.add(str(info["owner"]))
        for group in group_names:
            put(group, "corporate_group")

        def add_brand(name: Any, group: str | None) -> None:
            put(name, "brand")
            if group:
                for variant in _key_variants(name):
                    groups.setdefault(variant, normalize_entity_key(group))

        if target.get("name"):
            add_brand(target["name"], target.get("parent"))
            for alias in target.get("aliases") or []:
                add_brand(alias, target.get("parent"))
        ap_group = target.get("parent") or "AMOREPACIFIC"
        for entry in brands.get("amorepacific_brands") or []:
            if isinstance(entry, dict) and entry.get("name"):
                add_brand(entry["name"], ap_group)
                for alias in entry.get("aliases") or []:
                    add_brand(alias, ap_group)
        for entry in brands.get("competitor_brands") or []:
            if isinstance(entry, dict) and entry.get("name"):
                add_brand(entry["name"], None)
        for name in brands.get("korean_brands") or []:
            add_brand(name, None)
        for name, info in (brands.get("brand_ownership") or {}).items():
            add_brand(name, info.get("owner") if isinstance(info, dict) else None)
        for segment in brands.get("segments") or {}:
            put(segment, "segment")
        for entry in brands.get("competitor_brands") or []:
            if isinstance(entry, dict) and entry.get("tier"):
                put(entry["tier"], "segment")
        for country in ("Korea", "USA", "Japan", "France", "China"):
            put(country, "country")

        try:
            hierarchy = json.loads(
                (config_dir / "category_hierarchy.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            logger.warning("category_hierarchy.json을 읽지 못함", exc_info=True)
            hierarchy = {}
        for cat_id, info in (hierarchy.get("categories") or {}).items():
            put(cat_id, "category")
            if isinstance(info, dict) and info.get("amazon_node_id"):
                put(info["amazon_node_id"], "category")

        for metric in _METRIC_NAMES:
            put(metric, "metric")
        for placeholder in PLACEHOLDER_BRANDS:
            # 가짜 브랜드는 등록부보다 우선한다 (fresh는 실제 브랜드명이기도 하지만 KG의
            # 'fresh' 노드는 제목 오인이다 — 검토 보고서 §3.2).
            for variant in _key_variants(placeholder):
                types[variant] = "placeholder"
        return cls(types, groups, ontology=_load_ontology())

    def type_of(self, entity: Any) -> tuple[str | None, str | None]:
        """(타입, 출처) — 출처는 'ontology' | 'registry' | 'pattern'. 모르면 (None, None)."""
        if entity is None or str(entity).strip() == "":
            return None, None
        if self._ontology is not None:
            try:
                resolved = self._ontology.entity_type(str(entity))
            except Exception:
                resolved = None
            if resolved:
                return str(resolved), "ontology"
        for variant in _key_variants(entity):
            if variant in self._types:
                return self._types[variant], "registry"
        if _ASIN_RE.match(str(entity).strip()):
            return "product", "pattern"
        return None, None

    def group_of(self, brand: Any) -> str | None:
        """브랜드의 소속 그룹(정규화 키). 모르면 None."""
        for variant in _key_variants(brand):
            if variant in self._groups:
                return self._groups[variant]
        return None

    @property
    def source_label(self) -> str:
        return "ontology" if self._ontology is not None else "registry"


def _load_ontology() -> Any | None:
    """O1 온톨로지 로더가 있으면 쓴다. 없거나 entity_type을 제공하지 않으면 None."""
    try:
        from src.ontology.ontology import get_ontology  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        ontology = get_ontology()
    except Exception:
        logger.warning("온톨로지 로더 호출 실패 — config 등록부로 대체", exc_info=True)
        return None
    return ontology if callable(getattr(ontology, "entity_type", None)) else None


_DEFAULT_REGISTRY: EntityTypeRegistry | None = None


def default_type_registry() -> EntityTypeRegistry:
    """프로세스당 한 번 만드는 기본 등록부."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = EntityTypeRegistry.from_config()
    return _DEFAULT_REGISTRY


def gold_edge_types(gold_edges: list[str]) -> dict[str, str]:
    """골드 엣지의 술어 시그니처로 끝점 타입을 정한다 (도메인·범위가 한 타입일 때만).

    예: 'cosrx -ownedByGroup-> amorepacific' → cosrx=brand, amorepacific=corporate_group.
    같은 엔티티가 서로 다른 타입으로 나오면 첫 값을 유지한다(골드 자체의 모순은 여기서
    판정하지 않는다).
    """
    types: dict[str, str] = {}
    for edge in gold_edges or []:
        parsed = parse_edge(edge)
        if parsed is None:
            continue
        subject, predicate, obj = parsed
        signature = PREDICATE_SIGNATURES.get(canonical_predicate(predicate))
        if signature is None:
            continue
        domain, range_ = signature
        if len(domain) == 1:
            types.setdefault(normalize_entity_key(subject), next(iter(domain)))
        if range_ is not None and len(range_) == 1:
            types.setdefault(normalize_entity_key(obj), next(iter(range_)))
    return types


class OntologyValidator:
    """
    Validates ontology constraints for KG triples and inferences.

    Uses the relation types from src/domain/entities/relations.py
    to enforce type consistency.
    """

    # Allowed relation types: (subject_type, predicate, object_type)
    # Based on RelationType enum from src/domain/entities/relations.py
    ALLOWED_RELATIONS: set[tuple[str, str, str]] = {
        # Entity Relations
        ("brand", "hasProduct", "product"),
        ("product", "belongsToCategory", "category"),
        ("product", "ownedBy", "brand"),
        ("category", "hasSubcategory", "category"),
        ("category", "parentCategory", "category"),
        # Corporate Ownership
        ("brand", "ownedByGroup", "corporate_group"),
        ("corporate_group", "ownsBrand", "brand"),
        ("brand", "siblingBrand", "brand"),
        ("brand", "hasSegment", "segment"),
        ("brand", "originatesFrom", "country"),
        # Ranking Relations
        ("product", "rankedIn", "snapshot"),
        ("product", "hasRank", "rank"),
        ("product", "ranksAbove", "product"),
        ("product", "ranksBelow", "product"),
        # Competitive Relations
        ("brand", "competesWith", "brand"),
        ("product", "competesWithProduct", "product"),
        ("brand", "directCompetitor", "brand"),
        ("brand", "indirectCompetitor", "brand"),
        # Metric Relations
        ("metric", "indicates", "insight"),
        ("metric", "correlatesWith", "metric"),
        ("metric", "influences", "metric"),
        ("insight", "requiresAction", "action"),
        # Temporal Relations
        ("snapshot", "follows", "snapshot"),
        ("snapshot", "precedes", "snapshot"),
        ("entity", "hasTrend", "trend"),
        ("product", "hasHistory", "history"),
        # State Relations
        ("entity", "hasState", "state"),
        ("product", "hasAlert", "alert"),
        ("brand", "hasPosition", "position"),
        # Sentiment Relations
        ("product", "hasAISummary", "ai_summary"),
        ("product", "hasSentiment", "sentiment_tag"),
        ("sentiment_tag", "belongsToCluster", "sentiment_cluster"),
        ("product", "similarSentiment", "product"),
        ("brand", "brandSentiment", "sentiment_profile"),
    }

    # Entity type patterns for type inference
    # NOTE: Order matters! More specific patterns should be checked first.
    ENTITY_TYPE_PATTERNS: dict[str, list[str]] = {
        "metric": [
            r"^(sos|hhi|cpi|churn_rate|rank_volatility|rank_shock|streak_days)$",
        ],
        "category": [
            r"^(lip_care|skin_care|face_powder|lip_makeup|beauty)$",
            r"^\d+$",  # Node IDs
        ],
        "product": [
            r"^B0[A-Z0-9]{8}$",  # ASIN pattern
        ],
        "brand": [
            r"^(laneige|cosrx|tirtir|innisfree|etude|sulwhasoo)$",  # Known brands
        ],
        "sentiment_tag": [
            r"^(moisturizing|hydrating|value for money|easy to use)$",
        ],
    }

    # Required properties for specific relation types
    REQUIRED_PROPERTIES: dict[str, list[str]] = {
        "hasProduct": [],
        "belongsToCategory": [],
        "hasRank": ["rank"],
        "competesWith": [],
    }

    def __init__(self):
        """Initialize validator."""
        self._type_cache: dict[str, str] = {}

    def infer_entity_type(self, entity: str) -> str:
        """
        Infer entity type from entity string.

        Args:
            entity: Entity identifier

        Returns:
            Inferred type or "unknown"
        """
        # Check cache first
        if entity in self._type_cache:
            return self._type_cache[entity]

        # Check patterns - use original entity for some patterns, lowercase for others
        for entity_type, patterns in self.ENTITY_TYPE_PATTERNS.items():
            for pattern in patterns:
                # Try both original and lowercase for maximum match coverage
                if re.match(pattern, entity, re.IGNORECASE) or re.match(
                    pattern, entity.lower(), re.IGNORECASE
                ):
                    self._type_cache[entity] = entity_type
                    return entity_type

        # Default to "entity" (generic)
        return "entity"

    def validate_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_type: str | None = None,
        object_type: str | None = None,
    ) -> tuple[bool, str]:
        """
        Validate a single triple against type constraints.

        Args:
            subject: Subject entity
            predicate: Relation predicate
            obj: Object entity
            subject_type: Override subject type inference
            object_type: Override object type inference

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Infer types if not provided
        subj_type = subject_type or self.infer_entity_type(subject)
        obj_type = object_type or self.infer_entity_type(obj)

        # Check if relation is allowed
        triple_types = (subj_type, predicate, obj_type)

        # Try exact match first
        if triple_types in self.ALLOWED_RELATIONS:
            return True, ""

        # Try with generic "entity" type
        generic_triples = [
            (subj_type, predicate, "entity"),
            ("entity", predicate, obj_type),
            ("entity", predicate, "entity"),
        ]

        for generic in generic_triples:
            if generic in self.ALLOWED_RELATIONS:
                return True, ""

        # Not found - report error
        return False, (
            f"Invalid relation: ({subj_type}, {predicate}, {obj_type}). "
            f"Subject='{subject}', Object='{obj}'"
        )

    def validate_inference(self, inference: dict[str, Any]) -> list[str]:
        """
        Validate a single inference result.

        Args:
            inference: Inference dict with rule_name, insight_type, etc.

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []

        # Check required fields
        required_fields = ["rule_name", "insight_type", "insight"]
        for field in required_fields:
            if field not in inference:
                errors.append(f"Missing required field: {field}")

        # Check confidence bounds
        confidence = inference.get("confidence", 1.0)
        if not 0.0 <= confidence <= 1.0:
            errors.append(f"Invalid confidence: {confidence} (must be 0.0-1.0)")

        # Check insight_type is valid
        valid_insight_types = {
            "market_position",
            "market_dominance",
            "market_share",
            "competitive_threat",
            "competitive_advantage",
            "competitor_movement",
            "growth_opportunity",
            "growth_momentum",
            "entry_opportunity",
            "risk_alert",
            "rank_shock",
            "rating_decline",
            "price_position",
            "price_quality_gap",
            "price_dependency",
            "brand_strength",
            "stability",
            "volatility",
            "sentiment_strength",
            "sentiment_weakness",
            "sentiment_advantage",
            "sentiment_gap",
            "customer_perception",
            "sentiment_analysis",
            "rank_analysis",
            "market_structure",
        }

        insight_type = inference.get("insight_type", "")
        if insight_type and insight_type not in valid_insight_types:
            errors.append(f"Unknown insight_type: {insight_type}")

        return errors

    def validate_inferences(self, inferences: list[dict[str, Any]]) -> list[str]:
        """
        Validate multiple inferences.

        Args:
            inferences: List of inference dicts

        Returns:
            List of all validation errors
        """
        all_errors: list[str] = []

        for i, inference in enumerate(inferences):
            errors = self.validate_inference(inference)
            for error in errors:
                all_errors.append(f"Inference [{i}]: {error}")

        return all_errors

    def validate_kg_facts(self, facts: list[dict[str, Any]]) -> list[str]:
        """
        Validate KG facts/triples.

        Args:
            facts: List of fact dicts with type, entity, data fields

        Returns:
            List of validation errors
        """
        errors: list[str] = []

        for i, fact in enumerate(facts):
            fact_type = fact.get("type", "")
            entity = fact.get("entity", "")

            # Basic validation
            if not entity:
                errors.append(f"Fact [{i}]: Missing entity")
                continue

            # Type-specific validation
            if fact_type == "brand_products":
                data = fact.get("data", {})
                if "products" in data:
                    for prod in data["products"][:5]:  # Check first 5
                        asin = prod.get("asin", prod) if isinstance(prod, dict) else prod
                        if asin and not re.match(r"^B0[A-Z0-9]{8}$", str(asin)):
                            errors.append(f"Fact [{i}]: Invalid ASIN format: {asin}")

            elif fact_type == "competitors":
                data = fact.get("data", [])
                if not isinstance(data, list):
                    errors.append(f"Fact [{i}]: competitors data should be a list")

            elif fact_type == "category_hierarchy":
                data = fact.get("data", {})
                level = data.get("level", -1)
                if level < 0 or level > 5:
                    errors.append(f"Fact [{i}]: Invalid category level: {level}")

        return errors

    def check_type_consistency(
        self,
        entities: list[str],
        expected_types: dict[str, str] | None = None,
    ) -> tuple[float, list[str]]:
        """
        Check type consistency across a set of entities.

        Args:
            entities: List of entity identifiers
            expected_types: Optional dict of entity -> expected_type

        Returns:
            Tuple of (consistency_rate, list of inconsistencies)
        """
        if not entities:
            return 1.0, []

        inconsistencies: list[str] = []
        expected_types = expected_types or {}

        for entity in entities:
            inferred_type = self.infer_entity_type(entity)
            expected_type = expected_types.get(entity)

            if expected_type and inferred_type != expected_type:
                inconsistencies.append(
                    f"Entity '{entity}': inferred={inferred_type}, expected={expected_type}"
                )

        consistency_rate = 1.0 - (len(inconsistencies) / len(entities))
        return consistency_rate, inconsistencies

    # ------------------------------------------------------------------
    # [2026-09 사후] O0-A: 트레이스 타입 일관성 (골드·등록부 타입 기준)
    # ------------------------------------------------------------------

    # ontology_facts 종류 → (엔티티 필드가 가져야 할 타입, 하위 항목 추출 규칙).
    # 엣지를 만드는 fact(metric_edges·competitors·category_hierarchy 경로)는
    # kg_edges_found에서 시그니처로 검사하므로 여기서는 노드 타입만 본다.
    _FACT_ENTITY_TYPES: dict[str, str] = {
        "brand_products": "brand",
        "brand_info": "brand",
        "competitors": "brand",
        "competitor_network": "brand",
        "metric_edges": "brand",
        "category_brands": "category",
        "category_hierarchy": "category",
    }

    def check_trace_types(
        self,
        kg_edges: list[str],
        ontology_facts: list[dict[str, Any]],
        gold_types: dict[str, str] | None = None,
        registry: EntityTypeRegistry | None = None,
    ) -> dict[str, Any]:
        """시스템이 트레이스에 낸 타입 주장이 기대 타입과 맞는지 센다.

        **검사 단위(check)** — 끝점 하나의 타입 주장 하나:
          - 방출 엣지 ``s -p-> o``: p의 정식 시그니처(PREDICATE_SIGNATURES)가 요구하는 도메인에
            s의 기대 타입이, 범위(리터럴이 아니면)에 o의 기대 타입이 드는지. 끝점당 1건.
          - ontology_facts: fact 종류가 함의하는 엔티티 타입(예: category_brands의 entity는
            category, 그 top_brands[].brand는 brand, brand_products의 products[].asin은 product).
        **기대 타입** — 골드(명시 타입 > 골드 엣지 시그니처) → 온톨로지 로더 → config 등록부 →
        모양 규칙(ASIN) 순. 어느 원본도 모르는 엔티티는 ``untyped``로 세고 검사하지 않는다.
        가짜 브랜드(unknown·fresh·chi)는 타입 'placeholder'라 어떤 도메인·범위도 만족하지 않는다.
        시그니처가 없는 술어는 ``unknown_predicates``로만 센다.

        Returns:
            {checks, consistent, violations, untyped, unknown_predicates,
             sources: {출처: 건수}, violation_kinds: {라벨: 건수}, examples: [...최대 5]}
        """
        registry = registry or default_type_registry()
        gold_types = {normalize_entity_key(k): v for k, v in (gold_types or {}).items()}
        result: dict[str, Any] = {
            "checks": 0,
            "consistent": 0,
            "violations": 0,
            "untyped": 0,
            "unknown_predicates": 0,
            "sources": Counter(),
            "violation_kinds": Counter(),
            "examples": [],
        }

        def resolve(entity: Any) -> tuple[str | None, str | None]:
            key = normalize_entity_key(entity)
            if key in gold_types:
                return gold_types[key], "gold"
            return registry.type_of(entity)

        def check(entity: Any, allowed: frozenset[str], label: str) -> None:
            if entity is None or str(entity).strip() == "":
                entity_type, source = "empty", "trace"
            else:
                entity_type, source = resolve(entity)
            if entity_type is None:
                result["untyped"] += 1
                return
            result["checks"] += 1
            result["sources"][source] += 1
            if entity_type in allowed:
                result["consistent"] += 1
                return
            result["violations"] += 1
            kind = f"{label}={entity_type}"
            result["violation_kinds"][kind] += 1
            if len(result["examples"]) < 5:
                result["examples"].append(f"{kind} ({entity})")

        for edge in kg_edges or []:
            parsed = parse_edge(edge)
            if parsed is None:
                continue
            subject, predicate, obj = parsed
            canonical = canonical_predicate(predicate)
            signature = PREDICATE_SIGNATURES.get(canonical)
            if signature is None:
                result["unknown_predicates"] += 1
                continue
            domain, range_ = signature
            check(subject, domain, f"edge:{canonical}:subject")
            if range_ is not None:
                check(obj, range_, f"edge:{canonical}:object")

        seen: set[tuple[str, str, str]] = set()

        def check_once(entity: Any, entity_type: str, label: str) -> None:
            key = (normalize_entity_key(entity), entity_type, label)
            if key in seen:
                return
            seen.add(key)
            check(entity, frozenset({entity_type}), label)

        for fact in ontology_facts or []:
            if not isinstance(fact, dict):
                continue
            fact_type = str(fact.get("type", ""))
            expected = self._FACT_ENTITY_TYPES.get(fact_type)
            if expected is None:
                continue
            check_once(fact.get("entity"), expected, f"fact:{fact_type}:entity")
            data = fact.get("data")
            if fact_type == "category_brands" and isinstance(data, dict):
                for entry in data.get("top_brands") or []:
                    if isinstance(entry, dict):
                        check_once(entry.get("brand"), "brand", "fact:category_brands:brand")
            elif fact_type == "brand_products" and isinstance(data, dict):
                for product in data.get("products") or []:
                    if isinstance(product, dict):
                        check_once(product.get("asin"), "product", "fact:brand_products:asin")
                        if product.get("category"):
                            check_once(
                                product["category"], "category", "fact:brand_products:category"
                            )
            elif fact_type == "competitors" and isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        check_once(entry.get("brand"), "brand", "fact:competitors:brand")
            elif fact_type == "category_hierarchy" and isinstance(data, dict):
                for node in (data.get("ancestors") or []) + (data.get("descendants") or []):
                    if isinstance(node, dict):
                        check_once(node.get("id"), "category", "fact:category_hierarchy:node")

        result["sources"] = dict(result["sources"])
        result["violation_kinds"] = dict(result["violation_kinds"])
        return result

    # ------------------------------------------------------------------
    # [2026-09 사후] O0-A: 규칙 발화 결과의 제약 위반
    # ------------------------------------------------------------------

    # 비율 지표는 0~1 스케일이다 (결정 S2-0: SoS 스케일 0~1로 통일).
    _RATIO_KEYS: frozenset[str] = frozenset({"sos", "hhi", "sos_change"})
    _NUMERIC_SNAPSHOT_KEYS: frozenset[str] = frozenset(
        {
            "sos",
            "hhi",
            "cpi",
            "avg_rank",
            "rank",
            "rating_gap",
            "price",
            "category_avg_price",
            "sos_change",
            "rank_change_7d",
        }
    )

    def check_rule_inferences(
        self,
        inferences: list[dict[str, Any]],
        rule_evaluation: dict[str, Any] | None = None,
        registry: EntityTypeRegistry | None = None,
    ) -> dict[str, Any]:
        """발화한 규칙 추론 하나하나가 온톨로지 제약을 어기는지 판정한다.

        **분모** — 검사한 추론 = ``trace.l4_ontology.inferences``의 항목. ``rule_evaluation``이
        있으면 그 ``fired``에 든 규칙의 추론만 본다(규칙 계약 경로에서 실제로 발화한 것).
        ``fired``에는 있는데 추론 본문이 없는 규칙은 검사할 수 없어 ``fired_unchecked``로 센다.

        **위반** — 추론 하나가 아래 중 하나라도 해당하면 위반 1건(종류는 모두 기록):
          - ``schema``: 필수 필드(rule_name·insight_type·insight) 누락, confidence가 [0,1] 밖,
            알 수 없는 insight_type (기존 ``validate_inference``와 같은 검사).
          - ``subject_type``: evidence.context_snapshot.brand가 비었거나, 등록부상 brand가 아닌
            타입(가짜 브랜드 placeholder·그룹 corporate_group·카테고리 등)이다. 등록부가 모르는
            브랜드는 검사하지 않는다.
          - ``category_type``: context_snapshot.category가 비었거나 category가 아닌 타입이다.
          - ``related_entity_invalid``: related_entities에 빈 문자열이나 가짜 브랜드가 있다.
          - ``value_range``: 비율(sos·hhi)이 [0,1] 밖, rating_gap이 [-5,5] 밖, 순위(avg_rank·
            rank)가 [1,100] 밖, 가격·CPI가 0 이하, competitor_count가 음수, 또는 숫자여야 할
            값이 숫자가 아니다.
          - ``missing_as_of``: 수치 입력(sos·hhi·cpi·순위·평점·가격 등)으로 발화했는데
            context_snapshot에 as_of가 없다(날짜 없는 수치로 결론을 냈다).
          - ``ownership_mismatch``: context_snapshot.parent_group이 등록부의 그 브랜드 소속
            그룹과 다르다 (등록부가 그룹을 모르면 검사하지 않는다).

        Returns:
            {checked, violating, fired_unchecked, kinds: {종류: 건수}, examples: [...최대 5]}
        """
        registry = registry or default_type_registry()
        fired: set[str] | None = None
        if isinstance(rule_evaluation, dict) and rule_evaluation.get("fired") is not None:
            fired = {str(name) for name in rule_evaluation.get("fired") or []}

        considered = [
            inf
            for inf in inferences or []
            if isinstance(inf, dict) and (fired is None or inf.get("rule_name") in fired)
        ]
        inferred_names = {inf.get("rule_name") for inf in considered}
        fired_unchecked = len(fired - inferred_names) if fired is not None else 0

        kinds: Counter[str] = Counter()
        examples: list[str] = []
        violating = 0
        for inference in considered:
            found = self._inference_violations(inference, registry)
            if found:
                violating += 1
                kinds.update(found)
                if len(examples) < 5:
                    examples.append(f"{inference.get('rule_name')}: {','.join(sorted(found))}")
        return {
            "checked": len(considered),
            "violating": violating,
            "fired_unchecked": fired_unchecked,
            "kinds": dict(kinds),
            "examples": examples,
        }

    def _inference_violations(
        self, inference: dict[str, Any], registry: EntityTypeRegistry
    ) -> set[str]:
        found: set[str] = set()
        if self.validate_inference(inference):
            found.add("schema")

        evidence = inference.get("evidence") if isinstance(inference.get("evidence"), dict) else {}
        snapshot = evidence.get("context_snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else {}

        if "brand" in snapshot:
            brand_type, _ = registry.type_of(snapshot.get("brand"))
            if not str(snapshot.get("brand") or "").strip() or (
                brand_type is not None and brand_type != "brand"
            ):
                found.add("subject_type")
        if "category" in snapshot:
            category_type, _ = registry.type_of(snapshot.get("category"))
            if not str(snapshot.get("category") or "").strip() or (
                category_type is not None and category_type != "category"
            ):
                found.add("category_type")

        for entity in inference.get("related_entities") or []:
            entity_type, _ = registry.type_of(entity)
            if not str(entity).strip() or entity_type == "placeholder":
                found.add("related_entity_invalid")
                break

        numeric_seen = False
        for key in self._NUMERIC_SNAPSHOT_KEYS | {"competitor_count"}:
            if key not in snapshot or snapshot[key] is None:
                continue
            value = snapshot[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                found.add("value_range")
                continue
            if key in self._NUMERIC_SNAPSHOT_KEYS:
                numeric_seen = True
            if (
                (key in {"sos", "hhi"} and not 0.0 <= value <= 1.0)
                or (key == "sos_change" and not -1.0 <= value <= 1.0)
                or (key == "rating_gap" and not -5.0 <= value <= 5.0)
                or (key in {"avg_rank", "rank"} and not 1 <= value <= 100)
                or (key in {"price", "category_avg_price", "cpi"} and value <= 0)
                or (key == "competitor_count" and value < 0)
            ):
                found.add("value_range")
        if numeric_seen and not snapshot.get("as_of"):
            found.add("missing_as_of")

        parent_group = snapshot.get("parent_group")
        if parent_group and snapshot.get("brand"):
            registered = registry.group_of(snapshot["brand"])
            if registered is not None and registered != normalize_entity_key(parent_group):
                found.add("ownership_mismatch")
        return found

    def get_stats(self) -> dict[str, Any]:
        """Get validator statistics."""
        return {
            "allowed_relations": len(self.ALLOWED_RELATIONS),
            "entity_type_patterns": len(self.ENTITY_TYPE_PATTERNS),
            "type_cache_size": len(self._type_cache),
        }
