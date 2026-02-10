"""
Ontology Validator
==================
Validates ontology constraints including:
- Type consistency: (subject_type, predicate, object_type) matrix
- Cardinality constraints
- Required properties
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


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

    def get_stats(self) -> dict[str, Any]:
        """Get validator statistics."""
        return {
            "allowed_relations": len(self.ALLOWED_RELATIONS),
            "entity_type_patterns": len(self.ENTITY_TYPE_PATTERNS),
            "type_cache_size": len(self._type_cache),
        }
