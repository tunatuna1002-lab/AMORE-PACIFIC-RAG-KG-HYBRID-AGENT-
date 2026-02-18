"""
KG IRI Mixin
=============
IRI (Internationalized Resource Identifier) support methods extracted from KnowledgeGraph.

This mixin provides:
- D-1: IRI-aware query resolution (_resolve_id, query_iri)
- D-2: IRI migration and export (migrate_to_iri, export_as_iri, _detect_entity_type)

The mixin accesses `self.triples`, `self.subject_index`, `self.object_index`,
`self.predicate_index`, `self.entity_metadata`, `self.query()`, and `self._dirty`
from the main KnowledgeGraph class.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relations import Relation, RelationType

logger = logging.getLogger(__name__)


class KGIRIMixin:
    """Mixin providing IRI support for KnowledgeGraph."""

    # =========================================================================
    # IRI Support (D-1)
    # =========================================================================

    @staticmethod
    def _resolve_id(entity_id: str) -> str:
        """Strip IRI prefix to get bare entity ID string.

        If the value is an IRI (amore:brand/LANEIGE or full URI),
        extract and return the bare ID. Otherwise return as-is.
        """
        from src.domain.entities.relations import IRI

        if IRI.is_iri(entity_id):
            return IRI.extract_id(entity_id)
        return entity_id

    def query_iri(
        self,
        subject_iri: str | None = None,
        predicate: "RelationType | None" = None,
        object_iri: str | None = None,
        min_confidence: float = 0.0,
    ) -> "list[Relation]":
        """Query triples with IRI-aware subject/object resolution.

        Resolves IRI prefixes before delegating to self.query().

        Args:
            subject_iri: Subject (plain ID or IRI)
            predicate: Relation type
            object_iri: Object (plain ID or IRI)
            min_confidence: Minimum confidence threshold

        Returns:
            Matching relations
        """
        subject = self._resolve_id(subject_iri) if subject_iri else None
        object_ = self._resolve_id(object_iri) if object_iri else None
        return self.query(
            subject=subject,
            predicate=predicate,
            object_=object_,
            min_confidence=min_confidence,
        )

    # =========================================================================
    # IRI Migration (D-2)
    # =========================================================================

    @staticmethod
    def _detect_entity_type(entity_id: str) -> str:
        """Detect entity type from an entity ID string.

        Heuristics:
        - ASIN pattern (B0 + alphanumeric, 10 chars): "product"
        - Known category IDs: "category"
        - ALL CAPS or known brand patterns: "brand"
        - Default: "entity"
        """
        import re

        # Already an IRI - extract type
        from src.domain.entities.relations import IRI

        if IRI.is_iri(entity_id):
            entity_type, _ = IRI.from_iri(entity_id)
            return entity_type

        # ASIN pattern: starts with B0 and is 10 alphanumeric chars
        if re.match(r"^B0[A-Z0-9]{8}$", entity_id):
            return "product"

        # Known category IDs
        known_categories = {
            "beauty",
            "skin_care",
            "lip_care",
            "lip_makeup",
            "face_powder",
            "face_makeup",
            "makeup",
            "11060451",
            "3761351",
            "11059031",
            "11058971",
        }
        if entity_id.lower() in known_categories:
            return "category"

        # All uppercase (2+ chars) -> brand
        if len(entity_id) >= 2 and entity_id.isupper():
            return "brand"

        # Mixed case with capital start, no underscores -> brand
        if entity_id[0:1].isupper() and "_" not in entity_id and entity_id.isalpha():
            return "brand"

        return "entity"

    def migrate_to_iri(self) -> dict[str, int]:
        """Migrate all triple subject/object values to IRI form.

        Iterates all triples, converts subject/object to IRI format,
        and rebuilds indices.

        Returns:
            {"converted": N, "skipped": N}
        """
        from src.domain.entities.relations import IRI

        converted = 0
        skipped = 0

        for rel in self.triples:
            changed = False

            # Convert subject
            if not IRI.is_iri(rel.subject):
                entity_type = self._detect_entity_type(rel.subject)
                if entity_type != "entity":
                    new_subject = IRI.to_iri(entity_type, rel.subject)
                    # Remove from old index
                    if rel in self.subject_index.get(rel.subject, []):
                        self.subject_index[rel.subject].remove(rel)
                    rel.subject = new_subject
                    # Add to new index
                    self.subject_index[rel.subject].append(rel)
                    changed = True

            # Convert object
            if not IRI.is_iri(rel.object):
                entity_type = self._detect_entity_type(rel.object)
                if entity_type != "entity":
                    new_object = IRI.to_iri(entity_type, rel.object)
                    # Remove from old index
                    if rel in self.object_index.get(rel.object, []):
                        self.object_index[rel.object].remove(rel)
                    rel.object = new_object
                    # Add to new index
                    self.object_index[rel.object].append(rel)
                    changed = True

            if changed:
                converted += 1
            else:
                skipped += 1

        self._dirty = True
        logger.info(f"IRI migration: converted={converted}, skipped={skipped}")
        return {"converted": converted, "skipped": skipped}

    def export_as_iri(self) -> dict:
        """Export KG data with all entity IDs in IRI format.

        Returns same structure as save() but with IRI entity IDs.
        Does NOT modify the in-memory graph.
        """
        from src.domain.entities.relations import IRI

        iri_triples = []
        for rel in self.triples:
            triple_dict = rel.to_dict()

            # Convert subject
            if not IRI.is_iri(triple_dict["subject"]):
                entity_type = self._detect_entity_type(triple_dict["subject"])
                if entity_type != "entity":
                    triple_dict["subject"] = IRI.to_iri(entity_type, triple_dict["subject"])

            # Convert object
            if not IRI.is_iri(triple_dict["object"]):
                entity_type = self._detect_entity_type(triple_dict["object"])
                if entity_type != "entity":
                    triple_dict["object"] = IRI.to_iri(entity_type, triple_dict["object"])

            iri_triples.append(triple_dict)

        return {
            "version": "3.0-iri",
            "triples": iri_triples,
            "entity_metadata": self.entity_metadata,
            "stats": {
                "total_triples": len(self.triples),
                "unique_subjects": len(self.subject_index),
                "unique_objects": len(self.object_index),
            },
            "exported_at": datetime.now().isoformat(),
        }
