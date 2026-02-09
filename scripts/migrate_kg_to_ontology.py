#!/usr/bin/env python3
"""
KG → OntologyKnowledgeGraph 마이그레이션 스크립트
================================================
기존 KnowledgeGraph의 JSON 데이터를 OntologyKnowledgeGraph로 마이그레이션.
OWL 스키마 검증을 통해 데이터 품질 향상.

Usage:
    python scripts/migrate_kg_to_ontology.py [--kg-path data/kg_data.json] [--dry-run]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.ontology_knowledge_graph import OntologyKnowledgeGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("migrate")


async def migrate(kg_path: str, dry_run: bool = False):
    """
    KG 데이터를 OntologyKnowledgeGraph로 마이그레이션

    Args:
        kg_path: KG JSON 파일 경로
        dry_run: True면 실제 저장하지 않음
    """
    stats = {
        "total_relations": 0,
        "migrated": 0,
        "validation_passed": 0,
        "validation_warnings": 0,
        "errors": 0,
        "skipped": 0,
    }

    # 1. 기존 KG 로드
    logger.info(f"Loading KG from: {kg_path}")
    source_kg = KnowledgeGraph(auto_load=False)  # 자동 로드 비활성화

    kg_file = Path(kg_path)
    if kg_file.exists():
        try:
            with open(kg_file, encoding="utf-8") as f:
                data = json.load(f)

            version = data.get("version", "1.0")
            logger.info(f"Loading KG version {version}")

            # 트리플 데이터 직접 로드 (add_relation 사용하지 않음)
            from src.ontology.relations import Relation

            for triple_dict in data.get("triples", []):
                try:
                    relation = Relation.from_dict(triple_dict)
                    source_kg.triples.append(relation)
                except Exception as e:
                    logger.warning(f"Failed to parse relation: {e}")

            # 엔티티 메타데이터 로드
            source_kg.entity_metadata = data.get("entity_metadata", {})

            logger.info(f"Loaded {len(source_kg.triples)} triples from source KG")
        except Exception as e:
            logger.error(f"Failed to load KG: {e}")
            return stats
    else:
        logger.error(f"KG file not found: {kg_path}")
        return stats

    stats["total_relations"] = len(source_kg.triples)

    # 2. OntologyKnowledgeGraph 초기화
    logger.info("Initializing OntologyKnowledgeGraph...")
    target_kg = KnowledgeGraph(auto_load=False) if not dry_run else None
    ontology_kg = OntologyKnowledgeGraph(
        knowledge_graph=target_kg if target_kg else source_kg,
        enable_validation=True,
    )
    await ontology_kg.initialize()

    if dry_run:
        logger.info("[DRY RUN] No changes will be made")

    # 3. 관계 마이그레이션
    logger.info("Starting migration...")

    for i, rel in enumerate(source_kg.triples):
        try:
            if not dry_run:
                success, message = ontology_kg.add_validated_relation(
                    subject=rel.subject,
                    predicate=rel.predicate,
                    obj=rel.object,
                    metadata=rel.properties,
                )

                if success:
                    if "Validated" in message:
                        stats["validation_passed"] += 1
                    elif "warning" in message.lower():
                        stats["validation_warnings"] += 1
                else:
                    stats["validation_warnings"] += 1

            stats["migrated"] += 1

        except Exception as e:
            stats["errors"] += 1
            logger.warning(
                f"Error migrating relation ({rel.subject}, {rel.predicate.value}, {rel.object}): {e}"
            )

        # 진행률 표시
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{len(source_kg.triples)}")

    # 4. 엔티티 메타데이터 마이그레이션
    if not dry_run and source_kg.entity_metadata:
        logger.info("Migrating entity metadata...")
        for entity, meta in source_kg.entity_metadata.items():
            target_kg.set_entity_metadata(entity, meta)
        logger.info(f"  Migrated {len(source_kg.entity_metadata)} entity metadata entries")

    # 5. OWL 동기화
    if not dry_run and stats["migrated"] > 0:
        logger.info("Syncing OWL inferences...")
        try:
            sync_result = ontology_kg.sync_owl_inferences()
            logger.info(
                f"  OWL sync: added={sync_result.get('added', 0)}, "
                f"skipped={sync_result.get('skipped', 0)}, "
                f"errors={sync_result.get('errors', 0)}"
            )
        except Exception as e:
            logger.warning(f"OWL sync failed: {e}")

    # 6. 일관성 체크
    if not dry_run:
        logger.info("Checking consistency...")
        try:
            consistency = ontology_kg.check_consistency()
            issues = consistency.get("issues", [])
            warnings = consistency.get("warnings", [])

            if not issues:
                logger.info("  Consistency: OK")
            else:
                logger.warning(f"  Consistency: {len(issues)} issues found")
                for issue in issues[:5]:
                    logger.warning(f"    Issue: {issue}")

            if warnings:
                logger.info(f"  {len(warnings)} warnings:")
                for warning in warnings[:5]:
                    logger.info(f"    Warning: {warning}")

            # 통계 출력
            cons_stats = consistency.get("stats", {})
            logger.info(f"  Classified entities: {cons_stats.get('classified_entities', 0)}")

        except Exception as e:
            logger.warning(f"Consistency check failed: {e}")

    # 7. 저장 (dry-run 아닐 경우)
    if not dry_run and target_kg:
        logger.info("Saving migrated KG...")
        try:
            save_success = target_kg.save(force=True)
            if save_success:
                logger.info(f"  Saved to {target_kg.persist_path}")
            else:
                logger.warning("  Save failed or skipped")
        except Exception as e:
            logger.error(f"Failed to save migrated KG: {e}")

    # 8. 결과 보고
    logger.info("=" * 50)
    logger.info("Migration Complete!")
    logger.info(f"  Total relations:      {stats['total_relations']}")
    logger.info(f"  Migrated:            {stats['migrated']}")
    logger.info(f"  Validation passed:   {stats['validation_passed']}")
    logger.info(f"  Validation warnings: {stats['validation_warnings']}")
    logger.info(f"  Errors:              {stats['errors']}")
    logger.info(f"  Skipped:             {stats['skipped']}")
    if dry_run:
        logger.info("  [DRY RUN - no changes made]")
    else:
        success_rate = (
            (stats["validation_passed"] / stats["migrated"] * 100) if stats["migrated"] > 0 else 0
        )
        logger.info(f"  Validation success rate: {success_rate:.1f}%")
    logger.info("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate KG data to OntologyKnowledgeGraph")
    parser.add_argument(
        "--kg-path",
        default="data/knowledge_graph.json",
        help="Path to KG JSON file (default: data/knowledge_graph.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without writing",
    )

    args = parser.parse_args()

    stats = asyncio.run(migrate(args.kg_path, args.dry_run))

    # Exit code: 0 if no errors, 1 if errors
    sys.exit(0 if stats.get("errors", 0) == 0 else 1)


if __name__ == "__main__":
    main()
