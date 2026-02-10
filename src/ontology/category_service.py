"""
Category Service
================
KnowledgeGraph에서 분리된 카테고리 관리 서비스

책임:
- 카테고리 계층 구조 로드
- 카테고리 계층 조회
- 제품-카테고리 컨텍스트 조회
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.domain.entities import Relation, RelationType

if TYPE_CHECKING:
    from src.ontology.knowledge_graph import KnowledgeGraph


class CategoryService:
    """
    카테고리 계층 관리 서비스

    KnowledgeGraph의 카테고리 관련 로직을 분리하여 SRP 준수.
    KnowledgeGraph를 composition으로 사용합니다.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Args:
            knowledge_graph: 의존할 KnowledgeGraph 인스턴스
        """
        self._kg = knowledge_graph

    def load_hierarchy(self, hierarchy_path: str = "config/category_hierarchy.json") -> int:
        """
        카테고리 계층 구조 로드

        Args:
            hierarchy_path: 카테고리 계층 JSON 파일 경로

        Returns:
            추가된 관계 수
        """
        path = Path(hierarchy_path)
        if not path.exists():
            # 상대 경로로 시도
            base_path = Path(__file__).parent.parent.parent
            path = base_path / hierarchy_path

        if not path.exists():
            print(f"Warning: Category hierarchy file not found: {hierarchy_path}")
            return 0

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        added = 0
        categories = data.get("categories", {})

        for cat_id, cat_data in categories.items():
            # 카테고리 메타데이터 설정
            self._kg.set_entity_metadata(
                cat_id,
                {
                    "type": "category",
                    "name": cat_data.get("name", ""),
                    "amazon_node_id": cat_data.get("amazon_node_id", ""),
                    "level": cat_data.get("level", 0),
                    "parent_id": cat_data.get("parent_id"),
                    "path": cat_data.get("path", []),
                    "url": cat_data.get("url", ""),
                    "children": cat_data.get("children", []),
                },
            )

            # 부모-자식 관계 추가
            parent_id = cat_data.get("parent_id")
            if parent_id:
                # 자식 → 부모 관계
                rel_parent = Relation(
                    subject=cat_id,
                    predicate=RelationType.PARENT_CATEGORY,
                    object=parent_id,
                    properties={
                        "child_name": cat_data.get("name", ""),
                        "child_level": cat_data.get("level", 0),
                    },
                    source="config",
                )
                if self._kg.add_relation(rel_parent):
                    added += 1

                # 부모 → 자식 관계
                rel_child = Relation(
                    subject=parent_id,
                    predicate=RelationType.HAS_SUBCATEGORY,
                    object=cat_id,
                    properties={
                        "child_name": cat_data.get("name", ""),
                        "child_level": cat_data.get("level", 0),
                    },
                    source="config",
                )
                if self._kg.add_relation(rel_child):
                    added += 1

        return added

    def get_hierarchy(self, category_id: str) -> dict[str, Any]:
        """
        카테고리의 전체 계층 정보 조회

        Args:
            category_id: 카테고리 ID

        Returns:
            {
                "category": category_id,
                "name": str,
                "level": int,
                "path": List[str],
                "ancestors": List[Dict],  # 상위 카테고리들
                "descendants": List[Dict]  # 하위 카테고리들
            }
        """
        metadata = self._kg.get_entity_metadata(category_id)
        if not metadata or metadata.get("type") != "category":
            return {"category": category_id, "error": "Category not found"}

        result = {
            "category": category_id,
            "name": metadata.get("name", ""),
            "level": metadata.get("level", 0),
            "path": metadata.get("path", []),
            "ancestors": [],
            "descendants": [],
        }

        # 상위 카테고리 탐색
        current = category_id
        while True:
            parent_rels = self._kg.query(subject=current, predicate=RelationType.PARENT_CATEGORY)
            if not parent_rels:
                break
            parent_id = parent_rels[0].object
            parent_meta = self._kg.get_entity_metadata(parent_id)
            result["ancestors"].append(
                {
                    "id": parent_id,
                    "name": parent_meta.get("name", ""),
                    "level": parent_meta.get("level", 0),
                }
            )
            current = parent_id

        # 하위 카테고리 탐색 (1단계만)
        child_rels = self._kg.query(subject=category_id, predicate=RelationType.HAS_SUBCATEGORY)
        for rel in child_rels:
            child_id = rel.object
            child_meta = self._kg.get_entity_metadata(child_id)
            result["descendants"].append(
                {
                    "id": child_id,
                    "name": child_meta.get("name", ""),
                    "level": child_meta.get("level", 0),
                }
            )

        return result

    def get_product_category_context(self, product_asin: str) -> dict[str, Any]:
        """
        제품의 카테고리 컨텍스트 조회 (계층 포함)

        Args:
            product_asin: 제품 ASIN

        Returns:
            {
                "asin": str,
                "categories": [
                    {
                        "category_id": str,
                        "name": str,
                        "rank": int,
                        "hierarchy": {...}
                    }
                ]
            }
        """
        result = {"asin": product_asin, "categories": []}

        # 제품의 카테고리별 순위 조회
        rank_rels = self._kg.query(subject=product_asin, predicate=RelationType.RANKED_IN)

        for rel in rank_rels:
            cat_id = rel.object
            cat_meta = self._kg.get_entity_metadata(cat_id)
            hierarchy = self.get_hierarchy(cat_id)

            result["categories"].append(
                {
                    "category_id": cat_id,
                    "name": cat_meta.get("name", "") if cat_meta else "",
                    "rank": rel.properties.get("rank", 0),
                    "hierarchy": hierarchy,
                }
            )

        return result
