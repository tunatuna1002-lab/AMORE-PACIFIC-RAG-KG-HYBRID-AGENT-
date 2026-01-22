"""
Knowledge Graph Protocol
========================
Knowledge Graph에 대한 추상 인터페이스

구현체:
- KnowledgeGraph (src/ontology/knowledge_graph.py)
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class KnowledgeGraphProtocol(Protocol):
    """
    Knowledge Graph Protocol

    엔티티와 관계를 저장하고 쿼리하는 그래프 데이터베이스 인터페이스입니다.

    Methods:
        add_relation: 관계(Triple) 추가
        query: 그래프 쿼리 실행
        get_entity_relations: 엔티티의 모든 관계 조회
        get_competitors: 경쟁자 관계 조회
    """

    def add_relation(self, relation: Any) -> bool:
        """
        관계(Triple)를 그래프에 추가합니다.

        Args:
            relation: Relation 객체 (subject, predicate, object)

        Returns:
            추가 성공 여부
        """
        ...

    def add_relations(self, relations: List[Any]) -> int:
        """
        여러 관계를 그래프에 추가합니다.

        Args:
            relations: Relation 객체 목록

        Returns:
            성공적으로 추가된 관계 수
        """
        ...

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[Any] = None,
        object: Optional[str] = None
    ) -> List[Any]:
        """
        그래프를 쿼리합니다.

        Args:
            subject: 주체 필터 (선택)
            predicate: 관계 타입 필터 (선택)
            object: 객체 필터 (선택)

        Returns:
            매칭되는 Relation 목록
        """
        ...

    def get_entity_relations(
        self,
        entity: str,
        relation_types: Optional[List[Any]] = None
    ) -> List[Any]:
        """
        특정 엔티티와 관련된 모든 관계를 조회합니다.

        Args:
            entity: 엔티티 이름
            relation_types: 필터할 관계 타입 목록 (선택)

        Returns:
            관련 Relation 목록
        """
        ...

    def get_competitors(
        self,
        brand: str,
        category: Optional[str] = None
    ) -> List[str]:
        """
        특정 브랜드의 경쟁자를 조회합니다.

        Args:
            brand: 브랜드명
            category: 카테고리 필터 (선택)

        Returns:
            경쟁 브랜드 목록
        """
        ...

    def get_brand_products(self, brand: str) -> List[str]:
        """
        특정 브랜드의 제품 목록을 조회합니다.

        Args:
            brand: 브랜드명

        Returns:
            제품 ASIN 목록
        """
        ...

    def get_category_brands(self, category_id: str) -> List[str]:
        """
        특정 카테고리의 브랜드 목록을 조회합니다.

        Args:
            category_id: 카테고리 ID

        Returns:
            브랜드명 목록
        """
        ...

    def clear(self) -> None:
        """그래프의 모든 데이터를 삭제합니다."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """
        그래프 통계를 반환합니다.

        Returns:
            통계 딕셔너리 {"triple_count": int, "entity_count": int, ...}
        """
        ...
