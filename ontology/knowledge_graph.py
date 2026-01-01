"""
Knowledge Graph
온톨로지 기반 지식 그래프 구현

기능:
1. 트리플(Triple) 저장 및 관리
2. 패턴 기반 쿼리
3. 그래프 탐색 (BFS/DFS)
4. 관계 추론 지원
"""

import json
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime
from pathlib import Path

from .relations import (
    Relation,
    RelationType,
    MarketPosition,
    create_brand_product_relation,
    create_product_category_relation,
    create_competition_relation
)


class KnowledgeGraph:
    """
    지식 그래프

    트리플 저장 구조:
    - triples: List[Relation] - 모든 관계
    - subject_index: Dict[str, List[Relation]] - 주체별 인덱스
    - object_index: Dict[str, List[Relation]] - 객체별 인덱스
    - predicate_index: Dict[RelationType, List[Relation]] - 관계별 인덱스

    사용 예:
        kg = KnowledgeGraph()
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
        products = kg.get_objects("LANEIGE", RelationType.HAS_PRODUCT)
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Args:
            persist_path: 영속화 경로 (JSON 파일)
        """
        self.persist_path = Path(persist_path) if persist_path else None

        # 트리플 저장소
        self.triples: List[Relation] = []

        # 인덱스 (빠른 조회용)
        self.subject_index: Dict[str, List[Relation]] = defaultdict(list)
        self.object_index: Dict[str, List[Relation]] = defaultdict(list)
        self.predicate_index: Dict[RelationType, List[Relation]] = defaultdict(list)

        # 엔티티 메타데이터
        self.entity_metadata: Dict[str, Dict[str, Any]] = {}

        # 통계
        self._stats = {
            "total_triples": 0,
            "unique_subjects": 0,
            "unique_objects": 0,
            "relations_by_type": {}
        }

        # 영속화된 데이터 로드
        if self.persist_path and self.persist_path.exists():
            self._load()

    # =========================================================================
    # 기본 CRUD 연산
    # =========================================================================

    def add_relation(self, relation: Relation) -> bool:
        """
        관계 추가

        Args:
            relation: 추가할 관계

        Returns:
            성공 여부 (중복 시 False)
        """
        # 중복 체크
        if relation in self.triples:
            # 기존 관계 업데이트 (properties, confidence 등)
            existing_idx = self.triples.index(relation)
            existing = self.triples[existing_idx]
            existing.properties.update(relation.properties)
            existing.confidence = max(existing.confidence, relation.confidence)
            return False

        # 추가
        self.triples.append(relation)

        # 인덱스 업데이트
        self.subject_index[relation.subject].append(relation)
        self.object_index[relation.object].append(relation)
        self.predicate_index[relation.predicate].append(relation)

        # 통계 업데이트
        self._update_stats()

        return True

    def add_relations(self, relations: List[Relation]) -> int:
        """
        여러 관계 일괄 추가

        Args:
            relations: 관계 리스트

        Returns:
            추가된 관계 수
        """
        added = 0
        for relation in relations:
            if self.add_relation(relation):
                added += 1
        return added

    def remove_relation(self, relation: Relation) -> bool:
        """
        관계 삭제

        Args:
            relation: 삭제할 관계

        Returns:
            성공 여부
        """
        if relation not in self.triples:
            return False

        self.triples.remove(relation)

        # 인덱스에서 제거
        if relation in self.subject_index[relation.subject]:
            self.subject_index[relation.subject].remove(relation)
        if relation in self.object_index[relation.object]:
            self.object_index[relation.object].remove(relation)
        if relation in self.predicate_index[relation.predicate]:
            self.predicate_index[relation.predicate].remove(relation)

        self._update_stats()
        return True

    def clear(self) -> None:
        """전체 초기화"""
        self.triples.clear()
        self.subject_index.clear()
        self.object_index.clear()
        self.predicate_index.clear()
        self.entity_metadata.clear()
        self._update_stats()

    # =========================================================================
    # 쿼리 연산
    # =========================================================================

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[RelationType] = None,
        object_: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Relation]:
        """
        트리플 패턴 쿼리

        Args:
            subject: 주체 (None이면 모든 주체)
            predicate: 관계 유형 (None이면 모든 관계)
            object_: 객체 (None이면 모든 객체)
            min_confidence: 최소 신뢰도

        Returns:
            매칭되는 관계 리스트

        예:
            # LANEIGE의 모든 제품 조회
            kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)

            # Lip Care 카테고리의 모든 제품
            kg.query(predicate=RelationType.BELONGS_TO_CATEGORY, object_="lip_care")
        """
        # 가장 선택적인 인덱스 선택
        if subject and subject in self.subject_index:
            candidates = self.subject_index[subject]
        elif object_ and object_ in self.object_index:
            candidates = self.object_index[object_]
        elif predicate and predicate in self.predicate_index:
            candidates = self.predicate_index[predicate]
        else:
            candidates = self.triples

        # 필터링
        results = []
        for rel in candidates:
            if subject and rel.subject != subject:
                continue
            if predicate and rel.predicate != predicate:
                continue
            if object_ and rel.object != object_:
                continue
            if rel.confidence < min_confidence:
                continue
            results.append(rel)

        return results

    def get_subjects(
        self,
        predicate: RelationType,
        object_: str
    ) -> List[str]:
        """
        특정 관계와 객체에 대한 모든 주체 조회

        예: get_subjects(HAS_PRODUCT, "B08XYZ") → ["LANEIGE"]
        """
        relations = self.query(predicate=predicate, object_=object_)
        return list(set(r.subject for r in relations))

    def get_objects(
        self,
        subject: str,
        predicate: RelationType
    ) -> List[str]:
        """
        특정 주체와 관계에 대한 모든 객체 조회

        예: get_objects("LANEIGE", HAS_PRODUCT) → ["B08XYZ", "B09ABC"]
        """
        relations = self.query(subject=subject, predicate=predicate)
        return list(set(r.object for r in relations))

    def get_predicates(
        self,
        subject: str,
        object_: str
    ) -> List[RelationType]:
        """
        두 엔티티 간의 모든 관계 유형 조회

        예: get_predicates("LANEIGE", "COSRX") → [COMPETES_WITH]
        """
        relations = self.query(subject=subject, object_=object_)
        return list(set(r.predicate for r in relations))

    def exists(
        self,
        subject: str,
        predicate: RelationType,
        object_: str
    ) -> bool:
        """관계 존재 여부 확인"""
        return len(self.query(subject, predicate, object_)) > 0

    # =========================================================================
    # 그래프 탐색
    # =========================================================================

    def get_neighbors(
        self,
        entity: str,
        direction: str = "both",
        predicate_filter: Optional[List[RelationType]] = None
    ) -> Dict[str, List[Tuple[RelationType, str]]]:
        """
        엔티티의 이웃 조회

        Args:
            entity: 대상 엔티티
            direction: "outgoing", "incoming", "both"
            predicate_filter: 특정 관계만 필터링

        Returns:
            {
                "outgoing": [(predicate, object), ...],
                "incoming": [(predicate, subject), ...]
            }
        """
        result = {"outgoing": [], "incoming": []}

        if direction in ("outgoing", "both"):
            for rel in self.subject_index.get(entity, []):
                if predicate_filter and rel.predicate not in predicate_filter:
                    continue
                result["outgoing"].append((rel.predicate, rel.object))

        if direction in ("incoming", "both"):
            for rel in self.object_index.get(entity, []):
                if predicate_filter and rel.predicate not in predicate_filter:
                    continue
                result["incoming"].append((rel.predicate, rel.subject))

        return result

    def bfs_traverse(
        self,
        start_entity: str,
        max_depth: int = 3,
        predicate_filter: Optional[List[RelationType]] = None,
        direction: str = "outgoing"
    ) -> Dict[int, List[str]]:
        """
        BFS 그래프 탐색

        Args:
            start_entity: 시작 엔티티
            max_depth: 최대 탐색 깊이
            predicate_filter: 탐색할 관계 유형
            direction: 탐색 방향

        Returns:
            {depth: [entities at this depth]}
        """
        visited: Set[str] = set()
        result: Dict[int, List[str]] = defaultdict(list)
        queue: List[Tuple[str, int]] = [(start_entity, 0)]

        while queue:
            current, depth = queue.pop(0)

            if current in visited or depth > max_depth:
                continue

            visited.add(current)
            result[depth].append(current)

            if depth < max_depth:
                neighbors = self.get_neighbors(
                    current,
                    direction=direction,
                    predicate_filter=predicate_filter
                )

                for _, neighbor in neighbors.get(direction, []):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        return dict(result)

    def find_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5
    ) -> Optional[List[Relation]]:
        """
        두 엔티티 간 경로 탐색

        Args:
            start: 시작 엔티티
            end: 종료 엔티티
            max_depth: 최대 경로 길이

        Returns:
            경로를 구성하는 관계 리스트 (없으면 None)
        """
        if start == end:
            return []

        visited: Set[str] = set()
        queue: List[Tuple[str, List[Relation]]] = [(start, [])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current in visited:
                continue

            visited.add(current)

            for rel in self.subject_index.get(current, []):
                new_path = path + [rel]

                if rel.object == end:
                    return new_path

                if rel.object not in visited:
                    queue.append((rel.object, new_path))

        return None

    # =========================================================================
    # 도메인 특화 쿼리
    # =========================================================================

    def get_brand_products(
        self,
        brand: str,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        브랜드의 제품 목록 조회

        Args:
            brand: 브랜드명
            category: 카테고리 필터 (선택)

        Returns:
            제품 정보 리스트
        """
        products = []
        relations = self.query(
            subject=brand,
            predicate=RelationType.HAS_PRODUCT
        )

        for rel in relations:
            product_asin = rel.object
            props = rel.properties

            # 카테고리 필터
            if category and props.get("category") != category:
                continue

            products.append({
                "asin": product_asin,
                "name": props.get("product_name", ""),
                "category": props.get("category", ""),
                "rank": props.get("rank"),
                **props
            })

        return products

    def get_competitors(
        self,
        brand: str,
        category: Optional[str] = None,
        competition_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        브랜드의 경쟁사 조회

        Args:
            brand: 브랜드명
            category: 카테고리 필터
            competition_type: "direct", "indirect", "all"

        Returns:
            경쟁사 정보 리스트
        """
        competitors = []

        predicates = []
        if competition_type in ("direct", "all"):
            predicates.append(RelationType.DIRECT_COMPETITOR)
        if competition_type in ("indirect", "all"):
            predicates.append(RelationType.INDIRECT_COMPETITOR)
        if competition_type == "all":
            predicates.append(RelationType.COMPETES_WITH)

        for pred in predicates:
            relations = self.query(subject=brand, predicate=pred)
            for rel in relations:
                if category and rel.properties.get("category") != category:
                    continue

                competitors.append({
                    "brand": rel.object,
                    "type": rel.predicate.value,
                    "category": rel.properties.get("category", ""),
                    **rel.properties
                })

        # 중복 제거
        seen = set()
        unique = []
        for comp in competitors:
            key = (comp["brand"], comp.get("category"))
            if key not in seen:
                seen.add(key)
                unique.append(comp)

        return unique

    def get_category_brands(
        self,
        category: str,
        min_products: int = 1
    ) -> List[Dict[str, Any]]:
        """
        카테고리 내 브랜드 목록

        Args:
            category: 카테고리 ID
            min_products: 최소 제품 수

        Returns:
            브랜드별 정보
        """
        # 카테고리에 속한 제품들
        product_relations = self.query(
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object_=category
        )

        # 제품별 브랜드 매핑
        brand_products: Dict[str, List[str]] = defaultdict(list)

        for rel in product_relations:
            product_asin = rel.subject
            # 해당 제품의 브랜드 찾기
            brand_rels = self.query(
                predicate=RelationType.HAS_PRODUCT,
                object_=product_asin
            )
            for br in brand_rels:
                brand_products[br.subject].append(product_asin)

        # 결과 구성
        results = []
        for brand, products in brand_products.items():
            if len(products) >= min_products:
                results.append({
                    "brand": brand,
                    "product_count": len(products),
                    "products": products
                })

        return sorted(results, key=lambda x: x["product_count"], reverse=True)

    def get_entity_context(
        self,
        entity: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        엔티티의 전체 컨텍스트 수집

        Args:
            entity: 엔티티 ID
            depth: 탐색 깊이

        Returns:
            {
                "entity": str,
                "relations": {...},
                "metadata": {...}
            }
        """
        context = {
            "entity": entity,
            "relations": {},
            "metadata": self.entity_metadata.get(entity, {})
        }

        # 관계별 그룹핑
        neighbors = self.get_neighbors(entity, direction="both")

        for direction, rels in neighbors.items():
            context["relations"][direction] = defaultdict(list)
            for predicate, target in rels:
                context["relations"][direction][predicate.value].append(target)

        # 재귀 탐색 (depth > 1)
        if depth > 1:
            context["connected"] = {}
            for _, rels in neighbors.items():
                for _, target in rels[:5]:  # 상위 5개만
                    if target != entity:
                        context["connected"][target] = self.get_entity_context(
                            target, depth - 1
                        )

        return context

    # =========================================================================
    # 엔티티 메타데이터
    # =========================================================================

    def set_entity_metadata(
        self,
        entity: str,
        metadata: Dict[str, Any]
    ) -> None:
        """엔티티 메타데이터 설정"""
        if entity not in self.entity_metadata:
            self.entity_metadata[entity] = {}
        self.entity_metadata[entity].update(metadata)

    def get_entity_metadata(
        self,
        entity: str
    ) -> Dict[str, Any]:
        """엔티티 메타데이터 조회"""
        return self.entity_metadata.get(entity, {})

    # =========================================================================
    # 통계 및 분석
    # =========================================================================

    def _update_stats(self) -> None:
        """통계 업데이트"""
        self._stats["total_triples"] = len(self.triples)
        self._stats["unique_subjects"] = len(self.subject_index)
        self._stats["unique_objects"] = len(self.object_index)
        self._stats["relations_by_type"] = {
            pred.value: len(rels)
            for pred, rels in self.predicate_index.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        self._update_stats()
        return self._stats.copy()

    def get_entity_degree(self, entity: str) -> Dict[str, int]:
        """
        엔티티의 차수(degree) 계산

        Returns:
            {"in_degree": int, "out_degree": int, "total": int}
        """
        in_degree = len(self.object_index.get(entity, []))
        out_degree = len(self.subject_index.get(entity, []))

        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total": in_degree + out_degree
        }

    def get_most_connected(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        가장 많이 연결된 엔티티

        Returns:
            [(entity, degree), ...]
        """
        all_entities = set(self.subject_index.keys()) | set(self.object_index.keys())
        degrees = [
            (entity, self.get_entity_degree(entity)["total"])
            for entity in all_entities
        ]
        return sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]

    # =========================================================================
    # 영속화
    # =========================================================================

    def save(self, path: Optional[str] = None) -> None:
        """
        그래프 저장

        Args:
            path: 저장 경로 (None이면 초기화 시 설정된 경로)
        """
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            raise ValueError("No persist path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "triples": [r.to_dict() for r in self.triples],
            "entity_metadata": self.entity_metadata,
            "saved_at": datetime.now().isoformat()
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """영속화된 데이터 로드"""
        if not self.persist_path or not self.persist_path.exists():
            return

        with open(self.persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for triple_dict in data.get("triples", []):
            relation = Relation.from_dict(triple_dict)
            self.add_relation(relation)

        self.entity_metadata = data.get("entity_metadata", {})

    # =========================================================================
    # 데이터 로딩 헬퍼
    # =========================================================================

    def load_from_crawl_data(
        self,
        crawl_data: Dict[str, Any]
    ) -> int:
        """
        크롤링 데이터에서 관계 로드

        Args:
            crawl_data: CrawlerAgent 결과

        Returns:
            추가된 관계 수
        """
        added = 0

        for cat_key, cat_data in crawl_data.get("categories", {}).items():
            products = cat_data.get("rank_records", [])

            brand_products: Dict[str, List[Dict]] = defaultdict(list)

            for product in products:
                brand = product.get("brand", "Unknown")
                asin = product.get("product_asin", product.get("asin", ""))
                name = product.get("title", product.get("product_name", ""))
                rank = product.get("rank")

                if not asin:
                    continue

                brand_products[brand].append(product)

                # Brand → Product 관계
                rel1 = create_brand_product_relation(
                    brand=brand,
                    product_asin=asin,
                    product_name=name,
                    category=cat_key,
                    rank=rank,
                    rating=product.get("rating"),
                    price=product.get("price")
                )
                if self.add_relation(rel1):
                    added += 1

                # Product → Category 관계
                rel2 = create_product_category_relation(
                    product_asin=asin,
                    category_id=cat_key,
                    rank=rank
                )
                if self.add_relation(rel2):
                    added += 1

            # 브랜드 간 경쟁 관계 (같은 카테고리 내)
            brands = list(brand_products.keys())
            for i, brand1 in enumerate(brands[:10]):  # Top 10 브랜드만
                for brand2 in brands[i+1:10]:
                    comp_rel = create_competition_relation(
                        brand1=brand1,
                        brand2=brand2,
                        category=cat_key,
                        competition_type="direct" if i < 5 and brands.index(brand2) < 5 else "indirect"
                    )
                    if self.add_relation(comp_rel):
                        added += 1

                    # 역방향도 추가
                    comp_rel_rev = create_competition_relation(
                        brand1=brand2,
                        brand2=brand1,
                        category=cat_key,
                        competition_type="direct" if i < 5 and brands.index(brand2) < 5 else "indirect"
                    )
                    if self.add_relation(comp_rel_rev):
                        added += 1

        return added

    def load_from_metrics_data(
        self,
        metrics_data: Dict[str, Any]
    ) -> int:
        """
        지표 데이터에서 관계 로드

        Args:
            metrics_data: MetricsAgent 결과

        Returns:
            추가된 관계 수
        """
        added = 0

        # 브랜드 메타데이터 설정
        for brand_metric in metrics_data.get("brand_metrics", []):
            brand = brand_metric.get("brand_name")
            if brand:
                self.set_entity_metadata(brand, {
                    "type": "brand",
                    "sos": brand_metric.get("share_of_shelf"),
                    "avg_rank": brand_metric.get("avg_rank"),
                    "product_count": brand_metric.get("product_count"),
                    "is_target": brand_metric.get("is_laneige", False),
                    "category": brand_metric.get("category_id")
                })

        # 제품 메타데이터 설정
        for product_metric in metrics_data.get("product_metrics", []):
            asin = product_metric.get("asin")
            if asin:
                self.set_entity_metadata(asin, {
                    "type": "product",
                    "current_rank": product_metric.get("current_rank"),
                    "rank_change_1d": product_metric.get("rank_change_1d"),
                    "rank_change_7d": product_metric.get("rank_change_7d"),
                    "rank_volatility": product_metric.get("rank_volatility"),
                    "streak_days": product_metric.get("streak_days"),
                    "rating": product_metric.get("rating"),
                    "category": product_metric.get("category_id")
                })

        # 알림 기반 관계 추가
        for alert in metrics_data.get("alerts", []):
            asin = alert.get("asin")
            if asin:
                rel = Relation(
                    subject=asin,
                    predicate=RelationType.HAS_ALERT,
                    object=alert.get("type", "unknown"),
                    properties={
                        "severity": alert.get("severity"),
                        "message": alert.get("message"),
                        "details": alert.get("details", {})
                    },
                    source="metrics"
                )
                if self.add_relation(rel):
                    added += 1

        return added

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"KnowledgeGraph("
            f"triples={stats['total_triples']}, "
            f"subjects={stats['unique_subjects']}, "
            f"objects={stats['unique_objects']})"
        )
