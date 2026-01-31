"""
Knowledge Graph
================
온톨로지 기반 지식 그래프 구현 (In-Memory Triple Store)

## 핵심 개념
- **Triple**: (Subject, Predicate, Object) 형태의 관계 표현
  - 예: ("LANEIGE", HAS_PRODUCT, "B08XYZ")
  - 예: ("B08XYZ", BELONGS_TO_CATEGORY, "lip_care")
  - 예: ("LANEIGE", COMPETES_WITH, "COSRX")

## 데이터 구조
```
KnowledgeGraph
├── triples: List[Relation]              # 모든 관계 저장
├── subject_index: Dict[str, List]       # 주체별 빠른 조회
├── object_index: Dict[str, List]        # 객체별 빠른 조회
├── predicate_index: Dict[Type, List]    # 관계 유형별 인덱스
└── entity_metadata: Dict[str, Dict]     # 엔티티 메타데이터
```

## 관계 유형 (RelationType)
- HAS_PRODUCT: 브랜드 → 제품
- BELONGS_TO_CATEGORY: 제품 → 카테고리
- COMPETES_WITH / DIRECT_COMPETITOR: 브랜드 경쟁 관계
- PARENT_CATEGORY / HAS_SUBCATEGORY: 카테고리 계층
- HAS_SENTIMENT / HAS_AI_SUMMARY: 감성 분석 데이터

## 사용 패턴
```python
kg = KnowledgeGraph()

# 데이터 로드
kg.load_from_crawl_data(crawl_result)     # 크롤링 데이터에서 자동 관계 생성
kg.load_category_hierarchy()               # 카테고리 계층 로드

# 쿼리
products = kg.get_brand_products("LANEIGE")
competitors = kg.get_competitors("LANEIGE", category="lip_care")
hierarchy = kg.get_category_hierarchy("lip_care")

# 그래프 탐색
path = kg.find_path("LANEIGE", "lip_care")
neighbors = kg.get_neighbors("LANEIGE", direction="outgoing")
```

## 메모리 관리
- max_triples: 최대 트리플 수 (기본 50,000)
- 초과 시 FIFO로 오래된 트리플 자동 삭제 (10%)

## 기능
1. 트리플(Triple) 저장 및 관리
2. 패턴 기반 쿼리 (SPARQL-like)
3. 그래프 탐색 (BFS/DFS)
4. 관계 추론 지원
5. 카테고리 계층 관리
6. 감성 데이터 통합
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .relations import (
    Relation,
    RelationType,
    create_brand_product_relation,
    create_competition_relation,
    create_product_category_relation,
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

    # 기본값 (설정 파일 없을 때 fallback)
    DEFAULT_MAX_TRIPLES = 50000
    DEFAULT_PERSIST_PATH = "data/knowledge_graph.json"
    DEFAULT_SAVE_BATCH_THRESHOLD = 100
    RAILWAY_PERSIST_PATH = "/data/knowledge_graph.json"

    # 설정 파일 경로
    CONFIG_PATH = "config/thresholds.json"

    # 중요한 관계 유형 (eviction 시 보호)
    PROTECTED_RELATION_TYPES = {
        RelationType.OWNED_BY,
        RelationType.PARENT_CATEGORY,
        RelationType.HAS_SUBCATEGORY,
    }

    @classmethod
    def _load_config(cls) -> dict:
        """설정 파일에서 KG 관련 설정 로드"""
        from pathlib import Path as P

        project_root = P(__file__).parent.parent.parent
        config_path = project_root / cls.CONFIG_PATH

        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("system", {}).get("knowledge_graph", {})
            except Exception:
                pass

        return {}  # 설정 없으면 기본값 사용

    def __init__(
        self,
        persist_path: str | None = None,
        max_triples: int = None,
        auto_save: bool = None,
        auto_load: bool = True,
    ):
        """
        Args:
            persist_path: 영속화 경로 (JSON 파일, None이면 설정 파일 또는 기본 경로 사용)
            max_triples: 최대 트리플 수 (None이면 설정 파일 또는 기본값 50000)
            auto_save: True면 변경 시 자동 저장 (None이면 설정 파일에서 로드)
            auto_load: True면 초기화 시 자동 로드 (기본: True)
        """
        # 설정 파일 로드
        config = self._load_config()

        # Railway 환경 감지
        import os

        is_railway = bool(os.environ.get("RAILWAY_ENVIRONMENT"))

        # 기본 경로 설정 (Railway Volume > 설정 파일 > 파라미터 > 기본값)
        if persist_path is None:
            if is_railway:
                persist_path = self.RAILWAY_PERSIST_PATH
                import logging

                logging.getLogger(__name__).info(
                    f"Railway environment detected, using Volume path: {persist_path}"
                )
            else:
                from pathlib import Path as P

                project_root = P(__file__).parent.parent.parent
                config_path = config.get("persist_path", self.DEFAULT_PERSIST_PATH)
                persist_path = str(project_root / config_path)

        self.persist_path = Path(persist_path) if persist_path else None

        # max_triples (설정 파일 > 파라미터 > 기본값)
        self.max_triples = max_triples or config.get("max_triples", self.DEFAULT_MAX_TRIPLES)

        # auto_save (설정 파일 > 파라미터 > 기본값)
        if auto_save is None:
            auto_save = config.get("auto_save", True)
        self.auto_save = auto_save
        self._dirty = False  # 변경 추적
        self._save_batch_count = 0  # 배치 저장용 카운터
        self._save_batch_threshold = config.get(
            "save_batch_threshold", self.DEFAULT_SAVE_BATCH_THRESHOLD
        )

        # 트리플 저장소
        self.triples: list[Relation] = []

        # 인덱스 (빠른 조회용)
        self.subject_index: dict[str, list[Relation]] = defaultdict(list)
        self.object_index: dict[str, list[Relation]] = defaultdict(list)
        self.predicate_index: dict[RelationType, list[Relation]] = defaultdict(list)

        # 엔티티 메타데이터
        self.entity_metadata: dict[str, dict[str, Any]] = {}

        # 트리플 중요도 점수 (eviction 정책용)
        self._importance_scores: dict[int, float] = {}  # relation id -> score

        # 통계
        self._stats = {
            "total_triples": 0,
            "unique_subjects": 0,
            "unique_objects": 0,
            "relations_by_type": {},
        }

        # 영속화된 데이터 자동 로드
        if auto_load and self.persist_path and self.persist_path.exists():
            self._load()
            import logging

            logging.getLogger(__name__).info(
                f"KnowledgeGraph auto-loaded from {self.persist_path}: "
                f"{len(self.triples)} triples"
            )

    def _calculate_importance(self, relation: Relation) -> float:
        """
        트리플 중요도 점수 계산 (eviction 정책용)

        높은 점수 = 더 중요 = 삭제 우선순위 낮음
        """
        score = 0.5  # 기본 점수

        # 보호된 관계 유형은 높은 점수
        if relation.predicate in self.PROTECTED_RELATION_TYPES:
            score += 10.0

        # confidence가 높으면 높은 점수
        score += relation.confidence * 2.0

        # 브랜드 관련 관계는 중요
        if relation.predicate in {RelationType.HAS_PRODUCT, RelationType.COMPETES_WITH}:
            score += 1.0

        # 최근 생성된 관계는 약간 높은 점수
        if relation.created_at:
            from datetime import datetime, timedelta

            age = datetime.now() - relation.created_at
            if age < timedelta(days=7):
                score += 0.5
            elif age < timedelta(days=30):
                score += 0.2

        return score

    def _enforce_max_triples_smart(self) -> int:
        """
        중요도 기반 트리플 제거 (FIFO 대신)

        Returns:
            제거된 트리플 수
        """
        if len(self.triples) <= self.max_triples:
            return 0

        # 10% 제거 대상
        remove_count = int(len(self.triples) * 0.1)

        # 중요도 점수 계산 (캐시 업데이트)
        for rel in self.triples:
            if id(rel) not in self._importance_scores:
                self._importance_scores[id(rel)] = self._calculate_importance(rel)

        # 중요도 낮은 순으로 정렬
        sorted_by_importance = sorted(
            enumerate(self.triples),
            key=lambda x: self._importance_scores.get(id(x[1]), 0),
        )

        # 낮은 중요도부터 제거
        to_remove_indices = [idx for idx, _ in sorted_by_importance[:remove_count]]

        for idx in sorted(to_remove_indices, reverse=True):
            rel = self.triples[idx]
            self._remove_from_indices(rel)
            del self.triples[idx]

        import logging

        logging.getLogger(__name__).info(
            f"KnowledgeGraph eviction: removed {len(to_remove_indices)} low-importance triples"
        )

        return len(to_remove_indices)

    def _remove_from_indices(self, relation: Relation) -> None:
        """인덱스에서 관계 제거"""
        if relation in self.subject_index.get(relation.subject, []):
            self.subject_index[relation.subject].remove(relation)
        if relation in self.object_index.get(relation.object, []):
            self.object_index[relation.object].remove(relation)
        if relation in self.predicate_index.get(relation.predicate, []):
            self.predicate_index[relation.predicate].remove(relation)

    def _maybe_auto_save(self) -> None:
        """배치 자동 저장 (변경이 충분히 쌓이면)"""
        if not self.auto_save:
            return

        self._dirty = True
        self._save_batch_count += 1

        if self._save_batch_count >= self._save_batch_threshold:
            self.save()
            self._save_batch_count = 0
            self._dirty = False

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

        # 최대 크기 체크 - 초과 시 중요도 기반 삭제 (스마트 eviction)
        if len(self.triples) >= self.max_triples:
            self._enforce_max_triples_smart()

        # 추가
        self.triples.append(relation)

        # 중요도 점수 캐시
        self._importance_scores[id(relation)] = self._calculate_importance(relation)

        # 인덱스 업데이트
        self.subject_index[relation.subject].append(relation)
        self.object_index[relation.object].append(relation)
        self.predicate_index[relation.predicate].append(relation)

        # 통계 업데이트
        self._update_stats()

        # 자동 저장 (배치)
        self._maybe_auto_save()

        return True

    def _evict_oldest(self, count: int) -> int:
        """
        가장 오래된 트리플 삭제 (FIFO)

        Args:
            count: 삭제할 개수

        Returns:
            실제 삭제된 개수
        """
        if count <= 0 or not self.triples:
            return 0

        # 삭제할 트리플들
        to_remove = self.triples[:count]

        # 인덱스에서 제거
        for relation in to_remove:
            if relation in self.subject_index[relation.subject]:
                self.subject_index[relation.subject].remove(relation)
            if relation in self.object_index[relation.object]:
                self.object_index[relation.object].remove(relation)
            if relation in self.predicate_index[relation.predicate]:
                self.predicate_index[relation.predicate].remove(relation)

        # 트리플 리스트에서 제거
        self.triples = self.triples[count:]

        return len(to_remove)

    def add_relations(self, relations: list[Relation]) -> int:
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
        subject: str | None = None,
        predicate: RelationType | None = None,
        object_: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Relation]:
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

    def get_subjects(self, predicate: RelationType, object_: str) -> list[str]:
        """
        특정 관계와 객체에 대한 모든 주체 조회

        예: get_subjects(HAS_PRODUCT, "B08XYZ") → ["LANEIGE"]
        """
        relations = self.query(predicate=predicate, object_=object_)
        return list({r.subject for r in relations})

    def get_objects(self, subject: str, predicate: RelationType) -> list[str]:
        """
        특정 주체와 관계에 대한 모든 객체 조회

        예: get_objects("LANEIGE", HAS_PRODUCT) → ["B08XYZ", "B09ABC"]
        """
        relations = self.query(subject=subject, predicate=predicate)
        return list({r.object for r in relations})

    def get_predicates(self, subject: str, object_: str) -> list[RelationType]:
        """
        두 엔티티 간의 모든 관계 유형 조회

        예: get_predicates("LANEIGE", "COSRX") → [COMPETES_WITH]
        """
        relations = self.query(subject=subject, object_=object_)
        return list({r.predicate for r in relations})

    def exists(self, subject: str, predicate: RelationType, object_: str) -> bool:
        """관계 존재 여부 확인"""
        return len(self.query(subject, predicate, object_)) > 0

    # =========================================================================
    # 그래프 탐색
    # =========================================================================

    def get_neighbors(
        self,
        entity: str,
        direction: str = "both",
        predicate_filter: list[RelationType] | None = None,
    ) -> dict[str, list[tuple[RelationType, str]]]:
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
        predicate_filter: list[RelationType] | None = None,
        direction: str = "outgoing",
    ) -> dict[int, list[str]]:
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
        visited: set[str] = set()
        result: dict[int, list[str]] = defaultdict(list)
        queue: list[tuple[str, int]] = [(start_entity, 0)]

        while queue:
            current, depth = queue.pop(0)

            if current in visited or depth > max_depth:
                continue

            visited.add(current)
            result[depth].append(current)

            if depth < max_depth:
                neighbors = self.get_neighbors(
                    current, direction=direction, predicate_filter=predicate_filter
                )

                if direction == "both":
                    neighbor_list = neighbors.get("outgoing", []) + neighbors.get("incoming", [])
                else:
                    neighbor_list = neighbors.get(direction, [])

                for _, neighbor in neighbor_list:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        return dict(result)

    def find_path(self, start: str, end: str, max_depth: int = 5) -> list[Relation] | None:
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

        visited: set[str] = set()
        queue: list[tuple[str, list[Relation]]] = [(start, [])]

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

    def get_brand_products(self, brand: str, category: str | None = None) -> list[dict[str, Any]]:
        """
        브랜드의 제품 목록 조회

        Args:
            brand: 브랜드명
            category: 카테고리 필터 (선택)

        Returns:
            제품 정보 리스트
        """
        products = []
        relations = self.query(subject=brand, predicate=RelationType.HAS_PRODUCT)

        for rel in relations:
            product_asin = rel.object
            props = rel.properties

            # 카테고리 필터
            if category and props.get("category") != category:
                continue

            products.append(
                {
                    "asin": product_asin,
                    "name": props.get("product_name", ""),
                    "category": props.get("category", ""),
                    "rank": props.get("rank"),
                    **props,
                }
            )

        return products

    def get_competitors(
        self, brand: str, category: str | None = None, competition_type: str = "all"
    ) -> list[dict[str, Any]]:
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

                competitors.append(
                    {
                        "brand": rel.object,
                        "type": rel.predicate.value,
                        "category": rel.properties.get("category", ""),
                        **rel.properties,
                    }
                )

        # 중복 제거
        seen = set()
        unique = []
        for comp in competitors:
            key = (comp["brand"], comp.get("category"))
            if key not in seen:
                seen.add(key)
                unique.append(comp)

        return unique

    def get_category_brands(self, category: str, min_products: int = 1) -> list[dict[str, Any]]:
        """
        카테고리 내 브랜드 목록

        Args:
            category: 카테고리 ID
            min_products: 최소 제품 수

        Returns:
            브랜드별 정보
        """
        # 카테고리에 속한 제품들
        product_relations = self.query(predicate=RelationType.BELONGS_TO_CATEGORY, object_=category)

        # 제품별 브랜드 매핑
        brand_products: dict[str, list[str]] = defaultdict(list)

        for rel in product_relations:
            product_asin = rel.subject
            # 해당 제품의 브랜드 찾기
            brand_rels = self.query(predicate=RelationType.HAS_PRODUCT, object_=product_asin)
            for br in brand_rels:
                brand_products[br.subject].append(product_asin)

        # 결과 구성
        results = []
        for brand, products in brand_products.items():
            if len(products) >= min_products:
                results.append(
                    {
                        "brand": brand,
                        "product_count": len(products),
                        "products": products,
                    }
                )

        return sorted(results, key=lambda x: x["product_count"], reverse=True)

    def get_entity_context(self, entity: str, depth: int = 2) -> dict[str, Any]:
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
            "metadata": self.entity_metadata.get(entity, {}),
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
                        context["connected"][target] = self.get_entity_context(target, depth - 1)

        return context

    # =========================================================================
    # 엔티티 메타데이터
    # =========================================================================

    def set_entity_metadata(self, entity: str, metadata: dict[str, Any]) -> None:
        """엔티티 메타데이터 설정"""
        if entity not in self.entity_metadata:
            self.entity_metadata[entity] = {}
        self.entity_metadata[entity].update(metadata)

    def get_entity_metadata(self, entity: str) -> dict[str, Any]:
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
            pred.value: len(rels) for pred, rels in self.predicate_index.items()
        }

    def get_stats(self) -> dict[str, Any]:
        """통계 조회"""
        self._update_stats()
        return self._stats.copy()

    def get_entity_degree(self, entity: str) -> dict[str, int]:
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
            "total": in_degree + out_degree,
        }

    def get_most_connected(self, top_n: int = 10) -> list[tuple[str, int]]:
        """
        가장 많이 연결된 엔티티

        Returns:
            [(entity, degree), ...]
        """
        all_entities = set(self.subject_index.keys()) | set(self.object_index.keys())
        degrees = [(entity, self.get_entity_degree(entity)["total"]) for entity in all_entities]
        return sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]

    # =========================================================================
    # 영속화
    # =========================================================================

    def save(self, path: str | None = None, force: bool = False) -> bool:
        """
        그래프 저장

        Args:
            path: 저장 경로 (None이면 초기화 시 설정된 경로)
            force: True면 변경 없어도 강제 저장

        Returns:
            저장 성공 여부
        """
        import logging

        logger = logging.getLogger(__name__)

        # 변경 없으면 스킵 (force가 아닐 경우)
        if not force and not self._dirty:
            return False

        save_path = Path(path) if path else self.persist_path
        if not save_path:
            logger.warning("No persist path specified for KnowledgeGraph.save()")
            return False

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "2.0",  # 스마트 eviction 버전
                "triples": [r.to_dict() for r in self.triples],
                "entity_metadata": self.entity_metadata,
                "stats": {
                    "total_triples": len(self.triples),
                    "unique_subjects": len(self.subject_index),
                    "unique_objects": len(self.object_index),
                    "relation_types": {
                        str(k.value): len(v) for k, v in self.predicate_index.items()
                    },
                },
                "saved_at": datetime.now().isoformat(),
            }

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self._dirty = False
            logger.info(f"KnowledgeGraph saved to {save_path}: {len(self.triples)} triples")
            return True

        except Exception as e:
            logger.error(f"Failed to save KnowledgeGraph: {e}")
            return False

    def save_if_dirty(self) -> bool:
        """변경이 있을 때만 저장 (종료 시 호출용)"""
        if self._dirty:
            return self.save()
        return False

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 - 자동 저장"""
        self.save_if_dirty()
        return False

    def _load(self) -> None:
        """영속화된 데이터 로드"""
        import logging

        logger = logging.getLogger(__name__)

        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)

            # 버전 체크
            version = data.get("version", "1.0")
            logger.info(f"Loading KnowledgeGraph v{version} from {self.persist_path}")

            for triple_dict in data.get("triples", []):
                relation = Relation.from_dict(triple_dict)
                self.add_relation(relation)

            self.entity_metadata = data.get("entity_metadata", {})
            self._dirty = False  # 로드 직후에는 dirty 아님

        except Exception as e:
            logger.error(f"Failed to load KnowledgeGraph: {e}")
            raise

    # =========================================================================
    # 데이터 로딩 헬퍼
    # =========================================================================

    def load_from_crawl_data(self, crawl_data: dict[str, Any]) -> int:
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

            brand_products: dict[str, list[dict]] = defaultdict(list)

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
                    price=product.get("price"),
                )
                if self.add_relation(rel1):
                    added += 1

                # Product → Category 관계
                rel2 = create_product_category_relation(
                    product_asin=asin, category_id=cat_key, rank=rank
                )
                if self.add_relation(rel2):
                    added += 1

            # 브랜드 간 경쟁 관계 (같은 카테고리 내)
            brands = list(brand_products.keys())
            for i, brand1 in enumerate(brands[:10]):  # Top 10 브랜드만
                for brand2 in brands[i + 1 : 10]:
                    comp_rel = create_competition_relation(
                        brand1=brand1,
                        brand2=brand2,
                        category=cat_key,
                        competition_type="direct"
                        if i < 5 and brands.index(brand2) < 5
                        else "indirect",
                    )
                    if self.add_relation(comp_rel):
                        added += 1

                    # 역방향도 추가
                    comp_rel_rev = create_competition_relation(
                        brand1=brand2,
                        brand2=brand1,
                        category=cat_key,
                        competition_type="direct"
                        if i < 5 and brands.index(brand2) < 5
                        else "indirect",
                    )
                    if self.add_relation(comp_rel_rev):
                        added += 1

        return added

    def load_from_metrics_data(self, metrics_data: dict[str, Any]) -> int:
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
                self.set_entity_metadata(
                    brand,
                    {
                        "type": "brand",
                        "sos": brand_metric.get("share_of_shelf"),
                        "avg_rank": brand_metric.get("avg_rank"),
                        "product_count": brand_metric.get("product_count"),
                        "is_target": brand_metric.get("is_laneige", False),
                        "category": brand_metric.get("category_id"),
                    },
                )

        # 제품 메타데이터 설정
        for product_metric in metrics_data.get("product_metrics", []):
            asin = product_metric.get("asin")
            if asin:
                self.set_entity_metadata(
                    asin,
                    {
                        "type": "product",
                        "current_rank": product_metric.get("current_rank"),
                        "rank_change_1d": product_metric.get("rank_change_1d"),
                        "rank_change_7d": product_metric.get("rank_change_7d"),
                        "rank_volatility": product_metric.get("rank_volatility"),
                        "streak_days": product_metric.get("streak_days"),
                        "rating": product_metric.get("rating"),
                        "category": product_metric.get("category_id"),
                    },
                )

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
                        "details": alert.get("details", {}),
                    },
                    source="metrics",
                )
                if self.add_relation(rel):
                    added += 1

        return added

    def load_category_hierarchy(
        self, hierarchy_path: str = "config/category_hierarchy.json"
    ) -> int:
        """
        카테고리 계층 구조 로드

        Args:
            hierarchy_path: 카테고리 계층 JSON 파일 경로

        Returns:
            추가된 관계 수
        """
        import json
        from pathlib import Path

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
            self.set_entity_metadata(
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
                if self.add_relation(rel_parent):
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
                if self.add_relation(rel_child):
                    added += 1

        return added

    def get_category_hierarchy(self, category_id: str) -> dict[str, Any]:
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
        metadata = self.get_entity_metadata(category_id)
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
            parent_rels = self.query(subject=current, predicate=RelationType.PARENT_CATEGORY)
            if not parent_rels:
                break
            parent_id = parent_rels[0].object
            parent_meta = self.get_entity_metadata(parent_id)
            result["ancestors"].append(
                {
                    "id": parent_id,
                    "name": parent_meta.get("name", ""),
                    "level": parent_meta.get("level", 0),
                }
            )
            current = parent_id

        # 하위 카테고리 탐색 (1단계만)
        child_rels = self.query(subject=category_id, predicate=RelationType.HAS_SUBCATEGORY)
        for rel in child_rels:
            child_id = rel.object
            child_meta = self.get_entity_metadata(child_id)
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
                "product": asin,
                "categories": [
                    {
                        "category_id": str,
                        "rank": int,
                        "hierarchy": {...}
                    }
                ]
            }
        """
        result = {"product": product_asin, "categories": []}

        # 제품이 속한 카테고리 조회
        cat_rels = self.query(subject=product_asin, predicate=RelationType.BELONGS_TO_CATEGORY)

        for rel in cat_rels:
            cat_id = rel.object
            cat_info = {
                "category_id": cat_id,
                "rank": rel.properties.get("rank"),
                "hierarchy": self.get_category_hierarchy(cat_id),
            }
            result["categories"].append(cat_info)

        return result

    # =========================================================================
    # 감성 데이터 로딩 및 쿼리
    # =========================================================================

    def load_from_sentiment_data(self, sentiment_data: dict[str, Any]) -> int:
        """
        AI Customers Say 및 감성 태그 데이터에서 관계 로드

        Args:
            sentiment_data: AmazonProductScraper 결과
                {
                    "products": {
                        "B0BSHRYY1S": {
                            "asin": "B0BSHRYY1S",
                            "ai_customers_say": "Customers like the moisturizing...",
                            "sentiment_tags": ["Moisturizing", "Value for money"],
                            "success": True
                        },
                        ...
                    },
                    "collected_at": "2025-01-15"
                }

        Returns:
            추가된 관계 수
        """
        from .relations import (
            create_ai_summary_relation,
            create_sentiment_relation,
            get_cluster_for_sentiment,
        )

        added = 0
        collected_at = sentiment_data.get("collected_at", "")

        products = sentiment_data.get("products", {})
        for asin, product_data in products.items():
            if not product_data.get("success"):
                continue

            # AI Summary 관계 추가
            ai_summary = product_data.get("ai_customers_say")
            if ai_summary:
                rel = create_ai_summary_relation(
                    product_asin=asin, ai_summary=ai_summary, collected_at=collected_at
                )
                if self.add_relation(rel):
                    added += 1

                # 제품 메타데이터 업데이트
                self.set_entity_metadata(
                    asin,
                    {"ai_summary": ai_summary, "ai_summary_collected_at": collected_at},
                )

            # Sentiment Tag 관계 추가
            sentiment_tags = product_data.get("sentiment_tags", [])
            for tag in sentiment_tags:
                cluster = get_cluster_for_sentiment(tag)
                rel = create_sentiment_relation(
                    product_asin=asin, sentiment_tag=tag, sentiment_cluster=cluster
                )
                if self.add_relation(rel):
                    added += 1

        return added

    def get_product_sentiments(self, product_asin: str) -> dict[str, Any]:
        """
        제품의 감성 정보 조회

        Args:
            product_asin: 제품 ASIN

        Returns:
            {
                "asin": str,
                "ai_summary": str,
                "sentiment_tags": List[str],
                "sentiment_clusters": Dict[str, List[str]]
            }
        """
        result = {
            "asin": product_asin,
            "ai_summary": None,
            "sentiment_tags": [],
            "sentiment_clusters": {},
        }

        # AI Summary 조회
        summary_rels = self.query(subject=product_asin, predicate=RelationType.HAS_AI_SUMMARY)
        if summary_rels:
            result["ai_summary"] = summary_rels[0].properties.get("summary_text")

        # Sentiment Tags 조회
        sentiment_rels = self.query(subject=product_asin, predicate=RelationType.HAS_SENTIMENT)
        for rel in sentiment_rels:
            tag = rel.object
            cluster = rel.properties.get("cluster")
            result["sentiment_tags"].append(tag)
            if cluster:
                if cluster not in result["sentiment_clusters"]:
                    result["sentiment_clusters"][cluster] = []
                result["sentiment_clusters"][cluster].append(tag)

        return result

    def get_brand_sentiment_profile(self, brand: str) -> dict[str, Any]:
        """
        브랜드의 전체 감성 프로필 조회

        Args:
            brand: 브랜드명

        Returns:
            {
                "brand": str,
                "all_tags": List[str],
                "clusters": Dict[str, int],  # 클러스터별 언급 빈도
                "dominant_sentiment": str,
                "product_count": int
            }
        """
        result = {
            "brand": brand,
            "all_tags": [],
            "clusters": {},
            "dominant_sentiment": None,
            "product_count": 0,
        }

        # 브랜드의 모든 제품 조회
        products = self.get_brand_products(brand)
        result["product_count"] = len(products)

        # 각 제품의 감성 태그 집계
        tag_counts = {}
        cluster_counts = {}

        for product in products:
            asin = product.get("asin")
            if not asin:
                continue

            sentiments = self.get_product_sentiments(asin)
            for tag in sentiments.get("sentiment_tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            for cluster, tags in sentiments.get("sentiment_clusters", {}).items():
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + len(tags)

        result["all_tags"] = list(tag_counts.keys())
        result["clusters"] = cluster_counts

        # 가장 많이 언급된 태그
        if tag_counts:
            result["dominant_sentiment"] = max(tag_counts, key=tag_counts.get)

        return result

    def compare_product_sentiments(self, asin1: str, asin2: str) -> dict[str, Any]:
        """
        두 제품의 감성 비교

        Args:
            asin1: 제품 1 ASIN
            asin2: 제품 2 ASIN

        Returns:
            {
                "product1": {...},
                "product2": {...},
                "common_tags": List[str],
                "unique_to_1": List[str],
                "unique_to_2": List[str],
                "cluster_comparison": {...}
            }
        """
        sent1 = self.get_product_sentiments(asin1)
        sent2 = self.get_product_sentiments(asin2)

        tags1 = set(sent1.get("sentiment_tags", []))
        tags2 = set(sent2.get("sentiment_tags", []))

        return {
            "product1": sent1,
            "product2": sent2,
            "common_tags": list(tags1 & tags2),
            "unique_to_1": list(tags1 - tags2),
            "unique_to_2": list(tags2 - tags1),
            "cluster_comparison": {
                "product1_clusters": sent1.get("sentiment_clusters", {}),
                "product2_clusters": sent2.get("sentiment_clusters", {}),
            },
        }

    def find_products_by_sentiment(self, sentiment_tag: str, brand_filter: str = None) -> list[str]:
        """
        특정 감성 태그를 가진 제품 검색

        Args:
            sentiment_tag: 감성 태그
            brand_filter: 브랜드 필터 (선택)

        Returns:
            ASIN 리스트
        """
        rels = self.query(predicate=RelationType.HAS_SENTIMENT, object_=sentiment_tag)

        asins = [rel.subject for rel in rels]

        # 브랜드 필터 적용
        if brand_filter:
            filtered = []
            for asin in asins:
                meta = self.get_entity_metadata(asin)
                if meta.get("brand", "").lower() == brand_filter.lower():
                    filtered.append(asin)
            return filtered

        return asins

    # =========================================================================
    # 브랜드 소유권 데이터 로딩 (2026-01-26 추가)
    # =========================================================================

    def load_brand_ownership(self, brands_config_path: str = "config/brands.json") -> int:
        """
        브랜드 소유권 데이터 로드 (config/brands.json 기반)

        Args:
            brands_config_path: 브랜드 설정 JSON 파일 경로

        Returns:
            추가된 관계 수

        Note:
            - COSRX는 한국 브랜드 (중국 브랜드 아님)
            - 2024년 아모레퍼시픽에 인수됨
            - IR 2025 Q1-Q3에서 COSRX 실적 보고됨
        """
        import json
        from pathlib import Path

        path = Path(brands_config_path)
        if not path.exists():
            # 상대 경로로 시도
            base_path = Path(__file__).parent.parent.parent
            path = base_path / brands_config_path

        if not path.exists():
            print(f"Warning: Brands config file not found: {brands_config_path}")
            return 0

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        added = 0
        amorepacific_brands = data.get("amorepacific_brands", [])
        brand_ownership = data.get("brand_ownership", {})

        # CorporateGroup 메타데이터 설정
        self.set_entity_metadata(
            "AMOREPACIFIC",
            {
                "type": "corporate_group",
                "name": "아모레퍼시픽",
                "english_name": "Amorepacific Corporation",
                "country": "Korea",
                "stock_code": "090430.KS",
            },
        )

        # 아모레퍼시픽 브랜드 관계 추가
        for brand_data in amorepacific_brands:
            brand_name = brand_data.get("name", "")
            if not brand_name:
                continue

            segment = brand_data.get("segment", "Unknown")
            category = brand_data.get("category", "")
            acquired = brand_data.get("acquired")
            country = brand_data.get("country", "Korea")
            aliases = brand_data.get("aliases", [])

            # Brand → CorporateGroup 관계
            rel_ownership = Relation(
                subject=brand_name,
                predicate=RelationType.OWNED_BY_GROUP,
                object="AMOREPACIFIC",
                properties={
                    "segment": segment,
                    "category": category,
                    "acquired": acquired,
                    "aliases": aliases,
                },
                confidence=1.0,
                source="config/brands.json",
            )
            if self.add_relation(rel_ownership):
                added += 1

            # CorporateGroup → Brand 역관계
            rel_reverse = Relation(
                subject="AMOREPACIFIC",
                predicate=RelationType.OWNS_BRAND,
                object=brand_name,
                properties={"segment": segment, "category": category},
                confidence=1.0,
                source="config/brands.json",
            )
            if self.add_relation(rel_reverse):
                added += 1

            # Brand → Segment 관계
            rel_segment = Relation(
                subject=brand_name,
                predicate=RelationType.HAS_SEGMENT,
                object=segment,
                properties={"category": category},
                confidence=1.0,
                source="config/brands.json",
            )
            if self.add_relation(rel_segment):
                added += 1

            # Brand → Country 원산지 관계
            rel_origin = Relation(
                subject=brand_name,
                predicate=RelationType.ORIGINATES_FROM,
                object=country,
                properties={},
                confidence=1.0,
                source="config/brands.json",
            )
            if self.add_relation(rel_origin):
                added += 1

            # 인수 브랜드 추가 정보
            if acquired:
                acquired_year = str(acquired) if isinstance(acquired, int) else acquired
                if acquired_year and acquired_year != "true":
                    rel_acquired = Relation(
                        subject=brand_name,
                        predicate=RelationType.ACQUIRED_IN,
                        object=acquired_year,
                        properties={"original_country": country},
                        confidence=1.0,
                        source="config/brands.json",
                    )
                    if self.add_relation(rel_acquired):
                        added += 1

            # 브랜드 메타데이터 설정
            self.set_entity_metadata(
                brand_name,
                {
                    "type": "brand",
                    "segment": segment,
                    "category": category,
                    "country": country,
                    "parent_group": "AMOREPACIFIC",
                    "aliases": aliases,
                    "acquired": acquired,
                },
            )

        # brand_ownership 상세 정보 추가 (COSRX 등)
        for brand_name, ownership_info in brand_ownership.items():
            note = ownership_info.get("note", "")
            evidence = ownership_info.get("evidence", [])

            # 기존 메타데이터 업데이트
            existing_meta = self.get_entity_metadata(brand_name)
            existing_meta.update(
                {
                    "ownership_note": note,
                    "ownership_evidence": evidence,
                    "country_of_origin": ownership_info.get("country_of_origin", "Korea"),
                }
            )
            self.set_entity_metadata(brand_name, existing_meta)

        # 자매 브랜드 관계 추가 (siblingBrand)
        brand_names = [b.get("name") for b in amorepacific_brands if b.get("name")]
        for i, brand1 in enumerate(brand_names):
            for brand2 in brand_names[i + 1 :]:
                rel_sibling = Relation(
                    subject=brand1,
                    predicate=RelationType.SIBLING_BRAND,
                    object=brand2,
                    properties={"parent_group": "AMOREPACIFIC"},
                    confidence=1.0,
                    source="config/brands.json",
                )
                if self.add_relation(rel_sibling):
                    added += 1

                # 역방향도 추가 (대칭 관계)
                rel_sibling_rev = Relation(
                    subject=brand2,
                    predicate=RelationType.SIBLING_BRAND,
                    object=brand1,
                    properties={"parent_group": "AMOREPACIFIC"},
                    confidence=1.0,
                    source="config/brands.json",
                )
                if self.add_relation(rel_sibling_rev):
                    added += 1

        return added

    def get_brand_ownership(self, brand: str) -> dict[str, Any]:
        """
        브랜드 소유권 정보 조회

        Args:
            brand: 브랜드명

        Returns:
            {
                "brand": str,
                "parent_group": str or None,
                "segment": str,
                "country_of_origin": str,
                "acquired": str or None,
                "sibling_brands": List[str],
                "note": str
            }

        Example:
            >>> kg.get_brand_ownership("COSRX")
            {
                "brand": "COSRX",
                "parent_group": "AMOREPACIFIC",
                "segment": "K-Beauty",
                "country_of_origin": "Korea",  # NOT China
                "acquired": "2024",
                "sibling_brands": ["LANEIGE", "Sulwhasoo", ...],
                "note": "COSRX is a Korean brand, NOT Chinese..."
            }
        """
        result = {
            "brand": brand,
            "parent_group": None,
            "segment": None,
            "country_of_origin": None,
            "acquired": None,
            "sibling_brands": [],
            "note": None,
        }

        # 소유 그룹 조회
        ownership_rels = self.query(subject=brand, predicate=RelationType.OWNED_BY_GROUP)
        if ownership_rels:
            rel = ownership_rels[0]
            result["parent_group"] = rel.object
            result["segment"] = rel.properties.get("segment")
            result["acquired"] = rel.properties.get("acquired")

        # 원산지 조회
        origin_rels = self.query(subject=brand, predicate=RelationType.ORIGINATES_FROM)
        if origin_rels:
            result["country_of_origin"] = origin_rels[0].object

        # 자매 브랜드 조회
        sibling_rels = self.query(subject=brand, predicate=RelationType.SIBLING_BRAND)
        result["sibling_brands"] = [rel.object for rel in sibling_rels]

        # 메타데이터에서 추가 정보
        meta = self.get_entity_metadata(brand)
        if meta:
            result["note"] = meta.get("ownership_note")
            if not result["country_of_origin"]:
                result["country_of_origin"] = meta.get("country_of_origin", meta.get("country"))

        return result

    def is_amorepacific_brand(self, brand: str) -> bool:
        """
        브랜드가 아모레퍼시픽 소속인지 확인

        Args:
            brand: 브랜드명

        Returns:
            True if 아모레퍼시픽 소속
        """
        rels = self.query(
            subject=brand, predicate=RelationType.OWNED_BY_GROUP, object_="AMOREPACIFIC"
        )
        return len(rels) > 0

    def get_amorepacific_brands(self, segment_filter: str = None) -> list[dict[str, Any]]:
        """
        아모레퍼시픽 소속 브랜드 목록 조회

        Args:
            segment_filter: 세그먼트 필터 (예: "Premium", "Luxury")

        Returns:
            브랜드 정보 리스트
        """
        rels = self.query(subject="AMOREPACIFIC", predicate=RelationType.OWNS_BRAND)

        results = []
        for rel in rels:
            brand = rel.object
            segment = rel.properties.get("segment", "")

            if segment_filter and segment != segment_filter:
                continue

            results.append(
                {
                    "brand": brand,
                    "segment": segment,
                    "category": rel.properties.get("category", ""),
                }
            )

        return results

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"KnowledgeGraph("
            f"triples={stats['total_triples']}, "
            f"subjects={stats['unique_subjects']}, "
            f"objects={stats['unique_objects']})"
        )


# =============================================================================
# 싱글톤 패턴
# =============================================================================

_knowledge_graph_instance: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    """
    KnowledgeGraph 싱글톤 인스턴스 반환

    Returns:
        KnowledgeGraph 인스턴스
    """
    global _knowledge_graph_instance
    if _knowledge_graph_instance is None:
        _knowledge_graph_instance = KnowledgeGraph()
    return _knowledge_graph_instance
