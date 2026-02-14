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
- 초과 시 중요도 기반 삭제 (스마트 eviction)

## 기능
1. 트리플(Triple) 저장 및 관리
2. 패턴 기반 쿼리 (SPARQL-like)
3. 그래프 탐색 (BFS/DFS)
4. 관계 추론 지원
5. 카테고리 계층 관리
6. 감성 데이터 통합
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .kg_query import KGQueryMixin
from .kg_updater import KGUpdaterMixin
from .relations import Relation, RelationType

logger = logging.getLogger(__name__)


class KnowledgeGraph(KGQueryMixin, KGUpdaterMixin):
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
                logger.warning("Suppressed Exception", exc_info=True)

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
                f"KnowledgeGraph auto-loaded from {self.persist_path}: {len(self.triples)} triples"
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
