"""
Knowledge Graph Query & Traversal Mixin
========================================
KnowledgeGraph의 쿼리 및 그래프 탐색 기능을 담당하는 믹스인 클래스

## 주요 기능
1. 트리플 패턴 쿼리 (SPARQL-like)
2. 그래프 탐색 (BFS)
3. 경로 탐색
4. 도메인 특화 쿼리 (브랜드, 제품, 카테고리)
5. 엔티티 메타데이터 관리
6. 통계 및 분석
"""

from collections import defaultdict
from typing import Any

from .relations import Relation, RelationType


class KGQueryMixin:
    """
    KnowledgeGraph의 쿼리 및 탐색 기능을 제공하는 믹스인

    이 믹스인은 KnowledgeGraph 클래스에 의해 상속되며,
    다음 속성들을 사용합니다:
    - self.triples
    - self.subject_index
    - self.object_index
    - self.predicate_index
    - self.entity_metadata
    - self._stats
    """

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
    # SPARQL-like 쿼리
    # =========================================================================

    # RelationType name → enum 매핑 (예: "hasProduct" → RelationType.HAS_PRODUCT)
    _PREDICATE_MAP: dict[str, "RelationType"] | None = None

    @classmethod
    def _build_predicate_map(cls) -> dict[str, "RelationType"]:
        if cls._PREDICATE_MAP is None:
            mapping: dict[str, RelationType] = {}
            for rt in RelationType:
                mapping[rt.value] = rt
                # camelCase alias: HAS_PRODUCT → hasProduct
                parts = rt.name.split("_")
                camel = parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
                mapping[camel] = rt
            cls._PREDICATE_MAP = mapping
        return cls._PREDICATE_MAP

    def sparql_query(self, query_str: str) -> list[dict[str, str]]:
        """
        기본 SPARQL-like 쿼리 실행

        지원 구문:
            SELECT ?var1 ?var2
            WHERE {
              ?s <predicate> ?o .
              ?s <predicate> "literal" .
            }
            FILTER (?var >= value)

        Args:
            query_str: SPARQL-like 쿼리 문자열

        Returns:
            바인딩 딕셔너리 리스트 [{var: value, ...}, ...]

        예:
            kg.sparql_query('''
                SELECT ?brand ?product
                WHERE {
                    ?brand <hasProduct> ?product .
                    ?product <belongsToCategory> "lip_care" .
                }
            ''')
        """
        select_vars, patterns, filters = self._parse_sparql(query_str)

        if not patterns:
            return []

        # 첫 번째 패턴 매칭
        bindings = self._match_pattern(patterns[0])

        # 나머지 패턴과 JOIN
        for pattern in patterns[1:]:
            new_bindings = self._match_pattern(pattern)
            bindings = self._join_bindings(bindings, new_bindings)

        # FILTER 적용
        for filt in filters:
            bindings = self._apply_filter(bindings, filt)

        # SELECT 변수만 추출
        if select_vars:
            bindings = [{k: v for k, v in b.items() if k in select_vars} for b in bindings]

        return bindings

    def _parse_sparql(
        self, query_str: str
    ) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
        """
        SPARQL 문자열 파싱

        Returns:
            (select_vars, triple_patterns, filter_conditions)
        """
        import re as _re

        # SELECT 변수 추출
        select_match = _re.search(r"SELECT\s+(.*?)\s*WHERE", query_str, _re.IGNORECASE | _re.DOTALL)
        select_vars: list[str] = []
        if select_match:
            select_vars = _re.findall(r"\?(\w+)", select_match.group(1))

        # WHERE 블록 추출
        where_match = _re.search(r"WHERE\s*\{(.*?)\}", query_str, _re.IGNORECASE | _re.DOTALL)
        patterns: list[tuple[str, str, str]] = []
        filters: list[str] = []

        if where_match:
            where_body = where_match.group(1)

            # FILTER 추출
            for fm in _re.finditer(r"FILTER\s*\((.*?)\)", where_body, _re.IGNORECASE):
                filters.append(fm.group(1).strip())

            # 트리플 패턴 추출 (subject predicate object .)
            # 변수: ?var, URI: <pred>, 리터럴: "value"
            triple_re = _re.compile(
                r'(\?\w+|"[^"]*")\s+' r"(<[^>]+>|\?\w+)\s+" r'(\?\w+|"[^"]*")\s*\.'
            )
            for tm in triple_re.finditer(where_body):
                s = tm.group(1).strip()
                p = tm.group(2).strip()
                o = tm.group(3).strip()
                patterns.append((s, p, o))

        return select_vars, patterns, filters

    def _match_pattern(self, pattern: tuple[str, str, str]) -> list[dict[str, str]]:
        """
        단일 트리플 패턴 매칭

        패턴 요소:
            ?var  → 변수 (바인딩 대상)
            <uri> → 프레디케이트 URI
            "lit" → 리터럴 값
        """
        s_pat, p_pat, o_pat = pattern
        pred_map = self._build_predicate_map()

        # 프레디케이트 결정
        predicate: RelationType | None = None
        if p_pat.startswith("<") and p_pat.endswith(">"):
            pred_name = p_pat[1:-1]
            predicate = pred_map.get(pred_name)

        # 고정 주체/객체
        fixed_subject: str | None = None
        if s_pat.startswith('"') and s_pat.endswith('"'):
            fixed_subject = s_pat[1:-1]

        fixed_object: str | None = None
        if o_pat.startswith('"') and o_pat.endswith('"'):
            fixed_object = o_pat[1:-1]

        # 쿼리 실행
        relations = self.query(
            subject=fixed_subject,
            predicate=predicate,
            object_=fixed_object,
        )

        # 바인딩 생성
        bindings: list[dict[str, str]] = []
        for rel in relations:
            binding: dict[str, str] = {}

            if s_pat.startswith("?"):
                binding[s_pat[1:]] = rel.subject
            elif fixed_subject and rel.subject != fixed_subject:
                continue

            if o_pat.startswith("?"):
                binding[o_pat[1:]] = rel.object
            elif fixed_object and rel.object != fixed_object:
                continue

            # 프레디케이트가 변수인 경우
            if p_pat.startswith("?"):
                binding[p_pat[1:]] = rel.predicate.value

            bindings.append(binding)

        return bindings

    def _join_bindings(
        self,
        left: list[dict[str, str]],
        right: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        두 바인딩 세트를 공유 변수 기준으로 JOIN
        """
        if not left:
            return right
        if not right:
            return left

        # 공유 변수 찾기
        left_vars = set(left[0].keys()) if left else set()
        right_vars = set(right[0].keys()) if right else set()
        shared = left_vars & right_vars

        results: list[dict[str, str]] = []
        for lb in left:
            for rb in right:
                # 공유 변수 값이 일치하는지 확인
                if all(lb.get(v) == rb.get(v) for v in shared):
                    merged = {**lb, **rb}
                    results.append(merged)

        return results

    def _apply_filter(self, bindings: list[dict[str, str]], condition: str) -> list[dict[str, str]]:
        """
        FILTER 조건 적용

        지원 연산자: >=, <=, >, <, ==, !=
        예: "?rank >= 10", "?category == lip_care"
        """
        import re as _re

        match = _re.match(r"\?(\w+)\s*(>=|<=|>|<|==|!=)\s*(.+)", condition.strip())
        if not match:
            return bindings

        var_name = match.group(1)
        operator = match.group(2)
        raw_value = match.group(3).strip().strip('"').strip("'")

        results: list[dict[str, str]] = []
        for b in bindings:
            val = b.get(var_name)
            if val is None:
                continue

            # 숫자 비교 시도
            try:
                num_val = float(val)
                num_cmp = float(raw_value)
                passed = (
                    (operator == ">=" and num_val >= num_cmp)
                    or (operator == "<=" and num_val <= num_cmp)
                    or (operator == ">" and num_val > num_cmp)
                    or (operator == "<" and num_val < num_cmp)
                    or (operator == "==" and num_val == num_cmp)
                    or (operator == "!=" and num_val != num_cmp)
                )
            except ValueError:
                # 문자열 비교
                passed = (
                    (operator == "==" and val == raw_value)
                    or (operator == "!=" and val != raw_value)
                    or (operator == ">=" and val >= raw_value)
                    or (operator == "<=" and val <= raw_value)
                    or (operator == ">" and val > raw_value)
                    or (operator == "<" and val < raw_value)
                )

            if passed:
                results.append(b)

        return results
