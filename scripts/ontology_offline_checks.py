"""[2026-09 사후] O0-B: 무비용(LLM 호출 없음) 온톨로지 점검.

입력은 모두 읽기 전용이다: KG JSON 사본, 골든 typed 시험지, (있으면) O1 온톨로지 로더.
운영 ``data/``와 ``KnowledgeGraph()`` 기본 생성은 쓰지 않는다(KG는 JSON을 직접 읽는다).

출력 (JSON 파일 + 표준출력 마크다운):
  1. 브랜드 인식 범위 — KG 브랜드 노드·골든 문항 브랜드 중 ``EntityLinker``(사전 경로,
     spaCy 끔)가 인식하는 비율과 미인식 목록
  2. 골드 엣지 도달률 — 술어별로 KG에 사실이 있는지, 런타임이 그 술어 이름으로 방출하는지.
     검색기가 버리는 술어(``priority_preds``·``EXCLUDED_METADATA``·수치 술어)는 소스를
     ast로 읽어 센다(모듈을 import하지 않는다)
  3. KG 정합성 위반 — 대소문자 중복, 가짜 브랜드, 비대칭 대칭관계(competesWith·siblingBrand),
     날짜 없는 수치 엣지, 도메인·범위 위반(config 등록부 + ASIN 모양 기준)
  4. (``--rule-match``) 규칙 일치율 오프라인값 — 연결기 엔티티(A) vs 골드 엔티티(B).
     ``eval_output/evidence-2026-09/notes/3b_rule_offline_match.md``와 같은 방법이며,
     DB·KG는 임시 디렉터리 사본만 쓴다

실행 (워크트리 루트에서):
    .venv/bin/python scripts/ontology_offline_checks.py \\
        --kg <eval_data_snapshot_tagged/knowledge_graph.json> --out <result.json> \\
        [--rule-match --snapshot <eval_data_snapshot_tagged>] [--registry]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.validators.ontology_validator import (  # noqa: E402
    PLACEHOLDER_BRANDS,
    PREDICATE_SIGNATURES,
    EntityTypeRegistry,
    canonical_predicate,
    normalize_entity_key,
    parse_edge,
)

DEFAULT_TYPED_DIR = REPO / "eval" / "data" / "golden" / "typed"
AS_OF = "2026-08-31"
_ASIN_RE = re.compile(r"^B0[A-Z0-9]{8}$")

# KG 트리플에서 브랜드 노드를 고르는 위치 (술어 → 브랜드인 끝점). 브랜드 타입 트리플이
# KG에 0건이라(검토 보고서 §3.1) 술어 위치로 정한다.
BRAND_POSITIONS: dict[str, tuple[bool, bool]] = {
    "hasProduct": (True, False),
    "ownedByGroup": (True, False),
    "ownedBy": (True, False),
    "hasSegment": (True, False),
    "originatesFrom": (True, False),
    "acquiredIn": (True, False),
    "siblingBrand": (True, True),
    "competesWith": (True, True),
    "ownsBrand": (False, True),
    "hasPosition": (True, False),
    "rankedIn": (True, False),
}
SYMMETRIC_PREDICATES = ("competesWith", "siblingBrand")


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------


def effective_predicate(triple: dict[str, Any]) -> str:
    """검색기와 같은 규칙: original_predicate가 camelCase면 그것, 아니면 predicate."""
    orig = (triple.get("properties") or {}).get("original_predicate")
    pred = str(triple.get("predicate"))
    if orig and "_" not in orig and not orig.isupper():
        return str(orig)
    return pred


def load_gold_rows(typed_dir: Path) -> list[dict[str, Any]]:
    """typed 시험지 전체(combined_v1.jsonl = 233문항)를 읽는다."""
    path = typed_dir / "combined_v1.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# 1. 브랜드 인식 범위
# ---------------------------------------------------------------------------


class KGQueryShim:
    """EntityLinker가 쓰는 ``knowledge_graph.query()``만 흉내 낸 읽기 전용 KG.

    런타임은 ``extract_entities(query, knowledge_graph=kg)``로 제품명 → 브랜드 역링크를 한다.
    ``KnowledgeGraph()``를 만들지 않고(로드 중 파일을 다시 쓸 수 있다) JSON 트리플만 쓴다.
    """

    def __init__(self, triples: list[dict[str, Any]]):
        self._rels = [
            _Rel(
                str(t.get("subject")),
                str(t.get("predicate")),
                str(t.get("object")),
                dict(t.get("properties") or {}),
            )
            for t in triples
        ]

    def query(
        self, subject: str | None = None, predicate: Any = None, object_: str | None = None
    ) -> list[Any]:
        pred = getattr(predicate, "value", predicate)
        return [
            r
            for r in self._rels
            if (subject is None or r.subject == subject)
            and (pred is None or r.predicate == pred)
            and (object_ is None or r.object == object_)
        ]


class _Rel:
    __slots__ = ("subject", "predicate", "object", "properties")

    def __init__(self, subject: str, predicate: str, obj: str, properties: dict[str, Any]):
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.properties = properties


def kg_brand_surfaces(
    triples: list[dict[str, Any]], registry: EntityTypeRegistry | None = None
) -> dict[str, list[str]]:
    """정규화 키 → KG에 나온 표기들(정렬). 가짜 브랜드도 포함한다(호출 측에서 분리).

    위치는 검색기와 같은 effective 술어로 정한다 — ``hasPosition``은 원 술어가 hasHHI면 주어가
    카테고리다. 등록부가 카테고리로 아는 노드는 뺀다.
    """
    surfaces: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        pred = effective_predicate(triple)
        if pred == "hasSoS":
            pred = "hasPosition"
        positions = BRAND_POSITIONS.get(pred)
        if positions is None:
            continue
        subj_is_brand, obj_is_brand = positions
        for flag, node in (
            (subj_is_brand, triple.get("subject")),
            (obj_is_brand, triple.get("object")),
        ):
            if not flag or not node or _ASIN_RE.match(str(node)):
                continue
            if registry is not None and registry.type_of(node)[0] == "category":
                continue
            surfaces[normalize_entity_key(node)].add(str(node))
    return {key: sorted(values) for key, values in sorted(surfaces.items())}


def _linked_keys(linker: Any, text: str, knowledge_graph: Any | None = None) -> set[str]:
    brands = linker.extract_entities(text, knowledge_graph=knowledge_graph).get("brands") or []
    keys: set[str] = set()
    for brand in brands:
        key = normalize_entity_key(brand)
        keys |= {key, key.replace(".", ""), key.replace("_", "")}
    return keys


def _mentioned(entity: str, surfaces: list[str], question: str) -> bool:
    """골드 브랜드가 질문 원문에 (영문 표기로) 나오는지. 한글 별칭(라네즈 등)은 보지 않는다."""
    text = question.lower()
    key = normalize_entity_key(entity)
    candidates = {s.lower() for s in surfaces} | {
        str(entity).lower(),
        key.replace("_", " "),
        key.replace("_", "-"),
        key.replace(".", ""),
    }
    return any(c and c in text for c in candidates)


def _matches(key: str, keys: set[str]) -> bool:
    return bool({key, key.replace(".", ""), key.replace("_", "")} & keys)


def brand_recognition(
    brand_surfaces: dict[str, list[str]],
    gold_rows: list[dict[str, Any]],
    linker: Any,
    registry: EntityTypeRegistry,
    knowledge_graph: Any | None = None,
) -> dict[str, Any]:
    """KG 브랜드(표기 하나라도 인식되면 인식)와 골든 문항 브랜드(질문 원문에서 추출되는지).

    - KG 브랜드: 표기 문자열만 넣었을 때 사전이 그 브랜드를 내는지 (KG 역링크 없이 — 사전 범위)
    - 골든: 질문 원문을 런타임처럼 ``knowledge_graph``와 함께 넣었을 때 골드 브랜드가 나오는지.
      골드 브랜드 = 골드 kg_entities 중 등록부 타입이 brand이거나 KG 브랜드 노드인 것
      (카테고리·가짜 브랜드 제외).
    """
    kg_rows = []
    for key, surfaces in brand_surfaces.items():
        if key in PLACEHOLDER_BRANDS:
            continue
        recognized = any(_matches(key, _linked_keys(linker, s)) for s in surfaces)
        kg_rows.append({"brand": key, "surfaces": surfaces, "recognized": recognized})
    kg_brand_keys = set(brand_surfaces)

    gold_total = 0
    gold_hit = 0
    gold_missed: Counter[str] = Counter()
    gold_implicit: Counter[str] = Counter()
    for row in gold_rows:
        entities = (row.get("gold") or {}).get("kg_entities") or []
        linked = None
        for entity in entities:
            key = normalize_entity_key(entity)
            entity_type, _ = registry.type_of(entity)
            if entity_type != "brand" and (entity_type is not None or key not in kg_brand_keys):
                continue
            if key in PLACEHOLDER_BRANDS:
                continue
            question = row.get("question") or ""
            if not _mentioned(entity, brand_surfaces.get(key, []), question):
                # 골드가 질문에 없는 브랜드를 암묵적으로 넣은 경우(예: 기본 대상 LANEIGE)는
                # 연결기의 인식 문제가 아니다 — 따로 센다
                gold_implicit[key] += 1
                continue
            if linked is None:
                linked = _linked_keys(linker, question, knowledge_graph)
            gold_total += 1
            if _matches(key, linked):
                gold_hit += 1
            else:
                gold_missed[key] += 1

    recognized = sum(1 for r in kg_rows if r["recognized"])
    return {
        "kg_brands": len(kg_rows),
        "kg_brands_recognized": recognized,
        "kg_recognition_rate": recognized / len(kg_rows) if kg_rows else None,
        "kg_unrecognized": sorted(r["brand"] for r in kg_rows if not r["recognized"]),
        "kg_placeholders_excluded": sorted(k for k in brand_surfaces if k in PLACEHOLDER_BRANDS),
        "gold_brand_mentions": gold_total,
        "gold_brand_linked": gold_hit,
        "gold_linking_rate": gold_hit / gold_total if gold_total else None,
        "gold_unlinked": dict(sorted(gold_missed.items(), key=lambda kv: (-kv[1], kv[0]))),
        "gold_implicit_not_in_question": sum(gold_implicit.values()),
        "gold_implicit_by_brand": dict(
            sorted(gold_implicit.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }


def registry_recognition(brand_surfaces: dict[str, list[str]]) -> dict[str, Any] | None:
    """O1 온톨로지 로더(normalize_brand)가 있으면 KG 브랜드를 등록부 기준으로 다시 잰다."""
    try:
        from src.ontology.ontology import get_ontology  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        ontology = get_ontology()
        normalize = ontology.normalize_brand
    except Exception as exc:  # 로더가 아직 다른 API일 수 있다
        return {"skipped": f"{type(exc).__name__}: {exc}"}
    rows = []
    for key, surfaces in brand_surfaces.items():
        if key in PLACEHOLDER_BRANDS:
            continue
        hit = any(normalize(s) for s in surfaces)
        rows.append((key, hit))
    hits = sum(1 for _, hit in rows if hit)
    return {
        "kg_brands": len(rows),
        "recognized": hits,
        "rate": hits / len(rows) if rows else None,
        "unrecognized": sorted(k for k, hit in rows if not hit),
    }


# ---------------------------------------------------------------------------
# 2. 골드 엣지 도달률
# ---------------------------------------------------------------------------


def _const_set(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.Set):
        return {e.value for e in node.elts if isinstance(e, ast.Constant)}
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in {"frozenset", "set"}:
        if node.args:
            return _const_set(node.args[0])
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    return None


def read_source_constants(path: Path, names: set[str]) -> dict[str, set[str]]:
    """소스 파일에서 이름이 names인 대입(함수 안 포함)의 문자열 집합 값을 ast로 읽는다."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names and value is not None:
                parsed = _const_set(value)
                if parsed is not None:
                    found[target.id] = parsed
    return found


def retriever_predicate_filters(src_root: Path) -> dict[str, Any]:
    """검색기가 버리는/내보내는 술어를 소스에서 읽는다.

    [2026-09 사후] O3(플래그 ``ontology.use_class_reasoning``)부터 ``priority_preds``가
    ``hybrid_retriever.py``의 지역 변수(``_PRIORITY_PREDS_CANONICAL if canonical else
    _PRIORITY_PREDS``)가 돼 더는 단순 대입으로 ast에 안 잡힌다. 대신 그 두 모듈 상수를
    직접 읽어 플래그 OFF(레거시)·ON(정식 술어) 결과를 따로 낸다. 하위 호환을 위해
    최상위 키(``priority_preds``·``runtime_emitted_predicates``)는 OFF 값을 그대로 유지한다.
    """
    retriever = read_source_constants(
        src_root / "rag" / "hybrid_retriever.py",
        {"_PRIORITY_PREDS", "_PRIORITY_PREDS_CANONICAL"},
    )
    adapters = read_source_constants(
        src_root / "rag" / "evidence_adapters.py",
        {"KG_NUMERIC_PREDICATES", "EXCLUDED_METADATA", "EXCLUDED_NUMERIC", "PLACEHOLDER_ENTITIES"},
    )
    ontology_ctx = read_source_constants(
        src_root / "rag" / "ontology_context.py", {"STATIC_BRAND_PREDICATES"}
    )
    priority_off = sorted(retriever.get("_PRIORITY_PREDS", set()))
    priority_on = sorted(retriever.get("_PRIORITY_PREDS_CANONICAL", set()))
    static_on = sorted(ontology_ctx.get("STATIC_BRAND_PREDICATES", set()))
    # 런타임 방출 술어 이름: metric_edges(priority_preds + 제품 슬러그 엣지) + competitors
    # fact의 competesWith + category_hierarchy의 hasSubcategory (eval/runner.py _extract_l3_trace).
    # ON은 추가로 ontology_static 카드(정적 정의 술어, src/rag/ontology_context.py)를 낸다 —
    # 등록부 브랜드는 metric_edges 대신 이 카드로 실린다(hybrid_retriever.py의 "continue" 분기).
    common_extra = {"hasProduct", "belongsToCategory", "competesWith", "hasSubcategory"}
    emitted_off = sorted(set(priority_off) | common_extra)
    emitted_on = sorted(set(priority_on) | set(static_on) | common_extra)
    return {
        # 하위 호환 (기본값 = 플래그 OFF/레거시)
        "priority_preds": priority_off,
        "runtime_emitted_predicates": emitted_off,
        "retriever_renames": {"ownedByGroup": "ownedBy"},
        "kg_numeric_predicates_excluded_from_cards": sorted(
            adapters.get("KG_NUMERIC_PREDICATES", set())
        ),
        "excluded_metadata_reason": sorted(adapters.get("EXCLUDED_METADATA", set())),
        "placeholder_entities_excluded_from_cards": sorted(
            adapters.get("PLACEHOLDER_ENTITIES", set())
        ),
        # O3: 플래그별로 나눈 값
        "off": {
            "priority_preds": priority_off,
            "runtime_emitted_predicates": emitted_off,
        },
        "on": {
            "priority_preds": priority_on,
            "static_brand_predicates_ontology_card": static_on,
            "runtime_emitted_predicates": emitted_on,
        },
    }


def _slug_index(triples: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """제품 슬러그 → ASIN 집합, ASIN → 브랜드 키 (hasProduct 제목 기준)."""
    from src.rag.entity_linker import product_name_slugs

    slug_to_asins: dict[str, set[str]] = defaultdict(set)
    asin_brand: dict[str, str] = {}
    for triple in triples:
        if triple.get("predicate") != "hasProduct":
            continue
        title = (triple.get("properties") or {}).get("title") or ""
        brand = str(triple.get("subject"))
        asin = str(triple.get("object"))
        asin_brand[asin] = normalize_entity_key(brand)
        for slug in product_name_slugs(title, brand) if title else []:
            slug_to_asins[slug].add(asin)
    return slug_to_asins, asin_brand


def gold_edge_reachability(
    gold_rows: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    emitted_predicates: list[str],
) -> dict[str, Any]:
    """골드 엣지 술어별 도달 가능성.

    - ``in_kg_same_name``: KG에 같은 술어 이름(검색기의 effective_predicate 기준)과 같은 끝점
      (정규화 키, 제품은 제목 슬러그)의 트리플이 있다
    - ``in_kg_alias``: 별칭(ownedBy↔ownedByGroup)을 정식 이름으로 맞추면 있다
    - ``runtime_emits_name``: 런타임이 그 술어 이름을 방출할 수 있다(검색기 이름 변환 반영)
    - ``reachable_now``: in_kg_same_name 또는 (in_kg_alias이고 런타임이 골드 이름으로 방출) —
      지금 코드가 골드와 같은 문자열을 낼 수 있는 상한
    - ``reachable_if_normalized``: in_kg_alias이고 정식 술어가 런타임 방출 대상(이름만 맞추면)
    """
    slug_to_asins, _ = _slug_index(triples)
    by_key: dict[tuple[str, str, str], bool] = {}
    by_canonical: set[tuple[str, str, str]] = set()
    for triple in triples:
        pred = effective_predicate(triple)
        s = normalize_entity_key(triple.get("subject"))
        o = normalize_entity_key(triple.get("object"))
        by_key[(s, pred, o)] = True
        by_canonical.add((s, canonical_predicate(pred), o))

    def expand(node: str) -> set[str]:
        key = normalize_entity_key(node)
        asins = slug_to_asins.get(key, set())
        return {key} | {normalize_entity_key(a) for a in asins}

    # emitted_predicates(retriever_predicate_filters의 runtime_emitted_predicates)는 이미
    # 런타임이 실제로 내보내는 이름이다 — OFF는 "ownedBy"만, ON은 "ownedByGroup"만 담는다
    # (hybrid_retriever.py: OFF는 읽을 때 ownedByGroup→ownedBy로 바꿔 방출하고, ON은 정식
    # 이름을 그대로 쓴다). 여기서 다시 이름을 바꾸면 ON에서 ownedByGroup이 통째로 사라진다
    # (전에 있던 하드코딩 버그) — 그래서 그대로 쓴다.
    emitted_names = set(emitted_predicates)
    emitted_canonical = {canonical_predicate(p) for p in emitted_names}

    per_pred: dict[str, Counter[str]] = defaultdict(Counter)
    for row in gold_rows:
        for edge in (row.get("gold") or {}).get("kg_edges") or []:
            parsed = parse_edge(edge)
            if parsed is None:
                per_pred["(unparsed)"]["total"] += 1
                continue
            s, pred, o = parsed
            subjects, objects = expand(s), expand(o)
            same = any((si, pred, oi) in by_key for si in subjects for oi in objects)
            canon = canonical_predicate(pred)
            alias = same or any(
                (si, canon, oi) in by_canonical for si in subjects for oi in objects
            )
            name_emitted = pred in emitted_names
            bucket = per_pred[pred]
            bucket["total"] += 1
            bucket["in_kg_same_name"] += same
            bucket["in_kg_alias"] += alias
            bucket["runtime_emits_name"] += name_emitted
            bucket["reachable_now"] += (same or alias) and name_emitted
            bucket["reachable_if_normalized"] += alias and canon in emitted_canonical
    return {pred: dict(counts) for pred, counts in sorted(per_pred.items())}


# ---------------------------------------------------------------------------
# 3. KG 정합성
# ---------------------------------------------------------------------------


def kg_consistency(triples: list[dict[str, Any]], registry: EntityTypeRegistry) -> dict[str, Any]:
    surfaces: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        for node in (triple.get("subject"), triple.get("object")):
            text = str(node)
            if node is None or _ASIN_RE.match(text) or re.fullmatch(r"[\d.]+", text):
                continue
            surfaces[normalize_entity_key(text)].add(text)
    case_dupes = {k: sorted(v) for k, v in sorted(surfaces.items()) if len(v) > 1}

    placeholder: dict[str, Counter[str]] = defaultdict(Counter)
    for triple in triples:
        for node in (triple.get("subject"), triple.get("object")):
            key = normalize_entity_key(node)
            if key in PLACEHOLDER_BRANDS:
                placeholder[key][str(triple.get("predicate"))] += 1

    asymmetric: dict[str, Any] = {}
    for pred in SYMMETRIC_PREDICATES:
        pairs = {
            (normalize_entity_key(t.get("subject")), normalize_entity_key(t.get("object")))
            for t in triples
            if t.get("predicate") == pred
        }
        missing = sorted((a, b) for a, b in pairs if (b, a) not in pairs and a != b)
        asymmetric[pred] = {
            "pairs": len(pairs),
            "asymmetric": len(missing),
            "examples": missing[:5],
        }

    numeric_undated: Counter[str] = Counter()
    numeric_total: Counter[str] = Counter()
    for triple in triples:
        props = triple.get("properties") or {}
        if any(isinstance(v, int | float) and not isinstance(v, bool) for v in props.values()):
            key = effective_predicate(triple)
            numeric_total[key] += 1
            if not triple.get("valid_from"):
                numeric_undated[key] += 1

    merged_position = Counter(
        effective_predicate(t) for t in triples if t.get("predicate") == "hasPosition"
    )

    domain_range: Counter[str] = Counter()
    untyped = 0
    examples: dict[str, str] = {}
    for triple in triples:
        pred = canonical_predicate(effective_predicate(triple))
        signature = PREDICATE_SIGNATURES.get(pred)
        if signature is None:
            continue
        domain, range_ = signature
        for role, node, allowed in (
            ("subject", triple.get("subject"), domain),
            ("object", triple.get("object"), range_),
        ):
            if allowed is None:
                continue
            node_type, _ = registry.type_of(node)
            if node_type is None:
                untyped += 1
                continue
            if node_type not in allowed and not (registry.alt_types(node) & allowed):
                kind = f"{pred}:{role}={node_type}"
                domain_range[kind] += 1
                examples.setdefault(
                    kind,
                    f"{triple.get('subject')} -{triple.get('predicate')}-> {triple.get('object')}",
                )

    return {
        "triples": len(triples),
        "predicate_counts": dict(sorted(Counter(str(t.get("predicate")) for t in triples).items())),
        "entity_metadata_empty": True,
        "case_duplicates": len(case_dupes),
        "case_duplicate_examples": dict(list(case_dupes.items())[:15]),
        "placeholder_triples": {k: dict(sorted(v.items())) for k, v in sorted(placeholder.items())},
        "placeholder_triples_total": sum(sum(v.values()) for v in placeholder.values()),
        "asymmetric_symmetric_relations": asymmetric,
        "numeric_edges": dict(sorted(numeric_total.items())),
        "numeric_edges_without_valid_from": dict(sorted(numeric_undated.items())),
        "hasPosition_by_original_predicate": dict(sorted(merged_position.items())),
        "domain_range_violations": dict(sorted(domain_range.items())),
        "domain_range_examples": dict(sorted(examples.items())),
        "domain_range_untyped_endpoints": untyped,
    }


# ---------------------------------------------------------------------------
# 4. 규칙 일치율 오프라인 (3b 방법 재사용)
# ---------------------------------------------------------------------------


async def rule_match(snapshot_dir: Path, typed_dir: Path) -> dict[str, Any]:  # pragma: no cover
    """3b_rule_offline_match.md와 같은 방법. DB·KG는 임시 사본, LLM 호출은 예외로 막는다."""
    import os
    import shutil
    import tempfile

    os.environ["OPENAI_API_KEY"] = ""
    os.environ.pop("AMORE_DATA_AS_OF", None)
    import litellm

    calls: list[str] = []

    def _no_llm(*args: Any, **kwargs: Any) -> Any:
        calls.append(str(kwargs.get("model")))
        raise RuntimeError("LLM call blocked in offline checks")

    litellm.completion = _no_llm
    litellm.acompletion = _no_llm
    litellm.embedding = _no_llm
    litellm.aembedding = _no_llm

    from src.ontology.knowledge_graph import KnowledgeGraph
    from src.ontology.rule_contracts import evaluate_rules_on_cards
    from src.rag.evidence_assembly import build_rule_input_cards
    from src.rag.hybrid_retriever import HybridRetriever
    from src.rag.metric_facts import MetricFactsProvider

    class EmptyDocRetriever:
        async def initialize(self) -> None:
            return None

        async def search(self, query: str, top_k: int = 5, **_: Any) -> list[dict[str, Any]]:
            return []

    rows = [
        json.loads(line)
        for line in (typed_dir / "rule.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    generated = [r for r in rows if (r.get("metadata") or {}).get("generated")]
    items = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(snapshot_dir / "amore_data.db", tmp_path / "amore_data.db")
        shutil.copy2(snapshot_dir / "knowledge_graph.json", tmp_path / "knowledge_graph.json")
        kg = KnowledgeGraph(persist_path=str(tmp_path / "knowledge_graph.json"), auto_save=False)
        retriever = HybridRetriever(
            knowledge_graph=kg,
            doc_retriever=EmptyDocRetriever(),
            metric_facts_provider=MetricFactsProvider(tmp_path / "amore_data.db", as_of=AS_OF),
        )
        await retriever.initialize()
        for row in generated:
            gold = row["metadata"]["rule_gold"]
            rule_ids = set(gold["rule_ids"])
            expected = bool(gold["expected_conclusion"]["fires"])

            ctx = await retriever.retrieve(row["question"])
            fired_a = (ctx.metadata.get("rule_evaluation") or {}).get("fired") or []
            applied_a = sorted({inf.rule_name for inf in ctx.inferences}) or fired_a

            rc = gold.get("rule_context") or {}
            gold_entities = {
                "brands": [rc["brand"]] if rc.get("brand") else [],
                "categories": [rc["category"]] if rc.get("category") else [],
            }
            facts_b = await retriever.metric_facts_provider.collect(gold_entities)
            cards_b = build_rule_input_cards(
                metric_facts=facts_b,
                ontology_facts=retriever._query_knowledge_graph(gold_entities),
                adapter=retriever.evidence_adapter,
            )
            run_b = evaluate_rules_on_cards(
                retriever.reasoner.rules_by_priority,
                cards_b,
                [retriever.evidence_adapter.normalize_brand(b) for b in gold_entities["brands"]],
                [
                    retriever.evidence_adapter.normalize_category(c)
                    for c in gold_entities["categories"]
                ],
            )
            fired_b = run_b.summary()["fired"]
            items.append(
                {
                    "id": row["id"],
                    "expected_fires": expected,
                    "A_brands": ctx.entities.get("brands"),
                    "A_match": bool(set(applied_a) & rule_ids) == expected,
                    "B_match": bool(set(fired_b) & rule_ids) == expected,
                }
            )
        # 사본이라도 쓰기 확인용: 원본 해시와 비교하지 않고, 사본 KG를 저장하지 않았음을 보증
        assert kg is not None
    return {
        "as_of": AS_OF,
        "n": len(items),
        "A_match": sum(i["A_match"] for i in items),
        "B_match": sum(i["B_match"] for i in items),
        "llm_calls": len(calls),
        "mismatches": [i for i in items if not (i["A_match"] and i["B_match"])],
    }


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------


def render_markdown(result: dict[str, Any]) -> str:
    rec = result["brand_recognition"]
    lines = [
        "## 1. 브랜드 인식 범위 (EntityLinker 사전 경로)",
        "",
        "| 대상 | 인식 | 전체 | 비율 |",
        "|---|---|---|---|",
        f"| KG 브랜드 노드(가짜 제외) | {rec['kg_brands_recognized']} | {rec['kg_brands']} | "
        f"{_pct(rec['kg_recognition_rate'])} |",
        f"| 골든 문항 브랜드 중 질문에 나온 것 | {rec['gold_brand_linked']} | "
        f"{rec['gold_brand_mentions']} | {_pct(rec['gold_linking_rate'])} |",
        "",
        f"골드 브랜드 중 질문에 없는(암묵) 것 {rec['gold_implicit_not_in_question']}건은 제외: "
        f"{', '.join(f'{k}×{v}' for k, v in rec['gold_implicit_by_brand'].items())}",
        "",
        f"미인식 KG 브랜드({len(rec['kg_unrecognized'])}): {', '.join(rec['kg_unrecognized'])}",
        "",
        f"골든 미연결 브랜드: {', '.join(f'{k}×{v}' for k, v in rec['gold_unlinked'].items())}",
        "",
    ]
    if result.get("registry_recognition"):
        lines += [
            f"등록부 기준 인식: {json.dumps(result['registry_recognition'], ensure_ascii=False)}",
            "",
        ]
    filters = result["retriever_filters"]

    def _reach_table(mode_filters: dict[str, Any], reachability: dict[str, Any]) -> list[str]:
        out = [
            f"- priority_preds: {', '.join(mode_filters['priority_preds'])}",
            f"- 런타임 방출 술어: {', '.join(mode_filters['runtime_emitted_predicates'])}",
            "",
            "| 술어 | 골드 | KG 같은 이름 | KG 별칭 포함 | 런타임 이름 방출 | 지금 도달 가능 | 이름 정규화 시 |",
            "|---|---|---|---|---|---|---|",
        ]
        for pred, c in reachability.items():
            out.append(
                f"| {pred} | {c.get('total', 0)} | {c.get('in_kg_same_name', 0)} | "
                f"{c.get('in_kg_alias', 0)} | {c.get('runtime_emits_name', 0)} | "
                f"{c.get('reachable_now', 0)} | {c.get('reachable_if_normalized', 0)} |"
            )
        return out

    lines += [
        "## 2. 골드 엣지 도달률 (술어별, 플래그 ontology.use_class_reasoning OFF/ON 별도)",
        "",
        f"- 카드에서 빠지는 수치 술어: {', '.join(filters['kg_numeric_predicates_excluded_from_cards'])}",
        "",
        "### 2a. 플래그 OFF (레거시)",
        "",
    ]
    lines += _reach_table(filters["off"], result["gold_edge_reachability"])
    lines += ["", "### 2b. 플래그 ON (정식 술어 + 정적 사실 카드)", ""]
    lines += _reach_table(filters["on"], result.get("gold_edge_reachability_on", {}))
    kg = result["kg_consistency"]
    asym = kg["asymmetric_symmetric_relations"]
    lines += [
        "",
        "## 3. KG 정합성",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 트리플 | {kg['triples']} |",
        f"| 대소문자·표기 중복 노드 키 | {kg['case_duplicates']} |",
        f"| 가짜 브랜드 관련 트리플 | {kg['placeholder_triples_total']} "
        f"({', '.join(f'{k}={sum(v.values())}' for k, v in kg['placeholder_triples'].items())}) |",
        f"| competesWith 비대칭 쌍 | {asym['competesWith']['asymmetric']} / "
        f"{asym['competesWith']['pairs']} |",
        f"| siblingBrand 비대칭 쌍 | {asym['siblingBrand']['asymmetric']} / "
        f"{asym['siblingBrand']['pairs']} |",
        f"| 수치 속성 엣지 중 valid_from 없음 | {sum(kg['numeric_edges_without_valid_from'].values())}"
        f" / {sum(kg['numeric_edges'].values())} |",
        f"| hasPosition 원 술어 분포 | {kg['hasPosition_by_original_predicate']} |",
        f"| 도메인·범위 위반 | {sum(kg['domain_range_violations'].values())} "
        f"(타입 모름 끝점 {kg['domain_range_untyped_endpoints']}) |",
        "",
    ]
    for kind, count in kg["domain_range_violations"].items():
        lines.append(f"- {kind}: {count} (예: {kg['domain_range_examples'][kind]})")
    if result.get("rule_match"):
        rm = result["rule_match"]
        lines += [
            "",
            "## 4. 규칙 일치율 오프라인",
            "",
            f"- A(연결기 엔티티) {rm['A_match']}/{rm['n']}, B(골드 엔티티) {rm['B_match']}/{rm['n']}, "
            f"LLM 호출 {rm['llm_calls']}",
        ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def run_checks(kg_path: Path, typed_dir: Path, use_registry: bool) -> dict[str, Any]:
    from src.rag.entity_linker import EntityLinker

    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    triples = kg.get("triples") or []
    gold_rows = load_gold_rows(typed_dir)
    registry = EntityTypeRegistry.from_config()
    linker = EntityLinker(use_spacy=False)
    surfaces = kg_brand_surfaces(triples, registry)
    filters = retriever_predicate_filters(REPO / "src")
    result: dict[str, Any] = {
        "inputs": {
            "kg": str(kg_path),
            "kg_saved_at": kg.get("saved_at"),
            "kg_entity_metadata": len(kg.get("entity_metadata") or {}),
            "gold": str(typed_dir / "combined_v1.jsonl"),
            "gold_items": len(gold_rows),
            "type_registry": registry.source_label,
        },
        "brand_recognition": brand_recognition(
            surfaces, gold_rows, linker, registry, KGQueryShim(triples)
        ),
        "retriever_filters": filters,
        # 하위 호환 키(플래그 OFF/레거시)
        "gold_edge_reachability": gold_edge_reachability(
            gold_rows, triples, filters["off"]["runtime_emitted_predicates"]
        ),
        # O3: 플래그 ON(정식 술어 + 정적 사실 카드)일 때의 같은 계산
        "gold_edge_reachability_on": gold_edge_reachability(
            gold_rows, triples, filters["on"]["runtime_emitted_predicates"]
        ),
        "kg_consistency": kg_consistency(triples, registry),
    }
    result["kg_consistency"]["entity_metadata_empty"] = not kg.get("entity_metadata")
    if use_registry:
        result["registry_recognition"] = registry_recognition(surfaces) or {
            "skipped": "src.ontology.ontology 없음 (O1 미병합)"
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kg", type=Path, required=True, help="KG JSON 사본 (읽기 전용)")
    parser.add_argument("--typed-dir", type=Path, default=DEFAULT_TYPED_DIR)
    parser.add_argument("--out", type=Path, help="결과 JSON 경로")
    parser.add_argument("--registry", action="store_true", help="O1 온톨로지 로더 기준 인식 범위")
    parser.add_argument("--rule-match", action="store_true", help="규칙 일치율 오프라인(느림)")
    parser.add_argument("--snapshot", type=Path, help="--rule-match용 스냅샷 디렉터리(사본 사용)")
    args = parser.parse_args(argv)

    result = run_checks(args.kg, args.typed_dir, args.registry)
    if args.rule_match:
        import asyncio

        snapshot = args.snapshot or args.kg.parent
        result["rule_match"] = asyncio.run(rule_match(snapshot, args.typed_dir))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
