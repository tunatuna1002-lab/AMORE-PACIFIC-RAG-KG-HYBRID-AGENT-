"""O3 무비용 골드 엣지 도달률 (LLM 호출 없음) [2026-09 사후].

typed 골든(multihop·relation) 문항마다 **골드 엔티티를 입력으로** 검색기의 KG·DB·카드 조립을
돌려, 방출된 증거 카드(관계 카드 + 규칙 입력 카드)에 골드 엣지가 있는지 술어별로 센다.
플래그 ``ontology.use_class_reasoning`` OFF/ON을 비교한다. 연결기(O2)는 쓰지 않는다.

- KG·DB는 **사본**에서 읽는다: ``--snapshot`` 디렉터리의 ``knowledge_graph.json``·
  ``amore_data.db``를 임시 디렉터리로 복사한다(원본에 쓰지 않는다). KG는
  ``auto_save=False``로만 연다. 문서 검색은 빈 결과를 돌려주는 가짜다.
- 엔티티: 골드 ``kg_entities``를 등록부로 분류한다 — 브랜드(등록부 이름 소문자, 등록부 밖은
  밑줄→공백), 그룹(``groups``), 카테고리(id). 제품 슬러그 등 나머지는 넣지 않는다.
- ``--hints``: 질문의 "같은 그룹"·"자매"·"같은 세그먼트" 표현으로 ``relations_hint``를 만든다
  (O2 연결기 동작을 흉내 낸 변형 — 결과표에 따로 적는다).
- 엣지 표기는 평가 러너와 같다: 노드는 소문자 스네이크, ``s -p-> o``.

사용:
    .venv/bin/python scripts/o3_gold_edge_reachability.py \
        --snapshot eval_output/evidence-2026-09/eval_data_snapshot_tagged [--hints] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN_DIR = ROOT / "eval" / "data" / "golden" / "typed"
GOLDEN_FILES = ("multihop.jsonl", "relation.jsonl")
AS_OF = "2026-08-31"
FLAG = "FF_ONTOLOGY_USE_CLASS_REASONING"


def norm_node(name: Any) -> str:
    s = str(name).lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def edge_key(subject: Any, predicate: Any, obj: Any) -> str:
    return f"{norm_node(subject)} -{predicate}-> {norm_node(obj)}"


def gold_predicate(edge: str) -> str:
    return edge.split(" -", 1)[1].split("->", 1)[0]


def canonical_edge(onto: Any, edge: str) -> str:
    """술어 별칭을 정식 이름으로 맞춘 소문자 엣지 (``ownedBy`` ≡ ``ownedByGroup``).

    평가 L3가 골드·방출 양쪽 술어를 온톨로지 정식 이름으로 비교한다고 가정한 참고값이다.
    """
    subject, rest = edge.split(" -", 1)
    predicate, obj = rest.split("->", 1)
    canonical = onto.canonical_predicate(predicate) or predicate
    return f"{subject} -{canonical}->{obj}".lower()


def load_items() -> list[dict[str, Any]]:
    items = []
    for name in GOLDEN_FILES:
        for line in (GOLDEN_DIR / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


def hints_from_question(question: str) -> list[str]:
    hints = []
    if re.search(r"같은 그룹|자매", question):
        hints.append("sibling")
    if re.search(r"같은 세그먼트", question):
        hints.append("segment")
    return hints


def gold_entities(onto: Any, item: dict[str, Any], with_hints: bool) -> dict[str, list[str]]:
    brands: list[str] = []
    groups: list[str] = []
    categories: list[str] = []
    for raw in item["gold"].get("kg_entities") or []:
        text = str(raw).replace("_", " ")
        cid = onto.normalize_category(raw) or onto.normalize_category(text)
        gid = onto.normalize_group(raw) or onto.normalize_group(text)
        bid = onto.normalize_brand(raw) or onto.normalize_brand(text)
        if cid:
            categories.append(cid)
        elif bid:
            brands.append((onto.brand_name(bid) or bid).lower())
        elif gid:
            groups.append(gid)
        elif raw in {"sos", "hhi", "cpi", "usa", "korea"}:
            continue
    entities: dict[str, list[str]] = {
        "brands": brands,
        "categories": categories,
        "groups": groups,
    }
    if with_hints:
        entities["relations_hint"] = hints_from_question(item["question"])
    return entities


class EmptyDocRetriever:
    async def initialize(self) -> None:
        return None

    async def search(self, query: str, top_k: int = 5, **_: Any) -> list[dict[str, Any]]:
        return []


def emitted_edges(ctx: Any) -> set[str]:
    edges: set[str] = set()
    for card in ctx.evidence:
        if card.kind.value == "relation" and card.object is not None:
            edges.add(edge_key(card.subject, card.predicate, card.object))
    for fact in ctx.ontology_facts:  # 평가 러너가 읽는 metric_edges 원문 (수치 엣지 포함)
        if fact.get("type") in ("metric_edges", "ontology_static"):
            for e in fact["data"].get("edges", []):
                edges.add(edge_key(e.get("subject"), e.get("predicate"), e.get("object")))
    return edges


async def run(snapshot: Path, with_hints: bool) -> dict[str, Any]:
    from src.infrastructure.feature_flags import FeatureFlags
    from src.ontology.knowledge_graph import KnowledgeGraph
    from src.ontology.ontology import get_ontology
    from src.rag.hybrid_retriever import HybridRetriever
    from src.rag.metric_facts import MetricFactsProvider

    onto = get_ontology()
    items = load_items()
    results: dict[str, Any] = {"items": {}, "with_hints": with_hints}
    with tempfile.TemporaryDirectory() as tmp:
        kg_path = Path(tmp) / "knowledge_graph.json"
        db_path = Path(tmp) / "amore_data.db"
        shutil.copy2(snapshot / "knowledge_graph.json", kg_path)
        shutil.copy2(snapshot / "amore_data.db", db_path)
        for mode in ("off", "on"):
            os.environ[FLAG] = "true" if mode == "on" else "false"
            FeatureFlags.reset_instance()
            kg = KnowledgeGraph(persist_path=str(kg_path), auto_save=False)
            retriever = HybridRetriever(
                knowledge_graph=kg,
                doc_retriever=EmptyDocRetriever(),
                metric_facts_provider=MetricFactsProvider(db_path, as_of=AS_OF),
            )
            for item in items:
                entities = gold_entities(onto, item, with_hints)
                retriever.entity_extractor.extract = lambda q, knowledge_graph=None, e=entities: {
                    k: list(v) for k, v in e.items()
                }
                ctx = await retriever.retrieve(item["question"])
                # 평가 L3와 같이 엣지 전체를 소문자로 비교한다 (eval/metrics/l3_kg.py _norm_edges)
                emitted = emitted_edges(ctx)
                found = {e.lower() for e in emitted}
                found_alias = {canonical_edge(onto, e) for e in emitted}
                gold = [e.strip() for e in item["gold"].get("kg_edges") or []]
                row = results["items"].setdefault(item["id"], {"gold": gold})
                row[mode] = {
                    "hit": [e for e in gold if e.lower() in found],
                    "hit_alias": [e for e in gold if canonical_edge(onto, e) in found_alias],
                    "evidence": len(ctx.evidence),
                    "prompt": len(ctx.prompt_evidence),
                    "ontology_cards": (ctx.metadata.get("ontology") or {}).get("ontology_cards", 0),
                    "error": ctx.metadata.get("retrieval_error"),
                }
        # 원본 사본이 바뀌지 않았는지 (auto_save=False)
        results["kg_copy_unchanged"] = (
            kg_path.read_bytes() == (snapshot / "knowledge_graph.json").read_bytes()
        )
    return results


def summarize(results: dict[str, Any]) -> str:
    totals: Counter[str] = Counter()
    hits = {"off": Counter(), "on": Counter(), "off_alias": Counter(), "on_alias": Counter()}
    cards = {"off": [], "on": [], "on_onto": [], "off_prompt": [], "on_prompt": []}
    for row in results["items"].values():
        for edge in row["gold"]:
            predicate = gold_predicate(edge)
            totals[predicate] += 1
            for mode in ("off", "on"):
                if edge in row[mode]["hit"]:
                    hits[mode][predicate] += 1
                if edge in row[mode]["hit_alias"]:
                    hits[f"{mode}_alias"][predicate] += 1
        for mode in ("off", "on"):
            cards[mode].append(row[mode]["evidence"])
            cards[f"{mode}_prompt"].append(row[mode]["prompt"])
        cards["on_onto"].append(row["on"]["ontology_cards"])
    lines = [
        "| 술어 | 골드 | OFF | ON | OFF (별칭 동일시) | ON (별칭 동일시) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    columns = ("off", "on", "off_alias", "on_alias")
    for predicate, total in totals.most_common():
        cells = " | ".join(str(hits[c][predicate]) for c in columns)
        lines.append(f"| {predicate} | {total} | {cells} |")
    all_gold = sum(totals.values())
    cells = " | ".join(str(sum(hits[c].values())) for c in columns)
    lines.append(f"| **전체** | {all_gold} | {cells} |")
    n = len(results["items"])

    def avg(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    lines.append("")
    lines.append(
        f"문항 {n}개 · 평균 카드 수(전체 evidence) OFF {avg(cards['off']):.1f} → "
        f"ON {avg(cards['on']):.1f} · 프롬프트 카드 OFF {avg(cards['off_prompt']):.1f} → "
        f"ON {avg(cards['on_prompt']):.1f} · 그중 온톨로지 카드(ON) {avg(cards['on_onto']):.1f} · "
        f"최대 프롬프트 카드(ON) {max(cards['on_prompt']) if cards['on_prompt'] else 0}"
    )
    errors = [i for i, r in results["items"].items() if r["off"]["error"] or r["on"]["error"]]
    lines.append(
        f"검색 오류 문항: {errors or '없음'} · KG 사본 불변: {results['kg_copy_unchanged']}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--hints", action="store_true", help="질문 표현으로 relations_hint 생성")
    parser.add_argument("--json", type=Path, default=None, help="문항별 결과 저장 경로")
    args = parser.parse_args()
    results = asyncio.run(run(args.snapshot.resolve(), args.hints))
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summarize(results))


if __name__ == "__main__":
    main()
