"""골든셋 L2 골드를 '개념당 청크 집합'으로 재설계 (2026-08-30 사이클 6)

배경:
    골든셋 v2의 원래 골드는 `market_trend_01`, `competitor_analysis_01` 같은
    **문서 수준 가상 ID** 8종이었다. 2026-08-30 사이클 2의 재매핑
    (`scripts/remap_golden_chunk_ids.py`)이 각 개념을 "가장 canonical한 실제
    청크 1개"로 1:1 치환하면서, 160문항이 서로 다른 청크 7개만 가리키게 됐다.
    그중 119건이 `laneige_strategy_2026`의 291자·379자 문단 2개에 몰렸고,
    38청크로 쪼개진 문서에서 하필 그 문단이 top-8에 들었는지를 묻는 지표가 되어
    검색 품질이 아니라 라벨 입도를 측정했다 (사이클 5 진단: 청크 단위 recall
    0.102 vs 출처 문서 단위 0.555).

재설계 원칙:
    개념의 근거는 하나의 문단이 아니라 **그 개념이 서술된 절(section)** 에 있다.
    청크 ID는 `{doc_id}_{section}_{part}` 구조이므로, 각 개념을 "그 개념이 실린
    절의 전체 청크 집합"으로 매핑한다. 아래 CONCEPT_SECTIONS의 절 번호는 코퍼스
    청크 본문을 직접 확인해 지정했으며 근거를 주석으로 남긴다.

    점수가 오르도록 고른 것이 아니라, 골드가 원래 의도(문서 수준 개념)를 청크
    단위로 정확히 표현하도록 고른 것이다. 개념 집합이 커지므로 **평면 청크 recall은
    오히려 낮아진다** — 대신 개념 단위 판정(`context_recall_at_k_concept`)이
    "이 개념의 근거를 찾았는가"를 정확히 묻는다.

산출:
    gold.doc_chunk_groups — 개념당 청크 ID 리스트의 리스트 (신규)
    gold.doc_chunk_ids    — 위 집합의 평면 합집합 (기존 필드, 하위 호환)

사용법:
    python3 scripts/remap_golden_chunk_groups.py --dry-run
    python3 scripts/remap_golden_chunk_groups.py
"""

import argparse
import json
import subprocess
from pathlib import Path

# 개념(원래 가상 ID) → 청크 위치 지정자 리스트.
#   (doc_id, section)        → 그 절의 모든 청크
#   (doc_id, section, part)  → 그 청크 하나 (표처럼 절 안에서 갈라지는 경우)
# 절·표 지정 근거는 각 주석 참조 (코퍼스 청크 본문을 직접 확인해 지정).
CONCEPT_SECTIONS: dict[str, list[tuple[str, ...]]] = {
    # SoS 정의·산출식은 지표 정의서 A절(Level 1: Market & Brand)에,
    # 해석 가이드는 해석 가이드 A절에 실려 있다.
    "metric_guide_sos_01": [("strategic_indicators", "1"), ("metric_interpretation", "2")],
    # HHI 정의·산출식은 SoS와 같은 A절에 이어서 서술된다.
    "metric_guide_hhi_01": [("strategic_indicators", "1")],
    # HHI 해석 가이드는 해석 가이드 A절(집중/분산 시장 판단)에 있다.
    "metric_guide_hhi_02": [("metric_interpretation", "2")],
    # CPI는 정의서 B절(Level 2: Category & Price), 해석은 해석 가이드 B절.
    "metric_guide_cpi_01": [("strategic_indicators", "2"), ("metric_interpretation", "3")],
    # Rank Volatility·Rank Shock·Streak Days는 정의서 C절(Level 3: Product & Risk),
    # 해석은 해석 가이드 C절.
    "metric_guide_rank_01": [("strategic_indicators", "3"), ("metric_interpretation", "4")],
    # 시장 거시 환경/트렌드: 전략 문서 1절(서론: 거시적 환경 변화),
    # 2절(핵심 트렌드 — 모닝 쉐드), 3절(성분 트렌드 전쟁), 트렌드 요인 표.
    "market_trend_01": [
        ("laneige_strategy_2026", "1"),
        ("laneige_strategy_2026", "2"),
        ("laneige_strategy_2026", "3"),
        # table_0 = 트렌드 요인 × 소비자 행동 변화 표
        ("laneige_strategy_2026", "table", "0"),
    ],
    # 경쟁 브랜드 분석: 3절(Medicube·VT·Genabelle의 PDRN 공세),
    # 5절(Rhode Glazing Milk 등 토너 경쟁).
    "competitor_analysis_01": [
        ("laneige_strategy_2026", "3"),
        ("laneige_strategy_2026", "5"),
        # table_1 = LANEIGE Cream Skin vs Rhode Glazing Milk 비교표
        ("laneige_strategy_2026", "table", "1"),
    ],
    # 랭킹 변동 대응: 진단 문서 3절(필수 확인 체크리스트), 8절(실행 체크리스트).
    "playbook_ranking_01": [
        ("amazon_ranking_diagnosis", "3"),
        ("amazon_ranking_diagnosis", "8"),
    ],
}

TARGET_FILES = [
    "eval/data/golden/laneige_golden_v2.jsonl",
    "eval/data/golden/subset_nokg.jsonl",
]

# 재매핑 이전(가상 ID) 골든셋을 담고 있는 커밋 — 개념 라벨 복원용
PRE_REMAP_COMMIT = "15773bf^"


def load_corpus_chunk_ids(root: Path) -> list[str]:
    """ChromaDB `amore_docs` 컬렉션의 청크 ID 전체."""
    import chromadb

    client = chromadb.PersistentClient(path=str(root / "data" / "chroma"))
    return list(client.get_collection("amore_docs").get(include=[])["ids"])


def locators_of(chunk_id: str) -> tuple[tuple[str, ...], ...]:
    """청크 ID가 매칭될 수 있는 지정자들 — (doc, section)과 (doc, section, part)."""
    parts = chunk_id.rsplit("_", 2)
    if len(parts) != 3:
        return ((chunk_id,),)
    doc, section, part = parts
    return ((doc, section), (doc, section, part))


def build_concept_chunks(chunk_ids: list[str]) -> dict[str, list[str]]:
    """개념 → 해당 절에 속한 실제 청크 ID 집합."""
    by_locator: dict[tuple[str, ...], list[str]] = {}
    for cid in chunk_ids:
        for locator in locators_of(cid):
            by_locator.setdefault(locator, []).append(cid)

    concept_chunks: dict[str, list[str]] = {}
    for concept, sections in CONCEPT_SECTIONS.items():
        chunks: list[str] = []
        for key in sections:
            chunks.extend(sorted(by_locator.get(key, [])))
        if not chunks:
            raise SystemExit(f"개념 '{concept}'에 해당하는 코퍼스 청크가 없습니다: {sections}")
        concept_chunks[concept] = sorted(set(chunks))
    return concept_chunks


def load_pre_remap_concepts(root: Path, rel_path: str) -> dict[str, list[str]]:
    """재매핑 이전 커밋에서 문항별 개념(가상 ID) 목록을 복원."""
    raw = subprocess.run(
        ["git", "show", f"{PRE_REMAP_COMMIT}:{rel_path}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    concepts: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        concepts[item["id"]] = item.get("gold", {}).get("doc_chunk_ids") or []
    return concepts


def remap_file(
    root: Path, rel_path: str, concept_chunks: dict[str, list[str]], dry_run: bool
) -> tuple[int, int]:
    path = root / rel_path
    pre_remap = load_pre_remap_concepts(root, rel_path)

    items_changed = 0
    groups_written = 0
    out_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        concepts = pre_remap.get(item["id"], [])
        groups = [concept_chunks[c] for c in concepts if c in concept_chunks]
        if groups:
            flat: list[str] = []
            for group in groups:
                for cid in group:
                    if cid not in flat:
                        flat.append(cid)
            item["gold"]["doc_chunk_groups"] = groups
            item["gold"]["doc_chunk_ids"] = flat
            items_changed += 1
            groups_written += len(groups)
        out_lines.append(json.dumps(item, ensure_ascii=False))

    if not dry_run:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return items_changed, groups_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    concept_chunks = build_concept_chunks(load_corpus_chunk_ids(root))

    print("개념별 청크 집합 크기:")
    for concept, chunks in concept_chunks.items():
        print(f"  {concept:24s} {len(chunks):3d}개  {chunks[0]} …")

    for rel in TARGET_FILES:
        changed, groups = remap_file(root, rel, concept_chunks, args.dry_run)
        tag = "(dry-run)" if args.dry_run else ""
        print(f"{rel}: {changed} items, {groups} concept groups {tag}")


if __name__ == "__main__":
    main()
