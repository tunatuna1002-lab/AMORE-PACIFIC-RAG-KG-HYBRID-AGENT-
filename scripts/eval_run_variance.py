#!/usr/bin/env python3
"""동일 조건 반복 실행의 노이즈 폭을 잰다 (읽기 전용, 무비용).

배경
----
모든 baseline이 1회 실행이라 작은 델타를 해석할 수 없었다. 2026-08-31 Phase 4
검증에서 동일 코드 재실행 시 172문항 중 답변이 같은 문항은 7개, 통과는 12→9로
흔들렸다(docs/experiments/refactor_phase4_2026-08-31.md). 개선·회귀를 판정하려면
먼저 노이즈 폭을 알아야 한다.

이 스크립트는 같은 조건으로 돌린 report.json 여러 개를 받아
지표별 평균·표준편차·최대-최소와 통과 문항 집합의 교집합/합집합을 낸다.

사용법:
    python3 scripts/eval_run_variance.py eval_output/v9-run{1,2,3}/report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# 사이클 문서에서 추적하는 지표 (report.aggregates.by_layer 키)
TRACKED = [
    ("overall", None),
    ("pass_count", None),
    ("l1_entity_link_f1", "L1 Entity F1"),
    ("l1_concept_map_f1", "L1 Concept F1"),
    ("l2_context_recall_concept", "L2 개념 Recall"),
    ("l2_mrr", "L2 MRR"),
    ("l3_hits_at_k", "L3 Hits@8"),
    ("l3_kg_edge_recall", "L3 Edge Recall"),
    ("l5_groundedness", "L5 Groundedness"),
    ("l5_relevance", "L5 Relevance"),
    ("l5_answer_f1", "L5 Token F1"),
    ("l5_numeric_accuracy", "L5 수치 정확도"),
]


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = data["aggregates"]
    values = {"overall": agg["avg_overall_score"], "pass_count": float(agg["passed"])}
    values.update(dict(agg["by_layer"].items()))
    return {
        "path": path,
        "values": values,
        "passed": {i["item_id"] for i in data["items"] if i["passed"]},
        "answers": {
            i["item_id"]: (i.get("trace") or {}).get("l5_answer", {}).get("final_answer", "")
            for i in data["items"]
        },
        "errored": agg.get("errored", 0),
        "total": agg["total"],
        "cost": agg.get("total_cost_usd", 0.0),
        "tokens": agg.get("total_tokens", 0),
        "latency": agg.get("avg_latency_ms", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="같은 조건으로 돌린 report.json")
    args = parser.parse_args()

    runs = [load(p) for p in args.reports]
    if len(runs) < 2:
        print("2회 이상의 실행이 필요하다")
        return 1

    print(f"\n실행 {len(runs)}회 — {', '.join(r['path'].parent.name for r in runs)}")
    print("채점 문항: " + ", ".join(f"{r['total']}(제외 {r['errored']})" for r in runs))
    print("비용: " + ", ".join(f"${r['cost']:.4f}/{r['tokens']:,}토큰" for r in runs))

    print(
        "\n| 지표 | "
        + " | ".join(f"실행{i}" for i in range(1, len(runs) + 1))
        + " | 평균 | 표준편차 | 최대-최소 |"
    )
    print("|---" * (len(runs) + 4) + "|")
    for key, label in TRACKED:
        vals = [r["values"].get(key) for r in runs]
        if any(v is None for v in vals):
            continue
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        spread = max(vals) - min(vals)
        cells = " | ".join(f"{v:.3f}" for v in vals)
        print(f"| {label or key} | {cells} | {mean:.3f} | {sd:.3f} | {spread:.3f} |")

    inter = set.intersection(*(r["passed"] for r in runs))
    union = set.union(*(r["passed"] for r in runs))
    print(f"\n통과 문항: 교집합 {len(inter)}건 / 합집합 {len(union)}건")
    unstable = sorted(union - inter)
    if unstable:
        print(f"실행마다 통과 여부가 흔들린 문항 {len(unstable)}건: {', '.join(unstable)}")

    identical = sum(
        1
        for item_id in runs[0]["answers"]
        if len({r["answers"].get(item_id, "") for r in runs}) == 1
    )
    print(f"모든 실행에서 답변이 완전히 같은 문항: {identical} / {len(runs[0]['answers'])}")

    # 중앙값 실행 (overall 기준) — baseline으로 저장할 후보
    ordered = sorted(runs, key=lambda r: r["values"]["overall"])
    median_run = ordered[len(ordered) // 2]
    print(f"\n중앙값 실행(overall 기준): {median_run['path'].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
