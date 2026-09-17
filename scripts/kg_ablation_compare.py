#!/usr/bin/env python3
"""KG 효과 실험(ablation) 집계 — 구성별 반복 실행을 비교한다 (읽기 전용, 무비용).

배경: docs/experiments/kg_ablation_2026-09.md. 유효 ablation이 requires_kg=false 30문항·1회뿐이라
KG·규칙 추론의 효과를 말할 수 없었다(docs/portfolio/amore_architecture_evidence.md §5.5).

사용법:
    # 172문항 실행 리포트에서 부분집합 문항만 다시 집계 (같은 채점 코드)
    python3 scripts/kg_ablation_compare.py subset --report eval_output/v4-full-run1/report.json \
        --dataset eval/data/golden/subset_requires_kg_v2.jsonl --out eval_output/kg-full-run1

    # 구성 비교. 첫 구성이 기준이다.
    python3 scripts/kg_ablation_compare.py compare \
        --config full=eval_output/kg-full-run{1,2,3}/report.json \
        --config no-kg=eval_output/kg-nokg-run{1,2,3}/report.json

판정 규칙 ("차이 있음"은 둘 다 만족할 때만):
    1) 평균 차이의 절댓값이 노이즈 기준 이상 — 기준은 사이클 9 §2의 172문항 3회 측정값
       (종합·검색 계열 0.01, 근거성 0.03, 수치 정확도 0.05, 통과 수 5건)과
       이번 두 구성의 실행 간 폭(최대-최소) 중 큰 값
    2) 두 구성의 실행 값 범위가 겹치지 않는다
    실행이 1회뿐인 구성은 2)를 기준 구성 범위 밖인지로 본다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# (report.aggregates 키, 표시명, 노이즈 기준)
TRACKED = [
    ("overall", "종합 점수", 0.01),
    ("pass_count", "통과 수", 5.0),
    ("l1_entity_link_f1", "L1 엔티티 F1", 0.01),
    ("l2_context_recall_concept", "L2 개념 Recall", 0.01),
    ("l3_hits_at_k", "L3 Hits@8", 0.01),
    ("l3_kg_edge_recall", "L3 엣지 Recall", 0.01),
    ("l5_groundedness", "L5 근거성", 0.03),
    ("l5_relevance", "L5 관련성", 0.03),
    ("l5_answer_f1", "L5 토큰 F1", 0.01),
    ("l5_numeric_accuracy", "L5 수치 정확도", 0.05),
]


def _values(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = data["aggregates"]
    values = {"overall": agg["avg_overall_score"], "pass_count": float(agg["passed"])}
    values.update(agg["by_layer"])
    items = data["items"]
    return {
        "values": values,
        "total": agg["total"],
        "errored": agg.get("errored", 0),
        "cost": agg.get("total_cost_usd", 0.0),
        "latency": agg.get("avg_latency_ms", 0.0),
        "react_items": sum(
            1
            for i in items
            if ((i.get("trace") or {}).get("l5_answer") or {}).get("query_type") == "react"
        ),
        "commit": (data.get("config") or {}).get("git_commit"),
        "target": (data.get("config") or {}).get("target"),
    }


def cmd_subset(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.report import ReportGenerator
    from eval.schemas import EvalReport

    ids = {
        json.loads(line)["id"]
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    report = EvalReport.model_validate(json.loads(args.report.read_text(encoding="utf-8")))
    items = [i for i in report.items if i.item_id in ids]
    missing = ids - {i.item_id for i in items}
    if missing:
        print(f"리포트에 없는 문항 {len(missing)}건: {sorted(missing)[:10]}")
        return 1
    ReportGenerator(config=report.config).generate_report(items, args.out)
    print(f"{len(items)}문항 재집계 → {args.out}/report.json")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    configs: list[tuple[str, list[dict]]] = []
    for spec in args.config:
        name, _, paths = spec.partition("=")
        runs = [_values(Path(p)) for p in paths.split(",") if p]
        configs.append((name, runs))

    print("\n| 구성 | 실행 | 채점/제외 | 대상 | 커밋 | 비용 | 평균 지연 | ReAct 발동 문항 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, runs in configs:
        print(
            f"| {name} | {len(runs)} | "
            + ", ".join(f"{r['total']}/{r['errored']}" for r in runs)
            + f" | {runs[0]['target']} | {', '.join(sorted({str(r['commit']) for r in runs}))} | "
            + f"${sum(r['cost'] for r in runs):.3f} | "
            + f"{statistics.fmean(r['latency'] for r in runs) / 1000:.1f}s | "
            + ", ".join(str(r["react_items"]) for r in runs)
            + " |"
        )

    base_name, base_runs = configs[0]
    header = " | ".join(f"{n} 평균 [최소, 최대]" for n, _ in configs)
    diffs = " | ".join(f"{n}−{base_name}" for n, _ in configs[1:])
    print(f"\n| 지표 | {header} | {diffs} |")
    print("|---" * (1 + len(configs) + len(configs) - 1) + "|")
    for key, label, noise in TRACKED:
        cells = []
        stats = {}
        for name, runs in configs:
            vals = [r["values"].get(key) for r in runs]
            if any(v is None for v in vals):
                cells.append("—")
                continue
            stats[name] = (statistics.fmean(vals), min(vals), max(vals))
            mean, lo, hi = stats[name]
            fmt = "{:.0f}" if key == "pass_count" else "{:.3f}"
            if len(vals) == 1:
                cells.append(fmt.format(mean))
            else:
                cells.append(f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]")
        verdicts = []
        for name, _ in configs[1:]:
            if base_name not in stats or name not in stats:
                verdicts.append("—")
                continue
            b_mean, b_lo, b_hi = stats[base_name]
            c_mean, c_lo, c_hi = stats[name]
            delta = c_mean - b_mean
            threshold = max(noise, b_hi - b_lo, c_hi - c_lo)
            overlap = not (c_hi < b_lo or c_lo > b_hi)
            fmt = "{:+.1f}" if key == "pass_count" else "{:+.3f}"
            if abs(delta) >= threshold and not overlap:
                verdicts.append(f"{fmt.format(delta)} **차이 있음**")
            else:
                verdicts.append(f"{fmt.format(delta)} 차이 없음")
        print(f"| {label} | {' | '.join(cells)} | {' | '.join(verdicts)} |")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_subset = sub.add_parser("subset", help="리포트에서 부분집합 문항만 재집계")
    p_subset.add_argument("--report", type=Path, required=True)
    p_subset.add_argument("--dataset", type=Path, required=True)
    p_subset.add_argument("--out", type=Path, required=True)

    p_compare = sub.add_parser("compare", help="구성별 반복 실행 비교")
    p_compare.add_argument(
        "--config", action="append", required=True, help="name=report1,report2,... (첫 구성이 기준)"
    )

    args = parser.parse_args()
    return cmd_subset(args) if args.command == "subset" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
