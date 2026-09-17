#!/usr/bin/env python3
"""유형별 시험지 집계 — 통합 시험지 반복 실행 리포트를 유형별로 나눠 비교한다 (읽기 전용, 무비용).

배경: docs/experiments/evidence_pipeline_2026-09.md. 통합 시험지
(eval/data/golden/typed/combined_v1.jsonl)를 한 번 실행하고, 유형별 문항 id
(eval/data/golden/typed/{numeric,relation,rule,multihop}.jsonl)로 리포트를 나눠
같은 채점 코드(ReportGenerator._compute_aggregates)로 다시 집계한다.

사용법:
    python3 scripts/typed_eval_summary.py \
        --config base=eval_output/evidence-2026-09/base-run{1,2,3}/report.json \
        --config stage2=eval_output/evidence-2026-09/s2-run{1,2,3}/report.json

판정 규칙 (지시서 §2, scripts/kg_ablation_compare.py와 같음):
    평균 차이의 절댓값이 노이즈 기준과 두 구성의 실행 간 폭(최대-최소) 중 큰 값 이상이고,
    두 구성의 실행 값 범위가 겹치지 않을 때만 "차이 있음". 첫 구성이 기준이다.

규칙 정답 일치율 (rule 유형): 문항 metadata.rule_gold.rule_ids 중 하나라도 트레이스의
applied_rules에 있으면 "발화"로 보고, rule_gold.expected_conclusion.fires와 같으면 일치.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("numeric", "relation", "rule", "multihop")

# (키, 표시명, 노이즈 기준)
TRACKED = [
    ("overall", "종합 점수", 0.01),
    ("pass_count", "통과 수", 5.0),
    ("l2_context_recall_concept", "L2 개념 Recall", 0.01),
    ("l3_kg_edge_recall", "L3 엣지 Recall", 0.01),
    ("l5_groundedness", "L5 근거성", 0.03),
    ("l5_relevance", "L5 관련성", 0.03),
    ("l5_answer_f1", "L5 토큰 F1", 0.01),
    ("l5_numeric_accuracy", "L5 수치 정확도", 0.05),
    ("rule_agreement", "규칙 정답 일치율", 0.05),
    ("rule_fired_rate", "규칙 발화 문항 비율", 0.05),
]


def _load_type_ids(types_dir: Path) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = {}
    for name in TYPES:
        path = types_dir / f"{name}.jsonl"
        ids[name] = {
            json.loads(line)["id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return ids


def _load_rule_gold(types_dir: Path) -> dict[str, dict]:
    """문항 id → rule_gold. 리포트의 ItemMetadata는 이 키를 보존하지 않아 시험지에서 읽는다."""
    gold: dict[str, dict] = {}
    for line in (types_dir / "rule.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rule_gold = (record.get("metadata") or {}).get("rule_gold")
            if rule_gold:
                gold[record["id"]] = rule_gold
    return gold


RULE_GOLD: dict[str, dict] = {}


def _rule_agreement(items: list) -> float | None:
    """문항별 규칙 정답 일치율.

    리포트가 트랙 3-C 필드(ItemResult.rule_agreement)를 채운 경우 그것을 그대로
    쓴다. 없으면(구형 report.json) 예전처럼 시험지의 rule_gold + 트레이스의
    applied_rules로 다시 계산한다 — 하위 호환 폴백.
    """
    judged = 0
    agree = 0
    for item in items:
        if item.rule_agreement is not None:
            judged += 1
            agree += int(bool(item.rule_agreement))
            continue

        rule_gold = (getattr(item.metadata, "rule_gold", None)) or RULE_GOLD.get(item.item_id) or {}
        expected = (rule_gold.get("expected_conclusion") or {}).get("fires")
        rule_ids = set(rule_gold.get("rule_ids") or [])
        if expected is None or not rule_ids or item.trace is None:
            continue
        applied = set(item.trace.l4_ontology.applied_rules or [])
        judged += 1
        agree += int(bool(applied & rule_ids) == bool(expected))
    return agree / judged if judged else None


def _non_fire_top(items: list, top_n: int = 5) -> list[tuple[str, int]]:
    """채점된 문항들의 rule_evaluation.non_fire_top을 라벨별로 합산한 상위 top_n."""
    counts: Counter = Counter()
    for item in items:
        rule_evaluation = item.trace.rule_evaluation if item.trace is not None else None
        if not isinstance(rule_evaluation, dict):
            continue
        for entry in rule_evaluation.get("non_fire_top") or []:
            if isinstance(entry, list | tuple) and len(entry) == 2:
                label, count = entry
                counts[str(label)] += int(count)
    return counts.most_common(top_n)


def _summarize(items: list, generator) -> dict:
    agg = generator._compute_aggregates(items)
    values = {"overall": agg.avg_overall_score, "pass_count": float(agg.passed)}
    values.update(agg.by_layer)
    scored = [i for i in items if i.trace is None or not i.trace.error]
    values["rule_agreement"] = _rule_agreement(scored)
    values["rule_fired_rate"] = agg.rule_fired_items / len(scored) if scored else None
    return {
        "values": values,
        "total": agg.total,
        "errored": agg.errored,
        "routes": dict(agg.route_counts),
        "confidence": dict(agg.confidence_level_counts),
        "rule_fired_items": agg.rule_fired_items,
        "rule_inferences": agg.rule_inference_total,
        "non_fire_top5": _non_fire_top(scored, 5),
        "cost": sum((i.trace.cost.total_cost_usd if i.trace else 0.0) for i in items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", required=True, help="name=r1.json,r2.json")
    parser.add_argument("--types-dir", type=Path, default=ROOT / "eval/data/golden/typed")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from eval.report import ReportGenerator
    from eval.schemas import EvalReport

    type_ids = _load_type_ids(args.types_dir)
    RULE_GOLD.update(_load_rule_gold(args.types_dir))
    configs: list[tuple[str, list[dict]]] = []
    meta_rows = []
    for spec in args.config:
        name, _, paths = spec.partition("=")
        runs: list[dict] = []
        for path in [p for p in paths.split(",") if p]:
            report = EvalReport.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
            generator = ReportGenerator(config=report.config)
            by_type = {"all": _summarize(report.items, generator)}
            typed_union: set[str] = set()
            for type_name, ids in type_ids.items():
                subset = [i for i in report.items if i.item_id in ids]
                missing = ids - {i.item_id for i in subset}
                if missing:
                    print(f"{path}: {type_name} 문항 {len(missing)}건 없음 {sorted(missing)[:5]}")
                    return 1
                by_type[type_name] = _summarize(subset, generator)
                typed_union |= ids
            others = [i for i in report.items if i.item_id not in typed_union]
            by_type["other"] = _summarize(others, generator)
            runs.append(by_type)
            meta_rows.append(
                (
                    name,
                    path,
                    report.config.git_commit if report.config else None,
                    report.aggregates.total,
                    report.aggregates.errored,
                    report.aggregates.total_cost_usd,
                )
            )
        configs.append((name, runs))

    print("\n| 구성 | 리포트 | 커밋 | 채점 | 제외 | 비용(리포트) |")
    print("|---|---|---|---|---|---|")
    for row in meta_rows:
        print(f"| {row[0]} | `{row[1]}` | {row[2]} | {row[3]} | {row[4]} | ${row[5]:.3f} |")

    base_name, base_runs = configs[0]
    for type_name in ("numeric", "relation", "rule", "multihop", "other", "all"):
        n = base_runs[0][type_name]["total"]
        print(f"\n### {type_name} ({n}문항)\n")
        header = " | ".join(f"{name} 평균 [최소, 최대]" for name, _ in configs)
        diffs = " | ".join(f"{name}−{base_name}" for name, _ in configs[1:])
        print(f"| 지표 | {header} |" + (f" {diffs} |" if diffs else ""))
        print("|---" * (1 + len(configs) + max(0, len(configs) - 1)) + "|")
        for key, label, noise in TRACKED:
            cells, stats = [], {}
            for name, runs in configs:
                vals = [r[type_name]["values"].get(key) for r in runs]
                if any(v is None for v in vals):
                    cells.append("—")
                    continue
                stats[name] = (statistics.fmean(vals), min(vals), max(vals))
                mean, lo, hi = stats[name]
                fmt = "{:.1f}" if key == "pass_count" else "{:.3f}"
                cells.append(
                    fmt.format(mean)
                    if len(vals) == 1
                    else f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"
                )
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
                verdict = (
                    "**차이 있음**" if abs(delta) >= threshold and not overlap else "차이 없음"
                )
                verdicts.append(f"{fmt.format(delta)} {verdict}")
            row = f"| {label} | {' | '.join(cells)} |"
            if verdicts:
                row += f" {' | '.join(verdicts)} |"
            print(row)

        print(
            "\n| 구성 | 경로 분포(실행별) | 신뢰도 분포(실행별) | 규칙 발화 문항 / 추론 수 "
            "| 미발화 사유 상위 5 |"
        )
        print("|---|---|---|---|---|")
        for name, runs in configs:
            routes = "; ".join(
                ", ".join(f"{k} {v}" for k, v in sorted(r[type_name]["routes"].items()))
                for r in runs
            )
            conf = "; ".join(
                ", ".join(f"{k} {v}" for k, v in sorted(r[type_name]["confidence"].items()))
                for r in runs
            )
            fired = "; ".join(
                f"{r[type_name]['rule_fired_items']}/{r[type_name]['rule_inferences']}"
                for r in runs
            )
            non_fire = "; ".join(
                ", ".join(f"{label} {count}" for label, count in r[type_name]["non_fire_top5"])
                or "—"
                for r in runs
            )
            print(f"| {name} | {routes} | {conf} | {fired} | {non_fire} |")

    total_routes: Counter = Counter()
    for _, runs in configs:
        for r in runs:
            total_routes.update(r["all"]["routes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
