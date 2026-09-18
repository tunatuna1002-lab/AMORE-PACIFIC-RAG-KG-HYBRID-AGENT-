"""[2026-09 사후] O0-A: 저장된 평가 리포트를 새 L3·L4 지표로 재채점한다 (LLM 호출 없음).

리포트의 문항 trace(kg_edges_found·ontology_facts·l4_ontology.inferences·rule_evaluation)만
읽어 계산한다. 리포트에는 골드가 저장돼 있지 않아 골든 파일에서 문항 ID로 가져오고,
**기존 kg_edge_recall을 다시 계산해 저장값과 같은지 확인**한다(골드가 바뀌었으면 드러난다).

사용:
    .venv/bin/python scripts/rescore_l3_l4.py \\
        --base <eval_output/evidence-2026-09> --out <eval_output/ontology-2026-09/rescore> \\
        s6a-run1 s6a-run2 s6a-run3 s5-run1 s3-run1 ...

출력: 리포트별 ``<name>.json``(문항별 새 지표 + 집계)과 ``summary.json``, 표준출력에 마크다운 표.
구성(config) 묶음은 이름에서 ``-run<N>``을 뗀 것이다. ``--subset NAME=ids.jsonl``을 주면
그 구성의 문항을 해당 파일의 ID로 제한한 묶음(NAME)도 만든다(예: s3의 rule 42문항).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.metrics.l3_kg import L3KGMetrics, aggregate_l3_extended  # noqa: E402
from eval.metrics.l4_ontology import L4OntologyMetrics, aggregate_l4_extended  # noqa: E402
from eval.schemas import (  # noqa: E402
    GoldEvidence,
    KGQueryTrace,
    L3Metrics,
    L4Metrics,
    OntologyReasoningTrace,
)

DEFAULT_GOLD = REPO / "eval" / "data" / "golden" / "typed" / "combined_v1.jsonl"
KEY_PREDICATES = (
    "ownedByGroup",
    "ownedBy",
    "hasSegment",
    "originatesFrom",
    "siblingBrand",
    "acquiredIn",
    "belongsToCategory",
    "hasProduct",
    "rankedIn",
    "competesWith",
    "hasSoS",
    "hasHHI",
)


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["id"]: row for row in rows}


def rescore_report(
    report: dict[str, Any],
    gold_by_id: dict[str, dict[str, Any]],
    l3_calc: L3KGMetrics,
    l4_calc: L4OntologyMetrics,
) -> dict[str, Any]:
    """리포트 한 개를 재채점한다. 인프라 실패(trace.error) 문항은 리포트 집계처럼 뺀다."""
    items: list[dict[str, Any]] = []
    l3_list: list[L3Metrics] = []
    l4_list: list[L4Metrics] = []
    errored: list[str] = []
    gold_missing: list[str] = []
    recall_mismatch: list[str] = []
    missing_fields: dict[str, int] = {}

    for item in report.get("items") or []:
        item_id = item["item_id"]
        trace = item.get("trace") or {}
        if trace.get("error"):
            errored.append(item_id)
            continue
        gold_row = gold_by_id.get(item_id)
        if gold_row is None:
            gold_missing.append(item_id)
            continue
        for field in ("l3_kg_query", "l4_ontology", "rule_evaluation"):
            if trace.get(field) is None:
                missing_fields[field] = missing_fields.get(field, 0) + 1
        gold = GoldEvidence.model_validate(gold_row.get("gold") or {})
        kg_trace = KGQueryTrace.model_validate(trace.get("l3_kg_query") or {})
        onto_trace = OntologyReasoningTrace.model_validate(trace.get("l4_ontology") or {})

        l3 = l3_calc.compute(kg_trace, gold)
        l4 = l4_calc.compute(
            onto_trace, kg_trace, gold, rule_evaluation=trace.get("rule_evaluation")
        )
        stored_recall = (item.get("l3") or {}).get("kg_edge_recall")
        if stored_recall is not None and abs(stored_recall - l3.kg_edge_recall) > 1e-9:
            recall_mismatch.append(item_id)
        l3_list.append(l3)
        l4_list.append(l4)
        items.append({"item_id": item_id, "l3": l3.model_dump(), "l4": l4.model_dump()})

    return {
        "git_commit": (report.get("config") or {}).get("git_commit"),
        "items_in_report": len(report.get("items") or []),
        "scored": len(items),
        "errored": errored,
        "gold_missing": gold_missing,
        "legacy_recall_mismatch": recall_mismatch,
        "missing_trace_fields": missing_fields,
        "l3": aggregate_l3_extended(l3_list),
        "l4": aggregate_l4_extended(l4_list),
        "items": items,
    }


def aggregate_subset(result: dict[str, Any], ids: set[str]) -> dict[str, Any]:
    """재채점 결과를 문항 ID 부분집합으로 다시 집계한다."""
    kept = [it for it in result["items"] if it["item_id"] in ids]
    l3 = [L3Metrics.model_validate(it["l3"]) for it in kept]
    l4 = [L4Metrics.model_validate(it["l4"]) for it in kept]
    return {"scored": len(kept), "l3": aggregate_l3_extended(l3), "l4": aggregate_l4_extended(l4)}


def config_name(report_name: str) -> str:
    return re.sub(r"-run\d+$", "", report_name)


def _stats(values: list[float | None]) -> dict[str, Any]:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "min": None, "max": None, "runs": len(values), "n": 0}
    return {
        "mean": statistics.fmean(present),
        "min": min(present),
        "max": max(present),
        "runs": len(values),
        "n": len(present),
    }


def summarize(per_run: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """구성별로 run 간 mean/min/max를 낸다. per_run 값은 {'l3':..., 'l4':...} 집계."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, agg in per_run.items():
        groups.setdefault(config_name(name), []).append(agg)
    summary: dict[str, dict[str, Any]] = {}
    for group, runs in sorted(groups.items()):
        row: dict[str, Any] = {"runs": len(runs), "items": [r["scored"] for r in runs]}
        for key in (
            "kg_edge_recall_all",
            "kg_edge_recall_gold_only",
            "kg_edge_recall_micro",
            "gold_edge_items",
        ):
            row[f"l3.{key}"] = _stats([r["l3"][key] for r in runs])
        for predicate in KEY_PREDICATES:
            row[f"l3.pred.{predicate}"] = _stats(
                [(r["l3"]["recall_by_predicate"].get(predicate) or {}).get("recall") for r in runs]
            )
            totals = {
                (r["l3"]["recall_by_predicate"].get(predicate) or {}).get("total") for r in runs
            }
            row[f"l3.pred.{predicate}.total"] = sorted(t for t in totals if t is not None)
        for key in (
            "constraint_violation_rate_legacy",
            "type_consistency_rate_legacy",
            "rule_constraint_violation_rate",
            "rule_violation_micro",
            "rule_checked_items",
            "typed_consistency_rate",
            "typed_consistency_micro",
            "type_checked_items",
        ):
            row[f"l4.{key}"] = _stats([r["l4"][key] for r in runs])
        summary[group] = row
    return summary


def _fmt(stat: dict[str, Any], digits: int = 3) -> str:
    if stat["mean"] is None:
        return "—"
    if stat["runs"] == 1 or stat["min"] == stat["max"]:
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} ({stat['min']:.{digits}f}~{stat['max']:.{digits}f})"


def markdown(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| 구성 | runs | 문항 | L3 recall(기존) | L3 recall(골드 엣지 문항) | 골드 엣지 문항 | "
        "L3 micro | L4 위반(기존) | L4 규칙 위반율 | 규칙 검사 문항 | L4 타입 일관성(기존) | "
        "L4 타입 일관성 | 타입 검사 문항 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group, row in summary.items():
        lines.append(
            f"| {group} | {row['runs']} | {'/'.join(map(str, sorted(set(row['items']))))} | "
            f"{_fmt(row['l3.kg_edge_recall_all'])} | {_fmt(row['l3.kg_edge_recall_gold_only'])} | "
            f"{_fmt(row['l3.gold_edge_items'], 0)} | {_fmt(row['l3.kg_edge_recall_micro'])} | "
            f"{_fmt(row['l4.constraint_violation_rate_legacy'])} | "
            f"{_fmt(row['l4.rule_constraint_violation_rate'])} | "
            f"{_fmt(row['l4.rule_checked_items'], 0)} | "
            f"{_fmt(row['l4.type_consistency_rate_legacy'])} | "
            f"{_fmt(row['l4.typed_consistency_rate'])} | {_fmt(row['l4.type_checked_items'], 0)} |"
        )
    lines += [
        "",
        "술어별 recall (골드 표기 기준, 별칭 정규화 없음) — mean (min~max), [골드 엣지 수]",
        "",
    ]
    header = "| 구성 | " + " | ".join(KEY_PREDICATES) + " |"
    lines += [header, "|---|" + "---|" * len(KEY_PREDICATES)]
    for group, row in summary.items():
        cells = []
        for predicate in KEY_PREDICATES:
            totals = row[f"l3.pred.{predicate}.total"]
            total_txt = "/".join(map(str, totals)) if totals else "0"
            cells.append(f"{_fmt(row[f'l3.pred.{predicate}'])} [{total_txt}]")
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "reports", nargs="+", help="리포트 디렉터리 이름(<base>/<name>/report.json)"
    )
    parser.add_argument("--base", type=Path, required=True, help="리포트 디렉터리들의 상위 경로")
    parser.add_argument("--out", type=Path, required=True, help="출력 디렉터리 (새로 쓴다)")
    parser.add_argument(
        "--gold", type=Path, default=DEFAULT_GOLD, help="골든 jsonl (문항 ID로 조회)"
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        metavar="NAME:CONFIG=IDS.jsonl",
        help="구성 CONFIG의 run들을 IDS.jsonl의 ID로 제한한 묶음 NAME을 추가 (여러 번 가능)",
    )
    args = parser.parse_args(argv)

    gold_by_id = load_jsonl_by_id(args.gold)
    l3_calc = L3KGMetrics(default_k=8)
    l4_calc = L4OntologyMetrics()
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for name in args.reports:
        report = json.loads((args.base / name / "report.json").read_text(encoding="utf-8"))
        result = rescore_report(report, gold_by_id, l3_calc, l4_calc)
        result["report"] = str(args.base / name / "report.json")
        (args.out / f"{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )
        results[name] = result
        print(
            f"[{name}] scored={result['scored']} errored={len(result['errored'])} "
            f"gold_missing={len(result['gold_missing'])} "
            f"legacy_recall_mismatch={len(result['legacy_recall_mismatch'])} "
            f"missing_trace_fields={result['missing_trace_fields']}",
            file=sys.stderr,
        )

    per_run = {name: {k: r[k] for k in ("scored", "l3", "l4")} for name, r in results.items()}
    for spec in args.subset:
        label, _, rest = spec.partition(":")
        config, _, ids_path = rest.partition("=")
        ids = set(load_jsonl_by_id(Path(ids_path)))
        for name, result in results.items():
            if config_name(name) == config:
                run_suffix = name[len(config) :]
                per_run[f"{label}{run_suffix}"] = aggregate_subset(result, ids)

    summary = summarize(per_run)
    (args.out / "summary.json").write_text(
        json.dumps(
            {"gold": str(args.gold), "per_run": per_run, "summary": summary},
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    table = markdown(summary)
    (args.out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
