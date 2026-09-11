#!/usr/bin/env python3
"""저장된 리포트의 수치 정확도와 게이트를 현재 채점기로 다시 계산한다 (무비용).

배경 (사이클 10, 2026-09-12)
---------------------------
사이클 9에서 처음 배선한 numeric_accuracy가 거짓양성을 대량으로 냈다. 답변 본문이
아니라 시스템이 덧붙인 출처 목록 번호("2. 🧠"), 관계 개수("5개 관계"), 관련도
("0.52")의 숫자에 기대값이 걸렸고, "데이터에 명시되어 있지 않습니다"라고 답한
문항이 만점을 받았다. 채점기를 고친 뒤 LLM을 다시 부르지 않고, 리포트에 저장된
답변(trace.l5_answer.final_answer)으로 재채점한다.

바뀌는 것: l5.numeric_accuracy, passed, fail_reason_tags, 집계.
바뀌지 않는 것: 답변, L1~L4 지표, judge 점수, 종합 점수(수치 정확도는 공식에 없다).

사용법:
    python3 scripts/rescore_numeric_accuracy.py eval_output/v9-run{1,2,3}/report.json
    # → eval_output/v9-runN-rescored/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics.aggregator import MetricAggregator  # noqa: E402
from eval.metrics.l5_answer import numeric_accuracy  # noqa: E402
from eval.report import ReportGenerator  # noqa: E402
from eval.schemas import EvalReport, ItemMetadata  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "eval" / "data" / "golden" / "laneige_golden_v2.jsonl"


def load_gold(dataset: Path) -> dict[str, dict]:
    rows = {}
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def rescore(report_path: Path, gold: dict[str, dict], out_dir: Path) -> dict:
    report = EvalReport.model_validate(json.loads(report_path.read_text(encoding="utf-8")))
    aggregator = MetricAggregator()
    before = report.aggregates

    for item in report.items:
        if item.trace is None or item.trace.error:
            continue  # 인프라 실패 문항은 채점 대상이 아니다
        row = gold[item.item_id]
        item.metadata = ItemMetadata(**row["metadata"])
        item.l5.numeric_accuracy = numeric_accuracy(
            item.trace.l5_answer.final_answer, row["gold"].get("expected_values", {})
        )
        item.passed, item.fail_reason_tags = aggregator.check_gating(
            item.l1, item.l2, item.l3, item.l4, item.l5, item.metadata
        )

    after = ReportGenerator(config=report.config).generate_report(report.items, out_dir).aggregates
    return {
        "run": report_path.parent.name,
        "numeric_before": before.by_layer.get("l5_numeric_accuracy", 0.0),
        "numeric_after": after.by_layer.get("l5_numeric_accuracy", 0.0),
        "passed_before": before.passed,
        "passed_after": after.passed,
        "mismatch_before": before.top_fail_reasons.get("L5_numeric_mismatch", 0),
        "mismatch_after": after.top_fail_reasons.get("L5_numeric_mismatch", 0),
        "overall_before": before.avg_overall_score,
        "overall_after": after.avg_overall_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--suffix", default="-rescored", help="출력 디렉터리 접미사")
    args = parser.parse_args()

    gold = load_gold(args.dataset)
    rows = []
    for path in args.reports:
        out_dir = path.parent.with_name(path.parent.name + args.suffix)
        rows.append(rescore(path, gold, out_dir))

    print("\n| 실행 | 수치 정확도 | 통과 | L5_numeric_mismatch | 종합 점수 |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['run']} | {r['numeric_before']:.3f} → {r['numeric_after']:.3f} "
            f"| {r['passed_before']} → {r['passed_after']} "
            f"| {r['mismatch_before']} → {r['mismatch_after']} "
            f"| {r['overall_before']:.3f} → {r['overall_after']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
