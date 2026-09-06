#!/usr/bin/env python3
"""기존 baseline 리포트를 현재 종합 점수 공식으로 재집계한다 (읽기 전용).

배경 (2026-09-06)
-----------------
종합 점수 공식이 게이트와 다른 지표를 쓰고 있었다:

    공식: l2.context_recall_at_k (청크 단위) + l3.kg_edge_f1
    게이트: l2.context_recall_at_k_concept (개념 단위) + l3.kg_edge_recall

두 공식 지표 모두 판별력이 없다고 이미 판정된 것들이라(청크 단위는 라벨 입도를
재고, 엣지 set-F1은 규모 비대칭으로 상한 ~0.18) 게이트 지표로 교체했다.
그 결과 **v8.1 이전 baseline의 종합 점수와 이후 값은 정의가 다르다.**

이 스크립트는 저장된 baseline의 문항별 L1~L5 지표를 새 공식에 다시 통과시켜
연속성 있는 비교값을 만든다. 게이트·임계값은 건드리지 않으므로 pass/fail은
변하지 않는다.

주의: v8.1까지의 report.json은 `metadata`를 직렬화에서 잃어 requires_kg가 전부
기본값 True로 저장돼 있다. 공식이 requires_kg에 따라 갈리므로, 원 데이터셋에서
문항별 requires_kg를 복원해 쓴다(--dataset). 복원하지 않으면 requires_kg=False
문항 42개가 원 실행과 다른 가지로 계산된다.

사용법:
    python3 scripts/reaggregate_baseline_scores.py
    python3 scripts/reaggregate_baseline_scores.py --baseline v8.1-2026-08-30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics.aggregator import MetricAggregator  # noqa: E402
from eval.schemas import (  # noqa: E402
    ItemMetadata,
    L1Metrics,
    L2Metrics,
    L3Metrics,
    L4Metrics,
    L5Metrics,
)

DEFAULT_BASELINE_DIR = REPO_ROOT / "eval" / "baselines"
DEFAULT_DATASET = REPO_ROOT / "eval" / "data" / "golden" / "laneige_golden_v2.jsonl"


def load_requires_kg(dataset_path: Path) -> dict[str, bool]:
    """데이터셋에서 문항별 requires_kg를 복원."""
    mapping: dict[str, bool] = {}
    if not dataset_path.exists():
        return mapping
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mapping[row["id"]] = row.get("metadata", {}).get("requires_kg", True)
    return mapping


def reaggregate(report_path: Path, requires_kg: dict[str, bool]) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    aggregator = MetricAggregator()

    stored = data["aggregates"]["avg_overall_score"]
    items = data["items"]
    recomputed = []
    unknown_meta = 0

    # 새 공식이 쓰는 두 지표는 도중에 도입됐다 (엣지 recall: 사이클 4,
    # 개념 recall: 사이클 6). 그 이전 리포트에는 필드 자체가 없어 0으로 채워지므로
    # 재집계값이 의미가 없다. 없는 값을 채워 넣지 말고 비교 불가로 표시한다.
    first = items[0] if items else {"l2": {}, "l3": {}}
    comparable = "context_recall_at_k_concept" in first.get(
        "l2", {}
    ) and "kg_edge_recall" in first.get("l3", {})

    for item in items:
        item_id = item["item_id"]
        if item_id in requires_kg:
            meta = ItemMetadata(requires_kg=requires_kg[item_id])
        else:
            unknown_meta += 1
            meta = ItemMetadata(**item.get("metadata", {}))
        recomputed.append(
            aggregator.compute_overall_score(
                L1Metrics(**item["l1"]),
                L2Metrics(**item["l2"]),
                L3Metrics(**item["l3"]),
                L4Metrics(**item["l4"]),
                L5Metrics(**item["l5"]),
                meta,
            )
        )

    return {
        "baseline": report_path.parent.name,
        "items": len(items),
        "stored_score": stored,
        "recomputed_score": sum(recomputed) / len(recomputed) if recomputed else 0.0,
        "passed": sum(1 for i in items if i["passed"]),
        "unknown_metadata": unknown_meta,
        "comparable": comparable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--baseline", type=str, default=None, help="특정 baseline만")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    requires_kg = load_requires_kg(args.dataset)
    if not requires_kg:
        print(f"경고: 데이터셋을 읽지 못했다 ({args.dataset}) — 리포트의 metadata를 그대로 쓴다")

    names = (
        [args.baseline]
        if args.baseline
        else sorted(p.name for p in args.baseline_dir.iterdir() if p.is_dir())
    )

    rows = []
    for name in names:
        report_path = args.baseline_dir / name / "report.json"
        if not report_path.exists():
            print(f"건너뜀 (report.json 없음): {name}")
            continue
        rows.append(reaggregate(report_path, requires_kg))

    print()
    print("| baseline | 문항 | 기존 공식 | 새 공식 | Δ | 통과 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if not r["comparable"]:
            print(
                f"| {r['baseline']} | {r['items']} | {r['stored_score']:.3f} "
                f"| — | — | {r['passed']} |  ← 새 공식의 지표 미기록"
            )
            continue
        delta = r["recomputed_score"] - r["stored_score"]
        print(
            f"| {r['baseline']} | {r['items']} | {r['stored_score']:.3f} "
            f"| {r['recomputed_score']:.3f} | {delta:+.3f} | {r['passed']} |"
        )
    if any(not r["comparable"] for r in rows):
        print(
            "\n새 공식이 쓰는 개념 recall(사이클 6 도입)·엣지 recall(사이클 4 도입)이 "
            "기록되지 않은 baseline은 재집계하지 않는다. 그 값들을 0으로 채우면 "
            "지표 부재를 성능 저하로 오독하게 된다."
        )
    unknown = sum(r["unknown_metadata"] for r in rows)
    if unknown:
        print(f"\n데이터셋에서 찾지 못한 문항 {unknown}건은 리포트 metadata를 그대로 사용했다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
