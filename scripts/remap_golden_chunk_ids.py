"""골든셋 gold.doc_chunk_ids를 실제 ChromaDB 코퍼스 청크 ID로 재매핑

배경 (2026-08-30 사실 검증 감사):
    골든셋 v2의 doc_chunk_ids 8종은 가상 ID 체계(metric_guide_sos_01 등)로,
    실제 코퍼스(data/chroma, amore_docs 271청크)와 겹침이 0이어서
    L2(Context Recall) 지표가 구조적으로 0이었다.

매핑 원칙:
    가상 ID는 문서 수준 개념이므로, 각 개념의 정의/해석을 담은
    가장 canonical한 실제 청크 1개로 1:1 매핑한다 (recall 판정을 이진화).
    매핑 근거는 청크 본문 확인 결과 (strategic_indicators_*: 지표 정의·산출식,
    metric_interpretation_*: 해석 가이드, laneige_strategy_2026_*: 시장/경쟁 분석,
    amazon_ranking_diagnosis_*: 랭킹 대응 체크리스트).

사용법:
    python3 scripts/remap_golden_chunk_ids.py            # 적용
    python3 scripts/remap_golden_chunk_ids.py --dry-run  # 변경 내역만 출력
"""

import argparse
import json
from pathlib import Path

# 가상 골드 ID → 실제 코퍼스 청크 ID
CHUNK_ID_MAP: dict[str, str] = {
    "metric_guide_sos_01": "strategic_indicators_1_0",  # SoS 정의·산출식
    "metric_guide_hhi_01": "strategic_indicators_1_0",  # HHI 정의·산출식 (동일 청크 내)
    "metric_guide_hhi_02": "metric_interpretation_2_1",  # HHI 해석 가이드
    "metric_guide_cpi_01": "strategic_indicators_2_0",  # CPI 정의·산출식
    "metric_guide_rank_01": "strategic_indicators_3_0",  # Rank Volatility/Shock 정의
    "market_trend_01": "laneige_strategy_2026_1_0",  # 시장 거시 환경/트렌드
    "competitor_analysis_01": "laneige_strategy_2026_3_1",  # 경쟁 브랜드 분석
    "playbook_ranking_01": "amazon_ranking_diagnosis_3_0",  # 랭킹 변동 대응 체크리스트
}

TARGET_FILES = [
    "eval/data/golden/laneige_golden_v2.jsonl",
    "eval/data/golden/subset_nokg.jsonl",
]


def remap_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """파일 하나를 재매핑. (변경 항목 수, 치환된 ID 수) 반환."""
    items_changed = 0
    ids_replaced = 0
    out_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        chunk_ids = item.get("gold", {}).get("doc_chunk_ids") or []
        new_ids: list[str] = []
        changed = False
        for cid in chunk_ids:
            mapped = CHUNK_ID_MAP.get(cid, cid)
            if mapped != cid:
                changed = True
                ids_replaced += 1
            if mapped not in new_ids:  # 동일 청크로 합쳐지는 경우 중복 제거
                new_ids.append(mapped)
        if changed:
            item["gold"]["doc_chunk_ids"] = new_ids
            items_changed += 1
        out_lines.append(json.dumps(item, ensure_ascii=False))
    if not dry_run:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return items_changed, ids_replaced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    for rel in TARGET_FILES:
        path = root / rel
        changed, replaced = remap_file(path, args.dry_run)
        tag = "(dry-run)" if args.dry_run else ""
        print(f"{rel}: {changed} items changed, {replaced} ids replaced {tag}")


if __name__ == "__main__":
    main()
