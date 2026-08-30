"""골든셋에 IR(분기 실적) 도메인 문항 12개 추가 (2026-08-30 사이클 7)

배경:
    코퍼스의 24%(IR 분기보고서 본문 87청크)를 다루는 골든셋 문항이 0개였다.
    사이클 6에서 교차언어 검색을 고쳐 IR 문서가 실제로 검색 가능해졌으므로,
    그 능력을 상시 측정할 문항이 필요하다.

작성 원칙 (기존 문항의 실데이터 관례 준수):
    - gold.answer의 모든 수치는 `docs/ir/AP_*Q25_EN.md` 원문에서 그대로 인용했다.
      각 문항의 `evidence` 주석에 근거 청크와 원문 표현을 남긴다.
    - gold.doc_chunk_groups는 그 수치를 실제로 담고 있는 청크만 넣는다.
      한 그룹 안의 청크는 서로 대체 가능한 근거이므로 하나만 검색돼도 정답으로 본다.
    - 한국어 8문항 / 영어 4문항으로 섞어 교차언어 회수율을 골든셋 안에서 감시한다.
    - requires_kg=false — IR 사실은 KG가 아니라 문서에서 나온다.

주의:
    문항 추가는 overall·pass_rate의 **분모를 바꾼다**. v7.1 이전 baseline과
    집계 수치를 직접 비교하지 말고, 기존 160문항 부분집합으로 비교할 것.

사용법:
    python3 scripts/add_ir_golden_items.py --dry-run
    python3 scripts/add_ir_golden_items.py
"""

import argparse
import json
from pathlib import Path

TARGET = "eval/data/golden/laneige_golden_v2.jsonl"

IR_ITEMS: list[dict] = [
    {
        "id": "lg201",
        "question": "아모레퍼시픽 2025년 1분기 매출은 얼마인가요?",
        # evidence: ir_2025_q1_0_1 "Amorepacific revenue up 17.1% to 1.1 trillion KRW",
        #           "Revenue1) 911.5 100.0 1,067.5 100.0 +17.1", "Domestic Business ... +2.4"
        "answer": (
            "2025년 1분기 매출은 1조 675억원(1,067.5십억원)으로 전년 동기 9,115억원 대비 "
            "17.1% 증가했습니다. 국내 사업은 2.4%, 해외 사업은 40.5% 성장했습니다."
        ),
        "groups": [["ir_2025_q1_0_1", "ir_2025_q1_0_7", "ir_2025_q1_table_0"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "revenue"],
        "difficulty": "easy",
    },
    {
        "id": "lg202",
        "question": "AMOREPACIFIC 1Q25 operating profit and year-over-year change",
        # evidence: ir_2025_q1_0_2 "operating profit up 62.0% to 117.7 billion KRW",
        #           ir_2025_q1_0_3 "72.7* (8.0%) ... 117.7* (11.0%)"
        "answer": (
            "1Q 2025 operating profit was KRW 117.7 billion, up 62.0% year-over-year from "
            "KRW 72.7 billion. Operating profit margin improved from 8.0% to 11.0%."
        ),
        "groups": [["ir_2025_q1_0_2", "ir_2025_q1_0_3", "ir_2025_q1_0_7"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "operating_profit"],
        "difficulty": "easy",
    },
    {
        "id": "lg203",
        "question": "2025년 1분기 아모레퍼시픽 서구권 매출 성장률은 얼마인가요?",
        # evidence: ir_2025_q1_0_2 "Western Region 105.1 11.5 212.5 19.9 +102.1",
        #           ir_2025_q1_0_19 "Americas revenue increased 79%", "EMEA 17.3 55.3 +219%"
        "answer": (
            "서구권 매출은 1,051억원에서 2,125억원으로 102.1% 증가했습니다. "
            "미주가 79%, EMEA가 219% 성장했습니다."
        ),
        "groups": [["ir_2025_q1_0_2", "ir_2025_q1_0_19"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "regional_revenue"],
        "difficulty": "medium",
    },
    {
        "id": "lg204",
        "question": "2025년 1분기 아모레퍼시픽 중화권 실적은 어땠나요?",
        # evidence: ir_2025_q1_0_2 "Greater China 148.2 16.3 132.8 12.4 -10.4",
        #           ir_2025_q1_0_22 "China business turned to profit"
        "answer": (
            "중화권 매출은 1,482억원에서 1,328억원으로 10.4% 감소했습니다. "
            "다만 오프라인 채널 구조조정에도 주요 온라인 채널의 사업 구조 개선과 "
            "비용 절감으로 중국 사업은 흑자 전환했습니다."
        ),
        "groups": [["ir_2025_q1_0_2", "ir_2025_q1_0_22"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "regional_revenue"],
        "difficulty": "medium",
    },
    {
        "id": "lg205",
        "question": "아모레퍼시픽 2025년 2분기 영업이익은 얼마인가요?",
        # evidence: ir_2025_q2_0_3 "operating profit up 1673.4% to 73.7 billion KRW",
        #           ir_2025_q2_0_4 "+611% +164%", ir_2025_q2_0_7 "Operating Profit 4.2 0.5 73.7 7.3 +1673.4"
        "answer": (
            "2025년 2분기 영업이익은 737억원으로 전년 동기 42억원 대비 1673.4% 증가했습니다. "
            "국내 사업은 164%, 해외 사업은 611% 증가했습니다."
        ),
        "groups": [["ir_2025_q2_0_3", "ir_2025_q2_0_4", "ir_2025_q2_0_7"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "operating_profit"],
        "difficulty": "easy",
    },
    {
        "id": "lg206",
        "question": "AMOREPACIFIC 2Q25 revenue growth by region",
        # evidence: ir_2025_q2_0_2 "Revenue1) 904.8 ... 1,005.0 ... +11.1",
        #           "Western Region ... +12.2 Other Asia ... +9.3 Greater China ... +23.2"
        "answer": (
            "2Q 2025 revenue rose 11.1% to KRW 1,005.0 billion. Domestic business grew 8.2% and "
            "overseas 14.4%: Western Region +12.2%, Other Asia +9.3%, Greater China +23.2%."
        ),
        "groups": [["ir_2025_q2_0_2"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "regional_revenue"],
        "difficulty": "medium",
    },
    {
        "id": "lg207",
        "question": "2025년 2분기 아모레퍼시픽 매출총이익률은 어떻게 변했나요?",
        # evidence: ir_2025_q2_0_6 "Gross profit margin up 2.1%p ... Marketing expenses down 11%",
        #           ir_2025_q2_0_7 "Gross Profit 638.1 70.5 729.9 72.6 +14.4"
        "answer": (
            "매출총이익률은 70.5%에서 72.6%로 2.1%p 개선됐습니다. "
            "프로모션 통제 강화와 브랜드 믹스 개선이 요인이며, 마케팅비는 11% 감소했습니다."
        ),
        "groups": [["ir_2025_q2_0_6", "ir_2025_q2_0_7"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "profitability"],
        "difficulty": "hard",
    },
    {
        "id": "lg208",
        "question": "How did LANEIGE perform in the Americas in 2Q 2025?",
        # evidence: ir_2025_q2_0_19 "Americas revenue grew 10% ... [Laneige] ... 'Glaze Craze Lip
        #           Serum,' new launch of 'Bubble Tea Collection(Lip Sleeping Mask, Lip GlowyBalm)'"
        "answer": (
            "Americas revenue grew 10% in 2Q 2025. LANEIGE achieved robust sales growth in both "
            "lip and skincare categories, driven by 'Glaze Craze Lip Serum', the new 'Bubble Tea "
            "Collection' (Lip Sleeping Mask, Lip Glowy Balm), and continued demand for "
            "'Bouncy and Firm Serum'."
        ),
        "groups": [["ir_2025_q2_0_19"]],
        "entities": ["laneige"],
        "concepts": ["data_query", "regional_revenue"],
        "difficulty": "medium",
    },
    {
        "id": "lg209",
        "question": "아모레퍼시픽 2025년 3분기 영업이익 증가율은 얼마인가요?",
        # evidence: ir_2025_q3_0_2 "Operating Profit 65.2 6.7% 91.9 9.0% +2.4 +41.0",
        #           ir_2025_q3_0_3 "Operating profit increased 41.0% to 91.9 billion KRW"
        "answer": (
            "2025년 3분기 영업이익은 919억원으로 전년 동기 652억원 대비 41.0% 증가했습니다. "
            "영업이익률은 6.7%에서 9.0%로 개선됐습니다."
        ),
        "groups": [["ir_2025_q3_0_2", "ir_2025_q3_0_3"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "operating_profit"],
        "difficulty": "easy",
    },
    {
        "id": "lg210",
        "question": "AMOREPACIFIC 3Q25 net income",
        # evidence: ir_2025_q3_0_2 "Net Income 37.2 3.8% 68.2 6.7% +2.9 +83.6"
        "answer": (
            "3Q 2025 net income was KRW 68.2 billion, up 83.6% year-over-year from "
            "KRW 37.2 billion, improving from 3.8% to 6.7% of revenue."
        ),
        "groups": [["ir_2025_q3_0_2"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "net_income"],
        "difficulty": "easy",
    },
    {
        "id": "lg211",
        "question": "2025년 3분기 아모레퍼시픽 해외 사업 매출은 얼마인가요?",
        # evidence: ir_2025_q3_0_6 "Overseas Business 428.4 43.8% 440.8 43.4% +2.9",
        #           "Americas ... +6.9 EMEA ... -3.2"
        "answer": (
            "3분기 해외 사업 매출은 4,408억원으로 전년 동기 4,284억원 대비 2.9% 증가했습니다. "
            "미주는 6.9% 증가, EMEA는 3.2% 감소했습니다."
        ),
        "groups": [["ir_2025_q3_0_6"]],
        "entities": ["amorepacific"],
        "concepts": ["data_query", "regional_revenue"],
        "difficulty": "medium",
    },
    {
        "id": "lg212",
        "question": "2025년 3분기 라네즈 립 슬리핑 마스크 신제품 에디션은 무엇인가요?",
        # evidence: ir_2025_q3_0_18 "Launched new editions of the Lip Sleeping Mask
        #           ('Baskin Robbins,' 'Strawberry Shortcake')", ir_2025_q3_0_21 동일 제품 언급
        "answer": (
            "3분기에 립 슬리핑 마스크의 새 에디션인 '배스킨라빈스(Baskin Robbins)'와 "
            "'스트로베리 쇼트케이크(Strawberry Shortcake)'를 출시해 립 카테고리 리더십을 "
            "강화했습니다."
        ),
        "groups": [["ir_2025_q3_0_18", "ir_2025_q3_0_21"]],
        "entities": ["laneige", "lip_sleeping_mask"],
        "concepts": ["data_query", "product_launch"],
        "difficulty": "medium",
    },
]


def build_item(spec: dict) -> dict:
    flat: list[str] = []
    for group in spec["groups"]:
        for cid in group:
            if cid not in flat:
                flat.append(cid)
    return {
        "id": spec["id"],
        "question": spec["question"],
        "gold": {
            "answer": spec["answer"],
            "doc_chunk_ids": flat,
            "doc_chunk_groups": spec["groups"],
            "kg_entities": spec["entities"],
            "kg_edges": [],
            "concepts": spec["concepts"],
            "constraints": [],
            "expected_values": {},
        },
        "metadata": {
            "requires_kg": False,
            "domain": "ir",
            "difficulty": spec["difficulty"],
        },
    }


def verify_against_corpus(root: Path) -> None:
    """골드 청크가 실제 코퍼스에 존재하는지 확인."""
    import chromadb

    client = chromadb.PersistentClient(path=str(root / "data" / "chroma"))
    corpus = set(client.get_collection("amore_docs").get(include=[])["ids"])
    missing = sorted(
        {cid for spec in IR_ITEMS for group in spec["groups"] for cid in group} - corpus
    )
    if missing:
        raise SystemExit(f"코퍼스에 없는 골드 청크: {missing}")
    print(f"골드 청크 검증 통과 (코퍼스 {len(corpus)}청크)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    verify_against_corpus(root)

    path = root / TARGET
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    existing = {json.loads(ln)["id"] for ln in lines}

    added = 0
    for spec in IR_ITEMS:
        if spec["id"] in existing:
            continue
        lines.append(json.dumps(build_item(spec), ensure_ascii=False))
        added += 1

    tag = "(dry-run)" if args.dry_run else ""
    print(f"{TARGET}: {len(existing)} → {len(existing) + added} 문항 (추가 {added}) {tag}")
    if not args.dry_run and added:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
