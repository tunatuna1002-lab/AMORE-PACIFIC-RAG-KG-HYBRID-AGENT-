#!/usr/bin/env python3
"""골든셋 문항을 **질문 유형**(필요한 추론 능력)으로 분류해 유형별 시험지를 만든다.

배경 (docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md 1-A)
--------------------------------------------------------------------------
평가 지표를 유형별로 보려면(수치 질문엔 metric 증거, 관계 질문엔 relation 증거, 판단 질문엔
inference 증거가 필요한지) 분모가 되는 시험지가 먼저 있어야 한다. 이 스크립트는 LLM 없이
gold 필드와 질문 구조만 보고 **결정적으로** 분류한다. 같은 입력이면 같은 출력이다(`--check`).

유형
----
numeric   수치 조회. 이름이 주어진 엔티티의 지표·순위·가격을 한 번에 조회해 답한다.
relation  관계. 엔티티 사이의 관계(경쟁·소속·제품 구성)를 묻는다.
rule      규칙 판단. 규칙 엔진(src/ontology/rules)의 결론 개념(시장 구조·포지션·기회·
          가격 포지션)을 판정하거나, 지표 조건이 주어졌을 때의 판단을 묻는다.
multihop  다단계. 질문에 이름이 없는 엔티티를 관계·최상급으로 먼저 해석한 뒤 그 결과로
          다시 조회해야 한다(엔티티 해석 → 관계 → 수치 → 판단 중 2단계 이상의 연쇄).
other     위 넷이 아닌 것. 억지로 넣지 않는다 — subkind로 이유를 남긴다
          (definition·ir_document·out_of_scope·forecast·document_explanation·
          document_general).

분류 규칙 (위에서부터 처음 맞는 규칙이 주 유형, 나머지 탐지기는 secondary로 기록)
-------------------------------------------------------------------------------
C1_OUT_OF_SCOPE     concepts ∩ {out_of_scope, adversarial, ethical_boundary,
                    subjective_query}                                 → other/out_of_scope
C2_IR_DOCUMENT      metadata.domain == "ir" (IR 문서 속 실적 수치)     → other/ir_document
C3_FORECAST         concepts ∩ {future_forecast, forecasting}         → other/forecast
C4_DEFINITION       concepts ∩ {definition, calculation, data_requirements, kpi_framework}
                    또는 질문이 정의·계산법·지표 속성을 묻는 표현(DEFINITION_RE)  → other/definition
C5_RULE_HYPOTHETICAL 지표 이름 + (구체 값이 주어진 판정 | 조건절) + 판정·전략·원인·구조를
                    묻는 표현. 인과 "영향/효과"는 규칙 결론이 아니므로 제외   → rule
C6_MULTIHOP_CHAIN   질문에 이름이 없는 엔티티를 먼저 해석해야 하는 연쇄 표현(CHAIN_RE),
                    또는 카테고리명이 없는데 카테고리 수준 지표·1위를 묻는 경우 → multihop
C7_RELATION         관계를 묻는 표현(RELATION_RE). "경쟁 제품 대비 … 가격"처럼 관계가
                    비교 기준일 뿐 묻는 대상이 수치이면 제외           → relation
C8_RULE_CONCEPT     브랜드가 명시되고 규칙 결론 개념(시장 위치·성장 가능성·가성비 등,
                    RULE_CONCEPT_RE)을 묻는다                          → rule
C9_MULTIHOP_DOMAIN  domain == multi_hop 이고 DB 근거(snapshot 또는 expected_values)가
                    있는데 C6에 안 걸린 문항 — 약한 근거라 경계 문항으로 기록 → multihop
C10_NUMERIC         (expected_values 비어 있지 않고 gold_source=snapshot) 또는
                    snapshot 문항, 또는 수치를 묻는 표현(NUMERIC_ASK_RE) → numeric
C11_FALLBACK        나머지 → other/document_explanation (gold_source=document) 또는
                    other/document_general (domain_expectation)

시험지 포함 여부
----------------
`gold_source == "domain_expectation"` 문항은 주 유형이 넷 중 하나여도 시험지에 넣지 않는다
(`in_sheet=false`, `holdout_reason`). 골드가 DB에도 문서에도 없는 추정치라 정답 채점의
분모로 쓸 수 없기 때문이다(eval/schemas.py ItemMetadata.gold_source). 주 유형은 그대로
기록되므로 리드가 포함 여부를 다시 정할 수 있다.

생성 문항
---------
`eval/data/golden/typed/generated_{rule,multihop,relation}.jsonl`이 있으면 해당 유형
시험지 끝에 붙인다. 생성 문항의 유형은 생성기가 선언한 `metadata.question_type`을 따르고
(분류 규칙으로 다시 판정하지 않는다) classification.jsonl에 `G0_GENERATED`로 남긴다.

출력
----
eval/data/golden/typed/classification.jsonl   문항별 id·type·subkind·secondary·reasons·
                                               gold_source·in_sheet·boundary
eval/data/golden/typed/{numeric,relation,rule,multihop}.jsonl
    원본 레코드를 그대로 복사하고 metadata에 `question_type`만 추가한다(로더는 알 수 없는
    metadata 키를 무시한다 — tests/eval/test_classify_golden_types.py가 확인).

사용법:
    python3 scripts/classify_golden_types.py            # 파일 생성
    python3 scripts/classify_golden_types.py --check    # 디스크 파일과 재계산 결과 비교(다르면 1)
    python3 scripts/classify_golden_types.py --summary  # 파일을 쓰지 않고 요약만 출력
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "eval" / "data" / "golden" / "laneige_golden_v2.jsonl"
TYPED_DIR = REPO_ROOT / "eval" / "data" / "golden" / "typed"

TYPES = ("numeric", "relation", "rule", "multihop")
GENERATED_FILES = {
    "rule": "generated_rule.jsonl",
    "multihop": "generated_multihop.jsonl",
    "relation": "generated_relation.jsonl",
}

# ---------------------------------------------------------------------------
# 어휘 (질문 구조 탐지). 새 표현을 추가하면 --check 결과가 바뀌므로 테스트도 갱신할 것.
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_CONCEPTS = {"out_of_scope", "adversarial", "ethical_boundary", "subjective_query"}
FORECAST_CONCEPTS = {"future_forecast", "forecasting"}
DEFINITION_CONCEPTS = {"definition", "calculation", "data_requirements", "kpi_framework"}

DEFINITION_RE = re.compile(
    r"(란\s*무엇|이란|정의|계산\s*(방식|방법|공식)|계산에 필요한|의미하는 바|"
    r"(이|가) (마이너스|음수)가 될 수|해석 방법|가장 중요한 KPI)"
)

METRIC_RE = r"(SoS|HHI|CPI|이탈률|점유율)"
# (a) 구체 값이 주어진 판정: "SoS 5%는 좋은 수치인가요?", "HHI 0.15는 어떤 시장 구조"
RULE_GIVEN_VALUE_RE = re.compile(METRIC_RE + r"\s*[0-9.]+%?\s*(는|은)\s*.*(좋은|시장 구조|의미)")
# (b) 조건절 + 판정·전략·원인·상황
RULE_CONDITION_RE = re.compile(
    METRIC_RE + r".*(높으면|낮으면|높은데|미만일 때|이상일 때|초과일 때|하락할 때|"
    r"상승할 때|하락 중이라면|상승 중이라면)"
)
RULE_CONDITION_ASK_RE = re.compile(r"(상황|전략|원인|구조|좋은)")

CATEGORY_NAME_RE = re.compile(
    r"(lip care|lip makeup|face powder|face makeup|skin care|(?<!k-)beauty|뷰티|립케어|립스틱)",
    re.IGNORECASE,
)
CHAIN_RE = re.compile(
    r"(속한 카테고리|같은 카테고리|1위 브랜드의|카테고리에서 1위 브랜드|모회사|주력 카테고리|"
    r"가장 [^?]*?(카테고리|제품)(의|과|와)\s|브랜드 중 [^?]*가장)"
)
UNNAMED_CATEGORY_ASK_RE = re.compile(r"(HHI|집중도|SoS|점유율|1위|경쟁이 가장 심한)")

RELATION_RE = re.compile(
    r"(경쟁 (제품|관계|브랜드)|경쟁사와|와의 관계|포트폴리오에서 차지하는 위치|라인 제품 구성|"
    r"모회사|자매 브랜드|소속|세그먼트|원산지|인수)"
)
RELATION_AS_BASELINE_RE = re.compile(r"대비 [^?]*가격")

BRAND_RE = re.compile(
    r"(LANEIGE|라네즈|COSRX|TIRTIR|MEDICUBE|BIODANCE|Beauty of Joseon|ANUA|Anua|Aquaphor|"
    r"Burt's Bees|ChapStick|eos|NYX|Maybelline|Rare Beauty|e\.l\.f\.|아모레퍼시픽|AMOREPACIFIC|"
    r"Hera|HERA|Sulwhasoo|innisfree|ETUDE)"
)
RULE_CONCEPT_RE = re.compile(
    r"(시장 위치|시장 포지션|성장 가능성|가성비가 가장 좋은|분산 시장|집중 시장|도전자|지배자|"
    r"가격-품질 불일치|진입 기회)"
)
RULE_SECONDARY_RE = re.compile(r"(위협|기회|가격 경쟁력|포지셔닝|경쟁이 가장 심한)")

NUMERIC_ASK_RE = re.compile(
    r"(얼마|순위|현황|몇|평균 가격|Top \d+|상위|목록|가장 많은|가장 높은|ASIN|비교해주세요|"
    r"수준은|SoS는|HHI는|점유율은|비중은|규모는|추이는|변화는|패턴은|최고점|성장률|"
    r"차이는\?|show me)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 분류
# ---------------------------------------------------------------------------


def _q(item: dict[str, Any]) -> str:
    return item["question"]


def _concepts(item: dict[str, Any]) -> set[str]:
    return set(item["gold"].get("concepts") or [])


def _ev(item: dict[str, Any]) -> dict[str, Any]:
    return item["gold"].get("expected_values") or {}


def _source(item: dict[str, Any]) -> str:
    return item["metadata"].get("gold_source", "document")


def detect_hypothetical_rule(item: dict[str, Any]) -> str | None:
    q = _q(item)
    if RULE_GIVEN_VALUE_RE.search(q):
        return (
            f"C5_RULE_HYPOTHETICAL: 값이 주어진 지표 판정 '{RULE_GIVEN_VALUE_RE.search(q).group()}'"
        )
    cond = RULE_CONDITION_RE.search(q)
    if cond and RULE_CONDITION_ASK_RE.search(q):
        return (
            f"C5_RULE_HYPOTHETICAL: 지표 조건절 '{cond.group()}' + "
            f"판정 요청 '{RULE_CONDITION_ASK_RE.search(q).group()}'"
        )
    return None


def detect_chain(item: dict[str, Any]) -> str | None:
    q = _q(item)
    m = CHAIN_RE.search(q)
    if m:
        return f"C6_MULTIHOP_CHAIN: 연쇄 표현 '{m.group().strip()}'"
    if "카테고리" in q and not CATEGORY_NAME_RE.search(q):
        ask = UNNAMED_CATEGORY_ASK_RE.search(q)
        if ask:
            return (
                "C6_MULTIHOP_CHAIN: 카테고리명 없이 카테고리 수준 대상을 물음 "
                f"'{ask.group()}' → 카테고리를 먼저 해석해야 함"
            )
    return None


def detect_relation(item: dict[str, Any]) -> str | None:
    q = _q(item)
    m = RELATION_RE.search(q)
    if not m:
        return None
    if RELATION_AS_BASELINE_RE.search(q):
        return None
    return f"C7_RELATION: 관계 표현 '{m.group()}'"


def detect_rule_concept(item: dict[str, Any]) -> str | None:
    q = _q(item)
    m = RULE_CONCEPT_RE.search(q)
    b = BRAND_RE.search(q)
    if m and b:
        return f"C8_RULE_CONCEPT: 브랜드 '{b.group()}' + 규칙 결론 개념 '{m.group()}'"
    return None


def detect_numeric(item: dict[str, Any]) -> str | None:
    ev = _ev(item)
    src = _source(item)
    if ev and src == "snapshot":
        return f"C10_NUMERIC: gold_source=snapshot, expected_values 키 {sorted(ev)}"
    if src == "snapshot":
        return "C10_NUMERIC: gold_source=snapshot (expected_values 없음)"
    q = _q(item)
    m = NUMERIC_ASK_RE.search(q)
    named = BRAND_RE.search(q) or CATEGORY_NAME_RE.search(q)
    if m and named:
        return (
            f"C10_NUMERIC: 엔티티 '{named.group()}' + 수치를 묻는 표현 '{m.group()}' "
            f"(gold_source={src})"
        )
    return None


def classify_item(item: dict[str, Any]) -> dict[str, Any]:
    """원본 골든 문항 하나를 분류한다. 순수 함수(같은 입력 → 같은 출력)."""
    q = _q(item)
    concepts = _concepts(item)
    domain = item["metadata"].get("domain")
    src = _source(item)
    reasons: list[str] = []
    primary: str | None = None
    subkind: str | None = None

    hypo = detect_hypothetical_rule(item)
    chain = detect_chain(item)
    rel = detect_relation(item)
    rule_concept = detect_rule_concept(item)
    numeric = detect_numeric(item)

    if concepts & OUT_OF_SCOPE_CONCEPTS:
        primary, subkind = "other", "out_of_scope"
        reasons.append(f"C1_OUT_OF_SCOPE: concepts ∩ {sorted(concepts & OUT_OF_SCOPE_CONCEPTS)}")
    elif domain == "ir":
        primary, subkind = "other", "ir_document"
        reasons.append("C2_IR_DOCUMENT: metadata.domain=ir (IR 문서의 실적 수치, DB·KG 무관)")
    elif concepts & FORECAST_CONCEPTS:
        primary, subkind = "other", "forecast"
        reasons.append(f"C3_FORECAST: concepts ∩ {sorted(concepts & FORECAST_CONCEPTS)}")
    elif concepts & DEFINITION_CONCEPTS or DEFINITION_RE.search(q):
        primary, subkind = "other", "definition"
        hit = sorted(concepts & DEFINITION_CONCEPTS)
        m = DEFINITION_RE.search(q)
        reasons.append(
            "C4_DEFINITION: "
            + (f"concepts ∩ {hit}" if hit else "")
            + (" / " if hit and m else "")
            + (f"표현 '{m.group()}'" if m else "")
        )
    elif hypo:
        primary = "rule"
        reasons.append(hypo)
    elif chain:
        primary = "multihop"
        reasons.append(chain)
    elif rel:
        primary = "relation"
        reasons.append(rel)
    elif rule_concept:
        primary = "rule"
        reasons.append(rule_concept)
    elif domain == "multi_hop" and (src == "snapshot" or _ev(item)):
        primary = "multihop"
        reasons.append(
            f"C9_MULTIHOP_DOMAIN: domain=multi_hop, gold_source={src}, "
            f"expected_values 키 {sorted(_ev(item))} (연쇄 표현은 없음 — 약한 근거)"
        )
    elif numeric:
        primary = "numeric"
        reasons.append(numeric)
    else:
        primary = "other"
        subkind = "document_explanation" if src == "document" else "document_general"
        reasons.append(
            f"C11_FALLBACK: 네 유형 탐지기 모두 불일치 (gold_source={src}, domain={domain})"
        )

    # 보조 유형: 주 유형 외에 걸린 탐지기
    secondary: list[str] = []
    detectors = [
        ("rule", hypo or rule_concept or (RULE_SECONDARY_RE.search(q) and BRAND_RE.search(q))),
        ("multihop", chain),
        ("relation", rel or RELATION_RE.search(q)),
        ("numeric", numeric if (src == "snapshot" or _ev(item)) else None),
    ]
    for t, hit in detectors:
        if hit and t != primary:
            secondary.append(t)
            if isinstance(hit, str):
                reasons.append(f"secondary {t} ← {hit}")
            else:
                reasons.append(f"secondary {t} ← 표현 '{hit.group()}'")

    in_sheet = primary in TYPES and src != "domain_expectation"
    boundary: list[str] = []
    # multihop은 정의상 관계·수치 단계를 포함하므로 그 둘은 경계 사유로 보지 않는다
    implied = {"numeric", "relation"} if primary == "multihop" else set()
    conflicting = [t for t in secondary if t not in implied]
    if primary in TYPES and conflicting:
        boundary.append(f"주 유형 {primary}와 보조 유형 {conflicting} 기준을 동시에 만족")
    if primary in TYPES and reasons[0].startswith("C9_"):
        boundary.append("domain 라벨만으로 multihop — 연쇄 표현 없음")
    if primary in ("rule", "multihop") and src == "document":
        boundary.append(
            f"{primary} 구조지만 골드가 문서 서술(gold_source=document) — "
            "DB·KG로 검증 가능한 판정/연쇄 정답이 없음"
        )
    if primary == "other" and any(t in TYPES for t in secondary):
        boundary.append(f"other로 두었지만 {secondary} 탐지기가 걸림")

    return {
        "id": item["id"],
        "type": primary,
        "subkind": subkind,
        "secondary": secondary,
        "reasons": reasons,
        "gold_source": src,
        "domain": domain,
        "in_sheet": in_sheet,
        "holdout_reason": (
            "gold_source=domain_expectation (정답 검증 불가)"
            if primary in TYPES and not in_sheet
            else None
        ),
        "boundary": boundary,
    }


# ---------------------------------------------------------------------------
# 입출력
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def dumps_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def with_question_type(record: dict[str, Any], question_type: str) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out["metadata"]["question_type"] = question_type
    return out


def build_outputs(
    items: list[dict[str, Any]], generated: dict[str, list[dict[str, Any]]]
) -> dict[str, str]:
    """{파일명: 내용}. 파일 쓰기와 --check가 같은 함수를 쓴다."""
    rows = [classify_item(item) for item in items]
    sheets: dict[str, list[dict[str, Any]]] = {t: [] for t in TYPES}
    for item, row in zip(items, rows, strict=True):
        if row["in_sheet"]:
            sheets[row["type"]].append(with_question_type(item, row["type"]))

    for qtype, records in generated.items():
        for record in records:
            declared = record["metadata"].get("question_type")
            if declared != qtype:
                raise ValueError(f"{record['id']}: 선언 유형 {declared} != 파일 유형 {qtype}")
            rows.append(
                {
                    "id": record["id"],
                    "type": qtype,
                    "subkind": None,
                    "secondary": [],
                    "reasons": [
                        f"G0_GENERATED: {record['metadata'].get('generator')}가 선언한 유형"
                    ],
                    "gold_source": record["metadata"].get("gold_source"),
                    "domain": record["metadata"].get("domain"),
                    "in_sheet": True,
                    "holdout_reason": None,
                    "boundary": [],
                }
            )
            sheets[qtype].append(record)

    outputs = {"classification.jsonl": dumps_jsonl(rows)}
    for t in TYPES:
        outputs[f"{t}.jsonl"] = dumps_jsonl(sheets[t])
    return outputs


def load_generated(typed_dir: Path) -> dict[str, list[dict[str, Any]]]:
    generated: dict[str, list[dict[str, Any]]] = {}
    for qtype, name in GENERATED_FILES.items():
        path = typed_dir / name
        if path.exists():
            generated[qtype] = read_jsonl(path)
    return generated


def summarize(outputs: dict[str, str]) -> str:
    rows = [json.loads(line) for line in outputs["classification.jsonl"].splitlines()]
    lines = []
    generated = [r for r in rows if r["reasons"][0].startswith("G0_")]
    original = [r for r in rows if not r["reasons"][0].startswith("G0_")]
    by_type = collections.Counter(r["type"] for r in original)
    in_sheet = collections.Counter(r["type"] for r in original if r["in_sheet"])
    gen = collections.Counter(r["type"] for r in generated)
    lines.append(f"원본 {len(original)}문항 주 유형: {dict(sorted(by_type.items()))}")
    for t in TYPES:
        lines.append(
            f"  {t:9s} 원본 {by_type.get(t, 0):3d} (시험지 포함 {in_sheet.get(t, 0):3d}) "
            f"+ 생성 {gen.get(t, 0):3d} = 시험지 {in_sheet.get(t, 0) + gen.get(t, 0):3d}"
        )
    other_kinds = collections.Counter(r["subkind"] for r in original if r["type"] == "other")
    lines.append(f"  other {by_type.get('other', 0)}: {dict(sorted(other_kinds.items()))}")
    held = [r["id"] for r in original if r["type"] in TYPES and not r["in_sheet"]]
    lines.append(f"  domain_expectation 보류 {len(held)}: {held}")
    boundary = [r for r in original if r["boundary"]]
    lines.append(f"  경계 문항 {len(boundary)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=TYPED_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="디스크 파일과 재계산 결과 비교")
    mode.add_argument("--summary", action="store_true", help="파일을 쓰지 않고 요약만 출력")
    args = parser.parse_args()

    items = read_jsonl(args.dataset)
    outputs = build_outputs(items, load_generated(args.out))

    if args.check:
        stale = [
            name
            for name, content in outputs.items()
            if not (args.out / name).exists()
            or (args.out / name).read_text(encoding="utf-8") != content
        ]
        if stale:
            print(f"재계산 결과와 다른 파일: {stale}")
            return 1
        print("OK — 분류 결과가 디스크 파일과 같다")
        return 0

    print(summarize(outputs))
    if args.summary:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.out / name).write_text(content, encoding="utf-8")
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
