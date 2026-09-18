#!/usr/bin/env python3
"""신뢰도 임계값을 측정 데이터로 보정한다 (트랙 5-B).

무엇을 하는가
-------------
이미 끝난 평가 실행의 ``report.json``을 다시 읽어, 각 문항에 저장된 **증거 카드와 엔티티
링킹 결과**를 실제 채점 함수 ``src.core.confidence.score_evidence_fit()``에 그대로 다시
먹인다. LLM도 검색도 부르지 않는다 — 저장된 입력을 재생할 뿐이다. 그 다음 문항별 결과
(``passed``, L5 게이트)를 놓고 HIGH/MEDIUM/LOW 임계값을 고른다.

재생이 정확한 지점과 근사한 지점
--------------------------------
정확: 증거 카드. ``trace.evidence``는 러너가 ``prompt_evidence``를 그대로 덤프한 것이고
      (``eval/runner.py`` ``_extract_prompt_evidence``), 12,781장 전부 ``Evidence``로
      되살아난다. 실행 시점에 신뢰도 노드가 본 카드 집합과 같다.
정확: 질문 문자열.
근사: 엔티티. ``trace.l1_entity_linking``은 ``hybrid_ctx.entities``에서 brands·categories·
      indicators·products·concepts만 옮겨 담는다. 실행 시점 ``context.entities``에
      sentiments·time_range 같은 다른 키가 더 있었다면 여기서는 볼 수 없다. 다만 적합도
      점수가 쓰는 키는 brands·categories·products·indicators뿐이고 이 넷은 모두 트레이스에
      있으므로, 이 근사가 점수를 바꾸지는 않는다.
없음: ``evidence_all_count``(선별 전 전체 카드 수)는 남아 있지만 전체 카드 **목록**은
      트레이스에 없다. 그래서 "선별 전 카드로 점수를 매기면 어땠을까"는 계산할 수 없다.

과적합 방지
-----------
문항을 id의 sha1로 반씩 나눈다 — ``int(sha1(item_id).hexdigest(), 16) % 2 == 0``이면
보정(calibration), 아니면 검증(validation). 임계값은 **보정 절반에서만** 고르고, 검증
절반 수치는 따로 보고한다. 가중치와 조합 구조는 보정하지 않는다 — 설계에서 정한다.

보정 목표
---------
목표는 문항의 ``passed``(평가 게이트 전체 통과)다. 부차 지표로 답변 수준 결과
``answer_ok``(L5 게이트를 하나도 어기지 않음)를 같이 싣는다. HIGH는 "LLM 판단을
건너뛰고 증거로 바로 답한다"는 뜻이므로, 그 판단이 틀린 비율(=HIGH-but-failed)이
줄어드는지가 핵심이다.

임계값 선택 규칙 (보정 절반, 격자 0.00~1.00 step 0.01)
    THRESHOLD_HIGH   HIGH 묶음이 보정 절반의 25% 이상을 덮는다는 조건 아래
                     HIGH-but-failed(passed) 비율이 최소인 t.
                     25% 하한은 "HIGH가 실용적으로 남아 있어야 한다"는 설계 조건이다.
                     격자 상한은 ``MAX_HIGH_THRESHOLD``(0.95) — 이 위로 가면 (a)(b)가
                     완벽해도 (c) 때문에 HIGH가 막힌다.
                     최소 실패율과 ``FAIL_TOLERANCE``(0.02) 안쪽인 t들은 결과 데이터가
                     구별 못 한 것으로 보고 그중 가장 엄격한(큰) t를 쓴다.
    THRESHOLD_LOW    t 미만 묶음이 문항 5개 이상이면서 passed 실패율 0.95 이상인
                     가장 큰 t. 없으면 0.01 (점수 0 = 증거 없음만 UNKNOWN).
    THRESHOLD_MEDIUM [LOW, HIGH) 구간 문항 점수의 중앙값(격자에 맞춤). LOW와 겹치면
                     격자 한 칸 위로 올린다 — 겹치면 LOW 구간이 비어 버린다.
                     MEDIUM과 LOW는 둘 다 LLM 분기로 가므로 이 경계는 라우팅을 바꾸지
                     않는다 — DecisionMaker 프롬프트에 실리는 레벨 문자열만 달라진다.

버린 규칙 (기록)
    · J(t) = n_high × (2 × answer_ok_rate(≥t) − 1) 최대화 — answer_ok 비율이 t 전 구간에서
      평평해 J는 HIGH 묶음을 키우기만 한다. 목표 문장을 담지 못해 버렸다.
    · 동률일 때 더 넓은 t 고르기 — 곡선이 평평하면 임계값이 바닥까지 끌려 내려가 MEDIUM·LOW
      구간이 사라진다. 구별 못 하는 구간에서는 더 엄격한 쪽이 안전해서 뒤집었다.

사용법:
    python3 scripts/calibrate_confidence_thresholds.py REPORT.json [REPORT.json ...]
    python3 scripts/calibrate_confidence_thresholds.py REPORT.json -o NOTE.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.confidence import (  # noqa: E402
    FIT_WEIGHTS,
    MAX_HIGH_THRESHOLD,
    RETRIEVAL_SWING,
    ConfidenceAssessor,
    score_evidence_fit,
)
from src.domain.entities.evidence import Evidence  # noqa: E402

GRID = [round(i / 100, 2) for i in range(101)]
MIN_HIGH_COVERAGE = 0.25  # HIGH가 실용적으로 남아 있어야 한다는 설계 하한
FAIL_TOLERANCE = 0.02  # 실패율 차이가 이 안쪽이면 결과 데이터가 구별 못 한 것으로 본다
MIN_UNKNOWN_BUCKET = 5
UNKNOWN_FAIL_RATE = 0.95

_ENTITY_KEY_MAP = {
    "brands": "extracted_brands",
    "categories": "extracted_categories",
    "products": "extracted_products",
    "indicators": "extracted_indicators",
    "concepts": "extracted_concepts",
}


@dataclass(frozen=True)
class Replayed:
    """한 문항의 재생 결과."""

    item_id: str
    question: str
    score: float
    entity_coverage: float
    kind_fit: float
    retrieval_fit: float
    needs: tuple[str, ...]
    card_count: int
    passed: bool
    answer_ok: bool
    half: str
    old_level: str | None
    old_route: str | None
    rag_doc_ids: tuple[str, ...]


def split_half(item_id: str) -> str:
    """id의 sha1 최하위 비트로 보정/검증 절반을 정한다 (결정적, 실행과 무관)."""
    digest = int(hashlib.sha1(item_id.encode("utf-8")).hexdigest(), 16)
    return "calibration" if digest % 2 == 0 else "validation"


def replay_report(path: Path) -> list[Replayed]:
    """report.json 한 개를 실제 채점 함수로 다시 매긴다."""
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: list[Replayed] = []
    for item in report["items"]:
        trace = item.get("trace") or {}
        linking = trace.get("l1_entity_linking") or {}
        entities = {key: (linking.get(src) or []) for key, src in _ENTITY_KEY_MAP.items()}
        cards = [Evidence.model_validate(card) for card in (trace.get("evidence") or [])]

        fit = score_evidence_fit(item["question"], entities, cards)
        tags = item.get("fail_reason_tags") or []
        route_trace = trace.get("route_trace") or {}

        rows.append(
            Replayed(
                item_id=item["item_id"],
                question=item["question"],
                score=fit.score,
                entity_coverage=fit.entity_coverage,
                kind_fit=fit.kind_fit,
                retrieval_fit=fit.retrieval_fit,
                needs=fit.needs,
                card_count=fit.card_count,
                passed=bool(item.get("passed")),
                answer_ok=not any(tag.startswith("L5_") for tag in tags),
                half=split_half(item["item_id"]),
                old_level=route_trace.get("confidence_level"),
                old_route=route_trace.get("route"),
                rag_doc_ids=tuple((trace.get("l2_doc_retrieval") or {}).get("chunk_ids") or ()),
            )
        )
    return rows


def project_routes(rows: list[Replayed], thresholds: tuple[float, float, float]) -> dict[str, Any]:
    """새 임계값으로 각 문항이 어느 경로로 갔을지 센다.

    ``QueryGraph``의 실제 분기 규칙을 그대로 쓴다: HIGH → direct, UNKNOWN → clarify,
    나머지는 ReAct 에이전트가 붙어 있고 복잡한 질문이면 react, 아니면 decide.
    ``agents.use_react_agent`` 플래그가 기본 OFF이므로 두 경우를 모두 센다.

    주의: 이것은 **투영**이다. 실제로 경로가 갈리면 그 문항의 증거와 답변 자체가
    달라지므로, 성적까지 이 표로 예측할 수는 없다.
    """
    high, medium, low = thresholds
    try:
        from src.core.models import Context
        from src.core.query_graph import QueryGraph

        is_complex = QueryGraph._is_complex_query
    except (ImportError, AttributeError):  # pragma: no cover - 분기 헬퍼가 바뀐 경우
        return {}

    react_off: dict[str, int] = {}
    react_on: dict[str, int] = {}
    for row in rows:
        level = level_of(row.score, high, medium, low)
        if level == "HIGH":
            route_off = route_on = "direct"
        elif level == "UNKNOWN":
            route_off = route_on = "clarify"
        else:
            route_off = "decide"
            context = Context(
                query=row.question, rag_docs=[{"id": doc_id} for doc_id in row.rag_doc_ids]
            )
            route_on = "react" if is_complex(row.question, context) else "decide"
        react_off[route_off] = react_off.get(route_off, 0) + 1
        react_on[route_on] = react_on.get(route_on, 0) + 1
    return {"react_off": react_off, "react_on": react_on}


def _rate(rows: list[Replayed], attribute: str) -> float:
    return (sum(getattr(row, attribute) for row in rows) / len(rows)) if rows else 0.0


def _fmt_routes(counts: dict[str, int], total: int) -> str:
    return ", ".join(
        f"{name} {count} ({count / total:.1%})"
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )


def choose_thresholds(rows: list[Replayed]) -> tuple[float, float, float]:
    """보정 절반에서 HIGH/MEDIUM/LOW 임계값을 고른다 (모듈 docstring의 규칙)."""
    # 격자 상한은 MAX_HIGH_THRESHOLD다. 그 위로 올리면 (a)(b)가 완벽해도 검색 점수
    # 모양 때문에 HIGH가 막히는 첫 판의 결함이 되살아난다.
    candidates: list[tuple[float, float]] = []
    for t in GRID:
        if t > MAX_HIGH_THRESHOLD:
            break
        high = [row for row in rows if row.score >= t]
        if len(high) / len(rows) < MIN_HIGH_COVERAGE:
            continue
        candidates.append((t, 1 - _rate(high, "passed")))

    # 실패율이 최소인 t를 고르되, 최소값과 FAIL_TOLERANCE 안쪽인 t들은 결과 데이터가
    # 구별하지 못한 것으로 보고 그중 **가장 엄격한(큰)** t를 쓴다. 이 데이터에서는
    # 곡선이 실제로 평평해서(0.735~0.752, n=120에서 문항 2개 차이) 상한 0.95가 뽑힌다.
    # 0.95 = "(a)와 (b)가 둘 다 완전히 충족" 지점이므로 설계 정의와도 맞는다.
    best_fail = min(fail for _, fail in candidates)
    threshold_high = max(t for t, fail in candidates if fail <= best_fail + FAIL_TOLERANCE)

    threshold_low = 0.01
    for t in GRID:
        if t > threshold_high:
            break
        below = [row for row in rows if row.score < t]
        if len(below) >= MIN_UNKNOWN_BUCKET and (1 - _rate(below, "passed")) >= UNKNOWN_FAIL_RATE:
            threshold_low = t

    middle = [row.score for row in rows if threshold_low <= row.score < threshold_high]
    if middle:
        threshold_medium = round(statistics.median(middle), 2)
    else:
        threshold_medium = round((threshold_low + threshold_high) / 2, 2)
    # 중앙값이 LOW와 겹치면 격자 한 칸 위로 올린다 — 겹치면 LOW 구간이 비어 버린다.
    # MEDIUM과 LOW는 둘 다 LLM 분기라 이 조정은 라우팅을 바꾸지 않는다.
    threshold_medium = min(max(threshold_medium, round(threshold_low + 0.01, 2)), threshold_high)

    return threshold_high, threshold_medium, threshold_low


def level_of(score: float, high: float, medium: float, low: float) -> str:
    if score >= high:
        return "HIGH"
    if score >= medium:
        return "MEDIUM"
    if score >= low:
        return "LOW"
    return "UNKNOWN"


def bucket_table(rows: list[Replayed], high: float, medium: float, low: float) -> dict[str, Any]:
    """레벨별 문항 수와 실패율."""
    out: dict[str, Any] = {}
    for level in ("HIGH", "MEDIUM", "LOW", "UNKNOWN"):
        bucket = [row for row in rows if level_of(row.score, high, medium, low) == level]
        out[level] = {
            "n": len(bucket),
            "share": (len(bucket) / len(rows)) if rows else 0.0,
            "failed_rate_passed": 1 - _rate(bucket, "passed") if bucket else 0.0,
            "failed_rate_answer": 1 - _rate(bucket, "answer_ok") if bucket else 0.0,
        }
    return out


def _fmt_buckets(title: str, rows: list[Replayed], thresholds: tuple[float, float, float]) -> str:
    high, medium, low = thresholds
    table = bucket_table(rows, high, medium, low)
    lines = [
        f"**{title}** (n={len(rows)}, "
        f"passed={_rate(rows, 'passed'):.3f}, answer_ok={_rate(rows, 'answer_ok'):.3f})",
        "",
        "| 레벨 | 문항 수 | 비중 | HIGH-but-failed (passed) | HIGH-but-failed (answer) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for level in ("HIGH", "MEDIUM", "LOW", "UNKNOWN"):
        row = table[level]
        lines.append(
            f"| {level} | {row['n']} | {row['share']:.1%} | "
            f"{row['failed_rate_passed']:.3f} | {row['failed_rate_answer']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_note(reports: dict[str, list[Replayed]], thresholds: tuple[float, float, float]) -> str:
    high, medium, low = thresholds
    parts = [
        "# 5-B 신뢰도 임계값 보정",
        "",
        "`scripts/calibrate_confidence_thresholds.py`가 만든 표. 저장된 증거 카드와 엔티티",
        "링킹 결과를 실제 `score_evidence_fit()`에 다시 먹여 재생했다 (LLM·검색 호출 없음).",
        "",
        "## 점수식",
        "",
        "```",
        "base  = " + " + ".join(f"{weight:.2f}×{name}" for name, weight in FIT_WEIGHTS.items()),
        f"score = clamp(base + {RETRIEVAL_SWING:.2f} × (retrieval_fit − 0.5), 0, 1)",
        "```",
        "",
        "- `entity_coverage` — 질문이 이름을 부른 브랜드·카테고리·제품 중, 증거 카드의",
        "  주어·목적어(또는 metadata의 brand·categories·related_entities 표기)로 등장하는 비율.",
        "  질문이 엔티티를 부르지 않으면 1.0.",
        "- `kind_fit` — 질문이 요구하는 카드 종류 중 실제로 있는 비율. 이름 붙은 엔티티가",
        "  있으면 그 엔티티에 닻을 내린 카드만 요구를 채운 것으로 센다.",
        "- `retrieval_fit` — 문서 카드 검색 점수의 `(top − median) / top`. 점수 2개 미만이면 0.5.",
        "",
        "## 보정된 임계값",
        "",
        f"- `THRESHOLD_HIGH = {high}`",
        f"- `THRESHOLD_MEDIUM = {medium}`",
        f"- `THRESHOLD_LOW = {low}` (미만은 UNKNOWN)",
        "",
        "분할: `int(sha1(item_id).hexdigest(), 16) % 2 == 0` → 보정 절반, 나머지 → 검증 절반.",
        "임계값은 보정 절반에서만 골랐다. 가중치는 보정 대상이 아니다(설계에서 고정).",
        "",
        "## 읽을 때 주의",
        "",
        "1. **믿을 만한 이득은 아래쪽 꼬리(UNKNOWN)뿐이다.** HIGH 묶음의 실패율은 임계값을",
        "   어디에 두든 0.73~0.75로 평평하다 — 전체 평균과 같다. 233문항 중 184문항이",
        "   entity_coverage·kind_fit 둘 다 1.0이기 때문이다(검색이 질의마다 카드 ~70장을",
        "   비슷하게 담아 온다). 즉 이 시험지에서는 요건 신호가 거의 변별하지 못한다.",
        "   반면 UNKNOWN 묶음은 세 실행 모두 passed 0/n이다.",
        "   (1차 보정의 HIGH 실패율 0.63은 retrieval_fit 모양으로 자른 결과였고, 그 식은",
        "    (a)(b)가 완벽해도 HIGH가 불가능한 결함이 있어 폐기했다.)",
        "2. **`passed`의 상당 부분은 런타임 신호로 볼 수 없다.** s3-run1 실패 태그 상위는",
        "   L3_edge_fail 92, L1_concept_fail 86으로, 골드 주석과의 일치를 보는 게이트다.",
        "   적합도 점수로 예측할 수 있는 종류가 아니다. 그래서 `answer_ok`(L5 게이트만)를",
        "   부차 지표로 같이 싣는다.",
        "3. **재생 입력의 근사.** 엔티티는 `trace.l1_entity_linking`에서 읽는다 —",
        "   brands·categories·products·indicators는 그대로 있고 점수가 쓰는 키도 이 넷뿐이라",
        "   점수는 바뀌지 않지만, 실행 시점 `context.entities`에 다른 키가 더 있었는지는",
        "   확인할 수 없다. 증거 카드는 근사가 아니다(러너가 `prompt_evidence`를 그대로 덤프).",
        "   선별 **전** 전체 카드 목록은 트레이스에 없어서(`evidence_all_count` 수치만 있음)",
        "   '선별 전 카드로 매겼다면' 같은 계산은 불가능하다.",
        "4. **경로 분포는 재생값이다.** 실제 실행에서는 HIGH가 아닌 문항이 DecisionMaker·",
        "   ReAct·명확화로 가면서 증거·답변 자체가 달라진다. 이 표의 실패율은 '그때 그",
        "   답변'에 대한 것이지, 새 경로를 탄 뒤의 성적이 아니다. 실제 효과는 재실행으로만",
        "   확인할 수 있다.",
        "",
    ]
    for name, rows in reports.items():
        parts.append(f"## {name}")
        parts.append("")
        parts.append(_fmt_buckets("전체", rows, thresholds))
        parts.append(
            _fmt_buckets(
                "보정 절반", [row for row in rows if row.half == "calibration"], thresholds
            )
        )
        parts.append(
            _fmt_buckets("검증 절반", [row for row in rows if row.half == "validation"], thresholds)
        )
        old_high = [row for row in rows if row.old_level == "high"]
        parts.append(
            f"변경 전(개수 합 점수): HIGH {len(old_high)}/{len(rows)}문항, "
            f"그중 passed 실패율 {1 - _rate(old_high, 'passed'):.3f}, "
            f"answer 실패율 {1 - _rate(old_high, 'answer_ok'):.3f}\n"
        )

        routes = project_routes(rows, thresholds)
        if routes:
            old_routes: dict[str, int] = {}
            for row in rows:
                key = row.old_route or "(없음)"
                old_routes[key] = old_routes.get(key, 0) + 1
            total = len(rows)
            parts.append("경로 투영 (실제 분기 규칙 적용, 성적 예측 아님 — 주의 4 참조)\n")
            parts.append(f"- 변경 전(실측): {_fmt_routes(old_routes, total)}")
            parts.append(
                f"- 변경 후, ReAct OFF(기본 플래그): {_fmt_routes(routes['react_off'], total)}"
            )
            parts.append(f"- 변경 후, ReAct ON: {_fmt_routes(routes['react_on'], total)}\n")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="평가 report.json 경로")
    parser.add_argument("-o", "--output", type=Path, help="마크다운 표를 쓸 경로")
    parser.add_argument(
        "--apply-from",
        type=Path,
        help="임계값을 고를 기준 report.json (생략하면 첫 번째 report)",
    )
    args = parser.parse_args(argv)

    replayed = {path.parent.name or path.stem: replay_report(path) for path in args.reports}

    basis_path = args.apply_from or args.reports[0]
    basis = replayed[basis_path.parent.name or basis_path.stem]
    calibration = [row for row in basis if row.half == "calibration"]
    thresholds = choose_thresholds(calibration)

    note = build_note(replayed, thresholds)
    print(note)

    high, medium, low = thresholds
    current = (
        ConfidenceAssessor.THRESHOLD_HIGH,
        ConfidenceAssessor.THRESHOLD_MEDIUM,
        ConfidenceAssessor.THRESHOLD_LOW,
    )
    print(f"\n[보정] 고른 값 = {thresholds}, confidence.py 현재 값 = {current}")
    if thresholds != current:
        print("[보정] 두 값이 다르다 — confidence.py의 상수를 갱신할 것.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(note, encoding="utf-8")
        print(f"[보정] 표를 {args.output}에 썼다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
