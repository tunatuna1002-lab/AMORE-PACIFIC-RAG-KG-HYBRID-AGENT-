#!/usr/bin/env python3
"""프롬프트 실험 결과 요약표 생성.

사용법: python scripts/summarize_prompt_experiment.py eval_output/exp_YYYYMMDD_HHMM
각 하위 폴더(v0, v1, ...)의 report.json을 읽어 마크다운 표와 대표 실패 사례를 출력한다.
"""

import json
import sys
from pathlib import Path

VARIANT_DESC = {
    "v0": "baseline (현재 프롬프트)",
    "v1": "근거 인용 강제 [출처: …] + 데이터 없음 명시",
    "v2": "단계적 내부 추론(유형→지표→근거→결론) 후 결론만 출력",
    "v3": "출력 스키마 고정 (결론/근거3/유의사항)",
    "v4": "압축 (지시 중복 제거, 길이 약 40%)",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main(out_dir: Path) -> None:
    rows, failures = [], []
    for v in sorted(p.name for p in out_dir.iterdir() if p.is_dir()):
        rp = out_dir / v / "report.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        agg, items = r.get("aggregates", {}), r.get("items", [])
        g = mean([i.get("l5", {}).get("groundedness_score") for i in items])
        rel = mean([i.get("l5", {}).get("answer_relevance_score") for i in items])
        f1 = mean([i.get("l5", {}).get("answer_f1") for i in items])
        top_fail = (
            ", ".join(f"{k}({n})" for k, n in list(agg.get("top_fail_reasons", {}).items())[:2])
            or "-"
        )
        rows.append(
            f"| {v} | {VARIANT_DESC.get(v, '')} | {agg.get('total', 0)} | {agg.get('pass_rate', 0):.1%} | "
            f"{fmt(g)} | {fmt(rel)} | {fmt(f1)} | {agg.get('avg_latency_ms', 0):.0f} | "
            f"{agg.get('avg_tokens_per_item', 0):.0f} | {agg.get('total_cost_usd', 0):.4f} | {top_fail} |"
        )
        worst = sorted(
            (i for i in items if not i.get("passed")), key=lambda i: i.get("overall_score", 0)
        )[:1]
        for i in worst:
            ans = (
                (i.get("trace") or {}).get("l5_answer", {}).get("final_answer", "")
                if i.get("trace")
                else ""
            )
            failures.append(
                f"- **{v} / {i.get('item_id')}** — Q: {i.get('question')}\n"
                f"  - 실패 태그: {', '.join(i.get('fail_reason_tags', [])) or '-'} / overall {i.get('overall_score', 0):.2f}\n"
                f"  - 응답(앞 300자): {str(ans)[:300]!r}"
            )

    print(f"# 프롬프트 실험 결과 — {out_dir.name}\n")
    print(
        "고정 조건: 동일 데이터셋 / 동일 모델 / top-k 8 / temperature=$LLM_TEMPERATURE / judge gpt-4.1-mini / 각 1회 실행\n"
    )
    print(
        "| 버전 | 변경 | n | pass | Groundedness | Relevance | TokenF1 | 평균지연ms | 토큰/문항 | 비용USD | 실패태그 Top2 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print("\n## 대표 실패 사례 (버전별 최저 점수 1건)\n")
    print("\n".join(failures) if failures else "- 없음")
    print("\n## 결론·채택\n- 채택 버전: \n- 이유: \n- 트레이드오프: \n\n## 다음 실험\n- ")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: summarize_prompt_experiment.py <exp_output_dir>")
    main(Path(sys.argv[1]))
