#!/usr/bin/env python3
"""KG off 근거성 하락이 '답변이 나빠져서'인지 'judge 컨텍스트에서 KG 사실이 빠져서'인지 분리한다.

같은 문항에 대해 답변과 judge 컨텍스트의 출처를 교차한다 (유료: judge 호출).
  A = nokg 답변 × full 컨텍스트
  C = full 답변 × nokg 컨텍스트
원래 점수(full×full, nokg×nokg)는 리포트에 있다.

사용법:
    PYTHONPATH=. .venv/bin/python scripts/judge_context_swap.py FULL_REPORT NOKG_REPORT OUT_JSON
"""

import asyncio
import json
import statistics
import sys
from pathlib import Path


async def main() -> int:
    from eval.judge.llm import LLMJudge
    from eval.runner import EvalRunner
    from eval.schemas import EvalReport

    full_path, nokg_path, out_path = map(Path, sys.argv[1:4])
    full = {i.item_id: i for i in EvalReport.model_validate_json(full_path.read_text()).items}
    nokg = {i.item_id: i for i in EvalReport.model_validate_json(nokg_path.read_text()).items}
    ids = sorted(set(full) & set(nokg))

    usage = {"prompt": 0, "completion": 0}

    def on_usage(p: int, c: int) -> None:
        usage["prompt"] += p
        usage["completion"] += c

    judge = LLMJudge(model="gpt-4.1-mini", on_usage=on_usage)
    build = EvalRunner._build_context_string
    sem = asyncio.Semaphore(8)

    async def score(answer: str, context: str) -> float | None:
        async with sem:
            try:
                return await judge.score_groundedness(answer, context)
            except Exception as e:  # 문항 하나 실패로 전체를 버리지 않는다
                print(f"judge error: {e}", file=sys.stderr)
                return None

    async def one(item_id: str) -> dict:
        f, n = full[item_id], nokg[item_id]
        fa, na = f.trace.l5_answer.final_answer, n.trace.l5_answer.final_answer
        fc, nc = build(None, f.trace), build(None, n.trace)
        a, c = await asyncio.gather(score(na, fc), score(fa, nc))
        return {
            "item_id": item_id,
            "full_x_full": f.l5.groundedness_score,
            "nokg_x_nokg": n.l5.groundedness_score,
            "nokg_ans_x_full_ctx": a,
            "full_ans_x_nokg_ctx": c,
            "full_ctx_chars": len(fc),
            "nokg_ctx_chars": len(nc),
        }

    rows = await asyncio.gather(*(one(i) for i in ids))

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else float("nan")

    summary = {k: round(mean(k), 3) for k in rows[0] if k != "item_id"}
    summary["items"] = len(rows)
    # gpt-4.1-mini: $0.40 / 1M input, $1.60 / 1M output
    summary["judge_cost_usd"] = round(usage["prompt"] * 0.4e-6 + usage["completion"] * 1.6e-6, 4)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
