"""
골든셋 오프라인 회귀 게이트
==========================
eval/baselines/replay/<dataset>.jsonl 기록이 있으면 ReplayAgent + StubJudge로 L1~L4를 재계산해
저장된 baseline 대비 회귀가 없는지 검사한다. 기록이 없으면 skip (기록은 scripts/record_golden_replay.py).
"""

import json
from pathlib import Path

import pytest

from eval.loader import load_dataset
from eval.replay import ReplayAgent, ReplayHybridContext
from eval.runner import EvalRunner

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval" / "data" / "golden" / "subset_nokg.jsonl"
RECORDING = ROOT / "eval" / "baselines" / "replay" / "subset_nokg.jsonl"
BASELINE = ROOT / "eval" / "baselines" / "replay" / "subset_nokg.baseline.json"


def test_replay_agent_roundtrip(tmp_path):
    """기록 형식 → ReplayAgent 복원 계약."""
    rec = tmp_path / "r.jsonl"
    rec.write_text(
        json.dumps(
            {
                "question": "q1",
                "result": {
                    "response": "a",
                    "hybrid_context": {
                        "entities": {"brands": ["LANEIGE"]},
                        "inferences": [],
                        "rag_chunks": [],
                    },
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    agent = ReplayAgent(rec)
    assert len(agent) == 1
    import asyncio

    result = asyncio.run(agent.chat("q1"))
    assert result["response"] == "a"
    assert isinstance(result["hybrid_context"], ReplayHybridContext)
    assert result["hybrid_context"].entities == {"brands": ["LANEIGE"]}


@pytest.mark.skipif(
    not RECORDING.exists(), reason="replay recording missing (run scripts/record_golden_replay.py)"
)
@pytest.mark.asyncio
async def test_golden_replay_no_regression():
    items = load_dataset(DATASET)
    runner = EvalRunner(agent=ReplayAgent(RECORDING))
    results = await runner.run_dataset(items)
    summary = (
        runner.aggregator.summarize(results) if hasattr(runner.aggregator, "summarize") else None
    )
    scores = {
        "l1": sum(r.l1.overall for r in results if hasattr(r.l1, "overall")) / max(len(results), 1),
        "l3": sum(r.l3.overall for r in results if hasattr(r.l3, "overall")) / max(len(results), 1),
    }
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        pytest.skip(f"baseline created: {BASELINE}")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for k, v in baseline.items():
        assert scores.get(k, 0.0) >= v - 0.01, (
            f"{k} regressed: {scores.get(k)} < {v} (summary={summary})"
        )
