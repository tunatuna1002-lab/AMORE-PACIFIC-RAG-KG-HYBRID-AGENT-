"""
골든셋 재생 기록 생성
=====================
실제 HybridChatbotAgent로 골든셋 질문을 1회 실행해 eval/baselines/replay/<name>.jsonl 에 기록한다.
이후 tests/eval/test_golden_replay_gate.py 가 이 기록을 오프라인으로 재생해 회귀를 검사한다.

Usage:
    OPENAI_API_KEY=... python3 scripts/record_golden_replay.py --dataset eval/data/golden/subset_nokg.jsonl
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.loader import load_dataset  # noqa: E402
from eval.replay import RecordingAgent  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/data/golden/subset_nokg.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = (
        Path(args.output)
        if args.output
        else Path("eval/baselines/replay") / f"{dataset.stem}.jsonl"
    )

    from src.infrastructure.container import Container

    workflow = Container.get_chat_workflow()
    agent = RecordingAgent(workflow, output)
    items = load_dataset(dataset)
    for i, item in enumerate(items, 1):
        await agent.chat(item.question)
        print(f"[{i}/{len(items)}] recorded: {item.question[:50]}")
    agent.close()
    print(f"saved → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
