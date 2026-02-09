#!/usr/bin/env python3
"""
LANEIGE 골든셋 평가 실행 스크립트
===================================
AMORE RAG-KG Hybrid Agent의 성능을 평가합니다.

사용법:
    # 기본 평가 (NLI Judge, Semantic Similarity)
    python scripts/run_evaluation.py

    # LLM Judge 사용 (유료)
    python scripts/run_evaluation.py --judge llm

    # 상세 로그
    python scripts/run_evaluation.py --verbose
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval import generate_json_report, generate_markdown_summary, load_dataset
from eval.runner import EvalRunner
from eval.schemas import EvalConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LANEIGE 골든셋 평가 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/data/golden/laneige_golden_v1.jsonl",
        help="평가 데이터셋 경로 (기본: laneige_golden_v1.jsonl)",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="eval_output",
        help="출력 디렉토리 (기본: eval_output)",
    )

    parser.add_argument(
        "--judge",
        type=str,
        choices=["none", "nli", "llm"],
        default="nli",
        help="Judge 유형: none (없음), nli (무료, 로컬), llm (유료, OpenAI)",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM Judge 모델 (기본: gpt-4.1-mini)",
    )

    parser.add_argument(
        "--no-semantic",
        action="store_true",
        default=False,
        help="Semantic Similarity 비활성화",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="검색 top-k (기본: 8)",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="동시 평가 수 (기본: 1)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="상세 로그 활성화",
    )

    return parser.parse_args()


async def create_agent():
    """Create and initialize the HybridChatbotAgent."""
    try:
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()
        logger.info("HybridChatbotAgent 초기화 완료")
        return agent
    except ImportError as e:
        logger.error(f"HybridChatbotAgent 임포트 실패: {e}")
        logger.info("MockAgent를 사용합니다 (실제 평가에서는 agent 구현 필요)")
        return MockAgent()


class MockAgent:
    """Mock agent for testing when real agent is not available."""

    async def chat(self, question: str) -> dict:
        """Return mock response."""
        return {
            "answer": f"[Mock 응답] {question}에 대한 답변입니다.",
            "confidence": 0.5,
            "sources": [],
        }

    def get_last_hybrid_context(self):
        """Return mock context."""
        return {
            "extracted_entities": [],
            "retrieved_docs": [],
            "kg_results": [],
            "ontology_inferences": [],
        }


async def create_judge(judge_type: str, model: str):
    """Create judge based on type."""
    if judge_type == "none":
        from eval.judge.stub import StubJudge

        logger.info("StubJudge 사용 (점수 없음)")
        return StubJudge()
    elif judge_type == "nli":
        try:
            from eval.judge.nli import NLIJudge

            logger.info("NLIJudge 초기화 중 (무료, 로컬)...")
            return NLIJudge()
        except ImportError as e:
            logger.warning(f"NLIJudge 사용 불가: {e}")
            from eval.judge.stub import StubJudge

            return StubJudge()
    elif judge_type == "llm":
        try:
            from eval.judge.llm import LLMJudge

            logger.info(f"LLMJudge 초기화 중 (모델: {model})...")
            return LLMJudge(model=model)
        except ImportError as e:
            logger.warning(f"LLMJudge 사용 불가: {e}")
            from eval.judge.stub import StubJudge

            return StubJudge()
    else:
        from eval.judge.stub import StubJudge

        return StubJudge()


async def run_evaluation(args: argparse.Namespace) -> int:
    """Run the evaluation pipeline."""
    start_time = datetime.now()

    # 1. Load dataset
    logger.info(f"데이터셋 로드: {args.dataset}")
    try:
        dataset_path = project_root / args.dataset
        items = load_dataset(dataset_path)
        logger.info(f"  {len(items)}개 평가 항목 로드됨")
    except FileNotFoundError:
        logger.error(f"데이터셋 파일을 찾을 수 없습니다: {args.dataset}")
        return 1
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        return 1

    # 2. Initialize agent
    logger.info("Agent 초기화...")
    agent = await create_agent()

    # 3. Initialize judge
    judge = await create_judge(args.judge, args.judge_model)

    # 4. Create config
    use_semantic = not args.no_semantic
    config = EvalConfig(
        top_k=args.top_k,
        use_judge=(args.judge != "none"),
        judge_model=args.judge_model if args.judge == "llm" else None,
    )

    # 5. Initialize runner
    logger.info("평가 실행...")
    logger.info(f"  Judge: {args.judge}")
    logger.info(f"  Semantic Similarity: {use_semantic}")
    logger.info(f"  Top-k: {args.top_k}")

    runner = EvalRunner(
        agent=agent,
        config=config,
        judge=judge,
        use_semantic_similarity=use_semantic,
    )

    # 6. Run evaluation
    try:
        results = await runner.run_dataset(items, concurrency=args.concurrency)
    except Exception as e:
        logger.error(f"평가 실행 실패: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # 7. Generate reports
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"리포트 생성: {out_dir}")
    generate_json_report(results, out_dir)
    generate_markdown_summary(results, out_dir)

    # 8. Print summary
    elapsed = (datetime.now() - start_time).total_seconds()
    _print_summary(results, elapsed)

    return 0


def _print_summary(results, elapsed_seconds: float) -> None:
    """Print evaluation summary."""
    agg = results.aggregates

    print("\n" + "=" * 70)
    print("                        평가 결과 요약")
    print("=" * 70)
    print(f"  총 항목:      {agg.total}")
    print(f"  통과:         {agg.passed} ({agg.pass_rate:.1%})")
    print(f"  실패:         {agg.failed}")
    print(f"  평균 점수:    {agg.avg_overall_score:.3f}")
    print(f"  평균 지연:    {agg.avg_latency_ms:.0f}ms")
    print(f"  총 실행 시간: {elapsed_seconds:.1f}초")
    print()

    if agg.by_layer:
        print("레이어별 점수:")
        for layer, score in agg.by_layer.items():
            print(f"  - {layer}: {score:.3f}")
        print()

    if agg.by_difficulty:
        print("난이도별 통과율:")
        for diff, metrics in agg.by_difficulty.items():
            pass_rate = metrics.get("pass_rate", 0)
            print(f"  - {diff}: {pass_rate:.1%}")
        print()

    if agg.top_fail_reasons:
        print("주요 실패 원인:")
        for tag, count in list(agg.top_fail_reasons.items())[:5]:
            print(f"  - {tag}: {count}개")

    print("=" * 70)
    print()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 70)
    print("          AMORE RAG-KG Hybrid Agent 평가")
    print("          LANEIGE 골든셋 v1.0")
    print("=" * 70 + "\n")

    exit_code = asyncio.run(run_evaluation(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
