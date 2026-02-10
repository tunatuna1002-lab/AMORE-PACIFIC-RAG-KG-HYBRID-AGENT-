"""
Evaluation CLI
==============
Command-line interface for running evaluations.

Usage:
    python -m eval.cli --dataset path/to/dataset.jsonl --out ./eval_output

    # With NLI judge (free, local)
    python -m eval.cli --dataset data.jsonl --judge nli

    # With LLM judge (paid, more accurate)
    python -m eval.cli --dataset data.jsonl --judge llm --judge-model gpt-4.1-mini

    # With semantic similarity
    python -m eval.cli --dataset data.jsonl --semantic-similarity
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from eval.loader import load_dataset
from eval.report import ReportGenerator
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
        description="Run evaluation harness for AMORE RAG+KG Hybrid Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation (no judge)
    python -m eval.cli --dataset eval/data/examples/chatbot_eval.jsonl

    # With NLI judge (free, local)
    python -m eval.cli --dataset data.jsonl --judge nli

    # With LLM judge (paid, more accurate)
    python -m eval.cli --dataset data.jsonl --judge llm --judge-model gpt-4.1-mini

    # With semantic similarity enabled
    python -m eval.cli --dataset data.jsonl --semantic-similarity

    # Full evaluation with NLI + semantic similarity
    python -m eval.cli --dataset data.jsonl --judge nli --semantic-similarity

    # Custom output directory and top-k
    python -m eval.cli --dataset data.jsonl --out ./results --top-k 5
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset (JSONL format)",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./eval_output",
        help="Output directory for reports (default: ./eval_output)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k for retrieval metrics (default: 8)",
    )

    parser.add_argument(
        "--judge",
        type=str,
        choices=["none", "nli", "llm"],
        default="none",
        help="Judge type: none (no judge), nli (free, local NLI), llm (paid, OpenAI)",
    )

    parser.add_argument(
        "--use-judge",
        action="store_true",
        default=False,
        help="[Deprecated] Use --judge nli or --judge llm instead",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="Model to use for LLM judge (default: gpt-4.1-mini)",
    )

    parser.add_argument(
        "--semantic-similarity",
        action="store_true",
        default=False,
        help="Enable semantic similarity scoring using sentence-transformers",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent evaluations (default: 1)",
    )

    parser.add_argument(
        "--save-traces",
        action="store_true",
        default=False,
        help="Save individual traces to files",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Load dataset and validate without running evaluation",
    )

    return parser.parse_args()


async def run_evaluation(
    dataset_path: str,
    out_dir: str,
    top_k: int,
    judge_type: str,
    judge_model: str,
    use_semantic_similarity: bool,
    concurrency: int,
    save_traces: bool,
    dry_run: bool,
) -> int:
    """
    Run the full evaluation pipeline.

    Args:
        dataset_path: Path to evaluation dataset
        out_dir: Output directory
        top_k: Top-k for retrieval metrics
        judge_type: Judge type (none, nli, llm)
        judge_model: Model for LLM judge
        use_semantic_similarity: Whether to compute semantic similarity
        concurrency: Concurrent evaluation count
        save_traces: Whether to save individual traces
        dry_run: Whether to skip actual evaluation

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        items = load_dataset(Path(dataset_path))
        logger.info(f"Loaded {len(items)} evaluation items")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    if dry_run:
        logger.info("Dry run mode - skipping evaluation")
        logger.info(f"Dataset validation passed for {len(items)} items")
        for item in items[:5]:
            logger.info(f"  - {item.id}: {item.question[:50]}...")
        return 0

    # Determine if using judge
    use_judge = judge_type in ("nli", "llm")

    # Create config
    config = EvalConfig(
        top_k=top_k,
        use_judge=use_judge,
        judge_model=judge_model if judge_type == "llm" else None,
        save_traces=save_traces,
    )

    # Initialize agent
    logger.info("Initializing agent...")
    try:
        agent = await _create_agent()
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return 1

    # Initialize judge based on type
    judge = None
    if judge_type == "nli":
        logger.info("Initializing NLI judge (free, local)")
        try:
            judge = await _create_nli_judge()
        except Exception as e:
            logger.warning(f"Failed to initialize NLI judge, using stub: {e}")
    elif judge_type == "llm":
        logger.info(f"Initializing LLM judge with model: {judge_model}")
        try:
            judge = await _create_llm_judge(judge_model)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM judge, using stub: {e}")

    # Run evaluation
    logger.info(f"Starting evaluation with concurrency={concurrency}")
    logger.info(f"  Judge: {judge_type}")
    logger.info(f"  Semantic similarity: {use_semantic_similarity}")

    runner = EvalRunner(
        agent=agent,
        config=config,
        judge=judge,
        use_semantic_similarity=use_semantic_similarity,
    )

    try:
        results = await runner.run_dataset(items, concurrency=concurrency)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

    # Generate report
    logger.info(f"Generating reports to {out_dir}")
    report_generator = ReportGenerator(config=config)
    report = report_generator.generate_report(results, Path(out_dir))

    # Print summary
    _print_summary(report)

    return 0


async def _create_agent():
    """Create and initialize the HybridChatbotAgent."""
    try:
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()
        return agent
    except ImportError as e:
        logger.error(f"Could not import HybridChatbotAgent: {e}")
        raise


async def _create_nli_judge():
    """Create NLI judge instance (free, local)."""
    try:
        from eval.judge.nli import NLIJudge

        return NLIJudge()
    except ImportError as e:
        logger.warning(f"NLI judge not available: {e}")
        from eval.judge.stub import StubJudge

        return StubJudge()


async def _create_llm_judge(model: str):
    """Create LLM judge instance (paid, OpenAI)."""
    try:
        from eval.judge.llm import LLMJudge

        return LLMJudge(model=model)
    except ImportError as e:
        logger.warning(f"LLM judge not available: {e}")
        from eval.judge.stub import StubJudge

        return StubJudge()


def _print_summary(report) -> None:
    """Print evaluation summary to console."""
    agg = report.aggregates

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Items:  {agg.total}")
    print(f"Passed:       {agg.passed} ({agg.pass_rate:.1%})")
    print(f"Failed:       {agg.failed}")
    print(f"Avg Score:    {agg.avg_overall_score:.3f}")
    print(f"Avg Latency:  {agg.avg_latency_ms:.0f}ms")
    print()

    if agg.top_fail_reasons:
        print("Top Failure Reasons:")
        for tag, count in list(agg.top_fail_reasons.items())[:5]:
            print(f"  - {tag}: {count} items")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle deprecated --use-judge flag
    judge_type = args.judge
    if args.use_judge and judge_type == "none":
        logger.warning("--use-judge is deprecated. Use --judge llm instead.")
        judge_type = "llm"

    exit_code = asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            out_dir=args.out,
            top_k=args.top_k,
            judge_type=judge_type,
            judge_model=args.judge_model,
            use_semantic_similarity=args.semantic_similarity,
            concurrency=args.concurrency,
            save_traces=args.save_traces,
            dry_run=args.dry_run,
        )
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
