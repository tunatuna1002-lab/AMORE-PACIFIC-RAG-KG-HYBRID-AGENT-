"""
Evaluation CLI
==============
Command-line interface for running evaluations, comparing against baselines,
and managing baseline snapshots.

Usage:
    # Run evaluation (default subcommand)
    python -m eval run --dataset path/to/dataset.jsonl --out ./eval_output

    # Backward compatible (no subcommand = run)
    python -m eval --dataset path/to/dataset.jsonl --out ./eval_output

    # With NLI judge (free, local)
    python -m eval run --dataset data.jsonl --judge nli

    # With LLM judge (paid, more accurate)
    python -m eval run --dataset data.jsonl --judge llm --judge-model gpt-4.1-mini

    # Run and compare against baseline
    python -m eval run --dataset data.jsonl --baseline v1.0

    # Compare a report against a baseline
    python -m eval compare --baseline-name v1.0 --report reports/report.json

    # Save a baseline
    python -m eval set-baseline --name v1.0 --report reports/report.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from eval.loader import load_dataset
from eval.report import ReportGenerator
from eval.runner import EvalRunner
from eval.schemas import EvalConfig, EvalReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by the 'run' subcommand."""
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

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline name to compare against after evaluation completes",
    )

    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="eval/baselines",
        help="Directory for baselines (default: eval/baselines)",
    )


def _needs_run_prefix(argv: list[str]) -> bool:
    """Check if argv lacks a subcommand and should default to 'run'."""
    known_commands = {"run", "compare", "set-baseline"}
    if not argv:
        return False
    return argv[0] not in known_commands and argv[0].startswith("-")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments with subparsers."""
    raw = argv if argv is not None else sys.argv[1:]

    # Backward compatibility: if first arg is a flag (e.g. --dataset),
    # prepend 'run' so the old `python -m eval --dataset ...` still works.
    if _needs_run_prefix(raw):
        raw = ["run"] + list(raw)

    parser = argparse.ArgumentParser(
        description="Evaluation harness for AMORE RAG+KG Hybrid Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python -m eval run --dataset eval/data/examples/chatbot_eval.jsonl

    # Evaluate and compare against baseline
    python -m eval run --dataset data.jsonl --baseline v1.0

    # Compare existing report against baseline
    python -m eval compare --baseline-name v1.0 --report eval_output/report.json

    # Save current report as baseline
    python -m eval set-baseline --name v1.0 --report eval_output/report.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(run_parser)

    # --- compare subcommand ---
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare a report against a baseline (exits 1 on regression)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_parser.add_argument(
        "--baseline-dir",
        type=str,
        default="eval/baselines",
        help="Directory for baselines (default: eval/baselines)",
    )
    compare_parser.add_argument(
        "--baseline-name",
        type=str,
        default=None,
        help="Baseline name to compare against (default: latest)",
    )
    compare_parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to evaluation report JSON to compare",
    )
    compare_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    # --- set-baseline subcommand ---
    baseline_parser = subparsers.add_parser(
        "set-baseline",
        help="Save a report as a named baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    baseline_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Baseline name (e.g., v1.0, main-2026-02-18)",
    )
    baseline_parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to evaluation report JSON to save as baseline",
    )
    baseline_parser.add_argument(
        "--baseline-dir",
        type=str,
        default="eval/baselines",
        help="Directory for baselines (default: eval/baselines)",
    )
    baseline_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args(raw)

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    return args


# =============================================================================
# Run Evaluation
# =============================================================================


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
    baseline: str | None = None,
    baseline_dir: str = "eval/baselines",
) -> int:
    """
    Run the full evaluation pipeline.

    Returns:
        Exit code (0 for success, 1 for failure, 2 for regressions detected)
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

    # Post-eval baseline comparison
    if baseline:
        exit_code = _run_comparison(report, baseline_dir, baseline)
        if exit_code != 0:
            return exit_code

    return 0


# =============================================================================
# Compare Command
# =============================================================================


def cmd_compare(args: argparse.Namespace) -> int:
    """Run the compare subcommand."""
    report_path = Path(args.report)
    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return 1

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    report = EvalReport.model_validate(data)

    return _run_comparison(report, args.baseline_dir, args.baseline_name)


def _run_comparison(
    report: EvalReport,
    baseline_dir: str,
    baseline_name: str | None,
) -> int:
    """
    Compare a report against a baseline and print results.

    Returns:
        0 if no regressions, 1 if regressions detected.
    """
    from eval.regression import RegressionTester

    tester = RegressionTester(baseline_dir=baseline_dir)

    try:
        if baseline_name:
            comparison = tester.compare(baseline_name, report)
        else:
            comparison = tester.compare_with_latest(report)
    except FileNotFoundError as e:
        logger.error(f"Baseline not found: {e}")
        return 1

    if comparison is None:
        logger.info("No baselines found, skipping regression check")
        return 0

    # Print regression summary
    _print_regression_summary(comparison)

    if comparison.has_regressions:
        logger.error(
            f"REGRESSIONS DETECTED: {comparison.regression_count} metric(s) regressed, "
            f"{len(comparison.new_failures)} new failure(s)"
        )
        return 1

    logger.info("No regressions detected")
    return 0


def _print_regression_summary(comparison) -> None:
    """Print regression comparison summary to console."""
    print("\n" + "=" * 60)
    print("REGRESSION COMPARISON")
    print("=" * 60)
    print(f"Baseline:       {comparison.baseline_name}")
    print(f"Regressions:    {comparison.regression_count}")
    print(f"New Failures:   {len(comparison.new_failures)}")
    print(f"Fixed Items:    {len(comparison.fixed_items)}")
    print()

    regressed = [r for r in comparison.metric_results if r.is_regression]
    if regressed:
        print("Regressed Metrics:")
        for r in regressed:
            print(
                f"  - {r.metric_name}: {r.baseline_value:.4f} -> "
                f"{r.current_value:.4f} ({r.delta:+.4f}) [{r.severity}]"
            )
        print()

    if comparison.new_failures:
        print("New Failures:")
        for item_id in comparison.new_failures[:10]:
            print(f"  - {item_id}")
        if len(comparison.new_failures) > 10:
            print(f"  ... and {len(comparison.new_failures) - 10} more")
        print()

    if comparison.fixed_items:
        print(f"Fixed: {len(comparison.fixed_items)} item(s) now pass")
        print()

    print("=" * 60 + "\n")


# =============================================================================
# Set-Baseline Command
# =============================================================================


def cmd_set_baseline(args: argparse.Namespace) -> int:
    """Run the set-baseline subcommand."""
    report_path = Path(args.report)
    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return 1

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    report = EvalReport.model_validate(data)

    from eval.regression import RegressionTester

    tester = RegressionTester(baseline_dir=args.baseline_dir)
    saved_path = tester.save_baseline(args.name, report)

    print(f"Baseline '{args.name}' saved to {saved_path}")
    return 0


# =============================================================================
# Helpers
# =============================================================================


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


# =============================================================================
# Main
# =============================================================================


def main(argv: list[str] | None = None):
    """Main entry point."""
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "run":
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
                baseline=args.baseline,
                baseline_dir=args.baseline_dir,
            )
        )

    elif args.command == "compare":
        exit_code = cmd_compare(args)

    elif args.command == "set-baseline":
        exit_code = cmd_set_baseline(args)

    else:
        logger.error(f"Unknown command: {args.command}")
        exit_code = 2

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
