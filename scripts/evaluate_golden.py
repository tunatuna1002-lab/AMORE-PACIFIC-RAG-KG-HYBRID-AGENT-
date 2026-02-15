#!/usr/bin/env python3
"""
Golden Set Evaluation Script for AMORE RAG-KG Hybrid Agent

Usage:
    python scripts/evaluate_golden.py [--chatbot] [--report] [--verbose]

Options:
    --chatbot   Run chatbot golden set evaluation only
    --report    Run report golden set evaluation only
    --verbose   Show detailed results for each test case

Without options, runs both evaluations.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ChatbotTestResult:
    """Result of a single chatbot test case."""

    query: str
    passed: bool
    brands_found: list[str]
    expected_brands: list[str]
    facts_found: list[str]
    expected_facts: list[str]
    source_types_found: list[str]
    expected_source_types: list[str]
    latency_ms: float
    error: str | None = None


@dataclass
class ReportTestResult:
    """Result of a single report test case."""

    date: str
    passed: bool
    sections_found: list[str]
    expected_sections: list[str]
    kpis_found: list[str]
    expected_kpis: list[str]
    brands_mentioned: list[str]
    expected_brands: list[str]
    latency_ms: float
    error: str | None = None


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    total: int
    passed: int
    failed: int
    avg_latency_ms: float
    brand_recall: float
    fact_recall: float
    source_coverage: float


def load_golden_set(file_path: Path) -> list[dict[str, Any]]:
    """Load golden set from JSONL file."""
    cases = []
    if not file_path.exists():
        print(f"Warning: Golden set file not found: {file_path}")
        return cases

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def extract_brands_from_response(response: str) -> list[str]:
    """Extract brand names from chatbot response."""
    known_brands = [
        "LANEIGE",
        "COSRX",
        "TIRTIR",
        "SKIN1004",
        "Anua",
        "MEDICUBE",
        "Beauty of Joseon",
        "Innisfree",
        "MISSHA",
        "ETUDE",
        "Torriden",
        "CeraVe",
        "Neutrogena",
        "The Ordinary",
        "e.l.f.",
        "NYX",
    ]
    found = []
    response_upper = response.upper()
    for brand in known_brands:
        if brand.upper() in response_upper:
            found.append(brand)
    return found


def extract_facts_from_response(response: str) -> list[str]:
    """Extract mentioned facts/metrics from response."""
    fact_keywords = {
        "SoS": ["sos", "점유율", "share of shelf", "시장점유"],
        "HHI": ["hhi", "집중도", "herfindahl"],
        "CPI": ["cpi", "경쟁지수", "competitive position"],
        "rank": ["순위", "rank", "위", "등"],
        "trend": ["트렌드", "trend", "추세", "변동"],
        "competition": ["경쟁", "competition", "대비"],
        "product": ["제품", "product", "상품"],
        "strategy": ["전략", "strategy"],
        "recommendation": ["권고", "추천", "recommendation"],
    }

    found = []
    response_lower = response.lower()
    for fact, keywords in fact_keywords.items():
        if any(kw in response_lower for kw in keywords):
            found.append(fact)
    return found


def extract_source_types(sources: list[dict]) -> list[str]:
    """Extract source types from chatbot sources."""
    types = set()
    for source in sources:
        if isinstance(source, dict):
            source_type = source.get("type", source.get("source_type", ""))
            if source_type:
                types.add(source_type)
    return list(types)


def calculate_recall(found: list[str], expected: list[str]) -> float:
    """Calculate recall metric."""
    if not expected:
        return 1.0
    found_set = {item.upper() for item in found}
    expected_set = {item.upper() for item in expected}
    matches = len(found_set & expected_set)
    return matches / len(expected_set)


async def evaluate_chatbot_case(
    workflow, case: dict[str, Any], verbose: bool = False
) -> ChatbotTestResult:
    """Evaluate a single chatbot test case."""
    import time

    query = case["query"]
    expected_brands = case.get("expected_brands", [])
    expected_facts = case.get("expected_facts", [])
    expected_sources = case.get("expected_source_types", [])

    try:
        start = time.time()
        result = await workflow.chat(query)
        latency_ms = (time.time() - start) * 1000

        response = result.get("response", "")
        sources = result.get("sources", [])

        brands_found = extract_brands_from_response(response)
        facts_found = extract_facts_from_response(response)
        source_types_found = extract_source_types(sources)

        # Check pass/fail conditions
        brand_recall = calculate_recall(brands_found, expected_brands)
        fact_recall = calculate_recall(facts_found, expected_facts)
        source_recall = calculate_recall(source_types_found, expected_sources)

        passed = (
            brand_recall >= 0.5
            and fact_recall >= 0.5
            and len(response) > 50  # Minimum response length
        )

        if verbose:
            print(f"\n  Query: {query[:50]}...")
            print(f"  Brands: {brands_found} (expected: {expected_brands})")
            print(f"  Facts: {facts_found} (expected: {expected_facts})")
            print(f"  Sources: {source_types_found}")
            print(f"  Latency: {latency_ms:.0f}ms")
            print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

        return ChatbotTestResult(
            query=query,
            passed=passed,
            brands_found=brands_found,
            expected_brands=expected_brands,
            facts_found=facts_found,
            expected_facts=expected_facts,
            source_types_found=source_types_found,
            expected_source_types=expected_sources,
            latency_ms=latency_ms,
        )

    except Exception as e:
        if verbose:
            print(f"\n  Query: {query[:50]}...")
            print(f"  Status: ❌ ERROR - {str(e)}")

        return ChatbotTestResult(
            query=query,
            passed=False,
            brands_found=[],
            expected_brands=expected_brands,
            facts_found=[],
            expected_facts=expected_facts,
            source_types_found=[],
            expected_source_types=expected_sources,
            latency_ms=0,
            error=str(e),
        )


async def evaluate_chatbot(verbose: bool = False) -> EvaluationMetrics:
    """Run chatbot golden set evaluation."""
    print("\n" + "=" * 60)
    print("CHATBOT GOLDEN SET EVALUATION")
    print("=" * 60)

    golden_path = project_root / "tests" / "golden" / "chatbot_golden.jsonl"
    cases = load_golden_set(golden_path)

    if not cases:
        print("No test cases found.")
        return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0)

    print(f"Loaded {len(cases)} test cases")

    # Initialize workflow
    try:
        from src.application.workflows.batch_workflow import BatchWorkflow

        workflow = BatchWorkflow(use_hybrid=True)
    except ImportError as e:
        print(f"Error importing BatchWorkflow: {e}")
        print("Running in dry-run mode (no actual API calls)")
        return EvaluationMetrics(
            total=len(cases),
            passed=0,
            failed=len(cases),
            avg_latency_ms=0,
            brand_recall=0,
            fact_recall=0,
            source_coverage=0,
        )

    results: list[ChatbotTestResult] = []

    for i, case in enumerate(cases, 1):
        print(f"\nTest {i}/{len(cases)}: {case['query'][:40]}...", end="")
        result = await evaluate_chatbot_case(workflow, case, verbose)
        results.append(result)
        if not verbose:
            print(" ✅" if result.passed else " ❌")

    # Calculate metrics
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    brand_recalls = [calculate_recall(r.brands_found, r.expected_brands) for r in results]
    fact_recalls = [calculate_recall(r.facts_found, r.expected_facts) for r in results]
    source_recalls = [
        calculate_recall(r.source_types_found, r.expected_source_types) for r in results
    ]

    avg_brand_recall = sum(brand_recalls) / len(brand_recalls) if brand_recalls else 0
    avg_fact_recall = sum(fact_recalls) / len(fact_recalls) if fact_recalls else 0
    avg_source_coverage = sum(source_recalls) / len(source_recalls) if source_recalls else 0

    print("\n" + "-" * 40)
    print("CHATBOT EVALUATION RESULTS")
    print("-" * 40)
    print(f"Total Tests:     {len(results)}")
    print(f"Passed:          {passed} ({100 * passed / len(results):.1f}%)")
    print(f"Failed:          {failed}")
    print(f"Avg Latency:     {avg_latency:.0f}ms")
    print(f"Brand Recall:    {avg_brand_recall:.2f}")
    print(f"Fact Recall:     {avg_fact_recall:.2f}")
    print(f"Source Coverage: {avg_source_coverage:.2f}")

    return EvaluationMetrics(
        total=len(results),
        passed=passed,
        failed=failed,
        avg_latency_ms=avg_latency,
        brand_recall=avg_brand_recall,
        fact_recall=avg_fact_recall,
        source_coverage=avg_source_coverage,
    )


async def evaluate_report(verbose: bool = False) -> EvaluationMetrics:
    """Run report golden set evaluation."""
    print("\n" + "=" * 60)
    print("REPORT GOLDEN SET EVALUATION")
    print("=" * 60)

    golden_path = project_root / "tests" / "golden" / "report_golden.jsonl"
    cases = load_golden_set(golden_path)

    if not cases:
        print("No test cases found.")
        return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0)

    print(f"Loaded {len(cases)} test cases")
    print("Note: Report evaluation requires actual data and is expensive.")
    print("Skipping actual generation - using schema validation only.")

    # Schema validation only (no actual report generation)
    passed = 0
    for case in cases:
        expected_sections = case.get("expected_sections", [])
        expected_kpis = case.get("expected_kpis", [])

        # Validate schema
        if expected_sections and expected_kpis:
            passed += 1

    print("\n" + "-" * 40)
    print("REPORT EVALUATION RESULTS")
    print("-" * 40)
    print(f"Total Tests:     {len(cases)}")
    print(f"Schema Valid:    {passed}")
    print("Note: Full evaluation requires running actual report generation")

    return EvaluationMetrics(
        total=len(cases),
        passed=passed,
        failed=len(cases) - passed,
        avg_latency_ms=0,
        brand_recall=0,
        fact_recall=0,
        source_coverage=0,
    )


async def main():
    parser = argparse.ArgumentParser(description="Golden Set Evaluation")
    parser.add_argument("--chatbot", action="store_true", help="Run chatbot evaluation only")
    parser.add_argument("--report", action="store_true", help="Run report evaluation only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    args = parser.parse_args()

    # If no specific option, run both
    run_chatbot = args.chatbot or (not args.chatbot and not args.report)
    run_report = args.report or (not args.chatbot and not args.report)

    print("=" * 60)
    print("AMORE RAG-KG Hybrid Agent - Golden Set Evaluation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    if run_chatbot:
        chatbot_metrics = await evaluate_chatbot(args.verbose)

    if run_report:
        report_metrics = await evaluate_report(args.verbose)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    # Save results
    results_path = project_root / "tests" / "golden" / "evaluation_results.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "chatbot": chatbot_metrics.__dict__ if run_chatbot else None,
        "report": report_metrics.__dict__ if run_report else None,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
