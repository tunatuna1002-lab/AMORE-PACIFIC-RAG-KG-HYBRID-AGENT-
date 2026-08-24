"""
Ablation Study Runner
=====================
Runs the evaluation pipeline under multiple feature-flag configurations,
disabling one component at a time to measure each component's contribution.

Usage (via CLI):
    python3 -c "from eval.cli import main; main(['ablation', '--dataset', '...', '--out', '...'])"

Configs:
    full              – baseline (all components enabled)
    no-kg             – disable Knowledge Graph queries
    no-ontology       – disable OWL + rule-based reasoning
    no-reranker       – disable relevance grading / reranking
    no-query-rewrite  – disable query rewriting
    no-fusion         – disable confidence fusion scoring
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.loader import load_dataset
from eval.report import ReportGenerator
from eval.runner import EvalRunner
from eval.schemas import EvalConfig, EvalReport

logger = logging.getLogger(__name__)

# ─── Ablation configuration registry ────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict[str, str]] = {
    "full": {},
    "no-kg": {
        "FF_ONTOLOGY_USE_ONTOLOGY_KG": "false",
    },
    "no-ontology": {
        "FF_REASONER_USE_OWL_REASONER": "false",
        "FF_REASONER_USE_UNIFIED_REASONER": "false",
    },
    "no-reranker": {
        "FF_RETRIEVER_USE_RERANKER": "false",
    },
    "no-query-rewrite": {
        "FF_AGENTS_USE_QUERY_REWRITER": "false",
    },
    "no-fusion": {
        "FF_RETRIEVER_USE_CONFIDENCE_FUSION": "false",
    },
}


# ─── Result containers ──────────────────────────────────────────────────────


@dataclass
class AblationResult:
    config_name: str
    env_overrides: dict[str, str]
    report: EvalReport


@dataclass
class AblationReport:
    timestamp: str
    configs_run: list[str]
    results: list[AblationResult] = field(default_factory=list)
    comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    component_contributions: dict[str, dict[str, float]] = field(default_factory=dict)


# ─── Runner ──────────────────────────────────────────────────────────────────


class AblationRunner:
    """Run the same dataset under multiple feature-flag configurations."""

    def __init__(
        self,
        dataset_path: str,
        out_dir: str,
        eval_config: EvalConfig,
        judge: Any = None,
        use_semantic_similarity: bool = False,
        concurrency: int = 1,
    ) -> None:
        self.dataset_path = dataset_path
        self.out_dir = Path(out_dir) / "ablation"
        self.eval_config = eval_config
        self.judge = judge
        self.use_semantic_similarity = use_semantic_similarity
        self.concurrency = concurrency

    async def run_all(self, configs: list[str] | None = None) -> AblationReport:
        """Run each config sequentially and collect results."""
        configs = configs or list(ABLATION_CONFIGS.keys())

        # Validate config names
        for name in configs:
            if name not in ABLATION_CONFIGS:
                raise ValueError(
                    f"Unknown ablation config: {name}. "
                    f"Available: {list(ABLATION_CONFIGS.keys())}"
                )

        items = load_dataset(Path(self.dataset_path))
        logger.info(f"Loaded {len(items)} items for ablation study")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        report = AblationReport(
            timestamp=datetime.now().isoformat(),
            configs_run=configs,
        )

        for config_name in configs:
            overrides = ABLATION_CONFIGS[config_name]
            logger.info(f"\n{'='*60}")
            logger.info(f"ABLATION CONFIG: {config_name}")
            if overrides:
                logger.info(f"  Overrides: {overrides}")
            else:
                logger.info("  Baseline (no overrides)")
            logger.info(f"{'='*60}")

            saved_env = self._apply_env_overrides(overrides)
            try:
                result = await self._run_single_config(config_name, overrides, items)
                report.results.append(result)
            except Exception as e:
                logger.error(f"Config {config_name} failed: {e}")
                # Create a minimal failed report
                report.results.append(
                    AblationResult(
                        config_name=config_name,
                        env_overrides=overrides,
                        report=EvalReport(),
                    )
                )
            finally:
                self._restore_env(saved_env)

        # Build comparison
        report.comparison = self._build_comparison(report.results)
        report.component_contributions = self._build_contributions(report)

        # Save and print
        self._save_report(report)
        print_ablation_summary(report)

        return report

    async def _run_single_config(
        self,
        config_name: str,
        overrides: dict[str, str],
        items: list,
    ) -> AblationResult:
        """Run evaluation for a single ablation config."""
        from src.infrastructure.feature_flags import FeatureFlags

        FeatureFlags.reset_instance()

        agent = await _create_agent()
        runner = EvalRunner(
            agent=agent,
            config=self.eval_config,
            judge=self.judge,
            use_semantic_similarity=self.use_semantic_similarity,
        )

        results = await runner.run_dataset(items, concurrency=self.concurrency)

        # Generate per-config report
        config_out = self.out_dir / config_name
        config_out.mkdir(parents=True, exist_ok=True)
        report_gen = ReportGenerator(config=self.eval_config)
        eval_report = report_gen.generate_report(results, config_out)

        # Reset flags after run
        FeatureFlags.reset_instance()

        return AblationResult(
            config_name=config_name,
            env_overrides=overrides,
            report=eval_report,
        )

    def _apply_env_overrides(self, overrides: dict[str, str]) -> dict[str, str | None]:
        """Set env vars, return old values for restoration."""
        saved: dict[str, str | None] = {}
        for key, value in overrides.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value
        return saved

    def _restore_env(self, saved: dict[str, str | None]) -> None:
        """Restore original env vars."""
        for key, old_value in saved.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    def _build_comparison(self, results: list[AblationResult]) -> dict[str, dict[str, float]]:
        """Extract key metrics from each config's report."""
        metrics_to_compare = [
            "pass_rate",
            "avg_overall_score",
            "avg_latency_ms",
        ]
        layer_metrics = [
            "l1_entity_link_f1",
            "l1_concept_map_f1",
            "l2_context_recall_at_k",
            "l2_mrr",
            "l3_hits_at_k",
            "l3_kg_edge_f1",
            "l4_constraint_violation_rate",
            "l5_answer_f1",
            "l5_answer_exact_match",
        ]

        comparison: dict[str, dict[str, float]] = {}

        for metric in metrics_to_compare:
            comparison[metric] = {}
            for r in results:
                comparison[metric][r.config_name] = getattr(r.report.aggregates, metric, 0.0)

        for metric in layer_metrics:
            comparison[metric] = {}
            for r in results:
                comparison[metric][r.config_name] = r.report.aggregates.by_layer.get(metric, 0.0)

        return comparison

    def _build_contributions(self, report: AblationReport) -> dict[str, dict[str, float]]:
        """Calculate each component's contribution (delta from baseline)."""
        comparison = report.comparison
        contributions: dict[str, dict[str, float]] = {}

        baseline_values: dict[str, float] = {}
        for metric, config_vals in comparison.items():
            baseline_values[metric] = config_vals.get("full", 0.0)

        component_map = {
            "no-kg": "kg",
            "no-ontology": "ontology",
            "no-reranker": "reranker",
            "no-query-rewrite": "query-rewrite",
            "no-fusion": "fusion",
        }

        for config_name, component_name in component_map.items():
            if config_name not in report.configs_run:
                continue
            contributions[component_name] = {}
            for metric in ["pass_rate", "avg_overall_score", "l5_answer_f1"]:
                baseline = baseline_values.get(metric, 0.0)
                ablated = comparison.get(metric, {}).get(config_name, 0.0)
                contributions[component_name][f"delta_{metric}"] = ablated - baseline

        return contributions

    def _save_report(self, report: AblationReport) -> None:
        """Save ablation report as JSON."""
        report_data = {
            "timestamp": report.timestamp,
            "configs_run": report.configs_run,
            "results": [
                {
                    "config_name": r.config_name,
                    "env_overrides": r.env_overrides,
                    "aggregates": r.report.aggregates.model_dump(),
                }
                for r in report.results
            ],
            "comparison": report.comparison,
            "component_contributions": report.component_contributions,
        }
        out_path = self.out_dir / "ablation_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Ablation report saved to {out_path}")


# ─── Agent factory ───────────────────────────────────────────────────────────


async def _create_agent():
    """Create a fresh HybridChatbotAgent instance."""
    from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

    return HybridChatbotAgent()


# ─── Console output ─────────────────────────────────────────────────────────


def print_ablation_summary(report: AblationReport) -> None:
    """Print comparison table to console."""
    print("\n" + "=" * 72)
    print("ABLATION STUDY RESULTS")
    print("=" * 72)

    # Header
    header = f"{'Config':<24} | {'Pass Rate':>9} | {'Avg Score':>9} | {'F1':>6} | {'Latency':>8}"
    print(header)
    print("-" * 72)

    # Rows
    comparison = report.comparison
    for config_name in report.configs_run:
        pass_rate = comparison.get("pass_rate", {}).get(config_name, 0.0)
        avg_score = comparison.get("avg_overall_score", {}).get(config_name, 0.0)
        f1 = comparison.get("l5_answer_f1", {}).get(config_name, 0.0)
        latency = comparison.get("avg_latency_ms", {}).get(config_name, 0.0)

        label = f"{config_name} (baseline)" if config_name == "full" else config_name
        print(
            f"{label:<24} | {pass_rate:>8.1%} | {avg_score:>9.3f} | {f1:>5.3f} | {latency:>7.0f}ms"
        )

    # Contributions
    if report.component_contributions:
        print()
        print("COMPONENT CONTRIBUTIONS (delta from baseline):")
        for comp, deltas in report.component_contributions.items():
            d_pass = deltas.get("delta_pass_rate", 0.0)
            d_score = deltas.get("delta_avg_overall_score", 0.0)
            print(f"  {comp:<20} {d_pass:>+7.1%} pass rate, {d_score:>+7.3f} score")

    print("=" * 72 + "\n")
