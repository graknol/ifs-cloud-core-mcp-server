#!/usr/bin/env python3
"""
Evaluation and testing script for the PL/SQL fine-tuning pipeline.
This script helps validate the quality of training samples and provides metrics.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import statistics


class TrainingDataEvaluator:
    """Evaluates the quality of training data for fine-tuning."""

    def __init__(self, source_dir: str = "_work/source"):
        self.source_dir = Path(source_dir)
        self.samples = []
        self.stats = defaultdict(list)

    def load_training_samples(
        self, samples_file: str = "training_samples.jsonl"
    ) -> None:
        """Load training samples for evaluation."""
        if not os.path.exists(samples_file):
            print(f"❌ Training samples file not found: {samples_file}")
            return

        with open(samples_file, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        print(f"✅ Loaded {len(self.samples)} training samples")

    def analyze_sample_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of samples across various dimensions."""
        if not self.samples:
            return {}

        analysis = {
            "total_samples": len(self.samples),
            "api_distribution": Counter(),
            "module_distribution": Counter(),
            "operation_types": Counter(),
            "complexity_stats": {
                "cyclomatic": [],
                "code_lines": [],
                "estimated_tokens": [],
            },
            "pagerank_tiers": {
                "high": 0,  # > 0.01
                "medium": 0,  # 0.001 - 0.01
                "low": 0,  # < 0.001
            },
        }

        for sample in self.samples:
            context = sample.get("context", {})

            # API and module distribution
            api_name = context.get("api_name", "unknown")
            module = context.get("module", "unknown")
            operation_type = context.get("operation_type", "unknown")

            analysis["api_distribution"][api_name] += 1
            analysis["module_distribution"][module] += 1
            analysis["operation_types"][operation_type] += 1

            # Complexity metrics
            metrics = context.get("complexity_metrics", {})
            analysis["complexity_stats"]["cyclomatic"].append(
                metrics.get("cyclomatic_complexity", 0)
            )
            analysis["complexity_stats"]["code_lines"].append(
                metrics.get("code_lines", 0)
            )
            analysis["complexity_stats"]["estimated_tokens"].append(
                metrics.get("estimated_tokens", 0)
            )

            # PageRank tiers
            pagerank = context.get("pagerank_score", 0)
            if pagerank > 0.01:
                analysis["pagerank_tiers"]["high"] += 1
            elif pagerank > 0.001:
                analysis["pagerank_tiers"]["medium"] += 1
            else:
                analysis["pagerank_tiers"]["low"] += 1

        return analysis

    def calculate_diversity_metrics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate diversity metrics for the sample set."""
        metrics = {}

        # API diversity (normalized entropy)
        api_counts = list(analysis["api_distribution"].values())
        total = sum(api_counts)
        api_entropy = -sum(
            (count / total) * (count / total) for count in api_counts if count > 0
        )
        metrics["api_diversity"] = (
            api_entropy / len(api_counts) if len(api_counts) > 1 else 0
        )

        # Module diversity
        module_counts = list(analysis["module_distribution"].values())
        total = sum(module_counts)
        module_entropy = -sum(
            (count / total) * (count / total) for count in module_counts if count > 0
        )
        metrics["module_diversity"] = (
            module_entropy / len(module_counts) if len(module_counts) > 1 else 0
        )

        # Complexity spread (coefficient of variation)
        complexity_values = analysis["complexity_stats"]["cyclomatic"]
        if complexity_values:
            mean_complexity = statistics.mean(complexity_values)
            std_complexity = (
                statistics.stdev(complexity_values) if len(complexity_values) > 1 else 0
            )
            metrics["complexity_spread"] = (
                std_complexity / mean_complexity if mean_complexity > 0 else 0
            )

        return metrics

    def validate_sample_quality(self) -> List[Dict[str, Any]]:
        """Validate individual samples for quality issues."""
        quality_issues = []

        for i, sample in enumerate(self.samples):
            issues = []
            context = sample.get("context", {})
            code = sample.get("code", "")
            summary = sample.get("summary", "")

            # Check for missing required fields
            required_fields = ["api_name", "module", "function_name"]
            for field in required_fields:
                if not context.get(field):
                    issues.append(f"Missing {field}")

            # Check code quality
            if len(code) < 50:
                issues.append("Code too short")
            elif len(code) > 5000:
                issues.append("Code too long")

            # Check for common patterns that suggest low quality
            if "Get_" in context.get("function_name", "") and len(code) < 200:
                issues.append("Potential trivial getter")

            if "Set_" in context.get("function_name", "") and len(code) < 200:
                issues.append("Potential trivial setter")

            # Check summary quality (if available)
            if summary:
                if len(summary) < 10:
                    issues.append("Summary too short")
                elif len(summary) > 500:
                    issues.append("Summary too long")

                # Check for generic/template summaries
                generic_phrases = [
                    "this function",
                    "this procedure",
                    "performs operation",
                ]
                if any(phrase in summary.lower() for phrase in generic_phrases):
                    issues.append("Generic summary")

            if issues:
                quality_issues.append(
                    {
                        "sample_index": i,
                        "function_name": context.get("function_name", "unknown"),
                        "api_name": context.get("api_name", "unknown"),
                        "issues": issues,
                    }
                )

        return quality_issues

    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.samples:
            return "❌ No training samples loaded. Run load_training_samples() first."

        # Perform analysis
        analysis = self.analyze_sample_distribution()
        diversity_metrics = self.calculate_diversity_metrics(analysis)
        quality_issues = self.validate_sample_quality()

        # Generate report
        report = []
        report.append("📊 TRAINING DATA EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")

        # Basic statistics
        report.append(f"📈 Basic Statistics:")
        report.append(f"   Total Samples: {analysis['total_samples']}")
        report.append(f"   Unique APIs: {len(analysis['api_distribution'])}")
        report.append(f"   Unique Modules: {len(analysis['module_distribution'])}")
        report.append("")

        # PageRank distribution
        report.append(f"🎯 PageRank Tier Distribution:")
        tiers = analysis["pagerank_tiers"]
        report.append(
            f"   High Tier (>0.01): {tiers['high']} ({tiers['high']/analysis['total_samples']*100:.1f}%)"
        )
        report.append(
            f"   Medium Tier (0.001-0.01): {tiers['medium']} ({tiers['medium']/analysis['total_samples']*100:.1f}%)"
        )
        report.append(
            f"   Low Tier (<0.001): {tiers['low']} ({tiers['low']/analysis['total_samples']*100:.1f}%)"
        )
        report.append("")

        # Complexity statistics
        complexity_stats = analysis["complexity_stats"]
        report.append(f"🧮 Complexity Statistics:")
        if complexity_stats["cyclomatic"]:
            cyclo_mean = statistics.mean(complexity_stats["cyclomatic"])
            cyclo_median = statistics.median(complexity_stats["cyclomatic"])
            report.append(
                f"   Cyclomatic Complexity: Mean={cyclo_mean:.1f}, Median={cyclo_median:.1f}"
            )

        if complexity_stats["code_lines"]:
            lines_mean = statistics.mean(complexity_stats["code_lines"])
            lines_median = statistics.median(complexity_stats["code_lines"])
            report.append(
                f"   Code Lines: Mean={lines_mean:.1f}, Median={lines_median:.1f}"
            )
        report.append("")

        # Diversity metrics
        report.append(f"🌈 Diversity Metrics:")
        report.append(
            f"   API Diversity: {diversity_metrics.get('api_diversity', 0):.3f}"
        )
        report.append(
            f"   Module Diversity: {diversity_metrics.get('module_diversity', 0):.3f}"
        )
        report.append(
            f"   Complexity Spread: {diversity_metrics.get('complexity_spread', 0):.3f}"
        )
        report.append("")

        # Top APIs and modules
        report.append(f"🏆 Top 10 APIs:")
        for api, count in analysis["api_distribution"].most_common(10):
            report.append(f"   {api}: {count} samples")
        report.append("")

        report.append(f"🏆 Top 10 Modules:")
        for module, count in analysis["module_distribution"].most_common(10):
            report.append(f"   {module}: {count} samples")
        report.append("")

        # Quality issues
        report.append(f"⚠️ Quality Issues:")
        report.append(f"   Total Issues Found: {len(quality_issues)}")

        if quality_issues:
            issue_types = Counter()
            for issue_set in quality_issues:
                for issue in issue_set["issues"]:
                    issue_types[issue] += 1

            report.append(f"   Issue Breakdown:")
            for issue_type, count in issue_types.most_common():
                report.append(f"     {issue_type}: {count} samples")
        report.append("")

        # Recommendations
        report.append(f"💡 Recommendations:")

        # Check balance
        high_tier_ratio = tiers["high"] / analysis["total_samples"]
        if high_tier_ratio < 0.2:
            report.append(
                f"   ⚠️ Consider including more high-PageRank samples (currently {high_tier_ratio*100:.1f}%)"
            )

        if high_tier_ratio > 0.4:
            report.append(
                f"   ⚠️ Consider more diversity - high-PageRank samples are {high_tier_ratio*100:.1f}%"
            )

        # Check diversity
        if diversity_metrics.get("api_diversity", 0) < 0.1:
            report.append(f"   ⚠️ Low API diversity - consider samples from more APIs")

        if diversity_metrics.get("module_diversity", 0) < 0.1:
            report.append(
                f"   ⚠️ Low module diversity - consider samples from more modules"
            )

        # Check quality issues
        quality_ratio = len(quality_issues) / analysis["total_samples"]
        if quality_ratio > 0.1:
            report.append(
                f"   ⚠️ High quality issue rate ({quality_ratio*100:.1f}%) - review flagged samples"
            )

        if len(quality_issues) == 0:
            report.append(f"   ✅ No major quality issues detected!")

        report.append("")
        report.append("🎯 Overall Assessment:")

        # Calculate overall score
        score_components = []

        # Balance score (ideal is ~25% each tier, with some flexibility)
        balance_score = (
            1.0
            - abs(0.25 - tiers["high"] / analysis["total_samples"])
            - abs(0.5 - tiers["medium"] / analysis["total_samples"])
            - abs(0.25 - tiers["low"] / analysis["total_samples"])
        )
        score_components.append(max(0, balance_score))

        # Diversity score
        diversity_score = (
            diversity_metrics.get("api_diversity", 0)
            + diversity_metrics.get("module_diversity", 0)
        ) / 2
        score_components.append(
            min(1.0, diversity_score * 10)
        )  # Scale up since entropy is typically small

        # Quality score
        quality_score = 1.0 - min(1.0, quality_ratio * 2)  # Cap penalty at 50% issues
        score_components.append(quality_score)

        overall_score = statistics.mean(score_components) * 100

        if overall_score >= 80:
            report.append(f"   🏆 Excellent dataset (Score: {overall_score:.1f}/100)")
        elif overall_score >= 60:
            report.append(f"   ✅ Good dataset (Score: {overall_score:.1f}/100)")
        elif overall_score >= 40:
            report.append(
                f"   ⚠️ Fair dataset - improvements recommended (Score: {overall_score:.1f}/100)"
            )
        else:
            report.append(
                f"   ❌ Poor dataset - significant improvements needed (Score: {overall_score:.1f}/100)"
            )

        return "\n".join(report)

    def export_quality_report(self, output_file: str = "quality_report.json") -> None:
        """Export detailed quality analysis to JSON."""
        analysis = self.analyze_sample_distribution()
        diversity_metrics = self.calculate_diversity_metrics(analysis)
        quality_issues = self.validate_sample_quality()

        report_data = {
            "analysis": analysis,
            "diversity_metrics": diversity_metrics,
            "quality_issues": quality_issues,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Detailed quality report exported to {output_file}")


def main():
    """Main evaluation script."""
    print("🔍 PL/SQL Training Data Evaluator")
    print("=" * 40)

    evaluator = TrainingDataEvaluator()

    # Check if training samples exist
    if os.path.exists("training_samples.jsonl"):
        evaluator.load_training_samples()

        # Generate and display report
        report = evaluator.generate_report()
        print(report)

        # Export detailed report
        evaluator.export_quality_report()

    elif os.path.exists("extracted_samples.jsonl"):
        print("Found extracted samples, loading those...")
        evaluator.load_training_samples("extracted_samples.jsonl")

        report = evaluator.generate_report()
        print(report)

    else:
        print("❌ No training samples found.")
        print("Run extract_training_samples.py first to generate samples.")
        print("\nAlternatively, if you have samples in a different file,")
        print("rename it to 'training_samples.jsonl' and run this script again.")


if __name__ == "__main__":
    main()
