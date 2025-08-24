#!/usr/bin/env python3
"""
Parallel Processing Integration for Training Sample Generation

This script integrates the high-performance pipeline with existing components:
- Uses existing PageRank scores and file discovery
- Integrates with AST parser when available
- Optimizes for both UnixCoder and Claude summarization
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List
import time
import os

# Import existing components
from high_performance_pipeline import HighPerformancePipelineProcessor
from extract_training_samples import (
    load_pagerank_scores,
    find_plsql_files_with_pagerank,
    create_stratified_sample,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("parallel_pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ParallelExtractionPipeline:
    """Main orchestrator for parallel training sample generation."""

    def __init__(
        self,
        source_directory: str = "../ifs-cloud-analysis",
        parser_executable: str = "plsql_parser.exe",
        target_samples: int = 10000,
        use_vllm: bool = False,
    ):  # Default to False since vLLM needs setup

        self.source_directory = Path(source_directory)
        self.parser_executable = parser_executable
        self.target_samples = target_samples
        self.use_vllm = use_vllm

        # Verify parser exists
        self.has_parser = Path(parser_executable).exists()
        if not self.has_parser:
            logger.warning(f"⚠️ AST parser not found at {parser_executable}")
            logger.warning("   Will use fallback extraction method")

    async def generate_training_samples(self) -> Dict:
        """Generate training samples using parallel processing."""

        logger.info("🚀 Starting Parallel Training Sample Generation")
        logger.info("=" * 60)

        start_time = time.time()

        # Step 1: Load PageRank scores
        logger.info("📊 Loading PageRank scores...")
        pagerank_file = self.source_directory / "comprehensive_plsql_analysis.json"
        pagerank_scores = load_pagerank_scores(str(pagerank_file))
        logger.info(f"   Loaded {len(pagerank_scores)} PageRank scores")

        # Step 2: Find and stratify files
        logger.info("🔍 Finding and stratifying PL/SQL files...")
        all_files = find_plsql_files_with_pagerank(
            str(self.source_directory), pagerank_scores
        )
        selected_files = create_stratified_sample(
            all_files, pagerank_scores, self.target_samples
        )
        logger.info(f"   Selected {len(selected_files)} files for processing")

        # Step 3: Configure processing pipeline
        if self.has_parser:
            logger.info("🔧 Using AST parser for high-quality extraction")
            processor = HighPerformancePipelineProcessor(
                parser_executable=self.parser_executable,
                max_parse_workers=50,  # High I/O concurrency
                max_process_workers=8,  # CPU cores
                batch_size=32,  # Optimal for GPU
                use_vllm=self.use_vllm,
            )
        else:
            logger.info("🔧 Using fallback extraction method")
            processor = await self._create_fallback_processor()

        # Step 4: Run parallel processing pipeline
        logger.info("⚡ Starting parallel processing pipeline...")
        stats = await processor.process_files_pipeline(
            selected_files,
            pagerank_scores,
            output_file="parallel_training_samples.jsonl",
        )

        total_time = time.time() - start_time

        # Step 5: Generate summary report
        summary = self._generate_summary_report(stats, total_time)

        logger.info("🎉 Parallel processing complete!")
        self._log_summary_report(summary)

        return summary

    async def _create_fallback_processor(self) -> HighPerformancePipelineProcessor:
        """Create processor with fallback extraction when AST parser unavailable."""

        # Import the regex-based extractor as fallback
        from extract_training_samples import (
            extract_plsql_functions,
            calculate_complexity,
        )

        class FallbackProcessor(HighPerformancePipelineProcessor):
            """Processor using regex-based extraction as fallback."""

            def _parse_single_file(self, file_path: Path, pagerank_score: float):
                """Use regex-based extraction instead of AST parser."""
                try:
                    from high_performance_pipeline import FunctionData

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Use existing regex extraction
                    functions_data = extract_plsql_functions(content, str(file_path))

                    function_objects = []
                    for func_name, func_code in functions_data:
                        complexity = calculate_complexity(func_code)

                        function = FunctionData(
                            file_path=str(file_path),
                            function_name=func_name,
                            function_text=func_code,
                            complexity=complexity,
                            ast_metadata={"extraction_method": "regex"},
                            pagerank_score=pagerank_score,
                        )
                        function_objects.append(function)

                    return function_objects

                except Exception as e:
                    logger.error(f"Fallback extraction failed for {file_path}: {e}")
                    return []

        return FallbackProcessor(
            parser_executable="fallback",
            max_parse_workers=30,  # Slightly lower for regex processing
            max_process_workers=8,
            batch_size=32,
            use_vllm=self.use_vllm,
        )

    def _generate_summary_report(self, stats, total_time: float) -> Dict:
        """Generate comprehensive summary report."""

        return {
            "pipeline_performance": {
                "total_time": total_time,
                "total_files": stats.total_files,
                "parsed_files": stats.parsed_files,
                "processed_functions": stats.processed_functions,
                "summarized_functions": stats.summarized_functions,
                "failed_files": stats.failed_files,
                "success_rate": (
                    stats.parsed_files / stats.total_files
                    if stats.total_files > 0
                    else 0
                ),
                "throughput_files_per_sec": stats.total_files / total_time,
                "throughput_functions_per_sec": stats.processed_functions / total_time,
            },
            "stage_performance": {
                "parsing_time": stats.parse_time,
                "processing_time": stats.process_time,
                "summarization_time": stats.summarize_time,
                "parsing_throughput": (
                    stats.parsed_files / stats.parse_time if stats.parse_time > 0 else 0
                ),
                "processing_throughput": (
                    stats.processed_functions / stats.process_time
                    if stats.process_time > 0
                    else 0
                ),
                "summarization_throughput": (
                    stats.summarized_functions / stats.summarize_time
                    if stats.summarize_time > 0
                    else 0
                ),
            },
            "configuration": {
                "has_ast_parser": self.has_parser,
                "parser_executable": self.parser_executable,
                "target_samples": self.target_samples,
                "use_vllm": self.use_vllm,
                "source_directory": str(self.source_directory),
            },
            "quality_metrics": {
                "extraction_method": "ast" if self.has_parser else "regex",
                "average_function_size": None,  # To be calculated from output
                "complexity_distribution": None,  # To be calculated from output
            },
        }

    def _log_summary_report(self, summary: Dict):
        """Log comprehensive summary report."""

        perf = summary["pipeline_performance"]
        stages = summary["stage_performance"]
        config = summary["configuration"]

        logger.info("📊 PARALLEL PIPELINE SUMMARY REPORT")
        logger.info("=" * 60)
        logger.info(f"🎯 Target samples: {config['target_samples']}")
        logger.info(f"📁 Source directory: {config['source_directory']}")
        logger.info(
            f"🔧 Parser type: {'AST' if config['has_ast_parser'] else 'Regex fallback'}"
        )
        logger.info(
            f"🧠 Summarization: {'vLLM' if config['use_vllm'] else 'HuggingFace'}"
        )
        logger.info("")
        logger.info("📈 PERFORMANCE METRICS")
        logger.info(f"   Total processing time: {perf['total_time']:.2f}s")
        logger.info(
            f"   Files processed: {perf['parsed_files']}/{perf['total_files']} ({perf['success_rate']:.1%})"
        )
        logger.info(f"   Functions extracted: {perf['processed_functions']}")
        logger.info(f"   Summaries generated: {perf['summarized_functions']}")
        logger.info(
            f"   Overall throughput: {perf['throughput_files_per_sec']:.2f} files/sec"
        )
        logger.info(
            f"   Function throughput: {perf['throughput_functions_per_sec']:.2f} funcs/sec"
        )
        logger.info("")
        logger.info("⏱️ STAGE BREAKDOWN")
        logger.info(
            f"   Stage 1 (Parsing): {stages['parsing_time']:.2f}s ({stages['parsing_throughput']:.2f} files/sec)"
        )
        logger.info(
            f"   Stage 2 (Processing): {stages['processing_time']:.2f}s ({stages['processing_throughput']:.2f} funcs/sec)"
        )
        logger.info(
            f"   Stage 3 (Summarization): {stages['summarization_time']:.2f}s ({stages['summarization_throughput']:.2f} summaries/sec)"
        )

        # Save full report
        with open("parallel_pipeline_report.json", "w") as f:
            json.dump(summary, f, indent=2)


async def run_claude_summarization_parallel(
    input_file: str = "parallel_training_samples.jsonl",
    output_file: str = "claude_training_samples.jsonl",
    batch_size: int = 20,
):
    """Run Claude summarization on parallel-extracted samples."""

    logger.info("🧠 Starting Claude parallel summarization...")

    # Import Claude generation
    from generate_summaries_with_claude import generate_summaries_batch

    # Load samples
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]

    logger.info(f"📊 Processing {len(samples)} samples with Claude")

    # Process in batches for API rate limiting
    processed_samples = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        try:
            # Generate summaries for batch
            batch_with_summaries = await generate_summaries_batch(batch)
            processed_samples.extend(batch_with_summaries)

            logger.info(
                f"✅ Processed batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}"
            )

        except Exception as e:
            logger.error(f"❌ Batch failed: {e}")
            # Add without summaries as fallback
            processed_samples.extend(batch)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"🎉 Claude summarization complete: {output_file}")


async def main():
    """Main execution function."""

    print("🚀 HIGH-PERFORMANCE PARALLEL TRAINING PIPELINE")
    print("=" * 60)

    # Configuration options
    import argparse

    parser = argparse.ArgumentParser(
        description="High-performance parallel training sample generation"
    )
    parser.add_argument(
        "--source",
        default="../ifs-cloud-analysis",
        help="Source directory with PL/SQL files",
    )
    parser.add_argument(
        "--parser", default="plsql_parser.exe", help="Path to AST parser executable"
    )
    parser.add_argument(
        "--samples", type=int, default=10000, help="Target number of samples"
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM for summarization (requires setup)",
    )
    parser.add_argument(
        "--claude", action="store_true", help="Use Claude for summarization instead"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run with small sample for testing"
    )

    args = parser.parse_args()

    if args.test:
        args.samples = 100
        print("🧪 Running in test mode with 100 samples")

    # Initialize pipeline
    pipeline = ParallelExtractionPipeline(
        source_directory=args.source,
        parser_executable=args.parser,
        target_samples=args.samples,
        use_vllm=args.vllm,
    )

    try:
        # Run main pipeline
        summary = await pipeline.generate_training_samples()

        # Optional Claude post-processing
        if args.claude:
            await run_claude_summarization_parallel()

        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Check 'parallel_pipeline_report.json' for detailed metrics")
        print(f"📁 Output: 'parallel_training_samples.jsonl'")
        if args.claude:
            print(f"📁 Claude output: 'claude_training_samples.jsonl'")

    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
