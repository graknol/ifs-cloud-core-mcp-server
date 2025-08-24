#!/usr/bin/env python3
"""
Complete High-Performance Training Pipeline

This script demonstrates the complete workflow:
1. Run performance benchmark
2. Select optimal configuration
3. Execute parallel processing pipeline
4. Generate training samples

Usage:
    python run_complete_pipeline.py [--test] [--benchmark] [--samples N]
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main execution pipeline."""

    parser = argparse.ArgumentParser(
        description="Complete High-Performance Training Pipeline"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (100 samples)"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark first"
    )
    parser.add_argument(
        "--samples", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument(
        "--source", default="../ifs-cloud-analysis", help="Source directory"
    )
    parser.add_argument(
        "--parser", default="plsql_parser.exe", help="AST parser executable"
    )
    parser.add_argument(
        "--skip-vllm", action="store_true", help="Skip vLLM even if available"
    )

    args = parser.parse_args()

    print("🚀 COMPLETE HIGH-PERFORMANCE TRAINING PIPELINE")
    print("=" * 60)

    if args.test:
        args.samples = 100
        print("🧪 Running in TEST MODE (100 samples)")

    # Step 1: Optional benchmark
    if args.benchmark:
        try:
            print("\n📊 Running performance benchmark...")
            from performance_benchmark import run_benchmark_suite

            await run_benchmark_suite()

            # Load recommendations
            if Path("benchmark_results.json").exists():
                with open("benchmark_results.json", "r") as f:
                    recommendations = json.load(f)

                print(
                    f"\n💡 Recommended configuration: {recommendations.get('best_config', 'unknown')}"
                )

                # Ask user if they want to continue
                if not args.test:
                    response = input("\nContinue with full processing? (y/N): ")
                    if response.lower() != "y":
                        print("Benchmark complete. Exiting.")
                        return
        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            print("Continuing without benchmark...")

    # Step 2: Check system and dependencies
    print("\n🔍 Checking system configuration...")
    system_check = await check_system_dependencies()

    # Step 3: Configure pipeline based on system
    use_vllm = system_check.get("vllm_available", False) and not args.skip_vllm
    has_parser = system_check.get("parser_available", False)

    print(f"   CPU cores: {system_check.get('cpu_cores', 'unknown')}")
    print(f"   RAM: {system_check.get('ram_gb', 'unknown')}GB")
    print(f"   GPU: {'Yes' if system_check.get('gpu_available') else 'No'}")
    print(f"   AST parser: {'Yes' if has_parser else 'No (fallback)'}")
    print(f"   vLLM: {'Yes' if use_vllm else 'No'}")

    # Step 4: Run parallel processing pipeline
    print(f"\n⚡ Starting parallel processing for {args.samples} samples...")

    try:
        from parallel_extraction_pipeline import ParallelExtractionPipeline

        pipeline = ParallelExtractionPipeline(
            source_directory=args.source,
            parser_executable=args.parser,
            target_samples=args.samples,
            use_vllm=use_vllm,
        )

        # Execute pipeline
        summary = await pipeline.generate_training_samples()

        # Step 5: Show results
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        perf = summary["pipeline_performance"]
        print(f"📊 Files processed: {perf['parsed_files']}/{perf['total_files']}")
        print(f"🔧 Functions extracted: {perf['processed_functions']}")
        print(f"📝 Summaries generated: {perf['summarized_functions']}")
        print(f"⏱️ Total time: {perf['total_time']:.1f}s")
        print(f"🚀 Throughput: {perf['throughput_files_per_sec']:.2f} files/sec")

        # Step 6: Quality check
        await quality_check_samples("parallel_training_samples.jsonl")

        print(f"\n📁 Output files:")
        print(f"   • Training samples: parallel_training_samples.jsonl")
        print(f"   • Performance report: parallel_pipeline_report.json")
        print(f"   • Processing log: parallel_pipeline.log")

        print(f"\n✅ Ready for UnixCoder fine-tuning!")

    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


async def check_system_dependencies():
    """Check available system resources and dependencies."""

    import multiprocessing as mp
    import psutil

    system_info = {
        "cpu_cores": mp.cpu_count(),
        "ram_gb": psutil.virtual_memory().total // 1024**3,
        "gpu_available": False,
        "vllm_available": False,
        "parser_available": False,
    }

    # Check GPU
    try:
        import torch

        system_info["gpu_available"] = torch.cuda.is_available()
        if system_info["gpu_available"]:
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Check vLLM
    try:
        import vllm

        system_info["vllm_available"] = True
    except ImportError:
        pass

    # Check AST parser
    system_info["parser_available"] = Path("plsql_parser.exe").exists()

    return system_info


async def quality_check_samples(sample_file: str):
    """Quick quality check on generated samples."""

    if not Path(sample_file).exists():
        print("⚠️ Sample file not found for quality check")
        return

    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            samples = [
                json.loads(line) for line in f.readlines()[:10]
            ]  # Check first 10

        if not samples:
            print("⚠️ No samples found in output file")
            return

        print("\n🔍 Quality Check (first 10 samples):")

        # Check UnixCoder compatibility
        unixcoder_compatible = 0
        has_summaries = 0
        avg_complexity = 0

        for sample in samples:
            code_length = len(sample.get("code", ""))
            if code_length <= 2000:  # UnixCoder limit
                unixcoder_compatible += 1

            if sample.get("summary"):
                has_summaries += 1

            complexity = sample.get("context", {}).get("complexity_metrics", {})
            avg_complexity += complexity.get("cyclomatic_complexity", 0)

        avg_complexity /= len(samples)

        print(
            f"   ✅ UnixCoder compatible: {unixcoder_compatible}/{len(samples)} ({unixcoder_compatible/len(samples):.1%})"
        )
        print(
            f"   📝 Has summaries: {has_summaries}/{len(samples)} ({has_summaries/len(samples):.1%})"
        )
        print(f"   🧮 Avg complexity: {avg_complexity:.1f}")

        # Show sample
        if samples:
            sample = samples[0]
            print(f"\n📋 Sample preview:")
            print(
                f"   Function: {sample.get('context', {}).get('function_name', 'unknown')}"
            )
            print(f"   Code length: {len(sample.get('code', ''))} chars")
            print(f"   Summary: {sample.get('summary', 'No summary')[:100]}...")

    except Exception as e:
        print(f"⚠️ Quality check failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
