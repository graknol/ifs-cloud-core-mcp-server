#!/usr/bin/env python3
"""
Corrected Large Batch Size Analysis

This script tests large batch sizes with proper methodology - using sufficient
samples to properly demonstrate batch size efficiency differences.
"""

import asyncio
import logging
import time
from typing import List, Dict
from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_large_batch_sizes_corrected():
    """Test large batch sizes with proper sample counts."""

    print("🔬 Corrected Large Batch Size Analysis")
    print("=" * 70)

    # Test larger batch sizes with sufficient samples
    batch_sizes_to_test = [64, 128, 256, 384, 512, 768, 1024]

    # Use 1000 samples - enough to show real batch efficiency
    total_samples = 1000

    try:
        optimizer = RTX5070TiPyTorchOptimizer()
        success = await optimizer.initialize_model()

        if not success:
            print("❌ Could not initialize optimizer")
            return

        print(f"🎮 GPU: {optimizer.device}")
        print(f"📊 Testing {total_samples} samples with large batch sizes")
        print(f"⏱️ This will take a few minutes for accurate measurements...\n")

        # Generate test data
        print("📝 Generating test samples...")
        test_prompts = []
        function_names = []

        for i in range(total_samples):
            prompt = f"""# Summarize PL/SQL Function
Function: Large_Batch_Test_{i}
```plsql
PROCEDURE Process_Large_Batch_{i}(
   id_ IN NUMBER,
   status_ OUT VARCHAR2
) IS
   count_ NUMBER;
   temp_val_ VARCHAR2(100);
BEGIN
   SELECT COUNT(*) INTO count_
   FROM test_table_{i % 10}
   WHERE id = id_;
   
   IF count_ > 0 THEN
      SELECT value INTO temp_val_
      FROM test_table_{i % 10}
      WHERE id = id_;
      
      status_ := 'PROCESSED_' || temp_val_;
   ELSE
      status_ := 'NOT_FOUND';
   END IF;
   
EXCEPTION
   WHEN OTHERS THEN
      status_ := 'ERROR';
END Process_Large_Batch_{i};
```
Summary:"""
            test_prompts.append(prompt)
            function_names.append(f"Large_Batch_Test_{i}")

        print(f"✅ Generated {len(test_prompts)} unique test samples\n")

        results = []

        print(
            f"{'Batch Size':<12} {'Throughput':<15} {'Time (s)':<10} {'Batches':<8} {'Avg/Batch':<10} {'Notes'}"
        )
        print("-" * 85)

        for batch_size in batch_sizes_to_test:
            print(f"🧪 Testing batch size {batch_size}...")

            start_time = time.time()
            total_processed = 0
            batch_count = 0
            batch_times = []

            # Process all samples in specified batch sizes
            for i in range(0, total_samples, batch_size):
                batch_prompts = test_prompts[i : i + batch_size]
                batch_names = function_names[i : i + batch_size]

                batch_start_time = time.time()
                try:
                    batch_results = await optimizer.process_batch_optimized(
                        batch_prompts, batch_names
                    )
                    batch_end_time = time.time()

                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)
                    total_processed += len(batch_prompts)
                    batch_count += 1

                    # Progress indicator for large batches
                    if batch_count % 5 == 0 or batch_size >= 512:
                        progress = (total_processed / total_samples) * 100
                        print(
                            f"   Progress: {progress:.0f}% ({total_processed}/{total_samples} samples)"
                        )

                except Exception as e:
                    print(f"   ❌ Batch failed at {total_processed} samples: {e}")
                    break

            end_time = time.time()
            total_time = end_time - start_time

            if total_processed > 0:
                throughput = total_processed / total_time
                avg_batch_time = (
                    sum(batch_times) / len(batch_times) if batch_times else 0
                )

                # Determine notes
                note = ""
                if batch_size == 128:
                    note = "🏆 Original optimal"
                elif throughput > 215:
                    note = "✅ High performance"
                elif total_processed < total_samples:
                    note = "⚠️ Incomplete"

                results.append(
                    {
                        "batch_size": batch_size,
                        "throughput": throughput,
                        "total_time": total_time,
                        "batch_count": batch_count,
                        "avg_batch_time": avg_batch_time,
                        "samples_processed": total_processed,
                    }
                )

                print(
                    f"{batch_size:<12} {throughput:<15.1f} {total_time:<10.1f} {batch_count:<8} {avg_batch_time:<10.2f} {note}"
                )
            else:
                print(f"{batch_size:<12} FAILED")
                break

            print()  # Spacing between tests

        # Analysis
        if results:
            print("\n📊 LARGE BATCH SIZE ANALYSIS RESULTS")
            print("=" * 60)

            # Find peak performance
            best_result = max(results, key=lambda r: r["throughput"])
            print(f"\n🏆 Peak Performance:")
            print(f"   Batch Size: {best_result['batch_size']}")
            print(f"   Throughput: {best_result['throughput']:.1f} samples/sec")
            print(f"   Total Time: {best_result['total_time']:.1f} seconds")
            print(f"   Batch Count: {best_result['batch_count']}")

            # Compare to batch 128 baseline
            baseline_128 = next((r for r in results if r["batch_size"] == 128), None)
            if baseline_128:
                print(f"\n📈 Comparison to Batch 128:")
                for result in results:
                    if result["batch_size"] != 128:
                        improvement = (
                            (result["throughput"] - baseline_128["throughput"])
                            / baseline_128["throughput"]
                        ) * 100
                        symbol = (
                            "📈"
                            if improvement > 0
                            else "📉" if improvement < -1 else "➡️"
                        )
                        print(
                            f"   Batch {result['batch_size']:>3}: {improvement:+5.1f}% ({result['throughput']:.1f} vs {baseline_128['throughput']:.1f} samples/sec) {symbol}"
                        )

            # Performance trends
            print(f"\n📊 Performance Trends:")
            prev_throughput = None
            for result in results:
                if prev_throughput:
                    change = result["throughput"] - prev_throughput
                    trend = "📈" if change > 1 else "📉" if change < -1 else "➡️"
                    print(
                        f"   {result['batch_size']:>3}: {change:+5.1f} samples/sec {trend}"
                    )
                else:
                    print(f"   {result['batch_size']:>3}: baseline")
                prev_throughput = result["throughput"]

            # Memory efficiency
            print(f"\n💾 Efficiency Analysis:")
            print(f"   Fewer batches = better GPU utilization")
            for result in results:
                efficiency_score = result["throughput"] / result["batch_count"]
                print(
                    f"   Batch {result['batch_size']:>3}: {efficiency_score:>6.1f} samples/sec per batch iteration"
                )

        print(
            f"\n🎯 This corrected analysis should show the true performance characteristics!"
        )

    except Exception as e:
        print(f"❌ Large batch analysis failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_large_batch_sizes_corrected())
