#!/usr/bin/env python3
"""
RTX 5070 Ti Batch Size Optimization Benchmark

This script finds the optimal batch size for your RTX 5070 Ti by:
1. Testing increasing batch sizes until GPU memory limit
2. Measuring throughput and VRAM usage for each batch size
3. Finding the sweet spot for maximum performance
4. Providing recommendations for different use cases
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Tuple, Optional
import torch
import gc
from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchSizeOptimizer:
    """Find optimal batch size for RTX 5070 Ti."""

    def __init__(self):
        self.optimizer = None
        self.test_prompts = self._generate_test_prompts()
        self.results = []

        # RTX 5070 Ti specs
        self.gpu_memory_gb = 15.9
        self.safety_margin_gb = 2.0  # Keep 2GB free for system
        self.max_safe_memory_gb = self.gpu_memory_gb - self.safety_margin_gb

    def _generate_test_prompts(self) -> List[str]:
        """Generate realistic test prompts for benchmarking."""

        # Various PL/SQL function types and complexities
        templates = [
            # Simple getter function
            """# Summarize PL/SQL Function
Function: Get_{item}
```plsql
FUNCTION Get_{item}(id_ IN NUMBER) RETURN VARCHAR2 IS
   result_ VARCHAR2(100);
BEGIN
   SELECT {item}_name INTO result_
   FROM {table}_tab
   WHERE id = id_;
   RETURN result_;
EXCEPTION
   WHEN NO_DATA_FOUND THEN
      RETURN NULL;
END Get_{item};
```
Summary:""",
            # Complex validation function
            """# Summarize PL/SQL Function
Function: Validate_{entity}_Data
```plsql
PROCEDURE Validate_{entity}_Data(data_ IN {entity}_REC) IS
   count_ NUMBER;
   status_ VARCHAR2(20);
BEGIN
   -- Check mandatory fields
   IF data_.name IS NULL THEN
      Error_SYS.Record_General('{entity}_API', 'Name cannot be null');
   END IF;
   
   -- Check duplicates
   SELECT COUNT(*) INTO count_
   FROM {entity}_tab
   WHERE name = data_.name
   AND id != NVL(data_.id, -1);
   
   IF count_ > 0 THEN
      Error_SYS.Record_General('{entity}_API', 'Duplicate name');
   END IF;
   
   -- Validate status
   SELECT status INTO status_
   FROM status_types
   WHERE code = data_.status_code;
   
EXCEPTION
   WHEN OTHERS THEN
      Error_SYS.Record_General('{entity}_API', SQLERRM);
END Validate_{entity}_Data;
```
Summary:""",
            # Business logic function
            """# Summarize PL/SQL Function
Function: Calculate_{metric}
```plsql
FUNCTION Calculate_{metric}(period_start_ IN DATE, period_end_ IN DATE) RETURN NUMBER IS
   total_ NUMBER := 0;
   factor_ NUMBER;
BEGIN
   SELECT SUM(amount * multiplier) INTO total_
   FROM transaction_data
   WHERE transaction_date BETWEEN period_start_ AND period_end_
   AND status = 'CONFIRMED';
   
   -- Apply business rules
   IF total_ > 10000 THEN
      factor_ := 0.95; -- Volume discount
   ELSIF total_ > 5000 THEN
      factor_ := 0.97;
   ELSE
      factor_ := 1.0;
   END IF;
   
   total_ := total_ * factor_;
   
   -- Log calculation
   INSERT INTO calculation_log
   (period_start, period_end, result, calculated_at)
   VALUES (period_start_, period_end_, total_, SYSDATE);
   
   RETURN ROUND(total_, 2);
END Calculate_{metric};
```
Summary:""",
        ]

        # Generate varied prompts
        prompts = []
        entities = ["Customer", "Order", "Product", "Invoice", "Payment"]
        items = ["customer", "product", "order", "invoice"]
        tables = ["customer", "product", "order", "invoice"]
        metrics = ["Revenue", "Profit", "Discount", "Tax", "Total"]

        for i in range(200):  # Generate 200 varied prompts
            template = templates[i % len(templates)]

            if "Get_{item}" in template:
                item = items[i % len(items)]
                table = tables[i % len(tables)]
                prompt = template.format(item=item, table=table)
            elif "Validate_{entity}" in template:
                entity = entities[i % len(entities)]
                prompt = template.format(entity=entity)
            elif "Calculate_{metric}" in template:
                metric = metrics[i % len(metrics)]
                prompt = template.format(metric=metric)
            else:
                prompt = template

            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} test prompts")
        return prompts

    async def initialize_optimizer(self) -> bool:
        """Initialize the RTX 5070 Ti optimizer."""
        try:
            self.optimizer = RTX5070TiPyTorchOptimizer()
            success = await self.optimizer.initialize_model()

            if success:
                logger.info("✅ RTX 5070 Ti optimizer initialized")
                return True
            else:
                logger.error("❌ Optimizer initialization failed")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to initialize optimizer: {e}")
            return False

    async def benchmark_batch_size(
        self, batch_size: int, num_samples: int = 50
    ) -> Optional[Dict]:
        """Benchmark a specific batch size."""

        if not self.optimizer:
            logger.error("Optimizer not initialized")
            return None

        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Get baseline memory (use reserved memory for more accurate tracking)
            baseline_memory = torch.cuda.memory_reserved() / 1024**3

            # Select test samples
            test_samples = self.test_prompts[:num_samples]
            function_names = [f"TestFunc_{i}" for i in range(num_samples)]

            # Warmup run (don't count in timing)
            if batch_size <= 8:
                warmup_samples = test_samples[: min(4, len(test_samples))]
                warmup_names = function_names[: min(4, len(function_names))]
                await self.optimizer.process_batch_optimized(
                    warmup_samples, warmup_names
                )
                torch.cuda.empty_cache()

            # Benchmark run
            start_time = time.time()
            peak_memory = baseline_memory
            successful_batches = 0
            total_processed = 0

            # Process in specified batch sizes
            for batch_start in range(0, len(test_samples), batch_size):
                batch_samples = test_samples[batch_start : batch_start + batch_size]
                batch_names = function_names[batch_start : batch_start + batch_size]

                try:
                    # Process batch
                    results = await self.optimizer.process_batch_optimized(
                        batch_samples, batch_names
                    )

                    # Track memory usage (use reserved memory for better tracking)
                    current_memory = torch.cuda.memory_reserved() / 1024**3
                    peak_memory = max(peak_memory, current_memory)

                    successful_batches += 1
                    total_processed += len(batch_samples)

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"⚠️ OOM at batch size {batch_size}, batch {successful_batches + 1}"
                    )
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Error at batch size {batch_size}: {e}")
                    break

            end_time = time.time()

            # Calculate metrics
            if total_processed > 0:
                total_time = end_time - start_time
                throughput = total_processed / total_time
                memory_used = peak_memory - baseline_memory
                memory_efficiency = (
                    total_processed / memory_used if memory_used > 0 else 0
                )

                result = {
                    "batch_size": batch_size,
                    "samples_processed": total_processed,
                    "successful_batches": successful_batches,
                    "total_time": total_time,
                    "throughput_samples_sec": throughput,
                    "baseline_memory_gb": baseline_memory,
                    "peak_memory_gb": peak_memory,
                    "memory_used_gb": memory_used,
                    "memory_efficiency": memory_efficiency,
                    "gpu_utilization_percent": (peak_memory / self.gpu_memory_gb) * 100,
                    "success_rate": total_processed / num_samples,
                    "avg_batch_time": (
                        total_time / successful_batches if successful_batches > 0 else 0
                    ),
                }

                logger.info(
                    f"✅ Batch {batch_size}: {throughput:.1f} samples/sec, "
                    f"{memory_used:.2f}GB VRAM, {result['gpu_utilization_percent']:.1f}% GPU"
                )

                return result
            else:
                logger.error(f"❌ Batch size {batch_size} failed completely")
                return None

        except Exception as e:
            logger.error(f"❌ Benchmark failed for batch size {batch_size}: {e}")
            return None

    async def find_optimal_batch_sizes(self) -> Dict:
        """Find optimal batch sizes for different scenarios."""

        logger.info("🔍 Finding optimal batch sizes for RTX 5070 Ti...")

        # Test batch sizes from small to large
        batch_sizes_to_test = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            160,
            192,
            224,
            256,
        ]

        results = []
        max_successful_batch = 0

        for batch_size in batch_sizes_to_test:
            logger.info(f"🧪 Testing batch size: {batch_size}")

            result = await self.benchmark_batch_size(batch_size, num_samples=100)

            if result and result["success_rate"] > 0.8:  # At least 80% success
                results.append(result)
                max_successful_batch = batch_size
            else:
                logger.warning(
                    f"⚠️ Batch size {batch_size} failed or had low success rate"
                )
                # Try a few more smaller sizes around the failure point
                if batch_size > 32:
                    break

        # Find optimal configurations
        if not results:
            logger.error("❌ No successful batch sizes found")
            return {}

        # Sort by throughput
        results_by_throughput = sorted(
            results, key=lambda r: r["throughput_samples_sec"], reverse=True
        )
        results_by_efficiency = sorted(
            results, key=lambda r: r["memory_efficiency"], reverse=True
        )
        results_by_memory = sorted(results, key=lambda r: r["memory_used_gb"])

        analysis = {
            "all_results": results,
            "max_successful_batch_size": max_successful_batch,
            "optimal_configs": {
                "maximum_throughput": {
                    "config": results_by_throughput[0],
                    "description": "Best for maximum speed processing",
                },
                "memory_efficient": {
                    "config": results_by_efficiency[0],
                    "description": "Best samples per GB of VRAM",
                },
                "conservative": {
                    "config": results_by_memory[0],
                    "description": "Lowest memory usage, safe for long runs",
                },
                "balanced": {
                    "config": self._find_balanced_config(results),
                    "description": "Good balance of speed and memory usage",
                },
            },
            "recommendations": self._generate_recommendations(
                results, max_successful_batch
            ),
        }

        return analysis

    def _find_balanced_config(self, results: List[Dict]) -> Dict:
        """Find balanced configuration between throughput and memory."""

        if not results:
            return {}

        # Score each configuration (higher is better)
        scored_results = []

        max_throughput = max(r["throughput_samples_sec"] for r in results)

        # Handle case where memory_used_gb might be 0 or very small
        memory_values = [
            r["memory_used_gb"] for r in results if r["memory_used_gb"] > 0
        ]
        min_memory = min(memory_values) if memory_values else 0.01

        for result in results:
            throughput_score = result["throughput_samples_sec"] / max_throughput

            # Use peak memory for scoring if memory_used_gb is too small
            memory_to_use = max(result["memory_used_gb"], 0.01)
            memory_score = min_memory / memory_to_use

            # Weight throughput more heavily since memory usage is consistent
            balanced_score = (throughput_score * 0.7) + (memory_score * 0.3)

            scored_results.append((balanced_score, result))

        # Return the highest scoring configuration
        return max(scored_results, key=lambda x: x[0])[1]

    def _generate_recommendations(
        self, results: List[Dict], max_batch: int
    ) -> List[str]:
        """Generate specific recommendations."""

        recommendations = []

        if not results:
            recommendations.append("❌ No successful configurations found")
            return recommendations

        best_throughput = max(r["throughput_samples_sec"] for r in results)
        best_batch_size = max(
            r["batch_size"]
            for r in results
            if r["throughput_samples_sec"] == best_throughput
        )

        # Memory usage analysis
        high_memory_configs = [r for r in results if r["memory_used_gb"] > 8]
        safe_memory_configs = [r for r in results if r["memory_used_gb"] <= 6]

        recommendations.append(
            f"🏆 Maximum performance: batch size {best_batch_size} ({best_throughput:.1f} samples/sec)"
        )

        if safe_memory_configs:
            safe_best = max(
                safe_memory_configs, key=lambda r: r["throughput_samples_sec"]
            )
            recommendations.append(
                f"🛡️ Safe mode: batch size {safe_best['batch_size']} "
                f"({safe_best['throughput_samples_sec']:.1f} samples/sec, "
                f"{safe_best['memory_used_gb']:.1f}GB)"
            )

        if max_batch >= 64:
            recommendations.append(
                "✅ RTX 5070 Ti can handle large batches effectively"
            )
        elif max_batch >= 32:
            recommendations.append("✅ RTX 5070 Ti performs well with medium batches")
        else:
            recommendations.append(
                "⚠️ Consider reducing model precision or sequence length"
            )

        # Memory efficiency insights
        avg_memory_efficiency = sum(r["memory_efficiency"] for r in results) / len(
            results
        )
        recommendations.append(
            f"📊 Average efficiency: {avg_memory_efficiency:.1f} samples/GB VRAM"
        )

        return recommendations


async def run_batch_optimization():
    """Run the complete batch size optimization benchmark."""

    print("🚀 RTX 5070 Ti Batch Size Optimization Benchmark")
    print("=" * 70)

    optimizer = BatchSizeOptimizer()

    # Initialize
    if not await optimizer.initialize_optimizer():
        print("❌ Failed to initialize optimizer")
        return

    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"💾 VRAM: {optimizer.gpu_memory_gb}GB (using {optimizer.max_safe_memory_gb}GB max)"
    )
    print(f"📝 Test samples: {len(optimizer.test_prompts)}")

    # Run optimization
    analysis = await optimizer.find_optimal_batch_sizes()

    if not analysis:
        print("❌ Optimization failed")
        return

    # Display results
    print("\n" + "=" * 70)
    print("📊 BATCH SIZE OPTIMIZATION RESULTS")
    print("=" * 70)

    # Results table
    print(
        f"\n{'Batch Size':<12} {'Throughput':<15} {'VRAM Used':<12} {'GPU %':<8} {'Efficiency':<12}"
    )
    print("-" * 70)

    for result in analysis["all_results"]:
        print(
            f"{result['batch_size']:<12} "
            f"{result['throughput_samples_sec']:<15.1f} "
            f"{result['memory_used_gb']:<12.2f} "
            f"{result['gpu_utilization_percent']:<8.1f} "
            f"{result['memory_efficiency']:<12.1f}"
        )

    # Optimal configurations
    print(f"\n🏆 OPTIMAL CONFIGURATIONS")
    print("-" * 40)

    for config_name, config_data in analysis["optimal_configs"].items():
        config = config_data["config"]
        if config:  # Check if config exists
            print(f"\n📋 {config_name.replace('_', ' ').title()}:")
            print(f"   Batch size: {config['batch_size']}")
            print(f"   Throughput: {config['throughput_samples_sec']:.1f} samples/sec")
            print(
                f"   VRAM usage: {config['memory_used_gb']:.2f}GB ({config['gpu_utilization_percent']:.1f}%)"
            )
            print(f"   Description: {config_data['description']}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    for rec in analysis["recommendations"]:
        print(f"   {rec}")

    # Performance projection
    best_config = analysis["optimal_configs"]["maximum_throughput"]["config"]
    if best_config:
        time_for_10k = 10000 / best_config["throughput_samples_sec"]
        print(f"\n⏱️ PERFORMANCE PROJECTION")
        print(f"   Time for 10,000 samples: {time_for_10k/60:.1f} minutes")
        print(f"   VRAM required: {best_config['memory_used_gb']:.2f}GB")
        print(f"   GPU utilization: {best_config['gpu_utilization_percent']:.1f}%")

    # Save detailed results
    with open("batch_optimization_results.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n📁 Detailed results saved to: batch_optimization_results.json")
    print(f"🎉 Optimization complete!")


if __name__ == "__main__":
    asyncio.run(run_batch_optimization())
