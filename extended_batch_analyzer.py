#!/usr/bin/env python3
"""
Extended Batch Size Analysis - Testing Very Large Batch Sizes

This script tests batch sizes well beyond the initial optimal range to find
the exact point where throughput starts to decrease and understand why.
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


class ExtendedBatchAnalyzer:
    """Test very large batch sizes to find performance degradation points."""

    def __init__(self):
        self.optimizer = None
        # Generate a larger set of test prompts for big batches
        self.test_prompts = self._generate_large_test_set()

    def _generate_large_test_set(self) -> List[str]:
        """Generate a large set of varied test prompts."""

        base_prompt = """# Summarize PL/SQL Function
Function: Process_{entity}_{operation}
```plsql
PROCEDURE Process_{entity}_{operation}(
   id_ IN NUMBER,
   data_ IN {entity}_REC,
   result_ OUT VARCHAR2
) IS
   temp_value_ VARCHAR2(100);
   count_ NUMBER;
   status_ VARCHAR2(20);
BEGIN
   -- Validate input
   IF id_ IS NULL THEN
      Error_SYS.Record_General('{entity}_API', 'ID cannot be null');
   END IF;
   
   -- Check existence
   SELECT COUNT(*) INTO count_
   FROM {table}_tab 
   WHERE id = id_;
   
   IF count_ = 0 THEN
      Error_SYS.Record_General('{entity}_API', 'Record not found');
   END IF;
   
   -- Process data
   SELECT {field}_value INTO temp_value_
   FROM {table}_tab
   WHERE id = id_;
   
   -- Apply business logic
   IF temp_value_ = 'ACTIVE' THEN
      status_ := 'PROCESSED';
   ELSIF temp_value_ = 'PENDING' THEN  
      status_ := 'IN_PROGRESS';
   ELSE
      status_ := 'ERROR';
   END IF;
   
   -- Update record
   UPDATE {table}_tab
   SET status = status_,
       last_modified = SYSDATE
   WHERE id = id_;
   
   result_ := status_;
   
EXCEPTION
   WHEN OTHERS THEN
      Error_SYS.Record_General('{entity}_API', SQLERRM);
      ROLLBACK;
END Process_{entity}_{operation};
```
Summary:"""

        # Generate varied prompts
        entities = [
            "Customer",
            "Order",
            "Product",
            "Invoice",
            "Payment",
            "Shipment",
            "Inventory",
            "Account",
        ]
        operations = [
            "Validation",
            "Processing",
            "Update",
            "Creation",
            "Deletion",
            "Approval",
            "Completion",
        ]
        tables = [
            "customer",
            "order",
            "product",
            "invoice",
            "payment",
            "shipment",
            "inventory",
            "account",
        ]
        fields = [
            "status",
            "state",
            "condition",
            "flag",
            "type",
            "category",
            "level",
            "priority",
        ]

        prompts = []
        for i in range(2000):  # Generate 2000 prompts for large batch testing
            entity = entities[i % len(entities)]
            operation = operations[i % len(operations)]
            table = tables[i % len(tables)]
            field = fields[i % len(fields)]

            prompt = base_prompt.format(
                entity=entity, operation=operation, table=table, field=field
            )
            prompts.append(prompt)

        logger.info(
            f"Generated {len(prompts)} test prompts for extended batch analysis"
        )
        return prompts

    async def initialize_optimizer(self) -> bool:
        """Initialize the RTX 5070 Ti optimizer."""
        try:
            self.optimizer = RTX5070TiPyTorchOptimizer()
            success = await self.optimizer.initialize_model()

            if success:
                logger.info("✅ RTX 5070 Ti optimizer initialized for extended testing")
                return True
            else:
                logger.error("❌ Optimizer initialization failed")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to initialize optimizer: {e}")
            return False

    async def test_large_batch_size(
        self, batch_size: int, num_samples: int = 500
    ) -> Optional[Dict]:
        """Test a specific large batch size with more samples."""

        if not self.optimizer:
            logger.error("Optimizer not initialized")
            return None

        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Get detailed memory baseline
            baseline_allocated = torch.cuda.memory_allocated() / 1024**3
            baseline_reserved = torch.cuda.memory_reserved() / 1024**3
            max_memory = torch.cuda.max_memory_reserved() / 1024**3

            logger.info(
                f"🧪 Testing batch size {batch_size} with {num_samples} samples"
            )
            logger.info(
                f"📊 Memory baseline: {baseline_allocated:.3f}GB allocated, {baseline_reserved:.3f}GB reserved"
            )

            # Select test samples
            test_samples = self.test_prompts[:num_samples]
            function_names = [f"TestFunc_{i}" for i in range(num_samples)]

            # Warmup (smaller batch to ensure GPU is ready)
            if batch_size >= 64:
                warmup_size = min(16, batch_size // 4)
                warmup_samples = test_samples[:warmup_size]
                warmup_names = function_names[:warmup_size]
                await self.optimizer.process_batch_optimized(
                    warmup_samples, warmup_names
                )
                torch.cuda.empty_cache()
                logger.info(f"🔥 Warmup completed with batch size {warmup_size}")

            # Track detailed metrics
            start_time = time.time()
            peak_allocated = baseline_allocated
            peak_reserved = baseline_reserved
            successful_batches = 0
            total_processed = 0
            batch_times = []

            # Process in specified batch sizes
            for batch_start in range(0, len(test_samples), batch_size):
                batch_samples = test_samples[batch_start : batch_start + batch_size]
                batch_names = function_names[batch_start : batch_start + batch_size]
                actual_batch_size = len(batch_samples)

                try:
                    # Track batch processing time
                    batch_start_time = time.time()

                    # Process batch
                    results = await self.optimizer.process_batch_optimized(
                        batch_samples, batch_names
                    )

                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)

                    # Track memory usage
                    current_allocated = torch.cuda.memory_allocated() / 1024**3
                    current_reserved = torch.cuda.memory_reserved() / 1024**3
                    peak_allocated = max(peak_allocated, current_allocated)
                    peak_reserved = max(peak_reserved, current_reserved)

                    successful_batches += 1
                    total_processed += actual_batch_size

                    # Log progress for large batches
                    if batch_size >= 256 or successful_batches % 5 == 0:
                        throughput = actual_batch_size / batch_time
                        logger.info(
                            f"⚡ Batch {successful_batches}: {actual_batch_size} samples in {batch_time:.2f}s ({throughput:.1f} samples/sec)"
                        )

                except torch.cuda.OutOfMemoryError as oom:
                    logger.warning(
                        f"💥 OOM at batch {successful_batches + 1} with batch size {batch_size}"
                    )
                    logger.warning(
                        f"📊 Memory at OOM: {torch.cuda.memory_allocated() / 1024**3:.3f}GB allocated"
                    )
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Error at batch {successful_batches + 1}: {e}")
                    break

            end_time = time.time()

            # Calculate comprehensive metrics
            if total_processed > 0:
                total_time = end_time - start_time
                overall_throughput = total_processed / total_time
                memory_allocated_used = peak_allocated - baseline_allocated
                memory_reserved_used = peak_reserved - baseline_reserved

                # Calculate batch-level statistics
                avg_batch_time = (
                    sum(batch_times) / len(batch_times) if batch_times else 0
                )
                min_batch_time = min(batch_times) if batch_times else 0
                max_batch_time = max(batch_times) if batch_times else 0

                result = {
                    "batch_size": batch_size,
                    "samples_processed": total_processed,
                    "successful_batches": successful_batches,
                    "total_time": total_time,
                    "throughput_samples_sec": overall_throughput,
                    # Memory metrics
                    "baseline_allocated_gb": baseline_allocated,
                    "baseline_reserved_gb": baseline_reserved,
                    "peak_allocated_gb": peak_allocated,
                    "peak_reserved_gb": peak_reserved,
                    "memory_allocated_used_gb": memory_allocated_used,
                    "memory_reserved_used_gb": memory_reserved_used,
                    "max_memory_gb": max_memory,
                    # Performance metrics
                    "avg_batch_time": avg_batch_time,
                    "min_batch_time": min_batch_time,
                    "max_batch_time": max_batch_time,
                    "batch_time_variance": (
                        max_batch_time - min_batch_time if batch_times else 0
                    ),
                    # Efficiency metrics
                    "gpu_utilization_percent": (peak_reserved / 15.9) * 100,
                    "samples_per_gb_allocated": (
                        total_processed / memory_allocated_used
                        if memory_allocated_used > 0
                        else float("inf")
                    ),
                    "samples_per_gb_reserved": (
                        total_processed / memory_reserved_used
                        if memory_reserved_used > 0
                        else float("inf")
                    ),
                    "success_rate": total_processed / num_samples,
                    # Batch efficiency
                    "avg_samples_per_batch": (
                        total_processed / successful_batches
                        if successful_batches > 0
                        else 0
                    ),
                    "batch_processing_efficiency": (
                        batch_size / avg_batch_time if avg_batch_time > 0 else 0
                    ),
                }

                logger.info(
                    f"✅ Batch {batch_size}: {overall_throughput:.1f} samples/sec, "
                    f"allocated: {memory_allocated_used:.3f}GB, reserved: {memory_reserved_used:.3f}GB"
                )

                return result
            else:
                logger.error(f"❌ Batch size {batch_size} failed completely")
                return None

        except Exception as e:
            logger.error(
                f"❌ Extended benchmark failed for batch size {batch_size}: {e}"
            )
            return None


async def run_extended_batch_analysis():
    """Run extended batch size analysis to find degradation points."""

    print("🔍 Extended Batch Size Analysis - Finding Performance Limits")
    print("=" * 80)

    analyzer = ExtendedBatchAnalyzer()

    # Initialize
    if not await analyzer.initialize_optimizer():
        print("❌ Failed to initialize optimizer")
        return

    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: 15.9GB available")
    print(f"📝 Test samples: {len(analyzer.test_prompts)}")

    # Test progressively larger batch sizes
    large_batch_sizes = [
        128,  # Known optimal
        256,  # 2x optimal
        384,  # 3x optimal
        512,  # 4x optimal
        768,  # 6x optimal
        1024,  # 8x optimal
        1536,  # 12x optimal
        2048,  # 16x optimal
        # Will continue until we hit memory limits or performance degradation
    ]

    results = []
    previous_throughput = 0

    print(f"\n🧪 Testing Large Batch Sizes...")
    print(
        f"{'Batch Size':<12} {'Throughput':<15} {'Allocated':<12} {'Reserved':<12} {'Efficiency':<12}"
    )
    print("-" * 80)

    for batch_size in large_batch_sizes:
        result = await analyzer.test_large_batch_size(batch_size, num_samples=1000)

        if result and result["success_rate"] > 0.8:
            results.append(result)

            print(
                f"{result['batch_size']:<12} "
                f"{result['throughput_samples_sec']:<15.1f} "
                f"{result['memory_allocated_used_gb']:<12.3f} "
                f"{result['memory_reserved_used_gb']:<12.3f} "
                f"{result['batch_processing_efficiency']:<12.1f}"
            )

            # Check for performance degradation
            if (
                previous_throughput > 0
                and result["throughput_samples_sec"] < previous_throughput * 0.95
            ):
                print(
                    f"\n⚠️ Performance degradation detected at batch size {batch_size}"
                )
                print(
                    f"   Throughput dropped from {previous_throughput:.1f} to {result['throughput_samples_sec']:.1f} samples/sec"
                )
                print(
                    f"   That's a {((previous_throughput - result['throughput_samples_sec']) / previous_throughput * 100):.1f}% decrease"
                )

                # Continue testing a few more sizes to confirm the trend

            previous_throughput = result["throughput_samples_sec"]

        else:
            print(f"{batch_size:<12} FAILED - OOM or low success rate")
            break

    # Analysis
    if results:
        print(f"\n📊 EXTENDED BATCH SIZE ANALYSIS RESULTS")
        print("=" * 60)

        # Find optimal and degradation points
        max_throughput = max(r["throughput_samples_sec"] for r in results)
        optimal_result = next(
            r for r in results if r["throughput_samples_sec"] == max_throughput
        )

        print(f"\n🏆 Peak Performance:")
        print(f"   Batch Size: {optimal_result['batch_size']}")
        print(
            f"   Throughput: {optimal_result['throughput_samples_sec']:.1f} samples/sec"
        )
        print(
            f"   Memory Used (allocated): {optimal_result['memory_allocated_used_gb']:.3f}GB"
        )
        print(
            f"   Memory Used (reserved): {optimal_result['memory_reserved_used_gb']:.3f}GB"
        )

        # Check for degradation pattern
        throughputs = [r["throughput_samples_sec"] for r in results]
        batch_sizes = [r["batch_size"] for r in results]

        # Find where performance starts declining
        peak_idx = throughputs.index(max_throughput)
        if peak_idx < len(throughputs) - 1:
            declining_results = results[peak_idx + 1 :]
            if declining_results:
                print(f"\n📉 Performance Degradation Pattern:")
                for i, result in enumerate(declining_results):
                    prev_throughput = results[peak_idx + i]["throughput_samples_sec"]
                    degradation = (
                        (prev_throughput - result["throughput_samples_sec"])
                        / prev_throughput
                    ) * 100
                    print(
                        f"   Batch {result['batch_size']}: {result['throughput_samples_sec']:.1f} samples/sec ({degradation:+.1f}%)"
                    )

        # Memory scaling analysis
        print(f"\n💾 Memory Scaling Analysis:")
        for result in results:
            mem_efficiency = result["samples_per_gb_reserved"]
            print(
                f"   Batch {result['batch_size']}: {result['memory_reserved_used_gb']:.3f}GB reserved, "
                f"{mem_efficiency:.0f} samples/GB"
            )

        # Save results
        with open("extended_batch_analysis.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "peak_performance": optimal_result,
                    "analysis_summary": {
                        "optimal_batch_size": optimal_result["batch_size"],
                        "peak_throughput": max_throughput,
                        "memory_efficiency_range": [
                            min(r["samples_per_gb_reserved"] for r in results),
                            max(r["samples_per_gb_reserved"] for r in results),
                        ],
                    },
                },
                f,
                indent=2,
            )

        print(f"\n📁 Detailed results saved to: extended_batch_analysis.json")

    print(f"\n🎉 Extended analysis complete!")


if __name__ == "__main__":
    asyncio.run(run_extended_batch_analysis())
