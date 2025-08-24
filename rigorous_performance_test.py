#!/usr/bin/env python3
"""
Rigorous RTX 5070 Ti Performance Test - Verify actual processing
"""

import asyncio
import time
import torch
import traceback
from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer


async def rigorous_performance_test():
    print("🔬 Rigorous RTX 5070 Ti Performance Test")
    print("=" * 60)
    print("Verifying actual processing is happening...")
    print()

    # Initialize optimizer
    opt = RTX5070TiPyTorchOptimizer()
    success = await opt.initialize_model()

    if not success:
        print("❌ Failed to initialize optimizer")
        return

    print(f"✅ Model initialized")
    print(f"   Batch size: {opt.optimal_batch_size}")
    print(f"   FP16 enabled: {opt.use_fp16}")
    print(f"   Device: {opt.device}")
    print(f"   Model dtype: {opt.model.dtype}")
    print()

    # Create diverse test samples that will definitely produce different outputs
    test_samples = [
        "def calculate_fibonacci(n): return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
        "SELECT customer_id, SUM(amount) FROM orders WHERE status = 'completed' GROUP BY customer_id ORDER BY SUM(amount) DESC",
        "function processUserData(users) { return users.filter(u => u.active).map(u => ({id: u.id, name: u.name})); }",
        "class DatabaseManager: def __init__(self): self.connection = None",
        "UPDATE products SET price = price * 1.1 WHERE category = 'electronics' AND stock > 0",
        "const fetchApiData = async (url) => { const response = await fetch(url); return response.json(); }",
        "CREATE TABLE users (id SERIAL PRIMARY KEY, username VARCHAR(50), email VARCHAR(100))",
        "import pandas as pd; df = pd.read_csv('data.csv'); result = df.groupby('category').sum()",
    ]

    print("🧪 Testing with small batch first to verify processing...")

    # Test with just 3 samples and inspect results
    small_batch = test_samples[:3]

    try:
        start_time = time.perf_counter()
        results = await opt.process_batch_optimized(small_batch)
        end_time = time.perf_counter()

        processing_time = end_time - start_time

        print(f"✅ Small batch processed successfully")
        print(f"   Samples: {len(small_batch)}")
        print(f"   Results: {len(results) if results else 0}")
        print(f"   Processing time: {processing_time*1000:.3f}ms")
        print(f"   Throughput: {len(results)/processing_time:.1f} samples/sec")
        print()

        # Inspect actual results
        print("📋 Result inspection:")
        if results:
            for i, result in enumerate(results[:3]):
                result_preview = (
                    str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                )
                print(f"   Sample {i+1}: {type(result).__name__} - {result_preview}")
        else:
            print("   ⚠️  No results returned!")
            return
        print()

    except Exception as e:
        print(f"❌ Error in small batch processing:")
        print(f"   Error: {str(e)}")
        print(f"   Full traceback:")
        traceback.print_exc()
        return

    # Test with progressively larger batches and time each step
    batch_sizes = [8, 16, 32, 64, 128]

    print("📊 Progressive batch size testing:")
    print("-" * 60)

    for batch_size in batch_sizes:
        # Create batch by repeating and modifying samples
        batch = []
        for i in range(batch_size):
            base_sample = test_samples[i % len(test_samples)]
            # Add variation to ensure different processing
            modified_sample = f"// Batch {batch_size} Sample {i+1}\n{base_sample}"
            batch.append(modified_sample)

        try:
            # Measure actual processing time
            torch.cuda.synchronize()  # Ensure GPU operations complete
            start_time = time.perf_counter()

            results = await opt.process_batch_optimized(batch)

            torch.cuda.synchronize()  # Ensure GPU operations complete
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            throughput = len(results) / processing_time if processing_time > 0 else 0

            # Validate results
            if len(results) != len(batch):
                print(
                    f"Batch {batch_size:3d}: ❌ Result count mismatch! Expected {len(batch)}, got {len(results)}"
                )
                continue

            # Check if results are actually different (not cached/identical)
            unique_results = len(set(str(r) for r in results))

            print(
                f"Batch {batch_size:3d}: {throughput:8.1f} samples/sec ({processing_time*1000:7.2f}ms) - {unique_results} unique results"
            )

            if unique_results == 1:
                print(f"         ⚠️  All results identical - possible caching or error!")

        except Exception as e:
            print(f"Batch {batch_size:3d}: ❌ FAILED - {str(e)}")

    print("-" * 60)

    # Final comprehensive test with actual timing
    print("\n🎯 Final Comprehensive Test:")
    print("-" * 40)

    # Use optimal batch size but actually time it properly
    large_batch_size = min(256, opt.optimal_batch_size)  # Conservative size
    large_batch = []

    for i in range(large_batch_size):
        base_sample = test_samples[i % len(test_samples)]
        # Ensure each sample is unique
        unique_sample = f"/* Test {i+1} - Timestamp {time.time()} */\n{base_sample}\n/* End sample {i+1} */"
        large_batch.append(unique_sample)

    print(f"Testing with {large_batch_size} unique samples...")

    try:
        # Clear any caches
        torch.cuda.empty_cache()

        # Multiple timed runs for accuracy
        runs = 3
        times = []

        for run in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            results = await opt.process_batch_optimized(large_batch)

            torch.cuda.synchronize()
            end = time.perf_counter()

            run_time = end - start
            times.append(run_time)

            print(
                f"   Run {run+1}: {run_time*1000:.2f}ms ({len(results)/run_time:.1f} samples/sec)"
            )

        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_throughput = large_batch_size / avg_time

        print(f"\n✅ VERIFIED RESULTS:")
        print(f"   Batch size: {large_batch_size}")
        print(f"   Average time: {avg_time*1000:.2f}ms")
        print(f"   Average throughput: {avg_throughput:.1f} samples/sec")
        print(f"   Results returned: {len(results)}")
        print(f"   Unique results: {len(set(str(r) for r in results))}")

        # Memory usage
        print(f"   GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

        # Validate processing actually happened
        if len(results) == large_batch_size:
            print("   ✅ All samples processed")
        else:
            print(f"   ⚠️  Sample count mismatch: {len(results)} vs {large_batch_size}")

        unique_count = len(set(str(r) for r in results))
        if unique_count > large_batch_size * 0.8:  # Expect most results to be unique
            print("   ✅ Results appear to be genuinely processed")
        else:
            print(f"   ⚠️  Only {unique_count} unique results - possible issue")

    except Exception as e:
        print(f"❌ Final test failed:")
        print(f"   Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(rigorous_performance_test())
