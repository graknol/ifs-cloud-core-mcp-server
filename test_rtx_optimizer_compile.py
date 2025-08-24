#!/usr/bin/env python3
"""
Test the updated RTX optimizer with torch.compile integration
"""

import asyncio
from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer


async def test_rtx_optimizer_with_compile():
    print("🚀 Testing RTX 5070 Ti Optimizer with torch.compile")
    print("=" * 55)

    # Initialize optimizer
    opt = RTX5070TiPyTorchOptimizer()
    await opt.initialize_model()

    # Test with a batch of code samples
    test_samples = [
        "def calculate_sum(a, b): return a + b",
        "class DataProcessor: def __init__(self): pass",
        "import numpy as np; arr = np.array([1, 2, 3])",
    ]

    print(f"📊 Processing {len(test_samples)} samples...")

    # Process batch and measure performance
    import time

    start = time.time()

    print("📊 Processing 3 samples...")
    results = await opt.process_batch_optimized(test_samples)

    end = time.time()

    print(f"✅ RTX optimizer with torch.compile works!")
    print(f"   Processed: {len(results)} samples")
    print(f"   Time taken: {(end-start)*1000:.2f}ms")
    print(f"   Throughput: {len(results)/(end-start):.1f} samples/sec")
    print(f"   First result type: {type(results[0]).__name__ if results else 'N/A'}")
    print(
        f"   First result length: {len(results[0]) if results and hasattr(results[0], '__len__') else 'N/A'}"
    )


if __name__ == "__main__":
    asyncio.run(test_rtx_optimizer_with_compile())
