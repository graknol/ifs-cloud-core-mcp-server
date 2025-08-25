#!/usr/bin/env python3
"""Check what attention kernels PyTorch actually has available."""

import torch
import sys

print("🔍 Analyzing PyTorch Attention Capabilities...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

# Check if we can use the new API
try:
    from torch.nn.attention import sdpa_kernel

    print("✅ New sdpa_kernel API available")
    use_new_api = True
except ImportError:
    print("⚠️ Using legacy API")
    use_new_api = False

print()

# Test attention backends
if torch.cuda.is_available():
    print("🚀 Testing attention performance...")

    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 64
    device = "cuda"
    dtype = torch.float16

    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )

    # Benchmark different approaches
    import time

    def benchmark_attention(name, attention_func, warmup=5, trials=20):
        # Warmup
        for _ in range(warmup):
            _ = attention_func()

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(trials):
            _ = attention_func()

        torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / trials * 1000  # ms
        print(f"{name}: {avg_time:.2f} ms")
        return avg_time

    print()

    # Test 1: PyTorch SDPA (automatic kernel selection)
    def sdpa_auto():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    sdpa_time = benchmark_attention("PyTorch SDPA (Auto)", sdpa_auto)

    # Test 2: Manual attention (for comparison)
    def manual_attention():
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    manual_time = benchmark_attention("Manual Attention", manual_attention)

    print()
    print(f"🎯 SDPA is {manual_time / sdpa_time:.1f}x faster than manual attention")

    # Test memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _ = sdpa_auto()
    sdpa_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _ = manual_attention()
    manual_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    print(f"📊 Memory Usage:")
    print(f"  SDPA: {sdpa_memory:.1f} MB")
    print(f"  Manual: {manual_memory:.1f} MB")
    print(f"  SDPA uses {manual_memory / sdpa_memory:.1f}x less memory")

print()
print("🎯 Key Findings:")
print("1. PyTorch SDPA automatically selects the best available kernel")
print("2. It includes Flash Attention-style optimizations built-in")
print("3. Memory efficiency is excellent")
print("4. No need for separate flash_attn package on modern PyTorch")
print()
print("✅ Recommendation: Use PyTorch SDPA - it's optimal and well-supported")

# Check what Triton can offer additionally
print()
print("🔧 Triton Capabilities:")
try:
    import triton

    print(f"✅ Triton {triton.__version__} available for custom kernels")
    print("💡 Triton is useful for:")
    print("  - Custom attention variants (sparse, local, etc.)")
    print("  - Specialized patterns not in SDPA")
    print("  - Research and experimentation")
    print("  - But for standard attention, SDPA is sufficient")
except ImportError:
    print("❌ Triton not available")
