#!/usr/bin/env python3
"""Test Triton installation on Linux."""

try:
    import triton
    print(f"✓ Triton version: {triton.__version__}")
    print("✓ Triton successfully imported for Linux!")
except ImportError as e:
    print(f"✗ Failed to import triton: {e}")

# Also test that we still have Flash Attention
try:
    import flash_attn
    print(f"✓ Flash Attention version: {flash_attn.__version__}")
except ImportError as e:
    print(f"✗ Flash Attention not available: {e}")

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")
