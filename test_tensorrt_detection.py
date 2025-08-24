#!/usr/bin/env python3
"""
Test TensorRT Detection in RTX Optimizer
"""

from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer


def test_tensorrt_detection():
    """Test TensorRT detection in the RTX optimizer."""

    print("🔍 Testing TensorRT Detection in RTX Optimizer")
    print("=" * 50)

    # This will trigger the optimization status report
    opt = RTX5070TiPyTorchOptimizer()

    print("\n✨ Optimizer initialized with TensorRT detection!")


if __name__ == "__main__":
    test_tensorrt_detection()
