#!/usr/bin/env python3
"""
Test TensorRT Detection with Explicit Logging
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer


def test_tensorrt_with_logging():
    """Test TensorRT detection with proper logging setup."""

    print("🔍 Testing TensorRT Detection with Logging")
    print("=" * 50)

    # This will trigger the optimization status report with visible logs
    opt = RTX5070TiPyTorchOptimizer()

    print("\n✨ Optimizer initialized!")
    print("Check the logs above for TensorRT detection results")


if __name__ == "__main__":
    test_tensorrt_with_logging()
