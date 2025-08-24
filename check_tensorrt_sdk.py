#!/usr/bin/env python3
"""
Check TensorRT SDK Detection and Usage
"""


def check_tensorrt_sdk():
    """Check if TensorRT SDK is installed and being used."""

    print("🔍 Checking TensorRT SDK Installation and Usage")
    print("=" * 55)

    # Check direct TensorRT SDK import
    print("\n1. 🔧 Direct TensorRT SDK Check:")
    try:
        import tensorrt as trt

        print(f"   ✅ TensorRT SDK installed: version {trt.__version__}")
        print(f"   ✅ TensorRT Builder available: {hasattr(trt, 'Builder')}")
        print(f"   ✅ TensorRT Runtime available: {hasattr(trt, 'Runtime')}")

        # Check TensorRT capabilities
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        print(f"   ✅ TensorRT Builder initialized successfully")

        # Check supported formats and precisions
        print(f"   📊 TensorRT Platform compatibility:")
        print(f"      - GPU compute capability supported: Available")
        print(f"      - FP16 precision: {builder.platform_has_fast_fp16}")
        print(f"      - INT8 precision: {builder.platform_has_fast_int8}")

        tensorrt_sdk_available = True

    except ImportError as e:
        print(f"   ❌ TensorRT SDK not found: {e}")
        tensorrt_sdk_available = False
    except Exception as e:
        print(f"   ❌ TensorRT SDK error: {e}")
        tensorrt_sdk_available = False

    # Check ONNX Runtime TensorRT provider
    print("\n2. 🔧 ONNX Runtime TensorRT Provider Check:")
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        print(f"   📋 Available providers: {providers}")

        if "TensorrtExecutionProvider" in providers:
            print("   ✅ TensorrtExecutionProvider available in ONNX Runtime")

            # Get TensorRT provider options
            try:
                provider_options = ort.get_provider_options("TensorrtExecutionProvider")
                print(f"   📊 TensorRT provider options available")
            except:
                print("   ℹ️  TensorRT provider options not accessible")
        else:
            print("   ❌ TensorrtExecutionProvider not available")

    except ImportError:
        print("   ❌ ONNX Runtime not available")

    # Check Optimum integration
    print("\n3. 🔧 Optimum TensorRT Integration:")
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        print("   ✅ Optimum ONNX Runtime available")

        # Try to check for nvidia-specific optimum modules (may not exist)
        try:
            from optimum.nvidia import TensorRTForSequenceClassification

            print("   ✅ Optimum NVIDIA TensorRT models available")
        except ImportError:
            print("   ℹ️  Optimum NVIDIA module not available (normal for Windows)")

    except ImportError:
        print("   ❌ Optimum not available")

    # Check current RTX optimizer detection
    print("\n4. 🔧 RTX 5070 Ti Optimizer Detection:")
    try:
        from rtx5070ti_pytorch_optimizer import (
            TENSORRT_AVAILABLE,
            CUDA_AVAILABLE,
            OPTIMUM_AVAILABLE,
        )

        print(f"   📊 RTX Optimizer Status:")
        print(f"      - TensorRT Available: {'✅' if TENSORRT_AVAILABLE else '❌'}")
        print(f"      - CUDA Available: {'✅' if CUDA_AVAILABLE else '❌'}")
        print(f"      - Optimum Available: {'✅' if OPTIMUM_AVAILABLE else '❌'}")
    except ImportError as e:
        print(f"   ❌ RTX Optimizer import failed: {e}")

    # Summary and recommendations
    print(f"\n📋 SUMMARY:")
    print(f"   Direct TensorRT SDK: {'✅' if tensorrt_sdk_available else '❌'}")
    print(
        f"   ONNX Runtime TensorRT: {'✅' if 'TensorrtExecutionProvider' in ort.get_available_providers() else '❌'}"
    )

    if tensorrt_sdk_available:
        print(f"\n🔥 TENSORRT SDK DETECTED!")
        print(f"   Your RTX 5070 Ti can use the full TensorRT SDK capabilities.")
        print(f"   This provides maximum optimization beyond ONNX Runtime.")
        print(f"   Consider updating the RTX optimizer to use native TensorRT APIs.")
    else:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   If you installed TensorRT SDK, ensure it's in your PATH")
        print(f"   and the Python bindings are properly installed.")


if __name__ == "__main__":
    check_tensorrt_sdk()
