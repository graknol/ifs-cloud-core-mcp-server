#!/usr/bin/env python3
"""
Check Optimum TensorRT Capabilities
"""


def check_optimum_tensorrt():
    """Check what TensorRT capabilities are available via Optimum."""

    print("🔍 Checking Optimum TensorRT Capabilities")
    print("=" * 45)

    # Check optimum installation
    try:
        import optimum

        print("✅ Optimum installed")
    except ImportError:
        print("❌ Optimum not installed")
        return

    # Check optimum.nvidia
    try:
        from optimum.nvidia import TensorRTForSequenceClassification

        print("✅ optimum.nvidia.TensorRTForSequenceClassification available")
        tensorrt_via_nvidia = True
    except ImportError as e:
        print(f"❌ optimum.nvidia not available: {e}")
        tensorrt_via_nvidia = False

    # Check optimum.onnxruntime (which we have)
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        print("✅ optimum.onnxruntime.ORTModelForSequenceClassification available")
        ort_available = True
    except ImportError as e:
        print(f"❌ optimum.onnxruntime not available: {e}")
        ort_available = False

    # Check direct tensorrt import
    try:
        import tensorrt as trt

        print(f"✅ Direct TensorRT import available: version {trt.__version__}")
        direct_tensorrt = True
    except ImportError as e:
        print(f"❌ Direct TensorRT not available: {e}")
        direct_tensorrt = False

    # Check for TensorRT execution providers in ONNX Runtime
    if ort_available:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            print(f"📊 ONNX Runtime providers: {providers}")
            if "TensorrtExecutionProvider" in providers:
                print("✅ TensorRT execution provider available in ONNX Runtime")
            else:
                print("❌ TensorRT execution provider not available in ONNX Runtime")
        except Exception as e:
            print(f"❌ Error checking ONNX Runtime providers: {e}")

    print(f"\n📋 SUMMARY:")
    print(f"   Optimum NVIDIA: {'✅' if tensorrt_via_nvidia else '❌'}")
    print(f"   Optimum ONNX Runtime: {'✅' if ort_available else '❌'}")
    print(f"   Direct TensorRT: {'✅' if direct_tensorrt else '❌'}")

    if ort_available and not tensorrt_via_nvidia:
        print(f"\n💡 RECOMMENDATION:")
        print(
            f"   You have optimum[onnxruntime-gpu] which can provide GPU acceleration"
        )
        print(f"   through CUDA/DirectML execution providers, even without TensorRT.")
        print(f"   This is often sufficient for good performance on RTX 5070 Ti.")

    if not any([tensorrt_via_nvidia, direct_tensorrt]):
        print(f"\n🔧 TO GET TENSORRT:")
        print(f"   1. For Linux: pip install optimum[nvidia]")
        print(f"   2. For Windows: Download TensorRT from NVIDIA Developer site")
        print(f"   3. Alternative: Use current setup with torch.compile + max-autotune")


if __name__ == "__main__":
    check_optimum_tensorrt()
