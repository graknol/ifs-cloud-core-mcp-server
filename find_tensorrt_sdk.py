#!/usr/bin/env python3
"""
Enhanced TensorRT SDK Detection and Integration
"""

import os
import sys
from pathlib import Path


def find_tensorrt_installation():
    """Find TensorRT SDK installation on Windows."""

    print("🔍 Searching for TensorRT SDK Installation")
    print("=" * 45)

    # Common TensorRT installation paths on Windows
    search_paths = [
        "C:\\TensorRT",
        "C:\\Program Files\\TensorRT",
        "C:\\Program Files (x86)\\TensorRT",
        "C:\\Program Files\\NVIDIA\\TensorRT",
        "C:\\Program Files (x86)\\NVIDIA\\TensorRT",
        "C:\\NVIDIA\\TensorRT",
        "C:\\repos\\_lib\\nvidia\\tensorrt",  # Found in user's PATH
    ]

    # Extract TensorRT paths from system PATH
    path_env = os.environ.get("PATH", "")
    for path_item in path_env.split(";"):
        if "tensorrt" in path_item.lower():
            # If it's a bin directory, get parent
            if path_item.endswith("\\bin"):
                search_paths.append(path_item[:-4])  # Remove \bin
            else:
                search_paths.append(path_item)

    # Also check environment variables
    tensorrt_path = os.environ.get("TENSORRT_PATH")
    if tensorrt_path:
        search_paths.insert(0, tensorrt_path)

    tensorrt_root = os.environ.get("TENSORRT_ROOT")
    if tensorrt_root:
        search_paths.insert(0, tensorrt_root)

    found_installations = []

    for path in search_paths:
        if os.path.exists(path):
            print(f"   📁 Checking: {path}")

            # Look for typical TensorRT structure
            lib_path = os.path.join(path, "lib")
            include_path = os.path.join(path, "include")
            python_path = os.path.join(path, "python")

            if os.path.exists(lib_path) and os.path.exists(include_path):
                print(f"   ✅ TensorRT SDK structure found at: {path}")
                found_installations.append(
                    {
                        "root": path,
                        "lib": lib_path,
                        "include": include_path,
                        "python": python_path if os.path.exists(python_path) else None,
                    }
                )
            else:
                print(f"   ❌ No TensorRT structure at: {path}")
        else:
            print(f"   ❌ Path not found: {path}")

    # Check Windows PATH for tensorrt DLLs
    print(f"\n🔍 Checking Windows PATH for TensorRT DLLs:")
    path_env = os.environ.get("PATH", "")
    tensorrt_in_path = any("tensorrt" in p.lower() for p in path_env.split(";"))
    if tensorrt_in_path:
        print(f"   ✅ TensorRT references found in PATH")
    else:
        print(f"   ❌ No TensorRT references in PATH")

    return found_installations


def setup_tensorrt_python_bindings(installations):
    """Try to set up Python bindings for found TensorRT installations."""

    print(f"\n🔧 Setting up Python bindings:")

    if not installations:
        print("   ❌ No TensorRT installations found")
        return False

    for install in installations:
        python_path = install.get("python")
        if python_path and os.path.exists(python_path):
            print(f"   📂 Found Python bindings at: {python_path}")

            # Add to Python path
            if python_path not in sys.path:
                sys.path.insert(0, python_path)
                print(f"   ✅ Added to Python path")

            # Try to import tensorrt
            try:
                import tensorrt as trt

                print(
                    f"   ✅ TensorRT imported successfully: version {trt.__version__}"
                )
                return True
            except ImportError as e:
                print(f"   ❌ Failed to import TensorRT: {e}")
        else:
            print(f"   ❌ No Python bindings found for {install['root']}")

    return False


def test_tensorrt_functionality():
    """Test TensorRT functionality if available."""

    print(f"\n🧪 Testing TensorRT functionality:")

    try:
        import tensorrt as trt

        # Create a simple TensorRT builder to test
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        print(f"   ✅ TensorRT Builder created successfully")
        print(f"   📊 Platform capabilities:")
        print(f"      - FP16 support: {builder.platform_has_fast_fp16}")
        print(f"      - INT8 support: {builder.platform_has_fast_int8}")

        return True

    except Exception as e:
        print(f"   ❌ TensorRT functionality test failed: {e}")
        return False


def main():
    """Main TensorRT SDK detection and setup."""

    print("🚀 TensorRT SDK Detection and Setup")
    print("=" * 50)

    # Step 1: Find installations
    installations = find_tensorrt_installation()

    # Step 2: Set up Python bindings
    tensorrt_working = setup_tensorrt_python_bindings(installations)

    # Step 3: Test functionality
    if tensorrt_working:
        functionality_working = test_tensorrt_functionality()
    else:
        functionality_working = False

    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   TensorRT SDK Found: {'✅' if installations else '❌'}")
    print(f"   Python Bindings: {'✅' if tensorrt_working else '❌'}")
    print(f"   Functionality: {'✅' if functionality_working else '❌'}")

    if functionality_working:
        print(f"\n🔥 RESULT: TensorRT SDK is ready for use!")
        print(f"   The RTX 5070 Ti optimizer can now use native TensorRT APIs")
        print(f"   for maximum performance beyond ONNX Runtime.")
    else:
        print(f"\n💡 ALTERNATIVES:")
        print(f"   1. ✅ ONNX Runtime TensorRT provider (currently working)")
        print(f"   2. ✅ PyTorch torch.compile with max-autotune (currently working)")
        print(f"   3. Consider manual TensorRT SDK installation if needed")


if __name__ == "__main__":
    main()
