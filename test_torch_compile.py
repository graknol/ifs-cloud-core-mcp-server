#!/usr/bin/env python3
"""
Test torch.compile with max-autotune mode to diagnose issues
"""

import torch
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_torch_compile():
    """Test torch.compile functionality and identify missing dependencies."""

    print("🔬 Testing torch.compile with max-autotune")
    print("=" * 60)

    # Check system requirements
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}"
    )
    print(f"Compile available: {hasattr(torch, 'compile')}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    print("\n🧪 Loading model...")

    try:
        # Load a simple model first
        model_name = "microsoft/unixcoder-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

        print("✅ Model loaded successfully")

        # Move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = model.half()

        print(f"✅ Model moved to {device}")

        # Test basic inference first
        test_input = tokenizer(
            "def test_function():", return_tensors="pt", padding=True, truncation=True
        )
        test_input = {k: v.to(device) for k, v in test_input.items()}

        print("\n🧪 Testing basic inference...")
        with torch.no_grad():
            outputs = model(**test_input)
            print("✅ Basic inference works")

        # Now try torch.compile with different modes
        compile_modes = ["default", "reduce-overhead", "max-autotune"]

        for mode in compile_modes:
            print(f"\n🧪 Testing torch.compile with mode: {mode}")
            try:
                compiled_model = torch.compile(model, mode=mode)
                print(f"✅ Compilation with {mode} succeeded")

                # Test inference with compiled model
                with torch.no_grad():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)

                    start_time.record()
                    outputs = compiled_model(**test_input)
                    end_time.record()

                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)

                    print(
                        f"✅ Compiled inference with {mode} works - {elapsed_time:.2f}ms"
                    )

            except Exception as e:
                print(f"❌ Compilation with {mode} failed: {e}")
                import traceback

                traceback.print_exc()

                # Check for specific missing dependencies
                error_str = str(e).lower()

                if "triton" in error_str:
                    print("💡 Missing Triton compiler - try: pip install triton")
                elif "torchinductor" in error_str:
                    print(
                        "💡 TorchInductor issue - may need PyTorch nightly or different CUDA version"
                    )
                elif "symbolic" in error_str:
                    print("💡 Symbolic tracing issue - try simpler compilation mode")
                elif "dynamo" in error_str:
                    print("💡 TorchDynamo issue - check PyTorch installation")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback

        traceback.print_exc()


def check_dependencies():
    """Check for compilation-related dependencies."""

    print("\n🔍 Checking compilation dependencies...")
    print("=" * 40)

    dependencies = [
        ("torch", "PyTorch core"),
        ("triton", "Triton compiler for GPU kernels"),
        ("torchaudio", "TorchAudio (sometimes needed)"),
        ("functorch", "FuncTorch (functional transforms)"),
    ]

    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: Available ({description})")
        except ImportError:
            print(f"❌ {dep}: Missing ({description})")

            if dep == "triton":
                print("   Install with: pip install triton")
            elif dep == "functorch":
                print("   Usually included with PyTorch 2.0+")


if __name__ == "__main__":
    check_dependencies()
    test_torch_compile()
