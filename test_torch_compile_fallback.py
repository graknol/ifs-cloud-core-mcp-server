#!/usr/bin/env python3
"""
Test torch.compile fallback modes that don't require Triton
"""

import torch
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_torch_compile_fallback():
    """Test torch.compile with fallback modes that don't require Triton."""

    print("🔬 Testing torch.compile fallback modes (no Triton)")
    print("=" * 60)

    try:
        # Load model
        model_name = "microsoft/unixcoder-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).half()

        # Test input
        test_input = tokenizer(
            "def test_function():",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        test_input = {k: v.to(device) for k, v in test_input.items()}

        print("✅ Model loaded and test input prepared")

        # Test different backends that might not need Triton
        backends_to_test = [
            ("aot_eager", "AOT Eager compilation"),
            ("cudagraphs", "CUDA Graphs backend"),
            ("onnxrt", "ONNX Runtime backend"),
        ]

        # First test with backend specification
        for backend, description in backends_to_test:
            print(f"\n🧪 Testing backend: {backend} ({description})")

            try:
                # Set backend explicitly
                torch._inductor.config.triton.use_triton = False
                compiled_model = torch.compile(model, backend=backend)

                with torch.no_grad():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)

                    start_time.record()
                    outputs = compiled_model(**test_input)
                    end_time.record()

                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)

                    print(f"✅ Backend {backend} works - {elapsed_time:.2f}ms")

            except Exception as e:
                print(f"❌ Backend {backend} failed: {e}")

        # Test with Triton disabled and different modes
        print(f"\n🧪 Testing with Triton disabled...")

        try:
            # Disable Triton explicitly
            torch._inductor.config.triton.use_triton = False
            torch._inductor.config.cpp.enable_kernel_profile = False

            # Try different modes with Triton disabled
            modes_to_test = ["default", "reduce-overhead"]

            for mode in modes_to_test:
                print(f"\n🧪 Testing mode: {mode} (Triton disabled)")

                try:
                    compiled_model = torch.compile(model, mode=mode)

                    with torch.no_grad():
                        start_time = torch.cuda.Event(enable_timing=True)
                        end_time = torch.cuda.Event(enable_timing=True)

                        start_time.record()
                        outputs = compiled_model(**test_input)
                        end_time.record()

                        torch.cuda.synchronize()
                        elapsed_time = start_time.elapsed_time(end_time)

                        print(
                            f"✅ Mode {mode} works without Triton - {elapsed_time:.2f}ms"
                        )

                except Exception as e:
                    print(f"❌ Mode {mode} failed even without Triton: {e}")

        except Exception as e:
            print(f"❌ Triton disable configuration failed: {e}")

        # Test simple CPU fallback for comparison
        print(f"\n🧪 Testing CPU compilation for comparison...")

        try:
            model_cpu = model.to("cpu")
            test_input_cpu = {k: v.to("cpu") for k, v in test_input.items()}

            compiled_model_cpu = torch.compile(model_cpu, mode="default")

            with torch.no_grad():
                import time

                start = time.time()
                outputs = compiled_model_cpu(**test_input_cpu)
                end = time.time()

                print(f"✅ CPU compilation works - {(end-start)*1000:.2f}ms")

        except Exception as e:
            print(f"❌ CPU compilation failed: {e}")

    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_torch_compile_fallback()
