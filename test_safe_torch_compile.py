#!/usr/bin/env python3
"""
Test torch.compile with safer configurations to avoid overflow issues
"""

import torch
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_safe_torch_compile():
    """Test torch.compile with configurations that avoid overflow issues."""

    print("🔬 Testing Safe torch.compile Configurations")
    print("=" * 60)

    try:
        # Load model
        model_name = "microsoft/unixcoder-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP32 to avoid precision issues
            device_map="auto",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Use shorter input to avoid large tensor issues
        test_input = tokenizer(
            "def test():",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,  # Much shorter to avoid overflow
        )
        test_input = {k: v.to(device) for k, v in test_input.items()}

        print("✅ Model loaded with FP32 and shorter inputs")

        # Test configurations that should be more stable
        safe_configs = [
            {
                "mode": "default",
                "fullgraph": False,
                "description": "Default mode, partial graph",
            },
            {
                "mode": "default",
                "fullgraph": True,
                "description": "Default mode, full graph",
            },
            {"backend": "aot_eager", "description": "AOT Eager backend"},
        ]

        # Disable problematic autotuning features
        import torch._inductor.config

        torch._inductor.config.coordinate_descent_tuning = (
            False  # Disable the problematic tuning
        )
        torch._inductor.config.max_autotune = False
        torch._inductor.config.epilogue_fusion = False

        for config in safe_configs:
            print(f"\n🧪 Testing config: {config['description']}")

            try:
                # Compile with safe settings
                if "backend" in config:
                    compiled_model = torch.compile(model, backend=config["backend"])
                else:
                    compiled_model = torch.compile(
                        model,
                        mode=config["mode"],
                        fullgraph=config.get("fullgraph", False),
                    )

                # Test inference
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)

                    start_time.record()
                    outputs = compiled_model(**test_input)
                    end_time.record()

                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)

                    print(f"✅ {config['description']} works - {elapsed_time:.2f}ms")
                    print(f"   Output shape: {outputs.logits.shape}")

            except Exception as e:
                print(f"❌ {config['description']} failed: {e}")
                if "overflow" in str(e).lower():
                    print("   💡 This is the overflow issue we're trying to fix")

        # Try max-autotune with more conservative settings
        print(f"\n🧪 Testing max-autotune with conservative settings...")

        try:
            # Enable only safe autotuning features
            torch._inductor.config.coordinate_descent_tuning = False
            torch._inductor.config.max_autotune = True
            torch._inductor.config.max_autotune_gemm = True
            torch._inductor.config.epilogue_fusion = True
            torch._inductor.config.triton.autotune_remote_cache = False

            compiled_model = torch.compile(model, mode="max-autotune", fullgraph=False)

            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                outputs = compiled_model(**test_input)
                end_time.record()

                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)

                print(f"✅ Conservative max-autotune works - {elapsed_time:.2f}ms")

        except Exception as e:
            print(f"❌ Conservative max-autotune failed: {e}")

        # Performance comparison test
        print(f"\n📊 Performance comparison with working configuration...")

        try:
            # Find best working configuration
            torch._inductor.config.coordinate_descent_tuning = False
            torch._inductor.config.max_autotune = True
            torch._inductor.config.max_autotune_gemm = True

            compiled_model = torch.compile(model, mode="default", fullgraph=True)
            uncompiled_times = []
            compiled_times = []

            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    model(**test_input)
                    compiled_model(**test_input)

            # Benchmark uncompiled
            for _ in range(10):
                with torch.no_grad():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    outputs = model(**test_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    uncompiled_times.append(start_time.elapsed_time(end_time))

            # Benchmark compiled
            for _ in range(10):
                with torch.no_grad():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    outputs = compiled_model(**test_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    compiled_times.append(start_time.elapsed_time(end_time))

            avg_uncompiled = sum(uncompiled_times) / len(uncompiled_times)
            avg_compiled = sum(compiled_times) / len(compiled_times)
            speedup = avg_uncompiled / avg_compiled

            print(f"📈 Performance Results:")
            print(f"   Uncompiled: {avg_uncompiled:.2f}ms")
            print(f"   Compiled: {avg_compiled:.2f}ms")
            print(f"   Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"❌ Performance comparison failed: {e}")

    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_safe_torch_compile()
