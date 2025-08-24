#!/usr/bin/env python3
"""
Test torch.compile with FP16 to resolve max-autotune overflow issues.
"""

import asyncio
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time


async def test_torch_compile_fp16():
    print("🔬 Testing torch.compile with FP16 for max-autotune compatibility")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "microsoft/unixcoder-base"

    # Load model and tokenizer
    print(f"📦 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Simple binary classification
        torch_dtype=torch.float16,  # Force FP16 from start
        device_map="auto",
    )
    model.eval()

    print(f"✅ Model loaded on {device} with dtype {model.dtype}")

    # Configure torch for FP16 operations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Set inductor config for FP16 max-autotune
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True

    # Test different compilation modes with FP16
    test_configs = [
        {"name": "FP16 Default Mode", "mode": "default", "max_autotune": False},
        {"name": "FP16 Max-Autotune", "mode": "max-autotune", "max_autotune": True},
        {
            "name": "FP16 Reduce-Overhead",
            "mode": "reduce-overhead",
            "max_autotune": False,
        },
    ]

    # Test data
    test_texts = [
        "def calculate_sum(a, b): return a + b",
        "function processData() { return data.map(x => x * 2); }",
        "SELECT * FROM users WHERE active = 1",
    ]

    for config in test_configs:
        print(f"\n🧪 Testing {config['name']}...")

        try:
            # Clone model for each test to avoid compilation conflicts
            test_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, torch_dtype=torch.float16, device_map="auto"
            )
            test_model.eval()

            # Apply compilation
            if config["max_autotune"]:
                torch._inductor.config.max_autotune = True
                compiled_model = torch.compile(
                    test_model, mode=config["mode"], fullgraph=False
                )
                torch._inductor.config.max_autotune = False  # Reset after
            else:
                compiled_model = torch.compile(
                    test_model, mode=config["mode"], fullgraph=False
                )

            # Tokenize test data with FP16 compatible settings
            inputs = tokenizer(
                test_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            # Ensure input tensors are FP16
            if hasattr(inputs, "input_ids"):
                inputs["input_ids"] = inputs["input_ids"].to(torch.long)  # Keep as long
            if hasattr(inputs, "attention_mask"):
                inputs["attention_mask"] = inputs["attention_mask"].to(
                    torch.long
                )  # Keep as long

            # Warmup run
            print("   🔥 Warmup run...")
            with torch.no_grad():
                _ = compiled_model(**inputs)

            # Timed run
            print("   ⏱️  Timed run...")
            start = time.perf_counter()
            with torch.no_grad():
                outputs = compiled_model(**inputs)
            end = time.perf_counter()

            print(f"   ✅ SUCCESS: {config['name']}")
            print(f"      Time: {(end-start)*1000:.2f}ms")
            print(f"      Output shape: {outputs.logits.shape}")
            print(f"      Output dtype: {outputs.logits.dtype}")
            print(f"      Memory used: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

        except Exception as e:
            print(f"   ❌ FAILED: {config['name']}")
            print(f"      Error: {str(e)}")
            if "overflow" in str(e).lower() or "too large" in str(e).lower():
                print(
                    f"      🔍 Still overflow with FP16 - may need further optimization"
                )

        # Cleanup
        torch.cuda.empty_cache()
        if "test_model" in locals():
            del test_model
        if "compiled_model" in locals():
            del compiled_model


if __name__ == "__main__":
    asyncio.run(test_torch_compile_fp16())
