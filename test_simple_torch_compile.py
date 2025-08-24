#!/usr/bin/env python3
"""
Simple torch.compile test to isolate the overflow issue
"""

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def test_simple_compile():
    print("🔬 Simple torch.compile test")
    print("=" * 40)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load model
    model_name = "microsoft/unixcoder-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Simple input
    test_input = tokenizer(
        "def test():", return_tensors="pt", max_length=64, truncation=True
    )
    test_input = {k: v.to(device) for k, v in test_input.items()}

    print("✅ Model and input ready")

    # Test uncompiled first
    with torch.no_grad():
        outputs = model(**test_input)
        print(f"✅ Uncompiled works: {outputs.logits.shape}")

    # Try simple compilation
    try:
        print("\n🧪 Testing default compilation...")
        compiled_model = torch.compile(model, mode="default")

        with torch.no_grad():
            outputs = compiled_model(**test_input)
            print(f"✅ Default compiled works: {outputs.logits.shape}")

    except Exception as e:
        print(f"❌ Default compilation failed: {e}")

    # Try with disabled problematic features
    try:
        print("\n🧪 Testing with coordinate descent disabled...")
        import torch._inductor.config

        torch._inductor.config.coordinate_descent_tuning = False

        compiled_model2 = torch.compile(model, mode="default")

        with torch.no_grad():
            outputs = compiled_model2(**test_input)
            print(f"✅ Safe compiled works: {outputs.logits.shape}")

    except Exception as e:
        print(f"❌ Safe compilation failed: {e}")


if __name__ == "__main__":
    test_simple_compile()
