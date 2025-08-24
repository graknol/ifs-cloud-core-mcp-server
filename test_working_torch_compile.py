#!/usr/bin/env python3
"""
Working torch.compile test - fix the import issue
"""

import torch as pytorch_module
import torch._inductor.config as inductor_config
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def test_working_compile():
    print("🔬 Working torch.compile Configuration Test")
    print("=" * 50)

    print(f"PyTorch version: {pytorch_module.__version__}")
    print(f"CUDA available: {pytorch_module.cuda.is_available()}")
    print(
        f"Triton available: {'✅' if 'triton' in str(pytorch_module.compile) else '❌'}"
    )

    # Load model
    model_name = "microsoft/unixcoder-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    device = pytorch_module.device(
        "cuda" if pytorch_module.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # Simple input
    test_input = tokenizer(
        "def test():", return_tensors="pt", max_length=64, truncation=True
    )
    test_input = {k: v.to(device) for k, v in test_input.items()}

    print("✅ Model and input ready")

    # Test uncompiled baseline
    with pytorch_module.no_grad():
        baseline_outputs = model(**test_input)
        print(f"✅ Uncompiled baseline: {baseline_outputs.logits.shape}")

    # Configuration 1: Default mode with safe settings
    print("\n🧪 Test 1: Default mode with coordinate descent disabled")
    try:
        inductor_config.coordinate_descent_tuning = False
        inductor_config.max_autotune = False

        compiled_model_1 = pytorch_module.compile(model, mode="default")

        with pytorch_module.no_grad():
            outputs_1 = compiled_model_1(**test_input)
            print(f"✅ Configuration 1 works: {outputs_1.logits.shape}")

    except Exception as e:
        print(f"❌ Configuration 1 failed: {e}")

    # Configuration 2: Try max-autotune with coordinate descent off
    print("\n🧪 Test 2: max-autotune with coordinate descent disabled")
    try:
        inductor_config.coordinate_descent_tuning = False
        inductor_config.max_autotune = True
        inductor_config.max_autotune_gemm = True
        inductor_config.epilogue_fusion = True

        compiled_model_2 = pytorch_module.compile(model, mode="max-autotune")

        with pytorch_module.no_grad():
            outputs_2 = compiled_model_2(**test_input)
            print(f"✅ Configuration 2 works: {outputs_2.logits.shape}")

    except Exception as e:
        print(f"❌ Configuration 2 failed: {e}")
        if "overflow" in str(e).lower():
            print("   💡 Still getting overflow - this is a Triton/PyTorch issue")

    # Configuration 3: Try reduce-overhead mode
    print("\n🧪 Test 3: reduce-overhead mode")
    try:
        inductor_config.coordinate_descent_tuning = False
        inductor_config.max_autotune = False

        compiled_model_3 = pytorch_module.compile(model, mode="reduce-overhead")

        with pytorch_module.no_grad():
            outputs_3 = compiled_model_3(**test_input)
            print(f"✅ Configuration 3 works: {outputs_3.logits.shape}")

    except Exception as e:
        print(f"❌ Configuration 3 failed: {e}")

    # Configuration 4: Try AOT eager backend
    print("\n🧪 Test 4: AOT Eager backend (no Triton)")
    try:
        compiled_model_4 = pytorch_module.compile(model, backend="aot_eager")

        with pytorch_module.no_grad():
            outputs_4 = compiled_model_4(**test_input)
            print(f"✅ Configuration 4 works: {outputs_4.logits.shape}")

    except Exception as e:
        print(f"❌ Configuration 4 failed: {e}")

    print("\n📊 Summary:")
    print("If any configuration works, we can integrate it into RTX optimizer!")


if __name__ == "__main__":
    test_working_compile()
