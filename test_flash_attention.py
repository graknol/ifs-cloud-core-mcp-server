#!/usr/bin/env python3
"""
Flash Attention 2 Performance Test
Validates Flash Attention installation and performance gains
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_flash_attention():
    """Test Flash Attention 2 setup and performance."""
    print("🧪 Testing Flash Attention 2 Implementation...")
    
    # Check Flash Attention availability
    try:
        import flash_attn
        print(f"✅ Flash Attention 2 detected: v{flash_attn.__version__}")
        flash_available = True
    except ImportError:
        print("❌ Flash Attention 2 not available")
        flash_available = False
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU tests")
        return
    
    print(f"🎯 Testing on: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Test model loading with Flash Attention
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    try:
        print(f"\n🚀 Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
            "use_cache": True
        }
        
        if flash_available:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("⚡ Using Flash Attention 2")
        else:
            model_kwargs["attn_implementation"] = "sdpa"  # Use PyTorch SDPA
            print("⚡ Using PyTorch SDPA (optimized attention)")
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Test inference speed
        test_prompt = "def calculate_business_logic():"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        # Warmup
        with torch.no_grad():
            _ = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=False)
        
        # Performance test
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                use_cache=True
            )
        end_time = time.time()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = end_time - start_time
        tokens_per_second = 50 / generation_time
        
        print(f"\n📊 Performance Results:")
        print(f"   • Generation time: {generation_time:.2f}s")
        print(f"   • Speed: {tokens_per_second:.1f} tokens/sec")
        print(f"   • Memory used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"   • Flash Attention: {'✅ Enabled' if flash_available else '❌ Disabled'}")
        
        print(f"\n📝 Generated text:")
        print(f"   {response[len(test_prompt):]}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_flash_attention()
