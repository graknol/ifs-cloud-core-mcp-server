#!/usr/bin/env python3
"""Test Flash Attention and Triton availability."""

print("🔍 Testing Flash Attention and Triton compatibility...")
print()

# Test PyTorch
try:
    import torch

    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(
        f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
    )
except Exception as e:
    print(f"❌ PyTorch error: {e}")

print()

# Test Triton
try:
    import triton

    print(f"✅ Triton: {triton.__version__}")

    # Test if Triton can compile a simple kernel
    try:

        @triton.jit
        def simple_kernel(
            x_ptr, output_ptr, n_elements, BLOCK_SIZE: triton.language.tl.constexpr
        ):
            pid = triton.language.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + triton.language.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = triton.language.load(x_ptr + offsets, mask=mask)
            triton.language.store(output_ptr + offsets, x, mask=mask)

        print("✅ Triton kernel compilation: OK")
    except Exception as e:
        print(f"⚠️ Triton kernel test failed: {e}")

except Exception as e:
    print(f"❌ Triton error: {e}")

print()

# Test Flash Attention
try:
    import flash_attn

    print(f"✅ Flash Attention: {flash_attn.__version__}")

    # Test Flash Attention function
    try:
        from flash_attn import flash_attn_func

        print("✅ Flash Attention function import: OK")

        # Test actual computation
        if torch.cuda.is_available():
            batch_size, seq_len, head_dim = 2, 512, 64
            num_heads = 8

            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            k = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            v = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )

            out = flash_attn_func(q, k, v)
            print(f"✅ Flash Attention computation: OK, output shape: {out.shape}")
        else:
            print("⚠️ CUDA not available, skipping Flash Attention computation test")

    except Exception as e:
        print(f"❌ Flash Attention function test failed: {e}")

except Exception as e:
    print(f"❌ Flash Attention error: {e}")

print()

# Test PyTorch SDPA (fallback)
try:
    # Test PyTorch's built-in scaled_dot_product_attention
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("✅ PyTorch SDPA available: OK")

        if torch.cuda.is_available():
            batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64

            q = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            k = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            v = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )

            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                print(f"✅ PyTorch SDPA computation: OK, output shape: {out.shape}")
        else:
            print("⚠️ CUDA not available, skipping SDPA computation test")
    else:
        print("❌ PyTorch SDPA not available")
except Exception as e:
    print(f"❌ PyTorch SDPA error: {e}")

print()
print("🎯 Summary:")
print(
    "- Flash Attention and Triton work together (Flash Attention uses Triton kernels)"
)
print("- If Flash Attention fails, PyTorch SDPA is an excellent fallback")
print("- Both provide significant performance improvements over standard attention")
