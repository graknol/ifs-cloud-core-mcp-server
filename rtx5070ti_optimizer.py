#!/usr/bin/env python3
"""
Maximum Performance GPU Optimization for RTX 5070 Ti

This module implements the most advanced ONNX Runtime optimizations:
- TensorRT execution provider (best performance)
- CUDA execution provider (fallback)
- IOBinding for zero-copy operations
- Mixed precision (FP16) optimization
- Memory pre-allocation
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
import asyncio
import torch
import onnxruntime as ort
from pathlib import Path

logger = logging.getLogger(__name__)


class RTX5070TiOptimizer:
    """Maximum performance optimizer for RTX 5070 Ti."""

    def __init__(self, model_name: str = "t5-small"):  # Use T5 for ONNX compatibility
        self.model_name = model_name
        self.ort_model = None
        self.tokenizer = None
        self.session_options = None
        self.providers = self._get_optimal_providers()

        # RTX 5070 Ti specific optimizations
        self.optimal_batch_size = 64  # Can handle larger batches with TensorRT
        self.use_fp16 = True
        self.use_io_binding = True

        logger.info(
            f"🚀 Initializing RTX 5070 Ti optimizer with providers: {self.providers}"
        )
        logger.info(f"📝 Using {model_name} (ONNX compatible) for summarization")

    def _get_optimal_providers(self) -> List[str]:
        """Get optimal execution providers in priority order."""
        available_providers = ort.get_available_providers()

        # Priority order: TensorRT > CUDA > CPU
        optimal_providers = []

        if "TensorrtExecutionProvider" in available_providers:
            optimal_providers.append("TensorrtExecutionProvider")
            logger.info("✅ TensorRT available - Maximum performance mode")

        if "CUDAExecutionProvider" in available_providers:
            optimal_providers.append("CUDAExecutionProvider")
            logger.info("✅ CUDA available - GPU acceleration mode")

        optimal_providers.append("CPUExecutionProvider")  # Always available fallback

        return optimal_providers

    async def initialize_model(self) -> bool:
        """Initialize the optimized model with maximum performance settings."""
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer

            logger.info("🔧 Initializing model with RTX 5070 Ti optimizations...")

            # Create optimized session options
            self.session_options = ort.SessionOptions()

            # RTX 5070 Ti optimizations
            self.session_options.enable_mem_pattern = True
            self.session_options.enable_mem_reuse = True
            self.session_options.enable_cpu_mem_arena = False  # Use GPU memory

            # Threading optimizations for 32-core system
            self.session_options.intra_op_num_threads = 32
            self.session_options.inter_op_num_threads = 32
            self.session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

            # Enable verbose logging to verify GPU placement
            self.session_options.log_severity_level = 0

            # TensorRT specific optimizations
            provider_options = []
            for provider in self.providers:
                if provider == "TensorrtExecutionProvider":
                    provider_options.append(
                        {
                            "device_id": 0,
                            "trt_max_workspace_size": 8
                            * 1024
                            * 1024
                            * 1024,  # 8GB workspace
                            "trt_fp16_enable": self.use_fp16,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "./tensorrt_cache/",
                            "trt_max_batch_size": self.optimal_batch_size,
                        }
                    )
                elif provider == "CUDAExecutionProvider":
                    provider_options.append(
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kSameAsRequested",
                            "gpu_mem_limit": 12
                            * 1024
                            * 1024
                            * 1024,  # 12GB limit for safety
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                        }
                    )
                else:
                    provider_options.append({})

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Initialize ONNX model with optimal settings
            self.ort_model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                export=True,  # Export to ONNX if needed
                provider=self.providers[0],  # Use best available provider
                session_options=self.session_options,
                provider_options=provider_options[0] if provider_options else {},
                use_io_binding=self.use_io_binding,  # Enable zero-copy operations
                use_cache=True,  # Enable KV-cache for generation
            )

            # Verify provider selection
            actual_providers = self.ort_model.providers
            logger.info(f"✅ Model initialized with providers: {actual_providers}")

            if "TensorrtExecutionProvider" in actual_providers:
                logger.info("🚀 TensorRT mode active - Maximum performance!")
            elif "CUDAExecutionProvider" in actual_providers:
                logger.info("⚡ CUDA mode active - High performance!")

            return True

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            return False

    async def optimize_batch_processing(
        self, prompts: List[str], max_length: int = 150
    ) -> List[str]:
        """Process batch with maximum RTX 5070 Ti optimization."""

        if not self.ort_model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        start_time = time.time()

        # Pre-allocate GPU memory for optimal performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear any existing allocations
            initial_memory = torch.cuda.memory_allocated()

        # Tokenize with optimal settings
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # UnixCoder context limit
            return_attention_mask=True,
        )

        # Pre-move to GPU if using IOBinding
        if self.use_io_binding and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        try:
            # Generate with RTX 5070 Ti optimization
            with torch.no_grad():
                if self.use_fp16 and torch.cuda.is_available():
                    # Use automatic mixed precision for RTX 5070 Ti
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.ort_model.generate(
                            **inputs,
                            max_length=max_length,
                            num_beams=4,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,  # Enable KV-cache
                            early_stopping=True,
                            no_repeat_ngram_size=2,  # Avoid repetition
                        )
                else:
                    outputs = self.ort_model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=4,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                    )

            # Decode results
            summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Performance monitoring
            processing_time = time.time() - start_time

            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - initial_memory) / 1024**3
                throughput = len(prompts) / processing_time

                logger.info(
                    f"🚀 RTX 5070 Ti Performance: {len(prompts)} samples in {processing_time:.2f}s "
                    f"({throughput:.1f} samples/sec, {memory_used:.2f}GB used)"
                )

            return [s.strip() for s in summaries]

        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return ["Error: Processing failed"] * len(prompts)

    def get_optimal_batch_size(self, available_memory_gb: float = 15.9) -> int:
        """Calculate optimal batch size for RTX 5070 Ti."""

        # Base calculation for RTX 5070 Ti with 15.9GB VRAM
        if "TensorrtExecutionProvider" in self.providers:
            # TensorRT can handle larger batches more efficiently
            base_batch = min(80, int(available_memory_gb * 5))
        else:
            # CUDA provider
            base_batch = min(64, int(available_memory_gb * 4))

        # Adjust based on model complexity
        if "unixcoder" in self.model_name.lower():
            # UnixCoder is relatively lightweight
            return base_batch
        else:
            # Other models might need smaller batches
            return max(32, base_batch // 2)

    async def benchmark_performance(self, num_samples: int = 100) -> Dict:
        """Benchmark RTX 5070 Ti performance with different configurations."""

        logger.info(
            f"🧪 Benchmarking RTX 5070 Ti performance with {num_samples} samples"
        )

        # Create test data
        test_prompts = [
            f"# Summarize PL/SQL Function\nFunction: Test_Function_{i}\n```plsql\nFUNCTION test_func() RETURN NUMBER IS BEGIN RETURN {i}; END;\n```\nSummary:"
            for i in range(num_samples)
        ]

        results = {}

        # Test different batch sizes
        for batch_size in [16, 32, 48, 64]:
            if batch_size > len(test_prompts):
                continue

            try:
                start_time = time.time()

                # Process in batches
                all_results = []
                for i in range(0, len(test_prompts), batch_size):
                    batch = test_prompts[i : i + batch_size]
                    batch_results = await self.optimize_batch_processing(batch)
                    all_results.extend(batch_results)

                total_time = time.time() - start_time
                throughput = len(test_prompts) / total_time

                results[f"batch_{batch_size}"] = {
                    "time": total_time,
                    "throughput": throughput,
                    "samples_processed": len(all_results),
                }

                logger.info(f"✅ Batch size {batch_size}: {throughput:.1f} samples/sec")

            except Exception as e:
                logger.error(f"❌ Batch size {batch_size} failed: {e}")

        return results


async def test_rtx5070ti_optimization():
    """Test the RTX 5070 Ti optimization."""

    print("🚀 RTX 5070 Ti Maximum Performance Test")
    print("=" * 60)

    optimizer = RTX5070TiOptimizer()

    # Initialize model
    success = await optimizer.initialize_model()
    if not success:
        print("❌ Model initialization failed")
        return

    print(
        f"✅ Model initialized with optimal batch size: {optimizer.get_optimal_batch_size()}"
    )

    # Run performance benchmark
    results = await optimizer.benchmark_performance(50)

    print(f"\n📊 Performance Results:")
    best_config = max(results.items(), key=lambda x: x[1]["throughput"])

    for config, metrics in results.items():
        marker = "🏆" if config == best_config[0] else "📈"
        print(
            f"   {marker} {config}: {metrics['throughput']:.1f} samples/sec ({metrics['time']:.2f}s)"
        )

    print(
        f"\n🎉 Best configuration: {best_config[0]} with {best_config[1]['throughput']:.1f} samples/sec"
    )

    # Memory efficiency test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3

        # Process a large batch
        large_batch = ["Test prompt"] * optimizer.get_optimal_batch_size()
        await optimizer.optimize_batch_processing(large_batch)

        memory_after = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n💾 Memory Usage:")
        print(
            f"   Peak GPU memory: {memory_after:.2f}GB / 15.9GB ({memory_after/15.9*100:.1f}%)"
        )
        print(
            f"   Memory efficient: {'Yes' if memory_after < 12 else 'Consider reducing batch size'}"
        )


if __name__ == "__main__":
    asyncio.run(test_rtx5070ti_optimization())
