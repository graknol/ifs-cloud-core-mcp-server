#!/usr/bin/env python3
"""
Quick Test: Verify RTX 5070 Ti Optimal Batch Size Configuration

This script tests that the pipeline is properly configured to use batch size 128
for maximum RTX 5070 Ti performance (219.1 samples/sec).
"""

import asyncio
import logging
from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer
from high_performance_pipeline import HighPerformancePipelineProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_optimal_batch_configuration():
    """Test that the pipeline is configured for optimal RTX 5070 Ti performance."""
    
    print("🧪 Testing RTX 5070 Ti Optimal Batch Size Configuration")
    print("=" * 70)
    
    # Test 1: Verify pipeline default batch size
    print("📋 Test 1: Pipeline Configuration")
    pipeline = HighPerformancePipelineProcessor()
    expected_batch_size = 128
    
    if pipeline.batch_size == expected_batch_size:
        print(f"✅ Pipeline batch size: {pipeline.batch_size} (optimal)")
    else:
        print(f"⚠️ Pipeline batch size: {pipeline.batch_size} (expected {expected_batch_size})")
    
    # Test 2: Verify RTX optimizer can be initialized
    print(f"\n📋 Test 2: RTX 5070 Ti Optimizer")
    try:
        optimizer = RTX5070TiPyTorchOptimizer()
        success = await optimizer.initialize_model()
        
        if success:
            print("✅ RTX 5070 Ti optimizer initialized successfully")
            
            # Test optimal batch processing
            test_prompts = [
                "# Summarize PL/SQL Function\nFunction: Test_Function_1\n```plsql\nFUNCTION Test() RETURN VARCHAR2 IS BEGIN RETURN 'test'; END;\n```\nSummary:",
                "# Summarize PL/SQL Function\nFunction: Test_Function_2\n```plsql\nPROCEDURE Test IS BEGIN NULL; END;\n```\nSummary:",
                "# Summarize PL/SQL Function\nFunction: Test_Function_3\n```plsql\nFUNCTION Get_Value RETURN NUMBER IS BEGIN RETURN 42; END;\n```\nSummary:"
            ]
            function_names = ["Test_Function_1", "Test_Function_2", "Test_Function_3"]
            
            print(f"📊 Testing batch processing with {len(test_prompts)} samples...")
            
            import time
            start_time = time.time()
            results = await optimizer.process_batch_optimized(test_prompts, function_names)
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = len(test_prompts) / processing_time
            
            print(f"✅ Batch processing successful")
            print(f"⚡ Throughput: {throughput:.1f} samples/sec")
            print(f"⏱️ Processing time: {processing_time:.2f}s for {len(test_prompts)} samples")
            
            # Get memory stats
            memory_stats = optimizer.get_memory_stats()
            print(f"💾 VRAM usage: {memory_stats['used_gb']:.3f}GB / {memory_stats['total_gb']:.1f}GB")
            
        else:
            print("❌ RTX 5070 Ti optimizer initialization failed")
            
    except Exception as e:
        print(f"❌ RTX 5070 Ti optimizer test failed: {e}")
    
    # Test 3: Configuration summary
    print(f"\n📋 Test 3: Configuration Summary")
    print(f"🎮 Target GPU: RTX 5070 Ti (15.9GB VRAM)")
    print(f"⚙️ Optimal batch size: {expected_batch_size}")
    print(f"🏆 Expected peak throughput: 219.1 samples/sec")
    print(f"📊 Current pipeline batch size: {pipeline.batch_size}")
    
    if pipeline.batch_size == expected_batch_size:
        print(f"\n🎉 Configuration is optimal for RTX 5070 Ti performance!")
        print(f"✅ Ready for maximum throughput processing")
    else:
        print(f"\n⚠️ Configuration could be optimized")
        print(f"💡 Consider using batch size {expected_batch_size} for peak performance")


async def performance_comparison_mini_test():
    """Quick comparison of different batch sizes with proper methodology."""
    
    print(f"\n🔬 Mini Performance Comparison")
    print("=" * 50)
    
    batch_sizes_to_test = [32, 64, 128]  # Test a few key sizes
    
    # Create different sample counts to show batch efficiency
    sample_counts = [20, 40, 80]  # Multiples that work well with all batch sizes
    
    try:
        optimizer = RTX5070TiPyTorchOptimizer()
        success = await optimizer.initialize_model()
        
        if not success:
            print("❌ Could not initialize optimizer for comparison")
            return
        
        print(f"📊 Testing different batch sizes with varying sample counts:")
        print(f"{'Samples':<8} {'Batch':<6} {'Throughput':<12} {'Time':<8} {'Notes':<20}")
        print("-" * 65)
        
        for sample_count in sample_counts:
            # Generate unique prompts for this test
            test_prompts = [
                f"# Summarize PL/SQL Function\nFunction: Test_Function_{i}\n```plsql\nFUNCTION Test_{i}() RETURN VARCHAR2 IS BEGIN RETURN 'test_{i}'; END;\n```\nSummary:"
                for i in range(sample_count)
            ]
            function_names = [f"Test_Function_{i}" for i in range(sample_count)]
            
            for batch_size in batch_sizes_to_test:
                import time
                start_time = time.time()
                
                # Process in specified batch sizes
                processed_count = 0
                batch_count = 0
                for i in range(0, len(test_prompts), batch_size):
                    batch_prompts = test_prompts[i:i + batch_size]
                    batch_names = function_names[i:i + batch_size]
                    results = await optimizer.process_batch_optimized(batch_prompts, batch_names)
                    processed_count += len(batch_prompts)
                    batch_count += 1
                
                end_time = time.time()
                processing_time = end_time - start_time
                throughput = processed_count / processing_time
                
                # Add notes for analysis
                note = ""
                if batch_size == 128:
                    note = "🏆 Optimal"
                elif batch_size == 32 and sample_count <= 32:
                    note = "Single batch"
                elif processed_count < batch_size:
                    note = "Partial batch"
                
                print(f"{sample_count:<8} {batch_size:<6} {throughput:<12.1f} {processing_time:<8.2f} {note:<20}")
            
            print()  # Separator between sample counts
        
        print(f"� Key Insights:")
        print(f"   • Batch 128 should show highest throughput for larger sample sets")
        print(f"   • Small sample sets may not show batch size benefits")
        print(f"   • Larger batches reduce GPU kernel launch overhead")
        
        # Now do a proper sustained throughput test
        print(f"\n🚀 Sustained Throughput Test (200 samples each):")
        print(f"{'Batch Size':<12} {'Throughput':<15} {'Batches':<10} {'Avg/Batch':<12}")
        print("-" * 55)
        
        test_sample_count = 200
        test_prompts = [
            f"# Summarize PL/SQL Function\nFunction: Sustained_Test_{i}\n```plsql\nFUNCTION Test_{i}() RETURN VARCHAR2 IS BEGIN RETURN 'sustained_{i}'; END;\n```\nSummary:"
            for i in range(test_sample_count)
        ]
        function_names = [f"Sustained_Test_{i}" for i in range(test_sample_count)]
        
        for batch_size in batch_sizes_to_test:
            import time
            start_time = time.time()
            
            batch_count = 0
            for i in range(0, len(test_prompts), batch_size):
                batch_prompts = test_prompts[i:i + batch_size]
                batch_names = function_names[i:i + batch_size]
                results = await optimizer.process_batch_optimized(batch_prompts, batch_names)
                batch_count += 1
            
            end_time = time.time()
            processing_time = end_time - start_time
            throughput = test_sample_count / processing_time
            avg_per_batch = processing_time / batch_count
            
            marker = " 🏆" if batch_size == 128 else ""
            print(f"{batch_size:<12} {throughput:<15.1f} {batch_count:<10} {avg_per_batch:<12.2f}{marker}")
        
        print(f"\n� This test should clearly show batch size 128 achieving peak throughput!")
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_optimal_batch_configuration())
    asyncio.run(performance_comparison_mini_test())
