#!/usr/bin/env python3
"""
Test RTX 5070 Ti Optimizer with IFS Cloud Parser Integration
"""

import asyncio
from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer


async def test_rtx_with_ifs_parser():
    print("🚀 Testing RTX 5070 Ti Optimizer with IFS Cloud Parser")
    print("=" * 60)

    # Initialize optimizer
    opt = RTX5070TiPyTorchOptimizer()
    success = await opt.initialize_model()

    if not success:
        print("❌ Failed to initialize optimizer")
        return

    print("✅ RTX Optimizer initialized with IFS parser integration")
    print(f"   Parser available: {opt.ifs_parser.parser_available}")
    print()

    # Test with various IFS Cloud code samples
    test_samples = [
        "PROCEDURE Get_Customer_Info___ IS BEGIN SELECT * FROM customer; END;",
        "FUNCTION Calculate_Total___ RETURN NUMBER IS BEGIN FOR rec IN cursor LOOP total := total + rec.amount; END LOOP; RETURN total; END;",
        "PROCEDURE Validate_Order___ IS BEGIN IF order_status = 'PENDING' THEN UPDATE orders SET status = 'VALIDATED'; END IF; EXCEPTION WHEN OTHERS THEN Error_SYS.Record_General('ERROR', 'Validation failed'); END;",
        "FUNCTION Check_Authorization___ RETURN BOOLEAN IS BEGIN RETURN user_has_permission; END;",
        "PROCEDURE Process_Invoice___ IS cursor_rec NUMBER; BEGIN OPEN cursor_rec FOR SELECT * FROM invoices; LOOP FETCH cursor_rec INTO invoice_rec; EXIT WHEN cursor_rec%NOTFOUND; UPDATE invoice SET processed = 'Y'; END LOOP; END;",
    ]

    print("📝 Testing IFS Cloud code analysis:")
    print("-" * 50)

    # Process samples individually to see detailed analysis
    for i, code in enumerate(test_samples, 1):
        print(f"\n🔍 Sample {i}:")
        print(f"   Code: {code[:60]}...")

        try:
            # Test the enhanced summary creation
            func_name = f"test_function_{i}"
            summary = opt.create_unixcoder_summary(code, func_name)
            print(f"   Summary: {summary}")

            # Test parser analysis directly
            parsed_data = opt.ifs_parser.parse_code(code)
            complexity = opt.ifs_parser.analyze_complexity(parsed_data)
            print(f"   Parse method: {parsed_data.get('method', 'unknown')}")
            print(
                f"   Complexity: {complexity.get('complexity_level', 'unknown')} (score: {complexity.get('complexity_score', 0)})"
            )

            if parsed_data.get("patterns"):
                patterns = parsed_data["patterns"]
                detected = [k for k, v in patterns.items() if v]
                print(f"   Patterns: {', '.join(detected) if detected else 'none'}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Test batch processing
    print(f"\n🎯 Testing batch processing with {len(test_samples)} samples:")
    print("-" * 50)

    try:
        import time

        start = time.perf_counter()

        results = await opt.process_batch_optimized(test_samples)

        end = time.perf_counter()
        processing_time = end - start

        print(f"✅ Batch processing completed:")
        print(f"   Processed: {len(results)} samples")
        print(f"   Time: {processing_time*1000:.2f}ms")
        print(f"   Throughput: {len(results)/processing_time:.1f} samples/sec")
        print(f"   Unique results: {len(set(results))}")

        print(f"\n📊 Sample results:")
        for i, result in enumerate(results[:3], 1):
            print(f"   Result {i}: {result}")

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")

    # Memory usage stats
    print(f"\n💾 Memory Statistics:")
    memory_stats = opt.get_memory_stats()
    for key, value in memory_stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_rtx_with_ifs_parser())
