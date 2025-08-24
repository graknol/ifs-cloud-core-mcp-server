#!/usr/bin/env python3
"""
Test Optimized AST Context Extraction
"""

import asyncio
from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer
import json


async def test_optimized_context():
    """Test the optimized AST context extraction for embeddings."""

    print("🔬 Testing Optimized AST Context Extraction")
    print("=" * 55)

    # Initialize optimizer
    opt = RTX5070TiPyTorchOptimizer()

    # Test with realistic IFS Cloud code
    test_code = """
    FUNCTION Get_Customer_Order_Status___ (
        company_id_ IN VARCHAR2,
        customer_id_ IN VARCHAR2,
        order_no_ IN VARCHAR2
    ) RETURN CLOB IS
        cursor order_cursor IS
            SELECT order_status, delivery_date, total_amount
            FROM customer_order co, customer c
            WHERE co.customer_id = c.customer_id
            AND c.company = company_id_
            AND c.customer_id = customer_id_
            AND co.order_no = order_no_;
            
        order_status_ VARCHAR2(50);
        delivery_date_ DATE;
        total_amount_ NUMBER;
        result_json_ CLOB;
        
    BEGIN
        -- Validate input parameters
        IF company_id_ IS NULL OR customer_id_ IS NULL THEN
            Error_SYS.Record_General('INVALID_PARAMS', 'Required parameters missing');
        END IF;
        
        -- Get order information
        OPEN order_cursor;
        FETCH order_cursor INTO order_status_, delivery_date_, total_amount_;
        
        IF order_cursor%NOTFOUND THEN
            CLOSE order_cursor;
            RETURN '{"error": "Order not found"}';
        END IF;
        
        CLOSE order_cursor;
        
        -- Build JSON response
        result_json_ := '{' ||
            '"order_status": "' || order_status_ || '",' ||
            '"delivery_date": "' || TO_CHAR(delivery_date_, 'YYYY-MM-DD') || '",' ||
            '"total_amount": ' || NVL(total_amount_, 0) ||
        '}';
        
        -- Log access
        Activity_Log_API.New('ORDER_STATUS_ACCESS', customer_id_, order_no_);
        
        RETURN result_json_;
        
    EXCEPTION
        WHEN OTHERS THEN
            Error_SYS.Record_General('GET_ORDER_STATUS', 'Error: ' || SQLERRM);
            RETURN '{"error": "Internal error"}';
    END Get_Customer_Order_Status___;
    """

    function_name = "Get_Customer_Order_Status___"

    print("📊 Original Code Length:", len(test_code), "characters")
    print()

    # Test the summary generation (which creates optimized context)
    print("🏗️ Generating Summary with Optimized Context...")
    summary = opt.create_unixcoder_summary(test_code, function_name)
    print(f"Summary: {summary}")
    print()

    # Get the optimized context that was created
    optimized_context = opt.get_last_optimized_context()
    print("📝 Optimized Context Structure:")
    print(json.dumps(optimized_context, indent=2))
    print()

    # Test the embedding context string
    embedding_context = opt.create_embedding_context(test_code, function_name)
    print("🎯 Compact Embedding Context String:")
    print(f"'{embedding_context}'")
    print(f"Length: {len(embedding_context)} characters")
    print()

    # Compare space efficiency
    full_context_json = json.dumps(optimized_context, indent=2)
    print("📊 Space Efficiency Analysis:")
    print(f"Full JSON: {len(full_context_json)} characters")
    print(f"Compact String: {len(embedding_context)} characters")
    print(
        f"Space Savings: {len(full_context_json) - len(embedding_context)} characters ({((len(full_context_json) - len(embedding_context))/len(full_context_json)*100):.1f}%)"
    )
    print()

    # Test with multiple code samples
    test_samples = [
        (
            "Simple Procedure",
            """
        PROCEDURE Update_Customer_Status___ (customer_id_ IN VARCHAR2, status_ IN VARCHAR2) IS
        BEGIN
            UPDATE customer SET status = status_ WHERE id = customer_id_;
            IF SQL%ROWCOUNT = 0 THEN
                Error_SYS.Record_Not_Exist('Customer', customer_id_);
            END IF;
        END Update_Customer_Status___;
        """,
        ),
        (
            "Complex Function",
            """
        FUNCTION Calculate_Order_Total___ (order_id_ IN NUMBER) RETURN NUMBER IS
            cursor item_cursor IS SELECT price, quantity, discount FROM order_items WHERE order_id = order_id_;
            total_ NUMBER := 0;
            tax_rate_ NUMBER;
        BEGIN
            FOR item_rec IN item_cursor LOOP
                total_ := total_ + (item_rec.price * item_rec.quantity * (1 - item_rec.discount/100));
            END LOOP;
            
            SELECT tax_rate INTO tax_rate_ FROM tax_config WHERE ROWNUM = 1;
            total_ := total_ * (1 + tax_rate_/100);
            
            RETURN ROUND(total_, 2);
        EXCEPTION
            WHEN OTHERS THEN RETURN 0;
        END Calculate_Order_Total___;
        """,
        ),
    ]

    print("🔍 Testing Multiple Samples:")
    for sample_name, sample_code in test_samples:
        print(f"\n{sample_name}:")
        sample_context = opt.create_embedding_context(sample_code, sample_name)
        print(f"  Context: {sample_context}")
        print(f"  Length: {len(sample_context)} chars")

        # Get detailed context for comparison
        sample_summary = opt.create_unixcoder_summary(sample_code, sample_name)
        sample_optimized = opt.get_last_optimized_context()

        business_logic = sample_optimized.get("business_logic", {})
        behavior = sample_optimized.get("behavior_patterns", {})

        print(f"  Functions: {len(business_logic.get('functions', []))}")
        print(f"  Procedures: {len(business_logic.get('procedures', []))}")
        print(f"  Active Patterns: {sum(1 for v in behavior.values() if v)}")
        print(f"  Complexity: {sample_optimized.get('complexity', 'unknown')}")

    print(f"\n✅ Optimized AST Context Extraction Test Complete!")
    print(
        "💡 The optimized context reduces space usage while preserving business-critical information"
    )
    print(
        "🎯 This enables more efficient embedding generation and semantic search within context limits"
    )


if __name__ == "__main__":
    asyncio.run(test_optimized_context())
