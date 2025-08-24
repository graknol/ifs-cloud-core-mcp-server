#!/usr/bin/env python3
"""
Analyze AST Contents for Context Window Optimization
"""

from ifs_parser_integration import IFSCloudParserIntegration
import json


def analyze_ast_for_context_window():
    """Analyze what AST data would be most useful for embeddings/summarization"""

    parser = IFSCloudParserIntegration()

    # Test with a realistic IFS Cloud code sample
    realistic_code = """
    FUNCTION Get_Order_Status___ (
        company_id_ IN VARCHAR2,
        order_no_ IN VARCHAR2,
        line_no_ IN NUMBER DEFAULT NULL
    ) RETURN VARCHAR2 IS
        cursor get_order_info IS
            SELECT order_status, delivery_date, customer_id
            FROM customer_order
            WHERE company = company_id_
            AND order_no = order_no_
            AND (line_no_ IS NULL OR line_no = line_no_);
            
        status_ VARCHAR2(50);
        delivery_date_ DATE;
        customer_id_ VARCHAR2(20);
        result_ CLOB;
        
    BEGIN
        Client_SYS.Add_Info('PROCESSING_ORDER', 'Processing order: ' || order_no_);
        
        OPEN get_order_info;
        FETCH get_order_info INTO status_, delivery_date_, customer_id_;
        
        IF get_order_info%NOTFOUND THEN
            CLOSE get_order_info;
            Error_SYS.Record_Not_Exist('CustomerOrder', 'Order not found');
        END IF;
        
        CLOSE get_order_info;
        
        -- Build result JSON
        result_ := '{' ||
                   '"status": "' || status_ || '",' ||
                   '"delivery_date": "' || TO_CHAR(delivery_date_, 'YYYY-MM-DD') || '",' ||
                   '"customer": "' || customer_id_ || '"' ||
                   '}';
        
        -- Log the result
        Transaction_SYS.Log_Progress_Info('ORDER_STATUS_RETRIEVED');
        
        RETURN result_;
        
    EXCEPTION
        WHEN OTHERS THEN
            Error_SYS.Record_General('GET_ORDER_STATUS', 'Failed to get order status: ' || SQLERRM);
            RETURN NULL;
    END Get_Order_Status___;
    
    PROCEDURE Update_Order_Status___ (
        company_id_ IN VARCHAR2,
        order_no_ IN VARCHAR2,
        new_status_ IN VARCHAR2
    ) IS
        old_status_ VARCHAR2(50);
        
    BEGIN
        -- Check current status
        old_status_ := Get_Order_Status___(company_id_, order_no_);
        
        IF old_status_ IS NULL THEN
            Error_SYS.Record_General('UPDATE_ORDER', 'Order not found');
        END IF;
        
        -- Update the order
        UPDATE customer_order 
        SET order_status = new_status_,
            last_modified = SYSDATE,
            modified_by = Fnd_Session_API.Get_Fnd_User
        WHERE company = company_id_
        AND order_no = order_no_;
        
        IF SQL%ROWCOUNT = 0 THEN
            Error_SYS.Record_General('UPDATE_ORDER', 'No rows updated');
        END IF;
        
        -- Log the change
        Activity_Log_API.New(
            activity_type_ => 'ORDER_STATUS_CHANGE',
            reference1_ => company_id_,
            reference2_ => order_no_,
            note_ => 'Status changed from ' || old_status_ || ' to ' || new_status_
        );
        
    EXCEPTION
        WHEN OTHERS THEN
            Error_SYS.Record_General('UPDATE_ORDER_STATUS', SQLERRM);
            RAISE;
    END Update_Order_Status___;
    """

    print("🔬 Analyzing realistic IFS Cloud code for AST content...")
    print("=" * 60)

    # Parse the code
    result = parser.parse_code(realistic_code)

    print("📊 Parsed Structure:")
    print(f"Method: {result['method']}")
    print(f"Root Type: {result.get('root_type', 'N/A')}")
    print(f"Child Count: {result.get('child_count', 'N/A')}")
    print(f"Has Error: {result.get('has_error', 'N/A')}")

    print(f"\n🏗️ Structure Elements:")
    print(f"Functions: {result.get('functions', [])}")
    print(f"Procedures: {result.get('procedures', [])}")
    print(f"Variables: {result.get('variables', [])}")
    print(f"Parameters: {result.get('parameters', [])}")
    print(f"Data Types: {result.get('data_types', [])}")

    print(f"\n🎯 Pattern Recognition:")
    patterns = result.get("patterns", {})
    for pattern, detected in patterns.items():
        status = "✅" if detected else "❌"
        print(f"  {status} {pattern}: {detected}")

    # Analyze complexity
    complexity = parser.analyze_complexity(result)
    print(f"\n📈 Complexity Analysis:")
    print(f"Score: {complexity.get('complexity_score', 'N/A')}")
    print(f"Level: {complexity.get('complexity_level', 'N/A')}")

    print(f"\n📝 What's Most Useful for Context Windows:")
    print("=" * 50)

    # Analyze what would be most valuable for embeddings
    useful_for_embeddings = {
        "high_value": [
            "functions",
            "procedures",  # Core business logic identifiers
            "patterns.has_dml",
            "patterns.has_exception",  # Behavioral patterns
            "complexity_level",  # Complexity indicator
        ],
        "medium_value": [
            "variables",
            "parameters",  # Implementation details
            "patterns.has_conditional",
            "patterns.has_loop",  # Control flow
            "data_types",  # Type information
        ],
        "low_value": [
            "child_count",
            "root_type",  # Technical AST details
            "has_error",  # Parse status
        ],
    }

    for category, items in useful_for_embeddings.items():
        print(f"\n{category.upper().replace('_', ' ')} for embeddings:")
        for item in items:
            if "." in item:
                # Handle nested patterns
                main_key, sub_key = item.split(".")
                value = result.get(main_key, {}).get(sub_key, "N/A")
            else:
                value = result.get(item, "N/A")
            print(f"  • {item}: {value}")

    print(f"\n💡 Recommendations for Context Window Optimization:")
    print("=" * 55)
    recommendations = [
        "1. INCLUDE: Function/procedure names - essential for understanding purpose",
        "2. INCLUDE: Pattern flags (has_dml, has_exception) - indicate code behavior",
        "3. INCLUDE: Complexity level - helps prioritize important code",
        "4. SELECTIVE: Variable names only if unique/business-relevant",
        "5. EXCLUDE: Low-level AST details (child_count, root_type)",
        "6. COMPACT: Summarize data types to categories (STRING, NUMBER, DATE, etc.)",
        "7. FOCUS: Business logic patterns over syntactic details",
    ]

    for rec in recommendations:
        print(f"  {rec}")

    print(f"\n🎯 Optimized Context Format Example:")
    print("=" * 40)

    # Create an optimized context representation
    optimized_context = {
        "business_logic": {
            "functions": result.get("functions", []),
            "procedures": result.get("procedures", []),
        },
        "behavior_patterns": {
            "data_access": patterns.get("has_dml", False),
            "error_handling": patterns.get("has_exception", False),
            "complex_logic": patterns.get("has_conditional", False)
            or patterns.get("has_loop", False),
        },
        "complexity": complexity.get("complexity_level", "unknown"),
        "key_types": list(
            set([dt.split("(")[0] for dt in result.get("data_types", [])])
        ),  # Simplified types
    }

    print(json.dumps(optimized_context, indent=2))

    # Calculate token savings
    full_json = json.dumps(result, indent=2)
    optimized_json = json.dumps(optimized_context, indent=2)

    print(f"\n📊 Context Window Efficiency:")
    print(f"Full AST JSON: ~{len(full_json)} chars")
    print(f"Optimized JSON: ~{len(optimized_json)} chars")
    print(
        f"Space savings: ~{len(full_json) - len(optimized_json)} chars ({((len(full_json) - len(optimized_json))/len(full_json)*100):.1f}%)"
    )


if __name__ == "__main__":
    analyze_ast_for_context_window()
