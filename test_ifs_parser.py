#!/usr/bin/env python3
"""
Test IFS Cloud Parser Integration
"""

try:
    import tree_sitter_ifs_cloud_parser
    from tree_sitter import Language, Parser

    print("📦 Loading IFS Cloud Parser...")

    # Setup the language and parser
    language = Language(tree_sitter_ifs_cloud_parser.language())
    parser = Parser(language)

    print("✅ Parser setup successful!")

    # Test parsing a simple PL/SQL procedure
    test_code = b"""
    PROCEDURE Test_Procedure___ IS
    BEGIN
        NULL;
    END Test_Procedure___;
    """

    print("🔬 Parsing test code...")
    tree = parser.parse(test_code)

    print("📊 Parse tree:")
    print(tree.root_node.text)

    # Test with more complex code
    complex_code = b"""
    FUNCTION Get_Customer_Info___ (
        customer_id_ IN VARCHAR2
    ) RETURN VARCHAR2 IS
        result_ VARCHAR2(1000);
    BEGIN
        SELECT name INTO result_
        FROM customer
        WHERE id = customer_id_;
        
        RETURN result_;
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            RETURN 'Not found';
    END Get_Customer_Info___;
    """

    print("\n🔬 Parsing complex code...")
    complex_tree = parser.parse(complex_code)

    print("📊 Complex parse tree structure:")
    # Use available methods instead of sexp()
    root_node = complex_tree.root_node
    print(f"   Root type: {root_node.type}")
    print(f"   Text: {root_node.text}")

    # Walk through the tree structure
    def print_tree_structure(node, depth=0, max_depth=3):
        if depth > max_depth:
            return
        indent = "  " * depth
        print(f"{indent}- {node.type}: {node.text.decode('utf-8')[:50]}...")
        for child in node.children[:3]:  # Limit to first 3 children
            print_tree_structure(child, depth + 1, max_depth)

    print("📊 Tree structure (first 3 levels):")
    print_tree_structure(root_node)

    # Explore the tree structure
    print(f"\n📈 Tree stats:")
    print(f"   Root node type: {complex_tree.root_node.type}")
    print(f"   Child count: {complex_tree.root_node.child_count}")
    print(f"   Has error: {complex_tree.root_node.has_error}")

    if complex_tree.root_node.child_count > 0:
        print(f"   First child type: {complex_tree.root_node.children[0].type}")

except Exception as e:
    print(f"❌ Error testing parser: {e}")
    import traceback

    traceback.print_exc()
