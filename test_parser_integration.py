#!/usr/bin/env python3
"""
Test IFS Cloud Parser Integration
"""


def test_parser_basic():
    """Test basic parser functionality"""
    try:
        import ifs_cloud_parser

        print("✅ ifs_cloud_parser imported successfully")

        # Check what the language function returns
        lang_capsule = ifs_cloud_parser.language()
        print(f"✅ language() returned: {type(lang_capsule)}")
        print(f"   Capsule name: {lang_capsule}")

    except Exception as e:
        print(f"❌ Error importing ifs_cloud_parser: {e}")
        return False

    try:
        from tree_sitter import Language, Parser

        print("✅ tree_sitter imported successfully")
    except Exception as e:
        print(f"❌ Error importing tree_sitter: {e}")
        return False

    try:
        # Try creating the language
        language = Language(lang_capsule)
        print("✅ Language created successfully")
    except Exception as e:
        print(f"❌ Error creating Language: {e}")
        print(
            f"   This suggests a version mismatch between tree-sitter and ifs-cloud-parser"
        )
        return False

    try:
        # Try creating parser
        parser = Parser(language)
        print("✅ Parser created successfully")
    except Exception as e:
        print(f"❌ Error creating Parser: {e}")
        return False

    try:
        # Test parsing
        code = b"PROCEDURE Test___ IS BEGIN NULL; END;"
        tree = parser.parse(code)
        print("✅ Code parsed successfully")
        print(f"   Root node: {tree.root_node.type}")
        print(f"   Children: {tree.root_node.child_count}")
        print(f"   S-expression: {tree.root_node.sexp()}")
        return True
    except Exception as e:
        print(f"❌ Error parsing code: {e}")
        return False


if __name__ == "__main__":
    print("🔬 Testing IFS Cloud Parser Integration")
    print("=" * 50)
    success = test_parser_basic()
    if success:
        print("🎉 Parser integration test PASSED!")
    else:
        print("💥 Parser integration test FAILED!")
