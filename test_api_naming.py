#!/usr/bin/env python3
"""
Simple test for API naming fix.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.embedding_processor import BuiltInPageRankAnalyzer


def test_api_naming():
    print("Testing API naming fix...")

    analyzer = BuiltInPageRankAnalyzer(Path("_work"))

    test_cases = [
        ("XdCustomEventUtil", "XD_CUSTOM_EVENT_UTIL"),
        (
            "XD_CUSTOM_EVENT_UTIL_API",
            "XD_CUSTOM_EVENT_UTIL_API",
        ),  # Should stay unchanged
        ("CUSTOM_EVENT_API", "CUSTOM_EVENT_API"),  # Should stay unchanged
        ("CustomEventAPI", "CUSTOM_EVENT_API"),
        ("TestAPI", "TEST_API"),
        ("TEST_UTIL_API", "TEST_UTIL_API"),  # Should stay unchanged
    ]

    print("\nAPI Name Conversion Tests:")
    print("=" * 50)

    for input_name, expected in test_cases:
        result = analyzer._fix_api_naming(input_name)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"{status} | '{input_name}' -> '{result}' (expected: '{expected}')")

        if result != expected:
            print(f"      ERROR: Expected '{expected}' but got '{result}'")

    print("\nBM25S Preprocessing Test:")
    print("=" * 50)

    # Test with a sample that should show fixed API names
    from ifs_cloud_mcp_server.embedding_processor import BM25SIndexer, FileMetadata

    indexer = BM25SIndexer(Path("embedding_checkpoints/search_indexes"))

    # Create test metadata with properly formatted API name
    test_metadata = FileMetadata(
        rank=1,
        file_path="test/path.plsql",
        relative_path="test/path.plsql",
        file_name="TestFile.plsql",
        api_name="XD_CUSTOM_EVENT_UTIL_API",  # Already properly formatted
        file_size_mb=1.0,
        api_calls=[],
        changelog_lines=["Added test functionality"],
        procedure_function_names=["Test_Procedure"],
    )

    preprocessed = indexer.prepare_text_for_bm25(
        content="Test content for preprocessing",
        metadata=test_metadata,
        full_content="Full file content for testing",
    )

    print("Sample preprocessed text:")
    print(preprocessed[:200] + "...")

    # Check that API name appears correctly in output
    if "apiname:xd_custom_event_util_api" in preprocessed.lower():
        print("✓ PASS: API name correctly preserved in preprocessed text")
    else:
        print("✗ FAIL: API name not found or incorrectly formatted")
        print("Looking for: 'apiname:xd_custom_event_util_api'")


if __name__ == "__main__":
    test_api_naming()
