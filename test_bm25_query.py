#!/usr/bin/env python3
"""
Test BM25S querying to identify the error.
"""

import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_bm25_query():
    try:
        print("Testing BM25S querying...")

        from ifs_cloud_mcp_server.embedding_processor import BM25SIndexer

        # Initialize indexer
        print("1. Initializing BM25S indexer...")
        indexer = BM25SIndexer(Path("embedding_checkpoints/search_indexes"))

        # Try to load existing index
        print("2. Loading existing BM25S index...")
        if indexer.load_existing_index():
            print("✓ BM25S index loaded successfully")
        else:
            print("✗ Failed to load BM25S index")
            return

        # Test basic query
        print("3. Testing basic query...")
        if hasattr(indexer, "bm25_index") and indexer.bm25_index:
            print("✓ BM25S index object exists")

            # Test query preprocessing
            test_query = "custom event utility procedure"
            print(f"4. Testing query preprocessing for: '{test_query}'")

            # Use the same preprocessing as documents
            query_tokens = indexer._advanced_tokenize(test_query)
            stemmed_tokens = indexer._stem_tokens(query_tokens)

            print(f"   Original query: {test_query}")
            print(f"   Tokenized: {query_tokens}")
            print(f"   Stemmed: {stemmed_tokens}")

            # Test BM25S search
            print("5. Testing BM25S search...")
            try:
                # Use the new search method
                results = indexer.search(test_query, top_k=5)
                print(f"✓ BM25S search successful, found {len(results)} results")

                # Show results
                for i, result in enumerate(results[:3]):  # Show first 3 results
                    print(f"   Result {i+1}: {result['file_name']}")
                    print(f"      Score: {result['score']:.4f}")
                    print(f"      API: {result['api_name']}")
                    print(f"      Rank: {result['rank']}")
                    print()

            except Exception as e:
                print(f"✗ BM25S search failed: {e}")
                traceback.print_exc()

        else:
            print("✗ BM25S index object not found")

    except Exception as e:
        print(f"✗ Error during BM25S testing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_bm25_query()
