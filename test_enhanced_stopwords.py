#!/usr/bin/env python3
"""Test the enhanced stopwords list to see the improvements."""

import sys

sys.path.append("src")

from src.ifs_cloud_mcp_server.embedding_processor import BM25SIndexer
from pathlib import Path
import tempfile


def test_stopwords():
    # Create temporary indexer to see stopwords
    with tempfile.TemporaryDirectory() as temp_dir:
        indexer = BM25SIndexer(Path(temp_dir))

        print(f"📊 Total stopwords: {len(indexer.custom_stopwords)}")

        # Group by type
        special_chars = [w for w in indexer.custom_stopwords if not w.isalpha()]
        alphabetic = [w for w in indexer.custom_stopwords if w.isalpha()]

        print(f"🔤 Special characters/symbols: {len(special_chars)}")
        print(f"🔡 Alphabetic stopwords: {len(alphabetic)}")

        print("\n🔤 Sample special characters:")
        print(sorted(special_chars)[:20])

        print("\n🔡 Sample alphabetic stopwords:")
        print(sorted(alphabetic)[:20])

        # Test text preprocessing
        test_text = "SELECT * FROM table WHERE id = 123 AND name <> 'test' -- comment"
        print(f"\n🧪 Test preprocessing:")
        print(f"Original: {test_text}")

        # Tokenize
        tokens = indexer._advanced_tokenize(test_text)
        print(f"Tokens: {tokens}")

        # Apply stemming
        stemmed = indexer._stem_tokens(tokens)
        print(f"Stemmed: {stemmed}")


if __name__ == "__main__":
    test_stopwords()
