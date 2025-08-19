#!/usr/bin/env python3
"""
Show BM25S preprocessing results - what text actually gets tokenized and indexed.

This script demonstrates the full preprocessing pipeline:
1. Load a sample file
2. Show original content (first 500 chars)
3. Show preprocessed text ready for BM25S
4. Show tokenized output
5. Show stemmed tokens
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.embedding_processor import (
    BM25SIndexer,
    EmbeddingCheckpointManager,
    FileMetadata,
)


def main():
    print("🔍 BM25S Preprocessing Pipeline Demo\n")

    # Initialize BM25S indexer to use its preprocessing methods
    search_indexes_dir = Path("embedding_checkpoints/search_indexes")
    indexer = BM25SIndexer(search_indexes_dir)

    # Load existing results to get sample files
    checkpoint_manager = EmbeddingCheckpointManager(Path("embedding_checkpoints"))
    all_results = checkpoint_manager.load_all_results()

    if not all_results:
        print("❌ No results found. Please run the embedding pipeline first.")
        return

    # Get a few interesting files to demonstrate
    sample_files = []
    for result in all_results[:5]:  # First 5 files
        if result.success:
            sample_files.append(result)

    print(f"📂 Demonstrating preprocessing on {len(sample_files)} sample files:\n")

    for i, result in enumerate(sample_files, 1):
        print(f"{'='*80}")
        print(f"SAMPLE {i}: {result.file_metadata.file_name}")
        print(f"API: {result.file_metadata.api_name}")
        print(f"Rank: {result.file_metadata.rank}")
        print(f"{'='*80}")

        # Load original file content
        try:
            with open(
                result.file_metadata.file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                original_content = f.read()
        except Exception as e:
            print(f"❌ Could not read file: {e}")
            continue

        # Show original content (first 500 chars)
        print("\n📄 ORIGINAL CONTENT (first 500 chars):")
        print("-" * 60)
        print(repr(original_content[:500]) + "...")

        # Show AI summary/excerpt if available
        if result.summary:
            print(f"\n🤖 AI SUMMARY ({len(result.summary)} chars):")
            print("-" * 60)
            print(
                repr(result.summary[:300]) + "..."
                if len(result.summary) > 300
                else repr(result.summary)
            )

        # Show preprocessed text ready for BM25S
        print(f"\n🔧 PREPROCESSED TEXT FOR BM25S:")
        print("-" * 60)

        preprocessed_text = indexer.prepare_text_for_bm25(
            content=result.summary or result.content_excerpt,
            metadata=result.file_metadata,
            full_content=original_content,
        )

        # Show first 1000 chars of preprocessed text
        print(
            repr(preprocessed_text[:1000]) + "..."
            if len(preprocessed_text) > 1000
            else repr(preprocessed_text)
        )

        # Show tokenization results
        print(f"\n🎯 TOKENIZATION RESULTS:")
        print("-" * 60)
        tokens = indexer._advanced_tokenize(preprocessed_text)
        stemmed_tokens = indexer._stem_tokens(tokens)

        print(f"Original tokens (first 50): {tokens[:50]}")
        print(f"Stemmed tokens (first 50): {stemmed_tokens[:50]}")
        print(f"Total tokens: {len(tokens)} -> {len(stemmed_tokens)} after stemming")

        # Show structure breakdown
        parts = preprocessed_text.split()
        structured_parts = [
            part
            for part in parts
            if ":" in part
            and part.split(":")[0]
            in ["filename", "apiname", "rank", "changelog", "procedures"]
        ]

        print(f"\n📋 STRUCTURED COMPONENTS:")
        print("-" * 60)
        for part in structured_parts[:10]:  # First 10 structured parts
            print(f"  {part}")
        if len(structured_parts) > 10:
            print(f"  ... and {len(structured_parts) - 10} more structured components")

        print(f"\n📊 STATISTICS:")
        print("-" * 60)
        print(f"Original content length: {len(original_content):,} chars")
        print(
            f"AI summary length: {len(result.summary) if result.summary else 0:,} chars"
        )
        print(f"Preprocessed text length: {len(preprocessed_text):,} chars")
        print(f"Token count: {len(tokens):,}")
        print(f"Unique tokens: {len(set(tokens)):,}")
        print(f"Stemmed unique tokens: {len(set(stemmed_tokens)):,}")

        print("\n" + "=" * 80 + "\n")

    # Show global stopwords info
    print("🛑 STOPWORDS ANALYSIS:")
    print("-" * 60)
    print(f"Total custom stopwords: {len(indexer.custom_stopwords)}")
    print(f"Sample stopwords: {list(indexer.custom_stopwords)[:20]}")
    print(f"PL/SQL keywords preserved: {len(indexer.plsql_keywords)}")
    print(f"Sample PL/SQL keywords: {list(indexer.plsql_keywords)[:10]}")


if __name__ == "__main__":
    main()
