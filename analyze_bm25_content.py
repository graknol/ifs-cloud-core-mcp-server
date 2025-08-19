#!/usr/bin/env python3
"""
Test what content is actually in the BM25S index - summary vs full content.
"""

import sys
from pathlib import Path
import pickle
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.embedding_processor import (
    BM25SIndexer,
    EmbeddingCheckpointManager
)

def analyze_bm25_content():
    print("Analyzing BM25S Index Content")
    print("=" * 50)
    
    # Load the BM25S index and corpus
    search_indexes_dir = Path("embedding_checkpoints/search_indexes")
    
    # Check if index files exist
    bm25_corpus_file = search_indexes_dir / "bm25s_corpus.pkl"
    bm25_metadata_file = search_indexes_dir / "bm25s_metadata.json"
    doc_mapping_file = search_indexes_dir / "bm25s_doc_mapping.json"
    
    if not all(f.exists() for f in [bm25_corpus_file, bm25_metadata_file, doc_mapping_file]):
        print("❌ BM25S index files not found. Please run reindex first.")
        return
    
    # Load corpus texts
    print("📂 Loading BM25S corpus...")
    with open(bm25_corpus_file, 'rb') as f:
        corpus_texts = pickle.load(f)
    
    # Load document mapping
    with open(doc_mapping_file, 'r') as f:
        doc_mapping = json.load(f)
    
    # Load checkpoint manager to get original results
    checkpoint_manager = EmbeddingCheckpointManager(Path("embedding_checkpoints"))
    all_results = checkpoint_manager.load_all_results()
    
    print(f"📊 Found {len(corpus_texts)} documents in BM25S corpus")
    print(f"📊 Found {len(all_results)} original processing results")
    print(f"📊 Found {len(doc_mapping)} document mappings")
    
    # Analyze a few sample documents
    print("\n🔍 CONTENT ANALYSIS")
    print("=" * 50)
    
    for i in range(min(3, len(corpus_texts))):
        print(f"\n📄 DOCUMENT {i+1}:")
        print("-" * 30)
        
        # Get BM25S preprocessed text
        bm25_text = corpus_texts[i]
        
        # Find corresponding original result
        doc_info = doc_mapping.get(str(i))
        if not doc_info:
            print("❌ No document mapping found")
            continue
            
        file_path = doc_info['file_path'] if 'file_path' in doc_info else doc_info['relative_path']
        
        # Find original result by file path
        original_result = None
        for result in all_results:
            # Compare relative paths since doc mapping uses relative paths
            result_relative = str(Path(result.file_metadata.file_path).relative_to(Path.cwd())) if Path(result.file_metadata.file_path).is_absolute() else result.file_metadata.file_path
            if result_relative == doc_info.get('relative_path') or result.file_metadata.file_path == file_path:
                original_result = result
                break
                
        if not original_result:
            print(f"❌ No original result found for {file_path}")
            print(f"   Looking for relative path: {doc_info.get('relative_path')}")
            continue
        
        print(f"File: {doc_info['file_name']}")
        print(f"API: {doc_info['api_name']}")
        print(f"Rank: {doc_info['rank']}")
        
        # Load actual file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                actual_file_content = f.read()
        except Exception as e:
            print(f"❌ Could not read file: {e}")
            continue
        
        # Analyze what's in BM25S vs original sources
        print(f"\n📈 CONTENT STATISTICS:")
        print(f"  Original file size: {len(actual_file_content):,} chars")
        print(f"  AI summary size: {len(original_result.summary) if original_result.summary else 0:,} chars")
        print(f"  BM25S processed size: {len(bm25_text):,} chars")
        print(f"  BM25S token count: {len(bm25_text.split()):,}")
        
        # Check for content indicators
        print(f"\n🔍 CONTENT INDICATORS:")
        
        # Look for structured parts
        has_filename = "filename:" in bm25_text
        has_apiname = "apiname:" in bm25_text
        has_rank = "rank:" in bm25_text
        has_changelog = "changelog:" in bm25_text
        has_procedures = "procedures:" in bm25_text
        
        print(f"  ✓ Has filename: {has_filename}")
        print(f"  ✓ Has apiname: {has_apiname}")
        print(f"  ✓ Has rank: {has_rank}")
        print(f"  ✓ Has changelog: {has_changelog}")
        print(f"  ✓ Has procedures: {has_procedures}")
        
        # Check for source code patterns vs summary patterns
        source_indicators = [
            "procedure", "function", "begin", "end", "declare", "cursor",
            "select", "insert", "update", "delete", "create", "alter"
        ]
        
        summary_indicators = [
            "primary business purpose", "system importance", "key functionality",
            "business domain", "integration patterns", "architectural"
        ]
        
        source_count = sum(1 for indicator in source_indicators if indicator in bm25_text.lower())
        summary_count = sum(1 for indicator in summary_indicators if indicator in bm25_text.lower())
        
        print(f"  📝 Source code indicators found: {source_count}/{len(source_indicators)}")
        print(f"  🤖 AI summary indicators found: {summary_count}/{len(summary_indicators)}")
        
        # Show sample of the actual BM25S content
        print(f"\n📄 BM25S CONTENT SAMPLE (first 500 chars):")
        print("-" * 40)
        print(repr(bm25_text[:500]) + "...")
        
        # Compare with AI summary sample if available
        if original_result.summary:
            print(f"\n🤖 ORIGINAL AI SUMMARY SAMPLE (first 300 chars):")
            print("-" * 40)
            print(repr(original_result.summary[:300]) + "...")
            
            # Check if AI summary content appears in BM25S
            summary_words = original_result.summary.lower().split()[:20]  # First 20 words
            summary_in_bm25 = any(word in bm25_text.lower() for word in summary_words if len(word) > 3)
            print(f"\n🔍 AI summary words found in BM25S: {summary_in_bm25}")
        
        # Check if source code content appears in BM25S
        source_lines = [line.strip() for line in actual_file_content.split('\n')[:50] if line.strip() and not line.strip().startswith('--')]
        if source_lines:
            sample_source_line = source_lines[0].lower()
            source_words = sample_source_line.split()[:10]  # First 10 words from first source line
            source_in_bm25 = any(word in bm25_text.lower() for word in source_words if len(word) > 3)
            print(f"🔍 Source code words found in BM25S: {source_in_bm25}")
        
        print(f"\n{'='*50}")
    
    # Overall analysis
    print(f"\n📊 OVERALL ANALYSIS")
    print("=" * 50)
    
    # Count documents with full content flag
    full_content_count = 0
    total_source_indicators = 0
    total_summary_indicators = 0
    
    for i in range(len(doc_mapping)):
        doc_info = doc_mapping.get(str(i), {})
        if doc_info.get('has_full_content', False):
            full_content_count += 1
    
    print(f"Documents with full_content flag: {full_content_count}/{len(doc_mapping)}")
    
    # Sample token analysis
    if corpus_texts:
        avg_tokens = sum(len(text.split()) for text in corpus_texts[:100]) / min(100, len(corpus_texts))
        print(f"Average tokens per document (first 100): {avg_tokens:.1f}")
    
    # Quick content analysis on a sample
    sample_size = min(50, len(corpus_texts))
    source_indicators = ["procedure", "function", "begin", "end", "declare", "cursor"]
    summary_indicators = ["primary business purpose", "system importance", "key functionality"]
    
    for i in range(sample_size):
        text = corpus_texts[i].lower()
        total_source_indicators += sum(1 for indicator in source_indicators if indicator in text)
        total_summary_indicators += sum(1 for indicator in summary_indicators if indicator in text)
    
    print(f"Source indicators in sample: {total_source_indicators}")
    print(f"Summary indicators in sample: {total_summary_indicators}")
    
    print(f"\n🎯 CONCLUSION:")
    if full_content_count > len(doc_mapping) * 0.8:
        print("✅ BM25S index appears to contain FULL SOURCE CODE")
    elif total_summary_indicators > total_source_indicators:
        print("🤖 BM25S index appears to contain AI SUMMARIES") 
    else:
        print("� BM25S index appears to contain SOURCE CODE (based on content indicators)")

if __name__ == "__main__":
    analyze_bm25_content()
