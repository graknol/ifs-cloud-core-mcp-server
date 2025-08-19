#!/usr/bin/env python3
"""
Command-line interface for the Hybrid Search Engine.

Usage:
    python search_cli.py "query here" --top-k 10 --explain
    python search_cli.py --interactive
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.hybrid_search import HybridSearchEngine, SearchResponse


def format_results(response: SearchResponse, show_explanations: bool = False) -> str:
    """Format search results for console display."""

    output = []

    # Header
    output.append("=" * 80)
    output.append(f"🔍 HYBRID SEARCH RESULTS")
    output.append("=" * 80)
    output.append(f"Query: '{response.query}'")
    output.append(f"Query Type: {response.query_type.value}")
    output.append(
        f"Found: {response.total_found} documents ({len(response.results)} shown)"
    )
    output.append(f"Search Time: {response.search_time:.3f}s")
    output.append(
        f"Fusion: {response.fusion_method} (BM25S: {response.bm25s_count}, FAISS: {response.faiss_count})"
    )
    if response.rerank_applied:
        output.append("⚡ FlashRank reranking applied")
    output.append("")

    # Results
    for i, result in enumerate(response.results, 1):
        output.append(f"📄 RESULT #{i}")
        output.append("-" * 50)
        output.append(f"📁 File: {result.file_name}")
        output.append(f"🔗 API: {result.api_name}")
        output.append(f"📊 Rank: #{result.rank}")
        output.append(f"⚡ Score: {result.score:.4f}")
        output.append(f"🎯 Source: {result.source}")
        output.append(f"📝 Match: {result.match_type}")
        output.append(f"📄 Path: {result.file_path}")

        if result.snippet:
            output.append(f"📋 Snippet: {result.snippet}")

        if show_explanations and result.explanation:
            output.append(f"💡 Why: {result.explanation}")

        output.append("")

    # Suggestions
    if response.suggestions:
        output.append("💡 SUGGESTIONS:")
        for suggestion in response.suggestions:
            output.append(f"  • Try: '{suggestion}'")
        output.append("")

    return "\n".join(output)


def interactive_mode(engine: HybridSearchEngine):
    """Interactive search mode."""
    print("🚀 Interactive Hybrid Search Mode")
    print("Type 'quit' or 'exit' to stop, 'stats' for engine statistics")
    print("Add --explain to any query for detailed explanations")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n🔍 Search> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if user_input.lower() == "stats":
                stats = engine.get_stats()
                print("\n📊 ENGINE STATISTICS:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            # Check for --explain flag
            explain = "--explain" in user_input
            query = user_input.replace("--explain", "").strip()

            if not query:
                print("❌ Please provide a search query")
                continue

            # Perform search
            print(f"🔍 Searching for: '{query}'...")
            response = engine.search(query, top_k=10, explain_results=explain)

            # Display results
            formatted_output = format_results(response, show_explanations=explain)
            print(formatted_output)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Search error: {e}")


def main():
    parser = argparse.ArgumentParser(description="IFS Cloud Hybrid Search Engine")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to return"
    )
    parser.add_argument(
        "--explain", action="store_true", help="Include detailed explanations"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable FlashRank reranking"
    )
    parser.add_argument(
        "--checkpoint-dir", default="embedding_checkpoints", help="Checkpoint directory"
    )

    args = parser.parse_args()

    # Initialize search engine
    print("🚀 Initializing Hybrid Search Engine...")
    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("Please run the embedding pipeline first!")
        return 1

    engine = HybridSearchEngine(checkpoint_dir)

    # Check if engine initialized successfully
    stats = engine.get_stats()
    if not stats.get("initialized", False):
        print("❌ Failed to initialize search engine")
        return 1

    print("✅ Search engine ready!")
    print(
        f"📊 BM25S docs: {stats['bm25s_documents']:,}, FAISS embeddings: {stats['faiss_embeddings']:,}"
    )

    # Interactive mode
    if args.interactive or not args.query:
        interactive_mode(engine)
        return 0

    # Single query mode
    print(f"🔍 Searching for: '{args.query}'...")

    response = engine.search(
        query=args.query,
        top_k=args.top_k,
        enable_rerank=not args.no_rerank,
        explain_results=args.explain,
    )

    # Output results
    if args.json:
        # Convert to JSON-serializable format
        result_data = {
            "query": response.query,
            "query_type": response.query_type.value,
            "total_found": response.total_found,
            "search_time": response.search_time,
            "fusion_method": response.fusion_method,
            "bm25s_count": response.bm25s_count,
            "faiss_count": response.faiss_count,
            "rerank_applied": response.rerank_applied,
            "suggestions": response.suggestions,
            "results": [
                {
                    "rank": i + 1,
                    "file_name": r.file_name,
                    "api_name": r.api_name,
                    "file_rank": r.rank,
                    "score": r.score,
                    "source": r.source,
                    "match_type": r.match_type,
                    "file_path": r.file_path,
                    "snippet": r.snippet,
                    "explanation": r.explanation if args.explain else None,
                }
                for i, r in enumerate(response.results)
            ],
        }
        print(json.dumps(result_data, indent=2))
    else:
        formatted_output = format_results(response, show_explanations=args.explain)
        print(formatted_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
