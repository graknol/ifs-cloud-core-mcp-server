#!/usr/bin/env python3
"""
Demo script for the Hybrid Search Engine.

Runs comprehensive tests to demonstrate search capabilities across different query types.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.hybrid_search import HybridSearchEngine, QueryType


class SearchDemo:
    """Comprehensive search engine demonstration."""

    def __init__(self, checkpoint_dir: str = "embedding_checkpoints"):
        """Initialize demo with search engine."""
        print("🚀 Initializing Hybrid Search Engine...")

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint directory not found: {checkpoint_path}")

        self.engine = HybridSearchEngine(checkpoint_path)

        # Validate initialization
        stats = self.engine.get_stats()
        if not stats.get("initialized", False):
            raise RuntimeError("Failed to initialize search engine")

        print("✅ Search engine ready!")
        print(
            f"📊 BM25S docs: {stats['bm25s_documents']:,}, FAISS embeddings: {stats['faiss_embeddings']:,}"
        )
        print("")

    def run_demo(self):
        """Run comprehensive search demonstration."""
        print("🎯 HYBRID SEARCH ENGINE DEMONSTRATION")
        print("=" * 80)

        # Test queries organized by type
        test_queries = {
            "📝 Functional Queries": [
                "customer order processing",
                "invoice generation and payment",
                "inventory management system",
                "purchase order workflow",
                "general ledger accounting",
            ],
            "🔧 Technical Queries": [
                "database connection handling",
                "error logging mechanism",
                "data validation procedures",
                "authentication security",
                "API REST endpoints",
            ],
            "🏗️ Structural Queries": [
                "Customer_Order_API",
                "INVOICE_PROCESSING",
                "Inventory_Management",
                "Purchase_Order_Line",
                "General_Ledger_Entry",
            ],
            "🔍 Mixed Queries": [
                "customer order API validation",
                "invoice processing error handling",
                "inventory database procedures",
                "purchase order REST service",
                "ledger entry authentication",
            ],
        }

        # Run tests for each category
        for category, queries in test_queries.items():
            self._demo_category(category, queries)

        # Performance comparison
        self._demo_performance_comparison()

        # Interactive session
        self._demo_interactive()

    def _demo_category(self, category: str, queries: List[str]):
        """Demonstrate searches for a category."""
        print(f"\n{category}")
        print("-" * 60)

        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")

            start_time = time.time()
            response = self.engine.search(query, top_k=5, explain_results=True)
            search_time = time.time() - start_time

            print(f"⏱️  Time: {search_time:.3f}s | Type: {response.query_type.value}")
            print(
                f"📊 Found: {response.total_found} | Fusion: {response.fusion_method}"
            )
            if response.rerank_applied:
                print("⚡ FlashRank reranking applied")

            # Show top 3 results
            for j, result in enumerate(response.results[:3], 1):
                print(f"  {j}. 📄 {result.file_name} (Score: {result.score:.4f})")
                print(f"     🔗 {result.api_name}")
                print(f"     🎯 {result.source} | {result.match_type}")
                if result.explanation:
                    print(f"     💡 {result.explanation}")

            if response.suggestions:
                print(f"💡 Suggestions: {', '.join(response.suggestions[:2])}")

    def _demo_performance_comparison(self):
        """Compare performance with and without reranking."""
        print(f"\n🏃 PERFORMANCE COMPARISON")
        print("-" * 60)

        test_query = "customer order processing workflow"

        # Without reranking
        print(f"\n🔍 Testing: '{test_query}'")
        print("\n📊 WITHOUT FlashRank:")
        start_time = time.time()
        response_no_rerank = self.engine.search(
            test_query, top_k=10, enable_rerank=False
        )
        time_no_rerank = time.time() - start_time

        print(f"⏱️  Time: {time_no_rerank:.3f}s")
        print(f"📈 Fusion: {response_no_rerank.fusion_method}")
        print("Top 3 results:")
        for i, result in enumerate(response_no_rerank.results[:3], 1):
            print(f"  {i}. {result.file_name} ({result.score:.4f}) - {result.source}")

        # With reranking
        print("\n⚡ WITH FlashRank:")
        start_time = time.time()
        response_with_rerank = self.engine.search(
            test_query, top_k=10, enable_rerank=True
        )
        time_with_rerank = time.time() - start_time

        print(f"⏱️  Time: {time_with_rerank:.3f}s")
        print(f"📈 Fusion: {response_with_rerank.fusion_method}")
        print("Top 3 results:")
        for i, result in enumerate(response_with_rerank.results[:3], 1):
            print(f"  {i}. {result.file_name} ({result.score:.4f}) - {result.source}")

        # Performance summary
        overhead = ((time_with_rerank - time_no_rerank) / time_no_rerank) * 100
        print(
            f"\n📊 FlashRank overhead: +{overhead:.1f}% (+{(time_with_rerank - time_no_rerank)*1000:.1f}ms)"
        )

    def _demo_interactive(self):
        """Interactive demo session."""
        print(f"\n💬 INTERACTIVE DEMO")
        print("-" * 60)
        print("Try these example queries or enter your own:")
        print("  • 'customer order validation'")
        print("  • 'INVOICE_API'")
        print("  • 'database error handling'")
        print("  • 'purchase order line items'")
        print("Type 'quit' to exit, 'stats' for statistics")

        while True:
            try:
                query = input("\n🔍 Demo> ").strip()

                if not query:
                    continue

                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 Demo complete!")
                    break

                if query.lower() == "stats":
                    stats = self.engine.get_stats()
                    print("\n📊 ENGINE STATISTICS:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                # Perform search
                print(f"🔍 Searching...")
                response = self.engine.search(query, top_k=5, explain_results=True)

                print(f"⚡ Results for '{query}':")
                print(f"📊 Found {response.total_found} in {response.search_time:.3f}s")
                print(f"🎯 Query type: {response.query_type.value}")
                print(f"🔄 Fusion: {response.fusion_method}")

                if not response.results:
                    print("❌ No results found")
                    if response.suggestions:
                        print(f"💡 Try: {', '.join(response.suggestions)}")
                    continue

                for i, result in enumerate(response.results, 1):
                    print(f"\n  {i}. 📄 {result.file_name}")
                    print(f"     🔗 {result.api_name}")
                    print(f"     ⭐ Score: {result.score:.4f}")
                    print(f"     🎯 {result.source} | {result.match_type}")
                    if result.snippet:
                        snippet = (
                            result.snippet[:100] + "..."
                            if len(result.snippet) > 100
                            else result.snippet
                        )
                        print(f"     📝 {snippet}")
                    if result.explanation:
                        print(f"     💡 {result.explanation}")

            except KeyboardInterrupt:
                print("\n👋 Demo complete!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Run the search demo."""
    try:
        demo = SearchDemo()
        demo.run_demo()

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted!")
        return 1

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
