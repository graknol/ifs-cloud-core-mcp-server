"""
State-of-the-Art Hybrid Search Engine for IFS Cloud Code

Combines:
- BM25S for lexical/exact matching (full source code)
- FAISS for semantic similarity (AI summaries)
- FlashRank for intelligent result fusion
- Advanced query preprocessing and result ranking

Architecture:
1. Query preprocessing (intent detection, expansion)
2. Parallel BM25S + FAISS searches
3. FlashRank fusion with query-aware scoring
4. Result contextualization and explanation
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Import numpy lazily to avoid CLI slowdown
# import numpy as np  # Moved to lazy loading
from flashrank import Ranker, RerankRequest

from .analysis_engine import (
    BM25SIndexer,
    FAISSIndexManager,
    BGEM3EmbeddingGenerator,
    EmbeddingCheckpointManager,
)

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the search engine can handle."""

    EXACT_MATCH = "exact_match"  # API names, function names
    SEMANTIC = "semantic"  # Business logic, functionality
    CODE_PATTERN = "code_pattern"  # SQL queries, error patterns
    MIXED = "mixed"  # Combination of above


@dataclass
class SearchConfig:
    """Configuration options for search requests."""

    preset_name: str = ""

    # Performance settings
    enable_faiss: bool = True  # Disable for slower hardware (semantic search)
    enable_flashrank: bool = True  # Disable for slower hardware (neural reranking)

    # Search behavior
    fetch_multiplier: int = 2  # How many extra results to fetch before filtering
    min_score_threshold: float = 0.0  # Minimum score to include in results

    # Hardware optimization presets
    @classmethod
    def fast_hardware(cls) -> "SearchConfig":
        """Full features for fast hardware with GPU."""
        return cls(
            preset_name="fast_hardware",
            enable_faiss=True,
            enable_flashrank=True,
            fetch_multiplier=4,
        )

    @classmethod
    def medium_hardware(cls) -> "SearchConfig":
        """Balanced for medium hardware - hybrid without FlashRank."""
        return cls(
            preset_name="medium_hardware",
            enable_faiss=True,
            enable_flashrank=False,
            fetch_multiplier=2,
        )

    @classmethod
    def slow_hardware(cls) -> "SearchConfig":
        """Optimized for slow hardware - BM25S + PageRank only."""
        return cls(
            preset_name="slow_hardware",
            enable_faiss=False,
            enable_flashrank=False,
            fetch_multiplier=1,  # Don't over-fetch on slow hardware
        )


@dataclass
class SearchResult:
    """Individual search result with rich metadata."""

    doc_id: int
    file_path: str
    file_name: str
    api_name: str
    rank: int
    score: float
    source: str  # 'bm25s', 'faiss', or 'hybrid'
    title: str
    snippet: str
    explanation: str
    match_type: str  # 'exact', 'semantic', 'mixed'
    content_preview: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response with metadata."""

    query: str
    query_type: QueryType
    results: List[SearchResult]
    total_found: int
    search_time: float
    fusion_method: str
    bm25s_count: int
    faiss_count: int
    rerank_applied: bool
    suggestions: List[str] = None
    search_config: Optional["SearchConfig"] = None


class QueryAnalyzer:
    """Analyzes and preprocesses search queries."""

    def __init__(self):
        self.api_patterns = [
            r"[A-Z_]+_API$",  # CUSTOM_EVENT_API
            r"[A-Za-z]+API$",  # CustomEventAPI
            r"[A-Za-z_]+\.[A-Za-z_]+",  # API.method calls
        ]

        self.code_patterns = [
            r"SELECT\s+.*FROM",  # SQL queries
            r"PROCEDURE\s+\w+",  # Procedure definitions
            r"FUNCTION\s+\w+",  # Function definitions
            r"Error_SYS\.",  # Error handling
            r"RAISE_APPLICATION_ERROR",  # Exception patterns
        ]

        self.semantic_indicators = [
            "how to",
            "what is",
            "why does",
            "business logic",
            "functionality",
            "purpose",
            "workflow",
            "process",
        ]

    def analyze_query(self, query: str) -> Tuple[QueryType, Dict[str, Any]]:
        """Analyze query to determine type and extract metadata."""
        query_lower = query.lower()

        analysis = {
            "original_query": query,
            "normalized_query": query_lower,
            "tokens": query.split(),
            "has_api_pattern": any(
                self._matches_pattern(query, p) for p in self.api_patterns
            ),
            "has_code_pattern": any(
                self._matches_pattern(query, p) for p in self.code_patterns
            ),
            "has_semantic_indicators": any(
                indicator in query_lower for indicator in self.semantic_indicators
            ),
            "suggested_expansions": self._suggest_expansions(query),
            "search_weights": {},
        }

        # Determine query type
        if analysis["has_api_pattern"] or query.isupper():
            query_type = QueryType.EXACT_MATCH
            analysis["search_weights"] = {"bm25s": 0.8, "faiss": 0.2}

        elif analysis["has_code_pattern"]:
            query_type = QueryType.CODE_PATTERN
            analysis["search_weights"] = {"bm25s": 0.7, "faiss": 0.3}

        elif analysis["has_semantic_indicators"]:
            query_type = QueryType.SEMANTIC
            analysis["search_weights"] = {"bm25s": 0.3, "faiss": 0.7}

        else:
            query_type = QueryType.MIXED
            analysis["search_weights"] = {"bm25s": 0.5, "faiss": 0.5}

        return query_type, analysis

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches regex pattern."""
        import re

        return bool(re.search(pattern, text, re.IGNORECASE))

    def _suggest_expansions(self, query: str) -> List[str]:
        """Suggest query expansions for better results."""
        expansions = []

        # API name variations
        if query.isupper() and "_" in query:
            # Add camelCase version
            camel_case = "".join(word.capitalize() for word in query.split("_"))
            expansions.append(camel_case)

        # Add common synonyms
        synonyms = {
            "error": ["exception", "failure", "issue"],
            "create": ["insert", "add", "new"],
            "update": ["modify", "change", "edit"],
            "delete": ["remove", "drop"],
            "get": ["retrieve", "fetch", "select"],
        }

        for word in query.lower().split():
            if word in synonyms:
                expansions.extend(synonyms[word])

        return expansions[:3]  # Limit to top 3 expansions


class HybridSearchEngine:
    """State-of-the-art hybrid search engine combining BM25S + FAISS + FlashRank."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        # Put search indexes directly under the version directory (parent of checkpoint_dir)
        self.search_indexes_dir = self.checkpoint_dir.parent

        # Initialize components
        self.query_analyzer = QueryAnalyzer()
        self.bm25_indexer = None
        self.faiss_manager = None
        self.embedding_generator = None
        self.checkpoint_manager = EmbeddingCheckpointManager(self.checkpoint_dir)

        # FlashRank for result fusion
        self.ranker = None
        self.ranker_model = "ms-marco-MiniLM-L-12-v2"  # Fast, effective model

        # Caches
        self._document_cache = {}
        self._embedding_cache = {}

        # Lazy loading flag
        self._np_loaded = False

        # Initialize search components
        self._initialize_search_components()

    def _ensure_numpy_loaded(self):
        """Lazily load numpy when needed."""
        if not self._np_loaded:
            try:
                global np
                import numpy as np

                self._np_loaded = True
                logger.debug("📦 NumPy loaded for hybrid search")
            except ImportError:
                raise ImportError(
                    "NumPy is required but not installed. Install with: pip install numpy"
                )

    def _initialize_search_components(self) -> bool:
        """Initialize BM25S, FAISS, and FlashRank components."""
        try:
            # Get the version directory (parent of checkpoint_dir)
            version_dir = self.checkpoint_dir.parent

            # Set up specific directories for each component
            bm25s_dir = version_dir / "bm25s"
            faiss_dir = version_dir / "faiss"

            # Initialize BM25S indexer with specific directory
            logger.info("🔍 Initializing BM25S search...")
            self.bm25_indexer = BM25SIndexer(bm25s_dir)
            if not self.bm25_indexer.load_existing_index():
                logger.error("❌ Failed to load BM25S index")
                return False

            # Initialize FAISS manager with specific directory
            logger.info("🔍 Initializing FAISS search...")
            self.faiss_manager = FAISSIndexManager(faiss_dir)
            if not self.faiss_manager.load_existing_index():
                logger.error("❌ Failed to load FAISS index")
                return False

            # Initialize embedding generator for query embeddings
            logger.info("🔍 Initializing embedding generator...")
            self.embedding_generator = BGEM3EmbeddingGenerator()
            if not self.embedding_generator.initialize_model():
                logger.error("❌ Failed to initialize embedding model")
                return False

            # Initialize FlashRank
            logger.info("🔍 Initializing FlashRank...")
            self.ranker = Ranker(model_name=self.ranker_model)

            logger.info("✅ Hybrid search engine initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize search components: {e}")
            return False

    def search(
        self,
        query: str = None,
        semantic_query: str = None,
        lexical_query: str = None,
        top_k: int = 10,
        config: Optional[SearchConfig] = None,
        enable_rerank: Optional[bool] = None,
        enable_faiss: Optional[bool] = None,
        explain_results: bool = False,
    ) -> SearchResponse:
        """
        Perform hybrid search with configurable performance options.

        Args:
            query: Legacy single search query (used if semantic_query/lexical_query not provided)
            semantic_query: Query used for FAISS semantic search and FlashRank reranking
            lexical_query: Query used for BM25S exact/lexical matching
            top_k: Number of results to return
            config: Search configuration (overrides individual enable_* parameters)
            enable_rerank: Whether to apply FlashRank reranking (legacy parameter)
            enable_faiss: Whether to use FAISS semantic search (legacy parameter)
            explain_results: Whether to include detailed explanations

        Performance Modes:
            - Fast hardware: SearchConfig.fast_hardware() - Full features
            - Medium hardware: SearchConfig.medium_hardware() - No FlashRank
            - Slow hardware: SearchConfig.slow_hardware() - BM25S + PageRank only
        """
        # Handle query parameter combinations
        if semantic_query is None and lexical_query is None:
            if query is None:
                raise ValueError(
                    "Must provide either 'query' or both 'semantic_query' and 'lexical_query'"
                )
            # Legacy mode: use single query for both
            semantic_query = query
            lexical_query = query
            primary_query = query
        else:
            # New mode: separate queries
            if semantic_query is None:
                semantic_query = lexical_query or ""
            if lexical_query is None:
                lexical_query = semantic_query or ""
            primary_query = semantic_query or lexical_query
        # Use config if provided, otherwise fall back to legacy parameters
        if config is None:
            config = SearchConfig(
                enable_faiss=enable_faiss if enable_faiss is not None else True,
                enable_flashrank=enable_rerank if enable_rerank is not None else True,
            )

        start_time = time.time()

        # Step 1: Analyze primary query to get search weights
        query_type, analysis = self.query_analyzer.analyze_query(primary_query)
        bm25_weight = analysis["search_weights"]["bm25s"]
        faiss_weight = analysis["search_weights"]["faiss"]

        # Adjust weights if FAISS is disabled
        if not config.enable_faiss:
            bm25_weight = 1.0
            faiss_weight = 0.0
            logger.info(f"🔍 FAISS disabled - using BM25S only")
        else:
            logger.info(
                f"🔍 Query type: {query_type.value}, BM25S: {bm25_weight:.1f}, FAISS: {faiss_weight:.1f}"
            )

        # Step 2: Fetch results from enabled sources
        fetch_limit = top_k * config.fetch_multiplier
        bm25_results = self._search_bm25s(lexical_query, analysis, fetch_limit)

        if config.enable_faiss:
            faiss_results = self._search_faiss(semantic_query, analysis, fetch_limit)
        else:
            faiss_results = []

        logger.info(
            f"📊 Fetched BM25S: {len(bm25_results)}, FAISS: {len(faiss_results)}"
        )

        # Step 3: Apply PageRank boosting to each result set
        bm25_boosted = self._apply_pagerank_boosting(bm25_results)
        if config.enable_faiss:
            faiss_boosted = self._apply_pagerank_boosting(faiss_results)
        else:
            faiss_boosted = []

        # Step 4: Calculate how many results to take from each source
        if config.enable_faiss:
            # Respect ColBERT weights to get exactly top_k total documents
            total_weight = bm25_weight + faiss_weight
            bm25_count = int((bm25_weight / total_weight) * top_k)
            faiss_count = top_k - bm25_count  # Ensure exact total
        else:
            # FAISS disabled - take all from BM25S
            bm25_count = top_k
            faiss_count = 0

        logger.info(
            f"📊 Taking BM25S: {bm25_count}, FAISS: {faiss_count} (total: {top_k})"
        )

        # Step 5: Select weighted results from each source
        selected_bm25 = bm25_boosted[:bm25_count]
        selected_faiss = faiss_boosted[:faiss_count] if config.enable_faiss else []

        # Step 6: Combine selected results, handling duplicates
        combined_results = self._merge_weighted_results(selected_bm25, selected_faiss)

        # Step 7: Apply FlashRank for final ordering (if enabled)
        if config.enable_flashrank and len(combined_results) > 1:
            final_results = self._rerank_with_flashrank(
                semantic_query, combined_results
            )
            if config.enable_faiss:
                fusion_method = "pagerank_colbert_flashrank"
            else:
                fusion_method = "pagerank_bm25s_flashrank"
            rerank_applied = True
        else:
            final_results = combined_results
            if config.enable_faiss:
                fusion_method = "pagerank_colbert_weighted"
            else:
                fusion_method = "pagerank_bm25s_only"
            rerank_applied = False

        # Step 8: Take exactly top_k results
        final_results = final_results[:top_k]

        # Apply score threshold filtering if configured
        if config.min_score_threshold > 0:
            final_results = [
                r for r in final_results if r.score >= config.min_score_threshold
            ]

        # Step 9: Enrich results with explanations
        if explain_results:
            final_results = self._add_explanations(
                final_results, primary_query, analysis
            )

        # Step 10: Build response
        search_time = time.time() - start_time

        # Determine search mode for reporting
        if config.enable_faiss and config.enable_flashrank:
            search_mode = "Full Hybrid (BM25S + FAISS + PageRank + FlashRank)"
        elif config.enable_faiss and not config.enable_flashrank:
            search_mode = "Hybrid without FlashRank (BM25S + FAISS + PageRank)"
        elif not config.enable_faiss and config.enable_flashrank:
            search_mode = "BM25S with FlashRank (BM25S + PageRank + FlashRank)"
        else:
            search_mode = "BM25S Only (BM25S + PageRank)"

        logger.info(f"🎯 Search mode: {search_mode}")

        response = SearchResponse(
            query=primary_query,
            query_type=query_type,
            results=final_results,
            total_found=len(combined_results),
            search_time=search_time,
            fusion_method=fusion_method,
            bm25s_count=len(bm25_results),
            faiss_count=len(faiss_results),
            rerank_applied=rerank_applied,
            suggestions=analysis.get("suggested_expansions", []),
            search_config=config,
        )

        logger.info(
            f"✅ Search completed in {search_time:.3f}s: {len(final_results)} results"
        )
        return response

    def _search_bm25s(
        self, query: str, analysis: Dict, top_k: int
    ) -> List[SearchResult]:
        """Perform BM25S lexical search."""
        if not self.bm25_indexer or not self.bm25_indexer.bm25_index:
            return []

        try:
            # Preprocess query for BM25S
            query_tokens = self.bm25_indexer._advanced_tokenize(query)
            stemmed_tokens = self.bm25_indexer._stem_tokens(query_tokens)

            # Search BM25S index
            scores, doc_indices = self.bm25_indexer.bm25_index.retrieve(
                self.bm25_indexer.corpus_texts, stemmed_tokens, k=top_k
            )

            results = []
            for score, doc_idx in zip(scores[0], doc_indices[0]):
                if score > 0:  # Only include positive scores
                    doc_info = self.bm25_indexer.get_document_info(int(doc_idx))
                    if doc_info:
                        result = SearchResult(
                            doc_id=int(doc_idx),
                            file_path=doc_info["relative_path"],
                            file_name=doc_info["file_name"],
                            api_name=doc_info["api_name"],
                            rank=doc_info["rank"],
                            score=float(score),
                            source="bm25s",
                            title=f"{doc_info['api_name']} - {doc_info['file_name']}",
                            snippet=self._generate_snippet(
                                doc_info["relative_path"], query
                            ),
                            explanation="",
                            match_type="exact",
                        )
                        results.append(result)

            return results

        except Exception as e:
            logger.error(f"BM25S search failed: {e}")
            return []

    def _search_faiss(
        self, query: str, analysis: Dict, top_k: int
    ) -> List[SearchResult]:
        """Perform FAISS semantic search."""
        if not self.faiss_manager or not self.faiss_manager.faiss_index:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            if not query_embedding:
                return []

            # Search FAISS index
            self._ensure_numpy_loaded()
            query_vec = np.array([query_embedding], dtype=np.float32)
            scores, indices = self.faiss_manager.faiss_index.search(query_vec, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score > 0 and idx < len(self.faiss_manager.embedding_metadata):
                    metadata = self.faiss_manager.embedding_metadata[idx]

                    result = SearchResult(
                        doc_id=idx,
                        file_path=metadata["file_path"],
                        file_name=metadata["file_name"],
                        api_name=metadata["api_name"],
                        rank=metadata["rank"],
                        score=float(score),
                        source="faiss",
                        title=f"{metadata['api_name']} - {metadata['file_name']}",
                        snippet=self._generate_snippet(metadata["file_path"], query),
                        explanation="",
                        match_type="semantic",
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def _apply_pagerank_boosting(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Apply PageRank boosting to search results."""
        if not results:
            return results

        # Load PageRank scores if available
        pagerank_scores = self._load_pagerank_scores()
        if not pagerank_scores:
            logger.debug("No PageRank scores available, returning results unchanged")
            return sorted(results, key=lambda x: x.score, reverse=True)

        # Apply PageRank boosting
        boosted_results = []
        for result in results:
            # Get PageRank boost for this file
            pagerank_boost = pagerank_scores.get(result.file_path, 0.0)

            # Apply boost: base_score * (1 + pagerank_boost)
            # PageRank scores are typically 0.0-1.0, so this gives 1x-2x multiplier
            boosted_score = result.score * (1.0 + pagerank_boost)

            # Create boosted result
            boosted_result = SearchResult(
                doc_id=result.doc_id,
                file_path=result.file_path,
                file_name=result.file_name,
                api_name=result.api_name,
                rank=result.rank,
                score=boosted_score,
                source=result.source,
                title=result.title,
                snippet=result.snippet,
                explanation=f"{result.explanation} (PageRank boost: {pagerank_boost:.3f})",
                match_type=result.match_type,
            )
            boosted_results.append(boosted_result)

        # Sort by boosted scores
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Applied PageRank boosting to {len(boosted_results)} results")
        return boosted_results

    def _load_pagerank_scores(self) -> Dict[str, float]:
        """Load PageRank scores from ranked.jsonl file."""
        try:
            # Look for PageRank file in parent directory
            pagerank_file = self.checkpoint_dir.parent / "ranked.jsonl"
            if not pagerank_file.exists():
                return {}

            scores = {}
            with open(pagerank_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        file_path = data.get("relative_path") or data.get(
                            "file_path", ""
                        )
                        pagerank_score = data.get("pagerank_score", 0.0)
                        if file_path and isinstance(pagerank_score, (int, float)):
                            # Normalize PageRank score to 0.0-1.0 range for boosting
                            normalized_score = min(1.0, max(0.0, float(pagerank_score)))
                            scores[file_path] = normalized_score

            logger.info(f"Loaded PageRank scores for {len(scores)} files")
            return scores

        except Exception as e:
            logger.warning(f"Failed to load PageRank scores: {e}")
            return {}

    def _merge_weighted_results(
        self, bm25_results: List[SearchResult], faiss_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Merge results from both sources, handling duplicates."""
        seen_files = set()
        merged_results = []

        # Add BM25S results first (they've been pre-selected by count)
        for result in bm25_results:
            if result.file_path not in seen_files:
                seen_files.add(result.file_path)
                merged_results.append(result)

        # Add FAISS results, handling duplicates
        for result in faiss_results:
            if result.file_path not in seen_files:
                seen_files.add(result.file_path)
                merged_results.append(result)
            else:
                # File already exists, boost the existing result
                for existing in merged_results:
                    if existing.file_path == result.file_path:
                        # Average the scores and mark as hybrid
                        existing.score = (existing.score + result.score) / 2
                        existing.source = "hybrid"
                        existing.match_type = "mixed"
                        existing.explanation += f" + {result.source} match"
                        break

        # Sort by final scores
        merged_results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Merged results: {len(merged_results)} unique files")
        return merged_results

    def _combine_results(
        self,
        bm25_results: List[SearchResult],
        faiss_results: List[SearchResult],
        analysis: Dict,
    ) -> List[SearchResult]:
        """Combine and deduplicate results from BM25S and FAISS."""

        # Get weights from query analysis
        bm25_weight = analysis["search_weights"]["bm25s"]
        faiss_weight = analysis["search_weights"]["faiss"]

        # Normalize scores and apply weights
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results)
            for result in bm25_results:
                result.score = (result.score / max_bm25) * bm25_weight

        if faiss_results:
            max_faiss = max(r.score for r in faiss_results)
            for result in faiss_results:
                result.score = (result.score / max_faiss) * faiss_weight

        # Combine results, handling duplicates
        seen_files = set()
        combined = []

        # Add all unique results
        for result in bm25_results + faiss_results:
            file_key = result.file_path

            if file_key not in seen_files:
                seen_files.add(file_key)
                combined.append(result)
            else:
                # File already seen - boost score if it appears in both searches
                for existing in combined:
                    if existing.file_path == file_key:
                        existing.score += (
                            result.score * 0.5
                        )  # Boost for appearing in both
                        existing.source = "hybrid"
                        existing.match_type = "mixed"
                        break

        # Sort by score
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined

    def _rerank_with_flashrank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using FlashRank for optimal relevance."""
        if not self.ranker or len(results) <= 1:
            return results

        try:
            # Prepare passages for reranking
            passages = []
            for result in results:
                # Create rich passage text for reranking
                passage_text = f"{result.title}. {result.snippet}"
                if result.api_name:
                    passage_text = f"API: {result.api_name}. " + passage_text

                passages.append(
                    {
                        "id": result.doc_id,
                        "text": passage_text,
                        "meta": {
                            "original_score": result.score,
                            "source": result.source,
                        },
                    }
                )

            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)

            # Apply FlashRank
            reranked = self.ranker.rerank(rerank_request)

            # Update results with new scores
            reranked_results = []
            for item in reranked:
                # Find original result
                original_result = next(r for r in results if r.doc_id == item["id"])

                # Update with reranked score
                original_result.score = item["score"]
                original_result.source = f"{original_result.source}+flashrank"
                reranked_results.append(original_result)

            return reranked_results

        except Exception as e:
            logger.warning(f"FlashRank reranking failed: {e}")
            return results

    def _generate_snippet(
        self, file_path: str, query: str, max_chars: int = 200
    ) -> str:
        """Generate contextual snippet from file content."""
        try:
            if file_path in self._document_cache:
                content = self._document_cache[file_path]
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                self._document_cache[file_path] = content

            # Find best snippet containing query terms
            query_words = query.lower().split()
            lines = content.split("\n")

            best_line = ""
            best_score = 0

            for line in lines:
                line_lower = line.lower()
                score = sum(1 for word in query_words if word in line_lower)
                if score > best_score and len(line.strip()) > 10:
                    best_score = score
                    best_line = line.strip()

            # Truncate if too long
            if len(best_line) > max_chars:
                best_line = best_line[:max_chars] + "..."

            return best_line or "No preview available"

        except Exception as e:
            logger.warning(f"Failed to generate snippet for {file_path}: {e}")
            return "No preview available"

    def _add_explanations(
        self, results: List[SearchResult], query: str, analysis: Dict
    ) -> List[SearchResult]:
        """Add detailed explanations for why each result was returned."""

        for result in results:
            explanations = []

            # Explain match type
            if result.match_type == "exact":
                explanations.append("🎯 Exact match found in source code")
            elif result.match_type == "semantic":
                explanations.append("🧠 Semantic similarity to your query")
            else:
                explanations.append("🔀 Found through both exact and semantic matching")

            # Explain source
            if result.source == "bm25s":
                explanations.append("📝 Matched through lexical search")
            elif result.source == "faiss":
                explanations.append("🔍 Matched through embedding similarity")
            elif "flashrank" in result.source:
                explanations.append("⚡ Reranked by FlashRank for optimal relevance")

            # Explain ranking factors
            if result.rank <= 10:
                explanations.append(f"⭐ High importance file (rank #{result.rank})")

            result.explanation = " • ".join(explanations)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        stats = {
            "bm25s_documents": (
                len(self.bm25_indexer.corpus_texts) if self.bm25_indexer else 0
            ),
            "faiss_embeddings": (
                len(self.faiss_manager.embeddings) if self.faiss_manager else 0
            ),
            "cache_size": len(self._document_cache),
            "ranker_model": self.ranker_model,
            "initialized": all(
                [
                    self.bm25_indexer is not None,
                    self.faiss_manager is not None,
                    self.embedding_generator is not None,
                    self.ranker is not None,
                ]
            ),
        }
        return stats
