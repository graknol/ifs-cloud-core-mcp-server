#!/usr/bin/env python3
"""
FastAPI Web UI for IFS Cloud MCP Server with type-ahead search.
Provides a Meilisearch-like interface for exploring IFS Cloud codebases.
"""

import argparse
import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import tantivy
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict

from .indexer import IFSCloudIndexer, SearchResult
from .search_engine import IFSCloudSearchEngine
from .embedding_processor import ProductionEmbeddingFramework

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    limit: int = 20
    file_type: Optional[str] = None
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None


class SearchSuggestion(BaseModel):
    """Search suggestion model for type-ahead."""

    text: str
    type: str  # 'entity', 'function', 'module', 'file', 'frontend'
    score: float
    context: str  # Additional context like module or file


class WebUISearchResult(BaseModel):
    """Enhanced search result for web UI."""

    path: str
    name: str
    type: str
    content_preview: str
    score: float
    entities: List[str]
    functions: List[str] = []
    line_count: int
    complexity_score: float
    modified_time: datetime
    hash: str  # Unique content hash for React keys
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    entity_name: Optional[str] = None
    component: Optional[str] = None
    # Frontend elements
    pages: List[str] = []
    lists: List[str] = []
    groups: List[str] = []
    entitysets: List[str] = []
    iconsets: List[str] = []
    trees: List[str] = []
    navigators: List[str] = []
    contexts: List[str] = []
    # UI-specific fields
    highlight: str = ""
    tags: List[str] = []

    model_config = ConfigDict()


class FileContentResponse(BaseModel):
    """Enhanced file content response with search context."""

    path: str
    name: str
    type: str
    content: str
    highlighted_content: str
    size: int
    lines: int
    entities: List[str] = []
    functions: List[str] = []
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    entity_name: Optional[str] = None
    component: Optional[str] = None
    complexity_score: float = 0.0
    pagerank_score: float = 0.0
    modified_time: Optional[datetime] = None
    # Frontend elements
    pages: List[str] = []
    lists: List[str] = []
    groups: List[str] = []
    iconsets: List[str] = []
    trees: List[str] = []
    navigators: List[str] = []
    # Search context
    search_query: Optional[str] = None
    match_count: int = 0
    match_lines: List[int] = []


class IFSCloudWebUI:
    """Web UI application for IFS Cloud search."""

    def __init__(self, index_path: str = "./index"):
        self.indexer = IFSCloudIndexer(index_path)
        self.search_engine = IFSCloudSearchEngine(self.indexer)

        # Initialize embedding processor for hybrid search
        try:
            # Import get_data_directory function
            from .main import get_data_directory

            # Use proper data directory instead of current directory
            data_dir = get_data_directory()
            # For web UI, we could default to the first available version or require version selection
            # For now, let's look for any available version
            extracts_dir = data_dir / "extracts"

            work_dir = None
            analysis_file = None
            checkpoint_dir = None

            if extracts_dir.exists():
                # Find the first available version
                version_dirs = [d for d in extracts_dir.iterdir() if d.is_dir()]
                if version_dirs:
                    # Use the first available version
                    version_dir = version_dirs[0]
                    work_dir = version_dir / "_work"
                    analysis_file = version_dir / "comprehensive_plsql_analysis.json"
                    checkpoint_dir = version_dir / "embedding_checkpoints"

            # Only initialize if analysis file and checkpoint directory exist
            if (
                work_dir
                and analysis_file
                and analysis_file.exists()
                and checkpoint_dir
                and checkpoint_dir.exists()
            ):
                self.embedding_processor = ProductionEmbeddingFramework(
                    work_dir=work_dir,
                    analysis_file=analysis_file,
                    checkpoint_dir=checkpoint_dir,
                    model="phi4-mini:3.8b-q4_K_M",
                )

                # Add embedding processor to search engine for demo endpoints
                self.search_engine.embedding_processor = self.embedding_processor
                logger.info("✅ Embedding processor initialized successfully")
            else:
                self.embedding_processor = None
                self.search_engine.embedding_processor = None
                logger.info(
                    "⚠️ Embedding processor not initialized - missing required files or directories"
                )

        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize embedding processor: {e}")
            self.embedding_processor = None
            self.search_engine.embedding_processor = None

        self.app = FastAPI(
            title="IFS Cloud Explorer",
            description="Web UI for exploring IFS Cloud codebases with intelligent search",
            version="1.0.0",
        )
        self._setup_routes()
        self._setup_static_files()

        # Cache for suggestions
        self._suggestion_cache: Dict[str, List[SearchSuggestion]] = {}

    def _setup_static_files(self):
        """Setup static files and templates."""
        # Use package directory for static files and templates
        package_dir = Path(__file__).parent
        static_dir = package_dir / "static"
        templates_dir = package_dir / "templates"

        # Create directories if they don't exist
        static_dir.mkdir(exist_ok=True)
        templates_dir.mkdir(exist_ok=True)

        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Setup templates
        self.templates = Jinja2Templates(directory=str(templates_dir))

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Serve the main search interface."""
            return self.templates.TemplateResponse(
                "modern_react.html", {"request": request}
            )

        @self.app.get("/demo", response_class=HTMLResponse)
        async def demo_page(request: Request):
            """Serve the UniXcoder demo search interface."""
            return self.templates.TemplateResponse(
                "demo_search.html", {"request": request}
            )

        # ==================================================
        # HYBRID SEARCH DEMO ENDPOINTS (BM25S + FAISS)
        # ==================================================

        @self.app.get("/api/demo/status")
        async def demo_status():
            """Get status of the hybrid search engine."""
            try:
                # Check if embedding processor is available
                embedding_processor_available = (
                    hasattr(self.search_engine, "embedding_processor")
                    and self.search_engine.embedding_processor is not None
                )

                # Check BM25S index availability
                bm25s_available = False
                if embedding_processor_available:
                    bm25s_indexer = getattr(
                        self.search_engine.embedding_processor, "bm25_indexer", None
                    )
                    if bm25s_indexer:
                        bm25s_available = (
                            hasattr(bm25s_indexer, "bm25_index")
                            and bm25s_indexer.bm25_index is not None
                            and hasattr(bm25s_indexer, "doc_mapping")
                            and len(bm25s_indexer.doc_mapping) > 0
                        )

                # Check FAISS index availability
                faiss_available = False
                if embedding_processor_available:
                    faiss_manager = getattr(
                        self.search_engine.embedding_processor, "faiss_manager", None
                    )
                    if faiss_manager:
                        faiss_available = (
                            hasattr(faiss_manager, "faiss_index")
                            and faiss_manager.faiss_index is not None
                            and hasattr(faiss_manager, "embeddings")
                            and len(faiss_manager.embeddings) > 0
                        )

                status = {
                    "status": (
                        "ready" if (faiss_available and bm25s_available) else "partial"
                    ),
                    "available": True,
                    "components": {
                        "bm25s": bm25s_available,
                        "faiss": faiss_available,
                        "embedding_processor": embedding_processor_available,
                    },
                    "description": "Advanced hybrid search with BM25S lexical + FAISS semantic search",
                }
                return JSONResponse(status)
            except Exception as e:
                logger.error(f"Error getting demo status: {e}")
                return JSONResponse(
                    {"status": "error", "error": str(e), "available": False}
                )

        @self.app.get("/api/demo/search")
        async def demo_search_get(
            query: str = Query(..., description="Search query"),
            limit: int = Query(10, description="Maximum results"),
            search_type: str = Query(
                "hybrid", description="Search type: hybrid, bm25s, faiss"
            ),
            module: Optional[str] = Query(None, description="Filter by module"),
            min_score: float = Query(0.1, description="Minimum similarity score"),
        ):
            """Hybrid search demo endpoint (GET) - showcases BM25S + FAISS fusion."""
            try:
                logger.info(f"Demo search: {search_type} search for '{query}'")

                # Perform search using the configured search engine
                if search_type == "bm25s":
                    # BM25S lexical search only
                    results = await self._demo_bm25s_search(
                        query, limit, module, min_score
                    )
                elif search_type == "faiss":
                    # FAISS semantic search only
                    results = await self._demo_faiss_search(
                        query, limit, module, min_score
                    )
                else:
                    # Hybrid search (default)
                    results = self.search_engine.search(
                        query=query, limit=limit, module=module, include_related=True
                    )

                # Format results for demo
                demo_results = {
                    "query": query,
                    "search_type": search_type,
                    "total_results": len(results),
                    "results": [
                        {
                            "path": result.path,
                            "name": result.name,
                            "type": result.type,
                            "score": result.score,
                            "preview": (
                                result.content_preview[:200] + "..."
                                if len(result.content_preview) > 200
                                else result.content_preview
                            ),
                            "module": result.module,
                            "logical_unit": result.logical_unit,
                            "entity_name": result.entity_name,
                            "complexity_score": result.complexity_score,
                            "pagerank_score": result.pagerank_score,
                        }
                        for result in results
                    ],
                }

                return JSONResponse(demo_results)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Demo search error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/demo/search")
        async def demo_search_post(request: SearchRequest):
            """Hybrid search demo endpoint (POST)."""
            try:
                # Use the search engine for demonstration
                results = self.search_engine.search(
                    query=request.query,
                    limit=request.limit,
                    file_type=request.file_type,
                    module=request.module,
                    include_related=True,
                )

                demo_results = {
                    "query": request.query,
                    "search_type": "hybrid",
                    "total_results": len(results),
                    "results": [
                        {
                            "path": result.path,
                            "name": result.name,
                            "type": result.type,
                            "score": result.score,
                            "preview": (
                                result.content_preview[:300] + "..."
                                if len(result.content_preview) > 300
                                else result.content_preview
                            ),
                            "module": result.module,
                            "logical_unit": result.logical_unit,
                            "functions": result.functions,
                            "entities": result.entities,
                        }
                        for result in results
                    ],
                }

                return JSONResponse(demo_results)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Demo search POST error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/demo/compare")
        async def demo_compare_search_types(
            query: str = Query(..., description="Search query"),
            limit: int = Query(5, description="Results per search type"),
        ):
            """Compare different search types side by side."""
            try:
                comparison = {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "search_types": {},
                }

                # Try BM25S search
                try:
                    bm25s_results = await self._demo_bm25s_search(query, limit)
                    comparison["search_types"]["bm25s"] = {
                        "name": "BM25S Lexical Search",
                        "description": "Traditional keyword matching with advanced preprocessing",
                        "results": len(bm25s_results),
                        "top_results": [
                            {"path": r.path, "name": r.name, "score": r.score}
                            for r in bm25s_results[:3]
                        ],
                    }
                except Exception as e:
                    comparison["search_types"]["bm25s"] = {
                        "name": "BM25S Lexical Search",
                        "error": str(e),
                        "results": 0,
                    }

                # Try FAISS search
                try:
                    faiss_results = await self._demo_faiss_search(query, limit)
                    comparison["search_types"]["faiss"] = {
                        "name": "FAISS Semantic Search",
                        "description": "Vector similarity search using embeddings",
                        "results": len(faiss_results),
                        "top_results": [
                            {"path": r.path, "name": r.name, "score": r.score}
                            for r in faiss_results[:3]
                        ],
                    }
                except Exception as e:
                    comparison["search_types"]["faiss"] = {
                        "name": "FAISS Semantic Search",
                        "error": str(e),
                        "results": 0,
                    }

                # Try Hybrid search
                try:
                    hybrid_results = self.search_engine.search(
                        query=query, limit=limit, include_related=True
                    )
                    comparison["search_types"]["hybrid"] = {
                        "name": "Hybrid Search (BM25S + FAISS)",
                        "description": "Fusion of lexical and semantic search with intelligent ranking",
                        "results": len(hybrid_results),
                        "top_results": [
                            {"path": r.path, "name": r.name, "score": r.score}
                            for r in hybrid_results[:3]
                        ],
                    }
                except Exception as e:
                    comparison["search_types"]["hybrid"] = {
                        "name": "Hybrid Search",
                        "error": str(e),
                        "results": 0,
                    }

                return JSONResponse(comparison)
            except Exception as e:
                logger.error(f"Error in demo comparison: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        async def _demo_bm25s_search(
            self,
            query: str,
            limit: int = 10,
            module: Optional[str] = None,
            min_score: float = 0.1,
        ):
            """Helper method for BM25S-only search demo."""
            if (
                not hasattr(self.search_engine, "embedding_processor")
                or self.search_engine.embedding_processor is None
            ):
                return []

            # Use BM25S indexer directly
            bm25s_indexer = self.search_engine.embedding_processor.bm25s_indexer
            if bm25s_indexer is None:
                return []

            # Perform BM25S search
            bm25s_results = bm25s_indexer.search(query, top_k=limit)

            # Convert to SearchResult format
            results = []
            for doc_id, score in bm25s_results:
                if score < min_score:
                    continue

                # Get document info from mapping
                doc_info = bm25s_indexer.get_document_info(doc_id)
                if doc_info and (not module or doc_info.get("module") == module):
                    # Create a basic SearchResult
                    result = SearchResult(
                        path=doc_info["path"],
                        name=doc_info.get("name", doc_info["path"].split("/")[-1]),
                        type=doc_info.get("type", "unknown"),
                        content_preview=doc_info.get("content", "")[:500],
                        score=score,
                        entities=doc_info.get("entities", []),
                        line_count=doc_info.get("lines", 0),
                        complexity_score=doc_info.get("complexity_score", 0.0),
                        pagerank_score=doc_info.get("pagerank_score", 0.0),
                        modified_time=datetime.now(),
                        hash=doc_info.get("hash", ""),
                        module=doc_info.get("module"),
                        logical_unit=doc_info.get("logical_unit"),
                        entity_name=doc_info.get("entity_name"),
                    )
                    results.append(result)

            return results

        async def _demo_faiss_search(
            self,
            query: str,
            limit: int = 10,
            module: Optional[str] = None,
            min_score: float = 0.1,
        ):
            """Helper method for FAISS-only search demo."""
            if (
                not hasattr(self.search_engine, "embedding_processor")
                or self.search_engine.embedding_processor is None
            ):
                return []

            # Use FAISS search directly
            embedding_processor = self.search_engine.embedding_processor
            if embedding_processor.faiss_index is None:
                return []

            # Perform semantic search
            semantic_results = embedding_processor.search_semantic(query, top_k=limit)

            # Convert to SearchResult format
            results = []
            for result in semantic_results:
                if result.get("score", 0) < min_score:
                    continue

                if not module or result.get("module") == module:
                    search_result = SearchResult(
                        path=result["path"],
                        name=result.get("name", result["path"].split("/")[-1]),
                        type=result.get("type", "unknown"),
                        content_preview=result.get("content", "")[:500],
                        score=result.get("score", 0.0),
                        entities=result.get("entities", []),
                        line_count=result.get("lines", 0),
                        complexity_score=result.get("complexity_score", 0.0),
                        pagerank_score=result.get("pagerank_score", 0.0),
                        modified_time=datetime.now(),
                        hash=result.get("hash", ""),
                        module=result.get("module"),
                        logical_unit=result.get("logical_unit"),
                        entity_name=result.get("entity_name"),
                    )
                    results.append(search_result)

            return results

        @self.app.get("/api/search")
        async def search(
            query: str = Query(..., description="Search query"),
            limit: int = Query(20, description="Maximum results"),
            file_type: Optional[str] = Query(None, description="Filter by file type"),
            module: Optional[str] = Query(None, description="Filter by module"),
            logical_unit: Optional[str] = Query(
                None, description="Filter by logical unit"
            ),
            min_complexity: Optional[float] = Query(
                None, description="Minimum complexity"
            ),
            max_complexity: Optional[float] = Query(
                None, description="Maximum complexity"
            ),
        ):
            """Search endpoint with filtering."""
            try:
                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized")
                    raise HTTPException(
                        status_code=503,
                        detail="Search index is not initialized. Please rebuild the index.",
                    )

                logger.debug(
                    f"Performing search: query='{query}', limit={limit}, file_type={file_type}, module={module}, logical_unit={logical_unit}"
                )

                # Use intelligent search approach like the MCP tool
                results = self.search_engine.search(
                    query=query,
                    limit=limit,
                    file_type=file_type,
                    module=module,
                    logical_unit=logical_unit,
                    min_complexity=min_complexity,
                    max_complexity=max_complexity,
                    include_related=True,  # Enable intelligent search behavior
                )

                logger.debug(f"Search returned {len(results)} unique results")

                # Convert to web UI format with enhanced data
                web_results = []
                for result in results:
                    # Create tags based on content (excluding module/logical_unit as they have clickable badges now)
                    tags = []
                    if result.component:
                        tags.append(f"component:{result.component}")
                    if result.pages:
                        tags.append("has-pages")
                    if result.iconsets:
                        tags.append("has-iconsets")
                    if result.trees:
                        tags.append("has-trees")
                    if result.navigators:
                        tags.append("has-navigators")

                    # Create highlight snippet
                    highlight = self._create_highlight(result.content_preview, query)

                    web_result = WebUISearchResult(
                        path=result.path,
                        name=result.name,
                        type=result.type,
                        content_preview=result.content_preview,
                        score=result.score,
                        entities=result.entities,
                        line_count=result.line_count,
                        complexity_score=result.complexity_score,
                        modified_time=result.modified_time,
                        hash=result.hash,
                        module=result.module,
                        logical_unit=result.logical_unit,
                        entity_name=result.entity_name,
                        component=result.component,
                        pages=result.pages,
                        lists=result.lists,
                        groups=result.groups,
                        entitysets=result.entitysets,
                        iconsets=result.iconsets,
                        trees=result.trees,
                        navigators=result.navigators,
                        contexts=result.contexts,
                        highlight=highlight,
                        tags=tags,
                    )
                    web_results.append(web_result)

                # Use Pydantic's model_dump with mode='json' for proper serialization
                response_data = {
                    "results": [r.model_dump(mode="json") for r in web_results],
                    "total": len(web_results),
                    "query": query,
                    "filters": {
                        "file_type": file_type,
                        "module": module,
                        "logical_unit": logical_unit,
                        "complexity_range": [min_complexity, max_complexity],
                    },
                }
                return JSONResponse(response_data)

            except Exception as e:
                logger.error(f"Search error: {type(e).__name__}: {e}")
                logger.error(
                    f"Search parameters: query='{query}', limit={limit}, file_type={file_type}"
                )
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.post("/search")
        async def post_search(request: SearchRequest):
            """Search endpoint via POST."""
            try:
                # Check if search engine is ready
                if self.indexer._index is None:
                    logger.error("Index is not initialized")
                    raise HTTPException(
                        status_code=503,
                        detail="Search index is not initialized. Please rebuild the index.",
                    )

                logger.debug(
                    f"Performing POST search: query='{request.query}', limit={request.limit}"
                )

                # Perform search with filters using search engine
                results = self.search_engine.search(
                    query=request.query,
                    limit=request.limit,
                    file_type=getattr(request, "file_type", None),
                    module=getattr(request, "module", None),
                    logical_unit=getattr(request, "logical_unit", None),
                    min_complexity=getattr(request, "min_complexity", None),
                    max_complexity=getattr(request, "max_complexity", None),
                )

                # Convert results to web format
                web_results = []
                for result in results:
                    # Generate tags for better categorization (excluding module/logical_unit as they have clickable badges now)
                    tags = []
                    if result.component:
                        tags.append(f"component:{result.component}")
                    if result.pages:
                        tags.append("has-pages")
                    if result.iconsets:
                        tags.append("has-iconsets")
                    if result.trees:
                        tags.append("has-trees")
                    if result.navigators:
                        tags.append("has-navigators")

                    # Create highlight snippet
                    highlight = self._create_highlight(
                        result.content_preview, request.query
                    )

                    web_result = WebUISearchResult(
                        path=result.path,
                        name=result.name,
                        type=result.type,
                        content_preview=result.content_preview,
                        score=result.score,
                        entities=result.entities,
                        line_count=result.line_count,
                        complexity_score=result.complexity_score,
                        modified_time=result.modified_time,
                        hash=result.hash,
                        module=result.module,
                        logical_unit=result.logical_unit,
                        entity_name=result.entity_name,
                        component=result.component,
                        pages=result.pages,
                        lists=result.lists,
                        groups=result.groups,
                        iconsets=result.iconsets,
                        trees=result.trees,
                        navigators=result.navigators,
                        contexts=result.contexts,
                        tags=tags,
                        highlight=highlight,
                    )
                    web_results.append(web_result)

                # Use Pydantic's model_dump with mode='json' for proper serialization
                response_data = {
                    "results": [
                        result.model_dump(mode="json") for result in web_results
                    ],
                    "total": len(web_results),
                    "query": request.query,
                    "took_ms": 0,  # We could add timing if needed
                }
                return JSONResponse(response_data)

            except Exception as e:
                logger.error(f"POST Search error: {type(e).__name__}: {e}")
                logger.error(
                    f"Search parameters: query='{request.query}', limit={request.limit}"
                )
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.post("/search")
        async def post_search(request: SearchRequest):
            """Search endpoint via POST with JSON request."""
            try:
                # Check if search engine is ready
                if self.indexer._index is None:
                    logger.error("Index is not initialized")
                    raise HTTPException(
                        status_code=503,
                        detail="Search index is not initialized. Please rebuild the index.",
                    )

                logger.debug(
                    f"Performing POST search: query='{request.query}', limit={request.limit}"
                )

                # Perform search using search engine
                results = self.search_engine.search(
                    query=request.query,
                    limit=request.limit,
                    file_type=getattr(request, "file_type", None),
                    module=getattr(request, "module", None),
                    logical_unit=getattr(request, "logical_unit", None),
                    min_complexity=getattr(request, "min_complexity", None),
                    max_complexity=getattr(request, "max_complexity", None),
                )

                # Convert results to web UI format
                web_results = []
                for result in results:
                    # Create tags for classification
                    tags = []
                    if result.module:
                        tags.append(f"module:{result.module}")
                    if result.logical_unit:
                        tags.append(f"unit:{result.logical_unit}")
                    if result.component:
                        tags.append(f"component:{result.component}")
                    if result.pages:
                        tags.append("has-pages")
                    if result.iconsets:
                        tags.append("has-iconsets")
                    if result.trees:
                        tags.append("has-trees")
                    if result.navigators:
                        tags.append("has-navigators")

                    # Create highlight snippet
                    highlight = self._create_highlight(
                        result.content_preview, request.query
                    )

                    web_result = WebUISearchResult(
                        path=result.path,
                        name=result.name,
                        type=result.type,
                        content_preview=result.content_preview,
                        score=result.score,
                        entities=result.entities,
                        line_count=result.line_count,
                        complexity_score=result.complexity_score,
                        modified_time=result.modified_time,  # Keep as datetime, Pydantic will handle it
                        hash=result.hash,
                        module=result.module,
                        logical_unit=result.logical_unit,
                        entity_name=result.entity_name,
                        component=result.component,
                        pages=result.pages,
                        lists=result.lists,
                        groups=result.groups,
                        iconsets=result.iconsets,
                        trees=result.trees,
                        navigators=result.navigators,
                        contexts=result.contexts,
                        tags=tags,
                        highlight=highlight,
                    )
                    web_results.append(web_result)

                # Use Pydantic's model_dump with mode='json' for proper serialization
                response_data = {
                    "results": [
                        result.model_dump(mode="json") for result in web_results
                    ],
                    "total": len(web_results),
                    "query": request.query,
                }
                return JSONResponse(response_data)

            except Exception as e:
                logger.error(f"POST Search error: {type(e).__name__}: {e}")
                logger.error(
                    f"Search parameters: query='{request.query}', limit={request.limit}"
                )
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.get("/api/suggestions")
        async def get_suggestions(
            query: str = Query(..., description="Partial query for suggestions"),
            limit: int = Query(10, description="Maximum suggestions"),
        ):
            """Get type-ahead suggestions."""
            try:
                if len(query) < 2:  # Don't suggest for very short queries
                    return JSONResponse({"suggestions": []})

                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized for suggestions")
                    return JSONResponse(
                        {"suggestions": [], "error": "Index not initialized"}
                    )

                # Check cache first
                cache_key = f"{query}:{limit}"
                if cache_key in self._suggestion_cache:
                    return JSONResponse(
                        {
                            "suggestions": [
                                s.model_dump(mode="json")
                                for s in self._suggestion_cache[cache_key]
                            ]
                        }
                    )

                suggestions = await self._generate_suggestions(query, limit)
                self._suggestion_cache[cache_key] = suggestions

                return JSONResponse(
                    {"suggestions": [s.model_dump(mode="json") for s in suggestions]}
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/file-content")
        async def get_file_content(path: str = Query(..., description="File path")):
            """Get file content for viewing."""
            try:
                from pathlib import Path

                # Validate and resolve the file path
                file_path = Path(path)
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")

                if not file_path.is_file():
                    raise HTTPException(status_code=400, detail="Path is not a file")

                # Read file content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try with latin-1 encoding if UTF-8 fails
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()

                return JSONResponse(
                    {
                        "path": str(file_path),
                        "content": content,
                        "size": len(content),
                        "lines": content.count("\n") + 1 if content else 0,
                    }
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error reading file {path}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to read file: {str(e)}"
                )

        @self.app.post("/suggestions")
        async def post_suggestions(request: SearchRequest):
            """Get type-ahead suggestions via POST."""
            try:
                if len(request.query) < 2:  # Don't suggest for very short queries
                    return JSONResponse({"suggestions": []})

                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized for suggestions")
                    return JSONResponse(
                        {"suggestions": [], "error": "Index not initialized"}
                    )

                # Check cache first
                cache_key = f"{request.query}:{request.limit}"
                if cache_key in self._suggestion_cache:
                    return JSONResponse(
                        {
                            "suggestions": [
                                s.dict() for s in self._suggestion_cache[cache_key]
                            ]
                        }
                    )

                suggestions = await self._generate_suggestions(
                    request.query, request.limit
                )
                self._suggestion_cache[cache_key] = suggestions

                return JSONResponse(
                    {"suggestions": [s.model_dump(mode="json") for s in suggestions]}
                )

            except Exception as e:
                logger.error(f"Suggestions error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/stats")
        async def get_stats():
            """Get index statistics."""
            try:
                # Check if index is properly initialized
                if self.indexer._index is None:
                    return JSONResponse(
                        {
                            "total_files": 0,
                            "total_entities": 0,
                            "error": "Index not initialized. Please rebuild the index.",
                            "supported_extensions": list(
                                self.indexer.SUPPORTED_EXTENSIONS
                            ),
                            "index_path": str(self.indexer.index_path),
                        }
                    )

                stats = self.indexer.get_stats()
                return JSONResponse(stats)
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/index")
        async def rebuild_index(
            directory: str = Query(..., description="Directory to index")
        ):
            """Rebuild the index."""
            try:
                await self.indexer.index_directory(Path(directory))
                # Clear suggestion cache
                self._suggestion_cache.clear()
                return JSONResponse(
                    {"status": "success", "message": "Index rebuilt successfully"}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/file/{file_path:path}")
        async def get_file_content(
            file_path: str,
            search_query: Optional[str] = Query(
                None, description="Search query for highlighting"
            ),
        ):
            """Get enhanced file content with search highlighting and metadata."""
            try:
                path = Path(file_path)
                if not path.exists():
                    raise HTTPException(status_code=404, detail="File not found")

                content = path.read_text(encoding="utf-8", errors="ignore")

                # Get file metadata from search index if available
                file_metadata = await self._get_file_metadata(str(path))

                # Create highlighted content and find matches
                highlighted_content = content
                match_count = 0
                match_lines = []

                if search_query:
                    highlighted_content, match_count, match_lines = (
                        self._create_highlighted_content(content, search_query)
                    )

                response = FileContentResponse(
                    path=str(path),
                    name=path.name,
                    type=path.suffix,
                    content=content,
                    highlighted_content=highlighted_content,
                    size=len(content),
                    lines=len(content.split("\n")),
                    search_query=search_query,
                    match_count=match_count,
                    match_lines=match_lines,
                    **file_metadata,
                )

                return response

            except Exception as e:
                logger.error(f"Error getting file content for {file_path}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _get_file_metadata(self, file_path: str) -> dict:
        """Get file metadata from the search index."""
        try:
            if not self.indexer._index:
                return {}

            searcher = self.indexer._index.searcher()

            # Search for the specific file
            query = tantivy.Query.term_query(self.indexer._schema, "path", file_path)
            results = searcher.search(query, limit=1)

            if results.hits:
                _, doc_address = results.hits[0]
                doc = searcher.doc(doc_address)

                return {
                    "entities": (
                        doc.get_first("entities").split()
                        if doc.get_first("entities")
                        else []
                    ),
                    "functions": (
                        doc.get_first("functions").split()
                        if doc.get_first("functions")
                        else []
                    ),
                    "module": doc.get_first("module"),
                    "logical_unit": doc.get_first("logical_unit"),
                    "entity_name": doc.get_first("entity_name"),
                    "component": doc.get_first("component"),
                    "complexity_score": doc.get_first("complexity_score") or 0.0,
                    "pagerank_score": doc.get_first("pagerank_score") or 0.0,
                    "modified_time": datetime.fromisoformat(
                        str(doc.get_first("modified_time"))
                        if doc.get_first("modified_time")
                        else "1970-01-01T00:00:00"
                    ),
                    "pages": (
                        doc.get_first("pages").split() if doc.get_first("pages") else []
                    ),
                    "lists": (
                        doc.get_first("lists").split() if doc.get_first("lists") else []
                    ),
                    "groups": (
                        doc.get_first("groups").split()
                        if doc.get_first("groups")
                        else []
                    ),
                    "iconsets": (
                        doc.get_first("iconsets").split()
                        if doc.get_first("iconsets")
                        else []
                    ),
                    "trees": (
                        doc.get_first("trees").split() if doc.get_first("trees") else []
                    ),
                    "navigators": (
                        doc.get_first("navigators").split()
                        if doc.get_first("navigators")
                        else []
                    ),
                }
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")

        return {}

    def _create_highlighted_content(
        self, content: str, search_query: str
    ) -> tuple[str, int, List[int]]:
        """Create highlighted content with search matches marked.

        Returns:
            Tuple of (highlighted_content, match_count, match_lines)
        """
        if not search_query or not content:
            return content, 0, []

        lines = content.split("\n")
        highlighted_lines = []
        match_count = 0
        match_lines = []

        # Case-insensitive search
        query_lower = search_query.lower()

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            highlighted_line = line

            # Find all matches in this line
            start_pos = 0
            line_has_match = False

            while True:
                match_pos = line_lower.find(query_lower, start_pos)
                if match_pos == -1:
                    break

                # Mark this line as having a match
                if not line_has_match:
                    match_lines.append(line_num)
                    line_has_match = True

                # Highlight the match
                before = highlighted_line[:match_pos]
                match_text = highlighted_line[match_pos : match_pos + len(search_query)]
                after = highlighted_line[match_pos + len(search_query) :]

                highlighted_line = (
                    f"{before}<mark class='search-highlight'>{match_text}</mark>{after}"
                )
                match_count += 1

                # Update search position, accounting for added HTML
                start_pos = (
                    match_pos
                    + len(search_query)
                    + len("<mark class='search-highlight'></mark>")
                )
                line_lower = highlighted_line.lower()

            highlighted_lines.append(highlighted_line)

        highlighted_content = "\n".join(highlighted_lines)
        return highlighted_content, match_count, match_lines

    async def _generate_suggestions(
        self, query: str, limit: int
    ) -> List[SearchSuggestion]:
        """Generate type-ahead suggestions."""
        suggestions = []

        # Use both exact and wildcard searches for better suggestions
        query_lower = query.lower()

        # Try multiple search strategies
        search_queries = [
            query,  # Exact query
            f"{query}*",  # Prefix search
            f"*{query}*",  # Contains search
        ]

        all_results = []
        for search_query in search_queries:
            try:
                results = self.search_engine.search(query=search_query, limit=limit * 3)
                all_results.extend(results)
                if len(all_results) >= limit * 2:  # Stop early if we have enough
                    break
            except:
                continue  # Try next search strategy

        # If no results from advanced search, try basic search
        if not all_results:
            try:
                all_results = self.search_engine.search(query=query, limit=limit * 2)
            except:
                pass

        seen_suggestions = set()

        # Generate suggestions from results
        for result in all_results:
            if len(suggestions) >= limit:
                break

            # Add entity suggestions
            for entity in result.entities:
                if len(suggestions) >= limit:
                    break
                if (
                    query_lower in entity.lower()
                    and entity.lower() not in seen_suggestions
                ):
                    suggestions.append(
                        SearchSuggestion(
                            text=entity,
                            type="entity",
                            score=result.score,
                            context=f"in {result.name}",
                        )
                    )
                    seen_suggestions.add(entity.lower())

            # Add file name suggestions
            if (
                len(suggestions) < limit
                and query_lower in result.name.lower()
                and result.name.lower() not in seen_suggestions
            ):
                suggestions.append(
                    SearchSuggestion(
                        text=result.name,
                        type="file",
                        score=result.score,
                        context=f"{result.type} file",
                    )
                )
                seen_suggestions.add(result.name.lower())

            # Add module suggestions
            if (
                len(suggestions) < limit
                and result.module
                and query_lower in result.module.lower()
                and result.module.lower() not in seen_suggestions
            ):
                suggestions.append(
                    SearchSuggestion(
                        text=result.module,
                        type="module",
                        score=result.score,
                        context="IFS module",
                    )
                )
                seen_suggestions.add(result.module.lower())

            # Add frontend element suggestions with higher priority
            frontend_elements = [
                (result.pages, "page"),
                (result.lists, "list"),
                (result.groups, "group"),
                (result.iconsets, "iconset"),
                (result.trees, "tree"),
                (result.navigators, "navigator"),
                (result.contexts, "context"),
            ]

            for elements, element_type in frontend_elements:
                if len(suggestions) >= limit:
                    break
                for element in elements:
                    if len(suggestions) >= limit:
                        break
                    if (
                        query_lower in element.lower()
                        and element.lower() not in seen_suggestions
                    ):
                        suggestions.append(
                            SearchSuggestion(
                                text=element,
                                type="frontend",
                                score=result.score + 0.1,  # Boost frontend elements
                                context=f"{element_type} in {result.name}",
                            )
                        )
                        seen_suggestions.add(element.lower())

        # If still no suggestions, try common IFS terms
        if not suggestions and len(query) >= 2:
            common_terms = [
                "Activity",
                "Customer",
                "Order",
                "Project",
                "Delivery",
                "Structure",
                "Entity",
                "Projection",
                "Client",
                "Fragment",
                "Storage",
                "Views",
                "iconset",
                "tree",
                "navigator",
                "page",
                "list",
                "group",
            ]

            for term in common_terms:
                if query_lower in term.lower():
                    suggestions.append(
                        SearchSuggestion(
                            text=term,
                            type="common",
                            score=0.5,
                            context="common IFS term",
                        )
                    )
                if len(suggestions) >= limit:
                    break

        # Sort by score and limit
        suggestions.sort(key=lambda x: x.score, reverse=True)
        return suggestions[:limit]

    def _create_highlight(self, content: str, query: str) -> str:
        """Create highlighted content snippet."""
        if not query or not content:
            return content[:200] + "..." if len(content) > 200 else content

        # Simple highlighting - find query in content and add some context
        query_lower = query.lower()
        content_lower = content.lower()

        index = content_lower.find(query_lower)
        if index == -1:
            return content[:200] + "..." if len(content) > 200 else content

        # Get context around the match
        start = max(0, index - 50)
        end = min(len(content), index + len(query) + 50)

        snippet = content[start:end]

        # Add highlighting (will be handled by frontend)
        highlighted = snippet.replace(
            content[index : index + len(query)],
            f"<mark>{content[index:index + len(query)]}</mark>",
        )

        return highlighted


def find_available_port(start_port: int = 5700, max_port: int = 5799) -> int:
    """Find the first available port in the specified range.

    Args:
        start_port: Starting port number (default: 5700)
        max_port: Maximum port number to try (default: 5799)

    Returns:
        First available port number

    Raises:
        RuntimeError: If no available port is found in the range
    """
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"No available port found in range {start_port}-{max_port}")


# Main application entry point
if __name__ == "__main__":
    """Main entry point for Web UI."""
    from .main import get_data_directory

    # Use proper data directory by default
    default_index_path = get_data_directory() / "indexes" / "default"

    parser = argparse.ArgumentParser(description="IFS Cloud MCP Server Web UI")
    parser.add_argument(
        "--index-path",
        type=str,
        default=str(default_index_path),
        help=f"Path to store the Tantivy index (default: {default_index_path})",
    )

    args = parser.parse_args()

    # Create index directory if it doesn't exist
    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    import uvicorn

    # Create the web UI application
    web_ui = IFSCloudWebUI(index_path=index_path)

    # Find an available port in the 8000 series
    try:
        port = find_available_port()
        print("🚀 Starting IFS Cloud Web UI...")
        print(f"📊 Interface will be available at: http://localhost:{port}")
        print("🔍 Features:")
        print("  • Type-ahead search with intelligent suggestions")
        print("  • Frontend element discovery (pages, iconsets, trees, navigators)")
        print("  • Module-aware search and filtering")
        print("  • Real-time search with highlighting")
        print("  • Responsive design with modern UI")

        # Enable debug logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize demo search in background after server starts
        print("🎯 UniXcoder demo search will initialize after server starts")
        print("   Check /api/demo/status for hybrid search status")

        # Add startup event for hybrid search engine validation
        @web_ui.app.on_event("startup")
        async def startup_event():
            try:
                print("🔧 Validating hybrid search engine components...")
                # The search engine initialization happens in __init__, just log status
                if hasattr(web_ui.search_engine, "embedding_processor"):
                    print("✅ Hybrid search engine components ready!")
                else:
                    print("⚠️ Warning: Some search components may not be available")
            except Exception as e:
                print(f"⚠️ Search engine validation error: {e}")
                logger.error(f"Search engine validation error: {e}")

        # Start the server
        uvicorn.run(web_ui.app, host="localhost", port=port, reload=False)

    except RuntimeError as e:
        print(f"❌ Failed to start web UI: {e}")
        print(
            "💡 Please ensure ports 5700-5799 are available or close other applications using these ports."
        )
        exit(1)
