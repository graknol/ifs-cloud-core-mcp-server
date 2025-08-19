"""
FastAPI Web Service for the Hybrid Search Engine.

Provides REST API endpoints for searching IFS Cloud source code.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.hybrid_search import (
    HybridSearchEngine,
    SearchResponse,
    SearchResult,
    QueryType,
)


# Global search engine instance
search_engine: Optional[HybridSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global search_engine

    print("🚀 Initializing Hybrid Search Engine...")

    # Initialize search engine
    checkpoint_dir = Path("embedding_checkpoints")
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        raise RuntimeError(
            "Embedding checkpoints not found. Run embedding pipeline first!"
        )

    search_engine = HybridSearchEngine(checkpoint_dir)

    # Validate initialization
    stats = search_engine.get_stats()
    if not stats.get("initialized", False):
        raise RuntimeError("Failed to initialize search engine")

    print("✅ Search engine ready!")
    print(
        f"📊 BM25S docs: {stats['bm25s_documents']:,}, FAISS embeddings: {stats['faiss_embeddings']:,}"
    )

    yield

    # Cleanup
    print("🛑 Shutting down search engine...")


# FastAPI app with lifespan management
app = FastAPI(
    title="IFS Cloud Hybrid Search Engine",
    description="Advanced search engine combining lexical (BM25S) and semantic (FAISS) search with FlashRank fusion",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files for web UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# Request/Response models
class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of results to return"
    )
    enable_rerank: bool = Field(default=True, description="Enable FlashRank reranking")
    explain_results: bool = Field(
        default=False, description="Include result explanations"
    )


class SearchResponseAPI(BaseModel):
    """API search response model."""

    query: str
    query_type: str
    total_found: int
    search_time: float
    fusion_method: str
    bm25s_count: int
    faiss_count: int
    rerank_applied: bool
    suggestions: List[str]
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    engine_ready: bool
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    """Engine statistics response."""

    stats: Dict[str, Any]
    uptime: float


# Global state
app_start_time = time.time()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve search web interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IFS Cloud Hybrid Search</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .header h1 { color: #2c3e50; margin-bottom: 10px; }
            .header p { color: #7f8c8d; font-size: 18px; }
            .search-box { display: flex; gap: 10px; margin-bottom: 30px; }
            .search-input { flex: 1; padding: 15px; border: 2px solid #e1e8ed; border-radius: 8px; font-size: 16px; }
            .search-button { padding: 15px 30px; background: #3498db; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
            .search-button:hover { background: #2980b9; }
            .options { margin-bottom: 20px; display: flex; gap: 20px; align-items: center; }
            .option { display: flex; align-items: center; gap: 5px; }
            .results { margin-top: 30px; }
            .result { background: #f8f9fa; padding: 20px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #3498db; }
            .result-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
            .result-title { font-weight: bold; color: #2c3e50; font-size: 16px; }
            .result-score { background: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
            .result-details { color: #7f8c8d; margin-bottom: 10px; }
            .result-snippet { background: #ecf0f1; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 14px; }
            .loading { text-align: center; color: #3498db; margin: 40px 0; }
            .error { background: #e74c3c; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; }
            .stats { background: #2ecc71; color: white; padding: 10px; border-radius: 4px; margin-bottom: 20px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 IFS Cloud Hybrid Search</h1>
                <p>Advanced search combining lexical (BM25S) + semantic (FAISS) + FlashRank fusion</p>
            </div>
            
            <div class="search-box">
                <input type="text" id="searchInput" class="search-input" placeholder="Search IFS Cloud source code..." />
                <button onclick="search()" class="search-button">Search</button>
            </div>
            
            <div class="options">
                <div class="option">
                    <input type="number" id="topK" value="10" min="1" max="100" style="width: 60px; padding: 5px;">
                    <label for="topK">Results</label>
                </div>
                <div class="option">
                    <input type="checkbox" id="rerank" checked>
                    <label for="rerank">FlashRank</label>
                </div>
                <div class="option">
                    <input type="checkbox" id="explain">
                    <label for="explain">Explanations</label>
                </div>
            </div>
            
            <div id="results" class="results"></div>
        </div>
        
        <script>
            async function search() {
                const query = document.getElementById('searchInput').value.trim();
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="loading">🔍 Searching...</div>';
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: query,
                            top_k: parseInt(document.getElementById('topK').value),
                            enable_rerank: document.getElementById('rerank').checked,
                            explain_results: document.getElementById('explain').checked
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Search failed');
                    }
                    
                    displayResults(data);
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
                }
            }
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                
                if (!data.results || data.results.length === 0) {
                    resultsDiv.innerHTML = '<div class="error">No results found</div>';
                    return;
                }
                
                let html = `
                    <div class="stats">
                        📊 Found ${data.total_found} documents in ${data.search_time.toFixed(3)}s 
                        | Query: ${data.query_type} 
                        | Fusion: ${data.fusion_method}
                        ${data.rerank_applied ? ' ⚡ FlashRank' : ''}
                    </div>
                `;
                
                data.results.forEach((result, index) => {
                    html += `
                        <div class="result">
                            <div class="result-header">
                                <div class="result-title">${result.file_name}</div>
                                <div class="result-score">Score: ${result.score.toFixed(4)}</div>
                            </div>
                            <div class="result-details">
                                📁 ${result.file_path}<br>
                                🔗 API: ${result.api_name}<br>
                                🎯 Source: ${result.source} | Match: ${result.match_type}
                            </div>
                            ${result.snippet ? `<div class="result-snippet">${result.snippet}</div>` : ''}
                            ${result.explanation ? `<div style="margin-top: 10px; font-style: italic; color: #27ae60;">💡 ${result.explanation}</div>` : ''}
                        </div>
                    `;
                });
                
                if (data.suggestions && data.suggestions.length > 0) {
                    html += '<div style="margin-top: 20px;"><strong>💡 Suggestions:</strong><ul>';
                    data.suggestions.forEach(suggestion => {
                        html += `<li><a href="#" onclick="document.getElementById('searchInput').value='${suggestion}'; search();">${suggestion}</a></li>`;
                    });
                    html += '</ul></div>';
                }
                
                resultsDiv.innerHTML = html;
            }
            
            // Search on Enter key
            document.getElementById('searchInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    search();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global search_engine

    if search_engine is None:
        return HealthResponse(status="error", engine_ready=False, stats={})

    stats = search_engine.get_stats()

    return HealthResponse(
        status="healthy" if stats.get("initialized", False) else "error",
        engine_ready=stats.get("initialized", False),
        stats=stats,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get engine statistics."""
    global search_engine

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    stats = search_engine.get_stats()
    uptime = time.time() - app_start_time

    return StatsResponse(stats=stats, uptime=uptime)


@app.post("/api/search", response_model=SearchResponseAPI)
async def api_search(request: SearchRequest):
    """Search endpoint."""
    global search_engine

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        response = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            enable_rerank=request.enable_rerank,
            explain_results=request.explain_results,
        )

        # Convert to API response format
        return SearchResponseAPI(
            query=response.query,
            query_type=response.query_type.value,
            total_found=response.total_found,
            search_time=response.search_time,
            fusion_method=response.fusion_method,
            bm25s_count=response.bm25s_count,
            faiss_count=response.faiss_count,
            rerank_applied=response.rerank_applied,
            suggestions=response.suggestions,
            results=[
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
                    "explanation": r.explanation if request.explain_results else None,
                }
                for i, r in enumerate(response.results)
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/search", response_model=SearchResponseAPI)
async def api_search_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    rerank: bool = Query(True, description="Enable FlashRank"),
    explain: bool = Query(False, description="Include explanations"),
):
    """Search endpoint (GET method)."""
    request = SearchRequest(
        query=q, top_k=top_k, enable_rerank=rerank, explain_results=explain
    )
    return await api_search(request)


@app.post("/api/warmup")
async def warmup_engine(background_tasks: BackgroundTasks):
    """Warm up the search engine with sample queries."""
    global search_engine

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    async def perform_warmup():
        """Perform warmup searches in background."""
        warmup_queries = [
            "customer order",
            "invoice processing",
            "inventory management",
            "purchase order",
            "general ledger",
        ]

        for query in warmup_queries:
            try:
                search_engine.search(query, top_k=5, enable_rerank=False)
                await asyncio.sleep(0.1)  # Small delay between queries
            except Exception as e:
                print(f"Warmup query '{query}' failed: {e}")

    background_tasks.add_task(perform_warmup)

    return {"status": "warmup_started", "message": "Engine warming up in background"}


if __name__ == "__main__":
    uvicorn.run(
        "search_api:app", host="127.0.0.1", port=8001, reload=False, log_level="info"
    )
