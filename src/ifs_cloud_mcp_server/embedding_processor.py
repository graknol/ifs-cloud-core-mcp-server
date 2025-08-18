"""Production-Ready Embedding Framework for IFS Cloud.

A clean, sequential embedding framework that incorporates all learnings:
- Pre-processing with PageRank importance rankings
- Sequential processing for reliability
- Progress tracking and ETA estimates
- Checkpoint/cache system
- AI summarization on full files with metadata
- Phi-4 Mini with 64K context optimization
- Ollama CLI integration
- BM25S indexing for hybrid search
- ColBERT-style fusion preparation (see COLBERT_HYBRID_SEARCH_ARCHITECTURE.md)

The framework prepares data for ColBERT hybrid search combining:
- Dense retrieval (FAISS) for semantic understanding
- Sparse retrieval (BM25S) for lexical matching
- Late interaction fusion for optimal relevance
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import re
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

# Search engine imports
try:
    import faiss
    import numpy as np
    import bm25s
    import torch
    from transformers import AutoTokenizer, AutoModel

    SEARCH_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Search dependencies not available: {e}")
    SEARCH_DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class BuiltInPageRankAnalyzer:
    """Built-in PageRank analyzer for PL/SQL files."""

    def __init__(self, work_dir: Path, max_context_tokens: int = 65536):
        self.work_dir = Path(work_dir)
        self.max_context_tokens = max_context_tokens

    def extract_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract API calls from a PL/SQL file (fast and simple)."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Simple regex to find all Package_Name_API calls
            api_call_pattern = re.compile(r"([A-Z][A-Z0-9_]*_API)\.", re.IGNORECASE)
            api_calls = set()

            # Find all API calls in the content
            matches = api_call_pattern.findall(content)
            # API calls are already correctly formatted, just add them
            api_calls.update(match.upper() for match in matches)

            # Determine this file's API name from file path
            file_name = file_path.name
            if file_name.endswith(".plsql"):
                api_name = file_name[:-6].upper() + "_API"  # Remove .plsql extension
            else:
                api_name = file_path.stem.upper() + "_API"

            return {
                "file_path": str(file_path),
                "relative_path": str(file_path.relative_to(self.work_dir)),
                "file_name": file_name,
                "api_name": api_name,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "api_calls": list(api_calls),  # Who this file calls
            }

        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            return None

    def build_reference_graph(
        self, files_info: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """Build reference graph between files."""
        # Create API name to file mapping
        api_to_file = {}
        for info in files_info:
            if info:
                api_to_file[info["api_name"]] = info["file_path"]

        # Build reference graph
        reference_graph = {}
        for info in files_info:
            if not info:
                continue

            file_path = info["file_path"]
            reference_graph[file_path] = set()

            # Find references to other APIs
            for api_call in info["api_calls"]:
                if api_call in api_to_file and api_to_file[api_call] != file_path:
                    reference_graph[file_path].add(api_to_file[api_call])

        return reference_graph

    def analyze_files(self) -> Dict[str, Any]:
        """Perform complete PageRank analysis of PL/SQL files."""
        logger.info("🔍 Starting built-in PageRank analysis...")

        # Find all PL/SQL files
        plsql_files = list(self.work_dir.rglob("*.plsql"))
        logger.info(f"Found {len(plsql_files)} PL/SQL files")

        if not plsql_files:
            raise ValueError(f"No PL/SQL files found in {self.work_dir}")

        # Extract file information
        logger.info("📊 Extracting file metadata...")
        files_info = []
        for i, file_path in enumerate(plsql_files):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(plsql_files)} files")

            info = self.extract_file_info(file_path)
            if info:
                files_info.append(info)

        logger.info(f"Successfully processed {len(files_info)} files")

        # Build reference graph
        logger.info("🔗 Building reference graph...")
        reference_graph = self.build_reference_graph(files_info)

        # Simplified ranking - just sort alphabetically by file path for consistency
        file_rankings = []
        for info in files_info:
            if not info:
                continue

            file_rankings.append(info)

        # Sort by file path for consistent ordering
        file_rankings.sort(key=lambda x: x["file_path"])

        # Add rank numbers
        for i, ranking in enumerate(file_rankings):
            ranking["rank"] = i + 1

        # Prepare final analysis (simplified)
        analysis_metadata = {
            "work_directory": str(self.work_dir),
            "max_context_tokens": self.max_context_tokens,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_stats": {
                "total_files_found": len(plsql_files),
                "total_files_processed": len(files_info),
                "total_api_calls_found": sum(len(info["api_calls"]) for info in files_info),
            },
        }

        return {"analysis_metadata": analysis_metadata, "file_rankings": file_rankings}


@dataclass
class FileMetadata:
    """Metadata for a single file to be processed."""

    rank: int
    file_path: str
    relative_path: str
    file_name: str
    api_name: str
    file_size_mb: float
    procedure_count: int
    function_count: int
    api_calls: List[str] = field(default_factory=list)
    changelog_lines: List[str] = field(default_factory=list)
    procedure_function_names: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Result of processing a single file."""

    file_metadata: FileMetadata
    success: bool
    processing_time: float
    content_hash: str
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    bm25_text: Optional[str] = None  # Text used for BM25S indexing
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    has_ai_summary: bool = False  # True if summary is AI-generated, False if just content excerpt


@dataclass
class ProcessingProgress:
    """Current processing progress and statistics."""

    current_file: int
    total_files: int
    files_processed: int
    files_successful: int
    files_failed: int
    start_time: datetime
    estimated_completion: Optional[datetime]
    current_rate: float  # files per second
    total_processing_time: float
    checkpoint_saved: bool


class BGEM3EmbeddingGenerator:
    """Generates embeddings using BGE-M3 for semantic search of business content."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None

    def initialize_model(self) -> bool:
        """Initialize the BGE-M3 model for embedding generation."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            logger.warning("Embedding dependencies not available")
            return False

        try:
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load BGE-M3 tokenizer and model
            logger.info(f"Loading BGE-M3 model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info("✅ BGE-M3 model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 model: {e}")
            return False

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for the given text using BGE-M3."""
        if not self.model or not self.tokenizer:
            logger.warning("BGE-M3 model not initialized")
            return None

        try:
            # Tokenize input with BGE-M3's 8192 token limit
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,  # BGE-M3's max length
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token) for BGE-M3
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()

            # Convert to CPU numpy array then to list
            embedding_list = embedding.cpu().numpy().tolist()

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self.model:
            return 1024  # Default BGE-M3 dimension
        return self.model.config.hidden_size


class FAISSIndexManager:
    """Manages FAISS index creation and persistence for embeddings."""

    def __init__(self, index_dir: Path, embedding_dim: int = 1024):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.faiss_index_file = self.index_dir / "faiss_index.bin"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.metadata_file = self.index_dir / "faiss_metadata.json"

        # In-memory storage
        self.embeddings = []
        self.embedding_metadata = []
        self.faiss_index = None

    def add_embedding(self, embedding: List[float], metadata: dict) -> None:
        """Add an embedding and its metadata to the index."""
        self.embeddings.append(embedding)
        self.embedding_metadata.append(metadata)

    def build_faiss_index(self) -> bool:
        """Build FAISS index from collected embeddings."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            logger.warning("FAISS not available, skipping index build")
            return False

        if not self.embeddings:
            logger.warning("No embeddings available for FAISS indexing")
            return False

        try:
            logger.info(
                f"Building FAISS index with {len(self.embeddings)} embeddings..."
            )

            # Filter out None/invalid embeddings and check dimensions
            valid_embeddings = []
            for i, emb in enumerate(self.embeddings):
                if emb is not None and isinstance(emb, list) and len(emb) > 0:
                    valid_embeddings.append(emb)
                else:
                    logger.warning(f"Skipping invalid embedding at index {i}: {type(emb)}")

            if not valid_embeddings:
                logger.warning("No valid embeddings to index")
                return False

            # Check if all embeddings have the same dimension
            emb_dims = [len(emb) for emb in valid_embeddings]
            unique_dims = set(emb_dims)
            if len(unique_dims) > 1:
                logger.warning(f"Inconsistent embedding dimensions: {unique_dims}")
                # Use the most common dimension
                from collections import Counter
                most_common_dim = Counter(emb_dims).most_common(1)[0][0]
                logger.info(f"Using dimension {most_common_dim} for FAISS index")
                valid_embeddings = [emb for emb in valid_embeddings if len(emb) == most_common_dim]
                self.embedding_dim = most_common_dim

            # Convert embeddings to numpy array
            embeddings_np = np.array(valid_embeddings, dtype=np.float32)

            # Create FAISS index (using IndexFlatIP for cosine similarity)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_np)

            # Add embeddings to index
            self.faiss_index.add(embeddings_np)

            # Save to disk
            faiss.write_index(self.faiss_index, str(self.faiss_index_file))
            np.save(self.embeddings_file, embeddings_np)

            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "embedding_count": len(valid_embeddings),
                "embedding_dimension": self.embedding_dim,
                "index_type": "FAISS_IndexFlatIP",
                "model": "BAAI/bge-m3",
            }
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ FAISS index built successfully: {len(valid_embeddings)} embeddings"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False

    def load_existing_index(self) -> bool:
        """Load existing FAISS index if available."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            return False

        try:
            if (
                self.faiss_index_file.exists()
                and self.embeddings_file.exists()
                and self.metadata_file.exists()
            ):

                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
                embeddings_np = np.load(self.embeddings_file)
                self.embeddings = embeddings_np.tolist()

                logger.info(
                    f"Loaded existing FAISS index with {len(self.embeddings)} embeddings"
                )
                return True

        except Exception as e:
            logger.warning(f"Failed to load existing FAISS index: {e}")

        return False


class BM25SIndexer:
    """Handles BM25S indexing for lexical search capabilities."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # BM25S index files
        self.bm25_index_file = self.index_dir / "bm25s_index.pkl"
        self.bm25_corpus_file = self.index_dir / "bm25s_corpus.pkl"
        self.bm25_metadata_file = self.index_dir / "bm25s_metadata.json"

        # In-memory storage for batch processing
        self.corpus_texts = []
        self.corpus_metadata = []
        self.bm25_index = None

    def prepare_text_for_bm25(self, summary: str, metadata: FileMetadata) -> str:
        """Prepare text for BM25S indexing optimized for ColBERT hybrid fusion.

        Creates structured text that supports both lexical matching and
        late interaction with dense embeddings in ColBERT-style fusion.
        """
        # Structure text for optimal ColBERT fusion
        searchable_parts = [
            # Core identifiers (high weight in sparse retrieval)
            f"file:{metadata.file_name}",
            f"api:{metadata.api_name}",
            # Ranking context (for query-document alignment)
            f"rank:{metadata.rank}",
            # Code structure (for technical queries)
            f"procedures:{metadata.procedure_count}",
            f"functions:{metadata.function_count}",
            # Rich content (for semantic-lexical bridge)
            summary or "",
        ]

        # Join with spaces for proper tokenization in ColBERT late interaction
        return " ".join(filter(None, searchable_parts))

    def add_document(self, result: ProcessingResult) -> None:
        """Add a processed document to the BM25S corpus."""
        if not result.success or not result.summary:
            return

        bm25_text = self.prepare_text_for_bm25(result.summary, result.file_metadata)

        self.corpus_texts.append(bm25_text)
        self.corpus_metadata.append(
            {
                "file_path": result.file_metadata.file_path,
                "file_name": result.file_metadata.file_name,
                "api_name": result.file_metadata.api_name,
                "rank": result.file_metadata.rank,
                "content_hash": result.content_hash,
            }
        )

        # Update result with BM25 text for checkpointing
        result.bm25_text = bm25_text

    def build_index(self) -> bool:
        """Build the BM25S index from accumulated corpus."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            logger.warning("BM25S dependencies not available, skipping index build")
            return False

        if not self.corpus_texts:
            logger.warning("No corpus texts available for BM25S indexing")
            return False

        try:
            logger.info(
                f"Building BM25S index with {len(self.corpus_texts)} documents..."
            )

            # Tokenize corpus (BM25S will handle tokenization)
            corpus_tokens = bm25s.tokenize(self.corpus_texts, show_progress=True)

            # Build BM25S index
            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(corpus_tokens, show_progress=True)

            # Save index to disk
            with open(self.bm25_index_file, "wb") as f:
                pickle.dump(self.bm25_index, f)

            with open(self.bm25_corpus_file, "wb") as f:
                pickle.dump(self.corpus_texts, f)

            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "document_count": len(self.corpus_texts),
                "index_type": "BM25S",
                "version": "0.2.6+",
            }
            with open(self.bm25_metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ BM25S index built successfully: {len(self.corpus_texts)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build BM25S index: {e}")
            return False

    def load_existing_index(self) -> bool:
        """Load existing BM25S index if available."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            return False

        try:
            if (
                self.bm25_index_file.exists()
                and self.bm25_corpus_file.exists()
                and self.bm25_metadata_file.exists()
            ):

                with open(self.bm25_index_file, "rb") as f:
                    self.bm25_index = pickle.load(f)

                with open(self.bm25_corpus_file, "rb") as f:
                    self.corpus_texts = pickle.load(f)

                logger.info(
                    f"Loaded existing BM25S index with {len(self.corpus_texts)} documents"
                )
                return True

        except Exception as e:
            logger.warning(f"Failed to load existing BM25S index: {e}")

        return False


class EmbeddingCheckpointManager:
    """Manages checkpoints and caching for embedding processing."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = checkpoint_dir / "progress.json"
        self.results_file = checkpoint_dir / "results.jsonl"
        self.metadata_file = checkpoint_dir / "metadata.json"

    def archive_existing_files(self) -> None:
        """Archive existing checkpoint files before starting a new run to avoid duplicates."""
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        files_to_archive = [
            self.results_file,
            self.checkpoint_dir / "phase2a_results.jsonl",
            self.checkpoint_dir / "phase2b_results.jsonl",
            self.progress_file,
            self.metadata_file
        ]
        
        archived_any = False
        for file_path in files_to_archive:
            if file_path.exists():
                # Create archive filename with timestamp
                archive_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                archive_path = self.checkpoint_dir / "archives" / archive_name
                
                # Create archives directory if it doesn't exist
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the existing file to archive
                file_path.rename(archive_path)
                logger.info(f"📁 Archived {file_path.name} → {archive_name}")
                archived_any = True
        
        if archived_any:
            logger.info(f"✅ Existing checkpoint files archived with timestamp {timestamp}")
        else:
            logger.info("📝 No existing checkpoint files to archive - starting fresh")

    def save_progress(self, progress: ProcessingProgress) -> None:
        """Save current progress to checkpoint."""
        progress_data = asdict(progress)
        progress_data["start_time"] = progress.start_time.isoformat()
        if progress.estimated_completion:
            progress_data["estimated_completion"] = (
                progress.estimated_completion.isoformat()
            )

        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2)

    def load_progress(self) -> Optional[ProcessingProgress]:
        """Load progress from checkpoint if exists."""
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["start_time"] = datetime.fromisoformat(data["start_time"])
            if data.get("estimated_completion"):
                data["estimated_completion"] = datetime.fromisoformat(
                    data["estimated_completion"]
                )

            return ProcessingProgress(**data)
        except Exception as e:
            logger.warning(f"Failed to load progress checkpoint: {e}")
            return None

    def save_result(self, result: ProcessingResult) -> None:
        """Append processing result to results file."""
        result_data = asdict(result)
        result_data["timestamp"] = datetime.now().isoformat()

        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data) + "\n")

    def save_phase2a_result(self, result: ProcessingResult) -> None:
        """Append Phase 2A AI summary result to separate results file."""
        result_data = asdict(result)
        result_data["timestamp"] = datetime.now().isoformat()

        # Create phase2a_results.jsonl file
        phase2a_results_file = self.checkpoint_dir / "phase2a_results.jsonl"
        with open(phase2a_results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data) + "\n")

    def save_phase2b_result(self, result: ProcessingResult) -> None:
        """Append Phase 2B embedding result to separate results file."""
        result_data = asdict(result)
        result_data["timestamp"] = datetime.now().isoformat()

        # Create phase2b_results.jsonl file
        phase2b_results_file = self.checkpoint_dir / "phase2b_results.jsonl"
        with open(phase2b_results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data) + "\n")

    def load_processed_files(self) -> Set[str]:
        """Load set of already processed files."""
        processed = set()

        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            if result.get("success"):
                                processed.add(result["file_metadata"]["file_path"])
            except Exception as e:
                logger.warning(f"Failed to load processed files: {e}")

        return processed

    def load_all_results(self) -> List[ProcessingResult]:
        """Load all processing results from checkpoint files."""
        results = []

        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result_data = json.loads(line)
                            
                            # Handle both old and new checkpoint formats
                            if "file_metadata" in result_data:
                                # New format with FileMetadata object
                                file_metadata_data = result_data["file_metadata"]
                                file_metadata = FileMetadata(
                                    rank=file_metadata_data["rank"],
                                    file_path=file_metadata_data["file_path"],
                                    relative_path=file_metadata_data["relative_path"],
                                    file_name=file_metadata_data["file_name"],
                                    api_name=file_metadata_data["api_name"],
                                    file_size_mb=file_metadata_data["file_size_mb"],
                                    procedure_count=file_metadata_data.get("procedure_count", 0),
                                    function_count=file_metadata_data.get("function_count", 0),
                                    api_calls=file_metadata_data.get("api_calls", []),
                                    changelog_lines=file_metadata_data.get("changelog_lines", []),
                                    procedure_function_names=file_metadata_data.get("procedure_function_names", []),
                                )
                                
                                result = ProcessingResult(
                                    file_metadata=file_metadata,
                                    success=result_data["success"],
                                    processing_time=result_data.get("processing_time", 0.0),
                                    content_hash=result_data["content_hash"],
                                    summary=result_data.get("summary", ""),
                                    embedding=result_data.get("embedding"),
                                    has_ai_summary=result_data.get("has_ai_summary", False),  # Default to False for backward compatibility
                                )
                            else:
                                # Old format - skip for now as we're using new format
                                continue
                                
                            results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load all results: {e}")

        return results

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save processing metadata."""
        metadata["saved_at"] = datetime.now().isoformat()
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


class OllamaProcessor:
    """Handles AI processing using Ollama CLI for optimal performance."""

    def __init__(self, model: str = "phi4-mini:3.8b-q4_K_M", max_tokens: int = 65536):
        self.model = model
        self.max_tokens = max_tokens
        self.context_limit = 64000  # Leave some buffer for response
        self.restart_count = 0  # Track Ollama restart attempts
        self.max_restarts = 3  # Maximum number of restart attempts

    def check_ollama_availability(self) -> bool:
        """Check if Ollama is available and model is pulled."""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0 and self.model in result.stdout
        except Exception as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False

    def ensure_model(self) -> bool:
        """Ensure the model is available, pull if necessary."""
        if self.check_ollama_availability():
            return True

        logger.info(f"Pulling model {self.model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", self.model],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for model pull
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (chars / 4)."""
        return len(text) // 4

    def extract_valuable_content(
        self, content: str, metadata: FileMetadata
    ) -> Dict[str, List[str]]:
        """Extract valuable signals from PL/SQL content for balanced analysis."""
        lines = content.split("\n")

        signals = {
            "error_messages": [],
            "procedures": [],
            "functions": [],
            "comments": [],
            "api_calls": [],
            "database_operations": [],
        }

        current_block = []
        current_type = None
        in_block_comment = False

        for line in lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            # Handle block comments
            if "/*" in line:
                in_block_comment = True
            if "*/" in line:
                in_block_comment = False
                if line_stripped.startswith("--") or in_block_comment:
                    signals["comments"].append(line)
                continue

            # Extract error messages (highest priority signal)
            if (
                "Error_SYS.Record_General" in line
                or "RAISE_APPLICATION_ERROR" in line_upper
                or "Error_SYS.Appl_General" in line
                or "RAISE" in line_upper
                and ("EXCEPTION" in line_upper or "-20" in line)
            ):
                signals["error_messages"].append(line.strip())
                continue

            # Extract procedure/function signatures (key identification signal)
            if (
                line_upper.startswith("PROCEDURE ")
                or "PROCEDURE " in line_upper
                and ("AS" in line_upper or "BEGIN" in line_upper)
            ):
                if current_block and current_type:
                    signals[current_type].extend(current_block)
                current_block = [line]
                current_type = "procedures"
                continue

            elif (
                line_upper.startswith("FUNCTION ")
                or "FUNCTION " in line_upper
                and ("RETURN" in line_upper or "AS" in line_upper)
            ):
                if current_block and current_type:
                    signals[current_type].extend(current_block)
                current_block = [line]
                current_type = "functions"
                continue

            # Extract comments (valuable context signal)
            if line_stripped.startswith("--") or in_block_comment:
                signals["comments"].append(line)
                continue

            # Extract API calls (integration points signal)
            if "_API." in line and not line_upper.startswith("PACKAGE"):
                signals["api_calls"].append(line.strip())
                continue

            # Extract database operations (data access signal)
            if any(
                keyword in line_upper
                for keyword in [
                    "SELECT",
                    "INSERT",
                    "UPDATE",
                    "DELETE",
                    "FROM ",
                    "INTO ",
                ]
            ):
                signals["database_operations"].append(line.strip())
                continue

            # Add to current block if we're in a procedure/function
            if current_type and current_block:
                current_block.append(line)
                # Limit block size to prevent any single procedure from dominating
                if len(current_block) > 50:  # Max 50 lines per procedure/function
                    signals[current_type].extend(current_block[:50])
                    current_block = []
                    current_type = None

        # Add any remaining block
        if current_block and current_type:
            signals[current_type].extend(current_block)

        return signals

    def balance_signals(
        self,
        signals: Dict[str, List[str]],
        metadata: FileMetadata,
        available_tokens: int,
    ) -> str:
        """Create balanced content from extracted signals within token limits."""

        # Define signal priorities and allocation percentages
        signal_priorities = {
            "error_messages": 0.25,  # 25% - Critical for understanding failures
            "procedures": 0.30,  # 30% - Main functionality
            "functions": 0.25,  # 25% - Key operations
            "comments": 0.15,  # 15% - Context and documentation
            "api_calls": 0.03,  # 3% - Integration points
            "database_operations": 0.02,  # 2% - Data access patterns
        }

        # Adjust priorities based on file metadata
        if metadata.procedure_count > metadata.function_count * 2:
            signal_priorities["procedures"] = 0.35  # File is procedure-heavy
            signal_priorities["functions"] = 0.20
        elif metadata.function_count > metadata.procedure_count * 2:
            signal_priorities["functions"] = 0.35  # File is function-heavy
            signal_priorities["procedures"] = 0.20

        # Calculate token allocation for each signal
        balanced_content = []

        for signal_type, priority in signal_priorities.items():
            signal_data = signals.get(signal_type, [])
            if not signal_data:
                continue

            allocated_tokens = int(available_tokens * priority)
            allocated_chars = allocated_tokens * 4  # Rough char estimation

            # Join and truncate signal data
            signal_text = "\n".join(signal_data)
            if len(signal_text) > allocated_chars:
                signal_text = signal_text[:allocated_chars]
                # Try to end at line boundary
                last_newline = signal_text.rfind("\n")
                if (
                    last_newline > allocated_chars * 0.8
                ):  # If we're close to a line boundary
                    signal_text = signal_text[:last_newline]

            if signal_text.strip():
                balanced_content.append(
                    f"\n--- {signal_type.upper().replace('_', ' ')} ---"
                )
                balanced_content.append(signal_text)

        return "\n".join(balanced_content)

    def truncate_content(self, content: str, metadata: FileMetadata) -> Tuple[str, int]:
        """Intelligently extract and balance valuable signals within context limits."""
        estimated_tokens = self.estimate_tokens(content)

        if estimated_tokens <= self.context_limit:
            return content, estimated_tokens

        # Extract valuable signals
        signals = self.extract_valuable_content(content, metadata)

        # Calculate available tokens (reserve space for prompt and response)
        available_tokens = self.context_limit - 2000

        # Create balanced content
        balanced_content = self.balance_signals(signals, metadata, available_tokens)
        final_tokens = self.estimate_tokens(balanced_content)

        return balanced_content, final_tokens

    def create_summary_prompt(self, content: str, metadata: FileMetadata) -> str:
        """Create optimized prompt for file summarization using balanced signals."""
        
        # Format changelog lines if available
        changelog_section = ""
        if metadata.changelog_lines:
            changelog_text = "\n".join(metadata.changelog_lines)
            changelog_section = f"""
FILE CHANGELOG (SELECTED HEADER COMMENTS):
{changelog_text}
"""
        
        # Format procedure/function names if available
        procedures_section = ""
        if metadata.procedure_function_names:
            procedures_text = "\n".join(metadata.procedure_function_names)
            procedures_section = f"""
KEY PROCEDURES/FUNCTIONS (EVENLY SAMPLED):
{procedures_text}
"""

        return f"""Analyze this IFS Cloud PL/SQL file using the provided signals and create a comprehensive summary for semantic search purposes.

FILE METADATA & CONTEXT:
- File: {metadata.file_name}
- API: {metadata.api_name}  
- Rank: #{metadata.rank}
- Code Structure: {metadata.procedure_count} procedures, {metadata.function_count} functions
- API Calls: {len(metadata.api_calls)} external dependencies
{changelog_section}{procedures_section}
EXTRACTED SIGNALS (BALANCED FOR SEMANTIC ANALYSIS):
{content}

ANALYSIS INSTRUCTIONS:
Create a detailed summary optimized for semantic search and system understanding:

1. **PRIMARY BUSINESS PURPOSE**: What core business function does this API serve in IFS Cloud?

2. **SYSTEM IMPORTANCE**: Why is this file ranked #{metadata.rank}? What makes it architecturally important?

3. **KEY FUNCTIONALITY**: Based on the procedures/functions, what are the 3-5 most important operations this API provides?

4. **ERROR SCENARIOS**: From the error messages, what are the main failure cases and business rules this API enforces?

5. **INTEGRATION PATTERNS**: What other systems/APIs does this interact with? What role does it play in larger business processes?

6. **BUSINESS DOMAIN**: Which IFS Cloud functional area (Finance, Manufacturing, HR, etc.) does this primarily support?

7. **DATA OPERATIONS**: What main business entities/tables does this API manage or access?

8. **TECHNICAL ARCHITECTURE**: Any notable patterns, complexity, or architectural considerations?

9. **CHANGE HISTORY**: If changelog information is available, what major changes or evolution patterns are evident?

Focus on business value, system relationships, and searchable concepts rather than implementation details. This summary will be used for semantic similarity matching to help users find relevant code components.
"""

    def is_ollama_crashed_response(self, output: str) -> bool:
        """Detect if Ollama output indicates a crashed/corrupted state."""
        if not output:
            return False
            
        # Check for the specific "@" crash pattern
        stripped = output.strip()
        if len(stripped) > 20 and all(c == '@' for c in stripped):
            logger.warning(f"Detected Ollama crash pattern: {len(stripped)} '@' characters")
            return True
            
        # Check for other crash patterns
        crash_indicators = [
            "model not found",
            "connection refused",
            "server error",
            "internal error"
        ]
        
        lower_output = output.lower()
        for indicator in crash_indicators:
            if indicator in lower_output:
                return True
                
        return False

    def restart_ollama(self) -> bool:
        """Restart Ollama server to recover from crashes."""
        # Check if we've exceeded restart limit
        if self.restart_count >= self.max_restarts:
            logger.error(f"❌ Ollama restart limit exceeded ({self.max_restarts} attempts). "
                        f"The Ollama service appears to be in a persistently corrupted state.")
            return False
            
        self.restart_count += 1
        
        try:
            logger.warning(f"🔄 Restarting Ollama server (attempt {self.restart_count}/{self.max_restarts}) due to crash detection...")
            
            # Kill any existing Ollama processes
            logger.info("Terminating existing Ollama processes...")
            subprocess.run(
                ["taskkill", "/f", "/im", "ollama.exe"],
                capture_output=True,
                timeout=30
            )
            
            # Wait a moment for processes to terminate
            time.sleep(2)
            
            # Start Ollama server and ensure model is loaded
            logger.info(f"Restarting Ollama and loading model {self.model}...")
            subprocess.run(
                ["ollama", "run", self.model, "Ready to process files."],
                capture_output=True,
                timeout=120  # Allow time for model loading
            )
            
            logger.info("✅ Ollama server restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart Ollama: {e}")
            return False

    def reset_restart_counter(self):
        """Reset the restart counter after successful processing."""
        if self.restart_count > 0:
            logger.debug(f"Resetting Ollama restart counter (was {self.restart_count})")
            self.restart_count = 0

    def shutdown_ollama(self) -> bool:
        """Shutdown Ollama server to free VRAM for other models."""
        try:
            logger.info("🔄 Shutting down Ollama server to free VRAM...")
            
            # Kill all Ollama processes to free VRAM
            result = subprocess.run(
                ["taskkill", "/f", "/im", "ollama.exe"],
                capture_output=True,
                timeout=30
            )
            
            # Also try to stop any ollama_llama_server processes
            subprocess.run(
                ["taskkill", "/f", "/im", "ollama_llama_server.exe"],
                capture_output=True,
                timeout=30
            )
            
            # Wait a moment for processes to fully terminate and VRAM to be freed
            time.sleep(3)
            
            logger.info("✅ Ollama server shutdown complete - VRAM freed for next phase")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error during Ollama shutdown (non-critical): {e}")
            # Don't fail the entire process if shutdown has issues
            return False

    def process_file(self, content: str, metadata: FileMetadata) -> ProcessingResult:
        """Process a single file with AI summarization."""
        start_time = time.time()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        try:
            # Truncate content if necessary
            processed_content, tokens_used = self.truncate_content(content, metadata)

            # Create prompt
            prompt = self.create_summary_prompt(processed_content, metadata)

            # Call Ollama CLI
            logger.debug(f"Processing {metadata.file_name} with {tokens_used} tokens")

            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=300,  # 5 minutes per file
            )

            processing_time = time.time() - start_time

            # Check for crash/corruption patterns
            if result.returncode == 0 and result.stdout:
                # Check if output indicates Ollama crashed
                if self.is_ollama_crashed_response(result.stdout):
                    logger.warning(f"Ollama crash detected for {metadata.file_name}")
                    
                    # Attempt to restart Ollama
                    if self.restart_ollama():
                        # Retry the file processing
                        logger.info(f"Retrying processing of {metadata.file_name} after Ollama restart")
                        retry_start = time.time()
                        
                        retry_result = subprocess.run(
                            ["ollama", "run", self.model],
                            input=prompt,
                            text=True,
                            capture_output=True,
                            timeout=300,
                        )
                        
                        processing_time = time.time() - retry_start
                        
                        if retry_result.returncode == 0 and retry_result.stdout:
                            if not self.is_ollama_crashed_response(retry_result.stdout):
                                result = retry_result  # Use successful retry result
                                # Reset restart counter on successful retry
                                self.reset_restart_counter()
                                logger.info(f"✅ Successfully processed {metadata.file_name} after Ollama restart")
                            else:
                                # Check if this is due to restart limit being exceeded
                                if self.restart_count >= self.max_restarts:
                                    error_summary = (f"Failed: Ollama service is in a persistently corrupted state. "
                                                   f"Exceeded maximum restart attempts ({self.max_restarts}). "
                                                   f"Manual intervention required - try restarting your computer or "
                                                   f"reinstalling Ollama.")
                                else:
                                    error_summary = "Failed: Ollama crashed multiple times"
                                    
                                logger.error(f"Ollama still crashed after restart for {metadata.file_name}")
                                return ProcessingResult(
                                    summary=error_summary,
                                    metadata=asdict(metadata),
                                    hash=content_hash,
                                    processing_time=processing_time,
                                    tokens_used=tokens_used,
                                    success=False
                                )
                        else:
                            logger.error(f"Retry failed for {metadata.file_name}: {retry_result.stderr}")
                            return ProcessingResult(
                                summary="Failed: Retry unsuccessful after Ollama restart",
                                metadata=asdict(metadata),
                                hash=content_hash,
                                processing_time=processing_time,
                                tokens_used=tokens_used,
                                success=False
                            )
                    else:
                        # Provide descriptive error based on why restart failed
                        if self.restart_count >= self.max_restarts:
                            error_summary = (f"Failed: Ollama service is in a persistently corrupted state. "
                                           f"Exceeded maximum restart attempts ({self.max_restarts}). "
                                           f"Manual intervention required - try restarting your computer or "
                                           f"reinstalling Ollama.")
                            logger.error(f"Ollama restart limit exceeded for {metadata.file_name}")
                        else:
                            error_summary = "Failed: Could not restart Ollama service"
                            logger.error(f"Failed to restart Ollama for {metadata.file_name}")
                            
                        return ProcessingResult(
                            summary=error_summary,
                            metadata=asdict(metadata),
                            hash=content_hash,
                            processing_time=processing_time,
                            tokens_used=tokens_used,
                            success=False
                        )

            if result.returncode == 0 and result.stdout.strip():
                # Reset restart counter on successful processing
                self.reset_restart_counter()
                
                return ProcessingResult(
                    file_metadata=metadata,
                    success=True,
                    processing_time=processing_time,
                    content_hash=content_hash,
                    summary=result.stdout.strip(),
                    tokens_used=tokens_used,
                )
            else:
                error_msg = result.stderr or "No output from Ollama"
                return ProcessingResult(
                    file_metadata=metadata,
                    success=False,
                    processing_time=processing_time,
                    content_hash=content_hash,
                    error_message=error_msg,
                    tokens_used=tokens_used,
                )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                file_metadata=metadata,
                success=False,
                processing_time=processing_time,
                content_hash=content_hash,
                error_message=str(e),
            )


class ProductionEmbeddingFramework:
    """Production-ready embedding framework with all learnings incorporated."""

    def __init__(
        self,
        work_dir: Path,
        analysis_file: Path,
        checkpoint_dir: Path,
        model: str = "phi4-mini:3.8b-q4_K_M",
        max_files: Optional[int] = None,
    ):
        self.work_dir = Path(work_dir)
        self.analysis_file = Path(analysis_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_files = max_files

        # Initialize components
        self.checkpoint_manager = EmbeddingCheckpointManager(self.checkpoint_dir)
        self.ollama_processor = OllamaProcessor(model)
        self.bm25_indexer = BM25SIndexer(self.checkpoint_dir / "search_indexes")
        self.embedding_generator = BGEM3EmbeddingGenerator()
        self.faiss_manager = FAISSIndexManager(
            self.checkpoint_dir / "search_indexes",
            embedding_dim=1024,  # BGE-M3 default dimension
        )

        # Initialize built-in analyzer
        self.analyzer = BuiltInPageRankAnalyzer(self.work_dir)

        # Load or generate analysis data
        self.file_rankings = self._load_or_generate_analysis_data()

    def _fix_api_naming(self, api_name: str) -> str:
        """Fix API naming convention: add underscores before capitals except first and 'PI' in '_API'."""
        if not api_name or api_name == "":
            return api_name
        
        # Handle special case for _API suffix
        if api_name.upper().endswith("_API"):
            base_name = api_name[:-4]  # Remove _API
            suffix = "_API"
        elif api_name.upper().endswith("API"):
            base_name = api_name[:-3]  # Remove API
            suffix = "_API"
        else:
            base_name = api_name
            suffix = ""
        
        # Add underscores before capitals (except first character)
        fixed_name = ""
        for i, char in enumerate(base_name):
            if i > 0 and char.isupper():
                fixed_name += "_" + char
            else:
                fixed_name += char
        
        return fixed_name.upper() + suffix

    def _extract_changelog_lines(self, content: str) -> List[str]:
        """Extract 10 changelog messages from header: first, last, and 8 evenly spaced between."""
        try:
            # Find actual changelog entries and extract just the messages
            lines = content.split('\n')
            changelog_messages = []
            seen_messages = set()  # Track unique messages to avoid duplicates
            
            # Look for changelog entries that follow the pattern:
            # --  DDMMYY   sign    description
            # --  DD/MM/YYYY  sign    description  
            for i, line in enumerate(lines[:100]):  # Check first 100 lines for changelog
                stripped = line.strip()
                
                # Look for lines that contain actual changelog entries (dates + signatures + descriptions)
                if stripped.startswith('--') and len(stripped) > 10:
                    # Skip decorative lines (only dashes, headers, etc.)
                    if all(c in '- ' for c in stripped[2:].strip()):
                        continue
                    if 'Date' in stripped and 'Sign' in stripped and 'History' in stripped:
                        continue
                    if stripped.endswith('-'):
                        continue
                    if 'Logical unit:' in stripped or 'Component:' in stripped:
                        continue
                    if 'Template Version' in stripped:
                        continue
                    if 'layer Core' in stripped:
                        continue
                        
                    # Look for patterns that indicate actual changelog entries
                    # Pattern 1: --  DDMMYY   name   description
                    # Pattern 2: --  DD/MM/YYYY  name  description
                    stripped_content = stripped[2:].strip()  # Remove '--' prefix
                    
                    # Check if it starts with a date-like pattern
                    words = stripped_content.split()
                    if len(words) >= 3:
                        first_word = words[0]
                        # Date patterns: DDMMYY (6 digits) or DD/MM/YYYY
                        if (first_word.isdigit() and len(first_word) == 6) or \
                           ('/' in first_word and len(first_word.split('/')) == 3):
                            # Extract just the message part (skip date and developer name)
                            # Typical format: date developer_name message...
                            # Skip first two words (date and name) and take the rest
                            if len(words) > 2:
                                message = ' '.join(words[2:])  # Everything after date and name
                                if message and len(message.strip()) > 3:  # Only meaningful messages
                                    clean_message = message.strip()
                                    # Only add if we haven't seen this exact message before
                                    if clean_message.lower() not in seen_messages:
                                        changelog_messages.append(clean_message)
                                        seen_messages.add(clean_message.lower())
                    
            # If no changelog entries found, look for any meaningful comment lines with messages
            if not changelog_messages:
                for i, line in enumerate(lines[:50]):
                    stripped = line.strip()
                    if stripped.startswith('--') and len(stripped) > 15:
                        # Skip pure decoration
                        if all(c in '- ' for c in stripped[2:].strip()):
                            continue
                        if stripped.endswith('-' * 5):  # Lines ending with many dashes
                            continue
                        # Look for lines that seem to contain actual content descriptions
                        content_part = stripped[2:].strip()
                        if any(keyword in content_part.lower() for keyword in 
                               ['added', 'created', 'updated', 'fixed', 'changed', 'removed', 'modified']):
                            # Only add unique messages
                            if content_part.lower() not in seen_messages:
                                changelog_messages.append(content_part)
                                seen_messages.add(content_part.lower())
                        
            # Select 10 messages: first, last, and 8 evenly spaced
            if len(changelog_messages) <= 10:
                return changelog_messages
            elif len(changelog_messages) < 3:
                return changelog_messages
            else:
                selected = [changelog_messages[0]]  # First
                
                # Calculate 8 evenly spaced indices between first and last
                if len(changelog_messages) > 2:
                    middle_count = min(8, len(changelog_messages) - 2)
                    if middle_count > 0:
                        step = (len(changelog_messages) - 2) / (middle_count + 1)
                        for i in range(1, middle_count + 1):
                            idx = int(1 + step * i)
                            if idx < len(changelog_messages) - 1:
                                selected.append(changelog_messages[idx])
                
                selected.append(changelog_messages[-1])  # Last
                return selected[:10]
                
        except Exception as e:
            logger.warning(f"Could not extract changelog lines: {e}")
            return []

    def _extract_procedure_function_names(self, content: str) -> List[str]:
        """Extract 10 procedure/function names: balanced 50/50 public/private when available, excluding common IFS framework methods."""
        try:
            # Standard IFS framework method prefixes to exclude (these are generic implementation patterns)
            excluded_prefixes = [
                'GET_', 'SET_', 'CHECK_INSERT_', 'CHECK_UPDATE_', 'CHECK_DELETE_', 'CHECK_COMMON_',
                'DO_INSERT_', 'DO_UPDATE_', 'DO_DELETE_', 'DO_MODIFY_', 'DO_REMOVE_',
                'IS_', 'HAS_', 'EXIST_', 'EXISTS_', 'VALIDATE_', 'VERIFY_',
                'UNPACK_', 'PACK_', 'PREPARE_', 'FINISH_', 'COMPLETE_',
                'NEW__', 'MODIFY__', 'REMOVE__', 'DELETE__', 'INSERT__', 'UPDATE__',
                'PRE_', 'POST_', 'BEFORE_', 'AFTER_',
                'GET_OBJSTATE', 'GET_OBJVERSION', 'GET_OBJID', 'GET_STATE',
                'FINITE_STATE_', 'SET_STATE_', 'GET_DB_VALUES_', 'GET_CLIENT_VALUES_',
                'DECODE_', 'ENCODE_', 'GET_KEY_BY_', 'GET_KEYS_BY_'
            ]
            
            # Additional exact method names to exclude (common framework methods)
            excluded_exact = [
                'GET_OBJKEY', 'GET_VERSION_BY_KEYS', 'GET_VERSION_BY_ID', 'EXIST',
                'GET_FULL_NAME', 'GET_DESCRIPTION', 'GET_INFO', 'GET_BY_KEYS',
                'LOCK__', 'NEW__', 'MODIFY__', 'REMOVE__', 'DELETE__'
            ]
            
            # Separate collections for public and private methods
            public_names = []  # Methods without underscores at the end
            private_names = []  # Methods ending with ___
            seen_names = set()  # Track unique names to avoid duplicates
            
            lines = content.split('\n')
            
            # Look for procedure and function declarations
            for line in lines:
                stripped = line.strip().upper()
                
                # Match PROCEDURE declarations
                proc_match = re.match(r'^\s*PROCEDURE\s+(\w+)', stripped)
                if proc_match:
                    proc_name = proc_match.group(1)
                    if proc_name.lower() not in seen_names:
                        # Check if this should be excluded
                        should_exclude = False
                        
                        # Check prefixes (excluding the __ or ___ suffix for comparison)
                        base_name = proc_name.rstrip('_')
                        for prefix in excluded_prefixes:
                            if base_name.startswith(prefix):
                                should_exclude = True
                                break
                        
                        # Check exact matches (excluding suffix)
                        if base_name in excluded_exact:
                            should_exclude = True
                        
                        if not should_exclude:
                            method_entry = f"PROCEDURE {proc_name}"
                            if proc_name.endswith('___'):
                                private_names.append(method_entry)
                            else:
                                public_names.append(method_entry)
                            seen_names.add(proc_name.lower())
                
                # Match FUNCTION declarations  
                func_match = re.match(r'^\s*FUNCTION\s+(\w+)', stripped)
                if func_match:
                    func_name = func_match.group(1)
                    if func_name.lower() not in seen_names:
                        # Check if this should be excluded
                        should_exclude = False
                        
                        # Check prefixes (excluding the __ or ___ suffix for comparison)
                        base_name = func_name.rstrip('_')
                        for prefix in excluded_prefixes:
                            if base_name.startswith(prefix):
                                should_exclude = True
                                break
                        
                        # Check exact matches (excluding suffix)
                        if base_name in excluded_exact:
                            should_exclude = True
                        
                        if not should_exclude:
                            method_entry = f"FUNCTION {func_name}"
                            if func_name.endswith('___'):
                                private_names.append(method_entry)
                            else:
                                public_names.append(method_entry)
                            seen_names.add(func_name.lower())
            
            # Balance between public and private methods (50/50 when both available)
            total_needed = 10
            
            if len(public_names) == 0:
                # Only private methods available
                selected_names = private_names[:total_needed]
            elif len(private_names) == 0:
                # Only public methods available
                selected_names = public_names[:total_needed]
            else:
                # Both types available - aim for 50/50 balance
                public_needed = min(total_needed // 2, len(public_names))
                private_needed = min(total_needed - public_needed, len(private_names))
                
                # If one category doesn't have enough, take more from the other
                if private_needed < (total_needed - public_needed):
                    public_needed = min(total_needed - private_needed, len(public_names))
                elif public_needed < (total_needed - private_needed):
                    private_needed = min(total_needed - public_needed, len(private_names))
                
                # Select evenly spaced methods from each category
                selected_public = self._select_evenly_spaced(public_names, public_needed)
                selected_private = self._select_evenly_spaced(private_names, private_needed)
                
                # Combine them, maintaining some alternation for better representation
                selected_names = []
                pub_idx = priv_idx = 0
                for i in range(total_needed):
                    if pub_idx < len(selected_public) and priv_idx < len(selected_private):
                        # Alternate between public and private
                        if i % 2 == 0:
                            selected_names.append(selected_public[pub_idx])
                            pub_idx += 1
                        else:
                            selected_names.append(selected_private[priv_idx])
                            priv_idx += 1
                    elif pub_idx < len(selected_public):
                        selected_names.append(selected_public[pub_idx])
                        pub_idx += 1
                    elif priv_idx < len(selected_private):
                        selected_names.append(selected_private[priv_idx])
                        priv_idx += 1
            
            return selected_names[:total_needed]
                
        except Exception as e:
            logger.warning(f"Could not extract procedure/function names: {e}")
            return []

    def _select_evenly_spaced(self, items: List[str], count: int) -> List[str]:
        """Select evenly spaced items from a list."""
        if not items or count <= 0:
            return []
        if count >= len(items):
            return items
        if len(items) < 3:
            return items[:count]
            
        selected = [items[0]]  # First
        
        # Calculate evenly spaced indices between first and last
        if len(items) > 2 and count > 2:
            middle_count = min(count - 2, len(items) - 2)
            if middle_count > 0:
                step = (len(items) - 2) / (middle_count + 1)
                for i in range(1, middle_count + 1):
                    idx = int(1 + step * i)
                    if idx < len(items) - 1:
                        selected.append(items[idx])
        
        if count > 1 and len(items) > 1:
            selected.append(items[-1])  # Last
            
        return selected[:count]

        logger.info(
            f"Initialized embedding framework with {len(self.file_rankings)} files"
        )

        # Try to load existing indexes
        self.bm25_indexer.load_existing_index()
        self.faiss_manager.load_existing_index()

    def _load_analysis_data(self) -> List[FileMetadata]:
        """Load and parse the comprehensive analysis data."""
        try:
            with open(self.analysis_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            rankings = []
            for item in data.get("file_rankings", []):
                metadata = FileMetadata(**item)
                rankings.append(metadata)

                # Apply max_files limit if specified
                if self.max_files and len(rankings) >= self.max_files:
                    break

            logger.info(f"Loaded {len(rankings)} file rankings from analysis")
            return rankings

        except Exception as e:
            logger.error(f"Failed to load analysis data: {e}")
            raise

    def _load_or_generate_analysis_data(self) -> List[FileMetadata]:
        """Load existing analysis or generate simplified analysis if missing."""
        # Try to load existing analysis first
        if self.analysis_file.exists():
            logger.info(f"📊 Loading existing analysis from {self.analysis_file}")
            return self._load_analysis_data()

        # For testing/development: Create a simplified file list without full PageRank analysis
        if self.max_files and self.max_files <= 100:
            logger.info(f"📊 Creating simplified file list for {self.max_files} files (skipping PageRank)")
            return self._create_simplified_file_list()

        # Full PageRank analysis only for production runs
        logger.info(f"📊 Analysis file not found, generating new PageRank analysis...")
        logger.info(f"   This may take a few minutes for {self.work_dir}...")

        # Run built-in analyzer
        analysis_data = self.analyzer.analyze_files()

        # Save analysis to file
        with open(self.analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Analysis saved to {self.analysis_file}")

        # Convert to FileMetadata format
        rankings = []
        for item in analysis_data.get("file_rankings", []):
            metadata = FileMetadata(**item)
            rankings.append(metadata)

            # Apply max_files limit if specified
            if self.max_files and len(rankings) >= self.max_files:
                break

        logger.info(f"Generated analysis for {len(rankings)} files")
        return rankings

    def _create_simplified_file_list(self) -> List[FileMetadata]:
        """Create a simplified file list without PageRank analysis (for testing)."""
        plsql_files = list(self.work_dir.rglob("*.plsql"))
        rankings = []

        for i, file_path in enumerate(plsql_files[:self.max_files]):
            # Read content to extract actual metadata
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract metadata using patterns similar to OllamaProcessor
                procedure_count = len(re.findall(r'\bPROCEDURE\s+\w+', content, re.IGNORECASE))
                function_count = len(re.findall(r'\bFUNCTION\s+\w+', content, re.IGNORECASE))
                error_message_count = len(re.findall(r'\bError_SYS\.\w+|RAISE_APPLICATION_ERROR|EXCEPTION\b', content, re.IGNORECASE))
                comment_count = len(re.findall(r'--.*$|/\*.*?\*/', content, re.MULTILINE | re.DOTALL))
                
                # Extract API calls - look for patterns like Package_API.method
                api_call_pattern = r'\b([A-Z][A-Za-z_]*_API)\.([A-Za-z_]+)\s*\('
                api_matches = re.findall(api_call_pattern, content)
                # API calls are already correctly formatted, just combine them
                api_calls = [f"{pkg}.{method}" for pkg, method in api_matches[:10]]  # Limit to top 10
                
                # Extract changelog lines from header
                changelog_lines = self._extract_changelog_lines(content)
                
                # Extract procedure/function names
                procedure_function_names = self._extract_procedure_function_names(content)
                
            except Exception as e:
                logger.warning(f"Could not extract metadata from {file_path}: {e}")
                procedure_count = function_count = error_message_count = comment_count = 0
                api_calls = []
                changelog_lines = []
                procedure_function_names = []

            # Create metadata with extracted counts and changelog
            file_name = file_path.name
            if file_name.endswith(".plsql"):
                base_api_name = file_name[:-6] + "_API"  # Remove .plsql extension
            else:
                base_api_name = file_path.stem + "_API"
            
            api_name = self._fix_api_naming(base_api_name)
            
            metadata = FileMetadata(
                file_path=str(file_path),
                relative_path=str(file_path.relative_to(self.work_dir)),
                file_name=file_name,
                api_name=api_name,
                file_size_mb=file_path.stat().st_size / (1024 * 1024),
                procedure_count=procedure_count,
                function_count=function_count,
                api_calls=api_calls,
                changelog_lines=changelog_lines,
                procedure_function_names=procedure_function_names,
                rank=i + 1
            )
            rankings.append(metadata)

        logger.info(f"Created simplified file list with {len(rankings)} files")
        return rankings

    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content safely."""
        try:
            # Handle both absolute and relative paths
            if os.path.isabs(file_path):
                full_path = Path(file_path)
            else:
                full_path = self.work_dir / file_path.replace("_work\\", "").replace(
                    "_work/", ""
                )
            
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None

    def _calculate_progress_stats(
        self,
        current_idx: int,
        start_time: datetime,
        processed_count: int,
        successful_count: int,
        failed_count: int,
    ) -> ProcessingProgress:
        """Calculate current progress and ETA."""
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()

        # Calculate processing rate
        rate = processed_count / elapsed if elapsed > 0 else 0.0

        # Estimate completion time
        remaining_files = len(self.file_rankings) - processed_count
        estimated_completion = None
        if rate > 0:
            remaining_seconds = remaining_files / rate
            estimated_completion = now + timedelta(seconds=remaining_seconds)

        return ProcessingProgress(
            current_file=current_idx + 1,
            total_files=len(self.file_rankings),
            files_processed=processed_count,
            files_successful=successful_count,
            files_failed=failed_count,
            start_time=start_time,
            estimated_completion=estimated_completion,
            current_rate=rate,
            total_processing_time=elapsed,
            checkpoint_saved=False,
        )

    def _print_progress(self, progress: ProcessingProgress) -> None:
        """Print formatted progress information."""
        percent = (progress.files_processed / progress.total_files) * 100

        print(f"\n{'='*60}")
        print(f"🚀 EMBEDDING PROGRESS: {percent:.1f}% Complete")
        print(f"{'='*60}")
        print(f"📊 Files: {progress.files_processed:,}/{progress.total_files:,}")
        print(f"✅ Successful: {progress.files_successful:,}")
        print(f"❌ Failed: {progress.files_failed:,}")
        print(f"⚡ Rate: {progress.current_rate:.2f} files/sec")

        if progress.estimated_completion:
            eta_str = progress.estimated_completion.strftime("%H:%M:%S")
            remaining = progress.estimated_completion - datetime.now()
            remaining_str = str(remaining).split(".")[0]  # Remove microseconds
            print(f"⏰ ETA: {eta_str} (in {remaining_str})")

        elapsed = timedelta(seconds=progress.total_processing_time)
        elapsed_str = str(elapsed).split(".")[0]  # Remove microseconds
        print(f"⏱️  Elapsed: {elapsed_str}")

        if progress.current_file <= len(self.file_rankings):
            current_file = self.file_rankings[progress.current_file - 1]
            print(f"📄 Current: {current_file.file_name}")
            print(f"🎯 Importance: {current_file.combined_importance_score:.1f}")

        print(f"{'='*60}")

    async def run_embedding_pipeline(self, resume: bool = True) -> Dict[str, Any]:
        """Run the complete embedding pipeline with separated AI summarization."""
        logger.info("Starting production embedding pipeline (2-phase approach)")
        
        # Archive existing checkpoint files if not resuming to avoid duplicates
        if not resume:
            self.checkpoint_manager.archive_existing_files()

        # Phase 1: Basic processing and embeddings
        phase1_stats = await self._run_phase1_basic_processing(resume)

        # Phase 2: AI summarization pass
        phase2_stats = await self._run_phase2_ai_and_embeddings(resume)

        # Combine stats with compatibility keys
        final_stats = {
            **phase1_stats, 
            **phase2_stats, 
            "pipeline_completed": True,
            # Compatibility keys for main function
            "files_successful": phase1_stats.get("phase1_files_successful", 0) + phase2_stats.get("phase2_files_successful", 0),
            "files_failed": phase1_stats.get("phase1_files_failed", 0) + phase2_stats.get("phase2_files_failed", 0),
            "success_rate": (
                (phase1_stats.get("phase1_files_successful", 0) + phase2_stats.get("phase2_files_successful", 0)) / 
                max(1, phase1_stats.get("phase1_files_processed", 0) + phase2_stats.get("phase2_files_processed", 0))
            ),
            "average_rate": (
                phase1_stats.get("phase1_average_rate", 0) + phase2_stats.get("phase2_average_rate", 0)
            ) / 2,
            "total_processing_time": phase1_stats.get("phase1_processing_time", 0) + phase2_stats.get("phase2_processing_time", 0),
        }

        return final_stats

    async def _run_phase1_basic_processing(self, resume: bool = True) -> Dict[str, Any]:
        """Phase 1: Process files for basic embeddings and BM25S indexing without AI summaries."""
        logger.info("🚀 Phase 1: Basic processing and embeddings")
        
        # Print comprehensive explanation of Phase 1
        logger.info("")
        logger.info("=" * 80)
        logger.info("📋 PHASE 1 EXPLANATION - WHAT WE'RE DOING AND WHY")
        logger.info("=" * 80)
        logger.info("")
        logger.info("🎯 GOAL: Process all files for content indexing WITHOUT any GPU usage")
        logger.info("   Pure CPU/disk operations to prepare files for Phase 2 AI processing")
        logger.info("")
        logger.info("🔄 STEP-BY-STEP PROCESS:")
        logger.info("   1️⃣  For each of the 9,750 PL/SQL files:")
        logger.info("       • Read raw file content (disk I/O only)")
        logger.info("       • Create structured content excerpt (CPU processing)")
        logger.info("       • Extract metadata (procedures, functions, comments)")
        logger.info("       • Add to BM25S corpus (lexical/keyword search preparation)")
        logger.info("       • Save checkpoint with basic result (disk I/O only)")
        logger.info("   2️⃣  Build BM25S search index from all content excerpts (CPU only)")
        logger.info("   3️⃣  Prepare file list for Phase 2 AI processing")
        logger.info("")
        logger.info("🖥️  HARDWARE USAGE:")
        logger.info("   • CPU: Content processing, regex parsing, BM25S indexing")
        logger.info("   • Disk: File reading, checkpoint saving, index writing")
        logger.info("   • GPU: NONE - completely idle, no VRAM usage")
        logger.info("   • Memory: Minimal - only text processing buffers")
        logger.info("")
        logger.info("📊 OUTPUTS FROM PHASE 1:")
        logger.info("   • 9,750 content excerpts (keyword searchable)")
        logger.info("   • 9,750 metadata records (file info, rankings)")
        logger.info("   • BM25S index file (lexical search ready)")
        logger.info("   • Checkpoint files (Phase 2 ready)")
        logger.info("")
        logger.info("⚡ WHY NO GPU IN PHASE 1:")
        logger.info("   • Raw content is too large/noisy for good embeddings")
        logger.info("   • AI summaries (Phase 2) are much better for embeddings")
        logger.info("   • Keeps GPU completely free for intensive Phase 2 work")
        logger.info("   • Avoids any VRAM allocation during file processing")
        logger.info("")
        logger.info("🔜 AFTER PHASE 1:")
        logger.info("   • Phase 2 will load Ollama for AI summarization")
        logger.info("   • Phase 2 will load BGE-M3 for embedding generation")
        logger.info("   • Embeddings generated from clean AI summaries, not raw content")
        logger.info("   • FAISS index built from high-quality AI summary embeddings")
        logger.info("")
        logger.info("=" * 80)
        logger.info("🏁 STARTING PHASE 1 PROCESSING...")
        logger.info("=" * 80)
        logger.info("")

        # Phase 1 is CPU/Disk only - no GPU model initialization
        logger.info("💾 Phase 1: CPU and disk-based processing (no GPU usage)")

        # Load checkpoint if resuming
        processed_files = set()
        start_idx = 0
        start_time = datetime.now()
        successful_count = 0
        failed_count = 0

        if resume:
            processed_files = self.checkpoint_manager.load_processed_files()
            progress = self.checkpoint_manager.load_progress()
            if progress:
                start_idx = progress.files_processed
                start_time = progress.start_time
                successful_count = progress.files_successful
                failed_count = progress.files_failed
                logger.info(
                    f"Resuming Phase 1 from file {start_idx + 1}/{len(self.file_rankings)}"
                )

        # Save initial metadata
        metadata = {
            "framework_version": "1.0.0",
            "phase": "basic_processing",
            "total_files": len(self.file_rankings),
            "started_at": start_time.isoformat(),
            "work_directory": str(self.work_dir),
            "analysis_file": str(self.analysis_file),
        }
        self.checkpoint_manager.save_metadata(metadata)

        # Process files sequentially - Phase 1 (basic processing)
        for idx, file_metadata in enumerate(
            self.file_rankings[start_idx:], start=start_idx
        ):

            # Skip if already processed
            if file_metadata.file_path in processed_files:
                continue

            # Read file content
            content = self._read_file_content(file_metadata.file_path)
            if content is None:
                failed_count += 1
                continue

            # Phase 1: Create basic result with content excerpt (CPU only)
            logger.info(
                f"Phase 1: Processing {file_metadata.file_name} (rank {file_metadata.rank})"
            )

            # Create basic result with content hash and metadata (CPU only)
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Create a basic result for Phase 1 (no AI summary, no embeddings yet)
            basic_result = ProcessingResult(
                file_metadata=file_metadata,
                success=True,
                processing_time=0.0,
                content_hash=content_hash,
                summary=None,  # Will be filled in Phase 2A
                embedding=None,  # Will be generated in Phase 2B
            )

            # Create content excerpt for BM25S indexing (CPU processing only)
            content_excerpt = self._create_content_excerpt(content, file_metadata)
            # Store content excerpt for BM25S but keep summary as None for Phase 1
            
            # Save basic result and add to BM25S corpus (disk I/O only)
            self.checkpoint_manager.save_result(basic_result)
            self.bm25_indexer.add_document(basic_result)

            successful_count += 1
            logger.info(
                f"✅ Phase 1: {file_metadata.file_name} - excerpt: {len(content_excerpt)} chars"
            )

            processed_count = (idx - start_idx) + 1

            # Update progress every 10 files or at the end
            if processed_count % 10 == 0 or processed_count == len(self.file_rankings):
                progress = self._calculate_progress_stats(
                    idx, start_time, processed_count, successful_count, failed_count
                )
                progress.checkpoint_saved = True
                self.checkpoint_manager.save_progress(progress)
                self._print_progress(progress)

        # Phase 1 Final statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Build search indexes from Phase 1 data (CPU only)
        logger.info("🔍 Building Phase 1 BM25S index (CPU processing)...")
        bm25_success = self.bm25_indexer.build_index()

        # No FAISS index in Phase 1 - will be built in Phase 2 after AI summaries
        logger.info("⚠️ Skipping FAISS index (will be built in Phase 2 from AI summaries)")

        phase1_stats = {
            "phase1_completed_at": end_time.isoformat(),
            "phase1_processing_time": total_time,
            "phase1_files_processed": successful_count + failed_count,
            "phase1_files_successful": successful_count,
            "phase1_files_failed": failed_count,
            "phase1_success_rate": (
                successful_count / (successful_count + failed_count)
                if (successful_count + failed_count) > 0
                else 0
            ),
            "phase1_average_rate": (
                (successful_count + failed_count) / total_time if total_time > 0 else 0
            ),
            "bm25_index_built": bm25_success,
            "bm25_documents": (
                len(self.bm25_indexer.corpus_texts) if bm25_success else 0
            ),
            "faiss_index_built": False,  # Will be built in Phase 2
            "faiss_embeddings": 0,  # Will be created in Phase 2
        }

        logger.info("✅ Phase 1 completed: Content processing and BM25S indexing")
        logger.info(
            f"📊 Phase 1 stats: {successful_count}/{len(self.file_rankings)} files processed"
        )

        return phase1_stats

    def _create_content_excerpt(self, content: str, file_metadata: FileMetadata) -> str:
        """Create a content excerpt for basic BM25S indexing (Phase 1)."""
        # Extract key elements for search indexing
        lines = content.split("\n")

        # Get first few lines of comments/documentation
        comments = []
        procedures = []
        functions = []

        for line in lines[:100]:  # Look at first 100 lines
            stripped = line.strip()
            if stripped.startswith("--") and len(stripped) > 5:
                comments.append(stripped[2:].strip())
            elif "PROCEDURE" in line.upper():
                procedures.append(line.strip())
            elif "FUNCTION" in line.upper():
                functions.append(line.strip())

        # Create structured excerpt
        excerpt_parts = []
        if comments:
            excerpt_parts.append("Comments: " + " | ".join(comments[:3]))
        if procedures:
            excerpt_parts.append("Procedures: " + " | ".join(procedures[:3]))
        if functions:
            excerpt_parts.append("Functions: " + " | ".join(functions[:3]))

        excerpt = " | ".join(excerpt_parts)

        # Fallback to content start if no structured content found
        if not excerpt:
            excerpt = content[:1000]

        return excerpt[:2000]  # Limit excerpt size

    def _create_developer_focused_embedding_text(
        self, ai_summary: str, metadata: FileMetadata
    ) -> str:
        """Create structured text optimized for developer search patterns."""
        
        # Extract domain from API name (e.g., CustomerOrder_API -> Customer Order)
        domain = metadata.api_name.replace("_API", "").replace("_", " ")
        
        # Select top API dependencies for context
        top_api_calls = ", ".join(metadata.api_calls[:5]) if metadata.api_calls else "none"
        
        # Build structured text that helps with various query types
        structured_parts = [
            # 1. Business Context (40% weight) - What does this do?
            f"Business Function: {ai_summary}",
            
            # 2. API Identity (25% weight) - What is this component?
            f"Component: {metadata.api_name} in {metadata.file_name}",
            f"Domain: {domain}",
            
            # 3. System Importance (15% weight) - How critical is this?
            f"System Importance: Rank {metadata.rank} of 9750",
            
            # 4. Dependencies (10% weight) - What does it interact with?
            f"Dependencies: {len(metadata.api_calls)} API calls including {top_api_calls}",
            
            # 5. Implementation Details (10% weight) - What's the scope?
            f"Implementation: {metadata.procedure_count} procedures, {metadata.function_count} functions",
        ]
        
        return " | ".join(structured_parts)

    async def _run_phase2_ai_and_embeddings(self, resume: bool = True) -> Dict[str, Any]:
        """Phase 2: AI summarization using Ollama + embedding generation using BGE-M3."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("🤖 PHASE 2: AI SUMMARIZATION + EMBEDDING GENERATION")
        logger.info("=" * 80)
        logger.info("")
        logger.info("🎯 GOAL: Generate AI summaries and high-quality embeddings from summaries")
        logger.info("")
        logger.info("🔄 STEP-BY-STEP PROCESS:")
        logger.info("   1️⃣  Load Ollama model for AI summarization")
        logger.info("   2️⃣  PHASE 2A: AI Summarization (Ollama only)")
        logger.info("       • For each file: Generate AI summary")
        logger.info("       • Save summaries to checkpoints")
        logger.info("       • Unload Ollama (free VRAM)")
        logger.info("   3️⃣  PHASE 2B: Embedding Generation (BGE-M3 only)")
        logger.info("       • Load BGE-M3 model")
        logger.info("       • For each AI summary: Generate structured embedding")
        logger.info("       • Build FAISS index from embeddings")
        logger.info("       • Unload BGE-M3 (free all VRAM)")
        logger.info("   4️⃣  Build final FAISS vector index")
        logger.info("   5️⃣  Unload both models (free all VRAM)")
        logger.info("")
        logger.info("� VRAM STRATEGY:")
        logger.info("   • Both models loaded simultaneously (but manageable)")
        logger.info("   • Process AI summaries sequentially (no thrashing)")
        logger.info("   • Embeddings from clean AI summaries (higher quality)")
        logger.info("   • No model switching during processing")
        logger.info("")
        logger.info("=" * 80)
        logger.info("🏁 STARTING PHASE 2 PROCESSING...")
        logger.info("=" * 80)
        logger.info("")

        # Check Ollama availability
        if not self.ollama_processor.ensure_model():
            logger.warning("Ollama model not available - skipping AI summarization")
            return {
                "phase2_completed": False,
                "phase2_error": "Ollama model not available",
            }

        # DON'T Initialize BGE-M3 model yet - wait until Phase 2B
        # This ensures Ollama has full VRAM access during AI summarization
        logger.info("⚡ BGE-M3 model will be loaded AFTER AI summarization completes")
        embedding_available = True  # We'll check this later in Phase 2B

        start_time = datetime.now()
        summarized_count = 0
        failed_count = 0

        # Get all results that need AI summarization
        all_results = self.checkpoint_manager.load_all_results()

        logger.info(
            f"Phase 2: Processing {len(all_results)} files for AI summarization"
        )

        for idx, result in enumerate(all_results):
            # Skip if already has AI summary (not just content excerpt)
            if result.has_ai_summary:
                continue

            # Find the file metadata
            file_metadata = None
            for metadata in self.file_rankings:
                if metadata.file_path == result.file_metadata.file_path:
                    file_metadata = metadata
                    break

            if not file_metadata:
                failed_count += 1
                continue

            # Read file content again
            content = self._read_file_content(file_metadata.file_path)
            if content is None:
                failed_count += 1
                continue

            logger.info(
                f"Phase 2: AI summarizing {file_metadata.file_name} ({idx+1}/{len(all_results)})"
            )

            # Process with Ollama
            ai_result = self.ollama_processor.process_file(content, file_metadata)

            if ai_result.success and ai_result.summary:
                # Update the result with AI summary
                result.summary = ai_result.summary
                result.tokens_used = ai_result.tokens_used
                result.processing_time = ai_result.processing_time
                result.has_ai_summary = True  # Mark as AI-generated summary
                
                # DON'T generate embeddings yet - save that for Phase 2B
                # Save AI summary result to separate Phase 2A file
                self.checkpoint_manager.save_phase2a_result(result)

                summarized_count += 1
                embedding_status = "✓" if result.embedding is not None else "✗"
                logger.info(
                    f"✅ Phase 2: {file_metadata.file_name}: {len(ai_result.summary)} chars, embedding: {embedding_status}"
                )
            else:
                failed_count += 1
                logger.warning(
                    f"❌ Phase 2: {file_metadata.file_name}: {ai_result.error_message}"
                )

            # Progress update every 10 files
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Phase 2 Progress: {idx+1}/{len(all_results)} files processed"
                )

        # ============================================================================
        # PHASE 2A COMPLETE - AI Summarization done, now start Phase 2B
        # ============================================================================
        logger.info("")
        logger.info("🔥 PHASE 2A COMPLETE: AI Summarization finished")
        logger.info(f"📊 Phase 2A stats: {summarized_count} files summarized")
        
        # CRITICAL: Shutdown Ollama to free VRAM for BGE-M3
        logger.info("💾 Freeing VRAM by shutting down Ollama before BGE-M3 loading...")
        self.ollama_processor.shutdown_ollama()
        
        logger.info("")
        logger.info("🚀 STARTING PHASE 2B: EMBEDDING GENERATION...")
        logger.info("⚡ Now loading BGE-M3 model (Ollama shutdown complete)")
        
        # NOW initialize BGE-M3 model for embedding generation
        logger.info("Initializing BGE-M3 embedding model...")
        if not self.embedding_generator.initialize_model():
            logger.warning("BGE-M3 model initialization failed - embeddings will be skipped")
            embedding_available = False
        else:
            logger.info("✅ BGE-M3 model loaded successfully")
            embedding_available = True

        # Phase 2B: Generate embeddings from AI summaries
        embedding_count = 0
        if embedding_available:
            # Reload results to get all AI summaries
            all_results_with_summaries = self.checkpoint_manager.load_all_results()
            
            logger.info(f"Phase 2B: Generating embeddings for {len(all_results_with_summaries)} AI summaries...")
            
            for idx, result in enumerate(all_results_with_summaries):
                # Skip if no AI summary or already has embedding
                if not result.has_ai_summary or result.embedding is not None:
                    continue
                    
                # Find the file metadata
                file_metadata = None
                for metadata in self.file_rankings:
                    if metadata.file_path == result.file_metadata.file_path:
                        file_metadata = metadata
                        break
                        
                if not file_metadata:
                    continue
                    
                # Create structured embedding text combining AI summary with metadata
                embedding_text = self._create_developer_focused_embedding_text(
                    result.summary, file_metadata
                )
                logger.debug(f"Phase 2B: Generating embedding for {file_metadata.file_name}")
                embedding = self.embedding_generator.generate_embedding(embedding_text)
                
                if embedding:
                    result.embedding = embedding
                    embedding_count += 1
                    
                    # Add to FAISS manager
                    self.faiss_manager.add_embedding(
                        embedding,
                        {
                            "file_path": file_metadata.file_path,
                            "file_name": file_metadata.file_name,
                            "api_name": file_metadata.api_name,
                            "rank": file_metadata.rank,
                            "importance_score": file_metadata.combined_importance_score,
                            "reference_count": file_metadata.reference_count,
                            "content_hash": result.content_hash,
                            "summary_length": len(result.summary) if result.summary else 0,
                        },
                    )
                    
                    # Save updated result with embedding to Phase 2B file
                    self.checkpoint_manager.save_phase2b_result(result)
                    
                    # Progress reporting
                    if (embedding_count % 10) == 0:
                        logger.info(f"Phase 2B Progress: {embedding_count} embeddings generated")

        # Build FAISS index from all embeddings generated in Phase 2B
        faiss_success = False
        faiss_count = 0
        if embedding_available:
            logger.info("🧠 Building FAISS embedding index from AI summaries...")
            faiss_success = self.faiss_manager.build_faiss_index()
            faiss_count = len(self.faiss_manager.embeddings) if faiss_success else 0
        else:
            logger.info("⚠️ Skipping FAISS index (BGE-M3 not available)")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        phase2_stats = {
            "phase2_completed_at": end_time.isoformat(),
            "phase2_processing_time": total_time,
            "phase2_files_processed": summarized_count + failed_count,
            "phase2_files_successful": summarized_count,
            "phase2_files_failed": failed_count,
            "phase2_success_rate": (
                summarized_count / (summarized_count + failed_count)
                if (summarized_count + failed_count) > 0
                else 0
            ),
            "phase2_average_rate": (
                (summarized_count + failed_count) / total_time if total_time > 0 else 0
            ),
            "faiss_index_built": faiss_success,
            "faiss_embeddings": faiss_count,
        }

        logger.info("✅ Phase 2 completed: AI Summarization + Embedding Generation")
        logger.info(f"📊 Phase 2 stats: {summarized_count} files summarized, {faiss_count} embeddings generated")

        return phase2_stats


# CLI Interface Functions
def setup_embedding_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Set up required directories for embedding processing."""
    work_dir = base_dir / "_work"
    checkpoint_dir = base_dir / "embedding_checkpoints"
    analysis_file = base_dir / "comprehensive_plsql_analysis.json"

    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory not found: {work_dir}")

    # Analysis file will be generated automatically if it doesn't exist
    # (Removed the requirement check since we have built-in analyzer)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return work_dir, checkpoint_dir, analysis_file


async def run_embedding_command(args) -> int:
    """Handle the embedding command."""
    try:
        base_dir = Path.cwd()
        work_dir, checkpoint_dir, analysis_file = setup_embedding_directories(base_dir)

        # Initialize framework
        framework = ProductionEmbeddingFramework(
            work_dir=work_dir,
            analysis_file=analysis_file,
            checkpoint_dir=checkpoint_dir,
            model=args.model,
            max_files=args.max_files,
        )

        # Run the pipeline
        stats = await framework.run_embedding_pipeline(resume=not args.no_resume)

        # Print final summary
        print(f"\n🎉 Embedding pipeline completed!")
        print(f"✅ Successfully processed: {stats['files_successful']:,} files")
        print(f"❌ Failed: {stats['files_failed']:,} files")
        print(f"📊 Success rate: {stats['success_rate']:.1%}")
        print(f"⚡ Average rate: {stats['average_rate']:.2f} files/sec")
        print(f"⏱️  Total time: {timedelta(seconds=stats['total_processing_time'])}")

        if stats.get("bm25_index_built"):
            print(f"🔍 BM25S index built: {stats['bm25_documents']:,} documents")
        else:
            print(f"⚠️  BM25S index build failed (check logs)")

        if stats.get("faiss_index_built"):
            print(f"🧠 FAISS index built: {stats['faiss_embeddings']:,} embeddings")
            print(f"📁 Search indexes saved to: {framework.bm25_indexer.index_dir}")
        elif stats.get("faiss_embeddings", 0) > 0:
            print(
                f"⚠️  FAISS index build failed but {stats['faiss_embeddings']} embeddings generated"
            )
        else:
            print(f"⚠️  BGE-M3 embeddings skipped (dependencies not available)")

        return 0

    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}")
        return 1
