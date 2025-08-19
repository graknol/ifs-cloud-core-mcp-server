#!/usr/bin/env python3
"""
Rebuild BM25S index from existing Phase 1 results without touching other indexes.

This script:
1. Loads existing Phase 1 results (basic processing with content)
2. Rebuilds the BM25S index with enhanced preprocessing (punctuation handling, tiktoken, stemming)
3. Preserves FAISS embeddings and other indexes
4. Creates new document ID mapping for queries
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.embedding_processor import (
    BM25SIndexer,
    EmbeddingCheckpointManager,
    ProcessingResult,
)


def setup_logging():
    """Setup logging for the reindexing process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("reindex_bm25s.log"),
        ],
    )
    return logging.getLogger(__name__)


def load_file_content(file_path: str) -> str:
    """Load the full content of a file for BM25S processing."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Could not read file {file_path}: {e}")
        return ""


def main():
    logger = setup_logging()

    # Configuration
    checkpoint_dir = Path("embedding_checkpoints")
    work_dir = Path("_work")  # Adjust if your work directory is different

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory {checkpoint_dir} not found!")
        logger.error(
            "Please run the embedding pipeline first to generate Phase 1 results."
        )
        return 1

    # Initialize components
    checkpoint_manager = EmbeddingCheckpointManager(checkpoint_dir)
    search_indexes_dir = checkpoint_dir / "search_indexes"

    # Create new BM25S indexer (this will overwrite the existing index)
    logger.info("🔄 Initializing new BM25S indexer with enhanced preprocessing...")
    bm25_indexer = BM25SIndexer(search_indexes_dir)

    # Load existing Phase 1 results
    logger.info("📂 Loading existing Phase 1 processing results...")
    all_results = checkpoint_manager.load_all_results()

    if not all_results:
        logger.error("No existing processing results found!")
        logger.error(
            "Please run the embedding pipeline first: uv run python -m src.ifs_cloud_mcp_server.main embed --max-files 10"
        )
        return 1

    logger.info(f"Found {len(all_results)} existing processing results")

    # Filter for successful results only
    successful_results = [r for r in all_results if r.success]
    logger.info(f"Processing {len(successful_results)} successful results")

    # Rebuild BM25S index
    logger.info("🔍 Rebuilding BM25S index with enhanced preprocessing...")
    processed_count = 0

    for result in successful_results:
        try:
            # Load full file content for advanced preprocessing
            full_content = load_file_content(result.file_metadata.file_path)

            # Add document to new BM25S index
            bm25_indexer.add_document(result, full_content=full_content)
            processed_count += 1

            if processed_count % 100 == 0:
                logger.info(
                    f"Processed {processed_count}/{len(successful_results)} files..."
                )

        except Exception as e:
            logger.warning(f"Failed to process {result.file_metadata.file_name}: {e}")

    logger.info(f"✅ Processed {processed_count} files for BM25S indexing")

    # Build the enhanced BM25S index
    logger.info("🏗️ Building enhanced BM25S index...")
    if bm25_indexer.build_index():
        logger.info("🎉 BM25S index rebuilt successfully!")
        logger.info("Enhanced features:")
        logger.info("  ✓ Full-content indexing (not just excerpts)")
        logger.info("  ✓ Tiktoken tokenization (GPT-4 encoding)")
        logger.info("  ✓ English stemming with Snowball stemmer")
        logger.info("  ✓ Custom stopwords analysis")
        logger.info("  ✓ Source code punctuation handling")
        logger.info("  ✓ Document ID to filepath mapping")
        return 0
    else:
        logger.error("❌ Failed to rebuild BM25S index")
        return 1


if __name__ == "__main__":
    exit(main())
