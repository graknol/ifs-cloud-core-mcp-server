"""
Directory resolution utilities for IFS Cloud MCP Server.

This module centralizes all directory resolution logic to provide a single
point of maintenance for directory structure changes.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def get_data_directory() -> Path:
    """Get the platform-appropriate data directory for IFS Cloud files."""
    app_name = "ifs_cloud_mcp_server"

    if sys.platform == "win32":
        # Windows: %APPDATA%/ifs_cloud_mcp_server
        base_path = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        # macOS: ~/Library/Application Support/ifs_cloud_mcp_server
        base_path = Path.home() / "Library" / "Application Support"
    else:
        # Linux/Unix: ~/.local/share/ifs_cloud_mcp_server
        base_path = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )

    return base_path / app_name


def resolve_version_to_work_directory(version: str) -> Path:
    """
    Resolve a version to its work directory path.

    This function intelligently navigates nested directory structures
    to find the actual IFS source directory containing 'fndbas' and 'accrul' components.

    Args:
        version: The version identifier

    Returns:
        Path to the source directory containing IFS components

    Raises:
        FileNotFoundError: If no valid work directory is found
        ValueError: If multiple ambiguous work directories are found
    """
    data_dir = get_data_directory()
    extract_path = data_dir / "versions" / version / "source"

    if not extract_path.exists():
        available_versions = (
            [d.name for d in (data_dir / "versions").iterdir() if d.is_dir()]
            if (data_dir / "versions").exists()
            else []
        )
        if available_versions:
            logger.warning(f"Available versions: {', '.join(available_versions)}")
        raise FileNotFoundError(f"Version directory not found: {extract_path}")

    def _find_ifs_work_directory(current_path: Path) -> Path:
        """
        Recursively search for IFS work directory containing fndbas and accrul.

        The algorithm:
        1. First check if current directory contains fndbas and accrul - if so, return it
        2. If there's exactly one subdirectory, descend into it
        3. If multiple subdirectories are found, fail (return false/raise error)
        4. Continue until we find a directory with both components
        """
        logger.debug(f"Searching for IFS work directory in: {current_path}")

        # First check if current directory contains the required components
        if (current_path / "fndbas").exists() and (current_path / "accrul").exists():
            logger.info(f"Found IFS work directory: {current_path}")
            return current_path

        # Get all subdirectories (exclude files)
        subdirs = [d for d in current_path.iterdir() if d.is_dir()]

        if not subdirs:
            raise FileNotFoundError(
                f"No subdirectories found in {current_path}. "
                f"Expected to find a directory containing both 'fndbas' and 'accrul' subdirectories."
            )

        # If multiple subdirectories are found, fail
        if len(subdirs) > 1:
            subdir_names = [d.name for d in subdirs]
            raise ValueError(
                f"Multiple subdirectories found in {current_path}: {subdir_names}. "
                f"Expected exactly one subdirectory or a directory with fndbas and accrul."
            )

        # Exactly one subdirectory - recursively search it
        return _find_ifs_work_directory(subdirs[0])

    try:
        return _find_ifs_work_directory(extract_path)
    except (FileNotFoundError, ValueError):
        raise


def get_version_base_directory(version: str) -> Path:
    """Get the base directory for a specific version."""
    data_dir = get_data_directory()
    safe_version = "".join(c for c in version if c.isalnum() or c in "._-")
    return data_dir / "versions" / safe_version


def get_version_source_directory(version: str) -> Path:
    """Get the source directory for a specific version."""
    return resolve_version_to_work_directory(version)


def get_version_analysis_directory(version: str) -> Path:
    """Get the analysis directory for a specific version."""
    base_dir = get_version_base_directory(version)
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir


def get_version_analysis_file(version: str) -> Path:
    """Get the main analysis file path for a specific version."""
    analysis_dir = get_version_analysis_directory(version)
    return analysis_dir / "comprehensive_plsql_analysis.json"


def get_version_embedding_checkpoints_directory(version: str) -> Path:
    """Get the embedding checkpoints directory for a specific version."""
    base_dir = get_version_base_directory(version)
    checkpoint_dir = base_dir / "embedding_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_version_bm25s_directory(version: str) -> Path:
    """Get the BM25S index directory for a specific version."""
    base_dir = get_version_base_directory(version)
    bm25s_dir = base_dir / "bm25s"
    bm25s_dir.mkdir(parents=True, exist_ok=True)
    return bm25s_dir


def get_version_faiss_directory(version: str) -> Path:
    """Get the FAISS index directory for a specific version."""
    base_dir = get_version_base_directory(version)
    faiss_dir = base_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    return faiss_dir


def setup_embedding_directories(version: str) -> Tuple[Path, Path, Path]:
    """
    Set up required directories for embedding processing for a specific version.

    Args:
        version: The version identifier

    Returns:
        Tuple of (work_dir, checkpoint_dir, analysis_file)

    Raises:
        FileNotFoundError: If work directory not found
    """
    work_dir = get_version_source_directory(version)
    checkpoint_dir = get_version_embedding_checkpoints_directory(version)
    analysis_file = get_version_analysis_file(version)

    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory not found: {work_dir}")

    return work_dir, checkpoint_dir, analysis_file


def setup_analysis_engine_directories(
    version: str,
) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Set up all required directories for AnalysisEngine for a specific version.

    Args:
        version: The version identifier

    Returns:
        Tuple of (source_dir, checkpoint_dir, bm25s_dir, faiss_dir, analysis_file)

    Raises:
        FileNotFoundError: If source directory not found
    """
    source_dir = get_version_source_directory(version)
    checkpoint_dir = get_version_embedding_checkpoints_directory(version)
    bm25s_dir = get_version_bm25s_directory(version)
    faiss_dir = get_version_faiss_directory(version)
    analysis_file = get_version_analysis_file(version)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    return source_dir, checkpoint_dir, bm25s_dir, faiss_dir, analysis_file


def get_supported_extensions() -> Set[str]:
    """Get the set of file extensions that IFS Cloud MCP Server supports."""
    return {
        ".entity",
        ".plsql",
        ".views",
        ".storage",
        ".fragment",
        ".projection",
        ".client",
    }


def list_available_versions() -> List[str]:
    """List all available versions in the versions directory."""
    data_dir = get_data_directory()
    versions_dir = data_dir / "versions"

    if not versions_dir.exists():
        return []

    return [d.name for d in versions_dir.iterdir() if d.is_dir()]
