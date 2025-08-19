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
    extract_path = data_dir / "versions" / version

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
        1. If there's only one subdirectory, descend into it
        2. Continue until we find a directory structure with both fndbas and accrul
        3. Return that directory path
        """
        # Get all subdirectories (exclude files)
        subdirs = [d for d in current_path.iterdir() if d.is_dir()]

        # If there's exactly one subdirectory, keep going down
        if len(subdirs) == 1:
            return _find_ifs_work_directory(subdirs[0])

        # Check each subdirectory for the presence of "source" directory
        # and validate it contains both fndbas and accrul
        work_dirs_found = []
        for subdir in subdirs:
            potential_work = subdir / "source"
            if potential_work.exists() and potential_work.is_dir():
                fndbas_exists = (potential_work / "fndbas").exists()
                accrul_exists = (potential_work / "accrul").exists()

                if fndbas_exists and accrul_exists:
                    work_dirs_found.append(potential_work)

        if len(work_dirs_found) == 1:
            return work_dirs_found[0]
        elif len(work_dirs_found) > 1:
            dirs_str = "\n  ".join(str(d) for d in work_dirs_found)
            raise ValueError(
                f"Multiple valid work directories found. Please specify more precisely:\n  {dirs_str}"
            )
        else:
            # No valid work directories found
            raise FileNotFoundError(
                f"No valid work directory found in {current_path}. "
                f"Expected to find a 'source' directory containing both 'fndbas' and 'accrul' subdirectories."
            )

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


def get_version_faiss_directory(version: str) -> Path:
    """Get the FAISS index directory for a specific version."""
    base_dir = get_version_base_directory(version)
    faiss_dir = base_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    return faiss_dir


def get_version_bm25s_directory(version: str) -> Path:
    """Get the BM25S index directory for a specific version."""
    base_dir = get_version_base_directory(version)
    bm25s_dir = base_dir / "bm25s"
    bm25s_dir.mkdir(parents=True, exist_ok=True)
    return bm25s_dir


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


def get_supported_extensions() -> Set[str]:
    """Get the set of file extensions that IFS Cloud MCP Server supports."""
    return {
        ".entity",
        ".plsql",
        ".views",
        ".storage",
        ".cdb",
        ".fdb",
        ".apy",
        ".apv",
        ".svc",
        ".rpt",
        ".cpi",
        ".bi",
        ".fragment",
        ".projection",
        ".iac",
        ".client",
    }


def list_available_versions() -> List[str]:
    """List all available versions in the versions directory."""
    data_dir = get_data_directory()
    versions_dir = data_dir / "versions"

    if not versions_dir.exists():
        return []

    return [d.name for d in versions_dir.iterdir() if d.is_dir()]


def get_version_indexes_directory(version: str) -> Path:
    """Get the legacy indexes directory for compatibility (if needed)."""
    data_dir = get_data_directory()
    return data_dir / "indexes" / version


def validate_version_structure(version: str) -> bool:
    """
    Validate that a version has the expected directory structure.

    Args:
        version: The version identifier

    Returns:
        True if valid structure, False otherwise
    """
    try:
        base_dir = get_version_base_directory(version)
        work_dir = get_version_source_directory(version)

        # Check that required components exist
        fndbas_exists = (work_dir / "fndbas").exists()
        accrul_exists = (work_dir / "accrul").exists()

        return (
            base_dir.exists() and work_dir.exists() and fndbas_exists and accrul_exists
        )

    except (FileNotFoundError, ValueError):
        return False
