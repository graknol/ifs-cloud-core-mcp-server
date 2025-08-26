#!/usr/bin/env python3
"""
Extract top 10 PL/SQL files per module based on PageRank scores and copy them to top_10/ folder.
"""

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data_directory() -> Path:
    """Get the data directory path."""
    import os

    return (
        Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "ifs_cloud_mcp_server"
    )


def extract_module_from_path(file_path: str) -> str:
    """Extract module name from file path."""
    path = Path(file_path)
    parts = path.parts

    # Look for common module directory patterns in IFS Cloud
    for i, part in enumerate(parts):
        if part == "source" and i + 1 < len(parts):
            # Next part after 'source' is usually the module
            return parts[i + 1].upper()
        elif part == "database" and i - 1 >= 0:
            # Previous part before 'database' is usually the module
            return parts[i - 1].upper()
        elif part.endswith("ul") and part != "source":
            # Many IFS modules end with 'ul' (e.g., accrul, manuul, etc.)
            return part.upper()

    # Fallback: try to extract from beginning of path
    if len(parts) >= 2:
        return parts[1].upper()

    return "UNKNOWN"


def load_pagerank_data(version: str = "25.1.0") -> List[Dict]:
    """Load PageRank data from the ranked.jsonl file."""
    data_dir = get_data_directory()
    pagerank_file = data_dir / "versions" / version / "ranked.jsonl"

    if not pagerank_file.exists():
        raise FileNotFoundError(f"PageRank file not found: {pagerank_file}")

    logger.info(f"Loading PageRank data from: {pagerank_file}")

    ranked_files = []
    with open(pagerank_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ranked_files.append(json.loads(line))

    return ranked_files


def get_top_10_per_module(ranked_files: List[Dict]) -> Dict[str, List[Dict]]:
    """Get top 10 files per module based on PageRank scores."""
    module_files = defaultdict(list)

    # Group files by module
    for file_info in ranked_files:
        # Only process .plsql files
        if not file_info["file_name"].endswith(".plsql"):
            continue

        module = extract_module_from_path(file_info["relative_path"])
        module_files[module].append(file_info)

    # Sort each module's files by PageRank score and take top 10
    top_10_per_module = {}
    for module, files in module_files.items():
        # Files are already sorted by PageRank score in descending order
        sorted_files = sorted(files, key=lambda x: x["pagerank_score"], reverse=True)
        top_10_per_module[module] = sorted_files[:10]

    return top_10_per_module


def copy_files_to_top_10_structure(
    top_10_per_module: Dict[str, List[Dict]], source_root: Path, target_root: Path
):
    """Copy the top 10 files per module to the target directory structure."""
    target_root = Path(target_root)

    total_files = sum(len(files) for files in top_10_per_module.values())
    copied_files = 0
    failed_copies = 0

    logger.info(f"Copying {total_files} files from {len(top_10_per_module)} modules...")

    for module, files in top_10_per_module.items():
        module_dir = target_root / "top_10" / module.lower()
        module_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing module {module}: {len(files)} files")

        for i, file_info in enumerate(files, 1):
            try:
                # Source file path
                source_file = source_root / file_info["relative_path"]

                # Target file path
                target_file = module_dir / file_info["file_name"]

                if source_file.exists():
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    logger.info(
                        f"  {i:2d}. {file_info['file_name']} (PageRank: {file_info['pagerank_score']:.6f})"
                    )
                else:
                    logger.warning(
                        f"  {i:2d}. {file_info['file_name']} - SOURCE NOT FOUND: {source_file}"
                    )
                    failed_copies += 1

            except Exception as e:
                logger.error(f"  {i:2d}. {file_info['file_name']} - ERROR: {e}")
                failed_copies += 1

    logger.info(f"\n🎉 Copy complete!")
    logger.info(f"✅ Successfully copied: {copied_files} files")
    logger.info(f"❌ Failed to copy: {failed_copies} files")
    logger.info(f"📁 Files copied to: {target_root / 'top_10'}")


def main():
    """Main function to extract and copy top 10 files per module."""
    print("🏆 IFS Cloud Top 10 Files per Module Extractor")
    print("=" * 60)

    try:
        # Load PageRank data
        ranked_files = load_pagerank_data("25.1.0")
        logger.info(f"Loaded {len(ranked_files)} ranked files")

        # Get top 10 per module
        top_10_per_module = get_top_10_per_module(ranked_files)

        logger.info(f"Found {len(top_10_per_module)} modules:")
        for module, files in top_10_per_module.items():
            logger.info(f"  • {module}: {len(files)} files")

        # Determine source root (where the IFS source code is)
        source_root = Path(r"C:\repos\_ifs\25.1.0")
        if not source_root.exists():
            # Try alternative locations
            alternative_paths = [
                Path(r"C:\repos\ifs\25.1.0"),
                Path(r"C:\repos\IFS\25.1.0"),
                Path.cwd() / "ifs_source",
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    source_root = alt_path
                    break
            else:
                logger.error("❌ Could not find IFS source directory!")
                logger.error(
                    "   Please ensure IFS source code is available at one of these locations:"
                )
                logger.error(f"   • C:\\repos\\_ifs\\25.1.0")
                for alt_path in alternative_paths:
                    logger.error(f"   • {alt_path}")
                return

        logger.info(f"Using source root: {source_root}")

        # Copy files to top_10 structure
        target_root = Path.cwd()
        copy_files_to_top_10_structure(top_10_per_module, source_root, target_root)

        # Show summary of modules and their top files
        print("\n📊 Summary of Top 10 Files per Module:")
        print("=" * 60)
        for module in sorted(top_10_per_module.keys()):
            files = top_10_per_module[module]
            print(f"\n🔹 {module} ({len(files)} files):")
            for i, file_info in enumerate(files[:5], 1):  # Show top 5 in summary
                score = file_info["pagerank_score"]
                name = file_info["file_name"]
                print(f"   {i}. {name} ({score:.6f})")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
