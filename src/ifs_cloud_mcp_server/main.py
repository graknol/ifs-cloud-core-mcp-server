"""Main entry point for IFS Cloud MCP Server."""

import logging
import sys
from pathlib import Path

from .directory_utils import (
    get_data_directory,
    get_supported_extensions,
    resolve_version_to_work_directory,
)

# Lazy import: from .server_fastmcp import IFSCloudMCPServer
# Lazy import: from .embedding_processor import run_embedding_command


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )


def resolve_version_to_index_path(version: str) -> Path:
    """Resolve a version name to its corresponding index path.

    Args:
        version: Version identifier

    Returns:
        Path to the index directory for this version

    Raises:
        ValueError: If version doesn't exist or has no index
    """
    data_dir = get_data_directory()
    safe_version = "".join(c for c in version if c.isalnum() or c in "._-")

    extract_path = data_dir / "versions" / safe_version
    index_path = data_dir / "indexes" / safe_version

    if not extract_path.exists():
        raise ValueError(
            f"Version '{version}' not found. Available versions can be listed with: python -m src.ifs_cloud_mcp_server.main list"
        )

    if not index_path.exists():
        raise ValueError(
            f"Version '{version}' found but not indexed. Please re-import with: python -m src.ifs_cloud_mcp_server.main import <zip_file> --version {version}"
        )

    return index_path


def get_version_from_zip(zip_path: Path) -> str:
    """Extract version from version.txt file inside the ZIP.

    Args:
        zip_path: Path to the ZIP file

    Returns:
        Version string extracted from checkout/fndbas/source/version.txt

    Raises:
        FileNotFoundError: If ZIP file doesn't exist
        ValueError: If version.txt is not found or version cannot be determined
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Look for version.txt in the expected location
        version_file_path = "checkout/fndbas/source/version.txt"

        if version_file_path not in zip_ref.namelist():
            raise ValueError(f"Version file not found: {version_file_path}")

        try:
            with zip_ref.open(version_file_path) as version_file:
                version_content = version_file.read().decode("utf-8").strip()

            # Extract version number - should be in format like "25.1.0" or similar
            # The version.txt might have additional content, so we need to extract just the version
            lines = version_content.splitlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("//"):
                    # Take the first non-comment, non-empty line as version
                    version = line.split()[
                        0
                    ]  # Take first word in case there are additional details
                    if version:
                        return version

            raise ValueError("No valid version found in version.txt")

        except Exception as e:
            raise ValueError(f"Failed to read version from {version_file_path}: {e}")


def extract_ifs_cloud_zip(zip_path: Path, version: str) -> Path:
    """Extract IFS Cloud ZIP file to versioned directory with only supported files.

    Args:
        zip_path: Path to the ZIP file
        version: Version identifier for this extract

    Returns:
        Path to the extracted directory

    Raises:
        FileNotFoundError: If ZIP file doesn't exist
        zipfile.BadZipFile: If ZIP file is corrupted
        ValueError: If version contains invalid characters
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Sanitize version name for filesystem
    safe_version = "".join(c for c in version if c.isalnum() or c in "._-")
    if not safe_version:
        raise ValueError("Version must contain at least one alphanumeric character")

    # Get extraction directory
    data_dir = get_data_directory()
    extract_dir = data_dir / "versions" / safe_version

    # Remove existing extraction if it exists
    if extract_dir.exists():
        import shutil

        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = get_supported_extensions()
    extracted_count = 0

    logging.info(f"Extracting IFS Cloud files from {zip_path} to {extract_dir}")
    logging.info(f"Supported file types: {', '.join(sorted(supported_extensions))}")

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip directories
            if file_info.is_dir():
                continue

            file_path = Path(file_info.filename)

            # Skip files not under checkout directory
            if not file_info.filename.startswith("checkout/"):
                continue

            # Check if file has supported extension
            if file_path.suffix.lower() in supported_extensions:
                try:
                    # Remove "checkout/" from the path and extract to source/
                    relative_path = Path(*file_path.parts[1:])  # Skip "checkout"
                    target_path = extract_dir / "source" / relative_path

                    # Create target directory if it doesn't exist
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extract file data and write to target location
                    with zip_ref.open(file_info) as source_file:
                        with open(target_path, "wb") as target_file:
                            target_file.write(source_file.read())
                    extracted_count += 1

                    if extracted_count % 100 == 0:
                        logging.info(f"Extracted {extracted_count} files...")

                except Exception as e:
                    logging.warning(f"Failed to extract {file_info.filename}: {e}")
                    continue

    logging.info(
        f"Successfully extracted {extracted_count} supported files to {extract_dir}"
    )
    return extract_dir


def handle_import_command(args) -> int:
    """Handle the import command - auto-detect version and extract files to version directory."""
    try:
        zip_path = Path(args.zip_file)

        # Auto-detect version from ZIP file
        logging.info(f"🔍 Detecting version from {zip_path}")
        version = get_version_from_zip(zip_path)
        logging.info(f"📋 Detected version: {version}")

        # Extract ZIP file to version directory
        extract_path = extract_ifs_cloud_zip(zip_path, version)

        logging.info(f"✅ Import completed successfully!")
        logging.info(f"📁 Extracted files: {extract_path}")
        logging.info(f"🏷️  Version: {version}")
        logging.info("")
        logging.info("Files extracted successfully. To work with this version:")
        logging.info(
            f'  Analyze:    python -m src.ifs_cloud_mcp_server.main analyze --version "{version}"'
        )
        logging.info(
            f'  MCP Server: python -m src.ifs_cloud_mcp_server.main server --version "{version}"'
        )
        return 0

    except Exception as e:
        logging.error(f"❌ Import failed: {e}")
        return 1


def handle_list_command(args) -> int:
    """Handle the list command."""
    import json
    from datetime import datetime

    try:
        data_dir = get_data_directory()
        versions_dir = data_dir / "versions"
        indexes_dir = data_dir / "indexes"

        versions = []

        # Scan for available versions
        if versions_dir.exists():
            for version_dir in versions_dir.iterdir():
                if version_dir.is_dir():
                    index_path = indexes_dir / version_dir.name

                    # Get file count
                    file_count = 0
                    if version_dir.exists():
                        for ext in get_supported_extensions():
                            file_count += len(list(version_dir.rglob(f"*{ext}")))

                    # Get creation time
                    try:
                        created = datetime.fromtimestamp(version_dir.stat().st_ctime)
                        created_str = created.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        created_str = "Unknown"

                    version_info = {
                        "version": version_dir.name,
                        "extract_path": str(version_dir),
                        "index_path": str(index_path),
                        "has_index": index_path.exists(),
                        "file_count": file_count,
                        "created": created_str,
                    }
                    versions.append(version_info)

        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x["created"], reverse=True)

        if args.json:
            # Output JSON for programmatic use
            print(json.dumps(versions, indent=2))
        else:
            # Human-readable output
            if not versions:
                print("No IFS Cloud versions found.")
                print("")
                print("To import a version:")
                print(
                    "  python -m src.ifs_cloud_mcp_server.main import <zip_file> --version <version_name>"
                )
            else:
                print("Available IFS Cloud versions:")
                print("")
                for v in versions:
                    status = "✅ Indexed" if v["has_index"] else "⚠️  Not indexed"
                    print(f"📦 {v['version']}")
                    print(f"   Status: {status}")
                    print(f"   Files: {v['file_count']:,}")
                    print(f"   Created: {v['created']}")
                    if v["has_index"]:
                        print(
                            f"   Command: python -m src.ifs_cloud_mcp_server.main server --version \"{v['version']}\""
                        )
                    print("")

        return 0

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"❌ Failed to list versions: {e}")
        return 1


def handle_server_command(args) -> int:
    """Handle the server command."""
    try:
        # Resolve version to index path (required)
        index_path = resolve_version_to_index_path(args.version)
        logging.info(f"Using IFS Cloud version: {args.version}")
        logging.info(f"Index path: {index_path}")

        # Create index directory if it doesn't exist
        index_path.mkdir(parents=True, exist_ok=True)

        # Lazy import to avoid CLI slowdown
        from .server_fastmcp import IFSCloudMCPServer

        # Create server
        server = IFSCloudMCPServer(
            index_path=index_path, name=getattr(args, "name", "ifs-cloud-mcp-server")
        )

        # Try to run the server, handling asyncio context issues
        server.run(transport_type=getattr(args, "transport", "stdio"))

    except ValueError as e:
        logging.error(f"❌ Version error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except RuntimeError as e:
        if "Already running asyncio in this thread" in str(e):
            logging.error("❌ AsyncIO conflict detected")
            logging.error("This server must be run as a standalone process")
            logging.error("Please ensure no other asyncio event loop is running")
            return 1
        else:
            logging.error(f"Runtime error: {e}")
            return 1
    except Exception as e:
        logging.error(f"Server error: {e}")
        return 1
    finally:
        if "server" in locals() and server is not None:
            server.cleanup()

    return 0


def handle_bm25s_reindex_command(args) -> int:
    """Handle the BM25S reindex command."""
    try:
        from pathlib import Path
        from .analysis_processor import ProductionEmbeddingFramework
        import json

        # Determine work directory and file paths using version
        work_dir = resolve_version_to_work_directory(args.version)
        # Files go in the version's extract directory
        data_dir = get_data_directory()
        safe_version = "".join(c for c in args.version if c.isalnum() or c in "._-")
        base_dir = data_dir / "versions" / safe_version
        analysis_dir = base_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_file = (
            analysis_dir / "comprehensive_plsql_analysis.json"
        )  # Fixed filename
        checkpoint_dir = base_dir / "embedding_checkpoints"
        logging.info(f"Using IFS Cloud version: {args.version}")
        logging.info(f"Work directory: {work_dir}")

        if not work_dir.exists():
            logging.error(f"❌ Work directory not found: {work_dir}")
            return 1

        if not analysis_file.exists():
            logging.error(f"❌ Analysis file not found: {analysis_file}")
            logging.error(
                f"   Run analysis first: python -m src.ifs_cloud_mcp_server.main analyze --version {args.version}"
            )
            return 1

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"🔄 Starting BM25S index rebuild...")
        logging.info(f"📁 Work directory: {work_dir}")
        logging.info(f"📊 Analysis file: {analysis_file}")
        logging.info(f"💾 Checkpoint directory: {checkpoint_dir}")

        if args.max_files:
            logging.info(f"🔢 Max files limit: {args.max_files}")

        # Initialize the embedding framework (this will load existing indexes)
        framework = ProductionEmbeddingFramework(
            work_dir=work_dir,
            analysis_file=analysis_file,
            checkpoint_dir=checkpoint_dir,
            max_files=args.max_files,
        )

        # Get file rankings for BM25S indexing
        if args.analysis_file.endswith(".jsonl"):
            # Load JSONL format (e.g., ranked.jsonl from PageRank calculation)
            file_rankings = []
            with open(analysis_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        file_rankings.append(json.loads(line))
            logging.info(f"📋 Loaded {len(file_rankings)} files from JSONL format")
        else:
            # Load JSON format (e.g., comprehensive_plsql_analysis.json)
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
            file_rankings = analysis_data.get("file_rankings", [])
            logging.info(f"📋 Loaded {len(file_rankings)} files from JSON format")

        if args.max_files:
            file_rankings = file_rankings[: args.max_files]

        logging.info(f"📋 Processing {len(file_rankings)} files for BM25S indexing")

        # Clear existing BM25S index to force rebuild
        bm25s_indexer = framework.bm25_indexer
        bm25s_indexer.corpus_texts = []
        bm25s_indexer.corpus_metadata = []
        bm25s_indexer.doc_mapping = {}
        bm25s_indexer.bm25_index = None

        # Process files and build BM25S index
        processed_count = 0
        for file_info in file_rankings:
            file_path = work_dir / file_info["relative_path"]

            if not file_path.exists():
                logging.debug(f"⚠️ File not found: {file_path}")
                continue

            try:
                # Read full file content
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    full_content = f.read()

                # Create a minimal ProcessingResult for BM25S indexing
                from .analysis_processor import ProcessingResult, FileMetadata

                file_metadata = FileMetadata(
                    rank=file_info["rank"],
                    file_path=str(file_path),
                    relative_path=file_info["relative_path"],
                    file_name=file_info["file_name"],
                    api_name=file_info["api_name"],
                    file_size_mb=file_info["file_size_mb"],
                    changelog_lines=file_info.get("changelog_lines", []),
                    procedure_function_names=file_info.get(
                        "procedure_function_names", []
                    ),
                )

                # Create processing result
                processing_result = ProcessingResult(
                    file_metadata=file_metadata,
                    content_excerpt=full_content[:1000] if full_content else "",
                    summary="BM25S indexing",
                    success=True,
                )

                # Add to BM25S index with full content
                bm25s_indexer.add_document(processing_result, full_content=full_content)
                processed_count += 1

                if processed_count % 100 == 0:
                    logging.info(f"📄 Processed {processed_count} files...")

            except Exception as e:
                logging.warning(f"⚠️ Failed to process {file_path}: {e}")
                continue

        logging.info(f"✅ Processed {processed_count} files")

        # Build the BM25S index
        logging.info("🔨 Building BM25S index with enhanced preprocessing...")
        success = bm25s_indexer.build_advanced_index()

        if success:
            logging.info("✅ BM25S index rebuilt successfully!")
            logging.info(
                f"📊 Index contains {len(bm25s_indexer.corpus_texts)} documents"
            )
            logging.info(
                f"🗂️ Document mapping contains {len(bm25s_indexer.doc_mapping)} entries"
            )
            return 0
        else:
            logging.error("❌ Failed to build BM25S index")
            return 1

    except ImportError as e:
        logging.error(f"❌ Import error: {e}")
        logging.error("Make sure all dependencies are installed")
        return 1
    except Exception as e:
        logging.error(f"❌ BM25S reindex failed: {e}")
        import traceback

        logging.debug(traceback.format_exc())
        return 1


def handle_pagerank_command(args) -> int:
    """Handle the PageRank calculation command."""
    try:
        from pathlib import Path
        import json
        import numpy as np
        from collections import defaultdict, Counter
        import re

        # Determine work directory and file paths using version
        work_dir = resolve_version_to_work_directory(args.version)
        # Files go in the version's extract directory
        data_dir = get_data_directory()
        safe_version = "".join(c for c in args.version if c.isalnum() or c in "._-")
        base_dir = data_dir / "versions" / safe_version
        analysis_dir = base_dir / "analysis"
        analysis_file = (
            analysis_dir / "comprehensive_plsql_analysis.json"
        )  # Fixed input filename
        output_file = base_dir / "ranked.jsonl"  # Fixed output filename
        logging.info(f"Using IFS Cloud version: {args.version}")
        logging.info(f"Work directory: {work_dir}")

        if not work_dir.exists():
            logging.error(f"❌ Work directory not found: {work_dir}")
            return 1

        if not analysis_file.exists():
            logging.error(f"❌ Analysis file not found: {analysis_file}")
            logging.error(
                f"   Run analysis first: python -m src.ifs_cloud_mcp_server.main analyze --version {args.version}"
            )
            return 1

        logging.info(f"🧮 Starting PageRank calculation...")
        logging.info(f"📁 Work directory: {work_dir}")
        logging.info(f"📊 Analysis file: {analysis_file}")
        logging.info(f"💾 Output file: {output_file}")
        logging.info(f"🎛️ Damping factor: {args.damping_factor}")
        logging.info(f"🔄 Max iterations: {args.max_iterations}")

        # Load analysis file
        with open(analysis_file, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

        file_rankings = analysis_data.get("file_rankings", [])
        logging.info(f"📋 Analyzing {len(file_rankings)} files for PageRank")

        # Build file index and dependency graph
        file_index = {}  # relative_path -> index
        files_list = []  # index -> file_info

        for i, file_info in enumerate(file_rankings):
            file_index[file_info["relative_path"]] = i
            files_list.append(file_info)

        # Create adjacency matrix for PageRank
        n_files = len(files_list)
        adjacency_matrix = np.zeros((n_files, n_files))

        logging.info("🔗 Building dependency graph...")

        # Analyze file dependencies
        dependencies_found = 0
        for i, file_info in enumerate(files_list):
            file_path = work_dir / file_info["relative_path"]

            if not file_path.exists():
                logging.debug(f"⚠️ File not found: {file_path}")
                continue

            try:
                # Read file content to find dependencies
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().upper()  # Convert to uppercase for matching

                # Extract API calls from file content
                api_calls = file_info.get("api_calls", [])

                # For each API call, find files that might provide that API
                for api_call in api_calls:
                    # Find files whose API name matches this call
                    for j, target_file in enumerate(files_list):
                        if i == j:  # Don't self-reference
                            continue

                        target_api = target_file.get("api_name", "").upper()

                        # Check if this file provides the API being called
                        if api_call.upper() == target_api:
                            adjacency_matrix[i][j] = 1  # i depends on j
                            dependencies_found += 1
                            logging.debug(
                                f"Dependency: {file_info['file_name']} -> {target_file['file_name']}"
                            )

                        # Also check if API call appears in the target file name or content
                        elif api_call.upper() in target_file["file_name"].upper():
                            adjacency_matrix[i][j] = 0.5  # Weaker dependency
                            dependencies_found += 1

            except Exception as e:
                logging.warning(
                    f"⚠️ Failed to analyze dependencies for {file_path}: {e}"
                )
                continue

        logging.info(f"🔗 Found {dependencies_found} dependencies")

        # Convert adjacency matrix to transition matrix for PageRank
        # Transpose because PageRank flows from linked-to pages to linking pages
        transition_matrix = adjacency_matrix.T

        # Normalize rows (make it stochastic)
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_files):
            if row_sums[i] > 0:
                transition_matrix[i] /= row_sums[i]
            else:
                # If no outgoing links, distribute equally to all pages
                transition_matrix[i] = 1.0 / n_files

        logging.info("🧮 Running PageRank algorithm...")

        # Initialize PageRank vector
        pagerank_vector = np.ones(n_files) / n_files

        # Run PageRank iterations
        for iteration in range(args.max_iterations):
            prev_pagerank = pagerank_vector.copy()

            # PageRank formula: PR(i) = (1-d)/N + d * sum(PR(j)/L(j)) for all j linking to i
            pagerank_vector = (
                1 - args.damping_factor
            ) / n_files + args.damping_factor * transition_matrix @ pagerank_vector

            # Check convergence
            diff = np.abs(pagerank_vector - prev_pagerank).sum()
            if diff < args.convergence_threshold:
                logging.info(
                    f"✅ PageRank converged after {iteration + 1} iterations (diff: {diff:.2e})"
                )
                break

            if iteration % 10 == 0:
                logging.debug(f"Iteration {iteration + 1}: diff = {diff:.2e}")
        else:
            logging.warning(
                f"⚠️ PageRank did not converge after {args.max_iterations} iterations"
            )

        # Create ranked results
        ranked_results = []
        for i, (file_info, pagerank_score) in enumerate(
            zip(files_list, pagerank_vector)
        ):
            ranked_result = {
                **file_info,  # Copy all existing fields
                "pagerank_score": float(pagerank_score),
                "pagerank_rank": 0,  # Will be set after sorting
            }
            ranked_results.append(ranked_result)

        # Sort by PageRank score (descending)
        ranked_results.sort(key=lambda x: x["pagerank_score"], reverse=True)

        # Assign new ranks based on PageRank scores
        for rank, result in enumerate(ranked_results, 1):
            result["pagerank_rank"] = rank

        # Save results to JSONL file
        logging.info(f"💾 Saving PageRank results to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for result in ranked_results:
                f.write(json.dumps(result) + "\n")

        # Print summary
        logging.info("📊 PageRank Summary:")
        logging.info(f"   • Total files analyzed: {len(ranked_results)}")
        logging.info(f"   • Dependencies found: {dependencies_found}")
        logging.info(f"   • Output saved to: {output_file}")

        # Show top 10 files by PageRank
        logging.info("🏆 Top 10 files by PageRank score:")
        for i, result in enumerate(ranked_results[:10], 1):
            score = result["pagerank_score"]
            name = result["file_name"]
            logging.info(f"   {i:2d}. {name} (score: {score:.6f})")

        return 0

    except ImportError as e:
        logging.error(f"❌ Import error: {e}")
        logging.error("Make sure numpy is installed: uv add numpy")
        return 1
    except Exception as e:
        logging.error(f"❌ PageRank calculation failed: {e}")
        import traceback

        logging.debug(traceback.format_exc())
        return 1


def handle_analyze_command(args) -> int:
    """Handle the analyze command to generate comprehensive file analysis."""
    try:
        from pathlib import Path
        import json
        from datetime import datetime

        # Import the ProductionEmbeddingFramework class
        from .analysis_processor import ProductionEmbeddingFramework

        logging.info("🔍 Starting comprehensive file analysis...")

        # Determine work directory and output paths using version
        work_dir = resolve_version_to_work_directory(args.version)
        # Output files go in the version's extract directory
        data_dir = get_data_directory()
        safe_version = "".join(c for c in args.version if c.isalnum() or c in "._-")
        base_dir = data_dir / "versions" / safe_version
        analysis_dir = base_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            analysis_dir / "comprehensive_plsql_analysis.json"
        )  # Fixed filename
        logging.info(f"Using IFS Cloud version: {args.version}")
        logging.info(f"Work directory: {work_dir}")

        if not work_dir.exists():
            logging.error(f"❌ Work directory not found: {work_dir}")
            return 1

        logging.info(f"📁 Output file: {output_file}")

        # Find PL/SQL files
        plsql_files = list(work_dir.rglob("*.plsql"))
        logging.info(f"📁 Found {len(plsql_files)} PL/SQL files in {work_dir}")

        if not plsql_files:
            logging.error(f"❌ No PL/SQL files found in {work_dir}")
            return 1

        # Create a temporary framework instance just for analysis
        # We don't need all the embedding features, just the file analyzer
        framework = ProductionEmbeddingFramework(
            work_dir=work_dir,
            analysis_file=output_file,  # This won't be used for loading
            checkpoint_dir=base_dir
            / "embedding_checkpoints",  # Use version-specific directory
            max_files=args.max_files,  # Pass max_files to framework
            force=args.force,  # Pass force flag to framework
        )

        # Run the metadata extraction only (no PageRank calculation)
        logging.info("📊 Extracting file metadata and building dependency graph...")
        if args.max_files:
            logging.info(f"🔢 Limiting analysis to {args.max_files} files")

        analysis_results = framework.extract_metadata_only()

        # Save the analysis results
        logging.info(f"💾 Saving analysis results to {output_file}")

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2)

        # Print summary
        files_info = analysis_results.get("file_info", [])
        file_count = len(files_info)
        metadata = analysis_results.get("analysis_metadata", {})
        stats = metadata.get("processing_stats", {})

        logging.info("📊 Analysis Summary:")
        logging.info(f"   • Total files analyzed: {file_count}")
        logging.info(f"   • API calls found: {stats.get('total_api_calls_found', 0)}")
        logging.info(f"   • Output saved to: {output_file}")

        # Show top 10 files by API call count (since we don't have PageRank yet)
        if files_info:
            # Sort by number of API calls
            files_info_sorted = sorted(
                files_info, key=lambda x: len(x.get("api_calls", [])), reverse=True
            )
            logging.info("🏆 Top 10 files by API call count:")
            for i, file_info in enumerate(files_info_sorted[:10], 1):
                name = file_info.get("file_name", "Unknown")
                api_count = len(file_info.get("api_calls", []))
                logging.info(f"   {i:2d}. {name} ({api_count} API calls)")

        return 0

    except ValueError as e:
        logging.error(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        logging.error(f"❌ Analysis failed: {e}")
        import traceback

        logging.debug(traceback.format_exc())
        return 1


def main_sync():
    """Synchronous main entry point for console scripts."""
    import argparse

    # Parse arguments first to determine which command to run
    parser = argparse.ArgumentParser(
        description="IFS Cloud MCP Server with RAG search capabilities"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import command (requires async)
    import_parser = subparsers.add_parser(
        "import",
        help="Import IFS Cloud ZIP file to version directory (version auto-detected)",
    )
    import_parser.add_argument("zip_file", help="Path to IFS Cloud ZIP file")
    import_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # List command (synchronous)
    list_parser = subparsers.add_parser(
        "list", help="List available IFS Cloud versions and their index status"
    )
    list_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # Analyze command (synchronous) - Generate comprehensive file analysis
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Generate comprehensive file analysis (creates comprehensive_plsql_analysis.json)",
    )
    analyze_parser.add_argument(
        "--version", required=True, help="IFS Cloud version to analyze (e.g., 25.1.0)"
    )
    analyze_parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to analyze (for testing)",
    )
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration by skipping existing analysis file and overwriting it",
    )
    analyze_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # Server command (synchronous - default)
    server_parser = subparsers.add_parser(
        "server", help="Start the MCP server (default command)"
    )
    server_parser.add_argument(
        "--version", required=True, help="IFS Cloud version to use (e.g., 25.1.0)"
    )
    server_parser.add_argument(
        "--name", default="ifs-cloud-mcp-server", help="Server name"
    )
    server_parser.add_argument(
        "--transport", default="stdio", help="Transport type (stdio, sse)"
    )
    server_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # PageRank calculation command (synchronous) - Calculate PageRank scores for files
    pagerank_parser = subparsers.add_parser(
        "calculate-pagerank",
        help="Calculate PageRank scores based on file interdependencies",
    )
    pagerank_parser.add_argument(
        "--version", required=True, help="IFS Cloud version to analyze (e.g., 25.1.0)"
    )
    pagerank_parser.add_argument(
        "--damping-factor",
        type=float,
        default=0.85,
        help="PageRank damping factor (default: 0.85)",
    )
    pagerank_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum PageRank iterations (default: 100)",
    )
    pagerank_parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-6,
        help="PageRank convergence threshold (default: 1e-6)",
    )
    pagerank_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # Embedding command (requires async) - Create embeddings using production framework
    embedding_parser = subparsers.add_parser(
        "embed",
        help="⚠️  RESOURCE INTENSIVE: Create embeddings (requires NVIDIA GPU). Embeddings are published to GitHub releases and downloaded at runtime - only run if you need to generate new embeddings",
    )
    embedding_parser.add_argument(
        "--version", required=True, help="IFS Cloud version to process (e.g., 25.1.0)"
    )
    embedding_parser.add_argument(
        "--model",
        default="phi4-mini:3.8b-q4_K_M",
        help="Ollama model to use (default: phi4-mini:3.8b-q4_K_M)",
    )
    embedding_parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (for testing/partial runs)",
    )
    embedding_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint, start fresh",
    )
    embedding_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # BM25S reindex command (synchronous) - Rebuild BM25S index with enhanced preprocessing
    bm25s_parser = subparsers.add_parser(
        "reindex-bm25s",
        help="Rebuild BM25S lexical search index with enhanced preprocessing",
    )
    bm25s_parser.add_argument(
        "--version", required=True, help="IFS Cloud version to reindex (e.g., 25.1.0)"
    )
    bm25s_parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (for testing)",
    )
    bm25s_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Route to appropriate handler based on command
    if getattr(args, "command", None) == "import":
        # Import command is now synchronous (just file extraction)
        setup_logging(args.log_level)
        return handle_import_command(args)
    elif getattr(args, "command", None) == "embed":
        # Embedding command requires async - lazy import to avoid CLI slowdown
        import asyncio
        from .analysis_processor import run_embedding_command

        setup_logging(args.log_level)
        return asyncio.run(run_embedding_command(args))
    elif getattr(args, "command", None) == "reindex-bm25s":
        # BM25S reindex command is synchronous
        setup_logging(args.log_level)
        return handle_bm25s_reindex_command(args)
    elif getattr(args, "command", None) == "calculate-pagerank":
        # PageRank calculation command is synchronous
        setup_logging(args.log_level)
        return handle_pagerank_command(args)
    elif getattr(args, "command", None) == "list":
        # List command is synchronous
        return handle_list_command(args)
    elif getattr(args, "command", None) == "analyze":
        # Analyze command is synchronous
        setup_logging(args.log_level)
        return handle_analyze_command(args)
    else:
        # Server command (default) is synchronous and manages its own event loop
        setup_logging(getattr(args, "log_level", "INFO"))
        return handle_server_command(args)


if __name__ == "__main__":
    sys.exit(main_sync())
