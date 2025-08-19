"""Tantivy-based indexer for IFS Cloud files with enhanced caching."""

import os
import hashlib
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter

import tantivy
import aiofiles
from pydantic import BaseModel, ConfigDict

from .parsers import IFSFileParser


logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for tracking file changes and caching."""

    path: str
    size: int
    modified_time: float
    hash: str
    indexed_at: float


class SearchResult(BaseModel):
    """Search result model."""

    path: str
    name: str
    type: str
    content_preview: str
    score: float
    entities: List[str]
    line_count: int
    complexity_score: float
    pagerank_score: float
    modified_time: datetime
    hash: str  # Unique content hash for React keys
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    entity_name: Optional[str] = None
    component: Optional[str] = None
    pages: List[str] = []
    lists: List[str] = []
    groups: List[str] = []
    entitysets: List[str] = []
    iconsets: List[str] = []
    trees: List[str] = []
    navigators: List[str] = []
    contexts: List[str] = []

    model_config = ConfigDict()


class IFSCloudIndexer:
    """High-performance Tantivy-based indexer for IFS Cloud files."""

    # Supported IFS Cloud file extensions
    SUPPORTED_EXTENSIONS = {
        ".entity",
        ".plsql",
        ".views",
        ".storage",
        ".fragment",
        ".client",
        ".projection",
        ".plsvc",
    }

    def __init__(self, index_path: Union[str, Path], create_new: bool = False):
        """Initialize the Tantivy indexer with enhanced caching.

        Args:
            index_path: Path to store the Tantivy index
            create_new: Whether to create a new index (overwriting existing)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Cache metadata file path
        self.cache_metadata_path = self.index_path / "cache_metadata.json"
        self._file_cache: Dict[str, FileMetadata] = {}
        self._load_cache_metadata()

        self._schema = self._create_schema()
        self._index = self._create_or_open_index(create_new)
        self._writer = None  # Will be created on demand
        self._parser = IFSFileParser()

        logger.info(f"Initialized Tantivy indexer at {self.index_path}")
        logger.info(f"Cache contains {len(self._file_cache)} file entries")

    def _get_writer(self):
        """Get or create a writer instance."""
        if self._writer is None:
            try:
                self._writer = self._index.writer(heap_size=50_000_000)  # 50MB heap
            except Exception as e:
                logger.error(f"Failed to acquire writer: {e}")
                raise
        return self._writer

    def _close_writer(self):
        """Close the current writer instance and clean up lock files."""
        if self._writer is not None:
            try:
                self._writer.rollback()
            except:
                pass  # Ignore rollback errors

            # Explicitly delete the writer to release lock
            del self._writer
            self._writer = None

            # Force garbage collection to ensure cleanup
            import gc

            gc.collect()

            # Longer delay to ensure lock file is fully released
            import time

            time.sleep(1.0)  # Increased from 0.5 to 1.0 for better lock release

    def _commit_writer(self):
        """Commit and close the writer safely."""
        if self._writer is not None:
            try:
                self._writer.commit()
                self._writer = None
                return True
            except Exception as e:
                logger.error(f"Error committing writer: {e}")
                self._close_writer()
                return False
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            searcher = self._index.searcher()

            # Get total document count
            total_docs = searcher.num_docs

            # Get cache stats
            cache_files = len(self._file_cache.get("files", {}))

            # Count entities by doing a wildcard search
            try:
                all_results = self.search("*", limit=10000)
                total_entities = sum(len(result.entities) for result in all_results)

                # Count by file type
                file_types = {}
                modules = set()
                logical_units = set()

                for result in all_results:
                    file_type = result.type
                    file_types[file_type] = file_types.get(file_type, 0) + 1

                    if result.module:
                        modules.add(result.module)
                    if result.logical_unit:
                        logical_units.add(result.logical_unit)

            except Exception:
                # Fallback if search fails
                total_entities = 0
                file_types = {}
                modules = set()
                logical_units = set()

            return {
                "total_files": total_docs,
                "total_entities": total_entities,
                "cached_files": cache_files,
                "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
                "file_types": file_types,
                "modules": len(modules),
                "logical_units": len(logical_units),
                "index_path": str(self.index_path),
                "cache_enabled": True,
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_files": 0,
                "total_entities": 0,
                "cached_files": 0,
                "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
                "file_types": {},
                "modules": 0,
                "logical_units": 0,
                "index_path": str(self.index_path),
                "cache_enabled": True,
                "error": str(e),
            }

    def cleanup(self):
        """Clean up resources."""
        self._close_writer()
        logger.info("Indexer resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def _create_schema(self) -> tantivy.Schema:
        """Create the Tantivy schema for IFS Cloud files."""
        schema_builder = tantivy.SchemaBuilder()

        # File metadata fields
        schema_builder.add_text_field("path", stored=True)
        schema_builder.add_text_field("name", stored=True)
        schema_builder.add_text_field("type", stored=True)
        schema_builder.add_integer_field("size", stored=True, indexed=True)
        schema_builder.add_date_field("modified_time", stored=True, indexed=True)

        # Content fields - make sure these are properly indexed for search
        schema_builder.add_text_field("content", stored=False, index_option="position")
        schema_builder.add_text_field("content_preview", stored=True)

        # IFS-specific fields - ensure they're indexed for fuzzy search
        schema_builder.add_text_field("entities", stored=True, index_option="position")
        schema_builder.add_text_field(
            "dependencies", stored=True, index_option="position"
        )
        schema_builder.add_text_field("functions", stored=True, index_option="position")
        schema_builder.add_text_field("imports", stored=True, index_option="position")

        # Enhanced IFS structure fields
        schema_builder.add_text_field("module", stored=True, index_option="position")
        schema_builder.add_text_field(
            "logical_unit", stored=True, index_option="position"
        )
        schema_builder.add_text_field(
            "entity_name", stored=True, index_option="position"
        )
        schema_builder.add_text_field("component", stored=True, index_option="position")

        # Frontend UI elements - these need to be searchable
        schema_builder.add_text_field("pages", stored=True, index_option="position")
        schema_builder.add_text_field("lists", stored=True, index_option="position")
        schema_builder.add_text_field("groups", stored=True, index_option="position")
        schema_builder.add_text_field(
            "entitysets", stored=True, index_option="position"
        )
        schema_builder.add_text_field("iconsets", stored=True, index_option="position")
        schema_builder.add_text_field("trees", stored=True, index_option="position")
        schema_builder.add_text_field(
            "navigators", stored=True, index_option="position"
        )
        schema_builder.add_text_field("contexts", stored=True, index_option="position")

        # Metrics fields
        schema_builder.add_float_field("complexity_score", stored=True, indexed=True)
        schema_builder.add_float_field("pagerank_score", stored=True, indexed=True)
        schema_builder.add_integer_field("line_count", stored=True, indexed=True)
        schema_builder.add_text_field("hash", stored=True)

        return schema_builder.build()

    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        if not self.cache_metadata_path.exists():
            self._file_cache = {}
            return

        try:
            with open(self.cache_metadata_path, "r") as f:
                data = json.load(f)
                self._file_cache = {
                    path: FileMetadata(**metadata) for path, metadata in data.items()
                }
            logger.info(f"Loaded cache metadata for {len(self._file_cache)} files")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache metadata: {e}. Starting fresh.")
            self._file_cache = {}

    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            data = {
                path: {
                    "path": metadata.path,
                    "size": metadata.size,
                    "modified_time": metadata.modified_time,
                    "hash": metadata.hash,
                    "indexed_at": metadata.indexed_at,
                }
                for path, metadata in self._file_cache.items()
            }
            with open(self.cache_metadata_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved cache metadata for {len(self._file_cache)} files")
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _is_file_cached_and_current(self, file_path: Path) -> bool:
        """Check if file is cached and hasn't changed since last indexing.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file is cached and current
        """
        if not file_path.exists():
            return False

        file_key = str(file_path)
        if file_key not in self._file_cache:
            return False

        try:
            stat = file_path.stat()
            cached = self._file_cache[file_key]

            # Check if file has changed
            return cached.size == stat.st_size and cached.modified_time == stat.st_mtime
        except Exception:
            return False

    def _update_file_cache(self, file_path: Path, file_hash: str):
        """Update cache metadata for a file.

        Args:
            file_path: Path to the file
            file_hash: Content hash of the file
        """
        try:
            stat = file_path.stat()
            self._file_cache[str(file_path)] = FileMetadata(
                path=str(file_path),
                size=stat.st_size,
                modified_time=stat.st_mtime,
                hash=file_hash,
                indexed_at=datetime.now().timestamp(),
            )
        except Exception as e:
            logger.error(f"Failed to update cache for {file_path}: {e}")

    def cleanup_cache(self) -> int:
        """Remove cache entries for files that no longer exist.

        Returns:
            Number of stale entries removed
        """
        stale_files = []

        for file_path in self._file_cache.keys():
            if not Path(file_path).exists():
                stale_files.append(file_path)

        for file_path in stale_files:
            del self._file_cache[file_path]

        if stale_files:
            self._save_cache_metadata()
            logger.info(f"Removed {len(stale_files)} stale cache entries")

        return len(stale_files)

    def _create_or_open_index(self, create_new: bool) -> tantivy.Index:
        """Create or open a Tantivy index."""
        if create_new or not (self.index_path / "meta.json").exists():
            # Clear cache when creating new index
            self._file_cache = {}
            return tantivy.Index(self._schema, path=str(self.index_path))
        else:
            return tantivy.Index.open(str(self.index_path))

    def calculate_complexity_score(self, content: str, file_type: str) -> float:
        """Calculate complexity score for a file based on its content and type.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            Complexity score (0.0 - 1.0)
        """
        if not content:
            return 0.0

        # Use the parser to get complexity indicators
        parsed = self._parser.parse(content, file_type)

        lines = content.split("\n")
        line_count = len(lines)

        # Base complexity from line count (normalized)
        line_complexity = min(line_count / 1000.0, 0.3)

        # Type-specific complexity weights
        type_weights = {
            ".plsql": 1.0,  # PL/SQL is inherently complex
            ".plsvc": 1.0,  # PL/SQL service layer
            ".entity": 0.7,  # Entity definitions are moderately complex
            ".views": 0.6,  # Views are moderately complex
            ".projection": 0.8,  # Projections can be complex
            ".client": 0.6,  # Client code moderate complexity
            ".fragment": 0.9,  # Fragments can be very complex (mixed content)
            ".storage": 0.4,  # Storage configs are less complex
        }

        type_weight = type_weights.get(file_type, 0.5)

        # Complexity from parsed indicators (normalized)
        indicator_complexity = min(parsed.complexity_indicators / 100.0, 0.7)

        # Combine all factors
        total_complexity = (line_complexity + indicator_complexity) * type_weight

        return min(total_complexity, 1.0)

    def extract_entities(self, content: str, file_type: str) -> List[str]:
        """Extract IFS entities from file content using specialized parsers.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of extracted entities
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.entities

    def extract_dependencies(self, content: str, file_type: str) -> List[str]:
        """Extract dependencies from file content.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of dependencies
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.dependencies

    def extract_functions(self, content: str, file_type: str) -> List[str]:
        """Extract functions/procedures from file content.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of functions
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.functions

    def extract_imports(self, content: str, file_type: str) -> List[str]:
        """Extract imports/includes from file content.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of imports
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.imports

    def calculate_pagerank_scores(self) -> Dict[str, float]:
        """Calculate PageRank scores for all entities in the index.

        This implements a simplified PageRank algorithm where:
        - Each entity gets votes from files that reference it
        - More referenced entities get higher scores
        - Entities that reference important entities also get boosted

        Returns:
            Dictionary mapping entity names to PageRank scores
        """
        if not self._index:
            return {}

        searcher = self._index.searcher()

        # Build entity reference graph
        entity_graph = defaultdict(set)  # entity -> set of entities that reference it
        entity_dependencies = defaultdict(
            set
        )  # entity -> set of entities it depends on
        all_entities = set()

        try:
            # Get all documents to build the graph
            query = tantivy.Query.all_query()
            search_results = searcher.search(query, limit=10000)  # Get all docs

            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)

                # Get the primary entity for this file
                entity_name = doc.get_first("entity_name")
                if entity_name:
                    all_entities.add(entity_name)

                    # Get all entities this file depends on
                    dependencies_str = doc.get_first("dependencies") or ""
                    entities_str = doc.get_first("entities") or ""

                    # Combine dependencies and entities mentioned in the file
                    dependencies = []
                    if dependencies_str:
                        dependencies.extend(dependencies_str.split())
                    if entities_str:
                        dependencies.extend(entities_str.split())

                    # Remove duplicates and self-references
                    dependencies = list(
                        set(dep for dep in dependencies if dep and dep != entity_name)
                    )

                    # Build the graph
                    for dep in dependencies:
                        if dep:
                            all_entities.add(dep)
                            entity_graph[dep].add(
                                entity_name
                            )  # dep is referenced by entity_name
                            entity_dependencies[entity_name].add(
                                dep
                            )  # entity_name depends on dep

            # Initialize PageRank scores
            num_entities = len(all_entities)
            if num_entities == 0:
                return {}

            pagerank_scores = {entity: 1.0 / num_entities for entity in all_entities}

            # PageRank parameters
            damping_factor = 0.85
            iterations = 20
            convergence_threshold = 0.001

            # Run PageRank iterations
            for iteration in range(iterations):
                new_scores = {}

                for entity in all_entities:
                    # Base score (random surfer)
                    score = (1.0 - damping_factor) / num_entities

                    # Add scores from entities that reference this entity
                    for referencing_entity in entity_graph[entity]:
                        # Get the number of entities that the referencing entity depends on
                        out_degree = len(entity_dependencies[referencing_entity])
                        if out_degree > 0:
                            score += damping_factor * (
                                pagerank_scores[referencing_entity] / out_degree
                            )
                        else:
                            # If an entity has no dependencies, distribute its score equally
                            score += damping_factor * (
                                pagerank_scores[referencing_entity] / num_entities
                            )

                    new_scores[entity] = score

                # Check for convergence
                max_change = max(
                    abs(new_scores[entity] - pagerank_scores[entity])
                    for entity in all_entities
                )

                pagerank_scores = new_scores

                if max_change < convergence_threshold:
                    logger.info(f"PageRank converged after {iteration + 1} iterations")
                    break

            # Normalize scores to 0-1 range
            if pagerank_scores:
                max_score = max(pagerank_scores.values())
                min_score = min(pagerank_scores.values())
                score_range = max_score - min_score

                if score_range > 0:
                    normalized_scores = {
                        entity: (score - min_score) / score_range
                        for entity, score in pagerank_scores.items()
                    }
                else:
                    normalized_scores = {entity: 0.5 for entity in pagerank_scores}

                # Log top entities
                top_entities = sorted(
                    normalized_scores.items(), key=lambda x: x[1], reverse=True
                )[:10]
                logger.info("Top PageRank entities:")
                for entity, score in top_entities:
                    logger.info(f"  {entity}: {score:.3f}")

                return normalized_scores

        except Exception as e:
            logger.error(f"Failed to calculate PageRank scores: {e}")

        return {}

    def update_pagerank_scores(self):
        """Update PageRank scores for all documents in the index."""
        if not self._index:
            return

        pagerank_scores = self.calculate_pagerank_scores()
        if not pagerank_scores:
            logger.warning("No PageRank scores calculated, skipping update")
            return

        # Update documents with PageRank scores
        searcher = self._index.searcher()
        writer = self._index.writer()

        try:
            query = tantivy.Query.all_query()
            search_results = searcher.search(query, limit=10000)

            updated_count = 0
            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)

                # Get entity name for this document
                entity_name = doc.get_first("entity_name")
                pagerank_score = 0.0

                if entity_name and entity_name in pagerank_scores:
                    pagerank_score = pagerank_scores[entity_name]

                # Create updated document with PageRank score
                doc_dict = {}
                for field_name in [
                    "path",
                    "name",
                    "type",
                    "content",
                    "content_preview",
                    "entities",
                    "dependencies",
                    "functions",
                    "imports",
                    "module",
                    "logical_unit",
                    "entity_name",
                    "component",
                    "pages",
                    "lists",
                    "groups",
                    "entitysets",
                    "iconsets",
                    "trees",
                    "navigators",
                    "contexts",
                    "hash",
                ]:
                    value = doc.get_first(field_name)
                    if value is not None:
                        doc_dict[field_name] = value

                # Add numeric fields
                for field_name in ["size", "line_count"]:
                    value = doc.get_first(field_name)
                    if value is not None:
                        doc_dict[field_name] = value

                # Add float fields
                complexity_score = doc.get_first("complexity_score")
                if complexity_score is not None:
                    doc_dict["complexity_score"] = complexity_score

                # Add the PageRank score
                doc_dict["pagerank_score"] = pagerank_score

                # Add date field
                modified_time = doc.get_first("modified_time")
                if modified_time is not None:
                    doc_dict["modified_time"] = modified_time

                # Delete old document and add updated one
                writer.delete_term(
                    tantivy.Term.from_field_text(
                        self._schema.get_field("path"), doc_dict["path"]
                    )
                )
                writer.add_document(tantivy.Document.from_dict(doc_dict))
                updated_count += 1

            writer.commit()
            logger.info(f"Updated PageRank scores for {updated_count} documents")

        except Exception as e:
            logger.error(f"Failed to update PageRank scores: {e}")

    async def index_file(
        self, file_path: Union[str, Path], force_reindex: bool = False
    ) -> bool:
        """Index a single file with intelligent caching.

        Args:
            file_path: Path to the file to index
            force_reindex: Force re-indexing even if file hasn't changed

        Returns:
            True if file was indexed successfully
        """
        file_path = Path(file_path)

        if not file_path.exists() or file_path.suffix not in self.SUPPORTED_EXTENSIONS:
            return False

        # Check cache first (unless forced)
        if not force_reindex and self._is_file_cached_and_current(file_path):
            logger.debug(f"File unchanged, using cache: {file_path}")
            return True

        try:
            # Read file content
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()

            # Get file metadata
            stat = file_path.stat()
            file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

            # Check if we need to remove old version first
            if str(file_path) in self._file_cache:
                # Delete old document by path
                writer = self._get_writer()
                writer.delete_documents("path", str(file_path))
                logger.debug(f"Removed old version of {file_path}")

            # Extract IFS-specific data using enhanced parsers
            file_type = file_path.suffix
            parsed = self._parser.parse(content, file_type)

            entities = parsed.entities
            dependencies = parsed.dependencies
            functions = parsed.functions
            imports = parsed.imports

            # Extract new frontend elements (for .client and .fragment files)
            pages = parsed.pages or []
            lists = parsed.lists or []
            groups = parsed.groups or []
            entitysets = parsed.entitysets or []
            iconsets = parsed.iconsets or []
            trees = parsed.trees or []
            navigators = parsed.navigators or []
            contexts = parsed.contexts or []

            # Extract module and logical unit information from file path
            module, logical_unit = self._extract_module_info(file_path)

            # Extract primary entity name and component
            entity_name, component = self._extract_entity_info(
                content, file_type, entities
            )

            # Calculate metrics
            line_count = len(content.split("\n"))
            complexity_score = self.calculate_complexity_score(content, file_type)
            content_preview = content[:500] if content else ""

            # Create document with enhanced frontend elements
            doc = {
                "path": str(file_path),
                "name": file_path.name,
                "type": file_type,
                "size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime),
                "content": content,
                "content_preview": content_preview,
                "entities": " ".join(entities),
                "dependencies": " ".join(dependencies),
                "functions": " ".join(functions),
                "imports": " ".join(imports),
                "module": module,
                "logical_unit": logical_unit,
                "entity_name": entity_name,
                "component": component,
                "pages": " ".join(pages),
                "lists": " ".join(lists),
                "groups": " ".join(groups),
                "entitysets": " ".join(entitysets),
                "iconsets": " ".join(iconsets),
                "trees": " ".join(trees),
                "navigators": " ".join(navigators),
                "contexts": " ".join(contexts),
                "complexity_score": complexity_score,
                "pagerank_score": 0.0,  # Will be calculated later
                "line_count": line_count,
                "hash": file_hash,
            }

            # Add document to index - let radical retry handle failures
            try:
                writer = self._get_writer()
                writer.add_document(tantivy.Document(**doc))
                logger.debug(f"Successfully added document for {file_path}")
                success = True

            except Exception as e:
                logger.error(f"Error adding document for {file_path}: {e}")

                # Only close writer if it's a corruption or thread error
                error_msg = str(e)
                if (
                    "killed" in error_msg
                    or "worker thread" in error_msg
                    or "thread" in error_msg.lower()
                ):
                    logger.info(
                        f"Writer corruption detected, closing writer for {file_path}"
                    )
                    self._close_writer()
                else:
                    logger.info(
                        f"Non-corruption error for {file_path}, keeping writer open"
                    )

                success = False

            if not success:
                return False

            # IMPORTANT: Don't close writer here in batch processing mode
            # The writer will be closed and committed by index_directory() after each batch

            # Update cache
            self._update_file_cache(file_path, file_hash)

            logger.debug(
                f"Indexed file: {file_path} (entities: {len(entities)}, "
                f"complexity: {complexity_score:.2f})"
            )

            return True

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False

    async def index_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        force_reindex: bool = False,
        batch_size: int = 250,  # Process files in batches to avoid memory issues
    ) -> Dict[str, int]:
        """Index all supported files in a directory with intelligent caching and batch processing.

        Args:
            directory_path: Path to directory to index
            recursive: Whether to index subdirectories
            force_reindex: Force re-indexing even if files haven't changed
            batch_size: Number of files to process in each batch

        Returns:
            Dictionary with indexing statistics
        """
        directory_path = Path(directory_path)
        stats = {"indexed": 0, "skipped": 0, "errors": 0, "cached": 0}

        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return stats

        # Find all supported files
        pattern = "**/*" if recursive else "*"
        files = []

        for extension in self.SUPPORTED_EXTENSIONS:
            files.extend(directory_path.glob(f"{pattern}{extension}"))

        logger.info(f"Found {len(files)} files to index in {directory_path}")

        # Track failed files for retry after successful batches
        failed_files = []
        retry_queue = []

        # Process files in batches to avoid memory/resource exhaustion
        batch_start = 0
        while batch_start < len(files) or retry_queue:
            # Determine current batch: either from main files or retry queue
            if batch_start < len(files):
                batch = files[batch_start : batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(files) + batch_size - 1) // batch_size
                batch_start += batch_size
                batch_type = "regular"
            elif retry_queue:
                # Process retry batch after successful regular batch
                batch = retry_queue[:batch_size]
                retry_queue = retry_queue[batch_size:]
                batch_num = f"retry-{len(failed_files)}"
                total_batches = f"(+{len(retry_queue) + len(batch)} retries)"
                batch_type = "retry"
            else:
                break

            logger.info(
                f"Processing {batch_type} batch {batch_num}/{total_batches} ({len(batch)} files)"
            )

            # Process this batch
            batch_errors = 0
            batch_failed_files = []
            batch_success = False

            for file_path in batch:
                try:
                    # Check cache first
                    if not force_reindex and self._is_file_cached_and_current(
                        file_path
                    ):
                        stats["cached"] += 1
                        continue

                    success = await self.index_file(file_path, force_reindex)
                    if success:
                        stats["indexed"] += 1
                    else:
                        stats["skipped"] += 1
                        # Track failed file for potential retry
                        batch_failed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
                    stats["errors"] += 1
                    batch_errors += 1
                    batch_failed_files.append(file_path)

                    # Only close writer on corruption errors, not all errors
                    error_msg = str(e)
                    if (
                        "killed" in error_msg
                        or "worker thread" in error_msg
                        or "thread" in error_msg.lower()
                    ):
                        logger.info(
                            f"Writer corruption detected in batch, closing writer"
                        )
                        self._close_writer()

                    # If too many errors in this batch, skip to next batch
                    if batch_errors > batch_size // 2:
                        logger.warning(
                            f"Too many errors in batch {batch_num}, skipping remaining files in batch"
                        )
                        # Add remaining files in batch to failed list
                        remaining_files = batch[batch.index(file_path) + 1 :]
                        batch_failed_files.extend(remaining_files)
                        break

            # Commit this batch if we have indexed files
            if stats["indexed"] > 0:
                try:
                    if self._commit_writer():
                        logger.info(f"Batch {batch_num} committed successfully")
                        batch_success = True
                        # Reload index to make new documents searchable
                        self._index.reload()
                        self._save_cache_metadata()
                    else:
                        logger.error(f"Failed to commit batch {batch_num}")
                        stats["errors"] += 1
                except Exception as e:
                    logger.error(f"Error committing batch {batch_num}: {e}")
                    stats["errors"] += 1
                    self._close_writer()

            # RADICAL APPROACH: If this batch succeeded and we have failed files from previous batches,
            # add ALL failed files to retry queue for processing after this successful batch
            if batch_success and failed_files and batch_type == "regular":
                retry_queue.extend(failed_files)
                logger.info(
                    f"🔄 Adding {len(failed_files)} failed files to retry queue after successful batch"
                )
                failed_files = (
                    []
                )  # Clear the failed files list since they're now in retry queue

            # Add current batch failures to the failed files list
            if batch_failed_files:
                if batch_type == "retry":
                    # If retry batch failed, put files back at end of failed list
                    failed_files.extend(batch_failed_files)
                    logger.warning(
                        f"⚠️ Retry batch failed, {len(batch_failed_files)} files remain in failed queue"
                    )
                else:
                    # Regular batch failures go to failed list
                    failed_files.extend(batch_failed_files)
                    logger.info(
                        f"📝 {len(batch_failed_files)} files from batch {batch_num} added to failed list"
                    )

        # Report final failed files
        if failed_files:
            logger.warning(
                f"⚠️ {len(failed_files)} files could not be indexed after all retry attempts"
            )
            stats["final_failures"] = len(failed_files)

        logger.info(
            f"Indexing complete: {stats['indexed']} indexed, "
            f"{stats['cached']} cached, {stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )

        # Calculate PageRank scores after indexing is complete
        if stats["indexed"] > 0:
            logger.info("Calculating PageRank scores for entities...")
            try:
                self.update_pagerank_scores()
                logger.info("PageRank calculation completed")
            except Exception as e:
                logger.error(f"PageRank calculation failed: {e}")

        return stats

    def search_deduplicated(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search the index with deduplication to avoid duplicate results from multiple search strategies.

        This method wraps the base search method and removes duplicates based on path+hash combination.
        This is needed because the search method runs multiple strategies (exact, fuzzy, prefix) that
        can return the same documents.

        Args:
            query: Search query
            limit: Maximum number of results
            file_type: Filter by file type (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)

        Returns:
            List of unique search results
        """
        # Get raw results from base search method
        results = self.search(
            query=query,
            limit=limit * 3,  # Request more results to account for deduplication
            file_type=file_type,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
        )

        # Deduplicate results by path+hash combination
        seen_keys = set()
        unique_results = []
        for result in results:
            key = f"{result.path}-{result.hash}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_results.append(result)
                # Stop when we have enough unique results
                if len(unique_results) >= limit:
                    break

        logger.debug(
            f"Search returned {len(results)} results, {len(unique_results)} unique after deduplication"
        )
        return unique_results

    def search_with_related_files(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        include_related: bool = True,
        max_related_per_lu: int = 5,
    ) -> List[SearchResult]:
        """Search the index and include related files from the same logical units.

        This method extends the standard search to include related files when a logical unit
        is found. For example, when searching for "Activity", it will include Activity.entity,
        Activity.plsql, Activity.views, etc.

        Args:
            query: Search query
            limit: Maximum number of primary results
            file_type: Filter by file type (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)
            include_related: Whether to include related files
            max_related_per_lu: Maximum related files per logical unit

        Returns:
            List of search results including related files
        """
        # Get primary search results
        primary_results = self.search_deduplicated(
            query=query,
            limit=limit,
            file_type=file_type,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
        )

        if not include_related or not primary_results:
            return primary_results

        # Collect unique logical units from primary results
        logical_units = set()
        primary_paths = set()

        for result in primary_results:
            primary_paths.add(result.path)
            if result.logical_unit:
                logical_units.add(result.logical_unit)

        # Find related files for each logical unit
        related_results = []

        for lu in logical_units:
            related_files = self.find_related_files(lu)

            # Add related files not already in primary results
            related_count = 0
            for related in related_files:
                if (
                    related.path not in primary_paths
                    and related_count < max_related_per_lu
                ):
                    # Reduce score slightly to indicate it's a related result
                    related.score *= 0.8
                    related_results.append(related)
                    related_count += 1

        # Combine and sort results
        all_results = primary_results + related_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[
            : limit * 2
        ]  # Allow more results when including related files

    def intelligent_search(
        self,
        query: str,
        limit: int = 10,
        include_related: bool = True,
        boost_business_logic: bool = True,
    ) -> List[SearchResult]:
        """Perform intelligent search that combines index search with metadata ranking.

        This is the recommended search method that:
        1. Uses both index-based search and metadata-enhanced search
        2. Automatically includes related files from the same logical unit
        3. Applies business logic boosting for authorization queries
        4. Provides comprehensive results ranked by relevance

        Args:
            query: Search query
            limit: Maximum number of results
            include_related: Include related files from same logical units
            boost_business_logic: Apply extra boosting for business logic files

        Returns:
            List of intelligently ranked search results
        """
        # Step 1: Get enhanced search results if metadata is available
        if self._enhanced_search_engine:
            enhanced_results = self.enhanced_search(
                query=query,
                limit=limit,
                include_related=include_related,
                fuzzy_threshold=85.0,
            )

            # Convert enhanced results back to SearchResult format
            results = []
            for enhanced in enhanced_results:
                # Find the corresponding file in the index to get full metadata
                index_results = self.search_deduplicated(
                    query=enhanced.file_path.split("/")[-1].split(".")[
                        0
                    ],  # filename without extension
                    limit=1,
                )

                if index_results:
                    result = index_results[0]
                    # Apply enhanced confidence as score boost
                    result.score = result.score * (enhanced.confidence / 100.0) * 2.0
                    results.append(result)
                else:
                    # Create a basic result if not found in index
                    results.append(
                        SearchResult(
                            path=enhanced.file_path,
                            name=enhanced.file_path.split("/")[-1],
                            type=f".{enhanced.content_type}",
                            content_preview=enhanced.snippet,
                            score=enhanced.confidence,
                            entities=[],
                            line_count=0,
                            complexity_score=0.0,
                            pagerank_score=0.0,
                            modified_time=datetime.now(),
                            hash="",
                            module=enhanced.module,
                            logical_unit=enhanced.logical_unit,
                        )
                    )

            return results[:limit]

        # Step 2: Fallback to index-only search with related files
        else:
            results = self.search_with_related_files(
                query=query, limit=limit, include_related=include_related
            )

            # Apply business logic boosting if requested
            if boost_business_logic:
                results = self._apply_business_logic_boosting(query, results)

            return results[:limit]

    def _apply_business_logic_boosting(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Apply additional boosting for business logic files based on query context."""
        query_lower = query.lower()

        # Detect authorization-related queries
        authorization_keywords = [
            "authorize",
            "authorization",
            "approve",
            "approval",
            "reject",
            "workflow",
            "auth",
            "permission",
            "access",
            "security",
        ]

        is_authorization_query = any(kw in query_lower for kw in authorization_keywords)

        # Apply boosting
        for result in results:
            boost_factor = 1.0

            # Boost .plsql files for authorization queries
            if is_authorization_query and result.type == ".plsql":
                boost_factor *= 2.0

                # Extra boost for large business logic files
                if result.line_count > 1000:
                    boost_factor *= 1.5
                elif result.line_count > 500:
                    boost_factor *= 1.2

            # Boost entity files for entity queries
            elif result.type == ".entity" and result.name.lower().startswith(
                query_lower
            ):
                boost_factor *= 1.5

            # Apply the boost
            result.score *= boost_factor

        # Re-sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def search(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search the index with various filters and intelligent boosting.

        Args:
            query: Search query
            limit: Maximum number of results
            file_type: Filter by file type (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)

        Returns:
            List of search results with filename matches prioritized
        """
        searcher = self._index.searcher()

        try:
            # Build boosted query that prioritizes filename and entity name matches
            boosted_queries = []

            # 1. Highest priority: Exact filename matches (boost 10x)
            # For "Activity", this matches Activity.entity, Activity.plsql, etc.
            try:
                filename_query = tantivy.Query.term_query(self._schema, "name", query)
                boosted_queries.append((tantivy.Occur.Should, filename_query))
            except Exception as e:
                logger.debug(f"Failed to create filename query: {e}")

            # 2. High priority: Entity name matches (boost 5x)
            try:
                entity_name_query = tantivy.Query.term_query(
                    self._schema, "entity_name", query
                )
                boosted_queries.append((tantivy.Occur.Should, entity_name_query))
            except Exception as e:
                logger.debug(f"Failed to create entity name query: {e}")

            # 3. Medium priority: Fuzzy filename matches (boost 3x)
            try:
                fuzzy_name_query = tantivy.Query.fuzzy_term_query(
                    self._schema, "name", query.lower(), distance=1, prefix=False
                )
                boosted_queries.append((tantivy.Occur.Should, fuzzy_name_query))
            except Exception as e:
                logger.debug(f"Failed to create fuzzy filename query: {e}")

            # 4. Standard priority: Content and other field searches
            default_fields = [
                "content",
                "entities",
                "functions",
                "module",
                "logical_unit",
                "pages",
                "lists",
                "groups",
                "iconsets",
                "trees",
                "navigators",
                "contexts",
            ]

            # Create standard search across content fields
            try:
                content_query = self._index.parse_query(
                    query, default_field_names=default_fields
                )
                boosted_queries.append((tantivy.Occur.Should, content_query))
            except Exception as e:
                logger.debug(f"Failed to create content query: {e}")

            # 5. Fuzzy search for content fields (lower priority)
            if len(query.strip()) >= 3:
                try:
                    lowercase_query = query.lower()
                    fuzzy_queries = []
                    for field_name in default_fields:
                        try:
                            fuzzy_query = tantivy.Query.fuzzy_term_query(
                                self._schema,
                                field_name,
                                lowercase_query,
                                distance=2,
                                prefix=False,
                            )
                            fuzzy_queries.append((tantivy.Occur.Should, fuzzy_query))
                        except Exception as e:
                            logger.debug(
                                f"Failed to create fuzzy query for field {field_name}: {e}"
                            )
                            continue

                    if fuzzy_queries:
                        combined_fuzzy_query = tantivy.Query.boolean_query(
                            fuzzy_queries
                        )
                        boosted_queries.append(
                            (tantivy.Occur.Should, combined_fuzzy_query)
                        )
                except Exception as e:
                    logger.debug(f"Fuzzy search failed: {e}")

            # Combine all queries into a single boolean query
            if boosted_queries:
                final_query = tantivy.Query.boolean_query(boosted_queries)
                search_results = searcher.search(final_query, limit=limit)
            else:
                # Fallback to simple search if all boosted queries failed
                fallback_query = self._index.parse_query(
                    query, default_field_names=default_fields
                )
                search_results = searcher.search(fallback_query, limit=limit)

            # Convert to SearchResult objects with intelligent score boosting
            results = []

            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)

                # Apply additional scoring boost based on match type
                final_score = score
                filename = doc.get_first("name") or ""
                entity_name = doc.get_first("entity_name") or ""
                pagerank_score = doc.get_first("pagerank_score") or 0.0

                # Boost exact filename matches even higher
                file_type = doc.get_first("type") or ""

                if filename.lower().startswith(query.lower()):
                    final_score *= 10.0  # 10x boost for filename prefix match
                elif query.lower() in filename.lower():
                    final_score *= 5.0  # 5x boost for filename contains match
                elif entity_name and entity_name.lower() == query.lower():
                    final_score *= 3.0  # 3x boost for exact entity name match
                elif entity_name and query.lower() in entity_name.lower():
                    final_score *= 2.0  # 2x boost for entity name contains match

                # Special boost for .entity files when searching for entity names
                # Entity files are the core definition and should rank highest
                if file_type == ".entity" and (
                    query.lower() in filename.lower()
                    or (entity_name and query.lower() in entity_name.lower())
                ):
                    final_score *= 3.0  # Additional 3x boost for entity files

                # Additional file type prioritization for entity-related searches
                elif file_type == ".plsql" and query.lower() in filename.lower():
                    final_score *= 1.5  # Moderate boost for API files
                elif (
                    file_type in [".views", ".client", ".projection"]
                    and query.lower() in filename.lower()
                ):
                    final_score *= 1.3  # Small boost for UI/view files
                # .storage files get no additional boost (they should rank lower)

                # Apply PageRank boost (multiply by 1 + pagerank_score)
                # This gives entities with higher PageRank a 0-100% boost
                final_score *= 1.0 + pagerank_score

                search_result = SearchResult(
                    path=doc.get_first("path") or "",
                    name=filename,
                    type=doc.get_first("type") or "",
                    content_preview=doc.get_first("content_preview") or "",
                    score=final_score,
                    entities=(
                        doc.get_first("entities").split()
                        if doc.get_first("entities")
                        else []
                    ),
                    line_count=doc.get_first("line_count") or 0,
                    complexity_score=doc.get_first("complexity_score") or 0.0,
                    pagerank_score=doc.get_first("pagerank_score") or 0.0,
                    modified_time=datetime.fromisoformat(
                        str(doc.get_first("modified_time"))
                        if doc.get_first("modified_time")
                        else "1970-01-01T00:00:00"
                    ),
                    hash=doc.get_first("hash") or "",
                    module=doc.get_first("module") or None,
                    logical_unit=doc.get_first("logical_unit") or None,
                    entity_name=entity_name or None,
                    component=doc.get_first("component") or None,
                    pages=(
                        doc.get_first("pages").split() if doc.get_first("pages") else []
                    ),
                    lists=(
                        doc.get_first("lists").split() if doc.get_first("lists") else []
                    ),
                    groups=(
                        doc.get_first("groups").split()
                        if doc.get_first("groups")
                        else []
                    ),
                    entitysets=(
                        doc.get_first("entitysets").split()
                        if doc.get_first("entitysets")
                        else []
                    ),
                    iconsets=(
                        doc.get_first("iconsets").split()
                        if doc.get_first("iconsets")
                        else []
                    ),
                    trees=(
                        doc.get_first("trees").split() if doc.get_first("trees") else []
                    ),
                    navigators=(
                        doc.get_first("navigators").split()
                        if doc.get_first("navigators")
                        else []
                    ),
                    contexts=(
                        doc.get_first("contexts").split()
                        if doc.get_first("contexts")
                        else []
                    ),
                    dependencies=(
                        doc.get_first("dependencies").split()
                        if doc.get_first("dependencies")
                        else []
                    ),
                    functions=(
                        doc.get_first("functions").split()
                        if doc.get_first("functions")
                        else []
                    ),
                    imports=(
                        doc.get_first("imports").split()
                        if doc.get_first("imports")
                        else []
                    ),
                )

                # Apply filters
                if file_type and not search_result.type.endswith(file_type):
                    continue

                if (
                    min_complexity is not None
                    and search_result.complexity_score < min_complexity
                ):
                    continue

                if (
                    max_complexity is not None
                    and search_result.complexity_score > max_complexity
                ):
                    continue

                results.append(search_result)

            # Sort by final score (after our boosting) and return
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def find_similar_files(
        self, file_path: Union[str, Path], limit: int = 5
    ) -> List[SearchResult]:
        """Find files similar to the given file based on entities and content.

        Args:
            file_path: Path to the reference file
            limit: Maximum number of similar files to return

        Returns:
            List of similar files
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        try:
            # Read the reference file
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract entities and create similarity query
            entities = self.extract_entities(content, file_path.suffix)

            if not entities:
                return []

            # Create query from entities
            entity_query = " OR ".join(entities[:10])  # Use top 10 entities

            return self.search(entity_query, limit=limit + 1)  # +1 to exclude self

        except Exception as e:
            logger.error(f"Error finding similar files for {file_path}: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics including cache information."""
        searcher = self._index.searcher()

        cache_size = 0
        if self.cache_metadata_path.exists():
            cache_size = self.cache_metadata_path.stat().st_size

        return {
            "total_documents": searcher.num_docs,
            "index_size": sum(
                f.stat().st_size for f in self.index_path.glob("*") if f.is_file()
            ),
            "cache_size": cache_size,
            "cached_files": len(self._file_cache),
            "index_path": str(self.index_path),
            "cache_metadata_path": str(self.cache_metadata_path),
            "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
        }

    def _extract_module_info(self, file_path: Path) -> Tuple[str, str]:
        """Extract module and logical unit information from file path.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (module, logical_unit)

        Example:
            _work/proj/model/proj/Activity.entity -> ("proj", "Activity")
            _work/career/model/career/EmployeeActivity.entity -> ("career", "EmployeeActivity")
        """
        parts = file_path.parts

        # Find _work in the path
        try:
            work_index = parts.index("_work")
            if work_index + 1 < len(parts):
                module = parts[work_index + 1]  # e.g., "proj", "career"

                # Extract entity name from filename (remove extension)
                entity_name = file_path.stem  # e.g., "Activity", "EmployeeActivity"

                return module, entity_name
        except (ValueError, IndexError):
            pass

        # Fallback if _work structure not found
        return "unknown", file_path.stem

    def _extract_entity_info(
        self, content: str, file_type: str, entities: List[str]
    ) -> Tuple[str, str]:
        """Extract primary entity name and component from file content.

        Args:
            content: File content
            file_type: File extension
            entities: List of entities found by parser

        Returns:
            Tuple of (primary_entity_name, component)
        """
        primary_entity = ""
        component = ""

        if file_type == ".entity" and content:
            try:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(content)

                # Extract primary entity name
                name_elem = root.find(".//{urn:ifsworld-com:schemas:entity_entity}NAME")
                if name_elem is not None:
                    primary_entity = name_elem.text or ""

                # Extract component
                component_elem = root.find(
                    ".//{urn:ifsworld-com:schemas:entity_entity}COMPONENT"
                )
                if component_elem is not None:
                    component = component_elem.text or ""

            except ET.ParseError:
                # Fallback to using the parsed entities
                if entities:
                    primary_entity = entities[0]
        else:
            # For non-entity files, use the first parsed entity
            if entities:
                primary_entity = entities[0]

        return primary_entity, component

    def close(self):
        """Close the indexer and release resources."""
        try:
            # Commit any pending changes and close writer
            self._commit_writer()
            self._close_writer()
            # Save cache metadata before closing
            self._save_cache_metadata()
            logger.info("Indexer closed successfully")
        except Exception as e:
            logger.error(f"Error closing indexer: {e}")
