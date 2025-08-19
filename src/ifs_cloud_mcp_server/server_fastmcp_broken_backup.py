"""IFS Cloud MCP Server implementation using FastMCP with production database metadata extraction."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from fastmcp import FastMCP

from .indexer import IFSCloudIndexer, SearchResult
from .search_engine import IFSCloudSearchEngine
from .config import ConfigManager
from .plsql_analyzer import ConservativePLSQLAnalyzer
from .client_analyzer import ConservativeClientAnalyzer
from .projection_analyzer import ProjectionAnalyzer
from .fragment_analyzer import ConservativeFragmentAnalyzer


logger = logging.getLogger(__name__)


class IFSCloudMCPServer:
    """MCP Server for IFS Cloud with Tantivy search integration, production database metadata extraction, and Navigator GUI mappings using FastMCP."""

    def __init__(
        self, index_path: Union[str, Path], name: str = "ifs-cloud-mcp-server"
    ):
        """Initialize the MCP server.

        Args:
            index_path: Path to store the Tantivy index
            name: Server name
        """
        self.name = name
        self.mcp = FastMCP(name)
        self.indexer = IFSCloudIndexer(index_path)
        self.search_engine = IFSCloudSearchEngine(self.indexer)
        self.config_manager = ConfigManager()

        # Initialize analyzers for comprehensive code understanding
        self.plsql_analyzer = ConservativePLSQLAnalyzer(strict_mode=False)
        self.client_analyzer = ConservativeClientAnalyzer()
        self.projection_analyzer = ProjectionAnalyzer(strict_mode=False)
        self.fragment_analyzer = ConservativeFragmentAnalyzer()

        # Register tools
        self._register_tools()

        logger.info(f"Initialized IFS Cloud MCP Server: {name}")

    def _register_tools(self):
        """Register MCP tools for IFS Cloud development guidance."""

        @self.mcp.tool()
        async def get_ifs_development_guidance() -> str:
            """Get comprehensive guidance for IFS Cloud development and database analysis.

            **AGENT INSTRUCTIONS:**
            Primary search tool for finding IFS Cloud code, patterns, and examples.

            **When to use:**
            - Finding existing implementations of features
            - Searching for specific APIs or methods
            - Locating examples of business logic
            - Discovering patterns and naming conventions

            **Search strategies:**
            - Entity search: query="CustomerOrder" (finds all CustomerOrder-related files)
            - Function search: query="Calculate_Price" (finds pricing implementations)
            - Pattern search: query="validation business rule" (finds validation patterns)
            - API search: query="Get_Info" (finds info retrieval methods)

            **File type filters:**
            - .entity: Entity definitions (data models)
            - .plsql: PL/SQL business logic and APIs
            - .client: Frontend client definitions
            - .projection: Business projections and views
            - .fragment: UI fragments and components
            - .views: Database views
            - .storage: Storage configurations

            **Complexity filtering:**
            - min_complexity=0.0: Include simple files
            - max_complexity=0.5: Exclude very complex files
            - Use for finding simple examples or complex implementations
            - To search by complexity only: search_content("*", min_complexity=0.3, max_complexity=0.7)

            **Examples:**
            - Find order logic: search_content("CustomerOrder pricing", file_type=".plsql")
            - Simple entities: search_content("Product", file_type=".entity", max_complexity=0.3)
            - Validation patterns: search_content("Check_Insert___", limit=20)
            - Complex files only: search_content("*", min_complexity=0.8, limit=5)
            - Find by complexity: search_content("*", min_complexity=0.2, max_complexity=0.6)

            **Output:** Ranked search results with content previews and file information

            Args:
                query: Search query (supports full-text search)
                limit: Maximum number of results (default: 10)
                file_type: Filter by file type (.entity, .plsql, .views, etc.)
                min_complexity: Minimum complexity score (0.0-1.0)
                max_complexity: Maximum complexity score (0.0-1.0)
            """
            results = self.search_engine.search(
                query=query,
                limit=limit,
                file_type=file_type,
                min_complexity=min_complexity,
                max_complexity=max_complexity,
            )

            if not results:
                return f"No results found for query: '{query}'"

            return self._format_search_results(
                results, f"Content search results for '{query}'"
            )

        @self.mcp.tool()
        async def intelligent_search(
            query: str,
            limit: int = 10,
            include_related: bool = True,
            boost_business_logic: bool = True,
        ) -> str:
            """Perform intelligent search with metadata enhancement and related files.

            **AGENT INSTRUCTIONS:**
            Advanced search tool that combines index search with IFS Cloud metadata for optimal results.

            **Key Features:**
            - **Metadata-Enhanced Ranking**: Uses IFS business context for better relevance
            - **Related File Discovery**: Automatically includes related files (e.g., Activity.entity + Activity.plsql + Activity.views)
            - **Business Logic Boosting**: Prioritizes .plsql files for authorization/business rule queries
            - **Cross-Module Understanding**: Leverages IFS module relationships

            **When to use (PREFERRED over search_content):**
            - Finding core business logic for entities (e.g., "CustomerOrder authorization")
            - Discovering complete logical unit implementations
            - Authorization and approval workflow searches
            - Business rule and validation searches
            - When you need comprehensive results for an IFS concept

            **Examples:**
            - "expense authorization" → Gets ExpenseHeader.plsql + related files
            - "Activity" → Gets Activity.entity, Activity.plsql, Activity.views, etc.
            - "customer order approval" → Prioritizes business logic files
            - "purchase requisition workflow" → Finds workflow-related files

            **Advantages over basic search:**
            - Understands IFS business terminology
            - Groups related files by logical unit
            - Ranks by business importance, not just text matching
            - Includes files you might miss with basic keyword search

            Args:
                query: Search query (business terminology works best)
                limit: Maximum number of results (default: 10)
                include_related: Include related files from same logical units (default: True)
                boost_business_logic: Apply extra boosting for business logic files (default: True)
            """
            results = self.search_engine.search(
                query=query,
                limit=limit,
                include_related=include_related,
            )

            if not results:
                return f"No results found for intelligent search query: '{query}'"

            # Add metadata context if available
            context_info = ""
            if self.search_engine.has_metadata_enhancement():
                context_info = f"\n[Enhanced with IFS {self.search_engine.get_current_ifs_version()} metadata]"

                # Add related search suggestions
                suggestions = self.search_engine.suggest_related_searches(
                    query, limit=3
                )
                if suggestions:
                    context_info += f"\nRelated suggestions: {', '.join(suggestions)}"

            return self._format_search_results(
                results, f"Intelligent search results for '{query}'{context_info}"
            )

        @self.mcp.tool()
        async def set_ifs_version(version: str) -> str:
            """Set IFS version for metadata-enhanced search.

            **AGENT INSTRUCTIONS:**
            Configure the server to use specific IFS Cloud version metadata for enhanced search results.

            **Usage:**
            - Call this before using intelligent_search for best results
            - Required for metadata-enhanced features
            - Available versions: 25.1.0 and others (use get_available_ifs_versions to check)

            Args:
                version: IFS Cloud version (e.g., "25.1.0")
            """
            success = self.search_engine.set_ifs_version(version)

            if success:
                return f"Successfully set IFS version to {version}. Enhanced search features are now available."
            else:
                available = self.search_engine.get_available_ifs_versions()
                return f"Failed to set IFS version {version}. Available versions: {available}"

        @self.mcp.tool()
        async def get_available_ifs_versions() -> str:
            """Get list of available IFS versions with metadata.

            Returns:
                List of available IFS versions that can be used with enhanced search
            """
            versions = self.search_engine.get_available_ifs_versions()
            current = self.search_engine.get_current_ifs_version()

            result = f"Available IFS versions: {versions}\n"
            if current:
                result += f"Current version: {current}"
            else:
                result += "No version currently set. Use set_ifs_version() to enable enhanced features."

            return result

        @self.mcp.tool()
        async def find_related_files(logical_unit: str, limit: int = 10) -> str:
            """Find all files related to a specific logical unit.

            **AGENT INSTRUCTIONS:**
            Find all core files for an IFS logical unit (entity, plsql, views, client, etc.)

            **When to use:**
            - Understanding complete implementation of an entity
            - Finding all customization points for a logical unit
            - Discovering the full file structure for an IFS concept

            **Examples:**
            - find_related_files("CustomerOrder") → All CustomerOrder files
            - find_related_files("Activity") → Activity.entity, Activity.plsql, etc.

            Args:
                logical_unit: Name of the IFS logical unit (e.g., "Activity", "CustomerOrder")
                limit: Maximum number of related files to return
            """
            results = self.search_engine.find_related_files(logical_unit)

            if not results:
                return f"No files found for logical unit: '{logical_unit}'"

            return self._format_search_results(
                results[:limit], f"Files related to logical unit '{logical_unit}'"
            )

        @self.mcp.tool()
        async def get_module_statistics() -> str:
            """Get statistics about available IFS modules from metadata.

            Returns:
                Statistics about IFS modules, logical units, and metadata status
            """
            stats = self.search_engine.get_module_statistics()

            if "error" in stats:
                return f"Module statistics not available: {stats['error']}"

            result = "IFS Module Statistics:\n\n"

            if isinstance(stats, dict):
                for module, info in sorted(stats.items()):
                    if isinstance(info, dict):
                        result += (
                            f"**{module}**: {info.get('lu_count', 0)} logical units\n"
                        )
                        if "sample_entities" in info:
                            result += f"  Sample entities: {', '.join(info['sample_entities'][:3])}\n"
                        result += "\n"

            return result

        @self.mcp.tool()
        async def search_entities(entity: str, limit: int = 10) -> str:
            """Search for files containing specific IFS entities with related files.

            Args:
                entity: Entity name to search for
                limit: Maximum number of results (default: 10)
            """
            # Use intelligent search to find entities and related files
            results = self.search_engine.search(
                query=entity, limit=limit, include_related=True
            )

            if not results:
                return f"No files found containing entity: '{entity}'"

            return self._format_search_results(
                results, f"Files containing entity '{entity}' (with related files)"
            )

        @self.mcp.tool()
        async def search_by_module(
            query: str,
            module: Optional[str] = None,
            logical_unit: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            """Search IFS Cloud files with module and logical unit filtering.

            Args:
                query: Search query (supports full-text search)
                module: Filter by IFS module (e.g., "proj", "career", "finance")
                logical_unit: Filter by logical unit (entity name from filename)
                limit: Maximum number of results (default: 10)
            """
            # Enhance the query with module and logical unit filters
            enhanced_query = query
            if module:
                enhanced_query += f" module:{module}"
            if logical_unit:
                enhanced_query += f" logical_unit:{logical_unit}"

            results = self.search_engine.search(query=enhanced_query, limit=limit)

            if not results:
                filter_desc = ""
                if module and logical_unit:
                    filter_desc = (
                        f" in module '{module}' with logical unit '{logical_unit}'"
                    )
                elif module:
                    filter_desc = f" in module '{module}'"
                elif logical_unit:
                    filter_desc = f" with logical unit '{logical_unit}'"
                return f"No results found for '{query}'{filter_desc}"

            # Sort results to prioritize exact module and logical unit matches
            def sort_key(result):
                score = result.score
                # Boost score for exact module match
                if module and result.module == module:
                    score += 1.0
                # Boost score for exact logical unit match
                if logical_unit and result.logical_unit == logical_unit:
                    score += 1.0
                return -score  # Negative for descending order

            sorted_results = sorted(results, key=sort_key)

            title_parts = [f"Module-aware search results for '{query}'"]
            if module:
                title_parts.append(f"Module: {module}")
            if logical_unit:
                title_parts.append(f"Logical Unit: {logical_unit}")

            return self._format_search_results(sorted_results, " | ".join(title_parts))

        @self.mcp.tool()
        async def find_similar_files(file_path: str, limit: int = 5) -> str:
            """Find files similar to a given file based on entities and content.

            Args:
                file_path: Path to the reference file
                limit: Maximum number of similar files (default: 5)
            """
            results = self.indexer.find_similar_files(file_path, limit)

            if not results:
                return f"No similar files found for: {file_path}"

            # Filter out the original file if it appears in results
            filtered_results = [r for r in results if r.path != str(file_path)][:limit]

            return self._format_search_results(
                filtered_results, f"Files similar to '{file_path}'"
            )

        @self.mcp.tool()
        async def index_directory(path: str, recursive: bool = True) -> str:
            """Index all IFS Cloud files in a directory.

            Args:
                path: Directory path to index
                recursive: Index subdirectories recursively (default: true)
            """
            stats = await self.indexer.index_directory(path, recursive)

            response_lines = [
                f"Directory indexing completed for: {path}",
                "",
                "Statistics:",
                f"  • Files indexed: {stats['indexed']}",
                f"  • Files cached: {stats['cached']}",
                f"  • Files skipped: {stats['skipped']}",
                f"  • Errors: {stats['errors']}",
                f"  • Recursive: {recursive}",
                "",
                f"Index now contains {self.indexer.get_statistics()['total_documents']} total documents.",
            ]

            if stats["cached"] > 0:
                response_lines.extend(
                    [
                        "",
                        f"⚡ Performance: {stats['cached']} files were already cached and up-to-date!",
                        "   This significantly improved indexing speed.",
                    ]
                )

            return "\n".join(response_lines)

        @self.mcp.tool()
        async def index_file(file_path: str) -> str:
            """Index a single IFS Cloud file.

            Args:
                file_path: Path to the file to index
            """
            success = await self.indexer.index_file(file_path)

            if success:
                return f"Successfully indexed file: {file_path}"
            else:
                return f"Failed to index file: {file_path} (unsupported type or error)"

        @self.mcp.tool()
        async def get_index_statistics() -> str:
            """Get statistics about the search index."""
            stats = self.indexer.get_statistics()

            return f"""Search Index Statistics:

Total Documents: {stats['total_documents']}
Index Size: {stats['index_size']:,} bytes ({stats['index_size'] / 1024 / 1024:.1f} MB)
Index Path: {stats['index_path']}

Supported File Types:
{chr(10).join(f"  - {ext}" for ext in sorted(stats['supported_extensions']))}"""

        @self.mcp.tool()
        async def fuzzy_search(query: str, limit: int = 10) -> str:
            """Perform fuzzy search to handle typos and partial matches.

            Args:
                query: Search query (fuzzy matching enabled)
                limit: Maximum number of results (default: 10)
            """
            # Add fuzzy search operators to the query
            fuzzy_query = f"{query}~"  # Tantivy fuzzy search syntax

            results = self.search_engine.search(query=fuzzy_query, limit=limit)

            if not results:
                return f"No results found for fuzzy query: '{query}'"

            return self._format_search_results(
                results, f"Fuzzy search results for '{query}'"
            )

        @self.mcp.tool()
        async def set_core_codes_path(path: str) -> str:
            """Set the path to IFS Cloud Core Codes directory.

            Args:
                path: Path to the IFS Cloud Core Codes directory
            """
            if self.config_manager.set_core_codes_path(path):
                return f"Successfully set IFS Cloud Core Codes path to: {path}"
            else:
                return f"Failed to set path. Please ensure the directory exists: {path}"

        @self.mcp.tool()
        async def get_core_codes_path() -> str:
            """Get the currently configured IFS Cloud Core Codes path."""
            path = self.config_manager.get_core_codes_path()

            if path:
                return f"IFS Cloud Core Codes path: {path}"
            else:
                return "No IFS Cloud Core Codes path configured. Use 'set_core_codes_path' tool to configure it."

        @self.mcp.tool()
        async def index_core_codes(recursive: bool = True) -> str:
            """Index the configured IFS Cloud Core Codes directory.

            Args:
                recursive: Index subdirectories recursively (default: true)
            """
            core_path = self.config_manager.get_core_codes_path()
            if not core_path:
                return "No IFS Cloud Core Codes path configured. Use 'set_core_codes_path' tool first."

            try:
                stats = await self.indexer.index_directory(
                    core_path, recursive=recursive
                )

                # Update last indexed timestamp
                self.config_manager.set_last_indexed(datetime.now().isoformat())

                response_lines = [
                    f"Successfully indexed IFS Cloud Core Codes from: {core_path}",
                    "",
                    "Statistics:",
                    f"  • Files indexed: {stats['indexed']}",
                    f"  • Files cached: {stats['cached']}",
                    f"  • Files skipped: {stats['skipped']}",
                    f"  • Errors: {stats['errors']}",
                ]

                return "\n".join(response_lines)

            except Exception as e:
                logger.error(f"Error indexing core codes: {e}")
                return f"Error indexing core codes: {str(e)}"

        @self.mcp.tool()
        async def analyze_entity_dependencies(entity_name: str) -> str:
            """Analyze dependencies for a specific entity across all files.

            Args:
                entity_name: Name of the entity to analyze
            """
            # Search for files that contain this entity
            entity_results = self.search_engine.search(
                query=f"entities:{entity_name}", limit=50
            )

            # Search for files that depend on this entity
            dependency_results = self.search_engine.search(
                query=f"dependencies:{entity_name}", limit=50
            )

            response_lines = [
                f"Dependency Analysis for Entity: {entity_name}",
                "=" * 50,
                "",
            ]

            if entity_results:
                response_lines.extend(
                    [
                        f"Files defining/using this entity ({len(entity_results)} found):",
                        "-" * 40,
                    ]
                )
                for result in entity_results[:10]:  # Limit to first 10 for readability
                    response_lines.append(
                        f"  • {result.path} ({result.type}) - Score: {result.score:.2f}"
                    )

                if len(entity_results) > 10:
                    response_lines.append(
                        f"  ... and {len(entity_results) - 10} more files"
                    )
            else:
                response_lines.append(
                    f"No files found defining/using entity: {entity_name}"
                )

            response_lines.append("")

            if dependency_results:
                response_lines.extend(
                    [
                        f"Files depending on this entity ({len(dependency_results)} found):",
                        "-" * 40,
                    ]
                )
                for result in dependency_results[:10]:
                    response_lines.append(
                        f"  • {result.path} ({result.type}) - Score: {result.score:.2f}"
                    )

                if len(dependency_results) > 10:
                    response_lines.append(
                        f"  ... and {len(dependency_results) - 10} more files"
                    )
            else:
                response_lines.append(
                    f"No files found depending on entity: {entity_name}"
                )

            return "\n".join(response_lines)

        @self.mcp.tool()
        async def find_overrides_and_overtakes(
            entity_name: Optional[str] = None,
        ) -> str:
            """Find all @Override and @Overtake annotations in the codebase.

            Args:
                entity_name: Optional: filter by specific entity name
            """
            # Search for @Override and @Overtake patterns
            query = "@Override OR @Overtake"
            if entity_name:
                query += f" AND entities:{entity_name}"

            results = self.search_engine.search(query=query, limit=50)

            response_lines = ["Override and Overtake Analysis", "=" * 40, ""]

            if entity_name:
                response_lines[0] += f" for Entity: {entity_name}"

            if not results:
                search_scope = f" for entity '{entity_name}'" if entity_name else ""
                response_lines.append(
                    f"No @Override or @Overtake annotations found{search_scope}"
                )
            else:
                response_lines.extend(
                    [
                        f"Found {len(results)} files with override/overtake annotations:",
                        "-" * 50,
                    ]
                )

                for result in results:
                    response_lines.extend(
                        [
                            f"File: {result.path} ({result.type})",
                            f"  Entities: {', '.join(result.entities) if result.entities else 'None'}",
                            f"  Preview: {result.content_preview[:100]}...",
                            "",
                        ]
                    )

            return "\n".join(response_lines)

        @self.mcp.tool()
        async def force_reindex_directory(path: str, recursive: bool = True) -> str:
            """Force re-index all files in a directory, ignoring cache.

            Args:
                path: Directory path to force re-index
                recursive: Re-index subdirectories recursively (default: true)
            """
            try:
                directory_path = Path(path)
                if not directory_path.exists():
                    return f"Error: Directory not found: {path}"

                stats = await self.indexer.index_directory(
                    directory_path, recursive=recursive, force_reindex=True
                )

                response_lines = [
                    "Force Re-indexing Complete",
                    "=" * 30,
                    "",
                    f"Directory: {path}",
                    f"Recursive: {recursive}",
                    "",
                    "Results:",
                    f"  • Files indexed: {stats['indexed']}",
                    f"  • Files cached: {stats['cached']}",
                    f"  • Files skipped: {stats['skipped']}",
                    f"  • Errors: {stats['errors']}",
                    "",
                    f"Total processed: {stats['indexed'] + stats['cached'] + stats['skipped']}",
                ]

                return "\n".join(response_lines)

            except Exception as e:
                return f"Error during force re-indexing: {str(e)}"

        @self.mcp.tool()
        async def cleanup_cache() -> str:
            """Remove stale cache entries for files that no longer exist."""
            try:
                removed_count = self.indexer.cleanup_cache()

                response_lines = [
                    "Cache Cleanup Complete",
                    "=" * 25,
                    "",
                    f"Stale entries removed: {removed_count}",
                ]

                if removed_count > 0:
                    response_lines.extend(
                        [
                            "",
                            "Cache has been cleaned of entries for files that no longer exist.",
                            "This helps improve performance and accuracy.",
                        ]
                    )
                else:
                    response_lines.extend(
                        ["", "No stale entries found. Cache is clean!"]
                    )

                return "\n".join(response_lines)

            except Exception as e:
                return f"Error during cache cleanup: {str(e)}"

        @self.mcp.tool()
        async def get_cache_statistics() -> str:
            """Get detailed cache and index statistics."""
            try:
                stats = self.indexer.get_statistics()

                # Format file sizes
                def format_size(size_bytes):
                    for unit in ["B", "KB", "MB", "GB"]:
                        if size_bytes < 1024:
                            return f"{size_bytes:.1f} {unit}"
                        size_bytes /= 1024
                    return f"{size_bytes:.1f} TB"

                response_lines = [
                    "Index & Cache Statistics",
                    "=" * 30,
                    "",
                    "INDEX STATUS:",
                    f"  • Total documents: {stats['total_documents']:,}",
                    f"  • Index size: {format_size(stats['index_size'])}",
                    f"  • Index path: {stats['index_path']}",
                    "",
                    "CACHE STATUS:",
                    f"  • Cached files: {stats['cached_files']:,}",
                    f"  • Cache size: {format_size(stats['cache_size'])}",
                    f"  • Cache metadata: {stats['cache_metadata_path']}",
                    "",
                    "SUPPORTED FILE TYPES:",
                    f"  • {', '.join(stats['supported_extensions'])}",
                    "",
                    "PERFORMANCE BENEFITS:",
                    f"  • Cache hit ratio improves indexing speed significantly",
                    f"  • Only changed files are re-processed",
                    f"  • Metadata tracking prevents unnecessary work",
                ]

                return "\n".join(response_lines)

            except Exception as e:
                return f"Error getting cache statistics: {str(e)}"

        # ==== IFS CLOUD DEVELOPMENT ASSISTANCE TOOLS ====

        @self.mcp.tool()
        async def generate_entity_template(
            entity_name: str,
            module: str,
            logical_unit: Optional[str] = None,
            include_states: bool = False,
            include_associations: bool = False,
        ) -> str:
            """Generate a complete IFS Cloud entity template with proper XML structure.

            **AGENT INSTRUCTIONS:**
            This tool generates IFS Cloud entity XML files that match production standards.

            **When to use:**
            - Creating new business entities (Customer, Order, Product, etc.)
            - Setting up data models for IFS Cloud modules
            - Generating entity scaffolding for development

            **Parameter guidance:**
            - entity_name: Use PascalCase (CustomerOrder, ProductItem, WorkOrder)
            - module: IFS module name in UPPERCASE (ORDER, INVENTORY, MANUFACT)
            - include_states: Only set to True for entities that have lifecycle progression
              (orders: planned→active→completed, workflows, approvals)
            - include_associations: Set to True when entity relates to other entities

            **Examples:**
            - Simple reference data: generate_entity_template("ProductCategory", "INVENTORY")
            - Order entity: generate_entity_template("CustomerOrder", "ORDER", include_states=True)
            - Related entity: generate_entity_template("OrderLine", "ORDER", include_associations=True)

            **Output:** Complete XML entity definition ready for IFS Cloud development

            Args:
                entity_name: Name of the entity (e.g., 'CustomerOrder')
                module: IFS module name (e.g., 'Order', 'Inventory')
                logical_unit: Optional logical unit name
                include_states: Include state machine (only for entities that progress through states)
                include_associations: Include association placeholders
            """
            lu_name = logical_unit or f"{entity_name}Lu"
            component = module.upper()

            template = f"""<?xml version="1.0" encoding="UTF-8"?>
<ENTITY xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="urn:ifsworld-com:schemas:entity_entity">
   <CODE_GENERATION_PROPERTIES>
      <CODE_GENERATION_PROPERTIES>
         <DB_RMCOM_ACCESS>{entity_name}({entity_name.lower()}_id)</DB_RMCOM_ACCESS>
         <DB_DATA_SYNC>true</DB_DATA_SYNC>
         <DB_DATA_SYNC_SITE>CONTRACT</DB_DATA_SYNC_SITE>
         <TIMESTAMP_T_Z_REF>site(Contract)</TIMESTAMP_T_Z_REF>
      </CODE_GENERATION_PROPERTIES>
   </CODE_GENERATION_PROPERTIES>
   
   <COMMENTS>
      <COMMENT>
         <COMMENT_TYPE>ENTITY</COMMENT_TYPE>
         <TEXT>{entity_name} entity for {module} module</TEXT>
      </COMMENT>
   </COMMENTS>
   
   <NAME>Main</NAME>
   <COMPONENT>{component}</COMPONENT>
   
   <ATTRIBUTES>
      <ATTRIBUTE>
         <CODE_GENERATION_PROPERTIES>
            <COLUMN_NAME>{entity_name.upper()}_ID</COLUMN_NAME>
         </CODE_GENERATION_PROPERTIES>
         <NAME>{entity_name}Id</NAME>
         <IS_PRIMARY_KEY>1</IS_PRIMARY_KEY>
         <IS_PARENT_KEY>0</IS_PARENT_KEY>
         <IS_PUBLIC>0</IS_PUBLIC>
         <IS_MANDATORY>1</IS_MANDATORY>
         <IS_SERVER_GENERATED>0</IS_SERVER_GENERATED>
         <IS_UPDATE_ALLOWED>0</IS_UPDATE_ALLOWED>
         <IS_UPDATE_ALLOWED_IF_NULL>0</IS_UPDATE_ALLOWED_IF_NULL>
         <IS_DEFAULT_LOV>1</IS_DEFAULT_LOV>
         <IS_QUERYABLE>1</IS_QUERYABLE>
         <IS_DERIVED>0</IS_DERIVED>
         <DATATYPE>TEXT</DATATYPE>
         <LENGTH>50</LENGTH>
         <FORMAT>UPPERCASE</FORMAT>
         <PROMPT>{entity_name} ID</PROMPT>
      </ATTRIBUTE>
      
      <ATTRIBUTE>
         <NAME>Description</NAME>
         <IS_PRIMARY_KEY>0</IS_PRIMARY_KEY>
         <IS_PARENT_KEY>0</IS_PARENT_KEY>
         <IS_PUBLIC>1</IS_PUBLIC>
         <IS_MANDATORY>0</IS_MANDATORY>
         <IS_SERVER_GENERATED>0</IS_SERVER_GENERATED>
         <IS_UPDATE_ALLOWED>1</IS_UPDATE_ALLOWED>
         <IS_UPDATE_ALLOWED_IF_NULL>0</IS_UPDATE_ALLOWED_IF_NULL>
         <IS_DEFAULT_LOV>1</IS_DEFAULT_LOV>
         <IS_QUERYABLE>1</IS_QUERYABLE>
         <IS_DERIVED>0</IS_DERIVED>
         <DATATYPE>TEXT</DATATYPE>
         <LENGTH>200</LENGTH>
         <PROMPT>Description</PROMPT>
      </ATTRIBUTE>
      
      <ATTRIBUTE>
         <NAME>State</NAME>
         <IS_PRIMARY_KEY>0</IS_PRIMARY_KEY>
         <IS_PARENT_KEY>0</IS_PARENT_KEY>
         <IS_PUBLIC>1</IS_PUBLIC>
         <IS_MANDATORY>1</IS_MANDATORY>
         <IS_SERVER_GENERATED>0</IS_SERVER_GENERATED>
         <IS_UPDATE_ALLOWED>1</IS_UPDATE_ALLOWED>
         <IS_UPDATE_ALLOWED_IF_NULL>0</IS_UPDATE_ALLOWED_IF_NULL>
         <IS_DEFAULT_LOV>0</IS_DEFAULT_LOV>
         <IS_QUERYABLE>1</IS_QUERYABLE>
         <IS_DERIVED>0</IS_DERIVED>
         <DATATYPE>ENUMERATION</DATATYPE>
         <PROMPT>State</PROMPT>
      </ATTRIBUTE>
      
      <ATTRIBUTE>
         <NAME>CreatedDate</NAME>
         <IS_PRIMARY_KEY>0</IS_PRIMARY_KEY>
         <IS_PARENT_KEY>0</IS_PARENT_KEY>
         <IS_PUBLIC>1</IS_PUBLIC>
         <IS_MANDATORY>1</IS_MANDATORY>
         <IS_SERVER_GENERATED>1</IS_SERVER_GENERATED>
         <IS_UPDATE_ALLOWED>0</IS_UPDATE_ALLOWED>
         <IS_UPDATE_ALLOWED_IF_NULL>0</IS_UPDATE_ALLOWED_IF_NULL>
         <IS_DEFAULT_LOV>0</IS_DEFAULT_LOV>
         <IS_QUERYABLE>1</IS_QUERYABLE>
         <IS_DERIVED>0</IS_DERIVED>
         <DATATYPE>TIMESTAMP</DATATYPE>
         <PROMPT>Created Date</PROMPT>
      </ATTRIBUTE>
      
      <ATTRIBUTE>
         <NAME>CreatedBy</NAME>
         <IS_PRIMARY_KEY>0</IS_PRIMARY_KEY>
         <IS_PARENT_KEY>0</IS_PARENT_KEY>
         <IS_PUBLIC>1</IS_PUBLIC>
         <IS_MANDATORY>1</IS_MANDATORY>
         <IS_SERVER_GENERATED>1</IS_SERVER_GENERATED>
         <IS_UPDATE_ALLOWED>0</IS_UPDATE_ALLOWED>
         <IS_UPDATE_ALLOWED_IF_NULL>0</IS_UPDATE_ALLOWED_IF_NULL>
         <IS_DEFAULT_LOV>0</IS_DEFAULT_LOV>
         <IS_QUERYABLE>1</IS_QUERYABLE>
         <IS_DERIVED>0</IS_DERIVED>
         <DATATYPE>TEXT</DATATYPE>
         <LENGTH>30</LENGTH>
         <PROMPT>Created By</PROMPT>
      </ATTRIBUTE>
      
      <ATTRIBUTE>
         <NAME>Rowversion</NAME>
         <IS_PRIMARY_KEY>0</IS_PRIMARY_KEY>
         <IS_PARENT_KEY>0</IS_PARENT_KEY>
         <IS_PUBLIC>1</IS_PUBLIC>
         <IS_MANDATORY>1</IS_MANDATORY>
         <IS_SERVER_GENERATED>1</IS_SERVER_GENERATED>
         <IS_UPDATE_ALLOWED>0</IS_UPDATE_ALLOWED>
         <IS_UPDATE_ALLOWED_IF_NULL>0</IS_UPDATE_ALLOWED_IF_NULL>
         <IS_DEFAULT_LOV>0</IS_DEFAULT_LOV>
         <IS_QUERYABLE>0</IS_QUERYABLE>
         <IS_DERIVED>0</IS_DERIVED>
         <DATATYPE>TIMESTAMP</DATATYPE>
         <PROMPT>Row Version</PROMPT>
      </ATTRIBUTE>
   </ATTRIBUTES>"""

            # Only add states if explicitly requested (for entities that need state progression)
            if include_states:
                template += f"""
   
   <STATES>
      <STATE>
         <NAME>{{start}}</NAME>
         <DESCRIPTION>Initial state</DESCRIPTION>
      </STATE>
      <STATE>
         <NAME>Planned</NAME>
         <DESCRIPTION>Planned state</DESCRIPTION>
      </STATE>
      <STATE>
         <NAME>Active</NAME>
         <DESCRIPTION>Active state</DESCRIPTION>
      </STATE>
      <STATE>
         <NAME>Completed</NAME>
         <DESCRIPTION>Completed state</DESCRIPTION>
      </STATE>
      <STATE>
         <NAME>Cancelled</NAME>
         <DESCRIPTION>Cancelled state</DESCRIPTION>
      </STATE>
   </STATES>"""

            # Only add associations if requested
            if include_associations:
                template += f"""
   
   <ASSOCIATIONS>
      <!-- Add associations here as needed -->
      <!-- Example:
      <ASSOCIATION>
         <NAME>ParentRef</NAME>
         <IS_PARENT>1</IS_PARENT>
         <IS_VIEW_REFERENCE>0</IS_VIEW_REFERENCE>
         <TO_ENTITY>ParentEntity</TO_ENTITY>
         <ATTRIBUTES>
            <ATTRIBUTE>
               <FROM_ATTRIBUTE>ParentId</FROM_ATTRIBUTE>
               <TO_ATTRIBUTE>Id</TO_ATTRIBUTE>
            </ATTRIBUTE>
         </ATTRIBUTES>
      </ASSOCIATION>
      -->
   </ASSOCIATIONS>"""

            template += f"""
   
   <OBJECT_CONNECTIONS>
      <!-- Object connections will be auto-generated -->
   </OBJECT_CONNECTIONS>
</ENTITY>"""

            state_note = (
                " (with state machine)" if include_states else " (simple entity)"
            )
            assoc_note = " and associations" if include_associations else ""

            return f"Generated IFS Cloud Entity Template{state_note}:{assoc_note}\\n\\n```xml\\n{template}\\n```\\n\\n**Note**: This template creates a {'stateful' if include_states else 'simple'} entity without unnecessary diagrams. States are only included when the entity needs to progress through different lifecycle phases."

        @self.mcp.tool()
        async def generate_plsql_package_template(
            package_name: str,
            entity_name: str,
            include_crud: bool = True,
            include_business_logic: bool = True,
        ) -> str:
            """Generate PL/SQL package template for IFS Cloud entity.

            **AGENT INSTRUCTIONS:**
            This tool creates PL/SQL packages following IFS Cloud standards and patterns.

            **When to use:**
            - Creating API packages for entities
            - Building business logic packages
            - Setting up logical unit implementations
            - Creating utility packages

            **Package naming conventions:**
            - Entity APIs: EntityName_API (Customer_Order_API, Product_API)
            - Logical Units: LogicalUnitName (Customer_Order_Handling, Product_Mgmt)
            - Utilities: ModuleName_Util_API (Order_Util_API, Inventory_Util_API)

            **Parameter guidance:**
            - package_name: Follow IFS naming with underscores (Customer_Order_API)
            - entity_name: Related entity name (CustomerOrder, Product)
            - include_crud: True for entity APIs (creates New__, Modify__, Remove__ procedures)
            - include_business_logic: True for packages with business rules and validations

            **Examples:**
            - Entity API: generate_plsql_package_template("Customer_Order_API", "CustomerOrder")
            - Business logic: generate_plsql_package_template("Pricing_Engine", "CustomerOrder", include_crud=False)
            - Utility: generate_plsql_package_template("Order_Util_API", "OrderUtil", include_business_logic=False)

            **Output:** Complete PL/SQL package specification and body with IFS Cloud patterns

            Args:
                package_name: Package name (e.g., 'Customer_Order_API')
                entity_name: Related entity name
                include_crud: Include CRUD procedures
                include_business_logic: Include business logic templates
            """

            spec_template = f"""-- Generated PL/SQL Package Specification for {package_name}
-- =============================================================================
-- AGENT INSTRUCTIONS:
-- 1. CONSTANTS SECTION: Add module and logical unit constants below
-- 2. TYPES SECTION: Define record types and table types below  
-- 3. PUBLIC METHODS: Add procedure/function declarations below
-- 4. Follow IFS naming conventions (Pascal_Case for procedures/functions)
-- =============================================================================

CREATE OR REPLACE PACKAGE {package_name} IS

   -- =============================================================================
   -- CONSTANTS SECTION - AGENTS ADD MODULE/LU CONSTANTS HERE
   -- =============================================================================
   module_    CONSTANT VARCHAR2(25) := 'ORDER';  -- AGENT: Update module name
   lu_name_   CONSTANT VARCHAR2(25) := '{entity_name.upper()}';  -- AGENT: Logical unit name

   -- =============================================================================
   -- TYPES SECTION - AGENTS ADD RECORD TYPES AND TABLE TYPES HERE
   -- =============================================================================
   TYPE Public_Rec IS RECORD (
      -- AGENT: Add record fields based on table structure
      id             {entity_name.lower()}_tab.id%TYPE,
      description    {entity_name.lower()}_tab.description%TYPE,
      state          {entity_name.lower()}_tab.state%TYPE,
      created_date   {entity_name.lower()}_tab.created_date%TYPE,
      created_by     {entity_name.lower()}_tab.created_by%TYPE
      -- AGENT: Add more fields as needed
   );

   TYPE Public_Rec_Array IS TABLE OF Public_Rec INDEX BY PLS_INTEGER;
   -- AGENT: Add more collection types here if needed

   -- =============================================================================
   -- PUBLIC METHODS SECTION - AGENTS ADD METHOD DECLARATIONS HERE
   -- =============================================================================
"""

            if include_crud:
                spec_template += f"""
   -- =============================================================================
   -- CRUD OPERATIONS SECTION - AGENTS ADD CRUD METHOD DECLARATIONS HERE
   -- =============================================================================
   
   -- AGENT: Get method - retrieves single record
   FUNCTION Get (
      id_ IN VARCHAR2 ) RETURN Public_Rec;
      
   -- AGENT: New method - creates new record (follows IFS pattern)
   PROCEDURE New (
      info_ OUT VARCHAR2,        -- AGENT: Standard IFS info parameter for messages
      objid_ OUT VARCHAR2,       -- AGENT: Object ID output parameter
      objversion_ OUT VARCHAR2,  -- AGENT: Object version for optimistic locking
      attr_ IN OUT VARCHAR2,     -- AGENT: Attribute string with field values
      action_ IN VARCHAR2 DEFAULT 'DO' );  -- AGENT: Action (CHECK/DO/PREPARE)
      
   -- AGENT: Modify method - updates existing record (follows IFS pattern)  
   PROCEDURE Modify (
      info_ OUT VARCHAR2,        -- AGENT: Standard IFS info parameter for messages
      objid_ IN VARCHAR2,        -- AGENT: Object ID input parameter
      objversion_ IN OUT VARCHAR2, -- AGENT: Object version for optimistic locking
      attr_ IN OUT VARCHAR2,     -- AGENT: Attribute string with field values
      action_ IN VARCHAR2 DEFAULT 'DO' );  -- AGENT: Action (CHECK/DO/PREPARE)
      
   -- AGENT: Remove method - deletes record (follows IFS pattern)
   PROCEDURE Remove (
      info_ OUT VARCHAR2,        -- AGENT: Standard IFS info parameter for messages
      objid_ IN VARCHAR2,        -- AGENT: Object ID input parameter
      objversion_ IN VARCHAR2,   -- AGENT: Object version for optimistic locking
      action_ IN VARCHAR2 DEFAULT 'DO' );  -- AGENT: Action (CHECK/DO/PREPARE)
"""

            if include_business_logic:
                spec_template += f"""
   -- =============================================================================
   -- BUSINESS LOGIC SECTION - AGENTS ADD BUSINESS METHOD DECLARATIONS HERE
   -- =============================================================================
   
   -- AGENT: Validation methods - add data validation procedures
   PROCEDURE Validate_Id (
      id_ IN VARCHAR2 );         -- AGENT: Validates ID field format/existence
      
   -- AGENT: State management - add state transition validations
   PROCEDURE Check_State_Transition (
      id_ IN VARCHAR2,           -- AGENT: Record identifier
      new_state_ IN VARCHAR2 );  -- AGENT: Target state to validate
      
   -- AGENT: Calculation methods - add business calculation functions
   FUNCTION Calculate_Total (
      id_ IN VARCHAR2 ) RETURN NUMBER;  -- AGENT: Returns calculated business values
      
   -- AGENT: Business rule enforcement - add rule processing procedures
   PROCEDURE Process_Business_Rule (
      id_ IN VARCHAR2,           -- AGENT: Record identifier  
      action_ IN VARCHAR2 );     -- AGENT: Business action to process
      
   -- AGENT: Add more business logic methods here as needed
"""

            spec_template += f"""
   -- =============================================================================
   -- END OF PACKAGE SPECIFICATION
   -- =============================================================================

END {package_name};
/
"""

            body_template = f"""-- Generated PL/SQL Package Body for {package_name}
-- =============================================================================
-- AGENT INSTRUCTIONS:
-- 1. PRIVATE METHODS: Add internal/helper procedures below private section
-- 2. PUBLIC METHOD IMPLEMENTATIONS: Implement all public methods from spec
-- 3. VARIABLE DECLARATIONS: Add local variables in method bodies
-- 4. ERROR HANDLING: Use Error_SYS.Record_General for IFS standard errors
-- 5. LOGGING: Use Trace_SYS for debugging when needed
-- =============================================================================

CREATE OR REPLACE PACKAGE BODY {package_name} IS

   -- =============================================================================
   -- PRIVATE METHODS SECTION - AGENTS ADD HELPER PROCEDURES HERE
   -- =============================================================================
   
   -- AGENT: Add private helper methods here (suffix with ___)
   PROCEDURE Check_Exist___ (
      id_ IN VARCHAR2 ) IS
   BEGIN
      -- AGENT: Implement existence check logic
      -- AGENT: Raise error if record doesn't exist
      -- Example: Error_SYS.Record_Not_Exist(lu_name_);
      NULL;
   END Check_Exist___;

   -- AGENT: Add more private methods here as needed
   -- Examples: Validate_Fields___, Calculate_Derived___, etc.

   -- =============================================================================
   -- PUBLIC METHOD IMPLEMENTATIONS - AGENTS IMPLEMENT SPEC METHODS HERE
   -- =============================================================================

   -- AGENT: Implement Get function - return populated record
   FUNCTION Get (
      id_ IN VARCHAR2 ) RETURN Public_Rec IS
      
      rec_ Public_Rec;  -- AGENT: Declare return record variable
   BEGIN
      -- AGENT: Add cursor or direct SELECT to populate rec_
      -- AGENT: Handle NO_DATA_FOUND exception
      -- Example implementation needed here
      RETURN rec_;
   END Get;

   -- =============================================================================
   -- BUSINESS LOGIC IMPLEMENTATIONS - AGENTS ADD BUSINESS RULES HERE  
   -- =============================================================================

   -- AGENT: Implement validation logic
   PROCEDURE Validate_Id (
      id_ IN VARCHAR2 ) IS
   BEGIN
      -- AGENT: Add validation logic here
      -- AGENT: Check ID format, length, existence
      -- AGENT: Use Error_SYS.Record_General for validation failures
      -- Example: IF id_ IS NULL THEN Error_SYS.Record_General(...);
      NULL;
   END Validate_Id;

   -- AGENT: Add more business logic implementations here
   -- AGENT: Follow IFS patterns for error handling and logging

   -- =============================================================================
   -- INITIALIZATION SECTION - AGENTS ADD PACKAGE INITIALIZATION HERE
   -- =============================================================================

BEGIN
   -- AGENT: Add package initialization code here if needed
   -- AGENT: This section runs when package is first loaded
   NULL;

END {package_name};
/
"""

            return f"Generated PL/SQL Package Templates:\\n\\n**Specification:**\\n```sql\\n{spec_template}```\\n\\n**Body:**\\n```sql\\n{body_template}```"

        @self.mcp.tool()
        async def analyze_ifs_frontend_elements(
            query: Optional[str] = None,
            element_type: Optional[str] = None,
            limit: int = 20,
        ) -> str:
            """Analyze IFS Cloud frontend elements (pages, lists, groups, etc.).

            **AGENT INSTRUCTIONS:**
            This tool analyzes IFS Cloud frontend components from real production code.

            **When to use:**
            - Understanding existing UI patterns in IFS Cloud
            - Finding examples of specific frontend elements
            - Analyzing how IFS implements common UI components
            - Learning naming conventions for frontend elements

            **Element types available:**
            - pages: Main application pages and forms
            - lists: Data grids and list views
            - groups: UI grouping and layout containers
            - iconsets: Icon definitions and usage
            - trees: Hierarchical tree views
            - navigators: Navigation menu structures

            **Search strategies:**
            - Specific element: query="CustomerOrder" (finds all CustomerOrder-related UI)
            - By type: element_type="pages" (shows all page definitions)
            - Pattern search: query="search" (finds search-related components)
            - Workflow UI: query="approval workflow" (finds approval UI patterns)

            **Examples:**
            - Find order pages: analyze_ifs_frontend_elements("CustomerOrder", "pages")
            - Study list patterns: analyze_ifs_frontend_elements(element_type="lists", limit=10)
            - Search patterns: analyze_ifs_frontend_elements("search filter")

            **Output:** Analysis of frontend elements with patterns, naming, and usage examples

            Args:
                query: Optional search query for elements
                element_type: Filter by element type (pages, lists, groups, iconsets, trees, navigators)
                limit: Maximum results to return
            """

            # Build search query
            search_query = "*"
            if query:
                search_query = query

            results = self.indexer.search_deduplicated(
                query=search_query, limit=limit * 2
            )

            element_counts = {
                "pages": 0,
                "lists": 0,
                "groups": 0,
                "iconsets": 0,
                "trees": 0,
                "navigators": 0,
                "contexts": 0,
            }

            element_details = {
                "pages": [],
                "lists": [],
                "groups": [],
                "iconsets": [],
                "trees": [],
                "navigators": [],
                "contexts": [],
            }

            for result in results:
                # Count and collect elements from each result
                for element_name in [
                    "pages",
                    "lists",
                    "groups",
                    "iconsets",
                    "trees",
                    "navigators",
                    "contexts",
                ]:
                    elements = getattr(result, element_name, [])
                    if elements:
                        element_counts[element_name] += len(elements)
                        for element in elements:
                            if not element_type or element_type == element_name:
                                element_details[element_name].append(
                                    {
                                        "name": element,
                                        "file": result.name,
                                        "path": result.path,
                                        "module": result.module,
                                    }
                                )

            response_lines = ["IFS Cloud Frontend Elements Analysis", "=" * 40, ""]

            # Summary
            response_lines.extend(
                [
                    "Summary:",
                    f"  Pages: {element_counts['pages']}",
                    f"  Lists: {element_counts['lists']}",
                    f"  Groups: {element_counts['groups']}",
                    f"  Icon Sets: {element_counts['iconsets']}",
                    f"  Trees: {element_counts['trees']}",
                    f"  Navigators: {element_counts['navigators']}",
                    f"  Contexts: {element_counts['contexts']}",
                    "",
                ]
            )

            # Detailed listings
            for element_name, elements in element_details.items():
                if elements and (not element_type or element_type == element_name):
                    response_lines.extend(
                        [f"{element_name.title()}:", "-" * (len(element_name) + 1)]
                    )

                    for element in elements[:limit]:
                        response_lines.append(
                            f"  • {element['name']} ({element['file']})"
                        )
                        if element["module"]:
                            response_lines.append(f"    Module: {element['module']}")

                    if len(elements) > limit:
                        response_lines.append(
                            f"    ... and {len(elements) - limit} more"
                        )
                    response_lines.append("")

            return "\\n".join(response_lines)

        @self.mcp.tool()
        async def suggest_ifs_patterns(
            business_requirement: str,
            target_module: Optional[str] = None,
        ) -> str:
            """Suggest IFS Cloud implementation patterns for business requirements.

            Args:
                business_requirement: Description of the business requirement
                target_module: Target IFS module for implementation
            """

            patterns = {
                "order": {
                    "entities": ["Order", "OrderLine", "OrderStatus"],
                    "apis": ["Order_API", "Order_Line_API"],
                    "views": ["OrderView", "OrderLineView"],
                    "states": ["Planned", "Released", "Completed", "Cancelled"],
                },
                "inventory": {
                    "entities": ["InventoryPart", "InventoryTransaction", "Location"],
                    "apis": ["Inventory_Part_API", "Inventory_Transaction_API"],
                    "views": ["InventoryPartView", "LocationView"],
                    "states": ["Active", "Inactive", "Obsolete"],
                },
                "customer": {
                    "entities": ["Customer", "CustomerContact", "CustomerAddress"],
                    "apis": ["Customer_API", "Customer_Contact_API"],
                    "views": ["CustomerView", "ContactView"],
                    "states": ["Active", "Inactive", "Prospect"],
                },
                "financial": {
                    "entities": ["Account", "Transaction", "Period"],
                    "apis": ["Account_API", "Financial_Transaction_API"],
                    "views": ["AccountView", "TransactionView"],
                    "states": ["Open", "Closed", "Blocked"],
                },
            }

            # Analyze requirement to suggest patterns
            req_lower = business_requirement.lower()
            suggested_patterns = []

            for domain, pattern in patterns.items():
                if any(
                    keyword in req_lower
                    for keyword in [domain, *pattern["entities"][0].lower().split()]
                ):
                    suggested_patterns.append((domain, pattern))

            if not suggested_patterns:
                # Default comprehensive pattern
                suggested_patterns = [
                    (
                        "generic",
                        {
                            "entities": ["MainEntity", "DetailEntity", "StatusEntity"],
                            "apis": ["Main_Entity_API", "Detail_Entity_API"],
                            "views": ["MainEntityView", "DetailEntityView"],
                            "states": ["Draft", "Active", "Inactive"],
                        },
                    )
                ]

            response_lines = [
                f"IFS Cloud Implementation Patterns for: {business_requirement}",
                "=" * 60,
                "",
            ]

            if target_module:
                response_lines.extend([f"Target Module: {target_module}", ""])

            for domain, pattern in suggested_patterns:
                response_lines.extend(
                    [
                        f"Suggested Pattern: {domain.title()}",
                        "-" * 30,
                        "",
                        "Recommended Entities:",
                    ]
                )

                for entity in pattern["entities"]:
                    response_lines.append(f"  • {entity}.entity")

                response_lines.extend(
                    [
                        "",
                        "Required APIs:",
                    ]
                )

                for api in pattern["apis"]:
                    response_lines.append(f"  • {api}.plsql")

                response_lines.extend(
                    [
                        "",
                        "Frontend Views:",
                    ]
                )

                for view in pattern["views"]:
                    response_lines.append(f"  • {view}.views")

                response_lines.extend(
                    [
                        "",
                        "Suggested States:",
                    ]
                )

                for state in pattern["states"]:
                    response_lines.append(f"  • {state}")

                response_lines.append("")

            # Add implementation guidance
            response_lines.extend(
                [
                    "Implementation Steps:",
                    "1. Create entity definitions with proper attributes",
                    "2. Implement PL/SQL APIs with business logic",
                    "3. Create database views for data access",
                    "4. Define state machine if applicable",
                    "5. Add validation and business rules",
                    "6. Create frontend projections and pages",
                    "7. Implement integration points",
                    "",
                ]
            )

            return "\\n".join(response_lines)

        @self.mcp.tool()
        async def validate_ifs_coding_standards(
            file_path: Optional[str] = None,
            content: Optional[str] = None,
            file_type: Optional[str] = None,
        ) -> str:
            """Validate IFS Cloud coding standards and best practices.

            Args:
                file_path: Path to file to validate (will read from index)
                content: Direct content to validate
                file_type: File type (.entity, .plsql, .views, etc.)
            """

            if file_path:
                # Search for the file in index
                results = self.indexer.search_deduplicated(
                    query=f"path:{file_path}", limit=1
                )
                if not results:
                    return f"File not found in index: {file_path}"

                file_result = results[0]
                content = file_result.content if hasattr(file_result, "content") else ""
                file_type = file_result.type

            if not content:
                return "No content provided for validation"

            violations = []
            suggestions = []

            # Entity file standards
            if file_type == ".entity":
                if "crud = " not in content:
                    violations.append("Missing CRUD definition")
                    suggestions.append("Add: crud = Create, Read, Update, Delete;")

                if "ludependencies = " not in content:
                    violations.append("Missing ludependencies definition")
                    suggestions.append(
                        "Add: ludependencies = None; (or specify dependencies)"
                    )

                if (
                    "required = [true]" in content
                    and "insertable = [true]" not in content
                ):
                    suggestions.append(
                        "Consider adding insertable/updatable flags for required fields"
                    )

                # Check for standard attributes
                if "created_date" not in content.lower():
                    suggestions.append(
                        "Consider adding CreatedDate attribute for audit trail"
                    )

                if "created_by" not in content.lower():
                    suggestions.append(
                        "Consider adding CreatedBy attribute for audit trail"
                    )

            # PL/SQL standards
            elif file_type == ".plsql":
                if (
                    "PACKAGE" in content.upper()
                    and "PACKAGE BODY" not in content.upper()
                ):
                    if "PROCEDURE" in content.upper() or "FUNCTION" in content.upper():
                        violations.append(
                            "Package specification should not contain implementation"
                        )

                if "module_" not in content and "CONSTANT" in content:
                    suggestions.append("Consider adding module_ constant")

                if "lu_name_" not in content and "CONSTANT" in content:
                    suggestions.append("Consider adding lu_name_ constant")

                # Check error handling
                if (
                    "EXCEPTION" not in content.upper()
                    and "RAISE" not in content.upper()
                ):
                    suggestions.append("Consider adding proper exception handling")

            # Views standards
            elif file_type == ".views":
                if "entityset" in content.lower() and "from" not in content.lower():
                    violations.append("EntitySet should specify data source")

                if "page" in content.lower() and "entityset" not in content.lower():
                    suggestions.append("Consider using EntitySet for data binding")

            # General standards
            lines = content.split("\\n")
            for i, line in enumerate(lines, 1):
                if len(line) > 120:
                    violations.append(f"Line {i}: Exceeds 120 character limit")

                if line.strip().endswith(";") and file_type in [".entity", ".views"]:
                    # Check for proper semicolon usage
                    pass

            response_lines = [f"IFS Cloud Coding Standards Validation", "=" * 40, ""]

            if file_path:
                response_lines.append(f"File: {file_path}")
            response_lines.extend([f"File Type: {file_type}", ""])

            if violations:
                response_lines.extend(["❌ Violations Found:", "-" * 20])
                for violation in violations:
                    response_lines.append(f"  • {violation}")
                response_lines.append("")

            if suggestions:
                response_lines.extend(["💡 Suggestions:", "-" * 15])
                for suggestion in suggestions:
                    response_lines.append(f"  • {suggestion}")
                response_lines.append("")

            if not violations and not suggestions:
                response_lines.append(
                    "✅ No issues found - code follows IFS standards!"
                )

            return "\\n".join(response_lines)

        @self.mcp.tool()
        async def find_integration_patterns(
            integration_type: str,
            source_entity: Optional[str] = None,
            target_system: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            """Find IFS Cloud integration patterns and examples.

            **AGENT INSTRUCTIONS:**
            This tool discovers integration patterns between IFS Cloud and external systems.

            **When to use:**
            - Planning integrations with external systems
            - Understanding IFS Cloud integration capabilities
            - Finding examples of specific integration types
            - Learning how entities connect across modules

            **Integration types:**
            - "api": REST/SOAP API integrations
            - "event": Event-driven integrations
            - "batch": Scheduled batch processing
            - "realtime": Real-time data synchronization

            **Use cases:**
            - ERP integrations (SAP, Oracle, etc.)
            - CRM system connections
            - E-commerce platform links
            - Manufacturing system interfaces
            - Financial system integrations

            **Search strategies:**
            - By type: find_integration_patterns("api") - shows all API patterns
            - Entity-specific: find_integration_patterns("api", "CustomerOrder") - order API integrations
            - System-specific: find_integration_patterns("batch", target_system="SAP") - SAP batch patterns

            **Examples:**
            - API patterns: find_integration_patterns("api", "Customer")
            - Event handling: find_integration_patterns("event", "Order")
            - Batch processing: find_integration_patterns("batch", limit=15)

            **Output:** Integration patterns with code examples, endpoints, and implementation details

            Args:
                integration_type: Type of integration (api, event, batch, realtime)
                source_entity: Source entity for integration
                target_system: Target system name
                limit: Maximum examples to return
            """

            # Search for integration-related files
            search_terms = ["integration", "api", "event", "batch", "interface"]
            if source_entity:
                search_terms.append(source_entity.lower())
            if target_system:
                search_terms.append(target_system.lower())

            query = " OR ".join(search_terms)
            results = self.indexer.search_deduplicated(query=query, limit=limit * 2)

            # Categorize integration patterns
            patterns = {
                "api": {
                    "description": "REST/SOAP API integrations",
                    "files": [],
                    "patterns": ["HTTP calls", "JSON/XML processing", "Authentication"],
                },
                "event": {
                    "description": "Event-driven integrations",
                    "files": [],
                    "patterns": [
                        "Event publishing",
                        "Event subscription",
                        "Message queues",
                    ],
                },
                "batch": {
                    "description": "Batch processing integrations",
                    "files": [],
                    "patterns": [
                        "File processing",
                        "Scheduled jobs",
                        "Data transformation",
                    ],
                },
                "realtime": {
                    "description": "Real-time data synchronization",
                    "files": [],
                    "patterns": ["Triggers", "Change data capture", "Streaming"],
                },
            }

            # Analyze results for patterns
            for result in results:
                content = getattr(result, "content", "").lower()

                if any(term in content for term in ["http", "rest", "soap", "api"]):
                    patterns["api"]["files"].append(result)

                if any(
                    term in content
                    for term in ["event", "publish", "subscribe", "message"]
                ):
                    patterns["event"]["files"].append(result)

                if any(
                    term in content for term in ["batch", "job", "schedule", "file"]
                ):
                    patterns["batch"]["files"].append(result)

                if any(
                    term in content
                    for term in ["trigger", "realtime", "sync", "stream"]
                ):
                    patterns["realtime"]["files"].append(result)

            response_lines = [
                f"IFS Cloud Integration Patterns: {integration_type.title()}",
                "=" * 50,
                "",
            ]

            if source_entity:
                response_lines.append(f"Source Entity: {source_entity}")
            if target_system:
                response_lines.append(f"Target System: {target_system}")
            response_lines.append("")

            # Show specific pattern or all patterns
            patterns_to_show = (
                [integration_type] if integration_type in patterns else patterns.keys()
            )

            for pattern_type in patterns_to_show:
                pattern = patterns[pattern_type]
                response_lines.extend(
                    [
                        f"{pattern_type.upper()} Integration Pattern",
                        "-" * 30,
                        f"Description: {pattern['description']}",
                        "",
                        "Common Patterns:",
                    ]
                )

                for p in pattern["patterns"]:
                    response_lines.append(f"  • {p}")

                if pattern["files"]:
                    response_lines.extend(
                        [
                            "",
                            "Example Files:",
                        ]
                    )

                    for file_result in pattern["files"][:limit]:
                        response_lines.append(
                            f"  • {file_result.name} ({file_result.type})"
                        )
                        if file_result.module:
                            response_lines.append(f"    Module: {file_result.module}")
                        response_lines.append(f"    Path: {file_result.path}")
                        response_lines.append("")

                response_lines.append("")

            return "\\n".join(response_lines)

        @self.mcp.tool()
        async def generate_business_logic_template(
            entity_name: str,
            business_rule: str,
            rule_type: str = "validation",
        ) -> str:
            """Generate business logic template for IFS Cloud entities.

            Args:
                entity_name: Target entity name
                business_rule: Description of the business rule to implement
                rule_type: Type of rule (validation, calculation, workflow, trigger)
            """

            templates = {
                "validation": f"""-- =============================================================================
-- VALIDATION RULE TEMPLATE for {entity_name}
-- Rule: {business_rule}
-- AGENT INSTRUCTIONS:
-- 1. PARAMETER SECTION: rec_ contains all table fields for validation
-- 2. VALIDATION LOGIC: Add IF statements to check business rules  
-- 3. ERROR HANDLING: Use Error_SYS.Record_General for validation failures
-- 4. FIELD ACCESS: Use rec_.field_name to access table columns
-- =============================================================================

PROCEDURE Validate_{entity_name}_Rule___ (
   rec_ IN OUT {entity_name.lower()}_tab%ROWTYPE ) IS
BEGIN
   -- =============================================================================
   -- VALIDATION LOGIC SECTION - AGENTS ADD VALIDATION CHECKS HERE
   -- =============================================================================
   
   -- AGENT: Add field validation checks using IF statements
   -- AGENT: Example field validation:
   IF rec_.some_field IS NULL THEN
      Error_SYS.Record_General(
         lu_name_, 
         'FIELD_REQUIRED: Field :P1 is required',
         'Some Field');  -- AGENT: Replace with actual field name
   END IF;
   
   -- AGENT: Add business rule validations here
   -- AGENT: Example business rule check:
   -- IF rec_.start_date > rec_.end_date THEN
   --    Error_SYS.Record_General(lu_name_, 'INVALID_DATE_RANGE: Start date cannot be after end date');
   -- END IF;
   
   -- AGENT: Add more validation logic below
   
END Validate_{entity_name}_Rule___;""",
                "calculation": f"""-- =============================================================================
-- CALCULATION RULE TEMPLATE for {entity_name}
-- Rule: {business_rule}
-- AGENT INSTRUCTIONS:
-- 1. PARAMETER SECTION: rec_ contains input data for calculations
-- 2. VARIABLE SECTION: Declare local variables for intermediate results
-- 3. CALCULATION LOGIC: Implement business calculation formulas
-- 4. RETURN SECTION: Return calculated result
-- =============================================================================

FUNCTION Calculate_{entity_name}_Value___ (
   rec_ IN {entity_name.lower()}_tab%ROWTYPE ) RETURN NUMBER IS
   
   -- =============================================================================
   -- VARIABLE DECLARATIONS - AGENTS DECLARE CALCULATION VARIABLES HERE
   -- =============================================================================
   result_ NUMBER := 0;           -- AGENT: Main result variable
   intermediate_ NUMBER := 0;     -- AGENT: Add intermediate calculation variables
   factor_ NUMBER := 1;           -- AGENT: Add factor/multiplier variables
   
BEGIN
   -- =============================================================================
   -- CALCULATION LOGIC SECTION - AGENTS ADD CALCULATION FORMULAS HERE
   -- =============================================================================
   
   -- AGENT: Implement main calculation logic
   -- AGENT: Example basic calculation:
   result_ := rec_.quantity * rec_.unit_price;  -- AGENT: Replace with actual fields
   
   -- AGENT: Add conditional calculations based on business rules
   -- AGENT: Example discount calculation:
   IF rec_.discount_percent > 0 THEN
      result_ := result_ * (1 - rec_.discount_percent / 100);
   END IF;
   
   -- AGENT: Add more complex calculations here
   -- AGENT: Example multi-step calculation:
   -- intermediate_ := Calculate_Tax_Amount(result_, rec_.tax_code);
   -- result_ := result_ + intermediate_;
   
   -- =============================================================================
   -- RETURN CALCULATED RESULT
   -- =============================================================================
   RETURN result_;  -- AGENT: Return final calculated value
   
END Calculate_{entity_name}_Value___;""",
                "workflow": f"""-- =============================================================================
-- WORKFLOW RULE TEMPLATE for {entity_name}
-- Rule: {business_rule}
-- AGENT INSTRUCTIONS:
-- 1. PARAMETER SECTION: rec_ for record data, action_ for workflow action
-- 2. STATE MANAGEMENT: Update rec_.state based on action
-- 3. VALIDATION LOGIC: Check conditions before state transitions
-- 4. LOGGING: Record state changes for audit trail
-- =============================================================================

PROCEDURE Process_{entity_name}_Workflow___ (
   rec_ IN OUT {entity_name.lower()}_tab%ROWTYPE,   -- AGENT: Record to process
   action_ IN VARCHAR2 ) IS                          -- AGENT: Workflow action (START/COMPLETE/CANCEL/etc)
BEGIN
   -- =============================================================================
   -- WORKFLOW ACTION PROCESSING - AGENTS ADD ACTION HANDLING HERE
   -- =============================================================================
   
   -- AGENT: Use CASE statement to handle different workflow actions
   CASE action_
   WHEN 'START' THEN
      -- AGENT: Handle workflow start action
      rec_.state := 'InProgress';              -- AGENT: Set appropriate state
      rec_.started_date := SYSDATE;            -- AGENT: Set timestamp fields
      -- AGENT: Add additional start logic here
      
   WHEN 'COMPLETE' THEN
      -- AGENT: Handle workflow completion action
      -- AGENT: Add completion condition checks
      IF rec_.all_items_delivered = 'TRUE' THEN  -- AGENT: Replace with actual conditions
         rec_.state := 'Completed';
         rec_.completed_date := SYSDATE;
      ELSE
         Error_SYS.Record_General(
            lu_name_, 
            'CANNOT_COMPLETE: Cannot complete - conditions not met');
      END IF;
      
   WHEN 'CANCEL' THEN
      -- AGENT: Handle workflow cancellation action
      rec_.state := 'Cancelled';
      rec_.cancelled_date := SYSDATE;
      -- AGENT: Add cancellation logic here
      
   -- AGENT: Add more workflow actions here as needed
   ELSE
      Error_SYS.Record_General(lu_name_, 'INVALID_ACTION: Invalid workflow action :P1', action_);
      
   END CASE;
   
   -- =============================================================================
   -- AUDIT LOGGING SECTION - AGENTS ADD LOGGING HERE
   -- =============================================================================
   
   -- AGENT: Log state change for audit trail
   {entity_name}_History_API.Create_Entry(
      rec_.id,                                 -- AGENT: Record identifier
      action_,                                 -- AGENT: Action performed
      rec_.state,                              -- AGENT: New state
      Fnd_Session_API.Get_Fnd_User);          -- AGENT: Current user
      
END Process_{entity_name}_Workflow___;""",
                "trigger": f"""-- =============================================================================
-- TRIGGER RULE TEMPLATE for {entity_name}
-- Rule: {business_rule}
-- AGENT INSTRUCTIONS:
-- 1. PARAMETER SECTION: old_rec_, new_rec_ for before/after data, action_ for operation type
-- 2. ACTION HANDLING: Use IF/CASE to handle INSERT/UPDATE/DELETE operations
-- 3. CHANGE DETECTION: Compare old_rec_ vs new_rec_ to detect field changes
-- 4. SIDE EFFECTS: Call other APIs to update related entities
-- =============================================================================

PROCEDURE Handle_{entity_name}_Change___ (
   old_rec_ IN {entity_name.lower()}_tab%ROWTYPE,   -- AGENT: Record before change
   new_rec_ IN {entity_name.lower()}_tab%ROWTYPE,   -- AGENT: Record after change  
   action_ IN VARCHAR2 ) IS                          -- AGENT: Operation type (INSERT/UPDATE/DELETE)
BEGIN
   -- =============================================================================
   -- TRIGGER ACTION PROCESSING - AGENTS ADD ACTION HANDLING HERE
   -- =============================================================================
   
   -- AGENT: Handle different database operations
   IF action_ = 'INSERT' THEN
      -- =============================================================================
      -- INSERT HANDLING - AGENTS ADD NEW RECORD LOGIC HERE
      -- =============================================================================
      
      -- AGENT: Handle new record creation logic
      Generate_{entity_name}_Number___(new_rec_);     -- AGENT: Auto-generate numbers/codes
      -- AGENT: Add initialization logic here
      -- AGENT: Set default values, create related records, etc.
      
   ELSIF action_ = 'UPDATE' THEN
      -- =============================================================================
      -- UPDATE HANDLING - AGENTS ADD CHANGE DETECTION LOGIC HERE  
      -- =============================================================================
      
      -- AGENT: Handle record updates - check for specific field changes
      IF old_rec_.important_field != new_rec_.important_field THEN
         -- AGENT: Important field changed - trigger business logic
         Update_Related_Entities___(new_rec_);        -- AGENT: Update related data
         Send_Notification___(new_rec_, 'FIELD_CHANGED'); -- AGENT: Send notifications
      END IF;
      
      -- AGENT: Add more field change checks here
      -- IF old_rec_.status != new_rec_.status THEN
      --    Handle_Status_Change___(old_rec_, new_rec_);
      -- END IF;
      
   ELSIF action_ = 'DELETE' THEN
      -- =============================================================================
      -- DELETE HANDLING - AGENTS ADD DELETION LOGIC HERE
      -- =============================================================================
      
      -- AGENT: Handle record deletion
      Check_Delete_Restrictions___(old_rec_);         -- AGENT: Validate deletion is allowed
      Clean_Related_Data___(old_rec_);               -- AGENT: Clean up related records
      -- AGENT: Add archival, audit logging, etc.
      
   END IF;
   
   -- =============================================================================
   -- COMMON TRIGGER LOGIC - AGENTS ADD LOGIC FOR ALL OPERATIONS HERE
   -- =============================================================================
   
   -- AGENT: Add logic that runs for all trigger events
   -- Log_Entity_Change___(old_rec_, new_rec_, action_);
   
END Handle_{entity_name}_Change___;""",
            }

            template = templates.get(rule_type, templates["validation"])

            return f"""Business Logic Template Generated:

**Entity:** {entity_name}
**Rule Type:** {rule_type.title()}
**Business Rule:** {business_rule}

```sql
{template}
```

**Implementation Notes:**
1. Add this code to the {entity_name}_API package body
2. Call from appropriate trigger points (Insert___, Update___, Delete___)
3. Test thoroughly with various scenarios
4. Add proper error handling and logging
5. Consider performance implications for large datasets"""

        @self.mcp.tool()
        async def get_ifs_architecture_recommendations(
            use_case: str,
            complexity: str = "medium",
            integration_needs: Optional[str] = None,
        ) -> str:
            """Get IFS Cloud architecture recommendations for specific use cases.

            Args:
                use_case: Description of the use case or requirement
                complexity: Complexity level (simple, medium, complex)
                integration_needs: External integration requirements
            """

            recommendations = {
                "simple": {
                    "description": "Single entity with basic CRUD operations",
                    "components": ["Entity definition", "Basic API", "Simple views"],
                    "patterns": ["Single table", "Basic validation", "Standard CRUD"],
                },
                "medium": {
                    "description": "Multi-entity with business logic and workflows",
                    "components": [
                        "Master-detail entities",
                        "Business APIs",
                        "Projection/Client",
                        "State machines",
                    ],
                    "patterns": [
                        "Master-detail",
                        "State management",
                        "Business validation",
                        "Event handling",
                    ],
                },
                "complex": {
                    "description": "Full enterprise solution with integrations",
                    "components": [
                        "Entity hierarchy",
                        "Complex APIs",
                        "Integration points",
                        "Custom components",
                        "Reports",
                    ],
                    "patterns": [
                        "Microservices",
                        "Event sourcing",
                        "CQRS",
                        "Integration patterns",
                        "Performance optimization",
                    ],
                },
            }

            selected_complexity = recommendations.get(
                complexity, recommendations["medium"]
            )

            response_lines = [
                f"IFS Cloud Architecture Recommendations",
                "=" * 45,
                "",
                f"Use Case: {use_case}",
                f"Complexity Level: {complexity.title()}",
                "",
            ]

            if integration_needs:
                response_lines.extend([f"Integration Needs: {integration_needs}", ""])

            response_lines.extend(
                [
                    "Architecture Overview:",
                    f"  {selected_complexity['description']}",
                    "",
                    "Required Components:",
                ]
            )

            for component in selected_complexity["components"]:
                response_lines.append(f"  • {component}")

            response_lines.extend(
                [
                    "",
                    "Recommended Patterns:",
                ]
            )

            for pattern in selected_complexity["patterns"]:
                response_lines.append(f"  • {pattern}")

            # Add specific recommendations based on use case keywords
            use_case_lower = use_case.lower()

            response_lines.extend(
                [
                    "",
                    "Specific Recommendations:",
                ]
            )

            if "order" in use_case_lower or "sales" in use_case_lower:
                response_lines.extend(
                    [
                        "  • Implement order state machine (Planned → Released → Completed)",
                        "  • Use master-detail pattern for order headers and lines",
                        "  • Add pricing and discount calculations",
                        "  • Integrate with inventory for availability checks",
                    ]
                )
            elif "inventory" in use_case_lower or "stock" in use_case_lower:
                response_lines.extend(
                    [
                        "  • Implement inventory tracking with transactions",
                        "  • Use location-based inventory management",
                        "  • Add serial/lot number tracking if needed",
                        "  • Implement reservation mechanisms",
                    ]
                )
            elif "customer" in use_case_lower or "contact" in use_case_lower:
                response_lines.extend(
                    [
                        "  • Design customer hierarchy (companies, contacts, addresses)",
                        "  • Implement customer classification and segmentation",
                        "  • Add communication preferences and history",
                        "  • Consider GDPR compliance requirements",
                    ]
                )
            else:
                response_lines.extend(
                    [
                        "  • Follow IFS naming conventions and standards",
                        "  • Implement proper audit trail (created_by, created_date)",
                        "  • Add appropriate indexes for performance",
                        "  • Design with scalability in mind",
                    ]
                )

            if integration_needs:
                response_lines.extend(
                    [
                        "",
                        "Integration Architecture:",
                        "  • Use IFS Connect for external system integration",
                        "  • Implement REST APIs for real-time integration",
                        "  • Consider event-driven architecture for loose coupling",
                        "  • Plan for data synchronization and conflict resolution",
                    ]
                )

            response_lines.extend(
                [
                    "",
                    "Best Practices:",
                    "  • Start with core entities and expand incrementally",
                    "  • Design APIs first, then implement storage",
                    "  • Use IFS standard patterns and conventions",
                    "  • Plan for internationalization and localization",
                    "  • Implement proper security and authorization",
                    "  • Design for testability and maintainability",
                ]
            )

            return "\\n".join(response_lines)

        @self.mcp.tool()
        async def analyze_codebase_patterns(
            pattern_type: str = "all",
            limit: int = 20,
        ) -> str:
            """Analyze existing codebase patterns to learn IFS Cloud conventions.

            **AGENT INSTRUCTIONS:**
            This tool analyzes real IFS Cloud production code to extract patterns and conventions.

            **When to use:**
            - Learning IFS Cloud naming conventions
            - Understanding code organization patterns
            - Finding examples of how to structure new code
            - Discovering API patterns and best practices

            **Pattern types:**
            - "naming": Analyzes naming conventions for entities, APIs, modules
            - "structure": Examines file organization and architecture patterns
            - "api": Studies API design patterns and method conventions
            - "all": Comprehensive analysis of all pattern types

            **What you'll learn:**
            - Entity naming: CustomerOrder, ProductItem, WorkOrderHeader
            - API patterns: Entity_API, Logical_Unit_Util_API
            - Module structure: How ORDER module organizes entities/APIs
            - File types: Which files belong together (.entity, .plsql, .client)

            **Examples:**
            - Study naming: analyze_codebase_patterns("naming", 30)
            - API patterns: analyze_codebase_patterns("api", 15)
            - Full analysis: analyze_codebase_patterns("all", 50)

            **Output:** Detailed analysis of patterns with specific examples from production code

            Args:
                pattern_type: Type of patterns to analyze (naming, structure, api, all)
                limit: Maximum examples to show
            """

            # Get all files from index
            results = self.indexer.search_deduplicated(query="*", limit=limit * 3)

            patterns = {
                "naming": {
                    "entities": [],
                    "apis": [],
                    "modules": [],
                    "logical_units": [],
                },
                "structure": {"file_types": {}, "directories": set(), "modules": set()},
                "api": {
                    "crud_patterns": [],
                    "validation_patterns": [],
                    "business_logic_patterns": [],
                },
            }

            for result in results:
                # Analyze naming patterns
                if result.type == ".entity":
                    patterns["naming"]["entities"].append(result.name)
                elif result.type == ".plsql":
                    patterns["naming"]["apis"].append(result.name)

                if result.module:
                    patterns["structure"]["modules"].add(result.module)
                    patterns["naming"]["modules"].append(result.module)

                if result.logical_unit:
                    patterns["naming"]["logical_units"].append(result.logical_unit)

                # Count file types
                file_type = result.type
                patterns["structure"]["file_types"][file_type] = (
                    patterns["structure"]["file_types"].get(file_type, 0) + 1
                )

                # Extract API patterns from content
                content = getattr(result, "content", "").lower()
                if "procedure new" in content or "procedure modify" in content:
                    patterns["api"]["crud_patterns"].append(result.name)
                if "validate" in content or "check_" in content:
                    patterns["api"]["validation_patterns"].append(result.name)
                if "calculate" in content or "process_" in content:
                    patterns["api"]["business_logic_patterns"].append(result.name)

            response_lines = [f"IFS Cloud Codebase Pattern Analysis", "=" * 40, ""]

            if pattern_type in ["naming", "all"]:
                response_lines.extend(["NAMING PATTERNS:", "-" * 16, ""])

                if patterns["naming"]["entities"]:
                    response_lines.append("Entity Naming Examples:")
                    for entity in patterns["naming"]["entities"][:10]:
                        response_lines.append(f"  • {entity}")
                    response_lines.append("")

                if patterns["naming"]["apis"]:
                    response_lines.append("API Naming Examples:")
                    for api in patterns["naming"]["apis"][:10]:
                        response_lines.append(f"  • {api}")
                    response_lines.append("")

                if patterns["naming"]["modules"]:
                    modules = list(set(patterns["naming"]["modules"]))
                    response_lines.append(f"Modules Found ({len(modules)}):")
                    for module in modules[:10]:
                        response_lines.append(f"  • {module}")
                    response_lines.append("")

            if pattern_type in ["structure", "all"]:
                response_lines.extend(["STRUCTURE PATTERNS:", "-" * 18, ""])

                response_lines.append("File Type Distribution:")
                for file_type, count in sorted(
                    patterns["structure"]["file_types"].items()
                ):
                    response_lines.append(f"  • {file_type}: {count} files")
                response_lines.append("")

                if patterns["structure"]["modules"]:
                    response_lines.append(
                        f"Modules in Codebase ({len(patterns['structure']['modules'])}):"
                    )
                    for module in sorted(patterns["structure"]["modules"])[:15]:
                        response_lines.append(f"  • {module}")
                    response_lines.append("")

            if pattern_type in ["api", "all"]:
                response_lines.extend(["API PATTERNS:", "-" * 13, ""])

                if patterns["api"]["crud_patterns"]:
                    response_lines.append("Files with CRUD Patterns:")
                    for file in patterns["api"]["crud_patterns"][:10]:
                        response_lines.append(f"  • {file}")
                    response_lines.append("")

                if patterns["api"]["validation_patterns"]:
                    response_lines.append("Files with Validation Patterns:")
                    for file in patterns["api"]["validation_patterns"][:10]:
                        response_lines.append(f"  • {file}")
                    response_lines.append("")

                if patterns["api"]["business_logic_patterns"]:
                    response_lines.append("Files with Business Logic Patterns:")
                    for file in patterns["api"]["business_logic_patterns"][:10]:
                        response_lines.append(f"  • {file}")
                    response_lines.append("")

            # Add insights and recommendations
            response_lines.extend(["INSIGHTS & RECOMMENDATIONS:", "-" * 27, ""])

            # Naming convention insights
            if patterns["naming"]["entities"]:
                entity_names = patterns["naming"]["entities"]
                if any("_" in name for name in entity_names):
                    response_lines.append("• Entities use underscore naming convention")
                else:
                    response_lines.append("• Entities use CamelCase naming convention")

            # Module analysis
            if len(patterns["structure"]["modules"]) > 5:
                response_lines.append(
                    f"• Large codebase with {len(patterns['structure']['modules'])} modules - consider modular development"
                )
            elif len(patterns["structure"]["modules"]) < 3:
                response_lines.append(
                    "• Small focused codebase - good for targeted customizations"
                )

            # API pattern recommendations
            if len(patterns["api"]["crud_patterns"]) > len(
                patterns["api"]["validation_patterns"]
            ):
                response_lines.append(
                    "• Consider adding more validation patterns for data integrity"
                )

            return "\\n".join(response_lines)

        @self.mcp.tool()
        async def create_ifs_development_plan(
            project_name: str,
            requirements: str,
            timeline: str = "medium",
            team_size: str = "small",
        ) -> str:
            """Create a comprehensive development plan for IFS Cloud customization.

            Args:
                project_name: Name of the development project
                requirements: High-level requirements description
                timeline: Project timeline (short, medium, long)
                team_size: Team size (small, medium, large)
            """

            timelines = {
                "short": {"weeks": "2-6", "phases": 3},
                "medium": {"weeks": "8-16", "phases": 4},
                "long": {"weeks": "20-40", "phases": 5},
            }

            team_configs = {
                "small": {"developers": "1-2", "focus": "core functionality"},
                "medium": {"developers": "3-5", "focus": "full feature set"},
                "large": {"developers": "6+", "focus": "enterprise solution"},
            }

            timeline_info = timelines.get(timeline, timelines["medium"])
            team_info = team_configs.get(team_size, team_configs["small"])

            response_lines = [
                f"IFS Cloud Development Plan: {project_name}",
                "=" * (30 + len(project_name)),
                "",
                "PROJECT OVERVIEW:",
                f"  Timeline: {timeline_info['weeks']} weeks",
                f"  Team Size: {team_info['developers']} developers",
                f"  Focus: {team_info['focus']}",
                "",
                f"Requirements: {requirements}",
                "",
            ]

            # Phase planning
            response_lines.extend(["DEVELOPMENT PHASES:", "-" * 19, ""])

            phases = []
            if timeline_info["phases"] >= 3:
                phases.extend(
                    [
                        (
                            "Phase 1: Analysis & Design",
                            [
                                "Analyze existing codebase patterns",
                                "Design entity relationships",
                                "Define API specifications",
                                "Create database schema",
                                "Plan integration points",
                            ],
                        ),
                        (
                            "Phase 2: Core Development",
                            [
                                "Implement entity definitions",
                                "Create basic CRUD operations",
                                "Develop business logic APIs",
                                "Build validation rules",
                                "Create unit tests",
                            ],
                        ),
                        (
                            "Phase 3: Integration & Testing",
                            [
                                "Implement frontend views",
                                "Add integration points",
                                "Perform system testing",
                                "User acceptance testing",
                                "Performance optimization",
                            ],
                        ),
                    ]
                )

            if timeline_info["phases"] >= 4:
                phases.insert(
                    2,
                    (
                        "Phase 2.5: Business Logic",
                        [
                            "Implement complex business rules",
                            "Add state machines and workflows",
                            "Create calculation engines",
                            "Develop reporting features",
                            "Add advanced validations",
                        ],
                    ),
                )

            if timeline_info["phases"] >= 5:
                phases.append(
                    (
                        "Phase 4: Enterprise Features",
                        [
                            "Add security and authorization",
                            "Implement audit trails",
                            "Create integration adapters",
                            "Add monitoring and logging",
                            "Performance tuning",
                        ],
                    )
                )

            for phase_name, tasks in phases:
                response_lines.extend([phase_name, "-" * len(phase_name)])
                for task in tasks:
                    response_lines.append(f"  • {task}")
                response_lines.append("")

            # Deliverables
            response_lines.extend(["KEY DELIVERABLES:", "-" * 17, ""])

            deliverables = [
                "Entity definitions (.entity files)",
                "PL/SQL API packages (.plsql files)",
                "Database views and projections",
                "Frontend pages and components (.views files)",
                "Integration specifications",
                "Test suites and documentation",
                "Deployment scripts",
            ]

            if team_size in ["medium", "large"]:
                deliverables.extend(
                    [
                        "Performance monitoring setup",
                        "Security implementation",
                        "User training materials",
                    ]
                )

            for deliverable in deliverables:
                response_lines.append(f"  • {deliverable}")

            response_lines.extend(["", "RISK MITIGATION:", "-" * 16, ""])

            risks = [
                "Regular code reviews to maintain quality",
                "Continuous integration for early issue detection",
                "Prototype critical features early",
                "Maintain good communication with stakeholders",
                "Plan for IFS Cloud version compatibility",
            ]

            if timeline == "short":
                risks.append("Focus on MVP to meet tight deadline")
            elif timeline == "long":
                risks.append("Regular milestone reviews to prevent scope creep")

            for risk in risks:
                response_lines.append(f"  • {risk}")

            response_lines.extend(["", "SUCCESS METRICS:", "-" * 15, ""])

            metrics = [
                "All functional requirements implemented",
                "Performance meets IFS Cloud standards",
                "Code passes quality gates and reviews",
                "Integration tests pass successfully",
                "User acceptance criteria met",
                "Documentation complete and reviewed",
            ]

            for metric in metrics:
                response_lines.append(f"  • {metric}")

            response_lines.extend(["", "RECOMMENDED TOOLS:", "-" * 18, ""])

            tools = [
                "Use this MCP server for code generation, analysis, and production metadata insights",
                "IFS Developer Studio for development",
                "Version control (Git) for code management",
                "Continuous Integration pipeline",
                "Automated testing frameworks",
                "Code quality tools (SonarQube, etc.)",
            ]

            for tool in tools:
                response_lines.append(f"  • {tool}")

            return "\\n".join(response_lines)

        @self.mcp.tool()
        async def generate_projection_template(
            entity_name: str,
            projection_type: str = "standard",
            include_crud: bool = True,
            include_actions: bool = True,
        ) -> str:
            """Generate IFS Cloud projection template for frontend integration.

            **AGENT INSTRUCTIONS:**
            This tool creates IFS Cloud projection files (.projection) for frontend UI integration.

            **When to use:**
            - Creating backend projections for frontend pages
            - Exposing entity data to IFS Cloud clients
            - Setting up CRUD operations for UI components
            - Defining actions and functions for frontend consumption

            **Projection types:**
            - standard: Basic CRUD projection for single entity
            - masterdetail: Master-detail projection with nested entities
            - readonly: Read-only projection for reporting/viewing
            - custom: Custom projection with specific business logic

            **Examples:**
            - Basic projection: generate_projection_template("CustomerOrder")
            - Read-only view: generate_projection_template("OrderReport", "readonly", False, False)
            - Master-detail: generate_projection_template("CustomerOrder", "masterdetail")

            **Output:** Complete IFS Cloud projection file with CRUD operations and UI bindings

            Args:
                entity_name: Target entity name (e.g., 'CustomerOrder')
                projection_type: Type of projection (standard, masterdetail, readonly, custom)
                include_crud: Include Create/Read/Update/Delete operations
                include_actions: Include custom actions and functions
            """

            template = f"""-- =============================================================================
-- IFS CLOUD PROJECTION TEMPLATE for {entity_name}
-- Type: {projection_type.title()} Projection
-- AGENT INSTRUCTIONS:
-- 1. PROJECTION HEADER: Define projection name and version
-- 2. ENTITY SETS: Map database entities to projection entities
-- 3. CRUD OPERATIONS: Define Create/Read/Update/Delete operations
-- 4. ACTIONS: Add custom business actions
-- 5. FUNCTIONS: Add data retrieval functions
-- =============================================================================

projection {entity_name}Handling version "24.2.0";

-- =============================================================================
-- PROJECTION DESCRIPTION SECTION - AGENTS UPDATE DESCRIPTION HERE
-- =============================================================================
description = "{entity_name} management and handling operations";

-- =============================================================================
-- ENTITY SETS SECTION - AGENTS DEFINE DATA ENTITIES HERE
-- =============================================================================

-- AGENT: Main entity set - maps to database table/view
entityset {entity_name}Set for {entity_name} {{
   -- AGENT: Define entity attributes from database
   attribute Id Text;                    -- AGENT: Primary key field
   attribute Description Text;           -- AGENT: Description field
   attribute State Enumeration(State);   -- AGENT: State field for workflow
   attribute CreatedDate Timestamp;      -- AGENT: Created timestamp
   attribute CreatedBy Text;             -- AGENT: Created by user
   
   -- AGENT: Add more attributes based on entity structure
   -- attribute CustomField Text;
   -- attribute Amount Number;
   -- attribute Date Date;
   
   -- =============================================================================
   -- ENTITY REFERENCES SECTION - AGENTS ADD FOREIGN KEY REFERENCES HERE
   -- =============================================================================
   
   -- AGENT: Add references to related entities
   -- reference CustomerId to Customer(CustomerId) {{
   --    label = "Customer";
   -- }}
   
   -- =============================================================================
   -- ENTITY ARRAYS SECTION - AGENTS ADD CHILD/DETAIL COLLECTIONS HERE
   -- ============================================================================="""

            if projection_type == "masterdetail":
                template += f"""
   
   -- AGENT: Master-detail arrays for child entities
   array {entity_name}Lines({entity_name}Id) to {entity_name}Line {{
      -- AGENT: Define detail/line entity attributes
      attribute LineNo Number;
      attribute ItemId Text;
      attribute Quantity Number;
      attribute UnitPrice Number;
      -- AGENT: Add more line attributes as needed
   }}
"""

            template += f"""
}}

-- =============================================================================
-- ACTIONS SECTION - AGENTS ADD BUSINESS ACTIONS HERE
-- ============================================================================="""

            if include_actions:
                template += f"""

-- AGENT: Custom business actions that modify data
action Create{entity_name} {{
   parameter Description Text;           -- AGENT: Input parameters
   parameter CustomerId Text;           -- AGENT: Add parameters as needed
   -- AGENT: Add more input parameters here
}} returns {entity_name}Set;

action Update{entity_name}State {{
   parameter Id Text;                   -- AGENT: Record identifier
   parameter NewState Enumeration(State); -- AGENT: Target state
   parameter Reason Text;               -- AGENT: Reason for state change
}} returns {entity_name}Set;

-- AGENT: Add more custom actions here
-- action Process{entity_name} {{ ... }}
-- action Cancel{entity_name} {{ ... }}
"""

            template += f"""

-- =============================================================================
-- FUNCTIONS SECTION - AGENTS ADD DATA RETRIEVAL FUNCTIONS HERE
-- ============================================================================="""

            if include_crud:
                template += f"""

-- AGENT: Data retrieval functions (read-only operations)
function Get{entity_name}Details {{
   parameter Id Text;                   -- AGENT: Record identifier
}} returns {entity_name}Set;

function Search{entity_name}s {{
   parameter SearchText Text;           -- AGENT: Search criteria
   parameter State Enumeration(State);  -- AGENT: Filter by state
   parameter CreatedFrom Date;          -- AGENT: Date range filters
   parameter CreatedTo Date;
}} returns {entity_name}Set;

-- AGENT: Add more query functions here
-- function Calculate{entity_name}Total {{ ... }}
-- function Validate{entity_name}Data {{ ... }}
"""

            template += f"""

-- =============================================================================
-- ENUMERATIONS SECTION - AGENTS DEFINE ENUMERATION VALUES HERE
-- =============================================================================

-- AGENT: Define enumeration for state values
enumeration State {{
   value "Planned" {{
      identifier = "Planned";
      label = "Planned";
   }}
   value "Active" {{
      identifier = "Active"; 
      label = "Active";
   }}
   value "Completed" {{
      identifier = "Completed";
      label = "Completed";
   }}
   value "Cancelled" {{
      identifier = "Cancelled";
      label = "Cancelled";
   }}
   -- AGENT: Add more state values as needed
}}

-- AGENT: Add more enumerations here as needed
-- enumeration Priority {{ ... }}
-- enumeration Type {{ ... }}

-- =============================================================================
-- END OF PROJECTION
-- ============================================================================="""

            return f"Generated IFS Cloud Projection Template:\\n\\n**Entity:** {entity_name}\\n**Type:** {projection_type.title()}\\n**Include CRUD:** {include_crud}\\n**Include Actions:** {include_actions}\\n\\n```projection\\n{template}\\n```\\n\\n**Implementation Notes:**\\n1. Save as {entity_name}Handling.projection\\n2. Update entity attributes to match database schema\\n3. Add proper references to related entities\\n4. Test CRUD operations thoroughly\\n5. Validate enumeration values match business requirements"

        @self.mcp.tool()
        async def generate_client_template(
            entity_name: str,
            client_type: str = "standard",
            include_pages: bool = True,
            include_lists: bool = True,
        ) -> str:
            """Generate IFS Cloud client template for frontend UI definition.

            **AGENT INSTRUCTIONS:**
            This tool creates IFS Cloud client files (.client) for frontend UI components.

            **When to use:**
            - Creating frontend UI definitions
            - Defining pages, lists, and navigation
            - Setting up user interface layouts
            - Integrating with projection data sources

            **Client types:**
            - standard: Basic client with pages and lists
            - navigator: Navigation-focused client with menu structure
            - workspace: Workspace client with multiple tabs/views
            - assistant: Assistant/wizard style client

            **Examples:**
            - Basic client: generate_client_template("CustomerOrder")
            - Navigator: generate_client_template("OrderManagement", "navigator")
            - Workspace: generate_client_template("CustomerOrder", "workspace")

            **Output:** Complete IFS Cloud client file with UI components and navigation

            Args:
                entity_name: Target entity name (e.g., 'CustomerOrder')
                client_type: Type of client (standard, navigator, workspace, assistant)
                include_pages: Include page definitions
                include_lists: Include list/grid definitions
            """

            template = f"""-- =============================================================================
-- IFS CLOUD CLIENT TEMPLATE for {entity_name}
-- Type: {client_type.title()} Client
-- AGENT INSTRUCTIONS:
-- 1. CLIENT HEADER: Define client name and projection binding
-- 2. NAVIGATION: Set up menu and navigation structure  
-- 3. PAGES: Define form pages for data entry/viewing
-- 4. LISTS: Define grids/lists for data browsing
-- 5. COMMANDS: Add toolbar commands and actions
-- =============================================================================

client {entity_name}Handling using {entity_name}HandlingProjection;

-- =============================================================================
-- CLIENT DESCRIPTION SECTION - AGENTS UPDATE DESCRIPTION HERE
-- =============================================================================
description = "{entity_name} management client interface";

-- =============================================================================
-- NAVIGATION SECTION - AGENTS DEFINE MENU STRUCTURE HERE
-- ============================================================================="""

            if client_type == "navigator":
                template += f"""

-- AGENT: Navigator structure for menu-based navigation
navigator {{
   entry {entity_name}NavigatorEntry parent EnterprisePlatformNavigator.OrderMgtNavigator at index 100 {{
      label = "{entity_name} Management";
      
      -- AGENT: Add navigation entries for different views
      entry {entity_name}ListEntry {{
         label = "{entity_name} Overview";
         page List;
      }}
      
      entry New{entity_name}Entry {{
         label = "Create {entity_name}";
         page {entity_name}Page/New;
      }}
      
      -- AGENT: Add more navigation entries as needed
   }}
}}"""

            template += f"""

-- =============================================================================
-- PAGES SECTION - AGENTS DEFINE FORM PAGES HERE
-- ============================================================================="""

            if include_pages:
                template += f"""

-- AGENT: Main entity page for viewing/editing records
page {entity_name}Page using {entity_name}Set {{
   label = "{entity_name}";
   
   -- =============================================================================
   -- PAGE SELECTOR SECTION - AGENTS DEFINE SELECTION CRITERIA HERE
   -- =============================================================================
   selector {entity_name}Selector {{
      -- AGENT: Key fields for record selection
      field Id;
      field Description;
   }}
   
   -- =============================================================================
   -- PAGE GROUPS SECTION - AGENTS ORGANIZE FIELDS INTO LOGICAL GROUPS HERE
   -- =============================================================================
   
   -- AGENT: General information group
   group General{{
      label = "General Information";
      
      field Id {{
         size = Small;
         editable = [ETag == null];  -- AGENT: Only editable when creating new
      }}
      field Description {{
         size = Large;
      }}
      field State {{
         size = Small;
         editable = [false];  -- AGENT: State managed by workflow
      }}
   }}
   
   -- AGENT: Dates and audit information group
   group AuditInfo {{
      label = "Audit Information";
      
      field CreatedDate {{
         size = Small;
         editable = [false];
      }}
      field CreatedBy {{
         size = Small;
         editable = [false];
      }}
   }}
   
   -- AGENT: Add more groups for organizing fields
   -- group AdditionalInfo {{ ... }}
   -- group FinancialInfo {{ ... }}
   
   -- =============================================================================
   -- PAGE COMMANDS SECTION - AGENTS ADD TOOLBAR COMMANDS HERE
   -- =============================================================================
   
   command Create{entity_name}Command {{
      label = "Create";
      enabled = [true];
      execute {{
         call Create{entity_name}(Description, CustomerId) {{
            when Success {{
               navigate back;
            }}
         }}
      }}
   }}
   
   command Update{entity_name}StateCommand {{
      label = "Update State";
      enabled = [State != "Completed"];
      execute {{
         call Update{entity_name}State(Id, NewState, Reason);
      }}
   }}
   
   -- AGENT: Add more commands as needed
}}"""

            template += f"""

-- =============================================================================
-- LISTS SECTION - AGENTS DEFINE DATA GRIDS/LISTS HERE
-- ============================================================================="""

            if include_lists:
                template += f"""

-- AGENT: Main list for browsing multiple records
page List using {entity_name}Set {{
   label = "{entity_name} List";
   
   -- =============================================================================
   -- LIST STRUCTURE SECTION - AGENTS DEFINE COLUMNS AND LAYOUT HERE
   -- =============================================================================
   
   list {entity_name}List {{
      -- AGENT: Define list columns (fields to display)
      field Id {{
         size = Small;
      }}
      field Description {{
         size = Large;
      }}
      field State {{
         size = Small;
      }}
      field CreatedDate {{
         size = Small;
      }}
      field CreatedBy {{
         size = Small;
      }}
      
      -- AGENT: Add more display fields as needed
      
      -- =============================================================================
      -- LIST COMMANDS SECTION - AGENTS ADD ROW-LEVEL COMMANDS HERE
      -- =============================================================================
      
      command {entity_name}DetailsCommand {{
         label = "Details";
         mode = SingleRecord;
         execute {{
            navigate "{entity_name}Page/Form?$filter=Id eq '${{Id}}'";
         }}
      }}
      
      command Update{entity_name}StateCommand {{
         label = "Update State";
         mode = SelectedRecords;
         enabled = [State != "Completed"];
         execute {{
            call Update{entity_name}State(Id, NewState, Reason);
         }}
      }}
      
      -- AGENT: Add more row commands as needed
   }}
   
   -- =============================================================================
   -- PAGE COMMANDS SECTION - AGENTS ADD PAGE-LEVEL COMMANDS HERE
   -- =============================================================================
   
   command New{entity_name}Command {{
      label = "New";
      execute {{
         navigate "{entity_name}Page/New";
      }}
   }}
   
   command Refresh{entity_name}ListCommand {{
      label = "Refresh";
      execute {{
         refresh;
      }}
   }}
   
   -- AGENT: Add more page commands as needed
}}"""

            template += f"""

-- =============================================================================
-- SEARCH CONTEXTS SECTION - AGENTS DEFINE SEARCH/FILTER CAPABILITIES HERE  
-- =============================================================================

searchcontext PageSearchContext for {entity_name}Set {{
   -- AGENT: Define searchable fields
   field Id;
   field Description;
   field State;
   field CreatedBy;
   
   -- AGENT: Add more searchable fields as needed
}}

-- =============================================================================
-- END OF CLIENT DEFINITION
-- ============================================================================="""

            return f"Generated IFS Cloud Client Template:\\n\\n**Entity:** {entity_name}\\n**Type:** {client_type.title()}\\n**Include Pages:** {include_pages}\\n**Include Lists:** {include_lists}\\n\\n```client\\n{template}\\n```\\n\\n**Implementation Notes:**\\n1. Save as {entity_name}Client.client\\n2. Ensure projection name matches exactly\\n3. Customize field sizes and editability rules\\n4. Add proper command permissions and security\\n5. Test navigation and page transitions\\n6. Validate search functionality works correctly"

        @self.mcp.tool()
        async def generate_fragment_template(
            fragment_name: str,
            fragment_type: str = "data",
            target_entity: str = "",
        ) -> str:
            """Generate IFS Cloud fragment template for reusable UI components.

            **AGENT INSTRUCTIONS:**
            This tool creates IFS Cloud fragment files (.fragment) for reusable UI components.

            **When to use:**
            - Creating reusable UI components
            - Sharing common functionality across multiple clients
            - Building modular user interface elements
            - Creating consistent UI patterns

            **Fragment types:**
            - data: Data-focused fragment with entity binding
            - assistant: Wizard/assistant style fragment
            - selector: Selection dialog fragment
            - command: Reusable command fragment

            **Examples:**
            - Data fragment: generate_fragment_template("CustomerDetails", "data", "Customer")
            - Assistant: generate_fragment_template("OrderWizard", "assistant", "CustomerOrder")
            - Selector: generate_fragment_template("ProductSelector", "selector", "Product")

            **Output:** Complete IFS Cloud fragment file with reusable UI components

            Args:
                fragment_name: Name of the fragment (e.g., 'CustomerDetails')
                fragment_type: Type of fragment (data, assistant, selector, command)
                target_entity: Target entity name if applicable
            """

            template = f"""-- =============================================================================
-- IFS CLOUD FRAGMENT TEMPLATE for {fragment_name}
-- Type: {fragment_type.title()} Fragment
-- Target Entity: {target_entity if target_entity else 'N/A'}
-- AGENT INSTRUCTIONS:
-- 1. FRAGMENT HEADER: Define fragment name and parameters
-- 2. GROUPS/SECTIONS: Organize UI elements into logical groups
-- 3. FIELDS: Define input/display fields
-- 4. COMMANDS: Add fragment-specific commands
-- 5. REUSABILITY: Ensure fragment can be used in multiple contexts
-- =============================================================================

fragment {fragment_name};

-- =============================================================================
-- FRAGMENT DESCRIPTION SECTION - AGENTS UPDATE DESCRIPTION HERE
-- =============================================================================
description = "{fragment_name} reusable fragment component";"""

            if target_entity:
                template += f"""

-- =============================================================================
-- ENTITY BINDING SECTION - AGENTS BIND TO TARGET ENTITY HERE
-- =============================================================================
using {target_entity}Set;"""

            if fragment_type == "data":
                template += f"""

-- =============================================================================
-- DATA GROUP SECTION - AGENTS ORGANIZE DATA FIELDS HERE
-- =============================================================================

-- AGENT: Main data group for displaying/editing entity data
group {fragment_name}Group {{
   label = "{fragment_name}";
   
   -- =============================================================================
   -- FIELD DEFINITIONS - AGENTS ADD DATA FIELDS HERE
   -- =============================================================================
   
   -- AGENT: Primary fields for the entity
   field Id {{
      size = Small;
      editable = [ETag == null];  -- AGENT: Editable only when creating
   }}
   
   field Description {{
      size = Large;
      required = [true];           -- AGENT: Mark required fields
   }}
   
   field State {{
      size = Small;
      editable = [false];          -- AGENT: Read-only state field
   }}
   
   -- AGENT: Add more fields based on entity structure
   -- field CustomField {{
   --    size = Medium;
   --    required = [condition];
   -- }}
   
   -- =============================================================================
   -- REFERENCE FIELDS - AGENTS ADD LOOKUP/REFERENCE FIELDS HERE
   -- =============================================================================
   
   -- AGENT: Add reference fields for related entities
   -- lov CustomerRef with Customer {{
   --    label = "Customer";
   --    description = CustomerRef.Name;
   -- }}
}}"""

            elif fragment_type == "assistant":
                template += f"""

-- =============================================================================
-- ASSISTANT STRUCTURE SECTION - AGENTS DEFINE WIZARD STEPS HERE
-- =============================================================================

-- AGENT: Assistant/wizard component for guided data entry
assistant {fragment_name}Assistant {{
   label = "{fragment_name} Assistant";
   
   -- =============================================================================
   -- STEP DEFINITIONS - AGENTS ADD WIZARD STEPS HERE
   -- =============================================================================
   
   -- AGENT: Step 1 - Basic information
   step BasicInfo {{
      label = "Basic Information";
      
      group BasicInfoGroup {{
         field Description {{
            size = Large;
            required = [true];
         }}
         
         -- AGENT: Add basic fields for step 1
      }}
   }}
   
   -- AGENT: Step 2 - Detailed information  
   step DetailedInfo {{
      label = "Detailed Information";
      
      group DetailedInfoGroup {{
         -- AGENT: Add detailed fields for step 2
         field AdditionalInfo {{
            size = Large;
         }}
      }}
   }}
   
   -- AGENT: Add more steps as needed
   -- step ReviewAndConfirm {{ ... }}
   
   -- =============================================================================
   -- ASSISTANT COMMANDS - AGENTS ADD NAVIGATION COMMANDS HERE
   -- =============================================================================
   
   command NextStepCommand {{
      label = "Next";
      execute {{
         -- AGENT: Add validation and navigation logic
      }}
   }}
   
   command PreviousStepCommand {{
      label = "Previous";
      execute {{
         -- AGENT: Add navigation logic
      }}
   }}
   
   command FinishCommand {{
      label = "Finish";
      execute {{
         -- AGENT: Add completion logic
      }}
   }}
}}"""

            elif fragment_type == "selector":
                template += f"""

-- =============================================================================
-- SELECTOR STRUCTURE SECTION - AGENTS DEFINE SELECTION UI HERE
-- =============================================================================

-- AGENT: Selector component for choosing records
selector {fragment_name}Selector {{
   label = "Select {fragment_name}";
   
   -- =============================================================================
   -- SELECTION LIST - AGENTS DEFINE SELECTABLE ITEMS HERE
   -- =============================================================================
   
   list {fragment_name}List {{
      -- AGENT: Define columns for selection
      field Id {{
         size = Small;
      }}
      field Description {{
         size = Large;
      }}
      
      -- AGENT: Add more display fields
      
      -- =============================================================================
      -- SELECTION COMMANDS - AGENTS ADD SELECTION ACTIONS HERE
      -- =============================================================================
      
      command SelectCommand {{
         label = "Select";
         mode = SingleRecord;
         execute {{
            -- AGENT: Add selection logic
            exit Ok;
         }}
      }}
   }}
   
   -- =============================================================================
   -- SELECTOR COMMANDS - AGENTS ADD DIALOG COMMANDS HERE
   -- =============================================================================
   
   command OkCommand {{
      label = "OK";
      execute {{
         exit Ok;
      }}
   }}
   
   command CancelCommand {{
      label = "Cancel";
      execute {{
         exit Cancel;
      }}
   }}
}}"""

            template += f"""

-- =============================================================================
-- FRAGMENT PARAMETERS SECTION - AGENTS DEFINE INPUT PARAMETERS HERE
-- =============================================================================

-- AGENT: Define parameters that can be passed to the fragment
-- parameter EntityId Text;
-- parameter Mode Enumeration(FragmentMode);
-- parameter ReadOnly Boolean;

-- =============================================================================
-- END OF FRAGMENT DEFINITION
-- ============================================================================="""

            return f"Generated IFS Cloud Fragment Template:\\n\\n**Fragment:** {fragment_name}\\n**Type:** {fragment_type.title()}\\n**Target Entity:** {target_entity if target_entity else 'N/A'}\\n\\n```fragment\\n{template}\\n```\\n\\n**Implementation Notes:**\\n1. Save as {fragment_name}.fragment\\n2. Define clear parameters for reusability\\n3. Test fragment in multiple client contexts\\n4. Ensure proper data binding and validation\\n5. Add appropriate security and permission checks\\n6. Document fragment usage and integration points"

        @self.mcp.tool()
        async def analyze_plsql_file(content: str, strict_mode: bool = False) -> str:
            """Analyze IFS Cloud PLSQL file for business logic understanding.

            **AGENT INSTRUCTIONS:**
            Use this tool to deeply understand PLSQL business logic files (.plsql).

            **Perfect for:**
            - Understanding business validation patterns
            - Extracting public API methods and signatures
            - Finding error handling and exception patterns
            - Analyzing procedure/function organization
            - Understanding IFS Cloud architectural patterns

            **What you'll get:**
            - Complete AST structure of the PLSQL file
            - Public vs private method analysis
            - Business validation patterns and rules
            - Error handling mechanisms
            - Constants and type definitions
            - Logical unit and component information
            - Method signatures and parameters

            Args:
                content: PLSQL file content to analyze
                strict_mode: If True, applies stricter validation (default: False)

            Returns:
                Comprehensive analysis of PLSQL structure and business logic
            """
            try:
                result = self.plsql_analyzer.analyze(content)

                lines = []
                lines.append("🔍 **PLSQL Business Logic Analysis**")
                lines.append("=" * 50)
                lines.append("")

                # Basic validity
                lines.append(
                    f"✅ **Analysis Result:** {'Valid' if result.is_valid else 'Issues Found'}"
                )
                lines.append(f"🔍 **Errors:** {len(result.get_errors())}")
                lines.append(f"⚠️ **Warnings:** {len(result.get_warnings())}")
                lines.append("")

                # IFS Cloud metadata
                lines.append("🏗️ **IFS Cloud Structure:**")
                lines.append(
                    f"  • Logical Unit: {result.logical_unit or 'Not specified'}"
                )
                lines.append(f"  • Component: {result.component or 'Not specified'}")
                lines.append(f"  • Layer: {result.layer or 'Not specified'}")
                lines.append("")

                # Architecture overview
                lines.append("📊 **Architecture Overview:**")
                lines.append(f"  • Public Methods: {len(result.public_methods)}")
                lines.append(f"  • Private Methods: {len(result.private_methods)}")
                lines.append(f"  • Constants: {len(result.constants)}")
                lines.append(f"  • Type Definitions: {len(result.types)}")
                lines.append("")

                # Business logic analysis
                lines.append("🛡️ **Business Logic Analysis:**")
                lines.append(
                    f"  • Business Validations: {len(result.business_validations)}"
                )
                lines.append(f"  • Error Patterns: {len(result.error_patterns)}")
                lines.append("")

                # Public API
                if result.public_methods:
                    lines.append("🌟 **Public API Methods:**")
                    for method in result.public_methods[:10]:
                        lines.append(
                            f"  • {method['type'].title()}: **{method['name']}** (line {method['line']})"
                        )
                    if len(result.public_methods) > 10:
                        lines.append(
                            f"  ... and {len(result.public_methods) - 10} more"
                        )
                    lines.append("")

                # Business validations
                if result.business_validations:
                    lines.append("✅ **Business Validation Patterns:**")
                    validation_types = {}
                    for validation in result.business_validations:
                        vtype = validation["type"]
                        validation_types[vtype] = validation_types.get(vtype, 0) + 1

                    for vtype, count in validation_types.items():
                        lines.append(
                            f"  • {vtype.replace('_', ' ').title()}: {count} validations"
                        )

                    lines.append("")
                    lines.append("📝 **Sample Validations:**")
                    for validation in result.business_validations[:3]:
                        lines.append(
                            f"  • **{validation['error_method']}** ({validation['type']}) - Line {validation['line']}"
                        )
                        pattern = (
                            validation["pattern"][:80] + "..."
                            if len(validation["pattern"]) > 80
                            else validation["pattern"]
                        )
                        lines.append(f"    ↳ `{pattern}`")
                    lines.append("")

                # Constants for business logic
                if result.constants:
                    lines.append("🔤 **Business Constants:**")
                    for const in result.constants[:8]:
                        lines.append(
                            f"  • **{const['name']}**: {const['type']} (line {const['line']})"
                        )
                    if len(result.constants) > 8:
                        lines.append(f"  ... and {len(result.constants) - 8} more")
                    lines.append("")

                # AST structure
                if result.ast:
                    lines.append("🌳 **Code Structure (AST):**")
                    node_counts = {}
                    for child in result.ast.children:
                        node_type = child.node_type.value
                        node_counts[node_type] = node_counts.get(node_type, 0) + 1

                    for node_type, count in sorted(node_counts.items()):
                        lines.append(
                            f"  • {node_type.replace('_', ' ').title()}: {count}"
                        )
                    lines.append("")

                # Diagnostics
                if result.diagnostics:
                    lines.append("📋 **Issues & Recommendations:**")
                    for diag in result.diagnostics[:5]:
                        icon = {
                            "error": "❌",
                            "warning": "⚠️",
                            "info": "ℹ️",
                            "hint": "💡",
                        }[diag.severity.value]
                        lines.append(f"  {icon} Line {diag.line}: {diag.message}")
                        if diag.fix_suggestion:
                            lines.append(f"    💡 *Fix: {diag.fix_suggestion}*")
                    if len(result.diagnostics) > 5:
                        lines.append(f"  ... and {len(result.diagnostics) - 5} more")
                    lines.append("")

                lines.append("🎯 **AI Agent Insights:**")
                lines.append("This PLSQL file provides business logic for:")
                if result.logical_unit:
                    lines.append(f"• **{result.logical_unit}** logical unit operations")
                if result.public_methods:
                    lines.append(
                        f"• **{len(result.public_methods)} public APIs** for external integration"
                    )
                if result.business_validations:
                    lines.append(
                        f"• **{len(result.business_validations)} validation rules** ensuring data integrity"
                    )
                lines.append("• Business process automation and workflow management")
                lines.append(
                    "• Error handling and exception management for robust operations"
                )

                return "\\n".join(lines)

            except Exception as e:
                return f"❌ **Error analyzing PLSQL file:** {str(e)}"

        @self.mcp.tool()
        async def analyze_fragment_file(content: str, strict_mode: bool = False) -> str:
            """Analyze IFS Cloud Fragment file for comprehensive UI understanding.

            **AGENT INSTRUCTIONS:**
            Use this tool to understand Fragment files (.fragment) that combine client and projection logic.

            **Perfect for:**
            - Understanding complete UI component structure
            - Analyzing client-server integration patterns
            - Finding UI validation and business logic connections
            - Understanding fragment composition and reusability
            - Extracting navigation and command patterns

            **What you'll get:**
            - Complete AST with both client and projection sections
            - Header information (component, layer, dependencies)
            - Client elements (pages, navigators, commands, lists)
            - Projection elements (entities, queries, actions)
            - Section boundaries and organization
            - Zero false positives through conservative analysis

            Args:
                content: Fragment file content to analyze
                strict_mode: If True, applies stricter validation (default: False)

            Returns:
                Comprehensive analysis of fragment structure and components
            """
            try:
                result = self.fragment_analyzer.analyze(content)

                lines = []
                lines.append("🌟 **Fragment File Analysis**")
                lines.append("=" * 45)
                lines.append("")

                # Validity and structure
                lines.append(
                    f"✅ **Analysis Result:** {'Valid Structure' if result.get('valid', False) else 'Issues Found'}"
                )
                lines.append(f"🔍 **Errors:** {len(result.get('errors', []))}")
                lines.append(f"⚠️ **Warnings:** {len(result.get('warnings', []))}")
                lines.append("")

                # Fragment information
                fragment_info = result.get("fragment_info", {})
                lines.append("🏗️ **Fragment Structure:**")
                lines.append(
                    f"  • Fragment Name: {fragment_info.get('name', 'Not specified')}"
                )
                lines.append(
                    f"  • Component: {fragment_info.get('component', 'Not specified')}"
                )
                lines.append(
                    f"  • Layer: {fragment_info.get('layer', 'Not specified')}"
                )
                if fragment_info.get("description"):
                    lines.append(f"  • Description: {fragment_info['description']}")
                lines.append("")

                # Section detection
                sections = result.get("sections", {})
                lines.append("📋 **Section Structure:**")
                lines.append(
                    f"  • Header Section: {'✅' if sections.get('header', False) else '❌'}"
                )
                lines.append(
                    f"  • Client Fragments: {'✅' if sections.get('client_fragments', False) else '❌'}"
                )
                lines.append(
                    f"  • Projection Fragments: {'✅' if sections.get('projection_fragments', False) else '❌'}"
                )
                lines.append("")

                # AST analysis
                ast = result.get("ast")
                if ast:
                    lines.append("🌳 **AST Structure Analysis:**")

                    def count_nodes_by_type(node, counts=None):
                        if counts is None:
                            counts = {}

                        node_type = node.get("node_type", "unknown")
                        counts[node_type] = counts.get(node_type, 0) + 1

                        for child in node.get("children", []):
                            count_nodes_by_type(child, counts)

                        return counts

                    node_counts = count_nodes_by_type(ast)
                    total_nodes = sum(node_counts.values())
                    lines.append(f"  • Total Nodes: {total_nodes}")

                    # Show node type distribution
                    for node_type, count in sorted(node_counts.items()):
                        if node_type != "fragment_file":  # Skip root
                            icon = {
                                "header_section": "🔖",
                                "client_fragments_section": "🖥️",
                                "projection_fragments_section": "🗃️",
                                "fragment_declaration": "📋",
                                "component_declaration": "🏢",
                                "layer_declaration": "🏗️",
                                "page_declaration": "📄",
                                "navigator_section": "🧭",
                                "command_declaration": "⚡",
                                "entity": "🏛️",
                                "query": "🔍",
                                "action": "⚙️",
                                "function": "🔧",
                            }.get(node_type, "📌")
                            lines.append(
                                f"  {icon} {node_type.replace('_', ' ').title()}: {count}"
                            )
                    lines.append("")

                # Client section details
                client_elements = self._extract_client_elements_from_ast(ast)
                if client_elements:
                    lines.append("🖥️ **Client Fragment Elements:**")
                    for element_type, elements in client_elements.items():
                        if elements:
                            lines.append(
                                f"  • {element_type.title()}: {len(elements)} found"
                            )
                            for element in elements[:3]:  # Show first 3
                                name = element.get("properties", {}).get(
                                    "name", "Unnamed"
                                )
                                line_num = element.get("start_line", "?")
                                lines.append(f"    - {name} (line {line_num})")
                            if len(elements) > 3:
                                lines.append(f"    ... and {len(elements) - 3} more")
                    lines.append("")

                # Projection section details
                projection_elements = self._extract_projection_elements_from_ast(ast)
                if projection_elements:
                    lines.append("🗃️ **Projection Fragment Elements:**")
                    for element_type, elements in projection_elements.items():
                        if elements:
                            lines.append(
                                f"  • {element_type.title()}: {len(elements)} found"
                            )
                            for element in elements[:3]:  # Show first 3
                                name = element.get("properties", {}).get(
                                    "name", "Unnamed"
                                )
                                line_num = element.get("start_line", "?")
                                lines.append(f"    - {name} (line {line_num})")
                            if len(elements) > 3:
                                lines.append(f"    ... and {len(elements) - 3} more")
                    lines.append("")

                # Dependencies and includes
                includes = fragment_info.get("includes", [])
                if includes:
                    lines.append("📦 **Dependencies:**")
                    for include in includes:
                        lines.append(f"  • {include}")
                    lines.append("")

                lines.append("🎯 **AI Agent Insights:**")
                lines.append("This fragment provides:")
                if client_elements:
                    lines.append("• **Client-side UI components** for user interaction")
                if projection_elements:
                    lines.append(
                        "• **Server-side data access** and business logic integration"
                    )
                lines.append("• **Full-stack component** combining UI and data layers")
                lines.append(
                    "• **Reusable fragment** for modular application architecture"
                )

                return "\\n".join(lines)

            except Exception as e:
                return f"❌ **Error analyzing Fragment file:** {str(e)}"

        @self.mcp.tool()
        async def intelligent_context_analysis(
            business_requirement: str,
            domain: Optional[str] = None,
            max_files_to_analyze: int = 15,
        ) -> str:
            """Intelligently analyze IFS Cloud codebase context for a business requirement.

            **🎯 AI AGENT PRIORITY TOOL - USE THIS FIRST!**

            This tool automatically:
            1. Searches for relevant existing implementations in production IFS Cloud metadata
            2. Analyzes found files with specialized PLSQL, Entity, and Fragment analyzers
            3. Identifies patterns, APIs, and best practices from real IFS Cloud production code
            4. Leverages comprehensive database metadata including Navigator GUI mappings
            5. Provides implementation context using 11,570+ Logical Units and production patterns

            **🗃️ Enhanced with Production Metadata:**
            - **Navigator Mappings**: GUI-to-backend entity mappings from FND_NAVIGATOR_ALL
            - **Domain Mappings**: 10,000+ domain value mappings for validation patterns
            - **Module Structure**: Complete module hierarchy with 152 production modules
            - **Entity Relationships**: Real production entity dependencies and associations
            - **View Patterns**: Production view implementations for UI integration

            **When to use:**
            - Before implementing ANY new feature (especially important now with production context)
            - When you receive a business requirement (now with real IFS patterns)
            - To understand existing patterns and approaches (production-validated)
            - To ensure consistency with IFS Cloud standards (real production standards)
            - To find GUI navigation patterns for new functionality

            **Examples:**
            - intelligent_context_analysis("Create customer order validation", "ORDER")
            - intelligent_context_analysis("Add pricing calculation", "FINANCE")
            - intelligent_context_analysis("Build user interface for products", "PRODUCT")
            - intelligent_context_analysis("Add new navigation menu item", "NAVIGATOR")

            Args:
                business_requirement: Description of what needs to be implemented
                domain: Optional domain/module hint (ORDER, FINANCE, PROJECT, etc.)
                max_files_to_analyze: Maximum files to analyze in depth (default: 15)
            """
            try:
                lines = []
                lines.append("🧠 **Intelligent Context Analysis**")
                lines.append("=" * 60)
                lines.append(f"📋 **Requirement:** {business_requirement}")
                if domain:
                    lines.append(f"🏢 **Domain:** {domain}")
                lines.append("")

                # Phase 1: Extract key terms and concepts
                lines.append("🔍 **Phase 1: Concept Extraction**")
                lines.append("-" * 40)

                # Extract keywords from business requirement
                keywords = self._extract_business_keywords(business_requirement)
                lines.append(f"📝 **Key Terms:** {', '.join(keywords)}")
                lines.append("")

                # Phase 2: Strategic searches for relevant files
                lines.append("🔎 **Phase 2: Strategic Discovery**")
                lines.append("-" * 40)

                relevant_files = []
                search_strategies = []

                # Strategy 1: Direct keyword searches
                for keyword in keywords[:3]:  # Top 3 keywords
                    query = keyword
                    if domain:
                        query += f" module:{domain.lower()}"

                    search_results = self.search_engine.search(
                        query=query, limit=5, include_related=True
                    )

                    if search_results:
                        search_strategies.append(
                            f"'{keyword}' → {len(search_results)} files"
                        )
                        relevant_files.extend(search_results[:3])

                # Strategy 2: Pattern-based searches
                pattern_searches = [
                    ("validation", "business validation patterns"),
                    ("calculation", "calculation and pricing logic"),
                    ("API", "public API implementations"),
                    ("Check_", "validation methods"),
                    ("Get_", "information retrieval methods"),
                    ("Create_", "creation and insertion logic"),
                ]

                for pattern, description in pattern_searches:
                    if pattern.lower() in business_requirement.lower():
                        query = pattern
                        if domain:
                            query += f" module:{domain.lower()}"

                        search_results = self.search_engine.search(
                            query=query, limit=3, file_type=".plsql"
                        )

                        if search_results:
                            search_strategies.append(
                                f"{description} → {len(search_results)} files"
                            )
                            relevant_files.extend(search_results[:2])

                # Strategy 3: Entity-based searches
                if domain:
                    entity_query = f"module:{domain.lower()}"
                    entity_results = self.search_engine.search(
                        query=entity_query, limit=5, file_type=".entity"
                    )
                    if entity_results:
                        search_strategies.append(
                            f"Domain entities → {len(entity_results)} files"
                        )
                        relevant_files.extend(entity_results[:2])

                lines.append("📊 **Search Strategies:**")
                for strategy in search_strategies[:8]:  # Limit output
                    lines.append(f"   • {strategy}")
                lines.append("")

                # Remove duplicates and limit files
                seen_paths = set()
                unique_files = []
                for file in relevant_files:
                    if file.file_path not in seen_paths:
                        seen_paths.add(file.file_path)
                        unique_files.append(file)
                        if len(unique_files) >= max_files_to_analyze:
                            break

                lines.append(f"📁 **Selected Files for Analysis:** {len(unique_files)}")
                lines.append("")

                # Phase 3: Deep analysis of selected files
                lines.append("🔬 **Phase 3: Deep Analysis**")
                lines.append("-" * 40)

                api_methods = []
                validation_patterns = []
                business_logic = []

                for i, file_result in enumerate(unique_files[:max_files_to_analyze], 1):
                    try:
                        file_path = Path(file_result.file_path)
                        file_ext = file_path.suffix.lower()

                        lines.append(f"📄 **File {i}: {file_path.name}**")
                        lines.append(
                            f"   Type: {file_ext} | Complexity: {file_result.complexity_score:.2f}"
                        )

                        if file_result.module:
                            lines.append(f"   Module: {file_result.module}")
                        if file_result.logical_unit:
                            lines.append(f"   Logical Unit: {file_result.logical_unit}")

                        # Read and analyze file based on type
                        content = file_path.read_text(encoding="utf-8")

                        if file_ext == ".plsql":
                            # PLSQL Analysis
                            plsql_result = self.plsql_analyzer.analyze(content)

                            lines.append(
                                f"   🔧 Methods: {len(plsql_result.public_methods)} public, {len(plsql_result.private_methods)} private"
                            )
                            lines.append(
                                f"   ✅ Validations: {len(plsql_result.business_validations)}"
                            )

                            # Extract patterns
                            api_methods.extend(
                                [
                                    f"{plsql_result.logical_unit}.{m['name']}"
                                    for m in plsql_result.public_methods
                                ]
                            )
                            validation_patterns.extend(
                                plsql_result.business_validations
                            )
                            business_logic.append(
                                {
                                    "file": file_path.name,
                                    "logical_unit": plsql_result.logical_unit,
                                    "component": plsql_result.component,
                                    "public_apis": len(plsql_result.public_methods),
                                    "validations": len(
                                        plsql_result.business_validations
                                    ),
                                }
                            )

                        elif file_ext == ".entity":
                            # Entity files - extract structure info
                            entity_info = self._extract_entity_info(content)
                            lines.append(
                                f"   📊 Entity: {entity_info.get('name', 'Unknown')}"
                            )
                            if entity_info.get("attributes"):
                                lines.append(
                                    f"   📋 Attributes: {len(entity_info['attributes'])}"
                                )

                        elif file_ext == ".client":
                            # Client Analysis
                            client_result = self.client_analyzer.analyze(content)
                            if isinstance(client_result, dict):
                                lines.append(f"   🖥️ Client Structure Analyzed")
                            else:
                                lines.append(
                                    f"   🖥️ Pages: {len(getattr(client_result, 'pages', []))}"
                                )
                                lines.append(
                                    f"   📋 Commands: {len(getattr(client_result, 'commands', []))}"
                                )

                        elif file_ext == ".projection":
                            # Projection Analysis
                            proj_result = self.projection_analyzer.analyze(content)
                            lines.append(f"   🗃️ Entities: {len(proj_result.entities)}")
                            lines.append(f"   ⚙️ Actions: {len(proj_result.actions)}")

                        elif file_ext == ".fragment":
                            # Fragment Analysis
                            frag_result = self.fragment_analyzer.analyze(content)
                            sections = frag_result.get_section_analysis()
                            section_count = sum(1 for v in sections.values() if v)
                            lines.append(f"   🌟 Sections: {section_count}")

                        lines.append("")

                    except Exception as e:
                        lines.append(f"   ❌ Analysis error: {str(e)}")
                        lines.append("")
                        continue

                # Phase 4: Pattern synthesis and recommendations
                lines.append("🎯 **Phase 4: Implementation Guidance**")
                lines.append("-" * 40)

                lines.append("📈 **Discovered Patterns:**")
                if api_methods:
                    lines.append(f"   • {len(api_methods)} Public APIs available")
                    lines.append(
                        f"   • Common API patterns: {', '.join(sorted(set([m.split('.')[-1].split('_')[0] for m in api_methods[:10]]))[:5])}"
                    )

                if validation_patterns:
                    lines.append(
                        f"   • {len(validation_patterns)} Validation rules found"
                    )
                    common_validations = [v["type"] for v in validation_patterns[:5]]
                    lines.append(
                        f"   • Common validations: {', '.join(common_validations)}"
                    )

                if business_logic:
                    components = set(
                        [bl["component"] for bl in business_logic if bl["component"]]
                    )
                    if components:
                        lines.append(
                            f"   • Active components: {', '.join(sorted(components))}"
                        )

                lines.append("")
                lines.append("🚀 **Implementation Recommendations:**")

                # Generate specific recommendations based on analysis
                if business_logic:
                    avg_apis = sum(bl["public_apis"] for bl in business_logic) / len(
                        business_logic
                    )
                    lines.append(
                        f"   • Follow existing API patterns (avg {avg_apis:.1f} public methods per LU)"
                    )

                if validation_patterns:
                    lines.append(
                        "   • Implement validation rules following discovered patterns"
                    )
                    lines.append(
                        "   • Use Check_Insert___ / Check_Update___ naming conventions"
                    )

                if api_methods:
                    lines.append("   • Leverage existing APIs where possible:")
                    for api in api_methods[:5]:
                        lines.append(f"     - {api}")

                # Domain-specific recommendations
                if domain:
                    lines.append(
                        f"   • Ensure compliance with {domain} module standards"
                    )
                    lines.append(
                        f"   • Follow {domain} naming conventions and patterns"
                    )

                lines.append("")
                lines.append(
                    "✅ **Context Analysis Complete - Ready for Implementation!**"
                )

                return "\\n".join(lines)

            except Exception as e:
                return f"❌ Error in intelligent context analysis: {str(e)}"

        @self.mcp.tool()
        async def get_ifs_development_guidance() -> str:
            """Get comprehensive guidance for IFS Cloud development and database analysis.
            
            **AGENT INSTRUCTIONS:**
            Essential reference for understanding IFS Cloud architecture, development patterns, 
            and how to effectively research and implement features using both this MCP server
            and Oracle database tools.
            
            **When to use:**
            - Starting research on IFS Cloud features
            - Need understanding of IFS architecture patterns  
            - Looking for database analysis guidance
            - Want to know optimal search strategies
            - Beginning implementation work
            
            This tool provides strategic guidance rather than specific code - use other search
            tools to find actual implementations after reading this guidance.
            """
            
            guidance = """# 🏗️ IFS Cloud Development & Database Analysis Guide

## 🔧 **Oracle Database Tool Setup**

**CRITICAL**: Configure Oracle SQL Developer Extension for comprehensive database analysis:
- Extension ID: `Oracle.sql-developer` 
- Install in VS Code for direct database connectivity
- This provides RAG capabilities beyond this MCP server's search functionality
- Essential for accessing runtime/generated content not in source code

## 📊 **Key Database Tables for Research**

When connected to Oracle DB, these tables contain valuable metadata:

### **Reference & Hierarchy Information:**
- `DICTIONARY_SYS_TAB` - System dictionary and object references
- `MODULE_TAB` - Module definitions and hierarchies  
- `LOGICAL_UNIT_TAB` - Logical Unit metadata and relationships
- `ENTITY_TAB` - Entity definitions and attributes
- `PROJECTION_TAB` - Business projection metadata
- `CLIENT_TAB` - Client interface definitions

### **GUI & Navigation Mapping:**
- `NAVIGATOR_SYS_TAB` - Navigator entries (GUI → Backend mapping)
- `FND_PROJ_ENTITY_TAB` - Projection to Entity mappings  
- `CLIENT_PROJECTION_TAB` - Client to Projection relationships
- `MENU_TAB` - Menu structures and navigation

### **Title & Localization:**
- `LANGUAGE_SYS_TAB` - Translatable text and titles
- `FND_SETTING_TAB` - System settings and configurations
- `BASIC_DATA_TRANSLATION_TAB` - Basic data translations

*💡 These tables are more current than analyzing source files for metadata!*

## 🔄 **IFS Architecture Patterns**

### **File Type Relationships:**
```
Browser GUI → .client → .projection → .entity → .plsql
    ↑                                              ↓
    └─────────── Generated Framework Code ────────┘
```

### **Core Development Pattern:**
- **Entities** (.entity): Data models and business rules
- **PL/SQL** (.plsql): Business logic, APIs, validations  
- **Projections** (.projection): Business views and operations
- **Clients** (.client): Frontend interfaces and forms
- **Fragments** (.fragment): Reusable UI components

## 🎯 **Optimal Search Strategies**

### **Multi-File Search Approach:**
When researching features, search ALL related file types:
```
1. Start with Entity: find_related_files("CustomerOrder")
2. Examine PL/SQL: search_content("Customer_Order_API", file_type=".plsql") 
3. Check Projections: search_content("CustomerOrder", file_type=".projection")
4. Review Clients: search_content("CustomerOrder", file_type=".client")
```

### **Business Logic Location Strategy:**
- **Error Messages**: Almost always in .plsql files
- **Business Logic**: Almost always in .plsql files  
- **Validation Rules**: .plsql files (Check_Insert___, Check_Update___)
- **API Methods**: .plsql files (*_API packages)
- **Data Access**: .entity files (logical data model)
- **UI Logic**: .client and .fragment files

## 🔍 **PL/SQL Development Patterns**

### **Standard Attribute Handling:**
```plsql
-- ✅ IFS Pattern - Use attr_ parameters
PROCEDURE New___ (
   newrec_       IN OUT customer_order_tab%ROWTYPE,
   attr_         IN OUT VARCHAR2 )
   
-- ❌ Avoid direct DML - use framework methods
```

### **Common API Patterns:**
- `*_API.New__()` - Create records using attr_ 
- `*_API.Modify__()` - Update records using attr_
- `*_API.Get_Info()` - Retrieve information
- `*_API.Check_Insert___()` - Validation before insert
- `*_API.Check_Update___()` - Validation before update

### **Framework Integration:**
- Use `Client_SYS.Add_To_Attr()` for attribute manipulation
- Use `Error_SYS.Record_General()` for error handling
- Follow three-underscore naming for framework methods (`Method___`)

## ⚡ **Framework vs. Source Code**

### **Generated Content (Not in Source):**
- Standard CRUD operations  
- Basic validation logic
- Framework integration methods
- Standard getter/setter methods
- Default UI behaviors

### **Where to Find Generated Logic:**
- **Oracle Database**: Views like `USER_SOURCE`, `USER_PROCEDURES`
- **Runtime Analysis**: Use Oracle SQL Developer to examine actual procedures
- **Debug Mode**: IFS Developer Studio can show generated code

### **What IS in Source Code:**  
- Custom business logic
- Complex validations
- Specialized API methods  
- Custom UI components
- Integration logic

## 🚀 **Research Workflow**

### **Feature Implementation Research:**
1. **Search Strategy**: Use `search_content()` with business terms
2. **Multi-File Analysis**: Check entities, PL/SQL, projections, clients  
3. **Database Verification**: Query Oracle tables for complete metadata
4. **Pattern Recognition**: Look for similar implementations in codebase
5. **Framework Understanding**: Identify what's generated vs. custom

### **Debugging & Analysis:**
1. **Error Investigation**: Start with .plsql files for messages
2. **Business Logic**: Focus on .plsql API packages
3. **UI Issues**: Check .client and .projection files
4. **Data Problems**: Examine .entity definitions
5. **Integration Issues**: Look for .fragment and related files

## 📚 **Best Practices**

### **Code Research:**
- Search multiple file types for complete understanding
- Use `find_related_files()` to discover all related components
- Check complexity filtering to find simple examples first
- Use Oracle DB for metadata that's not in source files

### **Implementation:**
- Follow IFS naming conventions strictly  
- Use attr_ parameters instead of direct DML
- Implement validation in Check_Insert___ / Check_Update___
- Place business logic in .plsql API packages
- Reference existing patterns before creating new ones

---

*💡 **Remember**: IFS Cloud is a framework-heavy system. Much functionality is generated at runtime. Use both this MCP server for source code research AND Oracle database connectivity for complete analysis!*"""

            return guidance

    def _extract_client_elements_from_ast(self, ast):
        """Extract client elements from AST for analysis display"""
        if not ast:
            return {}

        elements = {"pages": [], "navigators": [], "commands": [], "lists": []}

        def extract_from_node(node):
            node_type = node.get("node_type", "")
            if node_type == "page_declaration":
                elements["pages"].append(node)
            elif node_type == "navigator_section":
                elements["navigators"].append(node)
            elif node_type == "command_declaration":
                elements["commands"].append(node)
            elif node_type == "list":
                elements["lists"].append(node)

            # Recurse through children
            for child in node.get("children", []):
                extract_from_node(child)

        extract_from_node(ast)
        return elements

    def _extract_projection_elements_from_ast(self, ast):
        """Extract projection elements from AST for analysis display"""
        if not ast:
            return {}

        elements = {"entities": [], "queries": [], "actions": [], "functions": []}

        def extract_from_node(node):
            node_type = node.get("node_type", "")
            if node_type == "entity":
                elements["entities"].append(node)
            elif node_type == "query":
                elements["queries"].append(node)
            elif node_type == "action":
                elements["actions"].append(node)
            elif node_type == "function":
                elements["functions"].append(node)

            # Recurse through children
            for child in node.get("children", []):
                extract_from_node(child)

        extract_from_node(ast)
        return elements

    def _format_search_results(self, results: List[SearchResult], title: str) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."

        lines = [f"{title}\n{'=' * len(title)}\n"]

        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.name} ({result.type})")
            lines.append(f"   Path: {result.path}")
            lines.append(
                f"   Score: {result.score:.3f} | Complexity: {result.complexity_score:.2f} | Lines: {result.line_count}"
            )

            # Add IFS Cloud structure information
            ifs_info_parts = []
            if result.module:
                ifs_info_parts.append(f"Module: {result.module}")
            if result.logical_unit:
                ifs_info_parts.append(f"Logical Unit: {result.logical_unit}")
            if result.entity_name:
                ifs_info_parts.append(f"Entity: {result.entity_name}")
            if result.component:
                ifs_info_parts.append(f"Component: {result.component}")

            if ifs_info_parts:
                lines.append(f"   IFS: {' | '.join(ifs_info_parts)}")

            if result.entities:
                entities_str = ", ".join(result.entities[:5])
                if len(result.entities) > 5:
                    entities_str += f" (and {len(result.entities) - 5} more)"
                lines.append(f"   Entities: {entities_str}")

            if result.content_preview:
                preview = result.content_preview.replace("\n", " ")[:100]
                if len(result.content_preview) > 100:
                    preview += "..."
                lines.append(f"   Preview: {preview}")

            lines.append("")  # Empty line between results

        return "\n".join(lines)

    def run(self, transport_type: str = "stdio", **kwargs):
        """Run the MCP server.

        Args:
            transport_type: Transport type ("stdio", "sse", etc.)
            **kwargs: Additional transport arguments
        """
        logger.info(f"Starting IFS Cloud MCP Server with {transport_type} transport")

        if transport_type == "stdio":
            self.mcp.run(transport="stdio")
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

    def _extract_business_keywords(self, business_requirement: str) -> List[str]:
        """Extract meaningful keywords from a business requirement."""
        import re

        # Common IFS Cloud business terms that should be preserved
        ifs_terms = [
            "customer",
            "order",
            "product",
            "invoice",
            "project",
            "supplier",
            "validation",
            "calculation",
            "pricing",
            "approval",
            "workflow",
            "entity",
            "api",
            "method",
            "function",
            "check",
            "get",
            "create",
            "update",
            "delete",
            "insert",
            "modify",
            "retrieve",
            "generate",
            "business",
            "logic",
            "rule",
            "constraint",
            "requirement",
        ]

        # Clean and tokenize the requirement
        cleaned = re.sub(r"[^\w\s]", " ", business_requirement.lower())
        words = cleaned.split()

        # Extract meaningful keywords
        keywords = []
        for word in words:
            if len(word) > 3 and word in ifs_terms:
                keywords.append(word)

        # Add compound terms
        for i in range(len(words) - 1):
            compound = f"{words[i]}_{words[i+1]}"
            if len(words[i]) > 2 and len(words[i + 1]) > 2:
                if any(
                    term in compound
                    for term in [
                        "customer_order",
                        "product_info",
                        "price_calc",
                        "order_line",
                    ]
                ):
                    keywords.append(compound)

        # Remove duplicates and limit to top terms
        keywords = list(dict.fromkeys(keywords))[:8]

        # If no specific terms found, use all significant words
        if not keywords:
            keywords = [word for word in words if len(word) > 4][:5]

        return keywords

    def _extract_entity_info(self, content: str) -> Dict[str, Any]:
        """Extract basic entity information from entity file content."""
        import re

        info = {}

        # Extract entity name
        entity_match = re.search(r"entity\s+(\w+)", content, re.IGNORECASE)
        if entity_match:
            info["name"] = entity_match.group(1)

        # Count attributes
        attributes = re.findall(
            r"^\s*attribute\s+\w+", content, re.MULTILINE | re.IGNORECASE
        )
        info["attributes"] = attributes

        # Extract keys
        keys_match = re.search(r"keys\s*=\s*([^;]+)", content, re.IGNORECASE)
        if keys_match:
            info["keys"] = keys_match.group(1).strip()

        return info

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "indexer"):
            self.indexer.close()
        logger.info("IFS Cloud MCP Server cleanup completed")
