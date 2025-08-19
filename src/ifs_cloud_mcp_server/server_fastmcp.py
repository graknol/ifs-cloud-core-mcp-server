"""
IFS Cloud MCP Server with FastMCP Framework

Clean implementation with focused IFS development guidance tool.
"""

import logging
from pathlib import Path

from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IFSCloudMCPServer:
    """IFS Cloud MCP Server providing development guidance."""

    def __init__(
        self, index_path: Path = None, name: str = "IFS Cloud Development Server"
    ):
        """Initialize the IFS Cloud MCP Server.

        Args:
            index_path: Path to the version directory containing analysis and search indexes
            name: Name for the MCP server
        """
        self.mcp = FastMCP(name)

        # Initialize paths
        if index_path:
            self.index_path = Path(index_path)
        else:
            from .directory_utils import get_data_directory

            self.index_path = get_data_directory()

        self.index_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using index path: {self.index_path}")

        # Initialize search engine
        self.search_engine = None
        self._initialize_search_engine()

        # Register tools
        self._register_tools()

    def _initialize_search_engine(self):
        """Initialize the hybrid search engine if embeddings are available."""
        try:
            # The index_path now points directly to versions/version_name
            version_dir = self.index_path

            # Look for the faiss directory which contains the embeddings
            faiss_dir = version_dir / "faiss"

            if faiss_dir.exists():
                from .hybrid_search import HybridSearchEngine

                self.search_engine = HybridSearchEngine(faiss_dir)
                logger.info("✅ Search engine initialized with FAISS directory")
            else:
                logger.warning(
                    f"⚠️  No FAISS embeddings found in {faiss_dir}. Search functionality will be limited."
                )
                logger.info(
                    f"Run embeddings creation: python -m src.ifs_cloud_mcp_server.main embed --version <version>"
                )
        except Exception as e:
            logger.error(f"❌ Failed to initialize search engine: {e}")
            self.search_engine = None

    def _register_tools(self):
        """Register MCP tools for IFS Cloud development guidance."""

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
- **Entities** (.entity): Data model definition
- **PL/SQL** (.plsql): Business logic, APIs, validations  
- **Projections** (.projection): Backend-for-frontend layer (OData definition defined in IFS's marble language)
- **Clients** (.client): Frontend interfaces and forms
- **Fragments** (.fragment): Reusable client and projection components baked into one file

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
- **Data Definition**: .entity files (logical data model)
- **UI Logic**: .client and .fragment files (a lot of frontend logic and elements have been tucked away in .fragment files. You can see which fragments are included by looking at the top of the .client file)

## 🔍 **PL/SQL Development Patterns**

### **Standard Attribute Handling:**
```plsql
-- ✅ IFS Pattern - Use attr_ parameters
PROCEDURE New___ (
   info_       OUT    VARCHAR2,
   objid_      OUT    VARCHAR2,
   objversion_ OUT    VARCHAR2,
   attr_       IN OUT NOCOPY VARCHAR2,
   action_     IN     VARCHAR2 )
   
-- ❌ Avoid direct DML - use framework methods
```

### **Common API Patterns:**
- `*_API.New__()` - Create records using attr_ 
- `*_API.Modify__()` - Update records using attr_
- `*_API.Get_*()` - Get the value of a public field
- `*_API.Check_Insert___()` - Validation before insert
- `*_API.Check_Update___()` - Validation before update

### **Framework Integration:**
- Use `Client_SYS.Add_To_Attr()` for attribute manipulation
- Use `Error_SYS.Record_General()` for error handling
- Follow three-underscore naming for private methods (`Method___`), two-underscore naming for protected methods (`Method__`) and no-underscore naming for public methods (`Method`).

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
4. **Data Problems**: Examine .entity and .plsql definitions

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

        @self.mcp.tool()
        async def search_ifs_codebase(
            query: str = None,
            semantic_query: str = None,
            lexical_query: str = None,
            search_mode: str = "hybrid",
            max_results: int = 10,
            explain_results: bool = True,
        ) -> str:
            """Search the IFS Cloud codebase using advanced hybrid search.

            **AGENT INSTRUCTIONS:**
            This tool provides intelligent search across your analyzed IFS Cloud codebase.
            Use this to find files, APIs, business logic, and implementations.

            **Search Modes:**
            - "hybrid": Use both semantic and lexical search (recommended for most queries)
            - "semantic": Semantic search only (best for concepts, "what does X do?")
            - "lexical": Exact matching only (best for specific API names, error codes)

            **Query Guidelines:**
            - **Hybrid mode**: Provide either `query` (used for both) OR separate `semantic_query` + `lexical_query`
            - **Semantic mode**: Use conceptual terms like "customer validation", "order processing"
            - **Lexical mode**: Use exact terms like "CUSTOMER_ORDER_API", "Check_Insert___"

            **Best Results Tips:**
            - For finding specific APIs: Use lexical mode with exact API names
            - For understanding functionality: Use semantic mode with business terms
            - For comprehensive research: Use hybrid mode with business concepts
            - For error investigation: Use lexical mode with error messages

            **Examples:**
            - Hybrid: query="customer order validation"
            - Semantic: semantic_query="how to validate customer data", lexical_query="Customer_Order_API"
            - Lexical: query="CUSTOMER_ORDER_API.Check_Insert___"

            Returns detailed search results with file paths, API names, snippets, and explanations.
            """
            if not self.search_engine:
                return """❌ **Search Not Available**

The search engine is not initialized. This usually means:
1. No embeddings have been created for this version
2. The analysis hasn't been completed yet

**To fix this:**
1. Ensure you've run: `uv run python -m src.ifs_cloud_mcp_server.main analyze --version <version>`
2. Create embeddings: `uv run python -m src.ifs_cloud_mcp_server.main embed --version <version>`
3. Restart the MCP server

**Alternative:** Use the guidance tool to understand IFS architecture patterns while waiting for search setup."""

            try:
                # Determine search configuration based on mode
                if search_mode == "semantic":
                    from .hybrid_search import SearchConfig

                    config = SearchConfig(enable_faiss=True, enable_flashrank=False)
                    # For semantic-only, prioritize the semantic query
                    if semantic_query is None:
                        semantic_query = query or ""
                    if lexical_query is None:
                        lexical_query = ""
                elif search_mode == "lexical":
                    from .hybrid_search import SearchConfig

                    config = SearchConfig(enable_faiss=False, enable_flashrank=False)
                    # For lexical-only, prioritize the lexical query
                    if lexical_query is None:
                        lexical_query = query or ""
                    if semantic_query is None:
                        semantic_query = ""
                else:  # hybrid mode
                    from .hybrid_search import SearchConfig

                    config = SearchConfig.medium_hardware()  # Balanced default

                # Perform the search
                response = self.search_engine.search(
                    query=query,
                    semantic_query=semantic_query,
                    lexical_query=lexical_query,
                    top_k=max_results,
                    config=config,
                    explain_results=explain_results,
                )

                # Format results
                if not response.results:
                    return f"""🔍 **No Results Found**

**Query:** {response.query}
**Search Mode:** {search_mode}
**Search Time:** {response.search_time:.3f}s

**Suggestions:**
- Try different search terms or synonyms
- Use broader or more specific terms
- Switch search modes (hybrid ↔ semantic ↔ lexical)
- Check if the functionality exists in your IFS Cloud version

**Query Analysis:** {response.query_type.value} query detected
**Sources Searched:** BM25S: {response.bm25s_count}, FAISS: {response.faiss_count}"""

                # Format successful results
                result_text = [
                    f"🔍 **Search Results** ({len(response.results)} of {response.total_found} found)",
                    f"**Query:** {response.query}",
                    f"**Mode:** {search_mode} | **Time:** {response.search_time:.3f}s",
                    f"**Fusion:** {response.fusion_method}",
                    "",
                ]

                for i, result in enumerate(response.results, 1):
                    result_text.extend(
                        [
                            f"## {i}. {result.title}",
                            f"**File:** `{result.file_path}`",
                            f"**API:** `{result.api_name}` | **Rank:** #{result.rank} | **Score:** {result.score:.4f}",
                            f"**Type:** {result.match_type} | **Source:** {result.source}",
                            "",
                            f"**Snippet:**",
                            f"```",
                            result.snippet,
                            f"```",
                            "",
                        ]
                    )

                    if explain_results and result.explanation:
                        result_text.extend(
                            [
                                f"**Why this result:** {result.explanation}",
                                "",
                            ]
                        )

                # Add suggestions if available
                if response.suggestions:
                    result_text.extend(
                        [
                            "**💡 Suggested refinements:**",
                            *[f"- {suggestion}" for suggestion in response.suggestions],
                            "",
                        ]
                    )

                return "\n".join(result_text)

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return f"❌ **Search Error:** {str(e)}\n\nPlease check the server logs for more details."

        @self.mcp.tool()
        async def search_ifs_semantic(
            semantic_query: str,
            max_results: int = 10,
        ) -> str:
            """Search IFS Cloud codebase using semantic/conceptual search only.

            **AGENT INSTRUCTIONS:**
            Use this for concept-based searches when you want to understand functionality,
            find business logic, or discover patterns. Best for questions like:
            - "How does customer validation work?"
            - "Where is order processing logic?"
            - "Find invoice approval workflow"

            **When to use semantic search:**
            - Finding functionality by business concepts
            - Understanding complex business logic
            - Discovering related components
            - Researching how features work

            This search uses AI embeddings to understand meaning and context.
            """
            return await search_ifs_codebase(
                query=None,
                semantic_query=semantic_query,
                lexical_query="",
                search_mode="semantic",
                max_results=max_results,
                explain_results=True,
            )

        @self.mcp.tool()
        async def search_ifs_lexical(
            lexical_query: str,
            max_results: int = 10,
        ) -> str:
            """Search IFS Cloud codebase using exact/lexical matching only.

            **AGENT INSTRUCTIONS:**
            Use this for exact matches when you know specific names, API calls, or error messages.
            Best for precise searches like:
            - "CUSTOMER_ORDER_API"
            - "Check_Insert___"
            - "Error_SYS.Record_General"
            - Specific procedure names or error codes

            **When to use lexical search:**
            - Finding specific APIs or procedures
            - Searching for exact error messages
            - Looking for specific method names
            - Finding exact code patterns

            This search uses BM25 text matching for precise results.
            """
            return await search_ifs_codebase(
                query=None,
                semantic_query="",
                lexical_query=lexical_query,
                search_mode="lexical",
                max_results=max_results,
                explain_results=True,
            )

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

    def cleanup(self):
        """Clean up resources."""
        logger.info("IFS Cloud MCP Server cleanup completed")
