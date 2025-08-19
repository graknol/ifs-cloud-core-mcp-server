"""
IFS Cloud MCP Server with FastMCP Framework

Clean implementation with focused IFS development guidance tool.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP
from .indexer import EnhancedIFSIndexer
from .embedding_processor import EmbeddingProcessor
from .search_engine import UnixCoderSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IFSCloudMCPServer:
    """IFS Cloud MCP Server providing development guidance."""

    def __init__(self, data_dir: str = None):
        """Initialize the IFS Cloud MCP Server.

        Args:
            data_dir: Data directory path for storing indices and embeddings
        """
        self.mcp = FastMCP("IFS Cloud Development Server")

        # Initialize data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            from platformdirs import user_data_dir

            self.data_dir = Path(user_data_dir("ifs-cloud-mcp-server"))

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using data directory: {self.data_dir}")

        # Initialize components
        self.indexer = EnhancedIFSIndexer(str(self.data_dir))
        self.embedding_processor = EmbeddingProcessor(str(self.data_dir))
        self.search_engine = UnixCoderSearchEngine(str(self.data_dir))

        # Register tools
        self._register_tools()

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
        if hasattr(self, "indexer"):
            self.indexer.close()
        logger.info("IFS Cloud MCP Server cleanup completed")
