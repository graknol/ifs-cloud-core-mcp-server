# 🧠 IFS Cloud Intelligent AI Agent

> \*\*Transform your IFS Cloud developme --connection "oracle://ifsapp:password@host:1521/IFSCDEV" 25.1.0

````

**📁 Custom ZIP Import** (For specific versions):deeply understands your codebase**

An intelligent Model Context Protocol (MCP) server that makes AI agents truly smart about IFS Cloud development. Features comprehensive code analysis, versioned ZIP catalog management, GUI-aware search with Navigator mappings, database metadata extraction, and intelligent context gathering for perfect architectural consistency.

---

## ✨ **What Makes This Special**

### 🎯 **Intelligent AI Agent**

- **Automatic Context Discovery**: AI proactively searches and analyzes your codebase before implementing
- **Pattern Recognition**: Discovers existing validation rules, APIs, and architectural patterns
- **Perfect Integration**: Every implementation matches your existing IFS Cloud conventions
- **Zero False Positives**: Conservative analysis ensures reliable insights

> **Transform your IFS Cloud development with AI that deeply understands your codebase**

An intelligent Model Context Protocol (MCP) server that makes AI agents truly smart about IFS Cloud development. Features comprehensive code analysis, versioned ZIP catalog management, GUI-aware search, and intelligent context gathering for perfect architectural consistency.

---

## ✨ **What Makes This Special**

### 🎯 **Intelligent AI Agent**

- **Automatic Context Discovery**: AI proactively searches and analyzes your codebase before implementing
- **Pattern Recognition**: Discovers existing validation rules, APIs, and architectural patterns
- **Perfect Integration**: Every implementation matches your existing IFS Cloud conventions
- **Zero False Positives**: Conservative analysis ensures reliable insights

### 🎨 **Modern Web Interface**

- **React + TypeScript**: Modern, responsive web UI with real-time search
- **Type-ahead Suggestions**: Smart autocomplete with context-aware suggestions
- **Faceted Filters**: Advanced filtering by file type, complexity, and module
- **File Viewer**: Built-in CodeMirror editor with IFS Marble syntax highlighting
- **Dark Theme**: Sleek, modern design optimized for developer productivity
- **Mobile Responsive**: Works perfectly on desktop, tablet, and mobile devices

### 🧠 **GUI-Aware Smart Search**

- **Real Production Mappings**: GUI labels mapped to backend entities using live IFS Cloud databases
- **60% Perfect Matches**: Search for "Customer Order" and find CustomerOrder entities instantly
- **15,000+ Files**: Handle complete IFS Cloud codebases efficiently
- **Sub-second Search**: Powered by Tantivy search engine with GUI enhancement
- **Smart Filtering**: Filter by module, file type, complexity, and more
- **Contextual Results**: Rich metadata with previews and relationships

### 📦 **Versioned Catalog Management**

- **ZIP Import**: Import entire IFS Cloud releases from ZIP files
- **Version Control**: Manage multiple IFS Cloud versions (24.1, 24.2, latest, dev builds)
- **Instant Switching**: Switch between versions for different projects
- **Smart Extraction**: Automatically filters and organizes IFS Cloud files

---

## 🚀 **Quick Start**

### 1. **Install & Setup**

```bash
git clone https://github.com/graknol/ifs-cloud-core-mcp-server.git
cd ifs-cloud-core-mcp-server
uv sync
````

### 2. **Launch the Web Interface**

```bash
# Start the modern web UI
uv run python -m src.ifs_cloud_mcp_server.web_ui

# Opens at http://localhost:5700
# Features: Real-time search, type-ahead, file viewer, dark theme
```

### 3. **Choose Your Data Source**

**🏭 Extract from Your Database** (Recommended - Environment-Specific):

- Get metadata tailored to YOUR specific IFS Cloud environment with Navigator GUI mappings
- Always current with your actual database schema, customizations, and navigation structure
- Includes FND_NAVIGATOR_ALL mappings connecting GUI elements to backend projections
- Perfect for developers who want the most accurate search results and UI understanding

```bash
# Secure extraction with environment variables
export IFS_DB_PASSWORD="your_secure_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host your-db-host --username ifsapp --service IFSCDEV 25.1.0

# Or use connection string for quick setup
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "oracle://ifsapp:password@host:1521/IFSCDEV" 25.1.0
```

**📦 Use Production Data** (Ready-to-use):

- Complete system with pre-extracted production metadata
- Enhanced search with business term matching and metadata enrichment
- Ready-to-use with real IFS Cloud files

```bash
cd production
uv run python test_setup.py  # Verify production setup
uv run python demos/demo_real_files.py  # See the magic happen!
```

**� Custom ZIP Import** (For specific versions):

```bash
# Import any IFS Cloud ZIP file to create versioned catalog
uv run python -m src.ifs_cloud_mcp_server.main import "IFS_Cloud_24.2.1.zip" --version "24.2.1"
```

```

```

### 3. **Start Intelligent AI Agent**

```bash
# Start with your imported version
uv run python -m src.ifs_cloud_mcp_server.main server --version "24.2.1"
```

### 4. **Connect GitHub Copilot**

Configure your MCP client to connect to the intelligent AI agent and experience AI that truly understands your IFS Cloud patterns!

---

## 🔧 **Intelligent Features**

<table>
<tr>
<td><strong>🧠 Intelligent Context Analysis</strong></td>
<td><strong>📊 Deep Code Analysis</strong></td>
</tr>
<tr>
<td>
• Automatic pattern discovery<br>
• Business requirement understanding<br>
• Existing API identification<br>
• Best practice recommendations
</td>
<td>
• PLSQL business logic analysis<br>
• Client UI pattern recognition<br>
• Projection data model mapping<br>
• Fragment full-stack understanding
</td>
</tr>
</table>

<table>
<tr>
<td><strong>📦 Version Management</strong></td>
<td><strong>⚡ High Performance</strong></td>
</tr>
<tr>
<td>
• ZIP file import/extraction<br>
• Multiple version support<br>
• Isolated environments<br>
• Easy switching between versions
</td>
<td>
• 1000+ files/second indexing<br>
• <100ms search response<br>
• Intelligent caching system<br>
• Batch processing optimization
</td>
</tr>
</table>

---

## 📁 **Supported IFS Cloud Files**

| File Type                    | Purpose               | AI Understanding                   |
| ---------------------------- | --------------------- | ---------------------------------- |
| **`.plsql`**                 | Business Logic        | APIs, validations, business rules  |
| **`.entity`**                | Data Models           | Entity relationships, attributes   |
| **`.client`**                | User Interface        | UI patterns, commands, navigation  |
| **`.projection`**            | Data Access           | Queries, actions, data surface     |
| **`.fragment`**              | Full-Stack Components | Complete UI-to-data integration    |
| **`.views`**, **`.storage`** | Database Layer        | Data structure and access patterns |

---

## 🎯 **Intelligent Workflow Example**

```
💬 User: "Add customer order validation to check credit limits"

🧠 AI Agent automatically:
   1. Searches for "validation", "customer", "order", "credit" patterns
   2. Finds existing CustomerOrder.plsql, validation methods
   3. Analyzes business logic with PLSQL analyzer
   4. Discovers Check_Insert___ validation patterns
   5. Identifies existing Customer_API methods
   6. Generates implementation matching your exact patterns

✅ Result: Perfect architectural consistency!
```

---

## 📋 **Commands Reference**

### **Database Metadata Extraction**

```bash
# Extract metadata from your database (recommended)
export IFS_DB_PASSWORD="secure_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host db-host --username ifsapp --service IFSCDEV 25.1.0

# Extract with connection string
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "oracle://user:pass@host:1521/service" 25.1.0

# JSON output for automation
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "oracle://..." --quiet --json 25.1.0
```

### **ZIP Management**

```bash
# Import IFS Cloud ZIP file
uv run python -m src.ifs_cloud_mcp_server.main import <zip_file> <version>

# List available versions
uv run python -m src.ifs_cloud_mcp_server.main list

# Start server with specific version
uv run python -m src.ifs_cloud_mcp_server.main server --version <version>
```

### **Server Management**

```bash
# Start MCP server (default - uses ./index)
uv run python -m src.ifs_cloud_mcp_server.main server

# Start with specific version
uv run python -m src.ifs_cloud_mcp_server.main server --version "25.1.0"

# Start with custom index path
uv run python -m src.ifs_cloud_mcp_server.main server --index-path ./my_index
```

---

## � **MCP Client Configuration**

### **GitHub Copilot**

```json
{
  "mcpServers": {
    "ifs-cloud-intelligent-agent": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "src.ifs_cloud_mcp_server.main",
        "server",
        "--version",
        "24.2.1"
      ],
      "cwd": "/path/to/ifs-cloud-core-mcp-server"
    }
  }
}
```

### **Claude Desktop**

```json
{
  "mcpServers": {
    "ifs-cloud": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "src.ifs_cloud_mcp_server.main",
        "server",
        "--version",
        "24.2.1"
      ],
      "cwd": "/path/to/ifs-cloud-core-mcp-server"
    }
  }
}
```

---

## 📚 **Documentation**

- **[� Metadata Extraction CLI](./METADATA_EXTRACTION_CLI.md)** - Extract metadata from YOUR database
- **[�📖 ZIP Indexing Walkthrough](./ZIP_WALKTHROUGH.md)** - Step-by-step import example
- **[📋 ZIP Indexing Instructions](./ZIP_INDEXING_INSTRUCTIONS.md)** - Complete import documentation
- **[🧠 Intelligent Agent Guide](./INTELLIGENT_AGENT.md)** - How the AI agent works
- **[🌐 Web UI Documentation](./WEB_UI_README.md)** - Interactive exploration interface

> **Note**: All metadata extraction including GUI mappings is now integrated into the main CLI. Use the `extract` command to gather data from your IFS Cloud database.

---

## 🎉 **The Result**

Your AI agent now has **comprehensive IFS Cloud intelligence** and will:

- ✅ **Automatically understand** your specific IFS Cloud patterns
- ✅ **Discover existing APIs** and validation approaches
- ✅ **Generate consistent code** that matches your architecture
- ✅ **Follow naming conventions** and business rule patterns
- ✅ **Leverage existing components** instead of reinventing
- ✅ **Maintain quality standards** across all implementations

**Transform your development workflow with AI that truly understands IFS Cloud!** 🚀

---

<div align="center">

**[⭐ Star this repo](https://github.com/graknol/ifs-cloud-core-mcp-server)** • **[📝 Report Issues](https://github.com/graknol/ifs-cloud-core-mcp-server/issues)** • **[💬 Discussions](https://github.com/graknol/ifs-cloud-core-mcp-server/discussions)**

_Built with ❤️ for IFS Cloud developers_

</div>
```

3. **Open browser:** Navigate to `http://localhost:5700` (or the port shown in the startup message) and start exploring!

### 🔌 **MCP Server Mode** (For AI integration)

1. **Start MCP server:**

```bash
# For Claude Desktop or other MCP clients
uv run python -m src.ifs_cloud_mcp_server.main
```

2. **Configure in Claude Desktop:**

```json
{
  "mcpServers": {
    "ifs-cloud": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.ifs_cloud_mcp_server.main"],
      "cwd": "/path/to/ifs-cloud-core-mcp-server"
    }
  }
}
```

## Installation

```bash
# Clone the repository
git clone https://github.com/graknol/ifs-cloud-core-mcp-server.git
cd ifs-cloud-core-mcp-server

# Install dependencies with UV (recommended)
uv sync

# Or with pip
pip install -e .

# For development
pip install -e ".[dev]"
```

### 🤖 **AI Intent Classification Models**

The server uses FastAI models for intelligent query classification. Models are automatically downloaded from GitHub releases when first needed:

```bash
# Models download automatically, but you can also:

# Download manually
uv run python -m src.ifs_cloud_mcp_server.model_downloader

# Train your own model (optional)
uv run python scripts/train_proper_fastai.py

# Prepare model for release (maintainers)
uv run python scripts/prepare_model_release.py
```

**Model Details:**

- **Size**: ~121MB (FastAI ULMFiT model)
- **Storage**: Downloaded from GitHub releases (not in repo)
- **Fallback**: Graceful degradation if model unavailable
- **GPU Support**: Automatic CUDA detection and acceleration

## Quick Start

```bash
# Start the MCP server
ifs-cloud-mcp-server --port 8000 --index-path ./index

# Index your IFS Cloud codebase
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/ifs/cloud/project"}'

# Search for entities
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CustomerOrder", "type": "entity"}'
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│   MCP Server    │◄──►│ Tantivy Index   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ File Processors │
                       │  - Entity       │
                       │  - PL/SQL       │
                       │  - Views        │
                       │  - Storage      │
                       │  - Fragment     │
                       │  - Client       │
                       │  - Projection   │
                       └─────────────────┘
```

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Performance

- **Indexing Speed**: ~1000 files/second on typical hardware
- **Search Response**: <100ms for most queries
- **Memory Usage**: ~200MB for 1GB codebase index
- **Incremental Updates**: Real-time file change tracking

## Future Roadmap

- 🤖 **AI Integration**: FastAI/PyTorch for semantic search
- 🧠 **Pattern Recognition**: ML-based code pattern detection
- 📈 **Analytics**: Advanced codebase insights and metrics
- 🔗 **IDE Integration**: VS Code and IntelliJ plugins

## License

Licensed under the terms specified in the LICENSE file.
