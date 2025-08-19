# 🧠 IFS Cloud MCP Server

> **AI-powered Model Context Protocol server for intelligent IFS Cloud codebase analysis**

A sophisticated Model Context Protocol (MCP) server that provides AI agents with deep understanding of IFS Cloud codebases through comprehensive analysis, PageRank importance ranking, and intelligent code search capabilities.

---

## ✨ **Key Features**

### 🎯 **Intelligent Code Analysis**

- **Comprehensive File Analysis**: Extracts API calls, procedure/function names, and dependency relationships
- **PageRank Ranking**: Identifies the most important files based on dependency network analysis
- **Reference Graph**: Maps inter-file dependencies for architectural understanding
- **Multi-format Support**: Handles `.plsql`, `.entity`, `.client`, `.projection`, `.fragment`, and more

### 📦 **Version Management**

- **ZIP Import**: Import complete IFS Cloud releases from ZIP files
- **Multiple Versions**: Manage and switch between different IFS Cloud versions
- **Isolated Analysis**: Each version maintains separate analysis data
- **Smart Extraction**: Automatically filters and organizes supported file types

### 🔍 **Advanced Hybrid Search & Discovery**

- **Dual-Query Hybrid Search**: Separate semantic and lexical queries for precision control
- **BGE-M3 Semantic Search**: AI-powered understanding using state-of-the-art embeddings
- **BM25S Lexical Search**: Fast exact matching for API names, functions, and keywords
- **FlashRank Fusion**: Neural reranking combines semantic and lexical results intelligently
- **Three Search Modes**: Comprehensive, semantic-only, or lexical-only via MCP tools
- **PageRank Importance**: Files ranked by their significance in the dependency network
- **CUDA Acceleration**: GPU-powered semantic search for maximum performance

---

## 🚀 **Quick Start**

### 1. **Installation**

```bash
git clone https://github.com/graknol/ifs-cloud-core-mcp-server.git
cd ifs-cloud-core-mcp-server
uv sync
```

### 2. **Import IFS Cloud Version**

```bash
# Import an IFS Cloud ZIP file
uv run python -m src.ifs_cloud_mcp_server.main import "IFS_Cloud_25.1.0.zip" --version "25.1.0"
```

### 3. **Analyze the Codebase**

```bash
# Perform comprehensive analysis
uv run python -m src.ifs_cloud_mcp_server.main analyze --version "25.1.0"

# Calculate PageRank importance scores
uv run python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version "25.1.0"
```

### 4. **Start the MCP Server**

```bash
# Start server with analyzed version
uv run python -m src.ifs_cloud_mcp_server.main server --version "25.1.0"
```

---

## 📋 **CLI Commands Reference**

### **Version Management**

```bash
# Import a ZIP file
uv run python -m src.ifs_cloud_mcp_server.main import <zip_file> --version <version_name>

# List all versions
uv run python -m src.ifs_cloud_mcp_server.main list

# Delete a version
uv run python -m src.ifs_cloud_mcp_server.main delete --version <version_name> [--force]
```

### **Analysis Commands**

```bash
# Analyze codebase (extract dependencies, API calls, etc.)
uv run python -m src.ifs_cloud_mcp_server.main analyze --version <version> [--max-files N] [--force]

# Calculate PageRank importance scores
uv run python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version <version>

# Create embeddings for semantic search (uses BGE-M3 model)
uv run python -m src.ifs_cloud_mcp_server.main embed --version <version> [--max-files N]

# Create test embeddings (top 10 files for quick testing)
uv run python -m src.ifs_cloud_mcp_server.main embed --version <version> --max-files 10
```

### **Server Operation**

```bash
# Start MCP server
uv run python -m src.ifs_cloud_mcp_server.main server --version <version>

# Start web UI (if available)
uv run python -m src.ifs_cloud_mcp_server.web_ui
```

---

## � **MCP Search Tools**

The server provides three sophisticated search tools for AI agents:

### **search_ifs_codebase** - Comprehensive Hybrid Search

```typescript
// Full hybrid search with separate semantic and lexical queries
search_ifs_codebase(
  query: "validation logic",           // Main query (fallback for both)
  semantic_query: "business rules",    // For FAISS semantic search
  lexical_query: "Check_Insert___",   // For BM25S exact matching
  max_results: 10,                    // Number of results
  explain: true                       // Include scoring explanations
)
```

### **search_ifs_semantic** - AI-Powered Understanding

```typescript
// Pure semantic search using BGE-M3 embeddings
search_ifs_semantic(
  semantic_query: "customer credit validation patterns",
  max_results: 10,
  explain: true
)
```

### **search_ifs_lexical** - Exact API & Keyword Matching

```typescript
// Pure lexical search using BM25S
search_ifs_lexical(
  lexical_query: "Customer_API.Get_Credit_Limit___",
  max_results: 10,
  explain: true
)
```

**Key Features:**

- **Dual Query Processing**: Different queries optimized for semantic vs lexical search
- **BGE-M3 Embeddings**: 1024-dimension vectors with 8192 token context
- **CUDA Acceleration**: GPU-powered semantic search when available
- **FlashRank Fusion**: Neural reranking for optimal result ordering
- **PageRank Integration**: Importance-weighted result scoring
- **Detailed Explanations**: Optional scoring breakdowns for transparency

---

## �🔧 **MCP Client Configuration**

### **Claude Desktop**

Add to your Claude Desktop configuration:

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
        "25.1.0"
      ],
      "cwd": "/path/to/ifs-cloud-core-mcp-server"
    }
  }
}
```

### **Other MCP Clients**

```bash
# Standard MCP server startup
uv run python -m src.ifs_cloud_mcp_server.main server --version "25.1.0"
```

---

## 📊 **Analysis Output**

The system generates comprehensive analysis data:

### **Dependency Analysis**

- **API Calls**: Which APIs each file calls
- **Reference Graph**: File-to-file dependency mappings
- **Incoming Links**: How many files depend on each file

### **PageRank Scoring**

- **Importance Ranking**: Files ranked by network centrality
- **Foundation APIs**: Infrastructure files (FndSession, Site, etc.) rank highest
- **Business Logic**: Domain-specific files ranked by usage patterns

### **File Metadata**

- **File Size & Type**: Basic file characteristics
- **Procedure/Function Names**: Code structure analysis
- **Change Information**: Extracted from comments and headers

---

## 🎯 **Intelligent Workflow Example**

### **AI Agent Search Workflow**

```
💬 User: "Find customer credit validation patterns in IFS Cloud"

🧠 AI Agent automatically uses hybrid search:

1️⃣ **Semantic Search** (search_ifs_semantic):
   Query: "customer credit validation business rules"
   → BGE-M3 finds conceptually similar code patterns
   → Returns files with credit checking logic, validation routines

2️⃣ **Lexical Search** (search_ifs_lexical):
   Query: "Customer_API Credit_Limit Check_Credit"
   → BM25S finds exact API names and function calls
   → Returns specific implementation methods

3️⃣ **Hybrid Fusion** (search_ifs_codebase):
   Semantic: "credit validation patterns"
   Lexical: "Customer_API.Check_Credit___"
   → FlashRank combines both approaches intelligently
   → PageRank boosts important foundation files

✅ Result: Comprehensive understanding across:
   - Business logic patterns (semantic)
   - Exact API implementations (lexical)
   - Architectural importance (PageRank)
   - Perfect architectural consistency!
```

### **Complete Setup Workflow**

```bash
# 1. Import IFS Cloud version
uv run python -m src.ifs_cloud_mcp_server.main import "IFS_Cloud_25.1.0.zip" --version "25.1.0"

# 2. Analyze the codebase (extract dependencies, API calls)
uv run python -m src.ifs_cloud_mcp_server.main analyze --version "25.1.0"

# 3. Calculate importance rankings (PageRank network analysis)
uv run python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version "25.1.0"

# 4. Create semantic embeddings (BGE-M3 model, ~5-10 minutes)
uv run python -m src.ifs_cloud_mcp_server.main embed --version "25.1.0"

# 5. Start MCP server with full hybrid search capabilities
uv run python -m src.ifs_cloud_mcp_server.main server --version "25.1.0"
```

**Result**: AI agents now have comprehensive hybrid search across your IFS Cloud codebase!

---

## 📁 **Supported File Types**

| File Type         | Purpose                   | Analysis Features                |
| ----------------- | ------------------------- | -------------------------------- |
| **`.plsql`**      | PL/SQL Business Logic     | API calls, procedures, functions |
| **`.entity`**     | Data Entity Definitions   | Entity relationships             |
| **`.client`**     | User Interface Components | UI patterns, commands            |
| **`.projection`** | Data Access Layer         | Queries, actions                 |
| **`.fragment`**   | Full-Stack Components     | Complete integration patterns    |
| **`.views`**      | Database Views            | Data structure                   |
| **`.storage`**    | Storage Definitions       | Database mappings                |

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Client     │◄──►│   MCP Server    │◄──►│ Analysis Data   │
│ (Claude, etc.)  │    │                 │    │ (JSON/JSONL)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Hybrid Search   │
                       │ • BGE-M3 FAISS  │
                       │ • BM25S Lexical │
                       │ • FlashRank     │
                       │ • PageRank      │
                       └─────────────────┘
```

### **Search Architecture Detail**

```
Query Input
     │
     ▼
┌─────────────────┐
│ Query Processor │ ◄─── Semantic Query + Lexical Query
│ • Intent detect │
│ • Query split   │
└─────────────────┘
     │
     ▼
┌─────────────────┐    ┌─────────────────┐
│ FAISS Search    │    │ BM25S Search    │
│ • BGE-M3 embed  │    │ • Exact match   │
│ • Semantic sim  │    │ • Keyword score │
│ • CUDA accel    │    │ • Fast retrieval│
└─────────────────┘    └─────────────────┘
     │                          │
     └─────────┬─────────────────┘
               ▼
        ┌─────────────────┐
        │ FlashRank Fusion│
        │ • Neural rerank │
        │ • Score fusion  │
        │ • PageRank boost│
        └─────────────────┘
               ▼
        Final Ranked Results
```

---

## 📈 **Performance**

### **Search Performance**

- **Hybrid Search Response**: <100ms for most queries with CUDA acceleration
- **BGE-M3 Embedding Generation**: ~50ms per query (GPU) / ~200ms (CPU)
- **BM25S Lexical Search**: <10ms across 10,000+ documents
- **FlashRank Neural Reranking**: <50ms for top-K candidate fusion
- **FAISS Vector Search**: <20ms with 1024-dim BGE-M3 embeddings

### **System Performance**

- **Analysis Speed**: 1,000+ files/second on modern hardware
- **Memory Efficient**: Handles 10,000+ file codebases
- **Scalable**: Version isolation prevents data conflicts
- **GPU Acceleration**: Automatic CUDA detection for semantic search

---

## 🛠️ **Development**

```bash
# Install for development
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black src/ tests/

# Type checking
uv run mypy src/
```

---

## 📚 **Data Storage**

The system stores data in versioned directories with separate indexes:

```
%APPDATA%/ifs_cloud_mcp_server/               # Windows
~/.local/share/ifs_cloud_mcp_server/          # Linux/macOS
├── versions/
│   └── 25.1.0/
│       ├── source/              # Extracted files
│       ├── analysis/            # Analysis results
│       ├── ranked.jsonl         # PageRank results
│       ├── bm25s/              # BM25S lexical index
│       │   ├── index.h5        # BM25S index data
│       │   └── corpus.jsonl    # Document corpus
│       └── faiss/              # FAISS semantic index
│           ├── index.faiss     # Vector index
│           ├── embeddings.npy  # BGE-M3 embeddings
│           └── metadata.jsonl  # Document metadata
└── models/                     # Downloaded models
    └── bge-m3/                # BGE-M3 model cache
```

---

## 🔮 **Future Enhancements**

- � **Advanced AI Models**: Integration with newer embedding models (BGE-M4, E5-v3)
- 🔍 **Query Understanding**: Natural language intent classification and query expansion
- 📊 **Visual Analytics**: Interactive dependency graph visualization
- 🌐 **Web Interface**: Enhanced browser-based exploration with search filtering
- 🚀 **Performance**: Further optimization of hybrid search pipeline
- 🎯 **Specialized Search**: Domain-specific search modes (UI patterns, business logic, etc.)

---

<div align="center">

**[⭐ Star this repo](https://github.com/graknol/ifs-cloud-core-mcp-server)** • **[🐛 Report Issues](https://github.com/graknol/ifs-cloud-core-mcp-server/issues)** • **[💬 Discussions](https://github.com/graknol/ifs-cloud-core-mcp-server/discussions)**

_Built with ❤️ for IFS Cloud developers_

</div>
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
