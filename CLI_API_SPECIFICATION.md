# IFS Cloud MCP Server CLI API Specification

**Version**: 1.0  
**Target**: AI Agent Implementation of VS Code Extension  
**Purpose**: Complete technical specification for implementing VS Code extension CLI integration

---

## 🎯 **CLI Overview**

The IFS Cloud MCP Server CLI provides comprehensive management of IFS Cloud codebase versions with advanced hybrid search capabilities. The CLI ---

### **9. SERVER Command** *(Default)*designed for programmatic integration and supports both interactive and automated workflows.

### **Core Architecture**

- **Entry Point**: `python -m src.ifs_cloud_mcp_server.main`
- **Alternative**: `uv run python -m src.ifs_cloud_mcp_server.main`
- **Command Structure**: `<entry_point> <command> [options] [arguments]`
- **Exit Codes**: 0 = success, 1 = error
- **Logging**: Structured JSON-compatible logging to stderr

---

## 📋 **Command Reference**

### **1. IMPORT Command**

**Purpose**: Import IFS Cloud ZIP files with automatic version detection

```bash
python -m src.ifs_cloud_mcp_server.main import <zip_file> [OPTIONS]
```

**Arguments:**

- `zip_file` (required): Path to IFS Cloud ZIP file

**Options:**

- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. Auto-detects version from `checkout/fndbas/source/version.txt` inside ZIP
2. Extracts only supported file types (see **Supported Files** section)
3. Creates versioned directory structure in data directory
4. Provides next-step guidance in output

**Output Format:**

```
🔍 Detecting version from path/to/file.zip
📋 Detected version: 25.1.0
✅ Import completed successfully!
📁 Extracted files: /path/to/data/versions/25.1.0/source
🏷️  Version: 25.1.0

Files extracted successfully. To work with this version:
  Analyze:    python -m src.ifs_cloud_mcp_server.main analyze --version "25.1.0"
  MCP Server: python -m src.ifs_cloud_mcp_server.main server --version "25.1.0"
```

**Error Handling:**

- FileNotFoundError: ZIP file doesn't exist
- ValueError: Invalid ZIP or version.txt not found
- Returns exit code 1 on failure

---

### **2. LIST Command**

**Purpose**: List all available IFS Cloud versions and their analysis status

```bash
python -m src.ifs_cloud_mcp_server.main list [OPTIONS]
```

**Options:**

- `--json`: Output in JSON format for programmatic parsing

**Human-Readable Output:**

```
📦 Available IFS Cloud Versions:

25.1.0
  📁 Directory: C:\Users\...\ifs_cloud_mcp_server\versions\25.1.0
  📊 Analysis: ✅ Complete (15,432 files analyzed)
  🔍 BM25S Index: ✅ Available (9,750 documents)
  🧠 FAISS Index: ✅ Available (9,733 embeddings)
  📅 Last Updated: 2025-01-15 14:30:22

24.2.1
  📁 Directory: C:\Users\...\ifs_cloud_mcp_server\versions\24.2.1
  📊 Analysis: ❌ Not analyzed
  🔍 BM25S Index: ❌ Not available
  🧠 FAISS Index: ❌ Not available
```

**JSON Output:**

```json
{
  "versions": [
    {
      "version": "25.1.0",
      "directory": "/path/to/versions/25.1.0",
      "analyzed": true,
      "analysis_file_size": 52428800,
      "file_count": 15432,
      "bm25s_available": true,
      "bm25s_document_count": 9750,
      "faiss_available": true,
      "faiss_embedding_count": 9733,
      "last_updated": "2025-01-15T14:30:22Z"
    }
  ]
}
```

---

### **3. DELETE Command**

**Purpose**: Remove a version and all associated data

```bash
python -m src.ifs_cloud_mcp_server.main delete --version <version> [OPTIONS]
```

**Arguments:**

- `--version` (required): Version identifier to delete

**Options:**

- `--force`: Skip interactive confirmation
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. Validates version exists
2. Shows confirmation prompt (unless --force)
3. Recursively deletes version directory and all indexes
4. Provides confirmation of deletion

**Interactive Output:**

```
⚠️  This will permanently delete version '25.1.0' and ALL associated data:
   📁 Source files: C:\...\versions\25.1.0
   📊 Analysis data
   🔍 Search indexes
   🧠 Embeddings

Are you sure? (y/N):
```

---

### **4. DOWNLOAD Command**

**Purpose**: Download pre-built indexes from GitHub releases for faster setup

```bash
python -m src.ifs_cloud_mcp_server.main download --version <version> [OPTIONS]
```

**Arguments:**

- `--version` (required): Version identifier to download indexes for

**Options:**

- `--force`: Overwrite existing indexes
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. Validates that version directory exists (must import first)
2. Checks GitHub releases for matching index files
3. Downloads BM25S and FAISS indexes if available
4. Extracts indexes to appropriate directories
5. Provides confirmation and next steps

**Output:**

```
🔄 Starting download for version 25.1.0...
🔍 Checking GitHub releases for version 25.1.0...
✅ Found release: IFS Cloud 25.1.0 Pre-built Indexes
📥 Downloading BM25S index (45,328,992 bytes)...
✅ BM25S index extracted to C:\...\versions\25.1.0\bm25s
📥 Downloading FAISS index (124,857,344 bytes)...
✅ FAISS index extracted to C:\...\versions\25.1.0\faiss
🎉 Successfully downloaded indexes for version 25.1.0
    BM25S: C:\...\versions\25.1.0\bm25s
    FAISS: C:\...\versions\25.1.0\faiss

✅ Download completed successfully for version 25.1.0!
    Ready to start MCP server:
    python -m src.ifs_cloud_mcp_server.main server --version "25.1.0"
```

**Error Handling:**

- Version directory doesn't exist → Guidance to import first
- No matching GitHub release → Lists available releases, suggests local generation
- Network errors → Clear error messages with fallback suggestions
- Missing or invalid index files → Detailed asset information

**GitHub Release Format:**

- **Release Tag**: `indexes-{version}` (e.g., `indexes-25.1.0`)
- **Assets**: `{version}-bm25s.zip`, `{version}-faiss.zip`
- **Repository**: `graknol/ifs-cloud-core-mcp-server`

---

### **5. ANALYZE Command**

**Purpose**: Generate comprehensive codebase analysis

```bash
python -m src.ifs_cloud_mcp_server.main analyze --version <version> [OPTIONS]
```

**Arguments:**

- `--version` (required): Version identifier to analyze

**Options:**

- `--max-files INTEGER`: Limit analysis to N files (testing)
- `--force`: Overwrite existing analysis
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. Scans source directory for supported files
2. Extracts API calls, dependencies, procedures, functions
3. Builds inter-file dependency graph
4. Creates comprehensive analysis JSON file
5. Updates version metadata

**Output:**

```
🔍 Starting comprehensive analysis for version 25.1.0...
📁 Source directory: /path/to/versions/25.1.0/source
📊 Found 15,432 supported files

Processing files: 100%|████████████| 15432/15432 [02:34<00:00, 125.7files/s]

✅ Analysis completed successfully!
📄 Analysis file: /path/to/analysis/comprehensive_plsql_analysis.json
📊 Total files analyzed: 15,432
🔗 Dependencies found: 89,541
⏱️  Processing time: 2m 34s
```

---

### **6. CALCULATE-PAGERANK Command**

**Purpose**: Calculate PageRank importance scores based on dependencies

```bash
python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version <version> [OPTIONS]
```

**Arguments:**

- `--version` (required): Version identifier

**Options:**

- `--damping-factor FLOAT`: PageRank damping factor (default: 0.85)
- `--max-iterations INTEGER`: Maximum iterations (default: 100)
- `--convergence-threshold FLOAT`: Convergence threshold (default: 1e-6)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. Loads dependency graph from analysis
2. Runs PageRank algorithm with specified parameters
3. Ranks files by importance in dependency network
4. Saves ranked results to JSONL file

**Output:**

```
📊 Starting PageRank calculation for version 25.1.0...
📈 Building dependency graph from 15,432 files
🔗 Graph contains 89,541 directed edges

Running PageRank algorithm:
  🎯 Damping factor: 0.85
  🔄 Max iterations: 100
  📏 Convergence threshold: 1e-06

Iteration 45: Converged (delta: 8.23e-07)

✅ PageRank calculation completed!
📄 Results saved: /path/to/analysis/ranked.jsonl
🏆 Top 5 most important files:
  1. FndSession.plsql (score: 0.0342)
  2. Site_API.plsql (score: 0.0298)
  3. Company_API.plsql (score: 0.0276)
  4. User_Default_API.plsql (score: 0.0245)
  5. Security_SYS.plsql (score: 0.0221)
```

---

### **7. EMBED Command**

**Purpose**: Create semantic embeddings using BGE-M3 model

```bash
python -m src.ifs_cloud_mcp_server.main embed --version <version> [OPTIONS]
```

**Arguments:**

- `--version` (required): Version identifier

**Options:**

- `--model STRING`: Ollama model for summarization (default: phi4-mini:3.8b-q4_K_M)
- `--max-files INTEGER`: Limit to N files for testing
- `--no-resume`: Start fresh, ignore checkpoints
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. **Phase 1**: Ollama summarization of code files
2. **Phase 2**: BGE-M3 embedding generation
3. **Phase 3**: FAISS index creation
4. Supports checkpointing and resume functionality
5. GPU acceleration when available

**Output:**

```
🚀 EMBEDDING GENERATION STARTED
📊 Version: 25.1.0 | Model: phi4-mini:3.8b-q4_K_M | Max files: unlimited

🏗️  PHASE 1: AI Summarization (Ollama)
   📄 Processing 15,432 files with Ollama model
   🤖 Generated summaries: 12,450/15,432 [80%]

🧠 PHASE 2: Embedding Generation (BGE-M3)
   🔧 Loading BGE-M3 model (CUDA detected)
   📊 Creating 1024-dim embeddings for 12,450 summaries
   ⚡ GPU acceleration: Active

🔍 PHASE 3: FAISS Index Creation
   💾 Building FAISS index with 9,733 embeddings
   📁 Saved to: /path/to/versions/25.1.0/faiss/

✅ EMBEDDING GENERATION COMPLETED
⏱️  Total time: 8m 42s
🎯 Ready for semantic search
```

---

### **8. REINDEX-BM25S Command**

**Purpose**: Rebuild BM25S lexical search index

```bash
python -m src.ifs_cloud_mcp_server.main reindex-bm25s --version <version> [OPTIONS]
```

**Arguments:**

- `--version` (required): Version identifier

**Options:**

- `--max-files INTEGER`: Limit to N files for testing
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. Loads analysis data
2. Preprocesses source code for lexical search
3. Creates BM25S index with enhanced tokenization
4. Saves index to version-specific directory

**Output:**

```
🔄 Starting BM25S index rebuild for version 25.1.0...
📁 Loading analysis from: /path/to/analysis.json
📊 Processing 15,432 files for indexing

🔧 Enhanced preprocessing:
   📝 Tokenizing source code
   🔍 Extracting API names and keywords
   📊 Creating BM25S corpus

💾 Building BM25S index...
✅ BM25S reindexing completed!
📁 Index saved: /path/to/versions/25.1.0/bm25s/
📊 Indexed documents: 9,750
🎯 Ready for lexical search
```

---

### **8. SERVER Command** _(Default)_

**Purpose**: Start MCP server for AI agent integration

```bash
python -m src.ifs_cloud_mcp_server.main server --version <version> [OPTIONS]
python -m src.ifs_cloud_mcp_server.main --version <version>  # Default command
```

**Arguments:**

- `--version` (required): Version identifier to serve

**Options:**

- `--name STRING`: Server name (default: ifs-cloud-mcp-server)
- `--transport STRING`: Transport type (default: stdio)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)

**Behavior:**

1. Validates version exists and is analyzed
2. Initializes hybrid search engine (BM25S + FAISS + FlashRank)
3. Loads BGE-M3 model for semantic search
4. Starts MCP server with stdio transport
5. Provides search tools to AI agents

**Startup Output:**

```
🚀 Starting IFS Cloud MCP Server
📊 Version: 25.1.0
📁 Directory: C:\...\ifs_cloud_mcp_server\versions\25.1.0

🔍 Initializing Hybrid Search Engine:
   📊 Loading BM25S index... ✅ (9,750 documents)
   🧠 Loading FAISS index... ✅ (9,733 embeddings)
   🤖 Loading BGE-M3 model... ✅ (CUDA acceleration)
   ⚡ Initializing FlashRank... ✅

🎯 MCP Tools Available:
   • search_ifs_codebase (hybrid search)
   • search_ifs_semantic (BGE-M3 embeddings)
   • search_ifs_lexical (BM25S exact match)

✅ Server ready on stdio transport
```

---

## 📁 **Directory Structure**

### **Data Directory Locations**

- **Windows**: `%APPDATA%\ifs_cloud_mcp_server`
- **macOS**: `~/Library/Application Support/ifs_cloud_mcp_server`
- **Linux**: `~/.local/share/ifs_cloud_mcp_server`

### **Version Directory Layout**

```
{data_dir}/
├── versions/
│   └── {version}/
│       ├── source/                    # Extracted IFS files
│       │   ├── fndbas/               # Foundation components
│       │   ├── accrul/               # Accounting components
│       │   ├── order/                # Order management
│       │   └── ...                   # Other IFS modules
│       ├── analysis/
│       │   └── comprehensive_plsql_analysis.json
│       ├── ranked.jsonl              # PageRank results
│       ├── bm25s/                    # BM25S lexical index
│       │   ├── index.h5             # BM25S index data
│       │   └── corpus.jsonl         # Document corpus
│       └── faiss/                    # FAISS semantic index
│           ├── index.faiss          # Vector index
│           ├── embeddings.npy       # BGE-M3 embeddings
│           └── metadata.jsonl       # Document metadata
└── models/                           # Downloaded models
    └── bge-m3/                      # BGE-M3 model cache
```

---

## 📄 **Supported File Types**

The CLI processes the following IFS Cloud file types:

| Extension     | Purpose                   | Analysis Features                         |
| ------------- | ------------------------- | ----------------------------------------- |
| `.entity`     | Data Entity Definitions   | Entity relationships, attributes          |
| `.plsql`      | PL/SQL Business Logic     | API calls, procedures, functions, queries |
| `.views`      | Database Views            | View definitions, column mappings         |
| `.storage`    | Storage Definitions       | Table mappings, database structure        |
| `.fragment`   | Full-Stack Components     | UI-to-data integration patterns           |
| `.projection` | Data Access Layer         | Projection queries, actions, CRUD         |
| `.client`     | User Interface Components | UI patterns, commands, navigation         |

**File Filtering**: During import, only files with these extensions are extracted and processed.

---

## 🔧 **Integration Guidelines for VS Code Extension**

### **Command Execution**

```typescript
// Recommended approach for VS Code extension
const executeCommand = async (
  command: string,
  args: string[]
): Promise<CommandResult> => {
  const fullCommand = [
    "python",
    "-m",
    "src.ifs_cloud_mcp_server.main",
    command,
    ...args,
  ];

  const result = await exec(fullCommand.join(" "), {
    cwd: workspaceRoot,
    encoding: "utf-8",
  });

  return {
    exitCode: result.code || 0,
    stdout: result.stdout,
    stderr: result.stderr,
  };
};
```

### **Progress Monitoring**

- Monitor stderr for structured logging output
- Parse progress indicators from log messages
- Handle long-running commands (analyze, embed) with progress bars

### **Error Handling**

```typescript
interface CLIError {
  exitCode: number;
  message: string;
  command: string;
  suggestions?: string[];
}

const handleCLIError = (command: string, result: CommandResult): CLIError => {
  // Parse structured error messages from stderr
  // Extract suggestions for next steps
  // Map common error patterns to user-friendly messages
};
```

### **Version Management**

```typescript
interface Version {
  version: string;
  directory: string;
  analyzed: boolean;
  bm25s_available: boolean;
  faiss_available: boolean;
  file_count?: number;
  last_updated: string;
}

const listVersions = async (): Promise<Version[]> => {
  const result = await executeCommand("list", ["--json"]);
  return JSON.parse(result.stdout).versions;
};
```

### **Workflow Automation**

```typescript
// Fast setup workflow - try download first (recommended)
const setupVersionFast = async (zipPath: string): Promise<void> => {
  // 1. Import ZIP file
  await executeCommand("import", [zipPath]);

  // 2. Get version from output
  const versions = await listVersions();
  const latestVersion = versions[0].version;

  // 3. Try to download pre-built indexes first
  const downloadResult = await executeCommand("download", [
    "--version",
    latestVersion,
  ]);

  if (downloadResult.exitCode === 0) {
    // Success! Ready to use immediately
    console.log("✅ Fast setup complete - ready for MCP server");
    return;
  }

  // 4. Fallback to local generation if download fails
  console.log("📦 Download failed, generating indexes locally...");
  await executeCommand("analyze", ["--version", latestVersion]);
  await executeCommand("calculate-pagerank", ["--version", latestVersion]);
  await executeCommand("reindex-bm25s", ["--version", latestVersion]);
  // Optional: await executeCommand('embed', ['--version', latestVersion]);
};

// Complete setup workflow for new version (local generation)
const setupVersion = async (zipPath: string): Promise<void> => {
  // 1. Import
  await executeCommand("import", [zipPath]);

  // 2. Get version from output
  const versions = await listVersions();
  const latestVersion = versions[0].version;

  // 3. Analyze (with progress monitoring)
  await executeCommand("analyze", ["--version", latestVersion]);

  // 4. Calculate PageRank
  await executeCommand("calculate-pagerank", ["--version", latestVersion]);

  // 5. Build BM25S index
  await executeCommand("reindex-bm25s", ["--version", latestVersion]);

  // 6. Optional: Create embeddings (resource intensive)
  // await executeCommand('embed', ['--version', latestVersion]);
};
```

---

## 🚀 **MCP Server Integration**

### **Server Lifecycle**

1. **Validation**: Ensure version exists and is analyzed
2. **Initialization**: Load search indexes and models
3. **Runtime**: Provide search tools via MCP protocol
4. **Cleanup**: Graceful shutdown and resource cleanup

### **Search Tools Available**

- `search_ifs_codebase`: Hybrid search with semantic + lexical
- `search_ifs_semantic`: Pure semantic search with BGE-M3
- `search_ifs_lexical`: Pure lexical search with BM25S

### **Performance Characteristics**

- **Cold Start**: 10-30 seconds (model loading)
- **Search Response**: <100ms with GPU, <500ms without
- **Memory Usage**: ~2GB for full indexes + models
- **CUDA Support**: Automatic detection and acceleration

---

## 📋 **CLI Return Codes & Error Handling**

### **Exit Codes**

- `0`: Success
- `1`: General error (file not found, validation failed, etc.)

### **Common Error Patterns**

```
❌ Import failed: <specific error>
❌ Version error: <validation issue>
❌ Analysis failed: <processing error>
❌ Server error: <runtime issue>
```

### **Structured Logging**

- All logging goes to stderr
- Structured format for programmatic parsing
- Progress indicators in standardized format
- Error messages include actionable suggestions

---

This specification provides complete technical details for implementing a VS Code extension that integrates with the IFS Cloud MCP Server CLI. The CLI is designed for programmatic use with comprehensive error handling, structured output, and clear integration patterns.
