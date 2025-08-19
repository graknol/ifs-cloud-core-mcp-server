"""IFS Cloud MCP Server with Tantivy search engine integration and production database metadata extraction.

A high-performance Model Context Protocol server for IFS Cloud codebases,
providing enterprise-grade search capabilities, direct database metadata extraction,
and Navigator GUI-to-backend mappings from live IFS Cloud environments.
"""

__version__ = "0.1.0"
__author__ = "IFS Cloud Team"

# Optional imports - only import if dependencies are available
__all__ = []

try:
    from .server_fastmcp import IFSCloudMCPServer

    __all__.append("IFSCloudMCPServer")
except ImportError:
    pass

try:
    from .embedding_processor import ProductionEmbeddingFramework

    __all__.append("ProductionEmbeddingFramework")
except ImportError:
    pass
