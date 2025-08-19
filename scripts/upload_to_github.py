"""
GitHub Release Upload Script for IFS Cloud MCP Server

This script automates the process of uploading safe files (embeddings, indexes, 
and PageRank data) to GitHub releases using the GitHub CLI.

Safe files for publishing:
- FAISS embeddings (vector representations)
- BM25S indexes (bag-of-### 🛠️ **Manual Generation**ords, no source code)
- PageRank results (file rankings only)

NOT included:
- Comprehensive analysis files (contain source code samples)
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from ifs_cloud_mcp_server.directory_utils import get_data_directory


class GitHubReleaseUploader:
    """Upload safe MCP server files to GitHub releases using GitHub CLI."""

    def __init__(self, repo: str = "graknol/ifs-cloud-core-mcp-server"):
        """Initialize the uploader."""
        self.repo = repo
        self.data_dir = get_data_directory() / "versions"

    def find_available_versions(self) -> List[str]:
        """Find all available versions with safe files to upload."""
        if not self.data_dir.exists():
            return []
        
        versions = []
        for version_dir in self.data_dir.iterdir():
            if version_dir.is_dir() and self._has_safe_files(version_dir):
                versions.append(version_dir.name)
        
        return sorted(versions)
    
    def _has_safe_files(self, version_dir: Path) -> bool:
        """Check if version directory has safe files to upload."""
        safe_files = self._get_safe_files(version_dir.name)
        return len(safe_files) > 0

    def _get_safe_files(self, version: str) -> List[Tuple[Path, str]]:
        """Get list of safe files for a version with their descriptions."""
        version_dir = self.data_dir / version
        safe_files = []

        # FAISS embeddings and indexes (safe - just vectors)
        faiss_dir = version_dir / "faiss"
        if faiss_dir.exists():
            for faiss_file in faiss_dir.glob("*"):
                if faiss_file.is_file():
                    if faiss_file.suffix == ".faiss":
                        desc = f"FAISS vector index for {version} (semantic search)"
                    elif faiss_file.suffix == ".pkl":
                        desc = f"FAISS metadata for {version} (embeddings)"
                    elif faiss_file.suffix == ".npy":
                        desc = f"FAISS embeddings for {version} (vectors)"
                    else:
                        desc = f"FAISS support file for {version}"
                    safe_files.append((faiss_file, desc))

        # BM25S indexes (safe - bag of words, no source code)
        bm25s_dir = version_dir / "bm25s"
        if bm25s_dir.exists():
            for bm25s_file in bm25s_dir.glob("*"):
                if bm25s_file.is_file():
                    if "index" in bm25s_file.name:
                        desc = f"BM25S lexical search index for {version}"
                    elif "corpus" in bm25s_file.name:
                        desc = f"BM25S tokenized corpus for {version}"
                    elif "metadata" in bm25s_file.name:
                        desc = f"BM25S index metadata for {version}"
                    else:
                        desc = f"BM25S support file for {version}"
                    safe_files.append((bm25s_file, desc))

        # PageRank results (safe - just file names and rankings)
        pagerank_file = version_dir / "ranked.jsonl"
        if pagerank_file.exists():
            desc = f"PageRank file rankings for {version} (importance scores)"
            safe_files.append((pagerank_file, desc))

        return safe_files

    def check_github_cli(self) -> bool:
        """Check if GitHub CLI is installed and available."""
        try:
            result = subprocess.run(
                ["gh", "--version"], capture_output=True, text=True, check=True
            )
            print(f"✅ GitHub CLI found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ GitHub CLI not found!")
            print("\nPlease install GitHub CLI:")
            print("- Windows: winget install GitHub.cli")
            print("- macOS: brew install gh")
            print("- Linux: https://cli.github.com/manual/installation")
            return False

    def check_authentication(self) -> bool:
        """Check if user is authenticated with GitHub CLI."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"], capture_output=True, text=True, check=True
            )
            print("✅ Already authenticated with GitHub CLI")
            return True
        except subprocess.CalledProcessError:
            print("⚠️ Not authenticated with GitHub CLI")
            return False

    def authenticate(self) -> bool:
        """Authenticate with GitHub using device flow."""
        print("\n🔐 Authenticating with GitHub...")
        print("This will open a browser window for device authentication.")

        try:
            # Use device flow authentication
            subprocess.run(["gh", "auth", "login", "--web"], check=True)
            print("✅ Authentication successful!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Authentication failed: {e}")
            return False

    def check_safe_files(self, version: str) -> Tuple[bool, List[Tuple[Path, str]], float]:
        """Check if safe files exist for a version and get their info."""
        print(f"\n📁 Checking safe files for version {version}...")

        safe_files = self._get_safe_files(version)
        total_size = 0

        if not safe_files:
            print(f"   ❌ No safe files found for version {version}")
            print(f"   Please run the embedding pipeline first:")
            print(f"   1. Import: python -m src.ifs_cloud_mcp_server.main import /path/to/{version}.zip")
            print(f"   2. Analyze: python -m src.ifs_cloud_mcp_server.main analyze --version {version}")
            print(f"   3. Embed: python -m src.ifs_cloud_mcp_server.main embed --version {version}")
            print(f"   4. PageRank: python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version {version}")
            return False, [], 0

        for file_path, description in safe_files:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"   ✅ {file_path.name} ({size_mb:.1f} MB) - {description}")
            else:
                print(f"   ❌ {file_path.name} - Not found!")

        print(f"\n📊 Total upload size: {total_size:.1f} MB")
        if total_size > 100:
            print("⚠️ Large upload detected - this may take some time")
        elif total_size < 1:
            print("⚠️ Very small upload - files might not be fully generated")

        return True, safe_files, total_size

    def get_existing_releases(self) -> List[Dict]:
        """Get list of existing releases."""
        try:
            result = subprocess.run(
                [
                    "gh",
                    "release",
                    "list",
                    "--repo",
                    self.repo,
                    "--json",
                    "name,tagName,isPrerelease",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except Exception as e:
            print(f"⚠️ Could not fetch existing releases: {e}")
            return []

    def create_release(
        self, tag: str, title: str, notes: str, prerelease: bool = False
    ) -> bool:
        """Create a new GitHub release."""
        print(f"\n🚀 Creating release {tag}...")

        try:
            cmd = [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                self.repo,
                "--title",
                title,
                "--notes",
                notes,
            ]

            if prerelease:
                cmd.append("--prerelease")

            subprocess.run(cmd, check=True)
            print(f"✅ Release {tag} created successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create release: {e}")
            return False

    def upload_assets(self, version: str, safe_files: List[Tuple[Path, str]]) -> bool:
        """Upload safe files to the release."""
        tag = f"v{version}"
        print(f"\n📤 Uploading assets to release {tag}...")

        success = True

        for file_path, description in safe_files:
            print(f"   Uploading {file_path.name}...")

            try:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "upload",
                        tag,
                        str(file_path),
                        "--repo",
                        self.repo,
                        "--clobber",  # Overwrite if exists
                    ],
                    check=True,
                )

                print(f"   ✅ {file_path.name} uploaded successfully")

            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to upload {file_path.name}: {e}")
                success = False

        return success

    def generate_release_notes(self, version: str, safe_files: List[Tuple[Path, str]], total_size: float) -> str:
        """Generate release notes for the IFS Cloud MCP server release."""
        
        # Categorize files
        faiss_files = [f for f in safe_files if "/faiss/" in str(f[0])]
        bm25s_files = [f for f in safe_files if "/bm25s/" in str(f[0])]
        pagerank_files = [f for f in safe_files if "ranked.jsonl" in str(f[0])]
        
        return f"""# IFS Cloud MCP Server - Version {version}

## 🚀 Pre-built Search Indexes Release

This release contains **pre-built search indexes and rankings** for IFS Cloud version {version}, allowing you to use the MCP server without running the resource-intensive embedding generation process.

### � Release Assets ({total_size:.1f} MB total)

#### 🧠 **Semantic Search (FAISS)**
{"".join([f"- `{f[0].name}` - {f[1]}\\n" for f in faiss_files]) if faiss_files else "- *(No FAISS files available)*\\n"}

#### 🔍 **Lexical Search (BM25S)**  
{"".join([f"- `{f[0].name}` - {f[1]}\\n" for f in bm25s_files]) if bm25s_files else "- *(No BM25S files available)*\\n"}

#### � **File Rankings (PageRank)**
{"".join([f"- `{f[0].name}` - {f[1]}\\n" for f in pagerank_files]) if pagerank_files else "- *(No PageRank files available)*\\n"}

### 🔧 **Usage**

The MCP server will automatically download and use these pre-built indexes:

```bash
# Start the MCP server for this version
python -m src.ifs_cloud_mcp_server.main server --version {version}

# The server will automatically:
# 1. Check for local indexes first
# 2. Download from this release if missing  
# 3. Provide instant search capabilities
```

### ✅ **What's Included (Safe for Distribution)**

These files contain **NO source code** and are safe for public distribution:
- **FAISS Embeddings**: Vector representations of semantic meaning only
- **BM25S Indexes**: Tokenized bag-of-words (impossible to extract source code)  
- **PageRank Results**: File importance rankings (names and scores only)

### ❌ **What's NOT Included**

The following files must be generated locally as they contain source code samples:
- Comprehensive analysis files (`comprehensive_plsql_analysis.json`)
- Raw source file excerpts or content

### �️ **Manual Generation**

If you need to generate these files yourself or customize the analysis:

```bash
# 1. Import IFS Cloud ZIP file  
python -m src.ifs_cloud_mcp_server.main import /path/to/ifscloud-{version}.zip

# 2. Generate analysis (contains source samples - not published)
python -m src.ifs_cloud_mcp_server.main analyze --version {version}

# 3. Create embeddings (requires NVIDIA GPU)
python -m src.ifs_cloud_mcp_server.main embed --version {version}

# 4. Calculate PageRank rankings
python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version {version}
```

### 🎯 **Benefits**

- **Instant Setup**: No need to run resource-intensive embedding generation
- **No GPU Required**: Use pre-built indexes without NVIDIA GPU
- **Fast Downloads**: Optimized file sizes for quick deployment
- **Production Ready**: Battle-tested indexes for reliable search

### 📋 **System Requirements**

- Python 3.11+
- IFS Cloud MCP Server package
- ~{total_size:.0f}MB disk space for indexes

---

*These indexes were generated using the production embedding framework with PageRank prioritization and comprehensive dependency analysis.*"""

    def interactive_release_flow(self) -> bool:
        """Interactive flow for creating and uploading release."""
        print("\n" + "=" * 60)
        print("🚀 IFS CLOUD MCP SERVER RELEASE WIZARD")
        print("=" * 60)

        # Find available versions
        available_versions = self.find_available_versions()
        if not available_versions:
            print("\n❌ No versions with safe files found!")
            print("\nTo create safe files for a version:")
            print("1. Import ZIP: python -m src.ifs_cloud_mcp_server.main import /path/to/version.zip")
            print("2. Analyze: python -m src.ifs_cloud_mcp_server.main analyze --version VERSION")
            print("3. Embed: python -m src.ifs_cloud_mcp_server.main embed --version VERSION")
            print("4. PageRank: python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version VERSION")
            return False

        # Get existing releases for reference
        existing_releases = self.get_existing_releases()
        if existing_releases:
            print("\n📋 Existing releases:")
            for release in existing_releases[:5]:  # Show last 5
                prerelease_mark = " (prerelease)" if release.get("isPrerelease") else ""
                print(f"   • {release['tagName']}{prerelease_mark}")

        # Show available versions
        print(f"\n📁 Available versions with safe files:")
        for i, version in enumerate(available_versions, 1):
            print(f"   {i}. {version}")

        # Get version selection
        print(f"\n📝 Version Selection:")
        version_choice = input(f"   Select version (1-{len(available_versions)}) or enter version name: ").strip()
        
        # Parse version choice
        if version_choice.isdigit():
            idx = int(version_choice) - 1
            if 0 <= idx < len(available_versions):
                version = available_versions[idx]
            else:
                print("❌ Invalid selection!")
                return False
        else:
            version = version_choice
            if version not in available_versions:
                print(f"❌ Version '{version}' not found or has no safe files!")
                return False

        # Check safe files for selected version
        files_exist, safe_files, total_size = self.check_safe_files(version)
        if not files_exist:
            return False

        # Get release details
        tag = f"v{version}"
        title = input(f"   Enter release title (default: 'IFS Cloud MCP Server {version} - Search Indexes'): ").strip()
        if not title:
            title = f"IFS Cloud MCP Server {version} - Search Indexes"

        prerelease = input("   Is this a prerelease? (y/N): ").strip().lower() == "y"

        # Generate release notes
        notes = self.generate_release_notes(version, safe_files, total_size)
        print(f"\n📄 Generated release notes preview:")
        print("-" * 40)
        print(notes[:800] + "..." if len(notes) > 800 else notes)
        print("-" * 40)

        confirm = (
            input(f"\nProceed with release creation and upload for {version}? (Y/n): ").strip().lower()
        )
        if confirm == "n":
            print("❌ Release cancelled by user")
            return False

        # Create release
        if not self.create_release(tag, title, notes, prerelease):
            return False

        # Upload assets
        if not self.upload_assets(version, safe_files):
            print("⚠️ Release created but some assets failed to upload")
            return False

        print(f"\n🎉 SUCCESS! Release {tag} created and {len(safe_files)} assets uploaded!")
        print(f"🔗 View release: https://github.com/{self.repo}/releases/tag/{tag}")
        print(f"\n📊 Release Summary:")
        print(f"   • Version: {version}")
        print(f"   • Total size: {total_size:.1f} MB")
        print(f"   • Files uploaded: {len(safe_files)}")

        return True

    def run(self) -> bool:
        """Run the complete upload process."""
        print("🚀 IFS Cloud MCP Server - GitHub Release Upload")
        print("=" * 50)
        print("📋 Uploading safe files only:")
        print("   ✅ FAISS embeddings (vector representations)")
        print("   ✅ BM25S indexes (tokenized search)")
        print("   ✅ PageRank results (file rankings)")
        print("   ❌ Source code analysis (must be generated locally)")
        print("=" * 50)

        # 1. Check GitHub CLI
        if not self.check_github_cli():
            return False

        # 2. Check authentication
        if not self.check_authentication():
            if not self.authenticate():
                return False

        # 3. Interactive release flow (includes file checking)
        return self.interactive_release_flow()


def main():
    """Main entry point."""
    uploader = GitHubReleaseUploader()

    try:
        success = uploader.run()

        if success:
            print("\n✅ Upload completed successfully!")
            print("\n📋 Next steps:")
            print("1. Test MCP server with published indexes")
            print("2. Update documentation with new release")
            print("3. Notify users about available pre-built indexes")
        else:
            print("\n❌ Upload failed or was cancelled")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n❌ Upload cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
