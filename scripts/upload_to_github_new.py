"""
GitHub Release Upload Script for IFS Cloud MCP Server

This script automates the process of uploading safe files (embeddings, indexes,
and PageRank data) to GitHub releases using the GitHub CLI.

The updated version:
- Asks user for release tag (e.g., v1.0.0) instead of deriving from version
- Validates that the tag is greater than the latest release
- Creates individual ZIP files for each version (e.g., "25.1.0.zip")
- Uploads all version ZIPs to a single release

Safe files for publishing:
- FAISS embeddings (vector representations)
- BM25S indexes (bag-of-words, no source code)
- PageRank results (file rankings only)

NOT included:
- Comprehensive analysis files (contain source code samples)
"""

import subprocess
import sys
import json
import time
import zipfile
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from packaging import version

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

    def get_latest_release_version(self) -> Optional[str]:
        """Get the latest release version from GitHub."""
        try:
            result = subprocess.run(
                [
                    "gh",
                    "release",
                    "list",
                    "--repo",
                    self.repo,
                    "--limit",
                    "1",
                    "--json",
                    "tagName",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            releases = json.loads(result.stdout)
            if releases:
                # Remove 'v' prefix if present
                tag = releases[0]["tagName"]
                return tag.lstrip("v")
            return None
        except Exception as e:
            print(f"⚠️ Could not fetch latest release: {e}")
            return None

    def validate_version_tag(
        self, new_tag: str, latest_version: Optional[str] = None
    ) -> bool:
        """Validate that the new version tag is greater than the latest release."""
        # Remove 'v' prefix if present for comparison
        new_version = new_tag.lstrip("v")

        # Basic version format validation
        if not re.match(r"^\d+\.\d+\.\d+$", new_version):
            print(f"❌ Invalid version format: {new_tag}")
            print("   Version should be in format: v1.0.0 or 1.0.0")
            return False

        if latest_version:
            try:
                if version.parse(new_version) <= version.parse(latest_version):
                    print(
                        f"❌ Version {new_tag} must be greater than latest release v{latest_version}"
                    )
                    return False
            except Exception as e:
                print(f"⚠️ Could not compare versions: {e}")
                # Continue anyway if version parsing fails

        return True

    def create_version_zip(
        self, version_name: str, safe_files: List[Tuple[Path, str]]
    ) -> Optional[Path]:
        """Create a ZIP file containing all safe files for a specific version."""
        if not safe_files:
            return None

        # Create ZIP file in the version directory
        version_dir = self.data_dir / version_name
        zip_path = version_dir / f"{version_name}.zip"

        print(f"   📦 Creating ZIP archive: {zip_path.name}")

        try:
            with zipfile.ZipFile(
                zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9
            ) as zipf:
                total_original_size = 0

                for file_path, description in safe_files:
                    if file_path.exists():
                        # Add file to ZIP with relative path structure
                        relative_path = file_path.relative_to(version_dir)
                        zipf.write(file_path, relative_path)

                        file_size = file_path.stat().st_size
                        total_original_size += file_size
                        print(
                            f"      + {relative_path} ({file_size / (1024*1024):.1f} MB)"
                        )
                    else:
                        print(f"      ! {file_path.name} - Not found, skipping")

                # Add a manifest file with version info
                manifest_content = f"""# IFS Cloud MCP Server - Version {version_name}
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
# Contains safe files only (no source code)

Files included:
"""
                for file_path, description in safe_files:
                    if file_path.exists():
                        relative_path = file_path.relative_to(version_dir)
                        manifest_content += f"- {relative_path} - {description}\\n"

                zipf.writestr("README.txt", manifest_content)

            zip_size = zip_path.stat().st_size
            compression_ratio = (
                (1 - zip_size / total_original_size) * 100
                if total_original_size > 0
                else 0
            )

            print(
                f"   ✅ ZIP created: {zip_size / (1024*1024):.1f} MB ({compression_ratio:.1f}% compressed)"
            )
            return zip_path

        except Exception as e:
            print(f"   ❌ Failed to create ZIP: {e}")
            return None

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
            print("\\nPlease install GitHub CLI:")
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
        print("\\n🔐 Authenticating with GitHub...")
        print("This will open a browser window for device authentication.")

        try:
            # Use device flow authentication
            subprocess.run(["gh", "auth", "login", "--web"], check=True)
            print("✅ Authentication successful!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Authentication failed: {e}")
            return False

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
        print(f"\\n🚀 Creating release {tag}...")

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

    def upload_assets(self, tag: str, version_zips: List[Path]) -> bool:
        """Upload version ZIP files to the release."""
        print(f"\\n📤 Uploading assets to release {tag}...")

        if not version_zips:
            print("❌ No ZIP files to upload!")
            return False

        success = True

        for zip_path in version_zips:
            print(f"   Uploading {zip_path.name}...")

            try:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "upload",
                        tag,
                        str(zip_path),
                        "--repo",
                        self.repo,
                        "--clobber",  # Overwrite if exists
                    ],
                    check=True,
                )

                zip_size = zip_path.stat().st_size / (1024 * 1024)
                print(
                    f"   ✅ {zip_path.name} uploaded successfully ({zip_size:.1f} MB)"
                )

            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to upload {zip_path.name}: {e}")
                success = False

        return success

    def generate_release_notes(
        self, tag: str, available_versions: List[str], version_zips: List[Path]
    ) -> str:
        """Generate release notes for the IFS Cloud MCP server release."""

        zip_info = ""
        for zip_path in version_zips:
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            zip_info += f"- **`{zip_path.name}`** - Pre-built indexes for IFS Cloud {zip_path.stem} ({size_mb:.1f} MB)\\n"

        versions_list = ""
        for ver in available_versions:
            versions_list += f"- **{ver}** - Ready for immediate use\\n"

        return f"""# IFS Cloud MCP Server - Release {tag}

## 🚀 Pre-built Search Indexes Release

This release contains **pre-built search indexes and rankings** for multiple IFS Cloud versions, allowing you to use the MCP server without running the resource-intensive embedding generation process.

### 📦 Release Assets

{zip_info}

Each ZIP file contains:
- **FAISS Embeddings**: Vector representations for semantic search
- **BM25S Indexes**: Tokenized lexical search indexes  
- **PageRank Results**: File importance rankings and metadata

### 🔧 Quick Start

1. **Download** the ZIP file for your IFS Cloud version
2. **Extract** to your MCP server data directory
3. **Start** the server - indexes will be automatically detected

```bash
# Extract indexes (example for version 25.1.0)
unzip 25.1.0.zip -d ~/.local/share/ifs_cloud_mcp_server/versions/

# Start the MCP server
python -m src.ifs_cloud_mcp_server.main server --version 25.1.0
```

### 🎯 Available Versions

{versions_list}

### ✅ What's Included (Safe for Distribution)

These files contain **NO source code** and are safe for public distribution:
- **FAISS Embeddings**: Vector representations of semantic meaning only
- **BM25S Indexes**: Tokenized bag-of-words (impossible to extract source code)  
- **PageRank Results**: File importance rankings (names and scores only)

### ❌ What's NOT Included

The following files must be generated locally as they contain source code samples:
- Comprehensive analysis files (`comprehensive_plsql_analysis.json`)
- Raw source file excerpts or content

### 🔧 Manual Generation

If you need to generate these files yourself or customize the analysis:

```bash
# 1. Import IFS Cloud ZIP file  
python -m src.ifs_cloud_mcp_server.main import /path/to/ifscloud-VERSION.zip

# 2. Generate analysis (contains source samples - not published)
python -m src.ifs_cloud_mcp_server.main analyze --version VERSION

# 3. Create embeddings (requires NVIDIA GPU)
python -m src.ifs_cloud_mcp_server.main embed --version VERSION

# 4. Calculate PageRank rankings
python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version VERSION
```

### 🎯 Benefits

- **Multiple Versions**: Download only what you need
- **Instant Setup**: No need to run resource-intensive embedding generation
- **No GPU Required**: Use pre-built indexes without NVIDIA GPU
- **Fast Downloads**: Optimized file sizes for quick deployment
- **Production Ready**: Battle-tested indexes for reliable search

### 📋 System Requirements

- Python 3.11+
- IFS Cloud MCP Server package
- Disk space varies by version (see individual ZIP sizes above)

---

*These indexes were generated using the production embedding framework with PageRank prioritization and comprehensive dependency analysis.*"""

    def interactive_release_flow(self) -> bool:
        """Interactive flow for creating and uploading release."""
        print("\\n" + "=" * 60)
        print("🚀 IFS CLOUD MCP SERVER RELEASE WIZARD")
        print("=" * 60)

        # Find available versions
        available_versions = self.find_available_versions()
        if not available_versions:
            print("\\n❌ No versions with safe files found!")
            print("\\nTo create safe files for a version:")
            print(
                "1. Import ZIP: python -m src.ifs_cloud_mcp_server.main import /path/to/version.zip"
            )
            print(
                "2. Analyze: python -m src.ifs_cloud_mcp_server.main analyze --version VERSION"
            )
            print(
                "3. Embed: python -m src.ifs_cloud_mcp_server.main embed --version VERSION"
            )
            print(
                "4. PageRank: python -m src.ifs_cloud_mcp_server.main calculate-pagerank --version VERSION"
            )
            return False

        # Get existing releases and latest version for comparison
        existing_releases = self.get_existing_releases()
        latest_version = self.get_latest_release_version()

        if existing_releases:
            print("\\n📋 Existing releases:")
            for release in existing_releases[:5]:  # Show last 5
                prerelease_mark = " (prerelease)" if release.get("isPrerelease") else ""
                print(f"   • {release['tagName']}{prerelease_mark}")

        if latest_version:
            print(f"\\n📌 Latest release: v{latest_version}")

        # Show available versions
        print(f"\\n📁 Available versions with safe files:")
        for i, version in enumerate(available_versions, 1):
            print(f"   {i}. {version}")

        # Get release tag from user
        print(f"\\n🏷️ Release Tag:")
        tag_input = input("   Enter release tag (e.g., v1.0.0): ").strip()
        if not tag_input:
            print("❌ Release tag is required!")
            return False

        # Validate version tag
        if not self.validate_version_tag(tag_input, latest_version):
            return False

        # Create ZIP files for each version
        print(f"\\n📦 Creating ZIP archives for all versions...")
        version_zips = []
        total_archive_size = 0

        for version_name in available_versions:
            print(f"\\n🔄 Processing version {version_name}...")

            # Get safe files for this version
            safe_files = self._get_safe_files(version_name)
            if not safe_files:
                print(f"   ⚠️ No safe files found for {version_name}, skipping")
                continue

            # Create ZIP for this version
            zip_path = self.create_version_zip(version_name, safe_files)
            if zip_path:
                version_zips.append(zip_path)
                zip_size = zip_path.stat().st_size
                total_archive_size += zip_size
                print(
                    f"   ✅ Created {zip_path.name} ({zip_size / (1024*1024):.1f} MB)"
                )
            else:
                print(f"   ❌ Failed to create ZIP for {version_name}")

        if not version_zips:
            print("❌ No ZIP files created - cannot proceed with release!")
            return False

        print(
            f"\\n📊 Summary: {len(version_zips)} ZIP files created, total size: {total_archive_size / (1024*1024):.1f} MB"
        )

        # Get release details
        print(f"\\n📝 Release Details:")
        title = input(
            f"   Enter release title (default: 'IFS Cloud MCP Server {tag_input} - Multi-Version Indexes'): "
        ).strip()
        if not title:
            title = f"IFS Cloud MCP Server {tag_input} - Multi-Version Indexes"

        prerelease = input("   Is this a prerelease? (y/N): ").strip().lower() == "y"

        # Generate release notes
        notes = self.generate_release_notes(tag_input, available_versions, version_zips)
        print(f"\\n📄 Generated release notes preview:")
        print("-" * 40)
        print(notes[:800] + "..." if len(notes) > 800 else notes)
        print("-" * 40)

        confirm = (
            input(
                f"\\nProceed with release creation and upload for {tag_input}? (Y/n): "
            )
            .strip()
            .lower()
        )
        if confirm == "n":
            print("❌ Release cancelled by user")
            return False

        # Create release
        if not self.create_release(tag_input, title, notes, prerelease):
            return False

        # Upload ZIP files
        if not self.upload_assets(tag_input, version_zips):
            print("⚠️ Release created but some assets failed to upload")
            return False

        print(
            f"\\n🎉 SUCCESS! Release {tag_input} created and {len(version_zips)} ZIP files uploaded!"
        )
        print(
            f"🔗 View release: https://github.com/{self.repo}/releases/tag/{tag_input}"
        )
        print(f"\\n📊 Release Summary:")
        print(f"   • Tag: {tag_input}")
        print(f"   • Versions included: {', '.join(available_versions)}")
        print(f"   • Total archive size: {total_archive_size / (1024*1024):.1f} MB")
        print(f"   • ZIP files uploaded: {len(version_zips)}")

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
            print("\\n✅ Upload completed successfully!")
            print("\\n📋 Next steps:")
            print("1. Test MCP server with published indexes")
            print("2. Update documentation with new release")
            print("3. Notify users about available pre-built indexes")
        else:
            print("\\n❌ Upload failed or was cancelled")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\\n\\n❌ Upload cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
