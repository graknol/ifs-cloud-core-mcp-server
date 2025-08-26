#!/usr/bin/env python3
"""
Dependency verification script for PLSQL analysis
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version compatibility"""
    print(f"🐍 Python version: {sys.version}")

    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    print("✅ Python version OK")
    return True


def check_dependencies():
    """Check if required dependencies are available"""
    required = {"requests": "requests", "openai": "openai", "anthropic": "anthropic"}

    available = {}
    missing = []

    for package, import_name in required.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            available[package] = version
            print(f"✅ {package}: {version}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}: Not found")

    if missing:
        print(f"\n📦 Missing dependencies: {', '.join(missing)}")
        print("💡 Install with:")
        print(f"   uv add {' '.join(missing)}")
        return False

    print("✅ All dependencies available")
    return True


def check_api_keys():
    """Check API key availability"""
    import os

    keys = {"OPENAI_API_KEY": "OpenAI GPT-4", "ANTHROPIC_API_KEY": "Claude Sonnet"}

    available_apis = []

    for env_var, api_name in keys.items():
        if os.getenv(env_var):
            print(f"✅ {api_name}: API key found")
            available_apis.append(api_name)
        else:
            print(f"ℹ️  {api_name}: No API key (optional)")

    print("✅ GitHub Copilot: Uses device authentication")
    available_apis.append("GitHub Copilot")

    return available_apis


def check_data_directory():
    """Check if top_10 directory exists with data"""
    top_10_dir = Path("top_10")

    if not top_10_dir.exists():
        print("❌ top_10 directory not found")
        print("💡 Run: python extract_top_10_per_module.py")
        return False

    # Count modules and files
    modules = [d for d in top_10_dir.iterdir() if d.is_dir()]
    total_files = 0

    for module_dir in modules:
        plsql_files = list(module_dir.glob("*.plsql"))
        total_files += len(plsql_files)

    print(f"✅ Data directory: {len(modules)} modules, {total_files} PLSQL files")

    if total_files == 0:
        print("❌ No PLSQL files found in top_10")
        return False

    return True


def estimate_processing():
    """Estimate processing requirements"""
    top_10_dir = Path("top_10")

    if not top_10_dir.exists():
        return

    modules = [d for d in top_10_dir.iterdir() if d.is_dir()]

    # Estimate files to process (20 per module max)
    estimated_files = min(
        len(modules) * 20,
        sum(len(list(module_dir.glob("*.plsql"))) for module_dir in modules),
    )

    print(f"\n📊 Processing Estimates:")
    print(f"   • Modules: {len(modules)}")
    print(f"   • Files to process: ~{estimated_files}")
    print(f"   • API calls: ~{estimated_files}")
    print(f"   • Estimated time: {estimated_files * 5 // 60} minutes (with delays)")


def main():
    print("🔍 PLSQL Analysis Dependencies Check")
    print("=" * 50)

    success = True

    # Check Python version
    if not check_python_version():
        success = False

    print("\n" + "-" * 30)

    # Check dependencies
    if not check_dependencies():
        success = False

    print("\n" + "-" * 30)

    # Check API keys
    print("🔐 API Keys:")
    available_apis = check_api_keys()

    print("\n" + "-" * 30)

    # Check data directory
    print("📁 Data Directory:")
    if not check_data_directory():
        success = False

    # Processing estimates
    estimate_processing()

    print("\n" + "=" * 50)

    if success:
        print("🎉 System ready for PLSQL analysis!")
        print(f"\n🚀 Available APIs: {', '.join(available_apis)}")
        print("\n📋 Ready to run:")
        print("   • python test_copilot_api.py")
        print("   • python run_plsql_analysis.py --api copilot")
        print("   • python analyze_plsql_procedures.py")
    else:
        print("❌ System not ready - fix issues above first")
        sys.exit(1)


if __name__ == "__main__":
    main()
