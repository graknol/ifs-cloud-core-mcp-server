#!/usr/bin/env python3
"""
Comprehensive PLSQL Procedure Analysis Runner
This script provides multiple API options and orchestrates the analysis workflow
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime


def check_and_install_dependencies():
    """Check and optionally install required packages"""
    required_packages = {
        "requests": "requests",
        "openai": "openai",
        "anthropic": "anthropic",
    }

    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/N): ").lower().strip()
        if install == "y":
            import subprocess

            for package in missing_packages:
                print(f"📦 Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print("❌ Cannot continue without required packages")
            return False

    return True


def get_available_apis():
    """Determine which APIs are available"""
    apis = {}

    # GitHub Copilot (always available since we have the implementation)
    apis["copilot"] = {
        "available": True,
        "requires_auth": True,
        "description": "GitHub Copilot API (requires GitHub authentication)",
    }

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai

            apis["openai"] = {
                "available": True,
                "requires_auth": False,
                "description": "OpenAI GPT-4 API",
            }
        except ImportError:
            apis["openai"] = {
                "available": False,
                "requires_auth": False,
                "description": "OpenAI GPT-4 API (package not installed)",
            }
    else:
        apis["openai"] = {
            "available": False,
            "requires_auth": False,
            "description": "OpenAI GPT-4 API (API key not set)",
        }

    # Claude/Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            apis["claude"] = {
                "available": True,
                "requires_auth": False,
                "description": "Claude Sonnet API",
            }
        except ImportError:
            apis["claude"] = {
                "available": False,
                "requires_auth": False,
                "description": "Claude Sonnet API (package not installed)",
            }
    else:
        apis["claude"] = {
            "available": False,
            "requires_auth": False,
            "description": "Claude Sonnet API (API key not set)",
        }

    return apis


def run_copilot_analysis(args):
    """Run analysis using GitHub Copilot API"""
    from analyze_plsql_procedures import PLSQLProcedureAnalyzer

    print("🚀 Starting analysis with GitHub Copilot API")
    analyzer = PLSQLProcedureAnalyzer(
        top_10_dir=args.top_10_dir, output_dir=args.output_dir + "_copilot"
    )

    analyzer.run_analysis(
        max_files_per_module=args.max_files, delay_between_calls=args.delay
    )


def run_openai_analysis(args):
    """Run analysis using OpenAI API"""
    from api_alternatives import PLSQLAnalyzerOpenAI

    print("🚀 Starting analysis with OpenAI GPT-4")
    analyzer = PLSQLAnalyzerOpenAI(
        top_10_dir=args.top_10_dir, output_dir=args.output_dir + "_openai"
    )

    # Implement the full analysis workflow (similar to copilot version)
    # This would need to be implemented in the PLSQLAnalyzerOpenAI class


def run_claude_analysis(args):
    """Run analysis using Claude API"""
    from api_alternatives import PLSQLAnalyzerClaude

    print("🚀 Starting analysis with Claude Sonnet")
    analyzer = PLSQLAnalyzerClaude(
        top_10_dir=args.top_10_dir, output_dir=args.output_dir + "_claude"
    )

    # Implement the full analysis workflow


def create_analysis_summary(output_dirs):
    """Create a summary comparing results from different APIs"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "apis_used": [],
        "total_procedures": 0,
        "procedures_by_api": {},
        "procedures_by_module": defaultdict(int),
    }

    all_procedures = []

    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            jsonl_file = Path(output_dir) / "procedures_analysis.jsonl"
            if jsonl_file.exists():
                api_name = output_dir.split("_")[-1]
                summary["apis_used"].append(api_name)

                procedures = []
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        procedure = json.loads(line)
                        procedures.append(procedure)
                        all_procedures.append(procedure)
                        summary["procedures_by_module"][procedure["module"]] += 1

                summary["procedures_by_api"][api_name] = len(procedures)
                summary["total_procedures"] += len(procedures)

    # Save combined results
    if all_procedures:
        combined_file = "plsql_analysis_combined.jsonl"
        with open(combined_file, "w", encoding="utf-8") as f:
            for procedure in all_procedures:
                json.dumps(procedure, ensure_ascii=False)
                f.write(json.dumps(procedure, ensure_ascii=False) + "\n")

        summary_file = "analysis_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Analysis Summary:")
        print(f"   • Total procedures extracted: {summary['total_procedures']}")
        print(f"   • APIs used: {', '.join(summary['apis_used'])}")
        print(f"   • Combined results: {combined_file}")
        print(f"   • Summary report: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive PLSQL Procedure Analysis"
    )
    parser.add_argument(
        "--api",
        choices=["copilot", "openai", "claude", "all", "available"],
        default="available",
        help="Which API to use",
    )
    parser.add_argument(
        "--top-10-dir",
        default="top_10",
        help="Directory containing top_10 module files",
    )
    parser.add_argument(
        "--output-dir",
        default="plsql_analysis",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--max-files", type=int, default=20, help="Maximum files to process per module"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Check and install missing dependencies",
    )
    parser.add_argument(
        "--list-apis", action="store_true", help="List available APIs and exit"
    )

    args = parser.parse_args()

    if args.install_deps:
        if not check_and_install_dependencies():
            sys.exit(1)

    # Get available APIs
    apis = get_available_apis()

    if args.list_apis:
        print("📋 Available APIs:")
        for api_name, info in apis.items():
            status = "✅" if info["available"] else "❌"
            print(f"   {status} {api_name}: {info['description']}")
        return

    print("🔍 Checking API availability...")
    for api_name, info in apis.items():
        status = "✅" if info["available"] else "❌"
        print(f"   {status} {api_name}: {info['description']}")

    # Determine which APIs to use
    if args.api == "available":
        # Use the first available API
        available_apis = [name for name, info in apis.items() if info["available"]]
        if not available_apis:
            print("❌ No APIs available. Please set up at least one API:")
            print("   • For OpenAI: export OPENAI_API_KEY='your-key'")
            print("   • For Claude: export ANTHROPIC_API_KEY='your-key'")
            print(
                "   • For Copilot: Run with --api copilot (will prompt for GitHub auth)"
            )
            sys.exit(1)

        api_to_use = available_apis[0]
        print(f"🎯 Using first available API: {api_to_use}")
        apis_to_run = [api_to_use]

    elif args.api == "all":
        apis_to_run = [name for name, info in apis.items() if info["available"]]
        if not apis_to_run:
            print("❌ No APIs available")
            sys.exit(1)

    else:
        if not apis[args.api]["available"]:
            print(f"❌ {args.api} API is not available")
            sys.exit(1)
        apis_to_run = [args.api]

    # Run analysis with selected APIs
    output_dirs = []

    for api_name in apis_to_run:
        try:
            if api_name == "copilot":
                run_copilot_analysis(args)
                output_dirs.append(args.output_dir + "_copilot")
            elif api_name == "openai":
                run_openai_analysis(args)
                output_dirs.append(args.output_dir + "_openai")
            elif api_name == "claude":
                run_claude_analysis(args)
                output_dirs.append(args.output_dir + "_claude")

        except Exception as e:
            print(f"❌ Error running {api_name} analysis: {e}")
            continue

    # Create combined summary
    if len(apis_to_run) > 1:
        create_analysis_summary(output_dirs)

    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()
