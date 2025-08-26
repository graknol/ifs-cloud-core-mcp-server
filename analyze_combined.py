#!/usr/bin/env python3
"""
Analyze the Combined Summary File
=================================

Quick analysis script to verify the quality and completeness of the combined summaries.
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_combined_summaries():
    """Analyze the combined summary file."""

    # Find the most recent combined file
    batch_dir = Path("batch_summaries")
    combined_files = list(batch_dir.glob("combined_summaries_*.json"))

    if not combined_files:
        print("❌ No combined summary files found!")
        return

    # Use the most recent file
    combined_file = max(combined_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Analyzing: {combined_file.name}")

    # Load the combined file
    with open(combined_file, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    print(f"\n✅ Successfully loaded {len(summaries)} summaries")

    # Analyze by module
    module_stats = defaultdict(list)
    complete_summaries = 0
    incomplete_summaries = 0

    for summary in summaries:
        module = summary.get("module", "unknown")
        module_stats[module].append(summary)

        # Check if summary is complete
        if summary.get("summary") and len(summary.get("summary", "")) > 100:
            complete_summaries += 1
        else:
            incomplete_summaries += 1

    print(f"\n📈 Module Breakdown:")
    for module, procedures in sorted(module_stats.items()):
        print(f"  {module:8s}: {len(procedures):3d} procedures")

    print(f"\n🔍 Summary Quality:")
    print(f"  Complete summaries: {complete_summaries}")
    print(f"  Incomplete summaries: {incomplete_summaries}")
    print(
        f"  Completion rate: {complete_summaries/(complete_summaries+incomplete_summaries)*100:.1f}%"
    )

    # Check for duplicates
    unique_keys = set()
    duplicates = []

    for summary in summaries:
        key = (
            summary.get("procedure_name", ""),
            summary.get("module", ""),
            summary.get("file_path", ""),
        )
        if key in unique_keys:
            duplicates.append(key)
        else:
            unique_keys.add(key)

    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} potential duplicates:")
        for dup in duplicates[:5]:  # Show first 5
            print(f"    {dup[1]}.{dup[0]}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")
    else:
        print(f"\n✅ No duplicates found!")

    # Sample a few summaries to check quality
    print(f"\n📝 Sample Summary Quality Check:")
    for i, summary in enumerate(summaries[:3]):
        summary_text = summary.get("summary", "")
        if summary_text:
            word_count = len(summary_text.split())
            print(
                f"  {i+1}. {summary['module']}.{summary['procedure_name']}: {word_count} words"
            )
        else:
            print(
                f"  {i+1}. {summary['module']}.{summary['procedure_name']}: No summary"
            )

    print(f"\n🎯 Analysis complete!")
    print(f"📁 File: {combined_file}")
    print(f"📊 Size: {combined_file.stat().st_size:,} bytes")


if __name__ == "__main__":
    analyze_combined_summaries()
