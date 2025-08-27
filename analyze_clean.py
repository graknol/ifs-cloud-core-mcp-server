#!/usr/bin/env python3
"""
Quick analysis of the clean dataset
"""

import json


def analyze_clean_dataset():
    """Analyze the clean dataset quality."""
    print("🎉 Clean Dataset Analysis")
    print("=" * 40)

    # Load dataset
    with open("plsql_analysis_combined/clean.jsonl", "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    print(f"Total entries: {len(entries):,}")

    # Check data quality
    complete_entries = 0
    for entry in entries:
        if (
            entry.get("procedure_name")
            and entry.get("parameters")
            and entry.get("summary")
            and entry.get("synthetic_queries")
            and entry.get("synthetic_statements")
            and len(entry.get("synthetic_queries", [])) >= 2
            and len(entry.get("synthetic_statements", [])) >= 2
        ):
            complete_entries += 1

    print(
        f"Complete entries with all fields: {complete_entries:,} ({complete_entries/len(entries)*100:.1f}%)"
    )

    # Sample entries
    print("\n🔍 Sample entries:")
    for i, entry in enumerate(entries[:2], 1):
        print(f"\nEntry {i}:")
        print(f"  Procedure: {entry.get('procedure_name', 'N/A')}")
        print(f"  Module: {entry.get('module', 'N/A')}")
        print(f"  Summary: {entry.get('summary', 'N/A')[:80]}...")
        queries = entry.get("synthetic_queries", [])
        statements = entry.get("synthetic_statements", [])
        print(f"  Queries ({len(queries)}): {queries}")
        print(f"  Statements ({len(statements)}): {statements}")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    analyze_clean_dataset()
