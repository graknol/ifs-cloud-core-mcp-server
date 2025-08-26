#!/usr/bin/env python3
"""
Combine Multiple Summary Files into One Comprehensive JSON
=========================================================

This script merges the three diversified summary files into one consolidated JSON file,
ensuring no duplicates and maintaining proper ID sequencing.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"Warning: {file_path} does not contain a list, skipping...")
                return []
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found, skipping...")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def clean_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and validate a summary entry."""
    # Ensure required fields exist
    required_fields = ["procedure_name", "module", "file_path", "timestamp"]

    # Skip entries that only have timestamp (incomplete entries)
    if len(entry) <= 2 and "timestamp" in entry:
        return None

    # Skip entries missing critical information
    if not entry.get("procedure_name") or not entry.get("module"):
        return None

    # Clean the entry
    cleaned = {}
    for key, value in entry.items():
        if value is not None and value != "":
            cleaned[key] = value

    return cleaned if len(cleaned) > 2 else None


def combine_summaries():
    """Combine all diversified summary files into one."""

    # Define the batch summaries directory
    batch_dir = Path("batch_summaries")
    if not batch_dir.exists():
        print(f"Error: Directory {batch_dir} does not exist!")
        sys.exit(1)

    # Find all diversified summary files
    summary_files = list(batch_dir.glob("diversified_summaries_*.json"))
    summary_files.sort()  # Sort by filename (which includes timestamp)

    if not summary_files:
        print("Error: No diversified summary files found!")
        sys.exit(1)

    print(f"🔍 Found {len(summary_files)} summary files:")
    for f in summary_files:
        print(f"  - {f.name}")

    # Load all summaries
    all_summaries = []
    file_stats = {}

    for file_path in summary_files:
        print(f"\n📂 Loading {file_path.name}...")
        summaries = load_json_file(file_path)

        # Clean and validate entries
        valid_summaries = []
        for entry in summaries:
            cleaned = clean_entry(entry)
            if cleaned:
                valid_summaries.append(cleaned)

        file_stats[file_path.name] = {
            "total_entries": len(summaries),
            "valid_entries": len(valid_summaries),
            "file_size": file_path.stat().st_size,
        }

        all_summaries.extend(valid_summaries)
        print(
            f"  ✅ Loaded {len(valid_summaries)} valid entries (out of {len(summaries)} total)"
        )

    print(f"\n📊 Summary Statistics:")
    total_loaded = 0
    for filename, stats in file_stats.items():
        print(
            f"  {filename}: {stats['valid_entries']} valid / {stats['total_entries']} total ({stats['file_size']:,} bytes)"
        )
        total_loaded += stats["valid_entries"]

    # Remove duplicates based on procedure_name, module, and file_path
    unique_summaries = []
    seen = set()

    for summary in all_summaries:
        # Create a unique key for deduplication
        key = (
            summary.get("procedure_name", ""),
            summary.get("module", ""),
            summary.get("file_path", ""),
        )

        if key not in seen:
            seen.add(key)
            unique_summaries.append(summary)

    print(f"\n🔄 Deduplication:")
    print(f"  Total entries loaded: {total_loaded}")
    print(f"  Unique entries: {len(unique_summaries)}")
    print(f"  Duplicates removed: {total_loaded - len(unique_summaries)}")

    # Sort by module, then by procedure_name for better organization
    unique_summaries.sort(
        key=lambda x: (x.get("module", ""), x.get("procedure_name", ""))
    )

    # Reassign sequential IDs
    for i, summary in enumerate(unique_summaries, 1):
        summary["id"] = i

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = batch_dir / f"combined_summaries_{timestamp}.json"

    # Save the combined summaries
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(unique_summaries, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Combined summaries saved to: {output_file}")
        print(f"📈 Final statistics:")
        print(f"  Total unique summaries: {len(unique_summaries)}")

        # Module breakdown
        module_counts = {}
        for summary in unique_summaries:
            module = summary.get("module", "unknown")
            module_counts[module] = module_counts.get(module, 0) + 1

        print(f"  Module breakdown:")
        for module, count in sorted(module_counts.items()):
            print(f"    {module}: {count} procedures")

        print(f"  Output file size: {output_file.stat().st_size:,} bytes")

        return output_file

    except Exception as e:
        print(f"Error saving combined summaries: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🎯 IFS Cloud Summary Combiner")
    print("=" * 50)

    try:
        output_file = combine_summaries()
        print(f"\n🎉 Successfully created combined summary file: {output_file.name}")

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
