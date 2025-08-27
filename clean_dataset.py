#!/usr/bin/env python3
"""
Clean synthetic questions dataset by:
1. Filter out entries with null original_summary
2. Correlate with procedures_analysis.jsonl to get accurate procedure info
3. Use summaries as correlation keys (module + file + summary should be 1-to-1)
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_procedures_analysis(file_path: Path) -> dict:
    """Load procedures analysis and create lookup by summary."""
    print(f"📖 Loading procedures analysis from {file_path}")

    procedures_by_summary = {}
    procedures_by_name_file = {}
    total_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1

                # Create lookup by summary (primary key for correlation)
                summary_key = data.get("summary", "").strip()
                if summary_key:
                    procedures_by_summary[summary_key] = data

                # Create secondary lookup by procedure name + file
                name_file_key = (
                    f"{data.get('procedure_name', '')}|{data.get('file', '')}"
                )
                procedures_by_name_file[name_file_key] = data

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error on line {line_num}: {e}")
                continue

    print(f"✅ Loaded {total_count} procedure entries")
    print(f"   - {len(procedures_by_summary)} unique summaries")
    print(f"   - {len(procedures_by_name_file)} unique name+file combinations")

    return procedures_by_summary, procedures_by_name_file


def clean_and_correlate_synthetic_questions(
    synthetic_file: Path,
    procedures_by_summary: dict,
    procedures_by_name_file: dict,
    output_file: Path,
) -> None:
    """Clean synthetic questions and correlate with procedures analysis."""

    print(f"\n🧹 Cleaning synthetic questions from {synthetic_file}")

    stats = {
        "total_entries": 0,
        "null_summary_filtered": 0,
        "summary_matched": 0,
        "name_file_matched": 0,
        "no_match": 0,
        "clean_entries": 0,
    }

    clean_entries = []

    with open(synthetic_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                stats["total_entries"] += 1

                # Filter out entries with null original_summary
                original_summary = data.get("original_summary")
                if original_summary is None:
                    stats["null_summary_filtered"] += 1
                    continue

                # Try to correlate with procedures analysis
                procedure_data = None
                match_type = None

                # Primary correlation: by summary
                summary_key = original_summary.strip()
                if summary_key in procedures_by_summary:
                    procedure_data = procedures_by_summary[summary_key]
                    match_type = "summary"
                    stats["summary_matched"] += 1
                else:
                    # Secondary correlation: by procedure name + file
                    name_file_key = (
                        f"{data.get('procedure_name', '')}|{data.get('file', '')}"
                    )
                    if name_file_key in procedures_by_name_file:
                        procedure_data = procedures_by_name_file[name_file_key]
                        match_type = "name_file"
                        stats["name_file_matched"] += 1
                    else:
                        stats["no_match"] += 1
                        # Keep the original data but mark it as unmatched
                        procedure_data = data
                        match_type = "no_match"

                # Create clean entry with enhanced data
                clean_entry = {
                    "procedure_name": procedure_data.get(
                        "procedure_name", data.get("procedure_name")
                    ),
                    "parameters": procedure_data.get("parameters", ""),
                    "signature": procedure_data.get("signature", ""),
                    "summary": procedure_data.get("summary", original_summary),
                    "synthetic_queries": data.get("synthetic_queries", []),
                    "synthetic_statements": data.get("synthetic_statements", []),
                    "module": procedure_data.get("module", data.get("module")),
                    "file": procedure_data.get("file", data.get("file")),
                    "file_path": procedure_data.get(
                        "file_path", data.get("source_file")
                    ),
                    "match_type": match_type,
                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                }

                clean_entries.append(clean_entry)
                stats["clean_entries"] += 1

                if line_num % 1000 == 0:
                    print(f"   Processed {line_num:,} entries...")

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Error processing line {line_num}: {e}")
                continue

    # Write clean entries to output file
    print(f"\n💾 Writing clean entries to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in clean_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print statistics
    print(f"\n📊 Cleaning Statistics:")
    print(f"{'='*50}")
    print(f"Total entries processed:     {stats['total_entries']:,}")
    print(f"Null summary filtered out:   {stats['null_summary_filtered']:,}")
    print(f"Summary-based matches:       {stats['summary_matched']:,}")
    print(f"Name+file-based matches:     {stats['name_file_matched']:,}")
    print(f"No correlation match:        {stats['no_match']:,}")
    print(f"Clean entries written:       {stats['clean_entries']:,}")
    print(f"")
    print(
        f"Cleaning efficiency:         {(stats['clean_entries']/stats['total_entries']*100):.1f}%"
    )
    print(
        f"Correlation success rate:    {((stats['summary_matched']+stats['name_file_matched'])/stats['clean_entries']*100):.1f}%"
    )


def analyze_results(clean_file: Path) -> None:
    """Analyze the cleaning results."""
    print(f"\n📈 Analyzing cleaned dataset: {clean_file}")

    # Load and analyze clean data
    modules = {}
    match_types = {}
    total_entries = 0

    with open(clean_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            total_entries += 1

            # Count by module
            module = data.get("module", "unknown")
            modules[module] = modules.get(module, 0) + 1

            # Count by match type
            match_type = data.get("match_type", "unknown")
            match_types[match_type] = match_types.get(match_type, 0) + 1

    print(f"\n🎯 Analysis Results:")
    print(f"{'='*40}")
    print(f"Total clean entries: {total_entries:,}")
    print(f"Modules covered: {len(modules)}")

    # Show match type breakdown
    print(f"\n🔗 Correlation breakdown:")
    for match_type, count in sorted(
        match_types.items(), key=lambda x: x[1], reverse=True
    ):
        pct = count / total_entries * 100
        print(f"   {match_type:12s}: {count:5,} ({pct:5.1f}%)")

    # Show top modules
    print(f"\n🏆 Top 10 modules by entry count:")
    sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
    for i, (module, count) in enumerate(sorted_modules[:10], 1):
        pct = count / total_entries * 100
        print(f"   {i:2d}. {module:12s}: {count:4,} ({pct:5.1f}%)")


def main():
    """Main execution function."""
    base_dir = Path("plsql_analysis_combined")

    # Input files
    synthetic_file = base_dir / "synthetic_questions.jsonl"
    procedures_file = base_dir / "procedures_analysis.jsonl"

    # Output file
    clean_file = base_dir / "clean.jsonl"

    # Validate input files
    if not synthetic_file.exists():
        print(f"❌ Synthetic questions file not found: {synthetic_file}")
        sys.exit(1)

    if not procedures_file.exists():
        print(f"❌ Procedures analysis file not found: {procedures_file}")
        sys.exit(1)

    print("🧹 IFS Cloud Synthetic Questions Dataset Cleaning")
    print("=" * 60)

    # Load procedures analysis for correlation
    procedures_by_summary, procedures_by_name_file = load_procedures_analysis(
        procedures_file
    )

    # Clean and correlate synthetic questions
    clean_and_correlate_synthetic_questions(
        synthetic_file, procedures_by_summary, procedures_by_name_file, clean_file
    )

    # Analyze results
    analyze_results(clean_file)

    print(f"\n✅ Dataset cleaning completed!")
    print(f"   Clean dataset: {clean_file}")
    print(f"   Ready for embedding fine-tuning with accurate procedure correlation.")


if __name__ == "__main__":
    main()
