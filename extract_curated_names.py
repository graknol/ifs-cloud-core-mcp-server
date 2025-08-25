#!/usr/bin/env python3
"""
Re-scan IFS codebase using curated word list to extract matching names.
This includes parameters, procedures, and variable names that contain
the same characters in the same order as curated words.
"""

import os
import re
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


class CuratedNameExtractor:
    def __init__(self, root_directory, curated_csv_file):
        self.root_directory = Path(root_directory)
        self.curated_words = self.load_curated_words(curated_csv_file)
        self.found_names = defaultdict(list)
        self.name_counts = Counter()

        # Comprehensive patterns for all name types
        self.name_patterns = [
            # Procedure/function definitions
            re.compile(
                r"(?:PROCEDURE|FUNCTION)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
            ),
            # Parameter declarations in parentheses
            re.compile(
                r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:IN|OUT|IN\s+OUT)?", re.IGNORECASE
            ),
            # Variable declarations
            re.compile(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+(?:VARCHAR2|NUMBER|DATE|BOOLEAN|CLOB|BLOB)",
                re.IGNORECASE | re.MULTILINE,
            ),
            # Assignment statements
            re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:=", re.IGNORECASE),
            # FOR loop variables
            re.compile(r"FOR\s+([A-Za-z_][A-Za-z0-9_]*)\s+IN", re.IGNORECASE),
            # Cursor declarations
            re.compile(r"CURSOR\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
            # Table/view references in SELECT
            re.compile(r"FROM\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
            # API calls and method parameters
            re.compile(
                r"[A-Za-z_][A-Za-z0-9_]*\.\w+\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)",
                re.IGNORECASE,
            ),
            # Record field access: record_name.field_name
            re.compile(
                r"([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]*", re.IGNORECASE
            ),
        ]

    def load_curated_words(self, csv_file):
        """Load the curated word list from CSV."""
        curated_words = {}

        if not Path(csv_file).exists():
            print(f"❌ Curated CSV file not found: {csv_file}")
            return {}

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["word"].lower().strip()
                insightful = int(row.get("insightful", 0))
                notes = row.get("notes", "").strip()

                if insightful == 1:  # Only include words marked as insightful
                    curated_words[word] = {
                        "occurrences": int(row.get("occurrences", 0)),
                        "notes": notes,
                    }

        print(f"📋 Loaded {len(curated_words)} curated insightful words")
        return curated_words

    def contains_word_pattern(self, name, target_word):
        """Check if name contains the same characters as target_word in the same order."""
        name = name.lower()
        target = target_word.lower()

        # Remove underscores for comparison
        name_clean = name.replace("_", "")

        # Check if target characters appear in order within name
        target_idx = 0
        for char in name_clean:
            if target_idx < len(target) and char == target[target_idx]:
                target_idx += 1

        return target_idx == len(target)

    def extract_matching_names(self, content, file_path):
        """Extract names that match curated word patterns."""
        all_names = set()

        # Extract all names using patterns
        for pattern in self.name_patterns:
            for match in pattern.finditer(content):
                name = match.group(1)
                if name and len(name) >= 2:
                    all_names.add(name)

        # Check each name against curated words
        for name in all_names:
            for curated_word in self.curated_words:
                if self.contains_word_pattern(name, curated_word):
                    self.found_names[curated_word].append(
                        {
                            "name": name,
                            "file": str(file_path.relative_to(self.root_directory)),
                        }
                    )
                    self.name_counts[name] += 1

    def process_file(self, file_path):
        """Process a single file for matching names."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            self.extract_matching_names(content, file_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def scan_codebase(self):
        """Scan entire codebase for names matching curated words."""
        print(f"🔍 Scanning codebase: {self.root_directory}")

        plsql_files = list(self.root_directory.rglob("*.plsql"))
        total_files = len(plsql_files)

        print(f"📁 Found {total_files} .plsql files")

        processed = 0
        for file_path in plsql_files:
            self.process_file(file_path)
            processed += 1

            if processed % 1000 == 0:
                print(
                    f"   Processed {processed}/{total_files} files ({processed/total_files*100:.1f}%)"
                )

        print(f"✅ Completed processing {processed} files")

    def generate_report(self, output_file):
        """Generate comprehensive report of findings."""
        report = {
            "summary": {
                "curated_words": len(self.curated_words),
                "matching_names_found": len(self.name_counts),
                "total_occurrences": sum(self.name_counts.values()),
                "analysis_date": "2025-08-25",
            },
            "curated_word_matches": {},
        }

        # Build detailed matches for each curated word
        for word in self.curated_words:
            matches = self.found_names[word]
            if matches:
                # Count unique names and their frequencies
                name_freq = Counter(match["name"] for match in matches)

                report["curated_word_matches"][word] = {
                    "original_frequency": self.curated_words[word]["occurrences"],
                    "notes": self.curated_words[word]["notes"],
                    "matching_names": len(name_freq),
                    "total_name_occurrences": sum(name_freq.values()),
                    "top_names": sorted(
                        name_freq.items(), key=lambda x: x[1], reverse=True
                    )[:10],
                    "example_files": list(set(match["file"] for match in matches[:5])),
                }

        # Save detailed report
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def show_results_preview(self, top_n=20):
        """Show preview of results."""
        print(f"\n📊 RESULTS PREVIEW (Top {top_n} Curated Words):")
        print("-" * 70)

        # Sort curated words by number of matches found
        word_results = []
        for word in self.curated_words:
            matches = self.found_names[word]
            if matches:
                unique_names = set(match["name"] for match in matches)
                word_results.append((word, len(unique_names), len(matches)))

        word_results.sort(key=lambda x: x[2], reverse=True)

        for i, (word, unique_names, total_matches) in enumerate(
            word_results[:top_n], 1
        ):
            print(
                f"{i:2d}. {word:<15} → {unique_names:3d} unique names, {total_matches:4d} total matches"
            )

            # Show a few example names
            examples = list(set(match["name"] for match in self.found_names[word][:5]))
            print(f"    Examples: {', '.join(examples[:3])}")
            if len(examples) > 3:
                print(f"              {', '.join(examples[3:])}")
            print()


def main():
    default_path = r"C:\repos\_work\25.1.0"
    csv_file = "ifs_parameter_procedure_words.csv"

    print("🎯 IFS CLOUD CURATED NAME EXTRACTOR")
    print("=" * 50)

    # Check if curated CSV exists
    if not Path(csv_file).exists():
        print(f"❌ Please run extract_parameter_procedure_names.py first")
        print(f"   and curate the results in {csv_file}")
        return

    root_dir = input(f"Enter IFS codebase path (default: {default_path}): ").strip()
    if not root_dir:
        root_dir = default_path

    if not Path(root_dir).exists():
        print(f"❌ Directory not found: {root_dir}")
        return

    extractor = CuratedNameExtractor(root_dir, csv_file)

    if not extractor.curated_words:
        print(f"❌ No insightful words found in {csv_file}")
        print(f"   Please mark words as insightful (1) in the CSV file")
        return

    print(f"\n🔍 SCANNING FOR MATCHING NAMES...")
    extractor.scan_codebase()

    print(f"\n📊 GENERATING RESULTS...")
    report_file = "curated_name_extraction_results.json"
    report = extractor.generate_report(report_file)

    extractor.show_results_preview(20)

    print(f"\n✅ EXTRACTION COMPLETE!")
    print(f"📄 Detailed report: {report_file}")
    print(f"🎯 Found {report['summary']['matching_names_found']} matching names")
    print(f"📊 Total occurrences: {report['summary']['total_occurrences']:,}")


if __name__ == "__main__":
    main()
