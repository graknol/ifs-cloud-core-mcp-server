#!/usr/bin/env python3
"""
Extract parameter and procedure names from IFS Cloud codebase.
Split by underscore, count occurrences, export to CSV for manual curation.
"""

import os
import re
import csv
from collections import Counter
from pathlib import Path


class ParameterProcedureExtractor:
    def __init__(self, root_directory):
        self.root_directory = Path(root_directory)
        self.word_counts = Counter()

        # Patterns for extracting names
        self.procedure_pattern = re.compile(
            r"(?:PROCEDURE|FUNCTION)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
        )

        self.parameter_patterns = [
            # Procedure/function parameters: (param1_ IN VARCHAR2, param2_ OUT NUMBER)
            re.compile(
                r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:IN|OUT|IN\s+OUT)?", re.IGNORECASE
            ),
            # Variable declarations: param_name_ VARCHAR2;
            re.compile(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+(?:VARCHAR2|NUMBER|DATE|BOOLEAN)",
                re.IGNORECASE | re.MULTILINE,
            ),
            # Assignment patterns: param_name_ := value;
            re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:=", re.IGNORECASE),
            # API calls with parameters: API.Method(param1_, param2_)
            re.compile(
                r"[A-Za-z_][A-Za-z0-9_]*\.\w+\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)",
                re.IGNORECASE,
            ),
        ]

    def extract_words_from_name(self, name):
        """Split name by underscore and extract individual words."""
        if not name or len(name) < 2:
            return []

        # Remove trailing underscore (common in PL/SQL)
        if name.endswith("_"):
            name = name[:-1]

        # Split by underscore and filter out empty/short parts
        words = [word.lower() for word in name.split("_") if word and len(word) >= 2]
        return words

    def process_file(self, file_path):
        """Extract parameter and procedure names from a single file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract procedure/function names
            for match in self.procedure_pattern.finditer(content):
                proc_name = match.group(1)
                words = self.extract_words_from_name(proc_name)
                for word in words:
                    self.word_counts[word] += 1

            # Extract parameter names using all patterns
            for pattern in self.parameter_patterns:
                for match in pattern.finditer(content):
                    param_name = match.group(1)
                    words = self.extract_words_from_name(param_name)
                    for word in words:
                        self.word_counts[word] += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def scan_directory(self):
        """Scan all .plsql files in the directory."""
        print(f"🔍 Scanning directory: {self.root_directory}")

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
        print(f"📊 Found {len(self.word_counts)} unique words")

    def export_to_csv(self, output_file):
        """Export word counts to CSV for manual curation."""

        # Sort by occurrence count (most frequent first)
        sorted_words = sorted(
            self.word_counts.items(), key=lambda x: x[1], reverse=True
        )

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(["word", "occurrences", "insightful", "notes"])

            # Data rows
            for word, count in sorted_words:
                writer.writerow([word, count, 0, ""])  # Default insightful = 0

        print(f"💾 Exported to: {output_file}")
        print(f"📝 Instructions:")
        print(f"   1. Open {output_file} in Excel/LibreOffice")
        print(f"   2. Review each word in the 'word' column")
        print(f"   3. Set 'insightful' to 1 for meaningful words, 0 for noise")
        print(f"   4. Add notes in the 'notes' column if needed")
        print(f"   5. Save the file when done")

    def show_preview(self, top_n=50):
        """Show preview of top words."""
        print(f"\n📋 TOP {top_n} WORDS PREVIEW:")
        print("-" * 50)

        sorted_words = sorted(
            self.word_counts.items(), key=lambda x: x[1], reverse=True
        )

        for i, (word, count) in enumerate(sorted_words[:top_n], 1):
            print(f"{i:3d}. {word:<20} ({count:,} occurrences)")

        if len(sorted_words) > top_n:
            print(f"... and {len(sorted_words) - top_n} more words")


def main():
    # Default to the IFS work directory structure
    default_path = r"C:\repos\_ifs\25.1.0"

    print("🏗️  IFS CLOUD PARAMETER & PROCEDURE NAME EXTRACTOR")
    print("=" * 60)

    # Use default path for automation
    root_dir = default_path
    print(f"📁 Using IFS codebase path: {root_dir}")

    if not Path(root_dir).exists():
        print(f"❌ Directory not found: {root_dir}")
        print(f"   Please ensure the IFS Cloud codebase is available at this location")
        return

    extractor = ParameterProcedureExtractor(root_dir)

    print("\n🔍 SCANNING CODEBASE...")
    extractor.scan_directory()

    print("\n📊 ANALYSIS RESULTS:")
    extractor.show_preview(50)

    # Export to CSV
    output_file = "ifs_parameter_procedure_words.csv"
    extractor.export_to_csv(output_file)

    print(f"\n✅ EXTRACTION COMPLETE!")
    print(f"📁 Results saved to: {output_file}")
    print(f"🎯 Total unique words found: {len(extractor.word_counts):,}")


if __name__ == "__main__":
    main()
