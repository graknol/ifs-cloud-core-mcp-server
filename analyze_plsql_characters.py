#!/usr/bin/env python3
"""
Analyze PL/SQL files to identify common special characters and noise patterns
that should be added to BM25S stopwords.
"""

import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Set, Dict, List, Tuple
import json


def get_plsql_files(work_dir: Path) -> List[Path]:
    """Find all PL/SQL files in the work directory."""
    plsql_files = []
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith((".plsql", ".entity", ".views", ".fragment", ".storage")):
                plsql_files.append(Path(root) / file)
    return plsql_files


def analyze_special_characters(
    files: List[Path], sample_size: int = 50
) -> Dict[str, int]:
    """Analyze special characters in PL/SQL files."""
    char_counter = Counter()
    files_processed = 0

    for file_path in files[:sample_size]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

                # Count each character
                for char in content:
                    if not char.isalnum() and not char.isspace():
                        char_counter[char] += 1

            files_processed += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Processed {files_processed} files")
    return dict(char_counter)


def analyze_token_patterns(files: List[Path], sample_size: int = 50) -> Dict[str, int]:
    """Analyze common noise tokens and patterns."""
    token_counter = Counter()
    files_processed = 0

    for file_path in files[:sample_size]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lower()

                # Extract tokens using basic tokenization
                # Split on whitespace and common punctuation
                tokens = re.findall(r"\b\w+\b", content)

                # Count short tokens (likely noise)
                for token in tokens:
                    if len(token) <= 3:  # Short tokens are often noise
                        token_counter[token] += 1

                # Find repetitive patterns
                patterns = re.findall(r"[-=]{2,}|[*]{2,}|[/]{2,}|[_]{2,}", content)
                for pattern in patterns:
                    token_counter[pattern] += 1

            files_processed += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Processed {files_processed} files for token analysis")
    return dict(token_counter)


def analyze_comment_patterns(
    files: List[Path], sample_size: int = 50
) -> Dict[str, int]:
    """Analyze common comment patterns and decorative elements."""
    comment_patterns = Counter()
    files_processed = 0

    for file_path in files[:sample_size]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

                for line in lines:
                    line = line.strip()

                    # SQL comments
                    if line.startswith("--"):
                        comment_content = line[2:].strip()

                        # Check for decorative patterns
                        if re.match(r"^[-=*_]{2,}$", comment_content):
                            comment_patterns[comment_content] += 1

                        # Check for pure whitespace or single chars
                        if len(comment_content) <= 2:
                            comment_patterns[comment_content] += 1

                    # C-style comments
                    c_comments = re.findall(r"/\*.*?\*/", line)
                    for comment in c_comments:
                        clean_comment = comment[2:-2].strip()
                        if len(clean_comment) <= 3:
                            comment_patterns[clean_comment] += 1

            files_processed += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Processed {files_processed} files for comment analysis")
    return dict(comment_patterns)


def main():
    work_dir = Path("_work")
    if not work_dir.exists():
        print("❌ _work directory not found")
        return

    print("🔍 Finding PL/SQL files...")
    plsql_files = get_plsql_files(work_dir)
    print(f"Found {len(plsql_files)} PL/SQL files")

    if not plsql_files:
        print("❌ No PL/SQL files found")
        return

    # Sample size for analysis (to avoid processing too many files)
    sample_size = min(100, len(plsql_files))
    print(f"Analyzing sample of {sample_size} files...")

    print("\n📊 Analyzing special characters...")
    special_chars = analyze_special_characters(plsql_files, sample_size)

    print("\n📊 Analyzing token patterns...")
    token_patterns = analyze_token_patterns(plsql_files, sample_size)

    print("\n📊 Analyzing comment patterns...")
    comment_patterns = analyze_comment_patterns(plsql_files, sample_size)

    # Results
    print("\n" + "=" * 60)
    print("🔤 MOST COMMON SPECIAL CHARACTERS:")
    print("=" * 60)
    for char, count in Counter(special_chars).most_common(30):
        print(f"'{char}' -> {count:,} occurrences")

    print("\n" + "=" * 60)
    print("🔡 MOST COMMON SHORT TOKENS (potential noise):")
    print("=" * 60)
    for token, count in Counter(token_patterns).most_common(30):
        if len(token) <= 3:
            print(f"'{token}' -> {count:,} occurrences")

    print("\n" + "=" * 60)
    print("📝 MOST COMMON COMMENT PATTERNS:")
    print("=" * 60)
    for pattern, count in Counter(comment_patterns).most_common(20):
        print(f"'{pattern}' -> {count:,} occurrences")

    # Generate recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDED ADDITIONAL STOPWORDS:")
    print("=" * 60)

    # Special characters with high frequency
    frequent_chars = {char for char, count in special_chars.items() if count > 100}
    print("\nSpecial Characters:")
    for char in sorted(frequent_chars):
        print(f"    '{char}'")

    # High-frequency short tokens
    frequent_tokens = {
        token
        for token, count in token_patterns.items()
        if count > 50 and len(token) <= 3
    }
    print(f"\nShort Tokens (≤3 chars):")
    for token in sorted(frequent_tokens):
        if token.isalpha():  # Only alphabetic tokens
            print(f"    '{token}'")

    # Save detailed results
    results = {
        "special_characters": special_chars,
        "token_patterns": token_patterns,
        "comment_patterns": comment_patterns,
        "recommendations": {
            "special_chars": list(frequent_chars),
            "short_tokens": list(frequent_tokens),
        },
    }

    with open("plsql_character_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Detailed results saved to plsql_character_analysis.json")


if __name__ == "__main__":
    main()
