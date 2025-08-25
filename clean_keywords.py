#!/usr/bin/env python3
"""
Clean up final_optimizer_keywords.csv to remove duplicate keywords.

This script identifies keywords that appear both as standalone keywords and as variants
of other keywords, then removes the standalone entries to avoid duplication.
"""

import pandas as pd
import re
from typing import Set, Dict, List


def extract_variants_from_string(variants_str: str) -> Set[str]:
    """Extract variant keyword names from the variants_with_scores string."""
    if pd.isna(variants_str) or not variants_str:
        return set()

    # Extract words before the (score) pattern
    pattern = r"(\w+)\([0-9.]+\)"
    matches = re.findall(pattern, variants_str)
    return set(matches)


def find_duplicate_keywords(df: pd.DataFrame) -> Dict[str, str]:
    """Find keywords that appear both as main keywords and as variants of others."""
    # Get all main keywords
    main_keywords = set(df["keyword"].str.lower())

    # Get all variant keywords
    all_variants = set()
    variant_to_main = {}  # Maps variant -> main keyword that contains it

    for _, row in df.iterrows():
        main_keyword = row["keyword"].lower()
        variants = extract_variants_from_string(row["variants_with_scores"])

        for variant in variants:
            variant_lower = variant.lower()
            all_variants.add(variant_lower)
            variant_to_main[variant_lower] = main_keyword

    # Find keywords that are both main keywords and variants
    duplicates = {}
    for keyword in main_keywords:
        if keyword in all_variants:
            duplicates[keyword] = variant_to_main[keyword]

    return duplicates


def analyze_keywords_csv(file_path: str):
    """Analyze the keywords CSV for duplicates and show statistics."""
    print(f"📊 Analyzing keywords CSV: {file_path}")

    # Read the CSV
    df = pd.read_csv(file_path)
    print(f"📈 Total keywords: {len(df)}")

    # Find duplicates
    duplicates = find_duplicate_keywords(df)
    print(f"🔍 Found {len(duplicates)} duplicate keywords:")

    for duplicate, parent in duplicates.items():
        duplicate_row = df[df["keyword"].str.lower() == duplicate].iloc[0]
        parent_row = df[df["keyword"].str.lower() == parent].iloc[0]

        print(
            f"  ❌ '{duplicate}' (occurrences: {duplicate_row['primary_occurrences']}) "
            f"-> variant of '{parent}' (occurrences: {parent_row['primary_occurrences']})"
        )

    return df, duplicates


def clean_keywords_csv(input_file: str, output_file: str):
    """Clean the keywords CSV by removing duplicates."""
    df, duplicates = analyze_keywords_csv(input_file)

    if not duplicates:
        print("✅ No duplicates found!")
        return

    print(f"\n🧹 Cleaning duplicates...")

    # Remove duplicate keywords (keep the parent keyword entries)
    duplicate_keywords = list(duplicates.keys())
    initial_count = len(df)

    # Remove rows where the keyword is a duplicate
    df_cleaned = df[~df["keyword"].str.lower().isin(duplicate_keywords)]

    final_count = len(df_cleaned)
    removed_count = initial_count - final_count

    print(f"📉 Removed {removed_count} duplicate keywords")
    print(f"📈 Final keyword count: {final_count}")

    # Save cleaned version
    df_cleaned.to_csv(output_file, index=False)
    print(f"💾 Saved cleaned keywords to: {output_file}")

    # Show some statistics
    total_occurrences_before = df["primary_occurrences"].sum()
    total_occurrences_after = df_cleaned["primary_occurrences"].sum()

    print(f"📊 Total occurrences before: {total_occurrences_before:,}")
    print(f"📊 Total occurrences after: {total_occurrences_after:,}")
    print(
        f"📊 Occurrences preserved: {(total_occurrences_after/total_occurrences_before)*100:.1f}%"
    )


if __name__ == "__main__":
    input_file = "final_optimizer_keywords.csv"
    output_file = "final_optimizer_keywords_cleaned.csv"

    clean_keywords_csv(input_file, output_file)

    print(f"\n✨ Keyword cleanup complete!")
    print(f"📄 Original file: {input_file}")
    print(f"📄 Cleaned file: {output_file}")
