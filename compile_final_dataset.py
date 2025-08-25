#!/usr/bin/env python3
"""
Compile final dataset of words/abbreviations for the optimizer.
Apply strict filtering rules to ensure high quality matches.
"""

import csv
import re
from collections import defaultdict, Counter
from pathlib import Path


class FinalDatasetCompiler:
    def __init__(self):
        self.word_groups = defaultdict(list)
        self.final_keywords = {}

        # Common plural suffixes to identify plurals
        self.plural_patterns = [
            r".*s$",  # ends with 's'
            r".*ies$",  # ends with 'ies' (like companies)
            r".*es$",  # ends with 'es' (like boxes)
        ]

        # Common abbreviation indicators (to avoid as primary keywords)
        self.abbreviation_indicators = [
            lambda w: len(w) <= 2,  # Very short words
            lambda w: w.count("") == len(w) and len(w) < 4,  # All consonants and short
            lambda w: not any(vowel in w for vowel in "aeiou")
            and len(w) < 5,  # No vowels and short
        ]

    def load_curated_data(self, csv_file):
        """Load curated data from the CSV with suggestions."""
        insightful_words = []

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["word"].strip().lower()
                insightful = int(row.get("insightful", 0))
                occurrences = int(row.get("occurrences", 0))
                notes = row.get("notes", "").strip()

                # Apply length filter
                if len(word) < 3:
                    continue

                if insightful == 1:
                    insightful_words.append(
                        {"word": word, "occurrences": occurrences, "notes": notes}
                    )

        print(f"📊 Loaded {len(insightful_words)} insightful words (3+ chars)")
        return insightful_words

    def load_clustering_data(self, clustering_csv):
        """Load high-quality matches from clustering report."""
        high_quality_matches = []

        with open(clustering_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                unchecked_word = row["unchecked_word"].strip().lower()
                similarity_score = float(row["similarity_score"])
                action = row["suggested_action"].strip()
                occurrences = int(row["occurrences"])
                best_match = row["best_match"].strip().lower()

                # Apply length filter
                if len(unchecked_word) < 3:
                    continue

                # Include matches with score > 0.58 as requested
                if similarity_score > 0.58:
                    high_quality_matches.append(
                        {
                            "word": unchecked_word,
                            "occurrences": occurrences,
                            "related_to": best_match,
                            "similarity_score": similarity_score,
                            "match_type": (
                                action.split(" -")[0] if " -" in action else action
                            ),
                        }
                    )

        print(
            f"🎯 Found {len(high_quality_matches)} high-quality clustered matches (score > 0.58)"
        )
        return high_quality_matches

    def is_plural(self, word):
        """Check if a word is likely a plural form."""
        for pattern in self.plural_patterns:
            if re.match(pattern, word):
                # Additional checks to avoid false positives
                if word.endswith("s") and len(word) > 3:
                    # Check if removing 's' gives a valid word
                    singular = word[:-1]
                    if len(singular) >= 3:
                        return True
                elif word.endswith("ies") and len(word) > 4:
                    return True
                elif word.endswith("es") and len(word) > 4:
                    return True
        return False

    def is_likely_abbreviation(self, word):
        """Check if word is likely a short abbreviation."""
        return any(indicator(word) for indicator in self.abbreviation_indicators)

    def find_stem(self, words):
        """Find the best stem word from a group of related words."""
        if not words:
            return None

        # Sort by preference criteria
        candidates = []

        for word_data in words:
            word = word_data["word"]
            score = 0

            # Prefer non-plurals
            if not self.is_plural(word):
                score += 100

            # Prefer longer words (avoid abbreviations)
            if not self.is_likely_abbreviation(word):
                score += 50

            # Prefer words with vowels (more readable)
            if any(vowel in word for vowel in "aeiou"):
                score += 25

            # Prefer higher frequency
            score += min(word_data["occurrences"] / 1000, 10)  # Cap at 10 points

            # Prefer shorter of two similar words (if one is substring of other)
            substring_bonus = 0
            for other_data in words:
                other_word = other_data["word"]
                if word != other_word:
                    if word in other_word:
                        substring_bonus += 20  # Shorter word that's contained in longer
                    elif other_word in word:
                        substring_bonus -= 10  # Longer word that contains shorter

            score += substring_bonus

            candidates.append((word, word_data, score))

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates[0][1] if candidates else None

    def group_related_words(self, insightful_words, clustered_matches):
        """Group related words together."""

        # Start with insightful words as base groups
        for word_data in insightful_words:
            word = word_data["word"]
            self.word_groups[word].append({**word_data, "source": "curated"})

        # Add clustered matches to related groups
        for match_data in clustered_matches:
            word = match_data["word"]
            related_to = match_data["related_to"]

            # Find the best group to add this word to
            best_group = None
            best_similarity = 0

            # Check if it's related to any existing group
            for group_key in self.word_groups:
                if group_key == related_to:
                    best_group = group_key
                    break
                # Check for similar words in group
                for group_word_data in self.word_groups[group_key]:
                    if (
                        self.calculate_word_similarity(word, group_word_data["word"])
                        > best_similarity
                    ):
                        best_similarity = self.calculate_word_similarity(
                            word, group_word_data["word"]
                        )
                        if best_similarity > 0.7:  # High similarity threshold
                            best_group = group_key

            if best_group:
                self.word_groups[best_group].append(
                    {**match_data, "source": "clustered"}
                )
            else:
                # Create new group
                self.word_groups[word].append({**match_data, "source": "clustered"})

    def calculate_word_similarity(self, word1, word2):
        """Simple similarity calculation."""
        if word1 == word2:
            return 1.0
        if word1 in word2 or word2 in word1:
            return 0.8
        # Simple character overlap
        chars1 = set(word1)
        chars2 = set(word2)
        overlap = len(chars1 & chars2)
        total = len(chars1 | chars2)
        return overlap / total if total > 0 else 0

    def compile_final_dataset(self):
        """Compile the final dataset with one keyword per group."""

        for group_key, word_list in self.word_groups.items():
            if not word_list:
                continue

            # Find the best representative word for this group
            best_word_data = self.find_stem(word_list)

            if best_word_data:
                # Calculate total occurrences and variants with details
                total_occurrences = sum(w["occurrences"] for w in word_list)
                variants_with_scores = []

                for w in word_list:
                    if w["word"] != best_word_data["word"]:
                        if "similarity_score" in w:
                            variants_with_scores.append(
                                f"{w['word']}({w['similarity_score']:.2f})"
                            )
                        else:
                            variants_with_scores.append(w["word"])

                self.final_keywords[best_word_data["word"]] = {
                    "keyword": best_word_data["word"],
                    "total_occurrences": total_occurrences,
                    "primary_occurrences": best_word_data["occurrences"],
                    "variants": [
                        w["word"]
                        for w in word_list
                        if w["word"] != best_word_data["word"]
                    ],
                    "variants_with_scores": variants_with_scores,
                    "variant_count": len(
                        [w for w in word_list if w["word"] != best_word_data["word"]]
                    ),
                    "sources": list(set(w["source"] for w in word_list)),
                    "notes": best_word_data.get("notes", ""),
                }

    def export_final_dataset(self, output_file):
        """Export the final dataset to CSV."""

        # Sort by total occurrences (most frequent first)
        sorted_keywords = sorted(
            self.final_keywords.items(),
            key=lambda x: x[1]["total_occurrences"],
            reverse=True,
        )

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "keyword",
                    "total_occurrences",
                    "primary_occurrences",
                    "variant_count",
                    "variants_with_scores",
                    "sources",
                    "notes",
                ]
            )

            for keyword, data in sorted_keywords:
                writer.writerow(
                    [
                        keyword,
                        data["total_occurrences"],
                        data["primary_occurrences"],
                        data["variant_count"],
                        ", ".join(
                            data["variants_with_scores"][:10]
                        ),  # Show variants with scores
                        ", ".join(data["sources"]),
                        data["notes"],
                    ]
                )

        print(f"💾 Final dataset exported to: {output_file}")

    def show_dataset_preview(self, top_n=30):
        """Show preview of final dataset."""

        print(f"\n📊 FINAL OPTIMIZER DATASET PREVIEW (Top {top_n}):")
        print("-" * 80)

        sorted_keywords = sorted(
            self.final_keywords.items(),
            key=lambda x: x[1]["total_occurrences"],
            reverse=True,
        )

        for i, (keyword, data) in enumerate(sorted_keywords[:top_n], 1):
            variants_str = (
                f" + {data['variant_count']} variants"
                if data["variant_count"] > 0
                else ""
            )
            sources_str = ",".join(data["sources"])

            print(
                f"{i:2d}. {keyword:<15} ({data['total_occurrences']:,} occur.){variants_str} [{sources_str}]"
            )

            if data["variants_with_scores"]:
                preview_variants = data["variants_with_scores"][:5]
                if len(data["variants_with_scores"]) > 5:
                    preview_variants.append(
                        f"...+{len(data['variants_with_scores'])-5} more"
                    )
                print(f"    Variants: {', '.join(preview_variants)}")

    def generate_summary_stats(self):
        """Generate summary statistics."""

        total_keywords = len(self.final_keywords)
        total_occurrences = sum(
            data["total_occurrences"] for data in self.final_keywords.values()
        )

        curated_count = sum(
            1 for data in self.final_keywords.values() if "curated" in data["sources"]
        )
        clustered_count = sum(
            1 for data in self.final_keywords.values() if "clustered" in data["sources"]
        )
        both_count = sum(
            1 for data in self.final_keywords.values() if len(data["sources"]) > 1
        )

        total_variants = sum(
            data["variant_count"] for data in self.final_keywords.values()
        )

        print(f"\n📈 FINAL DATASET STATISTICS:")
        print(f"   Total keywords: {total_keywords:,}")
        print(f"   Total occurrences: {total_occurrences:,}")
        print(f"   Total variants consolidated: {total_variants:,}")
        print(f"   Curated keywords: {curated_count}")
        print(f"   Clustered keywords: {clustered_count}")
        print(f"   Keywords from both sources: {both_count}")

        return {
            "total_keywords": total_keywords,
            "total_occurrences": total_occurrences,
            "total_variants": total_variants,
            "curated_count": curated_count,
            "clustered_count": clustered_count,
            "both_count": both_count,
        }


def main():
    curated_csv = "ifs_parameter_procedure_words_with_suggestions.csv"
    clustering_csv = "word_clustering_report.csv"

    print("🎯 FINAL OPTIMIZER DATASET COMPILER")
    print("=" * 60)

    # Check required files
    if not Path(curated_csv).exists():
        print(f"❌ Curated CSV file not found: {curated_csv}")
        return

    if not Path(clustering_csv).exists():
        print(f"❌ Clustering CSV file not found: {clustering_csv}")
        return

    compiler = FinalDatasetCompiler()

    print("📂 Loading curated data...")
    insightful_words = compiler.load_curated_data(curated_csv)

    print("🔍 Loading clustering data...")
    clustered_matches = compiler.load_clustering_data(clustering_csv)

    print("🔗 Grouping related words...")
    compiler.group_related_words(insightful_words, clustered_matches)

    print("🎯 Compiling final dataset...")
    compiler.compile_final_dataset()

    # Show results
    compiler.show_dataset_preview(30)

    # Generate statistics
    stats = compiler.generate_summary_stats()

    # Export final dataset
    output_file = "final_optimizer_keywords.csv"
    compiler.export_final_dataset(output_file)

    print(f"\n✅ FINAL DATASET COMPILATION COMPLETE!")
    print(f"📄 Dataset saved to: {output_file}")
    print(f"🎯 Ready for RTX optimizer integration!")


if __name__ == "__main__":
    main()
