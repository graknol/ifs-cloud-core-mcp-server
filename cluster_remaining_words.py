#!/usr/bin/env python3
"""
Cluster remaining CSV rows to find similar words to the curated insightful ones.
This will help discover additional abbreviations and variations.
"""

import csv
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


class WordClusterer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.insightful_words = []
        self.unchecked_words = []
        self.all_rows = []

    def load_csv_data(self):
        """Load the CSV data and separate insightful from unchecked words."""
        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                self.all_rows.append(row)

                word = row["word"].lower().strip()
                insightful = int(row.get("insightful", 0))
                occurrences = int(row.get("occurrences", 0))

                if i < 1005:  # First 1005 rows that were manually reviewed
                    if insightful == 1:
                        self.insightful_words.append(
                            {
                                "word": word,
                                "occurrences": occurrences,
                                "notes": row.get("notes", "").strip(),
                                "row_index": i,
                            }
                        )
                else:  # Remaining rows to cluster
                    self.unchecked_words.append(
                        {"word": word, "occurrences": occurrences, "row_index": i}
                    )

        print(
            f"📊 Loaded {len(self.insightful_words)} insightful words from first 1005 rows"
        )
        print(f"🔍 Found {len(self.unchecked_words)} unchecked words to cluster")

    def calculate_similarity_score(self, word1, word2):
        """Calculate similarity between two words using multiple metrics."""

        # 1. Sequence similarity (edit distance based)
        seq_similarity = SequenceMatcher(None, word1, word2).ratio()

        # 2. Character containment (one word contains the other)
        containment_score = 0
        if word1 in word2 or word2 in word1:
            containment_score = min(len(word1), len(word2)) / max(
                len(word1), len(word2)
            )

        # 3. Common prefix/suffix
        prefix_len = 0
        min_len = min(len(word1), len(word2))
        for i in range(min_len):
            if word1[i] == word2[i]:
                prefix_len += 1
            else:
                break

        suffix_len = 0
        for i in range(1, min_len + 1):
            if word1[-i] == word2[-i]:
                suffix_len += 1
            else:
                break

        prefix_suffix_score = (prefix_len + suffix_len) / max(len(word1), len(word2))

        # 4. Character overlap (same characters in different order)
        chars1 = set(word1)
        chars2 = set(word2)
        char_overlap = (
            len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0
        )

        # 5. Abbreviation pattern matching (consecutive characters)
        abbrev_score = self.check_abbreviation_pattern(word1, word2)

        # Weighted combination of all scores
        final_score = (
            seq_similarity * 0.3
            + containment_score * 0.25
            + prefix_suffix_score * 0.2
            + char_overlap * 0.1
            + abbrev_score * 0.15
        )

        return final_score, {
            "sequence": seq_similarity,
            "containment": containment_score,
            "prefix_suffix": prefix_suffix_score,
            "char_overlap": char_overlap,
            "abbreviation": abbrev_score,
        }

    def check_abbreviation_pattern(self, short_word, long_word):
        """Check if short_word could be an abbreviation of long_word."""
        if len(short_word) >= len(long_word):
            return 0

        # Check if characters of short_word appear in order in long_word
        short_idx = 0
        for char in long_word:
            if short_idx < len(short_word) and char == short_word[short_idx]:
                short_idx += 1

        if short_idx == len(short_word):
            return 0.8  # High score for abbreviation pattern
        else:
            return 0

    def find_clusters(self, similarity_threshold=0.6):
        """Find clusters of similar words."""
        clusters = defaultdict(list)

        print(
            f"\n🔍 CLUSTERING ANALYSIS (similarity threshold: {similarity_threshold})"
        )
        print("-" * 70)

        for unchecked in self.unchecked_words:
            best_matches = []

            for insightful in self.insightful_words:
                score, breakdown = self.calculate_similarity_score(
                    unchecked["word"], insightful["word"]
                )

                if score >= similarity_threshold:
                    best_matches.append(
                        {
                            "insightful_word": insightful["word"],
                            "score": score,
                            "breakdown": breakdown,
                            "insightful_occurrences": insightful["occurrences"],
                            "insightful_notes": insightful["notes"],
                        }
                    )

            if best_matches:
                # Sort by similarity score
                best_matches.sort(key=lambda x: x["score"], reverse=True)
                clusters[unchecked["word"]] = {
                    "unchecked_data": unchecked,
                    "matches": best_matches[:3],  # Top 3 matches
                }

        return clusters

    def generate_clustering_report(self, clusters, output_file):
        """Generate a detailed clustering report."""

        # Sort clusters by best similarity score
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: x[1]["matches"][0]["score"] if x[1]["matches"] else 0,
            reverse=True,
        )

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "unchecked_word",
                    "occurrences",
                    "best_match",
                    "similarity_score",
                    "match_occurrences",
                    "match_notes",
                    "sequence_sim",
                    "containment",
                    "prefix_suffix",
                    "char_overlap",
                    "abbreviation",
                    "suggested_action",
                ]
            )

            for unchecked_word, cluster_data in sorted_clusters:
                unchecked = cluster_data["unchecked_data"]
                best_match = cluster_data["matches"][0]
                breakdown = best_match["breakdown"]

                # Suggest action based on similarity type
                if breakdown["abbreviation"] > 0.5:
                    action = "ABBREVIATION - Consider marking as insightful"
                elif breakdown["containment"] > 0.7:
                    action = "VARIANT - Likely related to insightful word"
                elif breakdown["sequence"] > 0.8:
                    action = "SIMILAR - Review for relevance"
                else:
                    action = "WEAK_MATCH - Low priority"

                writer.writerow(
                    [
                        unchecked_word,
                        unchecked["occurrences"],
                        best_match["insightful_word"],
                        f"{best_match['score']:.3f}",
                        best_match["insightful_occurrences"],
                        best_match["insightful_notes"],
                        f"{breakdown['sequence']:.3f}",
                        f"{breakdown['containment']:.3f}",
                        f"{breakdown['prefix_suffix']:.3f}",
                        f"{breakdown['char_overlap']:.3f}",
                        f"{breakdown['abbreviation']:.3f}",
                        action,
                    ]
                )

        print(f"📄 Detailed clustering report saved to: {output_file}")

    def show_clustering_preview(self, clusters, top_n=20):
        """Show preview of top clustering results."""

        print(f"\n📊 TOP {top_n} CLUSTERING RESULTS:")
        print("-" * 80)

        # Sort by best similarity score
        sorted_items = sorted(
            clusters.items(),
            key=lambda x: x[1]["matches"][0]["score"] if x[1]["matches"] else 0,
            reverse=True,
        )

        for i, (unchecked_word, cluster_data) in enumerate(sorted_items[:top_n], 1):
            unchecked = cluster_data["unchecked_data"]
            best_match = cluster_data["matches"][0]
            breakdown = best_match["breakdown"]

            print(f"{i:2d}. {unchecked_word:<15} ({unchecked['occurrences']:,} occur.)")
            print(
                f"    → {best_match['insightful_word']:<15} (similarity: {best_match['score']:.3f})"
            )

            # Show breakdown of similarity components
            components = []
            if breakdown["abbreviation"] > 0.1:
                components.append(f"abbrev:{breakdown['abbreviation']:.2f}")
            if breakdown["containment"] > 0.1:
                components.append(f"contain:{breakdown['containment']:.2f}")
            if breakdown["sequence"] > 0.1:
                components.append(f"seq:{breakdown['sequence']:.2f}")

            print(f"    Components: {', '.join(components)}")
            print()

    def update_csv_with_suggestions(self, clusters, output_csv):
        """Update the original CSV with clustering suggestions."""

        # Create a mapping of row indices to suggested values
        suggestions = {}

        for unchecked_word, cluster_data in clusters.items():
            unchecked = cluster_data["unchecked_data"]
            best_match = cluster_data["matches"][0]
            breakdown = best_match["breakdown"]

            row_index = unchecked["row_index"]

            # Suggest insightful=1 for strong matches
            if (
                breakdown["abbreviation"] > 0.5
                or breakdown["containment"] > 0.8
                or best_match["score"] > 0.85
            ):
                suggestions[row_index] = {
                    "insightful": "1",
                    "notes": f"Auto-suggested: similar to '{best_match['insightful_word']}' (score: {best_match['score']:.3f})",
                }

        # Update the CSV
        updated_rows = []
        for i, row in enumerate(self.all_rows):
            if i in suggestions:
                row["insightful"] = suggestions[i]["insightful"]
                if not row["notes"].strip():
                    row["notes"] = suggestions[i]["notes"]
                else:
                    row["notes"] += f"; {suggestions[i]['notes']}"
            updated_rows.append(row)

        # Write updated CSV
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            if updated_rows:
                writer = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
                writer.writeheader()
                writer.writerows(updated_rows)

        print(f"📝 Updated CSV with {len(suggestions)} suggestions: {output_csv}")


def main():
    csv_file = "ifs_parameter_procedure_words.csv"

    print("🔍 IFS WORD CLUSTERING TOOL")
    print("=" * 50)

    if not Path(csv_file).exists():
        print(f"❌ CSV file not found: {csv_file}")
        return

    clusterer = WordClusterer(csv_file)

    print("📂 Loading CSV data...")
    clusterer.load_csv_data()

    if not clusterer.insightful_words:
        print("❌ No insightful words found in first 1005 rows")
        return

    print("\n🧮 Calculating similarities...")
    clusters = clusterer.find_clusters(similarity_threshold=0.55)

    print(f"📊 Found {len(clusters)} potential clusters")

    if clusters:
        clusterer.show_clustering_preview(clusters, top_n=25)

        # Generate detailed report
        report_file = "word_clustering_report.csv"
        clusterer.generate_clustering_report(clusters, report_file)

        # Update original CSV with suggestions
        updated_csv = "ifs_parameter_procedure_words_with_suggestions.csv"
        clusterer.update_csv_with_suggestions(clusters, updated_csv)

        print(f"\n✅ CLUSTERING COMPLETE!")
        print(f"📄 Detailed report: {report_file}")
        print(f"📝 Updated CSV: {updated_csv}")
    else:
        print("❌ No similar clusters found with current threshold")


if __name__ == "__main__":
    main()
