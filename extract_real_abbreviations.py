#!/usr/bin/env python3
"""
Extract REAL abbreviations from IFS Cloud comprehensive analysis.
Focus on shortened forms of words, not common full words.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path


class RealAbbreviationExtractor:
    def __init__(self):
        # Known IFS abbreviation patterns (shortened → full form)
        self.known_abbreviations = {
            "qty": "quantity",
            "addr": "address",
            "purch": "purchase",
            "ord": "order",
            "cust": "customer",
            "supp": "supplier",
            "mfg": "manufacturing",
            "inv": "inventory",
            "proj": "project",
            "req": "requirement",
            "ref": "reference",
            "desc": "description",
            "info": "information",
            "config": "configuration",
            "auth": "authorization",
            "temp": "temporary",
            "stat": "status",
            "proc": "process",
            "val": "value",
            "rec": "record",
            "num": "number",
            "std": "standard",
            "mgmt": "management",
            "dept": "department",
            "emp": "employee",
            "fin": "financial",
            "acct": "account",
            "trans": "transaction",
            "sched": "schedule",
            "del": "delivery",
            "manuf": "manufacture",
            "insp": "inspection",
            "ctrl": "control",
            "sys": "system",
            "admin": "administration",
            "oper": "operation",
            "prod": "production",
            "qual": "quality",
        }

        # Patterns that suggest abbreviations
        self.abbreviation_patterns = [
            # 2-4 character words ending in common abbreviation suffixes
            r"^[a-z]{2,4}$",  # Short words
            # Words with consonant clusters (missing vowels)
            r"^[bcdfghjklmnpqrstvwxyz]{2,}[aeiou]?[bcdfghjklmnpqrstvwxyz]*$",
            # Common abbreviation endings
            r".*[bcdfghjklmnpqrstvwxyz]{2}$",  # Ends with consonant cluster
            r".*mgmt$",  # management
            r".*ctrl$",  # control
            r".*proc$",  # process
            r".*sched$",  # schedule
            r".*manuf$",  # manufacture
            r".*auth$",  # authorization
            r".*admin$",  # administration
        ]

    def is_likely_abbreviation(self, word):
        """Check if a word is likely an abbreviation."""
        if len(word) < 2:
            return False

        # Skip common full words that aren't abbreviations
        full_words = {
            "number",
            "order",
            "value",
            "history",
            "customer",
            "quantity",
            "project",
            "description",
            "amount",
            "supplier",
            "purchase",
            "inventory",
            "information",
            "address",
            "delivery",
            "reference",
            "company",
            "part",
            "cost",
            "site",
            "currency",
            "invoice",
            "tax",
            "state",
            "status",
            "item",
            "cancelled",
            "planned",
            "condition",
            "closed",
            "active",
            "created",
            "updated",
            "process",
            "system",
            "control",
            "method",
            "function",
        }

        if word.lower() in full_words:
            return False

        # Check if it's a known abbreviation
        if word.lower() in self.known_abbreviations:
            return True

        # Check abbreviation patterns
        for pattern in self.abbreviation_patterns:
            if re.match(pattern, word.lower()):
                # Additional checks for likely abbreviations
                if len(word) <= 4 and word.isalpha():
                    return True
                if word.lower().endswith(
                    ("mgmt", "ctrl", "proc", "sched", "manuf", "auth", "admin")
                ):
                    return True

        return False

    def extract_abbreviations_from_results(self, results_file):
        """Extract real abbreviations from comprehensive analysis results."""
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        real_abbreviations = {}
        all_terms = Counter()

        # Process the top_abbreviations directly from the summary
        top_abbreviations = data.get("top_abbreviations", {})

        for term, count in top_abbreviations.items():
            if self.is_likely_abbreviation(term):
                real_abbreviations[term] = count

            all_terms[term] = count

        return real_abbreviations, all_terms

    def analyze_abbreviation_context(self, results_file):
        """Analyze where abbreviations appear most frequently."""
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        abbreviation_modules = defaultdict(list)

        # For now, just return the abbreviations we found - context analysis
        # would need the full module breakdown which might be in a different structure
        top_abbreviations = data.get("top_abbreviations", {})

        for term, count in top_abbreviations.items():
            if self.is_likely_abbreviation(term):
                abbreviation_modules[term].append(
                    {
                        "module": "ALL_MODULES",
                        "count": count,
                        "description": "Aggregated across all IFS modules",
                    }
                )

        return dict(abbreviation_modules)


def main():
    extractor = RealAbbreviationExtractor()
    results_file = "comprehensive_ifs_analysis_all_modules.json"

    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return

    print("🔍 EXTRACTING REAL ABBREVIATIONS FROM IFS ANALYSIS")
    print("=" * 60)

    # Extract real abbreviations
    real_abbreviations, all_terms = extractor.extract_abbreviations_from_results(
        results_file
    )

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Total terms analyzed: {len(all_terms)}")
    print(f"   Real abbreviations found: {len(real_abbreviations)}")

    # Show top real abbreviations
    print(f"\n🔤 TOP REAL ABBREVIATIONS:")
    print("-" * 40)
    sorted_abbreviations = sorted(
        real_abbreviations.items(), key=lambda x: x[1], reverse=True
    )

    for i, (abbrev, count) in enumerate(sorted_abbreviations[:20], 1):
        full_form = extractor.known_abbreviations.get(abbrev.lower(), "unknown")
        print(f"{i:2d}. {abbrev:<8} → {full_form:<15} ({count:,} occurrences)")

    # Analyze context
    print(f"\n🏗️  ABBREVIATION CONTEXT ANALYSIS:")
    print("-" * 40)

    context = extractor.analyze_abbreviation_context(results_file)

    for abbrev, modules in list(context.items())[:10]:
        total_count = sum(m["count"] for m in modules)
        top_module = modules[0] if modules else {}
        full_form = extractor.known_abbreviations.get(abbrev.lower(), "unknown")

        print(f"\n{abbrev.upper()} → {full_form}")
        print(f"  Total: {total_count:,} occurrences across {len(modules)} modules")
        if top_module:
            print(
                f"  Most used in: {top_module['module']} ({top_module['count']:,} times)"
            )

    # Save enhanced abbreviation mappings
    enhanced_mappings = {}
    for abbrev, count in sorted_abbreviations:
        if count >= 10:  # Only include abbreviations with significant usage
            full_form = extractor.known_abbreviations.get(abbrev.lower(), abbrev)
            enhanced_mappings[abbrev] = {
                "full_form": full_form,
                "occurrences": count,
                "is_confirmed_abbreviation": True,
            }

    output_file = "real_ifs_abbreviations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "abbreviation_mappings": enhanced_mappings,
                "total_abbreviations": len(enhanced_mappings),
                "analysis_date": "2025-08-25",
                "source": "IFS Cloud 25.1.0 comprehensive analysis",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n💾 RESULTS SAVED:")
    print(f"   File: {output_file}")
    print(f"   Confirmed abbreviations: {len(enhanced_mappings)}")
    print("\n✅ Real abbreviation extraction complete!")


if __name__ == "__main__":
    main()
