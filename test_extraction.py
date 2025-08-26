#!/usr/bin/env python3
"""Test the improved procedure extraction logic."""

import json
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from correlate_summaries_with_prompts import SummaryCorrelator


def test_procedure_extraction():
    """Test the new procedure extraction logic on a single procedure."""

    # Initialize the correlator
    correlator = SummaryCorrelator()

    # Load the first entry from final.json to test
    with open("batch_summaries/final.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No data found in final.json")
        return

    # Test the first procedure (Check_Common___)
    first_entry = data[0]
    print(f"Testing procedure: {first_entry['procedure_name']}")
    print(
        f"Original line range: {first_entry.get('line_start', 'N/A')}-{first_entry.get('line_end', 'N/A')}"
    )

    # Convert file path
    file_path_str = first_entry["file_path"]
    if file_path_str.startswith("C:\\repos\\_ifs\\25.1.0\\"):
        relative_path = file_path_str.replace("C:\\repos\\_ifs\\25.1.0\\", "")
        full_path = correlator.source_root / relative_path
    else:
        print(f"Unexpected path format: {file_path_str}")
        return

    print(f"File path: {full_path}")

    # Extract the procedure with the new logic
    extracted_code = correlator.extract_plsql_procedure(
        full_path,
        first_entry.get("line_start", 164),
        first_entry.get("line_end", 364),
        first_entry["procedure_name"],
    )

    if extracted_code:
        print(f"\n=== EXTRACTED PROCEDURE ({len(extracted_code)} chars) ===")
        print(extracted_code[:1000])  # Show first 1000 characters
        if len(extracted_code) > 1000:
            print("... (truncated)")

        # Count lines
        line_count = len(extracted_code.split("\n"))
        print(f"\nExtracted {line_count} lines (vs original range of ~200 lines)")

        # Check if it ends properly
        last_lines = extracted_code.strip().split("\n")[-3:]
        print("\nLast few lines:")
        for line in last_lines:
            print(f"  {line}")

    else:
        print("Failed to extract procedure")


if __name__ == "__main__":
    test_procedure_extraction()
