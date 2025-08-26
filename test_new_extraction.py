#!/usr/bin/env python3

import json
from correlate_summaries_with_prompts import SummaryCorrelator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Test the new extraction logic
correlator = SummaryCorrelator()

# Load final.json to get a test case
with open("batch_summaries/final.json", "r", encoding="utf-8") as f:
    final_data = json.load(f)

# Test with the first entry
test_entry = final_data[0]
print(f"Testing extraction for: {test_entry['procedure_name']}")
print(f"Available keys: {list(test_entry.keys())}")

# Get the file path (need to add the root directory)
summary_entry = {
    "procedure_name": test_entry["procedure_name"],
    "module_name": test_entry.get("module", "ACCRU"),  # Correct key
    "file_path": test_entry.get("file_path", ""),  # Correct key
    "line_start": test_entry.get("line_start", 164),
    "line_end": test_entry.get("line_end", 364),
}

print(f"Using summary entry: {summary_entry}")

# Test the new extraction
try:
    prompt = correlator.generate_prompt_for_procedure(summary_entry)

    if prompt:
        print("=" * 80)
        print("EXTRACTED PROCEDURE:")
        print("=" * 80)

        # Find where the procedure code starts in the prompt
        lines = prompt.split("\n")
        in_code_block = False
        procedure_lines = []

        for line in lines:
            if line.strip().startswith("```plsql"):
                in_code_block = True
                continue
            elif line.strip() == "```" and in_code_block:
                break
            elif in_code_block:
                procedure_lines.append(line)

        procedure_text = "\n".join(procedure_lines)
        print(f"Extracted {len(procedure_lines)} lines:")
        print(
            procedure_text[:800] + "..."
            if len(procedure_text) > 800
            else procedure_text
        )
    else:
        print("No prompt was generated - check the error above")

except Exception as e:
    print(f"Error during extraction: {e}")
    import traceback

    traceback.print_exc()
