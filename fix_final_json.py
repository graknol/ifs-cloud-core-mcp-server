#!/usr/bin/env python3
"""
Fix the regenerated_prompt field in final.json by using proper procedure extraction.
"""

import json
import logging
from pathlib import Path
from correlate_summaries_with_prompts import SummaryCorrelator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_final_json():
    """Fix the regenerated_prompt field in final.json with proper procedure extraction."""

    # Load the existing final.json
    final_path = Path("batch_summaries/final.json")

    if not final_path.exists():
        print(f"❌ Error: Could not find {final_path}")
        return

    # Create a backup
    backup_path = final_path.with_suffix(".json.backup_fixed")

    with open(final_path, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    # Save backup
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"📋 Loaded {len(final_data)} entries from {final_path}")
    print(f"💾 Created backup: {backup_path}")

    # Create correlator
    correlator = SummaryCorrelator()

    # Process each entry
    updated_count = 0
    failed_count = 0

    for i, entry in enumerate(final_data):
        try:
            logger.info(
                f"Processing {i+1}/{len(final_data)}: {entry['procedure_name']}"
            )

            # Generate the new prompt with correct procedure extraction
            new_prompt = correlator.generate_prompt_for_procedure(entry)

            if new_prompt:
                # Update the regenerated_prompt field
                entry["regenerated_prompt"] = new_prompt

                # Recalculate prompt token count
                entry["prompt_token_count"] = correlator.estimate_token_count(
                    new_prompt
                )

                # Add a field to track the fix
                entry["prompt_fixed_timestamp"] = (
                    "2025-08-26T" + str(i).zfill(2) + ":00:00Z"
                )

                updated_count += 1

                # Show improvement
                new_lines = len(new_prompt.split("\n"))
                print(
                    f"  ✅ Updated {entry['procedure_name']} - New prompt: {new_lines} lines, {entry['prompt_token_count']} tokens"
                )

            else:
                failed_count += 1
                print(f"  ❌ Failed to generate prompt for {entry['procedure_name']}")

        except Exception as e:
            failed_count += 1
            logger.error(
                f"Error processing {entry.get('procedure_name', 'unknown')}: {e}"
            )

    # Save the updated file
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Processing complete!")
    print(f"✅ Updated: {updated_count} entries")
    print(f"❌ Failed: {failed_count} entries")
    print(f"💾 Updated file: {final_path}")
    print(
        f"\n💡 The GUI should now show only the specific procedures instead of entire files!"
    )


if __name__ == "__main__":
    print("🔧 IFS Cloud Final.json Procedure Fix")
    print("=" * 50)
    fix_final_json()
