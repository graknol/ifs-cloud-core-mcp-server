#!/usr/bin/env python3
"""
GUI Review Tool for Batch Generated Summaries

This script loads previously generated summaries and allows you to review
and modify them using the same GUI interface as the training loop.
"""

import os
import sys
import json
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from supervised_training_loop import SupervisedTrainingLoop


class SummaryReviewGUI:
    """GUI for reviewing batch-generated summaries."""

    def __init__(self, summaries_file: str):
        self.summaries_file = Path(summaries_file)
        self.summaries_data = None
        self.modified_summaries = []

        # Load summaries
        self.load_summaries()

        if not self.summaries_data:
            raise ValueError("No summaries loaded")

        # Create a mock training loop for GUI functionality
        self.create_training_loop_gui()

    def load_summaries(self):
        """Load summaries from JSON file."""
        try:
            with open(self.summaries_file, "r", encoding="utf-8") as f:
                self.summaries_data = json.load(f)

            summaries = self.summaries_data.get("summaries", [])
            print(f"✅ Loaded {len(summaries)} summaries from {self.summaries_file}")

        except Exception as e:
            print(f"❌ Failed to load summaries: {e}")
            self.summaries_data = None

    def create_training_loop_gui(self):
        """Create a training loop instance for GUI functionality."""
        # Mock configuration
        config = {
            "summary_model_name": "mock",
            "training_model_name": "mock",
            "ifs_source_path": "mock",
            "save_dir": "./training_checkpoints",
            "target_summaries": len(self.summaries_data.get("summaries", [])),
        }

        # Create training loop (won't load models)
        self.training_loop = SupervisedTrainingLoop(**config)

        # Replace the summaries with our loaded data
        summaries = self.summaries_data.get("summaries", [])

        # Convert to training loop format
        converted_summaries = []
        for summary in summaries:
            converted = {
                "name": summary.get("name", "Unknown"),
                "module": summary.get("module", "Unknown"),
                "file_path": summary.get("file_path", ""),
                "full_text": summary.get("full_text", ""),
                "parameters": summary.get("parameters", []),
                "generated_summary": summary.get("generated_summary", ""),
                "human_summary": summary.get(
                    "generated_summary", ""
                ),  # Start with generated
                "status": "generated",
            }
            converted_summaries.append(converted)

        # Set up the GUI with our summaries
        self.training_loop.all_summaries = []  # Will be populated as we accept
        self.training_loop.batch_data = converted_summaries
        self.training_loop.current_index = 0

        # Override the save method to save our format
        original_save = self.training_loop.save_training_state

        def custom_save():
            """Custom save that updates our JSON file."""
            self.save_modified_summaries()
            return original_save()

        self.training_loop.save_training_state = custom_save

        print(f"🎯 Ready to review {len(converted_summaries)} summaries")
        print("Use the GUI to review and modify summaries as needed.")
        print("Modified summaries will be saved back to the JSON file.")

    def save_modified_summaries(self):
        """Save modified summaries back to the JSON file."""
        try:
            # Update the original data with modifications
            accepted_summaries = self.training_loop.all_summaries

            # Update summaries in the original data
            summaries = self.summaries_data.get("summaries", [])

            # Create a lookup for accepted summaries
            accepted_lookup = {s.get("name", ""): s for s in accepted_summaries}

            # Update the summaries with human modifications
            for i, summary in enumerate(summaries):
                name = summary.get("name", "")
                if name in accepted_lookup:
                    accepted = accepted_lookup[name]
                    summary["human_summary"] = accepted.get("human_summary", "")
                    summary["status"] = accepted.get("status", "accepted")
                    summary["modified"] = True

            # Save back to file
            backup_file = self.summaries_file.with_suffix(".backup.json")

            # Create backup
            if self.summaries_file.exists():
                import shutil

                shutil.copy2(self.summaries_file, backup_file)

            # Save updated version
            with open(self.summaries_file, "w", encoding="utf-8") as f:
                json.dump(self.summaries_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved modifications to {self.summaries_file}")
            print(f"📁 Backup created: {backup_file}")

        except Exception as e:
            print(f"❌ Failed to save modifications: {e}")

    def run_gui(self):
        """Start the GUI review process."""
        print("\n🚀 Starting GUI review...")
        print("=" * 50)
        print("GUI Controls:")
        print("  Ctrl+Enter → Accept current summary")
        print("  Ctrl+S → Skip this procedure")
        print("  Ctrl+E → Focus summary editor")
        print("  Ctrl+, / Ctrl+. → Navigate procedures")
        print("  Ctrl+Q → Save and continue")
        print("=" * 50)

        try:
            # Start the GUI
            self.training_loop.start_gui_review()

        except Exception as e:
            print(f"❌ GUI error: {e}")
            import traceback

            traceback.print_exc()


def find_latest_summaries_file() -> Optional[str]:
    """Find the latest generated summaries file."""
    batch_dir = Path("batch_summaries")

    if not batch_dir.exists():
        return None

    json_files = list(batch_dir.glob("generated_summaries_*.json"))

    if not json_files:
        return None

    # Return the most recent file
    latest = max(json_files, key=lambda x: x.stat().st_mtime)
    return str(latest)


def main():
    """Main execution function."""
    print("🔍 IFS Cloud Summary Review GUI")
    print("=" * 50)

    # Check for command line argument
    if len(sys.argv) > 1:
        summaries_file = sys.argv[1]
    else:
        # Try to find the latest file
        summaries_file = find_latest_summaries_file()

    if not summaries_file or not Path(summaries_file).exists():
        print("❌ No summaries file found!")
        print()
        print("Usage options:")
        print("1. python launch_gui_review.py <path_to_summaries.json>")
        print("2. python launch_gui_review.py  (auto-finds latest file)")
        print()
        print("Generate summaries first with:")
        print("  python generate_batch_summaries.py")
        return

    print(f"📁 Loading summaries from: {summaries_file}")

    try:
        # Create and run GUI
        review_gui = SummaryReviewGUI(summaries_file)
        review_gui.run_gui()

    except KeyboardInterrupt:
        print("\n🛑 Review interrupted by user")
    except Exception as e:
        print(f"❌ Error during review: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
