#!/usr/bin/env python3
"""
GUI Review Tool for Final Correlated Summaries

This script loads the final.json file with token counts and allows you to review
and modify the summaries using the enhanced GUI interface.
"""

import os
import sys
import json
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class FinalSummaryReviewGUI:
    """GUI for reviewing final correlated summaries with token counts."""

    def __init__(self, final_json_file: str):
        self.final_json_file = Path(final_json_file)
        self.final_data = None
        self.modified_summaries = []

        # Load summaries
        self.load_final_summaries()

        if not self.final_data:
            raise ValueError("No summaries loaded")

        # Create a mock training loop for GUI functionality
        self.create_training_loop_gui()

    def load_final_summaries(self):
        """Load summaries from final.json file."""
        try:
            with open(self.final_json_file, "r", encoding="utf-8") as f:
                self.final_data = json.load(f)

            print(
                f"✅ Loaded {len(self.final_data)} summaries from {self.final_json_file}"
            )

            # Show token count statistics
            total_summary_tokens = sum(
                entry.get("summary_token_count", 0) for entry in self.final_data
            )
            total_prompt_tokens = sum(
                entry.get("prompt_token_count", 0) for entry in self.final_data
            )
            avg_summary_tokens = (
                total_summary_tokens / len(self.final_data) if self.final_data else 0
            )
            avg_prompt_tokens = (
                total_prompt_tokens / len(self.final_data) if self.final_data else 0
            )

            print(f"📊 Token Statistics:")
            print(f"   • Total summary tokens: {total_summary_tokens:,}")
            print(f"   • Total prompt tokens: {total_prompt_tokens:,}")
            print(f"   • Average summary tokens: {avg_summary_tokens:.0f}")
            print(f"   • Average prompt tokens: {avg_prompt_tokens:.0f}")

        except Exception as e:
            print(f"❌ Failed to load summaries: {e}")
            self.final_data = None

    def create_training_loop_gui(self):
        """Create a training loop instance for GUI functionality."""

        # Import the GUI class directly to avoid full training loop initialization
        from supervised_training_loop import SummaryReviewGUI

        # Convert final.json format to training loop format
        converted_summaries = []
        for entry in self.final_data:
            # Extract just the procedure code from the regenerated prompt
            regenerated_prompt = entry.get("regenerated_prompt", "")
            procedure_code = self.extract_procedure_code(regenerated_prompt)

            converted = {
                "name": entry.get("procedure_name", "Unknown"),
                "module_name": entry.get("module", "unknown"),
                "file_path": entry.get("file_path", ""),
                "full_text": procedure_code,  # Use extracted procedure code
                "prompt": procedure_code,  # Clean procedure code only
                "parameters": [],  # Not available in final.json
                "generated_summary": entry.get("original_summary", ""),
                "human_summary": entry.get(
                    "original_summary", ""
                ),  # Start with original
                "status": "generated",
                # Add token count information for display
                "summary_token_count": entry.get("summary_token_count", 0),
                "prompt_token_count": entry.get("prompt_token_count", 0),
                "complexity_score": entry.get("complexity_score", 0),
                "content_length": entry.get("content_length", 0),
                "generation_time": entry.get("generation_time", 0),
                "correlation_timestamp": entry.get("correlation_timestamp", ""),
                "original_id": entry.get("id", 0),
            }
            converted_summaries.append(converted)

        # Create GUI directly without full training loop
        self.root = tk.Tk()
        self.gui = SummaryReviewGUI(self.root, converted_summaries)

        # Add required attributes that the GUI expects
        self.gui.all_summaries = []

        # Add custom save method that updates our JSON
        def custom_save():
            """Custom save that updates our JSON file."""
            try:
                # Update the original data with any changes
                for i, converted in enumerate(converted_summaries):
                    if i < len(self.final_data):
                        self.final_data[i]["original_summary"] = converted[
                            "human_summary"
                        ]

                # Save back to final.json
                backup_file = self.final_json_file.with_suffix(".json.backup")
                if self.final_json_file.exists():
                    import shutil

                    shutil.copy2(self.final_json_file, backup_file)
                    print(f"📁 Backup saved to {backup_file}")

                with open(self.final_json_file, "w", encoding="utf-8") as f:
                    json.dump(self.final_data, f, indent=2, ensure_ascii=False)

                print(f"💾 Saved changes to {self.final_json_file}")

            except Exception as e:
                print(f"❌ Error saving: {e}")

        # Add the save_training_state method to the GUI
        self.gui.save_training_state = custom_save

        print(
            f"🎯 Ready to review {len(converted_summaries)} final summaries with token counts"
        )

    def save_modified_summaries(self):
        """Save modified summaries back to the final.json file."""
        try:
            modified_data = []

            for proc in self.training_loop.batch_data:
                if proc.get("status") in ["accepted", "edited"]:
                    # Find original entry and update
                    original_id = proc.get("original_id", 0)
                    for entry in self.final_data:
                        if entry.get("id") == original_id:
                            entry["original_summary"] = proc["human_summary"]
                            break

            # Save updated data
            with open(self.final_json_file, "w", encoding="utf-8") as f:
                json.dump(self.final_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved modified summaries to {self.final_json_file}")

        except Exception as e:
            print(f"❌ Error saving modified summaries: {e}")

    def run_gui(self):
        """Start the GUI review process."""
        print("\n🚀 Starting Final Summary Review GUI...")
        print("=" * 50)
        print("GUI Controls:")
        print("  Ctrl+Enter → Accept current summary")
        print("  Ctrl+S → Skip this procedure")
        print("  Ctrl+E → Focus summary editor")
        print("  Ctrl+, / Ctrl+. → Navigate procedures")
        print("  Ctrl+Q → Save and continue")
        print("=" * 50)
        print("💡 Token counts are displayed in the status bar!")
        print("=" * 50)

        try:
            # Start the GUI main loop
            self.gui.run()

        except Exception as e:
            print(f"❌ GUI error: {e}")
            import traceback

            traceback.print_exc()


def find_final_json() -> Optional[str]:
    """Find the final.json file."""
    # Check current directory and batch_summaries
    possible_paths = [
        "final.json",
        "batch_summaries/final.json",
        Path("batch_summaries") / "final.json",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return str(path)

    return None


def main():
    """Main execution function."""
    print("🔍 IFS Cloud Final Summary Review GUI")
    print("=" * 50)

    # Check for command line argument
    if len(sys.argv) > 1:
        final_file = sys.argv[1]
    else:
        # Try to find the final.json file
        final_file = find_final_json()

    if not final_file or not Path(final_file).exists():
        print("❌ No final.json file found!")
        print()
        print("Usage options:")
        print("1. python launch_final_gui_review.py <path_to_final.json>")
        print("2. python launch_final_gui_review.py  (auto-finds final.json)")
        print()
        print("Expected locations:")
        print("  • ./final.json")
        print("  • ./batch_summaries/final.json")
        return

    print(f"📁 Loading summaries from: {final_file}")

    try:
        # Create and run GUI
        review_gui = FinalSummaryReviewGUI(final_file)
        review_gui.run_gui()

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
