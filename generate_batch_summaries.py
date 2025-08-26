#!/usr/bin/env python3
"""
Batch Summary Generator for IFS Cloud Procedures

This script generates summaries for 200 procedures using the large model
and saves them to a JSON file for later review in the GUI.

The generated summaries can then be loaded into the GUI for human verification
and editing before fine-tuning the smaller model.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from supervised_training_loop import SupervisedTrainingLoop

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BatchSummaryGenerator:
    """Generate summaries for procedures in batch mode."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None

        # Create a temporary training loop for extraction functionality
        self.training_loop = SupervisedTrainingLoop(**config)

        # Create output directory
        self.output_dir = Path("batch_summaries")
        self.output_dir.mkdir(exist_ok=True)

        # Output file for generated summaries
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"generated_summaries_{timestamp}.json"

    def load_model(self):
        """Load the large model for summary generation."""
        logger.info(f"Loading summary model: {self.config['summary_model_name']}")

        # Set CUDA memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["summary_model_name"]
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["summary_model_name"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )

            # Compile for better performance
            logger.info("🚀 Compiling model for optimal performance...")
            self.model = torch.compile(self.model, backend="inductor")
            logger.info("✅ Model loaded and compiled successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract_procedures(self) -> List[Dict]:
        """Extract procedures from IFS source code using existing training loop."""
        logger.info("🔍 Extracting procedures from IFS source...")

        try:
            # Use the training loop's extraction method
            self.training_loop.extract_procedures()
            procedures = self.training_loop.all_procedures

            target_count = self.config.get("target_summaries", 200)
            limited_procedures = procedures[:target_count]

            logger.info(
                f"✅ Extracted {len(limited_procedures)} procedures (from {len(procedures)} total)"
            )
            return limited_procedures

        except Exception as e:
            logger.error(f"Failed to extract procedures: {e}")
            return []

    def create_prompt(self, procedure: Dict) -> str:
        """Create a prompt for summary generation."""
        prompt = f"""Analyze this IFS Cloud procedure and provide a concise business summary.

Procedure: {procedure.get('name', 'Unknown')}
Module: {procedure.get('module', 'Unknown').upper()}

{procedure.get('full_text', '')}

Provide a clear, concise business summary focusing on:
1. Primary business purpose
2. Key functionality 
3. Important parameters and their roles
4. Business impact or use cases

Summary:"""

        return prompt

    def generate_summary(self, procedure: Dict) -> str:
        """Generate a summary for a single procedure."""
        try:
            prompt = self.create_prompt(procedure)

            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=self.config["max_length"],
                truncation=True,
            )

            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)

            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,  # Reasonable summary length
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the summary part
            if "Summary:" in response:
                summary = response.split("Summary:")[-1].strip()
            else:
                summary = response[len(prompt) :].strip()

            # Clean up the summary
            summary = summary.replace("\n\n", "\n").strip()
            if len(summary) > 1000:  # Reasonable length limit
                summary = summary[:1000] + "..."

            return summary

        except Exception as e:
            logger.error(
                f"Failed to generate summary for {procedure.get('name', 'Unknown')}: {e}"
            )
            return "Failed to generate summary due to processing error."

    def generate_batch_summaries(self):
        """Generate summaries for all procedures."""
        logger.info("🚀 Starting batch summary generation...")

        # Load model
        self.load_model()

        # Extract procedures
        procedures = self.extract_procedures()

        if not procedures:
            logger.error("No procedures found to process")
            return

        logger.info(f"🎯 Generating summaries for {len(procedures)} procedures...")

        # Generate summaries
        results = []
        batch_size = self.config.get("batch_size", 8)

        for i, procedure in enumerate(procedures):
            try:
                logger.info(
                    f"Processing {i+1}/{len(procedures)}: {procedure.get('name', 'Unknown')}"
                )

                summary = self.generate_summary(procedure)

                result = {
                    "id": i + 1,
                    "name": procedure.get("name", "Unknown"),
                    "module": procedure.get("module", "Unknown"),
                    "file_path": procedure.get("file_path", ""),
                    "full_text": procedure.get("full_text", ""),
                    "parameters": procedure.get("parameters", []),
                    "generated_summary": summary,
                    "status": "generated",  # Not yet reviewed
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)

                # Save progress periodically
                if (i + 1) % 20 == 0:
                    self.save_results(results, f"_progress_{i+1}")
                    logger.info(f"💾 Saved progress: {i+1} summaries")

                # Clear CUDA cache periodically
                if (i + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("🧹 Cleared CUDA cache")

            except Exception as e:
                logger.error(f"Failed to process procedure {i+1}: {e}")
                continue

        # Save final results
        self.save_results(results)
        logger.info(f"✅ Batch generation complete! Generated {len(results)} summaries")
        logger.info(f"📁 Saved to: {self.output_file}")

        # Cleanup
        self.cleanup_model()

        return results

    def save_results(self, results: List[Dict], suffix: str = ""):
        """Save results to JSON file."""
        output_file = self.output_file
        if suffix:
            stem = self.output_file.stem
            output_file = self.output_file.parent / f"{stem}{suffix}.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "total_summaries": len(results),
                            "generated_at": datetime.now().isoformat(),
                            "model_used": self.config["summary_model_name"],
                            "config": self.config,
                        },
                        "summaries": results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def cleanup_model(self):
        """Clean up model to free memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer

        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("🧹 Model cleanup completed")


def main():
    """Main execution function."""
    print("🎯 IFS Cloud Batch Summary Generator")
    print("=" * 50)

    # Configuration matching the two-phase training setup
    config = {
        "summary_model_name": "unsloth/Qwen2.5-7B-Instruct",  # Large model for generation
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "batch_size": 8,
        "max_length": 4096,
        "target_summaries": 200,
    }

    print(f"📊 Configuration:")
    print(f"  Model: {config['summary_model_name']}")
    print(f"  Target summaries: {config['target_summaries']}")
    print(f"  Max length: {config['max_length']} tokens")
    print(f"  IFS source: {config['ifs_source_path']}")
    print()

    # Check if IFS source exists
    if not Path(config["ifs_source_path"]).exists():
        print("⚠️  IFS source path not found. Please update the path in the script.")
        return

    try:
        # Create generator and run
        generator = BatchSummaryGenerator(config)
        results = generator.generate_batch_summaries()

        if results:
            print(f"🎉 Success! Generated {len(results)} summaries")
            print(f"📁 Output file: {generator.output_file}")
            print()
            print("Next steps:")
            print("1. Review the generated summaries in the output file")
            print("2. Use the GUI to modify summaries if needed:")
            print("   python launch_gui_review.py")
            print("3. Start fine-tuning with the reviewed summaries")

    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
