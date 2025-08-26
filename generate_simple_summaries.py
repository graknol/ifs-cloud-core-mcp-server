#!/usr/bin/env python3
"""
Simple Batch Summary Generator

Generates 200 summaries without the GUI, saving them for later review.
Uses minimal setup to avoid initialization issues.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleBatchGenerator:
    """Simple batch summary generator."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None

        # Create output directory
        self.output_dir = Path("batch_summaries")
        self.output_dir.mkdir(exist_ok=True)

        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"generated_summaries_{timestamp}.json"

    def load_model(self):
        """Load the summary generation model."""
        logger.info(f"Loading model: {self.config['model_name']}")

        # Set CUDA memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

            # Set up tokenizer properly to avoid attention mask warnings
            if self.tokenizer.pad_token is None:
                # Use a different token for padding if possible
                if (
                    hasattr(self.tokenizer, "unk_token")
                    and self.tokenizer.unk_token is not None
                ):
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    # Fallback to eos_token but we'll handle attention masks explicitly
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(
                f"Tokenizer setup: pad_token='{self.tokenizer.pad_token}', eos_token='{self.tokenizer.eos_token}'"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )

            logger.info("✅ Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def find_pls_files(self) -> List[Path]:
        """Find PLSQL files in the IFS source directory."""
        ifs_path = Path(self.config["ifs_source_path"])

        if not ifs_path.exists():
            raise FileNotFoundError(f"IFS source path not found: {ifs_path}")

        logger.info(f"Searching for PLSQL files in: {ifs_path}")

        # Find all .plsql files (not .pls)
        plsql_files = list(ifs_path.rglob("*.plsql"))
        logger.info(f"Found {len(plsql_files)} PLSQL files")

        return plsql_files

    def extract_simple_procedure(self, file_path: Path) -> Dict:
        """Extract a simple procedure representation from a PLSQL file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Simple heuristic extraction
            name = file_path.stem
            module = (
                file_path.parent.parent.name.lower()
                if file_path.parent.parent
                else "unknown"
            )

            # Find procedure/function declarations (simple approach)
            lines = content.split("\n")
            procedure_lines = []
            parameters = []

            for i, line in enumerate(lines):
                line_clean = line.strip().upper()
                if any(
                    keyword in line_clean for keyword in ["PROCEDURE ", "FUNCTION "]
                ):
                    # Take this line and next few lines for context
                    context_lines = lines[i : min(i + 10, len(lines))]
                    procedure_lines.extend(context_lines)
                    break

            # Simple parameter extraction
            param_section = " ".join(procedure_lines).upper()
            if "(" in param_section and ")" in param_section:
                param_text = param_section.split("(")[1].split(")")[0]
                if param_text.strip():
                    parameters = [p.strip() for p in param_text.split(",") if p.strip()]

            # Truncate content for context (first 2000 chars)
            full_text = content[:2000] + "..." if len(content) > 2000 else content

            return {
                "name": name,
                "module": module,
                "file_path": str(file_path),
                "full_text": full_text,
                "parameters": parameters[:5],  # Limit parameters
                "procedure_lines": "\n".join(procedure_lines),
            }

        except Exception as e:
            logger.warning(f"Failed to extract from {file_path}: {e}")
            return None

    def create_prompt(self, procedure: Dict) -> str:
        """Create a prompt for summary generation."""
        prompt = f"""Analyze this IFS Cloud procedure and provide a concise business summary.

Procedure: {procedure.get('name', 'Unknown')}
Module: {procedure.get('module', 'Unknown').upper()}
File: {Path(procedure.get('file_path', '')).name}

Code Context:
{procedure.get('procedure_lines', procedure.get('full_text', ''))}

Provide a clear, concise business summary focusing on:
1. Primary business purpose
2. Key functionality
3. Business impact

Summary:"""

        return prompt

    def generate_summary(self, procedure: Dict) -> str:
        """Generate summary for a procedure."""
        try:
            prompt = self.create_prompt(procedure)

            # Tokenize with proper attention mask
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config["max_length"],
                truncation=True,
                padding=True,
                return_attention_mask=True,
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            if torch.cuda.is_available():
                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)

            # Generate summary with attention mask
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract summary
            if "Summary:" in response:
                summary = response.split("Summary:")[-1].strip()
            else:
                summary = response[len(prompt) :].strip()

            # Clean up
            summary = summary.replace("\n\n", "\n").strip()
            if len(summary) > 800:
                summary = summary[:800] + "..."

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Generation failed: {str(e)}"

    def run_generation(self):
        """Run the batch generation process."""
        logger.info("🚀 Starting simple batch generation...")

        # Load model
        self.load_model()

        # Find files
        plsql_files = self.find_pls_files()

        if not plsql_files:
            logger.error("No PLSQL files found!")
            return

        target_count = min(self.config.get("target_summaries", 200), len(plsql_files))
        logger.info(f"🎯 Generating summaries for {target_count} files...")

        results = []

        for i, plsql_file in enumerate(plsql_files[:target_count]):
            try:
                logger.info(f"Processing {i+1}/{target_count}: {plsql_file.name}")

                # Extract procedure
                procedure = self.extract_simple_procedure(plsql_file)

                if not procedure:
                    logger.warning(f"Skipping {plsql_file.name} - extraction failed")
                    continue

                # Generate summary
                summary = self.generate_summary(procedure)

                # Create result
                result = {
                    "id": len(results) + 1,
                    "name": procedure["name"],
                    "module": procedure["module"],
                    "file_path": procedure["file_path"],
                    "full_text": procedure["full_text"],
                    "parameters": procedure["parameters"],
                    "generated_summary": summary,
                    "status": "generated",
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)

                # Save progress
                if len(results) % 10 == 0:
                    self.save_results(results, f"_progress_{len(results)}")
                    logger.info(f"💾 Progress saved: {len(results)} summaries")

                # Clear CUDA cache
                if len(results) % 25 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to process {plsql_file.name}: {e}")
                continue

        # Save final results
        self.save_results(results)
        logger.info(f"✅ Generation complete! Created {len(results)} summaries")
        logger.info(f"📁 Saved to: {self.output_file}")

        return results

    def save_results(self, results: List[Dict], suffix: str = ""):
        """Save results to JSON."""
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
                            "model_used": self.config["model_name"],
                            "config": self.config,
                        },
                        "summaries": results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except Exception as e:
            logger.error(f"Failed to save: {e}")


def main():
    """Main function."""
    print("🎯 Simple IFS Cloud Batch Summary Generator")
    print("=" * 50)

    config = {
        "model_name": "unsloth/Qwen2.5-7B-Instruct",
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "max_length": 2048,  # Conservative for generation
        "target_summaries": 200,
    }

    print(f"📊 Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Target: {config['target_summaries']} summaries")
    print(f"  Source: {config['ifs_source_path']}")
    print()

    # Check source path
    if not Path(config["ifs_source_path"]).exists():
        print("⚠️  IFS source path not found!")
        print("Update the path in the script or ensure IFS source is available.")
        return

    try:
        generator = SimpleBatchGenerator(config)
        results = generator.run_generation()

        if results:
            print(f"🎉 Success! Generated {len(results)} summaries")
            print(f"📁 File: {generator.output_file}")
            print()
            print("Next steps:")
            print("1. Review summaries with: python launch_gui_review.py")
            print("2. After review, start training with modified summaries")

    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
