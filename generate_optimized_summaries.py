#!/usr/bin/env python3
"""
Optimized Batch Summary Generator

This version:
1. Keeps entire model on GPU for maximum performance
2. Uses proper procedure extraction from supervised_training_loop
3. Resumes from existing progress if available
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


class OptimizedBatchGenerator:
    """Optimized batch generator with full GPU utilization and proper procedure extraction."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None

        # Create a training loop instance for proper extraction
        temp_config = {
            "summary_model_name": config["model_name"],
            "training_model_name": config["model_name"],  # Use same model for both
            "ifs_source_path": config["ifs_source_path"],
            "save_dir": "./temp_extraction",
            "target_summaries": config["target_summaries"],
            "max_length": config["max_length"],
            "batch_size": config.get("batch_size", 8),
            "training_epochs": config.get("training_epochs", 15),
            "data_augmentation": config.get("data_augmentation", True),
            "two_phase_training": config.get("two_phase_training", True),
        }
        self.training_loop = SupervisedTrainingLoop(**temp_config)

        # Output directory
        self.output_dir = Path("batch_summaries")
        self.output_dir.mkdir(exist_ok=True)

        # Find or create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"optimized_summaries_{timestamp}.json"

    def load_model_full_gpu(self):
        """Load model with aggressive GPU optimization."""
        logger.info(
            f"Loading model with full GPU optimization: {self.config['model_name']}"
        )

        # Set aggressive CUDA memory settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:128"
        )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

            # Set up tokenizer properly
            if self.tokenizer.pad_token is None:
                if (
                    hasattr(self.tokenizer, "unk_token")
                    and self.tokenizer.unk_token is not None
                ):
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(
                f"Tokenizer setup: pad_token='{self.tokenizer.pad_token}', eos_token='{self.tokenizer.eos_token}'"
            )

            # Load model with maximum GPU utilization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.bfloat16,
                device_map="cuda",  # Force all on GPU
                max_memory={0: "14GB"},  # Use most of 16GB, leave 2GB buffer
                low_cpu_mem_usage=False,  # Don't offload to CPU
                attn_implementation="sdpa",
                trust_remote_code=True,
            )

            # Compile model for maximum performance
            logger.info("🚀 Compiling model for maximum performance...")
            self.model = torch.compile(self.model, mode="max-autotune")

            logger.info("✅ Model loaded fully on GPU with optimizations")

            # Print memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )
                logger.info(
                    f"GPU Memory: {memory_used:.2f}GB / {memory_total:.2f}GB ({memory_used/memory_total*100:.1f}%)"
                )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract_procedures_properly(self) -> List[Dict]:
        """Extract procedures using the proper training loop method."""
        logger.info("🔍 Extracting procedures using training loop methods...")

        try:
            # Use the training loop's extraction - this gives us proper format
            self.training_loop.extract_procedures()
            procedures = self.training_loop.all_procedures

            if not procedures:
                logger.error("No procedures extracted!")
                return []

            target_count = self.config.get("target_summaries", 200)
            limited_procedures = procedures[:target_count]

            logger.info(
                f"✅ Extracted {len(limited_procedures)} procedures (from {len(procedures)} total)"
            )

            # Log sample of what we extracted
            if limited_procedures:
                sample = limited_procedures[0]
                logger.info(
                    f"Sample procedure: {sample.get('name', 'Unknown')} from {sample.get('module', 'Unknown')}"
                )
                logger.info(
                    f"Has parameters: {len(sample.get('parameters', []))} params"
                )
                logger.info(f"Text length: {len(sample.get('full_text', ''))} chars")

            return limited_procedures

        except Exception as e:
            logger.error(f"Failed to extract procedures: {e}")
            return []

    def create_enhanced_prompt(self, procedure: Dict) -> str:
        """Create enhanced prompt using context from training loop."""
        # Use the same context enhancement as the training loop
        context = procedure.get("full_text", "")
        enhanced_context = self.training_loop.enhance_context_with_keywords(
            context, procedure.get("name", "")
        )

        prompt = f"""Analyze this IFS Cloud procedure and provide a concise business summary.

Procedure: {procedure.get('name', 'Unknown')}
Module: {procedure.get('module', 'Unknown').upper()}
File: {procedure.get('file_path', 'Unknown')}

Parameters:"""

        # Add parameters properly
        parameters = procedure.get("parameters", [])
        if parameters:
            for param in parameters[:10]:  # Limit to first 10 parameters
                param_name = param.get("name", "Unknown")
                param_type = param.get("type", "")
                param_mode = param.get("mode", "")
                prompt += f"\n  - {param_name} ({param_mode} {param_type})"
        else:
            prompt += "\n  - No parameters"

        prompt += f"""

Code Context:
{enhanced_context[:1500]}...

Provide a clear, concise business summary focusing on:
1. Primary business purpose and functionality
2. Key parameters and their business roles  
3. Integration points and business impact
4. When and why this procedure would be used

Business Summary:"""

        return prompt

    def generate_summary_optimized(self, procedure: Dict) -> str:
        """Generate summary with optimized GPU utilization."""
        try:
            prompt = self.create_enhanced_prompt(procedure)

            # Tokenize with proper attention handling
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=min(
                    self.config["max_length"], 3072
                ),  # Leave room for generation
                truncation=True,
                padding=False,  # No padding for single sequence
                return_attention_mask=True,
            )

            input_ids = encoding["input_ids"].cuda()
            attention_mask = encoding["attention_mask"].cuda()

            # Generate with optimized settings
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=400,  # Longer summaries
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract summary
            if "Business Summary:" in response:
                summary = response.split("Business Summary:")[-1].strip()
            else:
                summary = response[len(prompt) :].strip()

            # Clean up
            summary = summary.replace("\n\n", "\n").strip()

            # Ensure reasonable length
            if len(summary) > 1200:
                summary = summary[:1200] + "..."

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Generation failed: {str(e)}"

    def find_existing_progress(self) -> tuple[List[Dict], int]:
        """Find and load existing progress files."""
        progress_files = list(self.output_dir.glob("*progress*.json"))

        if not progress_files:
            logger.info("No existing progress found, starting fresh")
            return [], 0

        # Get the most recent progress file
        latest_progress = max(progress_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_progress, "r", encoding="utf-8") as f:
                data = json.load(f)

            existing_results = data.get("summaries", [])
            start_index = len(existing_results)

            logger.info(
                f"📂 Found existing progress: {start_index} summaries in {latest_progress.name}"
            )

            # Update output file to continue the series
            self.output_file = latest_progress.parent / latest_progress.name.replace(
                "_progress_", "_continued_"
            )

            return existing_results, start_index

        except Exception as e:
            logger.warning(f"Failed to load progress file {latest_progress}: {e}")
            return [], 0

    def run_optimized_generation(self):
        """Run optimized batch generation."""
        logger.info("🚀 Starting optimized batch generation...")

        # Load model with full GPU optimization
        self.load_model_full_gpu()

        # Extract procedures properly
        procedures = self.extract_procedures_properly()

        if not procedures:
            logger.error("No procedures found to process!")
            return

        # Check for existing progress
        existing_results, start_index = self.find_existing_progress()

        target_count = self.config.get("target_summaries", 200)
        remaining = target_count - start_index

        logger.info(
            f"🎯 Generating {remaining} remaining summaries (starting from #{start_index + 1})"
        )

        results = existing_results.copy()

        # Process remaining procedures
        for i in range(start_index, min(target_count, len(procedures))):
            try:
                procedure = procedures[i]

                logger.info(
                    f"Processing {i+1}/{target_count}: {procedure.get('name', 'Unknown')}"
                )

                # Generate summary
                summary = self.generate_summary_optimized(procedure)

                # Create result in proper format
                result = {
                    "id": i + 1,
                    "name": procedure.get("name", "Unknown"),
                    "module": procedure.get("module", "Unknown"),
                    "file_path": procedure.get("file_path", ""),
                    "full_text": procedure.get("full_text", ""),
                    "parameters": procedure.get("parameters", []),
                    "generated_summary": summary,
                    "status": "generated",
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)

                # Save progress frequently
                if (i + 1) % 5 == 0:  # Save every 5 summaries
                    self.save_results(results, f"_progress_{len(results)}")
                    logger.info(f"💾 Progress saved: {len(results)} summaries")

                # Clear cache less frequently since we're fully on GPU
                if (i + 1) % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"🧹 Cache cleared, GPU memory: {memory_used:.2f}GB")

            except Exception as e:
                logger.error(f"Failed to process procedure {i+1}: {e}")
                continue

        # Save final results
        self.save_results(results)
        logger.info(
            f"✅ Optimized generation complete! Created {len(results)} summaries"
        )
        logger.info(f"📁 Saved to: {self.output_file}")

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
                            "model_used": self.config["model_name"],
                            "optimization": "full_gpu_mode",
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


def main():
    """Main function."""
    print("🎯 Optimized IFS Cloud Batch Summary Generator")
    print("=" * 55)

    config = {
        "model_name": "unsloth/Qwen2.5-7B-Instruct",
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "max_length": 3500,  # Leave room for generation
        "target_summaries": 200,
    }

    print(f"📊 Optimization Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Mode: Full GPU (no CPU offloading)")
    print(f"  Target: {config['target_summaries']} summaries")
    print(f"  Context: {config['max_length']} tokens + 400 generation")
    print(f"  Source: {config['ifs_source_path']}")
    print()

    # Check source path
    if not Path(config["ifs_source_path"]).exists():
        print("⚠️  IFS source path not found!")
        return

    try:
        generator = OptimizedBatchGenerator(config)
        results = generator.run_optimized_generation()

        if results:
            print(
                f"🎉 Success! Generated {len(results)} summaries with full GPU optimization"
            )
            print(f"📁 File: {generator.output_file}")
            print()
            print("Next steps:")
            print("1. Review summaries with: python launch_gui_review.py")
            print("2. Start fine-tuning with reviewed summaries")

    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
