#!/usr/bin/env python3
"""
Direct Batch Generator - Bypass Parser Issues
Processes raw file content without complex parsing
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectBatchGenerator:
    """Direct batch generator that processes raw file content."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None

        # Load IFS Cloud keywords
        self.important_keywords = self.load_important_keywords()

        # Output directory
        self.output_dir = Path("batch_summaries")
        self.output_dir.mkdir(exist_ok=True)

        # Find existing summary files or create new one
        self.output_file = self.find_or_create_output_file()

    def find_or_create_output_file(self) -> Path:
        """Find the most recent summary file or create a new one."""
        # Look for existing summary files
        existing_files = list(self.output_dir.glob("*_summaries_*.json"))

        if existing_files:
            # Sort by modification time and get the most recent
            most_recent = max(existing_files, key=lambda p: p.stat().st_mtime)
            print(f"📂 Found existing summary file: {most_recent.name}")
            return most_recent
        else:
            # Create new timestamped file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_file = self.output_dir / f"direct_summaries_{timestamp}.json"
            print(f"📂 Creating new summary file: {new_file.name}")
            return new_file

    def load_important_keywords(self) -> List[Dict[str, Any]]:
        """Load IFS Cloud important keywords."""
        keywords_file = Path("final_optimizer_keywords.csv")
        keywords = []

        if keywords_file.exists():
            import csv

            with open(keywords_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keywords.append(
                        {
                            "keyword": row["keyword"],
                            "variants": (
                                row.get("variants", "").split(",")
                                if row.get("variants")
                                else []
                            ),
                        }
                    )
            logger.info(f"Loaded {len(keywords)} important keywords")

        return keywords

    def clear_gpu_memory(self):
        """Clear GPU memory once."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Print memory status
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"🧹 GPU Memory cleared - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
            )

    def load_model_smart_gpu(self):
        """Load model with smart GPU/CPU distribution."""
        logger.info(
            f"Loading model with smart memory management: {self.config['model_name']}"
        )

        # Clear GPU memory once at startup
        self.clear_gpu_memory()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

            # Ensure we have proper tokens
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            print(f"✅ Tokenizer loaded with pad_token: {self.tokenizer.pad_token}")

            # Load model with 4-bit quantization and auto device mapping
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",  # Let transformers decide optimal placement
                "trust_remote_code": True,
                "use_cache": True,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
            }

            print("🚀 Loading quantized 14B model with auto device mapping...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"], **model_kwargs
            )

            # Check where the model ended up
            device_info = {}
            for name, param in self.model.named_parameters():
                device = str(param.device)
                if device not in device_info:
                    device_info[device] = 0
                device_info[device] += param.numel()

            print("📊 Model distribution:")
            for device, param_count in device_info.items():
                print(f"  {device}: {param_count:,} parameters")

            # Print memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(
                    f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
                )

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def extract_content_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract content from a PLSQL file (simple approach)."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Simple extraction - take first 4000 characters
            if len(content) > 4000:
                content = (
                    content[:4000]
                    + "\n... [content truncated for summary generation] ..."
                )

            return {
                "name": Path(file_path).stem,  # Use filename as procedure name
                "content": content,
                "file_path": file_path,
                "line_start": 1,
                "line_end": len(content.split("\n")),
            }

        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None

    def enhance_context_with_keywords(self, context: str) -> str:
        """Enhance context with relevant IFS Cloud keywords."""
        if not self.important_keywords:
            return context

        # Find relevant keywords in content
        relevant_keywords = []
        context_lower = context.lower()

        for kw_info in self.important_keywords:
            keyword = kw_info["keyword"].lower()
            variants = [
                v.strip().lower() for v in kw_info.get("variants", []) if v.strip()
            ]

            # Check if keyword or variants appear in context
            if keyword in context_lower:
                relevant_keywords.append(kw_info["keyword"])
            else:
                for variant in variants:
                    if variant in context_lower:
                        relevant_keywords.append(kw_info["keyword"])
                        break

        # Add keyword enhancement if relevant keywords found
        if relevant_keywords:
            keyword_note = "\n\n=== IFS Cloud Keywords Found ===\n"
            keyword_note += ", ".join(relevant_keywords[:10])  # Limit to top 10
            keyword_note += "\nThese keywords indicate specific IFS Cloud business patterns and should be emphasized in the summary.\n"
            return context + keyword_note

        return context

    def generate_summary(self, content_info: Dict[str, Any]) -> Optional[str]:
        """Generate summary for file content."""
        try:
            # Prepare context
            content = content_info["content"]

            # Enhance with keywords
            enhanced_content = self.enhance_context_with_keywords(content)

            # Create prompt
            prompt = f"""You are an expert IFS Cloud developer analyzing PL/SQL database code.

**IFS Cloud file to analyze:**
```plsql
{enhanced_content}
```

**Task:** Create a comprehensive technical summary for this IFS Cloud database file.

**Required format:**
- **Purpose**: Brief description of what this file/module does
- **Key Features**: Main functionality and capabilities  
- **Database Objects**: Important tables, procedures, functions mentioned
- **Business Logic**: Core business rules and logic
- **Integration Points**: How it connects with other IFS components
- **Technical Notes**: Important implementation details

**Guidelines:**
- Focus on IFS Cloud business context and patterns
- Highlight important business logic and validation rules
- Explain main procedures and their purposes
- Note any security checks or authorization logic
- Keep technical but accessible to other developers
- Aim for 200-400 words

**Summary:**"""

            # Tokenize with larger context window (we have 7GB free VRAM)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=6000,
            )

            # Move inputs to model's primary device (usually GPU if available)
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Generate with optimized settings (no_grad context already active)
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=700,  # Increased for better summaries
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the summary part
            if "**Summary:**" in generated_text:
                summary = generated_text.split("**Summary:**")[-1].strip()
                return summary
            else:
                return generated_text[len(prompt) :].strip()

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None

    def find_plsql_files(self) -> List[str]:
        """Find all PLSQL files."""
        source_path = Path(self.config["ifs_source_path"])
        plsql_files = list(source_path.rglob("*.plsql"))
        logger.info(f"Found {len(plsql_files)} PLSQL files")
        return [str(f) for f in plsql_files]

    def run_batch_generation(self):
        """Run the batch generation process."""
        logger.info("🚀 Starting direct batch generation...")

        # Clear GPU memory once at startup
        self.clear_gpu_memory()

        # Load model
        self.load_model_smart_gpu()

        # Find files
        plsql_files = self.find_plsql_files()

        # Load existing results if any
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                completed_files = set(item["file_path"] for item in existing_data)
        else:
            existing_data = []
            completed_files = set()

        # Process files
        target_summaries = min(self.config["target_summaries"], len(plsql_files))
        processed_count = len(existing_data)

        print(f"\n🎯 Target: {target_summaries} summaries")
        print(f"📊 Already completed: {processed_count}")
        print(f"🔄 Remaining: {target_summaries - processed_count}")

        start_time = time.time()

        # Use no_grad context for entire batch processing
        with torch.no_grad():
            for file_path in tqdm(
                plsql_files[:target_summaries], desc="Processing files"
            ):
                if file_path in completed_files:
                    continue

                # Light cache clearing every 20 files to prevent accumulation
                if (
                    processed_count % 20 == 0
                    and processed_count > 0
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

                try:
                    # Extract content
                    content_info = self.extract_content_from_file(file_path)

                    if (
                        content_info and len(content_info["content"]) > 100
                    ):  # Skip tiny files

                        # Generate summary
                        proc_start_time = time.time()
                        summary = self.generate_summary(content_info)
                        proc_time = time.time() - proc_start_time

                        if summary:
                            # Save result
                            result = {
                                "file_path": file_path,
                                "file_name": content_info["name"],
                                "summary": summary,
                                "generation_time": proc_time,
                                "timestamp": datetime.now().isoformat(),
                                "content_length": len(content_info["content"]),
                            }

                            existing_data.append(result)
                            processed_count += 1

                            # Save incrementally
                            with open(self.output_file, "w", encoding="utf-8") as f:
                                json.dump(existing_data, f, indent=2)

                            # Print progress
                            total_time = time.time() - start_time
                            avg_time = (
                                total_time / processed_count
                                if processed_count > 0
                                else 0
                            )
                            remaining = target_summaries - processed_count
                            eta_minutes = (remaining * avg_time) / 60
                            tokens_per_sec = (
                                700 / proc_time if proc_time > 0 else 0
                            )  # Updated token rate

                            print(
                                f"\n✅ {processed_count}/{target_summaries} - {content_info['name']}"
                            )
                            print(
                                f"⏱️  This: {proc_time:.1f}s ({tokens_per_sec:.1f} t/s), Avg: {avg_time:.1f}s, ETA: {eta_minutes:.1f}min"
                            )

                            # Show GPU memory usage periodically
                            if processed_count % 5 == 0 and torch.cuda.is_available():
                                memory_allocated = (
                                    torch.cuda.memory_allocated() / 1024**3
                                )
                                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                                print(
                                    f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
                                )

                            if processed_count >= target_summaries:
                                break

                    if processed_count >= target_summaries:
                        break

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    # Light cache clearing after errors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        total_time = time.time() - start_time
        avg_tokens_per_sec = (
            (processed_count * 700) / total_time if total_time > 0 else 0
        )

        print(
            f"\n🎉 Completed {processed_count} summaries in {total_time/60:.1f} minutes"
        )
        print(f"🚀 Average performance: {avg_tokens_per_sec:.1f} tokens/second")
        print(f"💾 Results saved to: {self.output_file}")


def main():
    """Main entry point."""
    # Check optimizations
    try:
        print("✅ PyTorch SDPA available:", hasattr(torch.backends.cuda, "sdp_kernel"))
    except:
        print("⚠️ PyTorch SDPA check failed")

    try:
        import triton

        print(
            f"✅ Triton available: {triton.__version__} (for custom kernel optimizations)"
        )
    except ImportError:
        print("⚠️ Triton not available")

    try:
        import optimum

        print("✅ Optimum available (for inference optimization)")
    except ImportError:
        print("⚠️ Optimum not available")

    print("\n🎯 Direct IFS Cloud Batch Summary Generator")
    print("=" * 55)

    # Configuration
    config = {
        "model_name": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "target_summaries": 200,
        "max_length": 4096,
        "batch_size": 8,
    }

    print(f"📊 Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Mode: 4-bit quantized with smart GPU distribution")
    print(f"  Target: {config['target_summaries']} summaries")
    print(f"  Context: 6000 tokens + 700 generation (increased)")
    print(f"  Source: {config['ifs_source_path']}")
    print(f"  Memory: ~8GB VRAM (vs ~24GB for FP16 14B)")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 RTX optimizations enabled on {gpu_name}")
        print(f"💾 Available VRAM: {total_memory:.1f}GB")
    else:
        print("❌ CUDA not available")
        return

    # Run generation
    generator = DirectBatchGenerator(config)
    generator.run_batch_generation()


if __name__ == "__main__":
    main()
