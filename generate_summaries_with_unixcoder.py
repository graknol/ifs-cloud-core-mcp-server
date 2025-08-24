#!/usr/bin/env python3
"""
UnixCoder-powered Summary Generation for PL/SQL Training Samples

This script uses a pre-trained UnixCoder model to generate initial summaries
for the extracted samples, eliminating the need for Claude API calls.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
import logging
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnixCoderSummaryGenerator:
    """Generate summaries using pre-trained UnixCoder for code summarization."""

    def __init__(self, model_name: str = "microsoft/unixcoder-base"):
        """
        Initialize UnixCoder model for summarization.

        Args:
            model_name: HuggingFace model name for UnixCoder
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load UnixCoder model and tokenizer."""
        logger.info(f"🤖 Loading UnixCoder model: {self.model_name}")
        logger.info(f"🔧 Device: {self.device}")

        try:
            # UnixCoder uses RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

            # For summarization, we need the T5 variant or use the base model differently
            # UnixCoder base is encoder-only, so we'll use a code-to-text model instead
            # Let's use CodeT5+ which is better for this task
            model_name = "Salesforce/codet5p-220m"
            logger.info(
                f"🔄 Using CodeT5+ instead for better summarization: {model_name}"
            )

            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)

            logger.info("✅ Model loaded successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def create_input_text(self, sample: Dict) -> str:
        """Create structured input text for the model."""
        context = sample["context"]
        code = sample["code"]
        truncation_meta = context.get("truncation_metadata", {})

        # Create a structured prompt for code summarization
        input_parts = [
            f"# Summarize this PL/SQL function",
            f"# API: {context['api_name']} | Module: {context['module']}",
            f"# Function: {context['function_name']}",
            f"# Complexity: {context['complexity_metrics']['cyclomatic_complexity']}",
            "",
        ]

        # Add truncation notice if applicable
        if truncation_meta.get("truncation_method") != "no_truncation":
            input_parts.extend(
                [
                    f"# Note: Code truncated from {truncation_meta['original_length']} to {truncation_meta['truncated_length']} chars",
                    "",
                ]
            )

        # Add the code
        input_parts.append(code)

        return "\n".join(input_parts)

    def generate_summary(self, input_text: str, max_length: int = 128) -> str:
        """Generate summary using the model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=10,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    do_sample=False,
                )

            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up the summary
            summary = summary.strip()
            if summary.lower().startswith("summarize"):
                # Remove any echoed prompt
                lines = summary.split("\n")
                summary = lines[-1] if len(lines) > 1 else summary

            return summary

        except Exception as e:
            logger.warning(f"⚠️ Failed to generate summary: {e}")
            return "Summary generation failed - manual review needed"

    def process_samples(self, input_file: str, output_file: str = None) -> List[Dict]:
        """Process all samples and generate summaries."""
        if output_file is None:
            output_file = input_file.replace(
                ".jsonl", "_with_unixcoder_summaries.jsonl"
            )

        logger.info(f"📖 Loading samples from: {input_file}")

        # Load samples
        samples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        logger.info(f"🔄 Processing {len(samples)} samples...")

        processed_samples = []
        start_time = time.time()

        for i, sample in enumerate(samples, 1):
            logger.info(
                f"Processing sample {i}/{len(samples)}: {sample['context']['function_name']}"
            )

            # Create input for model
            input_text = self.create_input_text(sample)

            # Generate summary
            summary = self.generate_summary(input_text)

            # Update sample with generated summary
            sample["summary"] = summary
            sample["summary_method"] = "unixcoder_codet5p"
            sample["summary_timestamp"] = time.time()

            processed_samples.append(sample)

            # Progress update every 10 samples
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(samples) - i) * avg_time
                logger.info(
                    f"⏱️ Progress: {i}/{len(samples)} | Avg: {avg_time:.1f}s/sample | ETA: {remaining/60:.1f}min"
                )

        # Save processed samples
        logger.info(f"💾 Saving results to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in processed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        total_time = time.time() - start_time
        logger.info(f"✅ Processing complete! Total time: {total_time/60:.1f} minutes")
        logger.info(f"⚡ Average: {total_time/len(samples):.1f} seconds per sample")

        return processed_samples

    def validate_summaries(self, samples: List[Dict]) -> Dict:
        """Validate the quality of generated summaries."""
        logger.info("🔍 Validating summary quality...")

        stats = {
            "total_samples": len(samples),
            "successful_summaries": 0,
            "failed_summaries": 0,
            "average_summary_length": 0,
            "summary_lengths": [],
            "quality_issues": [],
        }

        for sample in samples:
            summary = sample.get("summary", "")

            if summary and not summary.startswith("Summary generation failed"):
                stats["successful_summaries"] += 1
                stats["summary_lengths"].append(len(summary))

                # Check for quality issues
                if len(summary) < 10:
                    stats["quality_issues"].append(f"Too short: {sample['id']}")
                elif len(summary) > 300:
                    stats["quality_issues"].append(f"Too long: {sample['id']}")
                elif summary.lower().count("function") > 3:
                    stats["quality_issues"].append(f"Generic: {sample['id']}")
            else:
                stats["failed_summaries"] += 1
                stats["quality_issues"].append(f"Failed: {sample['id']}")

        if stats["summary_lengths"]:
            stats["average_summary_length"] = sum(stats["summary_lengths"]) / len(
                stats["summary_lengths"]
            )

        return stats

    def print_validation_report(self, stats: Dict):
        """Print summary validation report."""
        print("\n📊 UNIXCODER SUMMARY VALIDATION REPORT")
        print("=" * 50)
        print(f"Total samples: {stats['total_samples']}")
        print(
            f"Successful summaries: {stats['successful_summaries']} ({stats['successful_summaries']/stats['total_samples']*100:.1f}%)"
        )
        print(
            f"Failed summaries: {stats['failed_summaries']} ({stats['failed_summaries']/stats['total_samples']*100:.1f}%)"
        )

        if stats["summary_lengths"]:
            print(
                f"Average summary length: {stats['average_summary_length']:.0f} characters"
            )
            print(
                f"Summary length range: {min(stats['summary_lengths'])} - {max(stats['summary_lengths'])} characters"
            )

        if stats["quality_issues"]:
            print(f"\n⚠️ Quality Issues Found: {len(stats['quality_issues'])}")
            for issue in stats["quality_issues"][:10]:  # Show first 10
                print(f"  - {issue}")
            if len(stats["quality_issues"]) > 10:
                print(f"  ... and {len(stats['quality_issues']) - 10} more")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate summaries using UnixCoder/CodeT5+"
    )
    parser.add_argument(
        "--input",
        default="training_samples_for_claude.jsonl",
        help="Input JSONL file with samples",
    )
    parser.add_argument(
        "--output", help="Output JSONL file (auto-generated if not provided)"
    )
    parser.add_argument(
        "--model",
        default="Salesforce/codet5p-220m",
        help="Model to use for summarization",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after generating summaries",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        return

    # Initialize generator
    generator = UnixCoderSummaryGenerator(args.model)

    # Process samples
    samples = generator.process_samples(args.input, args.output)

    # Validate if requested
    if args.validate:
        stats = generator.validate_summaries(samples)
        generator.print_validation_report(stats)

    print("\n🎯 UNIXCODER SUMMARIZATION COMPLETE!")
    print("✅ Your samples now have AI-generated summaries")
    print("🚀 Ready for fine-tuning or direct use!")


if __name__ == "__main__":
    main()
