#!/usr/bin/env python3
"""
Claude-powered Summary Generation for PL/SQL Training Samples

This script processes the extracted samples and generates high-quality
summaries using Claude Sonnet for fine-tuning a local model.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaudeSummaryGenerator:
    """Generate summaries using Claude for training data."""

    def __init__(self):
        # You would need to add your Claude API client here
        # For now, this shows the structure and prompts
        pass

    def create_summary_prompt(self, sample: Dict) -> str:
        """Create a comprehensive prompt for Claude to generate summaries."""

        context = sample["context"]
        code = sample["code"]
        truncation_meta = context.get("truncation_metadata", {})

        # Add truncation notice if applicable
        truncation_notice = ""
        if truncation_meta.get("truncation_method") != "no_truncation":
            truncation_notice = f"""
## ⚠️ Code Truncation Notice:
This function was intelligently truncated for UnixCoder compatibility:
- Original length: {truncation_meta['original_length']} characters
- Truncated to: {truncation_meta['truncated_length']} characters  
- Truncation ratio: {truncation_meta['truncation_ratio']:.2f}
- Method: {truncation_meta['truncation_method']}

Please focus on the key business logic and purpose shown in this truncated view.
"""

        prompt = f"""
# PL/SQL Function Summarization Task

You are an expert PL/SQL developer tasked with writing concise, accurate summaries of database functions and procedures for **UnixCoder model fine-tuning**.

## Context Information:
- **API/Package**: {context['api_name']}
- **Module**: {context['module']}
- **File Purpose**: {context['file_summary']}
- **Function Name**: {context['function_name']}
- **Previous Function**: {context.get('previous_function', 'N/A')}
- **Next Function**: {context.get('next_function', 'N/A')}
- **Complexity**: {context['complexity_metrics']['cyclomatic_complexity']} (cyclomatic)
- **Code Lines**: {context['complexity_metrics']['code_lines']}
- **PageRank Score**: {context['pagerank_score']:.4f}{truncation_notice}

## Function Code:
```plsql
{code}
```

## Summary Requirements:
Write a **concise, technical summary** (1-3 sentences) that:

1. **States the main purpose** using active verbs
2. **Identifies key operations** (validations, calculations, data processing)
3. **Notes important side effects** (updates, exceptions, state changes)
4. **Uses proper PL/SQL terminology**

**Style Guidelines:**
- Start with action verbs: "Validates", "Calculates", "Updates", "Processes"
- Be specific about business operations
- Mention key data entities and relationships
- Note exception handling if significant
- Keep it under 150 words total

**Example Style:**
"Validates customer order line items against inventory availability and business rules, updating order status to 'Confirmed' and generating exception records for unfulfillable items with detailed error codes."

## Your Summary:

## Instructions:
Generate a concise, technical summary (1-3 sentences) that:
1. **States the main purpose** - What does this function DO?
2. **Identifies key operations** - Main business logic, validations, calculations
3. **Notes important side effects** - Database updates, exceptions, external calls
4. **Uses technical terminology** - Appropriate PL/SQL and business terms

## Style Guidelines:
- Start with an active verb (e.g., "Validates", "Calculates", "Updates", "Retrieves")
- Be specific about data being processed
- Mention key business rules or constraints
- Keep it concise but informative
- Avoid generic phrases like "This function..."

## Example Output Format:
"Validates customer order line items against inventory availability and business rules, updating order status and generating exception records for items that cannot be fulfilled within the specified delivery timeframe."

## Your Summary:
"""
        return prompt

    async def generate_summary(self, sample: Dict) -> str:
        """Generate summary for a single sample using Claude."""
        prompt = self.create_summary_prompt(sample)

        # This is where you would call Claude API
        # For demonstration, I'll show the structure:
        """
        response = await claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
        """

        # Placeholder - you'll need to implement actual Claude API call
        return "PLACEHOLDER_SUMMARY - Replace with actual Claude API call"

    async def process_samples_batch(
        self, samples: List[Dict], batch_size: int = 5
    ) -> List[Dict]:
        """Process samples in batches to avoid rate limits."""

        completed_samples = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(samples)-1)//batch_size + 1}"
            )

            # Process batch
            tasks = [self.generate_summary(sample) for sample in batch]
            summaries = await asyncio.gather(*tasks)

            # Add summaries to samples
            for sample, summary in zip(batch, summaries):
                sample["summary"] = summary
                completed_samples.append(sample)

            # Small delay to respect rate limits
            await asyncio.sleep(1)

        return completed_samples


def validate_training_data(samples: List[Dict]) -> Dict:
    """Validate the quality of generated training data."""

    stats = {
        "total_samples": len(samples),
        "valid_summaries": 0,
        "avg_summary_length": 0,
        "complexity_distribution": {},
        "module_coverage": set(),
    }

    summary_lengths = []
    complexities = []

    for sample in samples:
        if (
            sample.get("summary")
            and sample["summary"]
            != "PLACEHOLDER_SUMMARY - Replace with actual Claude API call"
        ):
            stats["valid_summaries"] += 1
            summary_lengths.append(len(sample["summary"].split()))

        complexity = sample["context"]["complexity_metrics"]["cyclomatic_complexity"]
        complexities.append(complexity)
        stats["module_coverage"].add(sample["context"]["module"])

    if summary_lengths:
        stats["avg_summary_length"] = sum(summary_lengths) / len(summary_lengths)

    # Complexity distribution
    for complexity in complexities:
        bucket = f"{(complexity//5)*5}-{(complexity//5)*5+4}"
        stats["complexity_distribution"][bucket] = (
            stats["complexity_distribution"].get(bucket, 0) + 1
        )

    stats["module_coverage"] = len(stats["module_coverage"])

    return stats


def create_fine_tuning_dataset(samples: List[Dict]) -> List[Dict]:
    """Convert samples to fine-tuning format (e.g., for Code T5+)."""

    fine_tuning_data = []

    for sample in samples:
        if not sample.get("summary"):
            continue

        context = sample["context"]

        # Create input text with rich context
        input_text = f"""
# API: {context['api_name']} | Module: {context['module']}
# Function: {context['function_name']} | Complexity: {context['complexity_metrics']['cyclomatic_complexity']}
# Context: {context['file_summary']}

{sample['code']}
""".strip()

        # Target summary
        target_text = sample["summary"]

        fine_tuning_data.append(
            {
                "input": input_text,
                "target": target_text,
                "metadata": {
                    "function_name": context["function_name"],
                    "api_name": context["api_name"],
                    "module": context["module"],
                    "complexity": context["complexity_metrics"][
                        "cyclomatic_complexity"
                    ],
                    "pagerank_score": context["pagerank_score"],
                },
            }
        )

    return fine_tuning_data


async def main():
    """Main processing pipeline."""

    # Load samples generated by extract_training_samples.py
    input_file = Path("training_samples_for_claude.jsonl")

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Run extract_training_samples.py first to generate samples")
        return

    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(samples)} samples for processing")

    # Generate summaries using Claude
    generator = ClaudeSummaryGenerator()

    # NOTE: You need to implement the actual Claude API calls
    # For now, this creates the structure
    logger.warning("⚠️  Claude API integration needed - currently using placeholders")

    # completed_samples = await generator.process_samples_batch(samples)
    completed_samples = samples  # Placeholder

    # Save completed samples
    output_file = Path("summarized_training_samples.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in completed_samples:
            f.write(json.dumps(sample, indent=None) + "\n")

    # Create fine-tuning dataset
    fine_tuning_data = create_fine_tuning_dataset(completed_samples)

    fine_tuning_file = Path("code_summarization_dataset.jsonl")
    with open(fine_tuning_file, "w", encoding="utf-8") as f:
        for item in fine_tuning_data:
            f.write(json.dumps(item, indent=None) + "\n")

    # Validate and report
    stats = validate_training_data(completed_samples)

    print(
        f"""
📊 Training Data Generation Complete:

✅ Files Generated:
  - {output_file} ({len(completed_samples)} samples)
  - {fine_tuning_file} ({len(fine_tuning_data)} training pairs)

📈 Quality Metrics:
  - Valid summaries: {stats['valid_summaries']}/{stats['total_samples']}
  - Avg summary length: {stats['avg_summary_length']:.1f} words
  - Module coverage: {stats['module_coverage']} modules
  - Complexity distribution: {stats['complexity_distribution']}

🚀 Next Steps:
  1. Implement Claude API integration in ClaudeSummaryGenerator
  2. Review generated summaries for quality
  3. Use code_summarization_dataset.jsonl for Code T5+ fine-tuning
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
