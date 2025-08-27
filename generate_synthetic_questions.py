#!/usr/bin/env python3
"""
Generate synthetic questions and statements from the summaries.
Takes the raw responses from the summarization step and creates correlation datasets.
"""

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from copilot_api import CopilotAPI


class SyntheticQuestionGenerator:
    def __init__(
        self,
        raw_responses_dir="plsql_analysis_copilot/raw_responses",
        output_dir="synthetic_questions",
    ):
        self.raw_responses_dir = Path(raw_responses_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize Copilot API
        self.copilot = CopilotAPI()

        # Create output subdirectories
        (self.output_dir / "raw_responses").mkdir(exist_ok=True)
        (self.output_dir / "extracted_questions").mkdir(exist_ok=True)

    def get_raw_response_files(self):
        """Get all raw response markdown files from the summarization step"""
        if not self.raw_responses_dir.exists():
            print(f"❌ Raw responses directory not found: {self.raw_responses_dir}")
            return []

        files = list(self.raw_responses_dir.glob("*.md"))
        print(f"📁 Found {len(files)} raw response files")
        return sorted(files)

    def extract_procedures_from_response(self, response_content):
        """Extract procedure summaries from the raw response content"""
        procedures = []

        # Look for procedure blocks in the response
        # Pattern: ## <procedure_signature>\n<summary>
        pattern = r"## ([^#\n]+?)\n([^#]+?)(?=## |$)"
        matches = re.findall(pattern, response_content, re.DOTALL)

        for match in matches:
            proc_signature, summary = match

            # Extract procedure name from signature (everything before the first parenthesis)
            proc_name_match = re.match(r"(\w+)", proc_signature.strip())
            if proc_name_match:
                proc_name = proc_name_match.group(1)
            else:
                proc_name = proc_signature.strip()

            # Clean up the summary
            summary = re.sub(r"\s+", " ", summary.strip())

            # Skip if too short or if it's metadata
            if len(summary) < 30 or "Analysis Result" in summary:
                continue

            procedures.append(
                {
                    "procedure_name": proc_name,
                    "full_signature": proc_signature.strip(),
                    "summary": summary,
                }
            )

        return procedures

    def create_correlation_prompt(self, procedures):
        """Create a prompt to generate synthetic questions and statements"""
        prompt = f"""We are fine-tuning an embeddings model, and are building a dataset consisting of hypothetical, synthetic questions mapped to the provided summary.

For each of the {len(procedures)} procedures provided below, write 2 brief query variants that has the same semantic meaning as the summary; as if the user is looking for something. And write 2 statements (not present tense, but in infinitive) that are not questions (example: "Account Group Validation" -> "Validates that the group of the parameters passed to the Check_Insert___ procedure is a valid group")

IMPORTANT: Do not include the original procedure text in your response. Only provide the generated questions and statements.

Clearly separate each procedure with the format:

### [Procedure Name]
**Query 1:** [question variant 1]
**Query 2:** [question variant 2]
**Statement 1:** [infinitive statement 1]
**Statement 2:** [infinitive statement 2]

Here are the procedures:

"""

        for i, proc in enumerate(procedures, 1):
            prompt += f"## {i}. {proc['procedure_name']}: {proc['summary']}\n\n"

        prompt += "\nPlease generate the synthetic questions and statements following the format above."

        return prompt

    def generate_questions_for_file(self, response_file):
        """Generate synthetic questions for a single response file"""
        try:
            # Read the raw response
            with open(response_file, "r", encoding="utf-8") as f:
                response_content = f.read()

            # Extract procedures from the response
            procedures = self.extract_procedures_from_response(response_content)

            if not procedures:
                print(f"⚠️  No procedures found in {response_file.name}")
                return None

            print(f"📝 Processing {response_file.name} ({len(procedures)} procedures)")

            # Create the correlation prompt
            prompt = self.create_correlation_prompt(procedures)

            # Call Copilot API
            response = self.copilot.copilot_completion(prompt)

            if not response:
                print(f"❌ No response from Copilot for {response_file.name}")
                return None

            # Create analysis result
            analysis_result = {
                "source_file": str(response_file),
                "original_procedures": procedures,
                "correlation_prompt": prompt,
                "correlation_response": response,
                "timestamp": datetime.now().isoformat(),
                "procedure_count": len(procedures),
            }

            # Save raw response (only the AI response, no original procedures)
            output_filename = response_file.stem + "_questions.md"
            raw_output_file = self.output_dir / "raw_responses" / output_filename

            with open(raw_output_file, "w", encoding="utf-8") as f:
                f.write(f"# Synthetic Questions for {response_file.name}\n\n")
                f.write(f"**Source:** {response_file.name}\n")
                f.write(f"**Procedures:** {len(procedures)}\n")
                f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
                f.write("## Generated Questions and Statements\n\n")
                f.write(response)

            print(f"✅ Generated questions for {response_file.name}")
            return analysis_result

        except Exception as e:
            print(f"❌ Error processing {response_file.name}: {e}")
            return None

    def extract_questions_from_response(self, analysis_result):
        """Extract structured questions and statements from the response"""
        response_content = analysis_result["correlation_response"]
        extracted_data = []

        # Pattern to match the expected format
        # ### [Procedure Name]
        # **Query 1:** ...
        # **Query 2:** ...
        # **Statement 1:** ...
        # **Statement 2:** ...

        pattern = r"### (.+?)\n\*\*Query 1:\*\*\s*(.+?)\n\*\*Query 2:\*\*\s*(.+?)\n\*\*Statement 1:\*\*\s*(.+?)\n\*\*Statement 2:\*\*\s*(.+?)(?=\n### |$)"
        matches = re.findall(pattern, response_content, re.DOTALL)

        for match in matches:
            proc_name, query1, query2, stmt1, stmt2 = match

            # Clean up the extracted text
            proc_name = proc_name.strip()
            query1 = re.sub(r"\s+", " ", query1.strip())
            query2 = re.sub(r"\s+", " ", query2.strip())
            stmt1 = re.sub(r"\s+", " ", stmt1.strip())
            stmt2 = re.sub(r"\s+", " ", stmt2.strip())

            # Find the original summary
            original_summary = None
            for proc in analysis_result["original_procedures"]:
                if proc["procedure_name"] == proc_name:
                    original_summary = proc["summary"]
                    break

            extracted_data.append(
                {
                    "procedure_name": proc_name,
                    "original_summary": original_summary,
                    "synthetic_queries": [query1, query2],
                    "synthetic_statements": [stmt1, stmt2],
                    "source_file": analysis_result["source_file"],
                    "timestamp": analysis_result["timestamp"],
                }
            )

        return extracted_data

    def run_generation(self, max_files=None, delay_between_calls=2):
        """Run the complete synthetic question generation workflow"""
        print("🚀 Starting Synthetic Question Generation")
        print(f"📁 Reading from: {self.raw_responses_dir}")
        print(f"📁 Output directory: {self.output_dir}")

        # Get all response files
        response_files = self.get_raw_response_files()

        if not response_files:
            print("❌ No raw response files found")
            return

        # Limit files if specified
        if max_files:
            response_files = response_files[:max_files]
            print(f"🔢 Limited to {max_files} files")

        print(f"📊 Processing {len(response_files)} response files")

        # Initialize Copilot token
        print("🔑 Initializing GitHub Copilot API...")
        self.copilot.get_token()

        all_analyses = []
        all_extracted_questions = []

        # Process each file
        for i, response_file in enumerate(response_files, 1):
            print(f"\n[{i}/{len(response_files)}] ", end="")

            # Generate questions for this file
            analysis = self.generate_questions_for_file(response_file)

            if analysis:
                all_analyses.append(analysis)

                # Extract structured questions
                extracted = self.extract_questions_from_response(analysis)
                all_extracted_questions.extend(extracted)

                print(f"   └── Extracted {len(extracted)} question sets")

            # Delay between API calls
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

        # Save results
        print(f"\n💾 Saving results...")

        # Save all analyses
        analyses_file = self.output_dir / "all_question_analyses.json"
        with open(analyses_file, "w", encoding="utf-8") as f:
            json.dump(all_analyses, f, ensure_ascii=False, indent=2)

        # Save extracted questions as JSONL
        questions_file = self.output_dir / "synthetic_questions.jsonl"
        with open(questions_file, "w", encoding="utf-8") as f:
            for item in all_extracted_questions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Print summary
        print(f"\n📈 Generation Complete!")
        print(f"   • Processed {len(all_analyses)} files")
        print(f"   • Generated {len(all_extracted_questions)} question sets")
        print(f"   • Total synthetic queries: {len(all_extracted_questions) * 2}")
        print(f"   • Total synthetic statements: {len(all_extracted_questions) * 2}")
        print(f"   • Results saved to: {self.output_dir}")

        return all_extracted_questions


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic questions from PLSQL procedure summaries"
    )
    parser.add_argument(
        "--max-files", type=int, help="Maximum number of files to process"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--raw-responses-dir",
        default="plsql_analysis_copilot/raw_responses",
        help="Directory with raw response files",
    )
    parser.add_argument(
        "--output-dir", default="synthetic_questions", help="Output directory"
    )

    args = parser.parse_args()

    generator = SyntheticQuestionGenerator(
        raw_responses_dir=args.raw_responses_dir, output_dir=args.output_dir
    )

    generator.run_generation(max_files=args.max_files, delay_between_calls=args.delay)


if __name__ == "__main__":
    main()
