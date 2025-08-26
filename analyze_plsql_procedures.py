#!/usr/bin/env python3
"""
Systematic PLSQL Procedure Analysis using GitHub Copilot API
This script analyzes the top PLSQL files from each module and extracts procedure summaries.
"""

import os
import json
import time
import glob
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime

from copilot_api import CopilotAPI


class PLSQLProcedureAnalyzer:
    def __init__(self, top_10_dir="top_10", output_dir="plsql_analysis"):
        self.top_10_dir = Path(top_10_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize Copilot API
        self.copilot = CopilotAPI()

        # Create subdirectories
        (self.output_dir / "raw_responses").mkdir(exist_ok=True)
        (self.output_dir / "extracted_procedures").mkdir(exist_ok=True)

        # Analysis prompt template
        self.analysis_prompt_template = """
Analyze this PL/SQL file and extract procedure/function definitions with summaries.

OBJECTIVE: Extract key procedures and functions for embedding fine-tuning.

TASK:
1. Identify actual PROCEDURE and FUNCTION declarations (not just calls)
2. Focus on the most important business logic procedures (max 25)
3. Provide conceptual summaries, not implementation details

FORMAT REQUIRED:
## PROCEDURE_NAME(param1 IN TYPE, param2 OUT TYPE)
A 2-3 sentence summary explaining the business purpose and what it conceptually accomplishes.

## FUNCTION_NAME(param IN TYPE) RETURN TYPE
A 2-3 sentence summary explaining what this function logically does and returns.

GUIDELINES:
- Skip simple getters/setters unless they're critical
- Focus on business logic and complex operations  
- Include parameter types when visible
- Emphasize conceptual understanding over code details
- Prioritize diversity in business domains
- Start each summary with a verb (e.g. "Validates", "Calculates", "Processes") instead of "This procedure..."

PL/SQL FILE CONTENT:
{file_content}

EXTRACT PROCEDURES/FUNCTIONS NOW:
"""

    def get_module_files(self):
        """Get all PLSQL files organized by module"""
        module_files = defaultdict(list)

        if not self.top_10_dir.exists():
            print(f"❌ Directory {self.top_10_dir} does not exist")
            return module_files

        for module_dir in self.top_10_dir.iterdir():
            if module_dir.is_dir():
                module_name = module_dir.name
                plsql_files = list(module_dir.glob("*.plsql"))
                if plsql_files:
                    module_files[module_name] = plsql_files

        return module_files

    def read_file_safely(self, file_path):
        """Read file with proper encoding handling"""
        encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all encodings fail, read as binary and decode with errors='ignore'
        try:
            with open(file_path, "rb") as f:
                content = f.read().decode("utf-8", errors="ignore")
                return content
        except Exception as e:
            print(f"❌ Could not read {file_path}: {e}")
            return None

    def truncate_file_if_needed(self, content, max_chars=300000):
        """Truncate file content if it's too large for the API"""
        if len(content) <= max_chars:
            return content, False

        # Try to truncate at a reasonable boundary (end of a procedure/function)
        truncated = content[:max_chars]

        # Find the last complete procedure/function
        last_procedure = max(
            truncated.rfind("\nPROCEDURE "),
            truncated.rfind("\nFUNCTION "),
            truncated.rfind("\n--"),
        )

        if last_procedure > max_chars // 2:  # If we found a reasonable boundary
            truncated = truncated[:last_procedure]

        return truncated, True

    def analyze_file_with_copilot(self, file_path, module_name):
        """Analyze a single PLSQL file using Copilot API"""

        # Check if this file has already been analyzed
        response_file = (
            self.output_dir / "raw_responses" / f"{module_name}_{file_path.stem}.md"
        )

        if response_file.exists():
            print(f"⏭️  Skipping {module_name}/{file_path.name} (already analyzed)")
            return None

        print(f"📝 Analyzing {module_name}/{file_path.name}")

        # Read file content
        content = self.read_file_safely(file_path)
        if content is None:
            return None

        # Truncate if needed
        content, was_truncated = self.truncate_file_if_needed(content)
        if was_truncated:
            print(f"⚠️  File truncated due to size ({len(content)} chars)")

        # Create the analysis prompt
        prompt = self.analysis_prompt_template.format(file_content=content)

        try:
            # Call Copilot API
            response = self.copilot.copilot_completion(
                prompt, max_tokens=32000, temperature=0.3
            )

            if response:
                # Save raw response
                response_file = (
                    self.output_dir
                    / "raw_responses"
                    / f"{module_name}_{file_path.stem}.md"
                )
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(f"# Analysis of {module_name}/{file_path.name}\n\n")
                    f.write(f"File: {file_path}\n")
                    f.write(f"Module: {module_name}\n")
                    f.write(f"Analyzed: {datetime.now().isoformat()}\n")
                    f.write(f"Truncated: {was_truncated}\n")
                    f.write(f"Content Length: {len(content)} chars\n\n")
                    f.write("## Analysis Result\n\n")
                    f.write(response)

                print(f"✅ Analysis completed for {file_path.name}")
                return {
                    "module": module_name,
                    "file": file_path.name,
                    "file_path": str(file_path),
                    "response": response,
                    "truncated": was_truncated,
                    "content_length": len(content),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                print(f"❌ No response received for {file_path.name}")
                return None

        except Exception as e:
            print(f"❌ Error analyzing {file_path.name}: {e}")
            return None

    def extract_procedures_from_response(self, analysis_result):
        """Extract individual procedures and their summaries from the API response"""
        if not analysis_result:
            return []

        response = analysis_result["response"]
        procedures = []

        # More flexible regex to catch procedure headers
        # Matches: ## PROCEDURE_NAME, ## FUNCTION_NAME, etc.
        sections = re.split(
            r"\n##\s*([A-Z][A-Z0-9_]*[^\n]*)",
            response,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Process sections (skip the first one which is before any procedure)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                proc_signature = sections[i].strip()
                proc_summary = sections[i + 1].strip()

                # Skip empty or very short summaries
                if len(proc_summary) < 20:
                    continue

                # Extract procedure name (everything before first parenthesis or space)
                proc_name_match = re.match(
                    r"([A-Z][A-Z0-9_]*)", proc_signature, re.IGNORECASE
                )
                if proc_name_match:
                    proc_name = proc_name_match.group(1)
                else:
                    proc_name = (
                        proc_signature.split("(")[0].split()[0]
                        if proc_signature
                        else "Unknown"
                    )

                # Extract parameters if present
                if "(" in proc_signature:
                    start_paren = proc_signature.find("(")
                    end_paren = proc_signature.rfind(")")
                    if end_paren > start_paren:
                        parameters = proc_signature[start_paren : end_paren + 1]
                    else:
                        parameters = proc_signature[start_paren:] + ")"
                else:
                    parameters = "()"

                # Clean up the summary
                # Remove extra formatting, newlines, and common artifacts
                clean_summary = re.sub(r"\n+", " ", proc_summary)
                clean_summary = re.sub(r"\s+", " ", clean_summary)
                clean_summary = clean_summary.strip()

                # Skip very short or generic summaries
                if len(clean_summary) < 30 or "Summary text here" in clean_summary:
                    continue

                procedures.append(
                    {
                        "procedure_name": proc_name,
                        "parameters": parameters,
                        "signature": proc_signature,
                        "summary": clean_summary,
                        "module": analysis_result["module"],
                        "file": analysis_result["file"],
                        "file_path": analysis_result["file_path"],
                        "timestamp": analysis_result["timestamp"],
                    }
                )

        return procedures

    def save_procedures_jsonl(
        self, all_procedures, filename="procedures_analysis.jsonl"
    ):
        """Save all extracted procedures to a JSONL file"""
        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            for procedure in all_procedures:
                json_line = json.dumps(procedure, ensure_ascii=False)
                f.write(json_line + "\n")

        print(f"💾 Saved {len(all_procedures)} procedures to {output_file}")

    def run_analysis(self, max_files_per_module=20, delay_between_calls=2):
        """Run the complete analysis workflow"""
        print("🚀 Starting PLSQL Procedure Analysis")
        print(f"📁 Analyzing files from: {self.top_10_dir}")
        print(f"📁 Output directory: {self.output_dir}")

        # Get all module files
        module_files = self.get_module_files()

        if not module_files:
            print("❌ No PLSQL files found in the top_10 directory")
            return

        print(f"📊 Found {len(module_files)} modules with PLSQL files")

        # Initialize Copilot token
        print("🔑 Initializing GitHub Copilot API...")
        self.copilot.get_token()

        all_analyses = []
        all_procedures = []

        total_files = sum(
            min(len(files), max_files_per_module) for files in module_files.values()
        )
        processed_files = 0

        # Process each module
        for module_name, files in module_files.items():
            print(f"\n📂 Processing module: {module_name} ({len(files)} files)")

            # Limit files per module
            files_to_process = files[:max_files_per_module]

            for file_path in files_to_process:
                processed_files += 1
                print(f"[{processed_files}/{total_files}] ", end="")

                # Analyze the file
                analysis = self.analyze_file_with_copilot(file_path, module_name)

                if analysis:
                    all_analyses.append(analysis)

                    # Extract procedures from the analysis
                    procedures = self.extract_procedures_from_response(analysis)
                    all_procedures.extend(procedures)

                    print(f"   └── Extracted {len(procedures)} procedures")

                # Delay between API calls to avoid rate limiting
                if delay_between_calls > 0:
                    time.sleep(delay_between_calls)

        # Save results
        print(f"\n💾 Saving results...")

        # Save all analyses
        analyses_file = self.output_dir / "all_analyses.json"
        with open(analyses_file, "w", encoding="utf-8") as f:
            json.dump(all_analyses, f, ensure_ascii=False, indent=2)

        # Save procedures as JSONL
        self.save_procedures_jsonl(all_procedures)

        # Print summary
        print(f"\n📈 Analysis Complete!")
        print(f"   • Processed {len(all_analyses)} files")
        print(f"   • Extracted {len(all_procedures)} procedures")
        print(f"   • Results saved to: {self.output_dir}")

        # Print per-module breakdown
        module_counts = defaultdict(int)
        for proc in all_procedures:
            module_counts[proc["module"]] += 1

        print(f"\n📊 Procedures per module:")
        for module, count in sorted(module_counts.items()):
            print(f"   • {module}: {count} procedures")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze PLSQL files using GitHub Copilot API"
    )
    parser.add_argument(
        "--top-10-dir",
        default="top_10",
        help="Directory containing top_10 module files",
    )
    parser.add_argument(
        "--output-dir", default="plsql_analysis", help="Output directory for results"
    )
    parser.add_argument(
        "--max-files", type=int, default=20, help="Maximum files to process per module"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Delay between API calls (seconds)"
    )

    args = parser.parse_args()

    # Create and run analyzer
    analyzer = PLSQLProcedureAnalyzer(
        top_10_dir=args.top_10_dir, output_dir=args.output_dir
    )

    analyzer.run_analysis(
        max_files_per_module=args.max_files, delay_between_calls=args.delay
    )
