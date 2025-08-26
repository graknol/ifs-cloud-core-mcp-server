#!/usr/bin/env python3
"""
Correlate Cleaned Summaries with Source Code and Regenerate Prompts
==================================================================

This script takes the cleaned summary file and correlates each entry with its source code,
extracts the procedure, and generates the complete prompt that would be used for generation.
This creates a file suitable for UI review.
"""

import json
import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaryCorrelator:
    """Correlate summaries with source code and regenerate prompts."""

    def __init__(self):
        self.source_root = Path("C:/repos/_ifs/25.1.0")
        self.important_keywords = self.load_important_keywords()

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count using a simple tokenization approach.

        This provides a rough estimate based on:
        - Word boundaries (spaces, punctuation)
        - Common programming tokens (operators, keywords)
        - Typical GPT-style tokenization patterns

        Note: This is an approximation. Actual token counts may vary by ~10-20%.
        """
        if not text:
            return 0

        # Basic tokenization patterns
        # Split on whitespace and common punctuation, but keep some programming constructs together
        tokens = []

        # Replace common programming patterns with single tokens
        text = re.sub(r"\b\d+\.\d+\b", " <NUMBER> ", text)  # Decimals
        text = re.sub(r"\b\d+\b", " <NUMBER> ", text)  # Integers
        text = re.sub(r'["\'][^"\']*["\']', " <STRING> ", text)  # String literals
        text = re.sub(r"--.*$", " <COMMENT> ", text, flags=re.MULTILINE)  # SQL comments
        text = re.sub(
            r"/\*.*?\*/", " <COMMENT> ", text, flags=re.DOTALL
        )  # Block comments

        # Split on whitespace and punctuation, but preserve some structure
        parts = re.findall(r"\w+|[^\w\s]", text)

        # Estimate tokens - some heuristics:
        # - Short words (1-3 chars) typically = 1 token
        # - Medium words (4-8 chars) typically = 1-2 tokens
        # - Long words (9+ chars) typically = 2-3 tokens
        # - Punctuation typically = 1 token each

        estimated_tokens = 0
        for part in parts:
            if part.isalnum():
                if len(part) <= 3:
                    estimated_tokens += 1
                elif len(part) <= 8:
                    estimated_tokens += max(1, len(part) // 4)
                else:
                    estimated_tokens += max(2, len(part) // 3)
            else:
                estimated_tokens += 1

        # Add some padding for subword tokenization (common in modern tokenizers)
        estimated_tokens = int(estimated_tokens * 1.2)

        return estimated_tokens

    def load_important_keywords(self) -> List[str]:
        """Load important IFS Cloud keywords for context enhancement."""
        try:
            with open("comprehensive_plsql_analysis.json", "r", encoding="utf-8") as f:
                analysis = json.load(f)
                return analysis.get("important_keywords", [])
        except Exception as e:
            logger.warning(f"Could not load keywords: {e}")
            return []

    def extract_plsql_procedure(
        self, file_path: Path, line_start: int, line_end: int, procedure_name: str
    ) -> Optional[str]:
        """Extract a specific procedure from a PL/SQL file using proper IFS parser integration."""
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None

            # Read the full file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_content = f.read()

            # Create a minimal parser integration instance
            from ifs_parser_integration import IFSCloudParserIntegration

            parser_integration = IFSCloudParserIntegration()

            # Parse the file using the IFS parser
            parsed_result = parser_integration.parse_code(full_content)

            # Try to extract the procedure using the parsed result
            procedure_body = self._extract_procedure_from_parsed(
                full_content, procedure_name, parsed_result
            )

            if procedure_body:
                logger.info(
                    f"✅ Successfully extracted procedure {procedure_name} using IFS parser"
                )
                return procedure_body
            else:
                # Fallback to the original method if parser fails
                logger.warning(
                    f"Parser extraction failed for {procedure_name}, falling back to line-based extraction"
                )
                return self._extract_plsql_procedure_fallback(
                    file_path, line_start, line_end, procedure_name
                )

        except Exception as e:
            logger.error(
                f"Error extracting procedure {procedure_name} from {file_path}: {e}"
            )
            # Fallback to the original method
            return self._extract_plsql_procedure_fallback(
                file_path, line_start, line_end, procedure_name
            )

    def _extract_procedure_from_parsed(
        self, content: str, procedure_name: str, parsed_result: dict
    ) -> Optional[str]:
        """Extract procedure using the parsed AST information."""
        try:
            lines = content.split("\n")

            # Find the procedure declaration in the content
            for i, line in enumerate(lines):
                line_upper = line.upper().strip()
                if f"PROCEDURE {procedure_name.upper()}" in line_upper:
                    start_line = i

                    # Find the matching END statement
                    procedure_depth = 0
                    found_procedure_start = False

                    for j in range(start_line, len(lines)):
                        current_line = lines[j].upper().strip()

                        # Track if we've found the procedure start
                        if (
                            not found_procedure_start
                            and f"PROCEDURE {procedure_name.upper()}" in current_line
                        ):
                            found_procedure_start = True
                            procedure_depth = 1
                            continue

                        if found_procedure_start:
                            # Look for nested procedures/functions
                            if any(
                                keyword in current_line
                                for keyword in ["PROCEDURE ", "FUNCTION "]
                            ):
                                procedure_depth += 1
                            # Look for END statements
                            elif current_line.startswith("END"):
                                if f"END {procedure_name.upper()}" in current_line:
                                    # Found the exact matching END statement
                                    return "\n".join(lines[start_line : j + 1])
                                elif current_line == "END;":
                                    # Generic END statement - only match if depth is 1
                                    procedure_depth -= 1
                                    if procedure_depth == 0:
                                        return "\n".join(lines[start_line : j + 1])

                    # If we didn't find a proper end, return reasonable default
                    end_line = min(start_line + 50, len(lines))
                    return "\n".join(lines[start_line:end_line])

            return None

        except Exception as e:
            logger.error(f"Error extracting from parsed result: {e}")
            return None

    def _extract_plsql_procedure_fallback(
        self, file_path: Path, line_start: int, line_end: int, procedure_name: str
    ) -> Optional[str]:
        """Fallback extraction method using line numbers."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if line_start < 1:
                line_start = 1

            # Convert to 0-based indexing
            start_idx = line_start - 1

            if start_idx >= len(lines):
                logger.warning(
                    f"Start line {line_start} exceeds file length {len(lines)} in {file_path}"
                )
                return None

            # Find the actual procedure boundaries using smarter detection
            actual_end_idx = self._find_procedure_end(lines, start_idx, procedure_name)

            if actual_end_idx is None:
                # Fallback to original line_end if provided and reasonable
                if line_end <= len(lines):
                    actual_end_idx = line_end - 1
                else:
                    # Use a reasonable default (50 lines max for a single procedure)
                    actual_end_idx = min(start_idx + 50, len(lines) - 1)

            # Extract the procedure lines
            procedure_lines = lines[start_idx : actual_end_idx + 1]
            procedure_text = "".join(procedure_lines)

            return procedure_text.strip()

        except Exception as e:
            logger.error(
                f"Error in fallback extraction for {procedure_name} from {file_path}: {e}"
            )
            return None

    def _find_procedure_end(
        self, lines: List[str], start_idx: int, procedure_name: str
    ) -> Optional[int]:
        """Find the actual end of a procedure using smart boundary detection."""
        try:
            procedure_name_upper = procedure_name.upper()

            # Look for the procedure declaration first to understand the structure
            found_begin = False
            paren_count = 0
            in_parameters = False

            # Start searching from the procedure declaration
            for i in range(start_idx, min(start_idx + 100, len(lines))):
                line = lines[i].strip()
                line_upper = line.upper()

                # Skip empty lines and comments
                if not line or line_upper.startswith("--"):
                    continue

                # Track parentheses for parameter detection
                if not found_begin:
                    if "(" in line:
                        in_parameters = True
                        paren_count += line.count("(") - line.count(")")
                    elif in_parameters:
                        paren_count += line.count("(") - line.count(")")

                    # Found the BEGIN keyword
                    if "BEGIN" in line_upper and not in_parameters and paren_count <= 0:
                        found_begin = True
                        continue

                # After finding BEGIN, look for END
                if found_begin:
                    # Look for END statement that matches this procedure
                    if line_upper.startswith("END "):
                        # Check if it's the end of our specific procedure
                        if procedure_name_upper in line_upper:
                            return i
                        # Also accept END; as a generic end
                        elif line_upper.strip() == "END;" and i < start_idx + 20:
                            # Only accept generic END if it's close to start (single procedure)
                            return i
                    elif line_upper.strip() == "END;":
                        # Generic END statement - accept if close to procedure start
                        if i < start_idx + 20:
                            return i
                    elif (
                        line_upper.startswith("PROCEDURE ")
                        or line_upper.startswith("FUNCTION ")
                    ) and i > start_idx + 5:
                        # Found start of next procedure/function, so previous line is end
                        return i - 1

            # If we didn't find a proper end, return a reasonable default
            return min(
                start_idx + 30, len(lines) - 1
            )  # Reduced from 50 to 30 lines max

        except Exception as e:
            logger.debug(f"Error finding procedure end for {procedure_name}: {e}")
            return None

    def enhance_context_with_keywords(self, context: str) -> str:
        """Add keyword context note if important keywords are found."""
        if not self.important_keywords:
            return context

        # Find keywords that appear in the context
        found_keywords = []
        context_lower = context.lower()

        for keyword in self.important_keywords:
            if keyword.lower() in context_lower:
                found_keywords.append(keyword)

        if found_keywords:
            keyword_note = f"\n\n**Important IFS Cloud Keywords Found**: {', '.join(found_keywords[:10])}"  # Limit to first 10
            if len(found_keywords) > 10:
                keyword_note += f" (and {len(found_keywords) - 10} more)"
            return context + keyword_note

        return context

    def generate_prompt_for_procedure(
        self, summary_entry: Dict[str, Any]
    ) -> Optional[str]:
        """Generate the complete prompt for a procedure based on its summary entry."""
        try:
            # Convert file path to Path object and make it relative to source root
            file_path_str = summary_entry["file_path"]

            # Handle the path conversion
            if file_path_str.startswith("C:\\repos\\_ifs\\25.1.0\\"):
                relative_path = file_path_str.replace("C:\\repos\\_ifs\\25.1.0\\", "")
                full_path = self.source_root / relative_path
            else:
                # Try to find the file by searching for it
                logger.warning(f"Non-standard path format: {file_path_str}")
                return None

            # Extract the procedure
            procedure_text = self.extract_plsql_procedure(
                full_path,
                summary_entry["line_start"],
                summary_entry["line_end"],
                summary_entry["procedure_name"],
            )

            if not procedure_text:
                logger.warning(
                    f"Could not extract procedure {summary_entry['procedure_name']} from {full_path}"
                )
                return None

            # Enhance context with keywords
            enhanced_context = self.enhance_context_with_keywords(procedure_text)

            # Generate the complete prompt (same format as used in diversified generator)
            prompt = f"""You are an expert IFS Cloud developer analyzing PL/SQL procedures. Provide a comprehensive technical summary of the following procedure from the IFS Cloud {summary_entry.get('module', summary_entry.get('module_name', 'UNKNOWN')).upper()} module.

**Analysis Guidelines:**
- Focus on business logic, integration points, and technical implementation
- Explain the procedure's purpose within the broader IFS Cloud architecture  
- Identify key database objects, APIs, and dependencies
- Describe any complex business rules or validations
- Note integration patterns with other IFS Cloud modules
- Keep the summary concise but thorough (aim for 300-500 words)

**Procedure Information:**
- Module: {summary_entry.get('module', summary_entry.get('module_name', 'UNKNOWN')).upper()}
- Procedure: {summary_entry['procedure_name']}
- File: {Path(summary_entry['file_path']).name}
- Lines: {summary_entry['line_start']}-{summary_entry['line_end']}
- Complexity Score: {summary_entry.get('complexity_score', 'N/A')}

**Procedure Code:**
```plsql
{enhanced_context}
```

**Summary:**
Provide a detailed analysis following the above format. Your summary should cover the key aspects of the procedure while fitting the word count guidelines."""

            return prompt

        except Exception as e:
            logger.error(
                f"Error generating prompt for {summary_entry.get('procedure_name', 'unknown')}: {e}"
            )
            return None

    def correlate_summaries_with_prompts(self, clean_summaries_file: Path) -> str:
        """Correlate cleaned summaries with their source code and generate prompts."""

        # Load the cleaned summaries
        try:
            with open(clean_summaries_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)
            logger.info(
                f"Loaded {len(summaries)} cleaned summaries from {clean_summaries_file}"
            )
        except Exception as e:
            logger.error(f"Error loading summaries file {clean_summaries_file}: {e}")
            return ""

        # Process each summary
        correlated_data = []
        successful_correlations = 0
        failed_correlations = 0

        for i, summary in enumerate(summaries):
            try:
                logger.info(
                    f"Processing {i+1}/{len(summaries)}: {summary['module']}.{summary['procedure_name']}"
                )

                # Generate the prompt for this procedure
                prompt = self.generate_prompt_for_procedure(summary)

                if prompt:
                    # Estimate token counts
                    summary_tokens = self.estimate_token_count(summary["summary"])
                    prompt_tokens = self.estimate_token_count(prompt)

                    # Create the correlated entry
                    correlated_entry = {
                        "id": summary["id"],
                        "procedure_name": summary["procedure_name"],
                        "module": summary["module"],
                        "file_path": summary["file_path"],
                        "line_start": summary["line_start"],
                        "line_end": summary["line_end"],
                        "complexity_score": summary.get("complexity_score", 0),
                        "content_length": summary.get("content_length", 0),
                        "original_summary": summary["summary"],
                        "summary_token_count": summary_tokens,
                        "generation_time": summary.get("generation_time", 0),
                        "original_timestamp": summary.get("timestamp", ""),
                        "regenerated_prompt": prompt,
                        "prompt_token_count": prompt_tokens,
                        "correlation_timestamp": datetime.now().isoformat(),
                    }

                    correlated_data.append(correlated_entry)
                    successful_correlations += 1

                else:
                    logger.warning(
                        f"Failed to generate prompt for {summary['procedure_name']}"
                    )
                    failed_correlations += 1

            except Exception as e:
                logger.error(f"Error processing summary {i+1}: {e}")
                failed_correlations += 1

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            clean_summaries_file.parent
            / f"correlated_summaries_with_prompts_{timestamp}.json"
        )

        # Save the correlated data
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(correlated_data, f, indent=2, ensure_ascii=False)

            # Calculate token statistics
            if correlated_data:
                total_summary_tokens = sum(
                    entry["summary_token_count"] for entry in correlated_data
                )
                total_prompt_tokens = sum(
                    entry["prompt_token_count"] for entry in correlated_data
                )
                avg_summary_tokens = total_summary_tokens / len(correlated_data)
                avg_prompt_tokens = total_prompt_tokens / len(correlated_data)

                logger.info(f"✅ Correlated summaries saved to: {output_file}")
                logger.info(
                    f"📊 Results: {successful_correlations} successful, {failed_correlations} failed correlations"
                )
                logger.info(f"🔤 Token Statistics:")
                logger.info(f"   • Average summary tokens: {avg_summary_tokens:.0f}")
                logger.info(f"   • Average prompt tokens: {avg_prompt_tokens:.0f}")
                logger.info(f"   • Total summary tokens: {total_summary_tokens:,}")
                logger.info(f"   • Total prompt tokens: {total_prompt_tokens:,}")
            else:
                logger.info(f"✅ Correlated summaries saved to: {output_file}")
                logger.info(
                    f"📊 Results: {successful_correlations} successful, {failed_correlations} failed correlations"
                )

            return str(output_file)

        except Exception as e:
            logger.error(f"Error saving correlated data: {e}")
            return ""


def main():
    """Main function to correlate summaries with prompts."""
    print("🔗 IFS Cloud Summary-Source Correlator")
    print("=" * 50)

    # Find the cleaned summaries file
    batch_dir = Path("batch_summaries")
    clean_file = batch_dir / "combined_summaries_clean.json"

    if not clean_file.exists():
        print(f"❌ Error: Could not find cleaned summaries file: {clean_file}")
        return

    # Create correlator and process
    correlator = SummaryCorrelator()

    try:
        output_file = correlator.correlate_summaries_with_prompts(clean_file)

        if output_file:
            print(f"🎉 Successfully created correlated file: {Path(output_file).name}")
            print(f"📂 Location: {output_file}")
            print(
                f"\n💡 You can now use this file in your UI to review summaries with their complete prompts!"
            )
        else:
            print("❌ Failed to create correlated file")

    except Exception as e:
        print(f"💥 Error during correlation: {e}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    main()
