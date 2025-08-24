#!/usr/bin/env python3
"""
AST-based PL/SQL Function Extractor

This module will integrate with your PL/SQL parser executable to extract
functions using proper AST parsing instead of regex-based approaches.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ASTBasedFunctionExtractor:
    """Extract PL/SQL functions using AST parser executable."""

    def __init__(self, parser_executable: str = "plsql_parser.exe"):
        self.parser_executable = parser_executable
        self.verify_parser_available()

    def verify_parser_available(self) -> bool:
        """Check if the PL/SQL parser executable is available."""
        try:
            result = subprocess.run(
                [self.parser_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"✅ PL/SQL parser available: {self.parser_executable}")
                return True
            else:
                logger.warning(
                    f"⚠️ Parser executable found but returned error: {result.stderr}"
                )
                return False
        except FileNotFoundError:
            logger.warning(f"⚠️ Parser executable not found: {self.parser_executable}")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking parser: {e}")
            return False

    def extract_functions_from_file(self, file_path: Path) -> List[Dict]:
        """Extract functions from PL/SQL file using AST parser."""
        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Create temporary file for parser input
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".plsql", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                # Call the parser executable
                result = subprocess.run(
                    [
                        self.parser_executable,
                        "--extract-functions",
                        "--format",
                        "json",
                        temp_file_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    # Parse the JSON output from the parser
                    parser_output = json.loads(result.stdout)
                    return self._process_parser_output(parser_output, file_path)
                else:
                    logger.error(f"Parser failed for {file_path}: {result.stderr}")
                    return []

            finally:
                # Clean up temporary file
                Path(temp_file_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error extracting functions from {file_path}: {e}")
            return []

    def _process_parser_output(
        self, parser_output: Dict, file_path: Path
    ) -> List[Dict]:
        """Process the output from the AST parser into our expected format."""
        functions = []

        # Expected parser output format:
        # {
        #   "functions": [
        #     {
        #       "name": "function_name",
        #       "type": "FUNCTION" | "PROCEDURE",
        #       "start_line": 123,
        #       "end_line": 456,
        #       "parameters": [...],
        #       "body": "function body text",
        #       "complexity_metrics": {
        #         "cyclomatic_complexity": 15,
        #         "decision_points": 10,
        #         "nesting_depth": 3
        #       }
        #     }
        #   ]
        # }

        for func_data in parser_output.get("functions", []):
            # Calculate our standard complexity metrics
            complexity = self._calculate_complexity_metrics(func_data)

            # Filter out functions that are too simple or complex
            if complexity["code_lines"] < 10 or complexity["code_lines"] > 500:
                continue

            # Skip trivial functions
            if self._is_trivial_function(func_data):
                continue

            functions.append(
                {
                    "function_name": func_data.get("name", "unknown"),
                    "function_text": func_data.get("body", ""),
                    "function_type": func_data.get("type", "UNKNOWN"),
                    "start_line": func_data.get("start_line", 0),
                    "end_line": func_data.get("end_line", 0),
                    "parameters": func_data.get("parameters", []),
                    "complexity": complexity,
                    "file_path": file_path,
                    "position": len(functions),
                }
            )

        # Sort by complexity and return best candidates
        functions.sort(
            key=lambda x: x["complexity"]["cyclomatic_complexity"], reverse=True
        )
        return functions[:10]  # Top 10 most complex functions per file

    def _calculate_complexity_metrics(self, func_data: Dict) -> Dict:
        """Calculate complexity metrics from parser data."""
        body = func_data.get("body", "")
        lines = body.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        # Use parser's complexity if available, otherwise calculate
        cyclomatic_complexity = func_data.get("complexity_metrics", {}).get(
            "cyclomatic_complexity"
        )
        if cyclomatic_complexity is None:
            # Fallback calculation
            decision_keywords = [
                "IF",
                "CASE",
                "WHEN",
                "LOOP",
                "FOR",
                "WHILE",
                "EXCEPTION",
            ]
            cyclomatic_complexity = 1  # Base complexity

            for line in non_empty_lines:
                line_upper = line.upper()
                for keyword in decision_keywords:
                    cyclomatic_complexity += line_upper.count(keyword)

        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "cyclomatic_complexity": cyclomatic_complexity,
            "estimated_tokens": len(body.split()) * 1.3,
            "nesting_depth": func_data.get("complexity_metrics", {}).get(
                "nesting_depth", 0
            ),
            "decision_points": func_data.get("complexity_metrics", {}).get(
                "decision_points", 0
            ),
        }

    def _is_trivial_function(self, func_data: Dict) -> bool:
        """Check if this is a trivial function that should be skipped."""
        name = func_data.get("name", "").upper()
        body = func_data.get("body", "")

        # Skip obvious getters/setters
        if any(pattern in name for pattern in ["GET_", "SET_", "IS_", "HAS_"]):
            if len(body.split("\n")) < 20:
                return True

        # Skip functions that are mostly assignments
        lines = [line.strip() for line in body.split("\n") if line.strip()]
        if not lines:
            return True

        assignment_lines = sum(
            1 for line in lines if ":=" in line or "RETURN " in line.upper()
        )
        if assignment_lines > 0 and assignment_lines / len(lines) > 0.7:
            return True  # More than 70% assignments - likely trivial

        return False

    def extract_function_signatures(self, file_path: Path) -> List[Dict]:
        """Extract only function signatures (for lighter processing)."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".plsql", delete=False, encoding="utf-8"
            ) as temp_file:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    temp_file.write(f.read())
                temp_file_path = temp_file.name

            try:
                result = subprocess.run(
                    [
                        self.parser_executable,
                        "--extract-signatures",
                        "--format",
                        "json",
                        temp_file_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    return json.loads(result.stdout).get("signatures", [])
                else:
                    logger.error(
                        f"Signature extraction failed for {file_path}: {result.stderr}"
                    )
                    return []

            finally:
                Path(temp_file_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error extracting signatures from {file_path}: {e}")
            return []


# Integration point for the main extraction script
class PLSQLSampleExtractorWithAST:
    """Enhanced sample extractor using AST-based parsing."""

    def __init__(self, version_path: Path, parser_executable: str = "plsql_parser.exe"):
        self.version_path = version_path
        self.ast_extractor = ASTBasedFunctionExtractor(parser_executable)
        self.pagerank_scores = self.load_pagerank_scores()

    def load_pagerank_scores(self) -> Dict[str, float]:
        """Load PageRank scores from ranked.jsonl"""
        scores = {}
        ranked_file = Path("ranked.jsonl")

        if ranked_file.exists():
            with open(ranked_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    scores[item["file_path"]] = item.get("pagerank_score", 0.0)

        return scores

    def extract_functions_from_file(self, file_path: Path) -> List[Dict]:
        """Extract functions using AST parser."""
        return self.ast_extractor.extract_functions_from_file(file_path)

    def extract_stratified_samples(self, target_count: int = 200) -> List[Dict]:
        """Extract samples using AST-based parsing and stratified sampling."""
        source_dir = self.version_path / "source"
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return []

        plsql_files = list(source_dir.rglob("*.plsql"))
        logger.info(f"Found {len(plsql_files)} PL/SQL files")

        # Sort by PageRank score
        scored_files = [(f, self.pagerank_scores.get(str(f), 0.0)) for f in plsql_files]
        scored_files.sort(key=lambda x: x[1], reverse=True)

        # Stratified sampling
        high_tier = scored_files[:50]  # Top 50 by PageRank
        mid_tier = scored_files[50:150]  # Middle tier
        low_tier = scored_files[150:300]  # Lower tier (if available)

        all_samples = []

        for tier, tier_name in [
            (high_tier, "high"),
            (mid_tier, "mid"),
            (low_tier, "low"),
        ]:
            tier_samples = []

            for file_path, score in tier:
                functions = self.extract_functions_from_file(file_path)
                tier_samples.extend(functions)

            # Sort by complexity and diversity
            tier_samples.sort(
                key=lambda x: x["complexity"]["cyclomatic_complexity"], reverse=True
            )

            # Take best samples from this tier
            samples_per_tier = min(target_count // 3, len(tier_samples))
            selected = tier_samples[:samples_per_tier]

            logger.info(
                f"Selected {len(selected)} samples from {tier_name} tier using AST parser"
            )
            all_samples.extend(selected)

        return all_samples[:target_count]


def main():
    """Test the AST-based extractor."""
    print("🔍 AST-Based PL/SQL Function Extractor")
    print("=" * 50)

    extractor = ASTBasedFunctionExtractor()

    if not extractor.verify_parser_available():
        print("❌ PL/SQL parser executable not available.")
        print("   Please build your parser executable and place it in the PATH")
        print("   or specify the path with --parser-executable")
        return

    print("✅ Parser is ready for integration!")
    print("\nTo use with your existing extraction script:")
    print("1. Build your PL/SQL parser executable")
    print("2. Place it in PATH or specify location")
    print("3. Replace the extract_functions_from_file method")


if __name__ == "__main__":
    main()
