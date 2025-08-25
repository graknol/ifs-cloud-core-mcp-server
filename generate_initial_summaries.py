#!/usr/bin/env python3
"""
Generate 200 initial procedure summaries using AST analysis and curated keywords.
Sample diverse procedures across modules and complexity levels for evaluation.
"""

import json
import csv
import re
import ast
import random
from pathlib import Path
from collections import defaultdict, Counter
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class ProcedureSummaryGenerator:
    def __init__(self, keywords_file, tensorrt_enabled=True):
        self.keywords_file = keywords_file
        self.tensorrt_enabled = tensorrt_enabled
        self.keywords = self.load_keywords()
        self.procedures = []
        self.summaries = []

        # Initialize tokenizer and model
        print("🤖 Initializing AI model...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")

        # PL/SQL patterns for AST-like analysis
        self.plsql_patterns = {
            "procedure": re.compile(
                r"PROCEDURE\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
            ),
            "function": re.compile(
                r"FUNCTION\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
            ),
            "parameters": re.compile(r"\(\s*([^)]+)\s*\)", re.IGNORECASE),
            "if_statements": re.compile(r"\bIF\b", re.IGNORECASE),
            "loops": re.compile(r"\b(FOR|WHILE)\b", re.IGNORECASE),
            "exceptions": re.compile(r"\bEXCEPTION\b", re.IGNORECASE),
            "api_calls": re.compile(
                r"([A-Za-z_][A-Za-z0-9_]*_API\.[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
            ),
            "comments": re.compile(r"--.*$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
            "assignments": re.compile(r":=", re.IGNORECASE),
            "selects": re.compile(r"\bSELECT\b", re.IGNORECASE),
            "inserts": re.compile(r"\bINSERT\b", re.IGNORECASE),
            "updates": re.compile(r"\bUPDATE\b", re.IGNORECASE),
            "deletes": re.compile(r"\bDELETE\b", re.IGNORECASE),
        }

    def load_keywords(self):
        """Load curated keywords from CSV."""
        keywords = {}
        variants_map = {}

        with open(self.keywords_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keyword = row["keyword"].strip().lower()
                variants_str = row.get("variants_with_scores", "").strip()

                keywords[keyword] = {
                    "occurrences": int(row["total_occurrences"]),
                    "variants": [],
                }

                # Parse variants (format: "variant1(0.85), variant2(0.70)")
                if variants_str:
                    for variant_score in variants_str.split(", "):
                        if "(" in variant_score and ")" in variant_score:
                            variant = variant_score.split("(")[0].strip()
                            score = float(variant_score.split("(")[1].split(")")[0])
                            keywords[keyword]["variants"].append(
                                {"word": variant, "score": score}
                            )
                            variants_map[variant.lower()] = keyword
                        else:
                            variant = variant_score.strip()
                            keywords[keyword]["variants"].append(
                                {"word": variant, "score": 1.0}
                            )
                            variants_map[variant.lower()] = keyword

        print(f"📋 Loaded {len(keywords)} keywords with {len(variants_map)} variants")
        return {"main": keywords, "variants": variants_map}

    def analyze_procedure_complexity(self, code):
        """Analyze procedure complexity using PL/SQL patterns."""
        complexity_score = 0
        features = {}

        # Count various complexity indicators
        features["if_count"] = len(self.plsql_patterns["if_statements"].findall(code))
        features["loop_count"] = len(self.plsql_patterns["loops"].findall(code))
        features["api_calls"] = len(self.plsql_patterns["api_calls"].findall(code))
        features["assignments"] = len(self.plsql_patterns["assignments"].findall(code))
        features["sql_operations"] = (
            len(self.plsql_patterns["selects"].findall(code))
            + len(self.plsql_patterns["inserts"].findall(code))
            + len(self.plsql_patterns["updates"].findall(code))
            + len(self.plsql_patterns["deletes"].findall(code))
        )
        features["exceptions"] = len(self.plsql_patterns["exceptions"].findall(code))
        features["line_count"] = len(code.split("\n"))
        features["char_count"] = len(code)

        # Calculate complexity score
        complexity_score = (
            features["if_count"] * 2
            + features["loop_count"] * 3
            + features["api_calls"] * 1
            + features["sql_operations"] * 2
            + features["exceptions"] * 4
            + features["line_count"] * 0.1
        )

        # Classify complexity
        if complexity_score < 10:
            complexity_level = "simple"
        elif complexity_score < 30:
            complexity_level = "medium"
        else:
            complexity_level = "complex"

        return {
            "score": complexity_score,
            "level": complexity_level,
            "features": features,
        }

    def extract_procedure_keywords(self, code):
        """Extract keywords from procedure code using curated list."""
        found_keywords = []
        code_lower = code.lower()

        # Check main keywords
        for keyword, data in self.keywords["main"].items():
            if keyword in code_lower:
                count = code_lower.count(keyword)
                found_keywords.append(
                    {
                        "keyword": keyword,
                        "count": count,
                        "type": "main",
                        "importance": data["occurrences"],
                    }
                )

        # Check variants
        for variant, main_keyword in self.keywords["variants"].items():
            if variant in code_lower:
                count = code_lower.count(variant)
                found_keywords.append(
                    {
                        "keyword": variant,
                        "main_keyword": main_keyword,
                        "count": count,
                        "type": "variant",
                        "importance": self.keywords["main"][main_keyword][
                            "occurrences"
                        ],
                    }
                )

        # Sort by importance and frequency
        found_keywords.sort(key=lambda x: (x["importance"], x["count"]), reverse=True)
        return found_keywords[:20]  # Top 20 keywords

    def generate_context_summary(self, code, keywords, complexity):
        """Generate context summary using keywords and complexity analysis."""

        # Extract procedure name
        proc_match = self.plsql_patterns["procedure"].search(code)
        func_match = self.plsql_patterns["function"].search(code)
        name = (
            proc_match.group(1)
            if proc_match
            else func_match.group(1) if func_match else "Unknown"
        )

        # Extract parameters
        param_match = self.plsql_patterns["parameters"].search(code)
        param_count = len(param_match.group(1).split(",")) if param_match else 0

        # Business context from keywords
        business_keywords = [
            k["keyword"]
            for k in keywords[:5]
            if k["type"] == "main" and k["importance"] > 5000
        ]

        # Operation type based on patterns
        operations = []
        if complexity["features"]["sql_operations"] > 0:
            operations.append("data operations")
        if complexity["features"]["api_calls"] > 3:
            operations.append("API integration")
        if complexity["features"]["if_count"] > 2:
            operations.append("business logic")
        if complexity["features"]["exceptions"] > 0:
            operations.append("error handling")

        # Generate summary components
        summary_parts = []

        # Function type and complexity
        func_type = "Function" if func_match else "Procedure"
        summary_parts.append(f"{func_type} '{name}' ({complexity['level']} complexity)")

        # Parameters
        if param_count > 0:
            summary_parts.append(
                f"with {param_count} parameter{'s' if param_count > 1 else ''}"
            )

        # Business context
        if business_keywords:
            summary_parts.append(f"handles {', '.join(business_keywords[:3])}")

        # Operations
        if operations:
            summary_parts.append(f"performs {', '.join(operations)}")

        return " ".join(summary_parts) + "."

    def extract_procedures_from_file(self, file_path):
        """Extract individual procedures/functions from a file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            procedures = []

            # Find all procedures and functions
            proc_pattern = re.compile(
                r"((?:PROCEDURE|FUNCTION)\s+[A-Za-z_][A-Za-z0-9_]*.*?)(?=(?:PROCEDURE|FUNCTION|\Z))",
                re.IGNORECASE | re.DOTALL,
            )

            matches = proc_pattern.findall(content)

            for match in matches:
                if len(match.strip()) > 100:  # Skip very short procedures
                    # Extract module name from the IFS path structure
                    # Path should be like: .../MODULE/source/MODULE/database/...
                    module_name = "unknown"
                    path_parts = file_path.parts
                    if "database" in path_parts:
                        db_index = path_parts.index("database")
                        if db_index >= 1:  # Should have at least module/database
                            module_name = path_parts[
                                db_index - 1
                            ]  # The MODULE right before database

                    procedures.append(
                        {
                            "code": match.strip(),
                            "file_path": str(file_path),
                            "module": module_name,
                        }
                    )

            return procedures

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def scan_codebase_for_procedures(self, root_directory):
        """Scan codebase and extract all procedures with analysis."""
        root_path = Path(root_directory)
        all_procedures = []

        print(f"🔍 Scanning {root_directory} for database procedures...")

        # Only look for database scripts in IFS modules (exclude replication, wadaco)
        excluded_modules = {
            "replication",
            "wadaco",
            "fndrpl",
        }  # Also exclude foundation replication

        # Find all database directories in modules
        database_paths = []
        for module_dir in root_path.iterdir():
            if (
                module_dir.is_dir()
                and module_dir.name.lower() not in excluded_modules
                and not module_dir.name.startswith(".")
            ):  # Skip hidden directories

                # IFS structure: MODULE/source/MODULE/database/
                database_dir = module_dir / "source" / module_dir.name / "database"
                if database_dir.exists():
                    database_paths.append(database_dir)

        print(f"📁 Found {len(database_paths)} module database directories")

        # Collect all .plsql files from database directories
        plsql_files = []
        for db_path in database_paths:
            module_files = list(db_path.rglob("*.plsql"))
            plsql_files.extend(module_files)
            print(
                f"   {db_path.parent.parent.parent.name}/source/{db_path.parent.name}/database: {len(module_files)} files"
            )

        # Limit for initial sampling if too many
        if len(plsql_files) > 1000:
            print(
                f"🎯 Limiting to first 1000 files from {len(plsql_files)} total files"
            )
            plsql_files = plsql_files[:1000]

        processed = 0
        for file_path in plsql_files:
            procedures = self.extract_procedures_from_file(file_path)

            for proc in procedures:
                # Analyze complexity
                complexity = self.analyze_procedure_complexity(proc["code"])

                # Extract keywords
                keywords = self.extract_procedure_keywords(proc["code"])

                # Generate initial context
                context = self.generate_context_summary(
                    proc["code"], keywords, complexity
                )

                all_procedures.append(
                    {
                        **proc,
                        "complexity": complexity,
                        "keywords": keywords,
                        "context": context,
                        "char_length": len(proc["code"]),
                    }
                )

            processed += 1
            if processed % 100 == 0:
                print(
                    f"   Processed {processed}/{len(plsql_files)} files, found {len(all_procedures)} procedures"
                )

        print(
            f"✅ Found {len(all_procedures)} total database procedures from IFS modules"
        )
        return all_procedures

    def stratified_sample_procedures(self, procedures, target_count=10):
        """Create stratified sample across modules and complexity levels."""

        # Group by module and complexity
        groups = defaultdict(lambda: defaultdict(list))

        for proc in procedures:
            module = proc["module"]
            complexity = proc["complexity"]["level"]
            groups[module][complexity].append(proc)

        # Calculate sampling distribution
        total_modules = len(groups)
        complexity_distribution = {
            "simple": 0.4,
            "medium": 0.4,
            "complex": 0.2,
        }  # 40-40-20 split

        sampled_procedures = []
        procedures_per_module = max(1, target_count // total_modules)

        print(f"\n📊 SAMPLING STRATEGY:")
        print(f"   Target samples: {target_count}")
        print(f"   Modules found: {total_modules}")
        print(f"   Samples per module: ~{procedures_per_module}")
        print(f"   Complexity distribution: {complexity_distribution}")

        for module, complexity_groups in groups.items():
            module_samples = []

            # Sample from each complexity level
            for complexity_level in ["simple", "medium", "complex"]:
                available = complexity_groups.get(complexity_level, [])
                if not available:
                    continue

                target_for_level = int(
                    procedures_per_module * complexity_distribution[complexity_level]
                )
                sample_count = min(target_for_level, len(available))

                if sample_count > 0:
                    samples = random.sample(available, sample_count)
                    module_samples.extend(samples)

            # Add remaining samples randomly if under target
            if len(module_samples) < procedures_per_module:
                all_module_procs = []
                for complexity_procs in complexity_groups.values():
                    all_module_procs.extend(complexity_procs)

                remaining_needed = procedures_per_module - len(module_samples)
                available_remaining = [
                    p for p in all_module_procs if p not in module_samples
                ]

                if available_remaining:
                    additional_samples = random.sample(
                        available_remaining,
                        min(remaining_needed, len(available_remaining)),
                    )
                    module_samples.extend(additional_samples)

            sampled_procedures.extend(module_samples[:procedures_per_module])

            print(
                f"   {module}: {len(module_samples)} samples (simple: {len(complexity_groups.get('simple', []))}, "
                f"medium: {len(complexity_groups.get('medium', []))}, complex: {len(complexity_groups.get('complex', []))})"
            )

        # If we're under target, sample more randomly
        if len(sampled_procedures) < target_count:
            remaining_needed = target_count - len(sampled_procedures)
            available_remaining = [p for p in procedures if p not in sampled_procedures]

            if available_remaining:
                additional = random.sample(
                    available_remaining, min(remaining_needed, len(available_remaining))
                )
                sampled_procedures.extend(additional)

        return sampled_procedures[:target_count]

    def generate_enhanced_summary(self, procedure_data):
        """Generate enhanced summary using AI model and context."""

        code = procedure_data["code"]
        keywords = procedure_data["keywords"]
        complexity = procedure_data["complexity"]
        context = procedure_data["context"]

        # Prepare context for AI model
        keyword_context = ", ".join([k["keyword"] for k in keywords[:5]])

        # Create enhanced prompt
        prompt = f"""
        Analyze this IFS Cloud PL/SQL procedure:
        
        Context: {context}
        Key business terms: {keyword_context}
        Complexity: {complexity['level']} ({complexity['score']:.1f})
        
        Code snippet:
        {code[:1000]}...
        
        Generate a concise business summary focusing on:
        1. What business process it supports
        2. Key operations performed
        3. Main data entities involved
        """

        try:
            # Tokenize and get embeddings
            inputs = self.tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # For now, use the context summary as base and enhance it
            # In a full implementation, this would use a proper generation model
            enhanced_summary = self.enhance_context_summary(
                context, keywords, complexity
            )

            return enhanced_summary

        except Exception as e:
            print(f"Warning: AI enhancement failed, using context summary: {e}")
            return context

    def enhance_context_summary(self, base_summary, keywords, complexity):
        """Enhance the context summary with additional business intelligence."""

        # Extract business domain from keywords
        domains = []
        for keyword_data in keywords[:5]:
            keyword = keyword_data["keyword"]
            if keyword in ["order", "purchase", "sales"]:
                domains.append("order management")
            elif keyword in ["inventory", "part", "item"]:
                domains.append("inventory management")
            elif keyword in ["payment", "invoice", "cost", "amount"]:
                domains.append("financial processing")
            elif keyword in ["customer", "supplier", "party"]:
                domains.append("party management")
            elif keyword in ["project", "activity", "task"]:
                domains.append("project management")

        # Create enhanced summary
        enhanced_parts = [base_summary]

        if domains:
            unique_domains = list(set(domains))
            enhanced_parts.append(
                f"Supports {', '.join(unique_domains[:2])} business domain."
            )

        if complexity["features"]["api_calls"] > 5:
            enhanced_parts.append(
                "Heavy API integration for cross-module communication."
            )

        if complexity["features"]["sql_operations"] > 10:
            enhanced_parts.append("Performs extensive database operations.")

        return " ".join(enhanced_parts)

    def export_summaries_for_evaluation(self, sampled_procedures, output_file):
        """Export sampled procedures with summaries for manual evaluation."""

        evaluation_data = []

        for i, proc in enumerate(sampled_procedures, 1):
            # Generate enhanced summary
            enhanced_summary = self.generate_enhanced_summary(proc)

            evaluation_data.append(
                {
                    "id": i,
                    "module": proc["module"],
                    "file_path": proc["file_path"],
                    "complexity_level": proc["complexity"]["level"],
                    "complexity_score": round(proc["complexity"]["score"], 2),
                    "char_length": proc["char_length"],
                    "keyword_count": len(proc["keywords"]),
                    "top_keywords": ", ".join(
                        [k["keyword"] for k in proc["keywords"][:5]]
                    ),
                    "context_summary": proc["context"],
                    "enhanced_summary": enhanced_summary,
                    "code_snippet": (
                        proc["code"][:500] + "..."
                        if len(proc["code"]) > 500
                        else proc["code"]
                    ),
                    "manual_rating": "",
                    "manual_corrected_summary": "",
                    "notes": "",
                }
            )

            if i % 20 == 0:
                print(f"   Generated {i}/{len(sampled_procedures)} summaries")

        # Export to CSV for evaluation
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            if evaluation_data:
                writer = csv.DictWriter(f, fieldnames=evaluation_data[0].keys())
                writer.writeheader()
                writer.writerows(evaluation_data)

        print(f"💾 Evaluation data exported to: {output_file}")
        return evaluation_data


def main():
    keywords_file = "final_optimizer_keywords.csv"
    root_directory = r"C:\repos\_ifs\25.1.0"

    print("🎯 IFS PROCEDURE SUMMARY GENERATOR")
    print("=" * 60)

    if not Path(keywords_file).exists():
        print(f"❌ Keywords file not found: {keywords_file}")
        return

    if not Path(root_directory).exists():
        print(f"❌ IFS directory not found: {root_directory}")
        return

    # Set random seed for reproducible sampling
    random.seed(42)

    generator = ProcedureSummaryGenerator(keywords_file)

    print("\n🔍 SCANNING CODEBASE...")
    all_procedures = generator.scan_codebase_for_procedures(root_directory)

    if len(all_procedures) < 10:
        print(f"⚠️  Only found {len(all_procedures)} procedures, using all of them")
        target_samples = len(all_procedures)
    else:
        target_samples = 10  # Start small for iterative fine-tuning

    print(f"\n📊 CREATING STRATIFIED SAMPLE...")
    sampled_procedures = generator.stratified_sample_procedures(
        all_procedures, target_samples
    )

    print(f"\n🤖 GENERATING ENHANCED SUMMARIES...")
    output_file = "procedure_summaries_for_evaluation.csv"
    evaluation_data = generator.export_summaries_for_evaluation(
        sampled_procedures, output_file
    )

    # Generate summary statistics
    complexity_counts = Counter(proc["complexity_level"] for proc in evaluation_data)
    module_counts = Counter(proc["module"] for proc in evaluation_data)

    print(f"\n📈 SAMPLING RESULTS:")
    print(f"   Total samples: {len(evaluation_data)}")
    print(f"   Complexity distribution: {dict(complexity_counts)}")
    print(f"   Top 5 modules: {dict(module_counts.most_common(5))}")
    print(
        f"   Average keywords per procedure: {np.mean([int(proc['keyword_count']) for proc in evaluation_data]):.1f}"
    )

    print(f"\n✅ SUMMARY GENERATION COMPLETE!")
    print(f"📄 Evaluation file: {output_file}")
    print(f"🎯 Next step: Review and rate summaries, then fine-tune model")

    # Show preview of first few summaries
    print(f"\n📋 PREVIEW OF FIRST 3 SUMMARIES:")
    print("-" * 60)
    for i, proc in enumerate(evaluation_data[:3], 1):
        print(f"{i}. [{proc['module']}] {proc['complexity_level'].upper()} complexity")
        print(f"   Keywords: {proc['top_keywords']}")
        print(f"   Summary: {proc['enhanced_summary']}")
        print()


if __name__ == "__main__":
    main()
