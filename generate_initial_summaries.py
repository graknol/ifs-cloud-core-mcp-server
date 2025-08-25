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
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np


class ProcedureSummaryGenerator:
    def __init__(self, keywords_file, tensorrt_enabled=True):
        self.keywords_file = keywords_file
        self.tensorrt_enabled = tensorrt_enabled
        self.keywords = self.load_keywords()
        self.procedures = []
        self.summaries = []

        # Initialize tokenizer and model for code generation
        print("🤖 Initializing AI model...")
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )

        # Load model with optimal settings for 16GB VRAM
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use fp16 for memory efficiency
                device_map="auto",  # Auto-distribute across GPU
                low_cpu_mem_usage=True,
                # Skip flash attention on Windows to avoid compatibility issues
            )
        except Exception as e:
            print(f"⚠️  Flash attention not available, using standard attention: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

        # Move to GPU if available
        if torch.cuda.is_available():
            if not hasattr(self.model, "device") or self.model.device.type != "cuda":
                self.model = self.model.cuda()
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM Usage: ~{self.estimate_vram_usage():.1f}GB")

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def estimate_vram_usage(self):
        """Estimate VRAM usage for the loaded model."""
        try:
            if hasattr(self.model, "get_memory_footprint"):
                return self.model.get_memory_footprint() / (1024**3)  # Convert to GB
            else:
                # Estimate based on model parameters (7B model ~14GB in fp16)
                return 14.0
        except:
            return 14.0  # Conservative estimate for 7B model

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

    def extract_business_relevant_code(self, code):
        """Extract only business-relevant parts: procedure name, parameter names, and body logic with prioritized control structures."""
        lines = code.split("\n")

        # Find procedure/function declaration line
        proc_name = "Unknown"
        param_names = []

        for i, line in enumerate(lines):
            line_clean = line.strip()
            if line_clean.upper().startswith(("PROCEDURE", "FUNCTION")):
                # Extract procedure name
                match = re.search(
                    r"(?:PROCEDURE|FUNCTION)\s+([A-Za-z_][A-Za-z0-9_]*)",
                    line_clean,
                    re.IGNORECASE,
                )
                if match:
                    proc_name = match.group(1)

                # Extract parameter names (skip types, just get meaningful names)
                # Look for parameters in current and next few lines until we hit 'IS' or 'AS'
                param_section = ""
                j = i
                while j < len(lines) and j < i + 10:  # Look ahead max 10 lines
                    param_section += lines[j] + " "
                    if re.search(r"\b(IS|AS)\b", lines[j], re.IGNORECASE):
                        break
                    j += 1

                # Extract parameter names (avoid SQL keywords, types, and common CRUD parameters)
                param_matches = re.findall(
                    r"\b([a-z_][a-z0-9_]*)\s*_?\s*(?:IN|OUT|IN\s+OUT)?\s+(?:VARCHAR2|NUMBER|DATE|BOOLEAN|TIMESTAMP)\b",
                    param_section,
                    re.IGNORECASE,
                )
                # Filter out common IFS CRUD parameters that are just noise
                crud_params = {
                    "info",
                    "objid",
                    "objversion",
                    "attr",
                    "action",
                    "in",
                    "out",
                    "is",
                    "as",
                    "begin",
                    "end",
                }
                param_names = [
                    p.replace("_", "")
                    for p in param_matches
                    if len(p) > 2 and p.lower() not in crud_params
                ]
                break

        # Find the actual business logic (after IS/AS, before END)
        body_start = -1
        body_end = -1

        for i, line in enumerate(lines):
            if body_start == -1 and re.search(
                r"\b(IS|AS)\s*$", line.strip(), re.IGNORECASE
            ):
                body_start = i + 1
            elif body_start != -1 and re.match(
                r"^END\s*[A-Za-z_]*\s*;?\s*$", line.strip(), re.IGNORECASE
            ):
                body_end = i
                break

        # Extract and prioritize business logic body
        if body_start != -1:
            if body_end == -1:
                body_end = len(lines)
            body_lines = lines[body_start:body_end]

            # Prioritize control structures and important business logic
            prioritized_logic = self.prioritize_control_structures(body_lines)
        else:
            prioritized_logic = code[:2000]  # Fallback to original if parsing fails

        # Create business-focused summary
        business_code = f"Procedure: {proc_name}\n"
        if param_names:
            business_code += (
                f"Processing: {', '.join(param_names[:5])}\n"  # Max 5 params
            )
        business_code += f"Logic:\n{prioritized_logic}"

        return business_code

    def prioritize_control_structures(self, body_lines):
        """Prioritize control structures and key business logic in code body."""
        # Remove empty lines and pure comments
        meaningful_lines = []
        for line in body_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("--"):
                meaningful_lines.append(line)

        # If short enough, return as-is
        if len("\n".join(meaningful_lines)) <= 2000:
            return "\n".join(meaningful_lines)

        # Categorize lines by importance
        priority_lines = []
        context_lines = []

        for i, line in enumerate(meaningful_lines):
            line_upper = line.upper()
            is_priority = (
                # Control structures
                any(
                    pattern in line_upper
                    for pattern in ["IF ", "ELSIF", "ELSE", "END IF"]
                )
                or any(
                    pattern in line_upper
                    for pattern in ["FOR ", "WHILE ", "LOOP", "END LOOP"]
                )
                or
                # Exception handling
                "EXCEPTION" in line_upper
                or "RAISE" in line_upper
                or
                # API calls (business integration)
                "_API." in line
                or
                # SQL operations (data manipulation)
                any(
                    pattern in line_upper
                    for pattern in ["SELECT", "INSERT", "UPDATE", "DELETE", "CURSOR"]
                )
                or
                # Variable assignments with business meaning
                ":=" in line
                and any(
                    keyword in line_upper
                    for keyword in ["_ID", "_NO", "_CODE", "_STATUS", "_STATE", "_TYPE"]
                )
            )

            if is_priority:
                priority_lines.append((i, line))
            else:
                context_lines.append((i, line))

        # Build result with priority lines and selective context
        result_lines = []
        char_count = 0
        max_chars = 2000

        # Always include priority lines
        last_included_idx = -1
        for idx, line in priority_lines:
            if char_count + len(line) + 1 > max_chars:
                break

            # Add context gap marker if there's a significant gap
            if last_included_idx != -1 and idx - last_included_idx > 3:
                result_lines.append("   ...")
                char_count += 8

            result_lines.append(line)
            char_count += len(line) + 1
            last_included_idx = idx

        # Fill remaining space with context lines near priority lines
        remaining_chars = max_chars - char_count
        if remaining_chars > 100:  # Only if we have meaningful space left
            context_added = 0
            for idx, line in context_lines:
                if (
                    char_count + len(line) + 1 > max_chars or context_added > 5
                ):  # Limit context lines
                    break

                # Only add context lines that are near priority lines
                near_priority = any(
                    abs(idx - p_idx) <= 2 for p_idx, _ in priority_lines
                )
                if near_priority:
                    result_lines.append(line)
                    char_count += len(line) + 1
                    context_added += 1

        # Sort by original line position to maintain flow
        result_with_positions = []
        for line in result_lines:
            if line == "   ...":
                result_with_positions.append((999999, line))  # Sort gaps to end
                continue
            # Find original position
            for idx, orig_line in enumerate(meaningful_lines):
                if orig_line == line:
                    result_with_positions.append((idx, line))
                    break

        result_with_positions.sort()
        final_result = "\n".join(line for _, line in result_with_positions)

        # Add final truncation marker if we hit the limit
        if char_count >= max_chars:
            final_result += "\n   ..."

        return final_result

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

        # Extract business-relevant parts of the code (maximize space for code)
        business_code = self.extract_business_relevant_code(code)

        prompt = f"""
        IFS Cloud business procedure analysis. Focus on business value and operational impact.
        
        Context: {context}
        Keywords: {keyword_context}
        Complexity: {complexity['level']} ({complexity['score']:.1f})
        
        {business_code}
        
        Write a business-focused summary explaining what business problem this solves and its operational value.
        Use natural business language for functional consultants.
        """

        try:
            # Format as chat messages for Qwen2.5-Coder
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert IFS Cloud business analyst. Create concise, business-focused summaries of PL/SQL procedures. Respond with only the summary text, no headers or markdown.",
                },
                {
                    "role": "user",
                    "content": f"""Analyze this IFS Cloud procedure and write a concise business summary (1-2 sentences):

Context: {context}
Keywords: {keyword_context}
Complexity: {complexity['level']} ({complexity['score']:.1f})

{business_code}

Write ONLY a plain text business summary that explains what business problem this solves. Start with an action verb like "Validates", "Calculates", "Updates", or "Processes".""",
                },
            ]

            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize for generation (use larger context window)
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=4096,  # Use larger context for Qwen2.5-Coder
                truncation=True,
                padding=True,
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate summary using the model
            with torch.no_grad():
                generation_config = GenerationConfig(
                    max_new_tokens=80,  # Shorter output for concise summaries
                    temperature=0.2,  # Lower temperature for more focused output
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetition
                )

                outputs = self.model.generate(
                    **inputs, generation_config=generation_config, use_cache=True
                )

                # Decode the generated text
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()

                # Clean up the generated summary
                if generated_text:
                    # Remove markdown headers, formatting, and other artifacts
                    cleaned_text = (
                        generated_text.replace("###", "")
                        .replace("##", "")
                        .replace("#", "")
                    )
                    cleaned_text = cleaned_text.replace("**", "").replace("*", "")
                    cleaned_text = cleaned_text.strip()

                    # Split by lines and take the first substantial sentence
                    lines = [
                        line.strip()
                        for line in cleaned_text.split("\n")
                        if line.strip()
                    ]
                    if lines:
                        # Find the first line that looks like a proper business summary
                        for line in lines:
                            if (
                                len(line) > 20
                                and any(
                                    line.startswith(verb)
                                    for verb in [
                                        "Validates",
                                        "Calculates",
                                        "Updates",
                                        "Processes",
                                        "Manages",
                                        "Handles",
                                        "Creates",
                                        "Retrieves",
                                        "Deletes",
                                        "Modifies",
                                        "Controls",
                                        "Executes",
                                        "Performs",
                                        "Checks",
                                        "Ensures",
                                    ]
                                )
                                or any(
                                    word in line.lower()
                                    for word in [
                                        "business",
                                        "process",
                                        "data",
                                        "system",
                                        "order",
                                        "customer",
                                        "financial",
                                    ]
                                )
                            ):
                                enhanced_summary = line
                                # Ensure reasonable length
                                if len(enhanced_summary) > 250:
                                    enhanced_summary = enhanced_summary[:250] + "..."
                                return enhanced_summary

                        # If no perfect match, take the first substantial line
                        enhanced_summary = lines[0]
                        if len(enhanced_summary) > 250:
                            enhanced_summary = enhanced_summary[:250] + "..."
                        return enhanced_summary

            # Fallback to context summary if generation fails
            return self.enhance_context_summary(context, keywords, complexity)

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

            # Extract business-relevant code for display
            business_code = self.extract_business_relevant_code(proc["code"])

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
                        business_code[:800] + "..."
                        if len(business_code) > 800
                        else business_code
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
