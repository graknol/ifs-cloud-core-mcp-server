#!/usr/bin/env python3
"""
Alternative implementation using OpenAI API (ChatGPT) for PLSQL analysis
This provides a fallback option if GitHub Copilot API doesn't work as expected
"""

import openai
import os
import json
import time
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime


class PLSQLAnalyzerOpenAI:
    def __init__(self, top_10_dir="top_10", output_dir="plsql_analysis_openai"):
        self.top_10_dir = Path(top_10_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "raw_responses").mkdir(exist_ok=True)
        (self.output_dir / "extracted_procedures").mkdir(exist_ok=True)

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Analysis system prompt
        self.system_prompt = """You are an expert PL/SQL code analyst specializing in Oracle database procedures and functions. Your task is to analyze PL/SQL files and extract procedure/function summaries for embedding training purposes.

INSTRUCTIONS:
1. Identify up to 50 procedures and functions from the provided PL/SQL code
2. Prioritize procedures/functions that show diversity in complexity and business logic
3. For each procedure/function, provide:
   - The exact name and parameter signature
   - A 2-3 sentence conceptual summary focusing on WHAT it does, not HOW
   - Focus on business purpose and conceptual understanding
4. Avoid implementation details - focus on the business concept and purpose
5. These summaries will be used for training search embeddings, so clarity is crucial

FORMAT your response exactly as shown:
## PROCEDURE_NAME(parameter1 TYPE, parameter2 TYPE)
Clear 2-3 sentence summary explaining the business purpose and what this procedure conceptually accomplishes in the system.

## FUNCTION_NAME(parameters) RETURN TYPE  
Clear 2-3 sentence summary explaining what this function conceptually provides or calculates."""

    def analyze_file_with_openai(self, file_path, module_name, content):
        """Analyze a single PLSQL file using OpenAI API"""
        print(f"📝 Analyzing {module_name}/{file_path.name} with OpenAI")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better code analysis
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Analyze the following PL/SQL file from module '{module_name}':\n\nFILE: {file_path.name}\n\n{content}",
                    },
                ],
                max_tokens=4000,
                temperature=0.2,  # Lower temperature for more consistent analysis
            )

            response_text = response.choices[0].message.content

            if response_text:
                # Save raw response
                response_file = (
                    self.output_dir
                    / "raw_responses"
                    / f"{module_name}_{file_path.stem}_openai.md"
                )
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(f"# OpenAI Analysis of {module_name}/{file_path.name}\n\n")
                    f.write(f"File: {file_path}\n")
                    f.write(f"Module: {module_name}\n")
                    f.write(f"Model: gpt-4\n")
                    f.write(f"Analyzed: {datetime.now().isoformat()}\n")
                    f.write(f"Content Length: {len(content)} chars\n\n")
                    f.write("## Analysis Result\n\n")
                    f.write(response_text)

                print(f"✅ OpenAI analysis completed for {file_path.name}")
                return {
                    "module": module_name,
                    "file": file_path.name,
                    "file_path": str(file_path),
                    "response": response_text,
                    "content_length": len(content),
                    "timestamp": datetime.now().isoformat(),
                    "model": "gpt-4",
                }
            else:
                print(f"❌ No response received for {file_path.name}")
                return None

        except Exception as e:
            print(f"❌ Error analyzing {file_path.name} with OpenAI: {e}")
            return None


class PLSQLAnalyzerClaude:
    def __init__(self, top_10_dir="top_10", output_dir="plsql_analysis_claude"):
        self.top_10_dir = Path(top_10_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "raw_responses").mkdir(exist_ok=True)

        # Note: You'll need to install anthropic: pip install anthropic
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            print("❌ anthropic package not installed. Run: pip install anthropic")
            raise

    def analyze_file_with_claude(self, file_path, module_name, content):
        """Analyze using Claude Sonnet"""
        print(f"📝 Analyzing {module_name}/{file_path.name} with Claude")

        prompt = f"""Analyze the following PL/SQL file and extract procedure summaries.

TASK: Pick 50 procedures (diversity and complexity). Write a 2-3 sentence summary about their purpose and what they actually do (do not write implementation specifics, but rather "the idea"). These will be used for fine-tuning sparse search engine embeddings, so keep that in mind. Write each summary in its own block and include the name and parameters of each procedure.

FORMAT:
## PROCEDURE_NAME(parameters)
Summary explaining what this procedure conceptually does...

MODULE: {module_name}
FILE: {file_path.name}

PL/SQL CODE:
{content}"""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Save raw response
            response_file = (
                self.output_dir
                / "raw_responses"
                / f"{module_name}_{file_path.stem}_claude.md"
            )
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(f"# Claude Analysis of {module_name}/{file_path.name}\n\n")
                f.write(f"File: {file_path}\n")
                f.write(f"Module: {module_name}\n")
                f.write(f"Model: claude-3-sonnet-20240229\n")
                f.write(f"Analyzed: {datetime.now().isoformat()}\n\n")
                f.write("## Analysis Result\n\n")
                f.write(response_text)

            return {
                "module": module_name,
                "file": file_path.name,
                "file_path": str(file_path),
                "response": response_text,
                "timestamp": datetime.now().isoformat(),
                "model": "claude-3-sonnet",
            }

        except Exception as e:
            print(f"❌ Error analyzing {file_path.name} with Claude: {e}")
            return None


def test_api_availability():
    """Test which APIs are available"""
    available_apis = []

    # Test OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai

            available_apis.append("openai")
        except ImportError:
            print("⚠️  OpenAI package not installed")
    else:
        print("⚠️  OPENAI_API_KEY not set")

    # Test Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            available_apis.append("anthropic")
        except ImportError:
            print("⚠️  Anthropic package not installed")
    else:
        print("⚠️  ANTHROPIC_API_KEY not set")

    return available_apis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test alternative API implementations")
    parser.add_argument(
        "--api",
        choices=["openai", "claude", "test"],
        default="test",
        help="Which API to use (test shows available options)",
    )
    parser.add_argument("--file", help="Test with a single file")

    args = parser.parse_args()

    if args.api == "test":
        print("🧪 Testing API availability...")
        available = test_api_availability()
        print(f"Available APIs: {available}")

        if "openai" in available:
            print("✅ OpenAI API ready")
        if "anthropic" in available:
            print("✅ Claude API ready")

        if not available:
            print("❌ No APIs available. Please set API keys:")
            print("   export OPENAI_API_KEY='your-key'")
            print("   export ANTHROPIC_API_KEY='your-key'")

    elif args.file and os.path.exists(args.file):
        # Test with a single file
        file_path = Path(args.file)

        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()[:50000]  # Truncate for testing

        if args.api == "openai":
            analyzer = PLSQLAnalyzerOpenAI()
            result = analyzer.analyze_file_with_openai(file_path, "test", content)
        elif args.api == "claude":
            analyzer = PLSQLAnalyzerClaude()
            result = analyzer.analyze_file_with_claude(file_path, "test", content)

        if result:
            print(f"✅ Analysis successful: {len(result['response'])} characters")
