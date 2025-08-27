#!/usr/bin/env python3
"""
Lightweight test script to verify training data format without heavy ML dependencies.
"""

import json
import random
from pathlib import Path


def load_clean_dataset(dataset_path):
    """Load the clean dataset from JSONL file."""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_summarization_prompts():
    """Get diverse summarization prompt templates."""
    return [
        "Summarize this procedure:",
        "Provide a summary of this procedure:",
        "Give a brief overview of this procedure:",
        "Describe what this procedure does:",
        "Explain the purpose of this procedure:",
        "What does this procedure accomplish?",
        "Briefly describe this procedure's functionality:",
        "Provide a concise explanation of this procedure:",
        "Outline what this procedure performs:",
        "Give a short description of this procedure:"
    ]


def get_question_generation_prompts():
    """Get diverse question generation prompt templates."""
    return [
        "Write a question that is answered by the following procedure:",
        "Generate a question that this procedure addresses:",
        "What question does this procedure answer?",
        "Create a question for which this procedure is the solution:",
        "What problem does this procedure solve? Write as a question:",
        "Formulate a question that this procedure responds to:",
        "What inquiry would lead to this procedure being used?",
        "Generate a relevant question for this procedure:",
        "What question would this procedure help answer?",
        "Create an appropriate question for this procedure:"
    ]


def convert_to_training_format(data):
    """Convert dataset to training format with diverse prompts."""
    training_examples = []
    summarization_prompts = get_summarization_prompts()
    question_prompts = get_question_generation_prompts()
    
    for entry in data:
        # Create input with standardized format
        input_text = f"Module: {entry['module']}\nFile: {entry['file']}\nSignature: {entry['signature']}"
        
        # Generate summarization examples
        for prompt_template in summarization_prompts:
            training_examples.append({
                'input': f"{prompt_template}\n{input_text}",
                'output': entry['summary'],
                'task_type': 'summarization'
            })
        
        # Generate question examples using synthetic queries
        if entry.get('synthetic_queries'):
            queries = entry['synthetic_queries'][:2]  # Use first 2 queries
            for i, query in enumerate(queries):
                if i < len(question_prompts):
                    training_examples.append({
                        'input': f"{question_prompts[i]}\n{input_text}",
                        'output': query,
                        'task_type': 'question_generation'
                    })
    
    return training_examples


def format_phi4_chat_template(example):
    """Format example using Phi-4 chat template."""
    system_message = "You are a helpful assistant that analyzes PL/SQL procedures and provides accurate information."
    
    formatted_text = f"<|system|>{system_message}<|end|><|user|>{example['input']}<|end|><|assistant|>{example['output']}<|end|>"
    
    return {
        'text': formatted_text,
        'input': example['input'],
        'output': example['output'],
        'task_type': example['task_type']
    }


def test_lightweight():
    """Test without heavy dependencies."""
    print("🧪 Lightweight Training Data Test")
    print("=" * 40)
    
    # Check if dataset exists
    dataset_path = "plsql_analysis_combined/clean.jsonl"
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print("📖 Loading sample data...")
    data = load_clean_dataset(dataset_path)
    sample_data = data[:3]  # Use first 3 entries
    
    print(f"✅ Loaded {len(sample_data)} sample entries")
    
    print("\n🔄 Converting to training format...")
    training_examples = convert_to_training_format(sample_data)
    
    print(f"✅ Generated {len(training_examples)} training examples")
    
    # Count by task type
    summarization_count = len([ex for ex in training_examples if ex['task_type'] == 'summarization'])
    question_count = len([ex for ex in training_examples if ex['task_type'] == 'question_generation'])
    
    print(f"   • Summarization: {summarization_count}")
    print(f"   • Question generation: {question_count}")
    
    print("\n📝 Sample Examples:")
    print("=" * 25)
    
    # Show samples of each type
    for task_type in ['summarization', 'question_generation']:
        examples = [ex for ex in training_examples if ex['task_type'] == task_type]
        if examples:
            sample = examples[0]
            formatted = format_phi4_chat_template(sample)
            
            print(f"\n🔹 {task_type.title()}:")
            print("Input preview:", sample['input'][:100] + "...")
            print("Output preview:", sample['output'][:100] + "...")
            
            # Check Phi-4 format
            text = formatted['text']
            tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
            token_status = [token in text for token in tokens]
            
            print("Format check:", "✅ Valid" if all(token_status) else "❌ Invalid")
            print("Length:", len(text), "characters")
    
    # Dataset integrity check
    print(f"\n🔍 Dataset Integrity (first 10 entries):")
    print("=" * 30)
    
    required_fields = ['procedure_name', 'signature', 'summary', 'module', 'file', 'synthetic_queries']
    valid_count = 0
    
    for entry in data[:10]:
        valid = all(field in entry and entry[field] for field in required_fields)
        if valid and entry.get('synthetic_queries') and len(entry['synthetic_queries']) >= 2:
            valid_count += 1
    
    print(f"✅ Valid entries: {valid_count}/10")
    
    if valid_count >= 8:
        print("🎯 Dataset looks good for fine-tuning!")
    else:
        print("⚠️  Dataset may need cleaning")


if __name__ == "__main__":
    test_lightweight()
