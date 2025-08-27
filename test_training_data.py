#!/usr/bin/env python3
"""
Test script to verify the training data conversion and format.
Run this before starting fine-tuning to ensure data is properly formatted.
"""

import json
import random
from pathlib import Path
from finetune_phi4 import load_clean_dataset, convert_to_training_format, format_phi4_chat_template


def test_data_conversion():
    """Test the data conversion pipeline."""
    print("🧪 Testing Training Data Conversion")
    print("=" * 50)
    
    # Load a small subset of the dataset
    print("📖 Loading dataset sample...")
    data = load_clean_dataset("plsql_analysis_combined/clean.jsonl")
    
    # Take a small sample for testing
    sample_data = random.sample(data, min(5, len(data)))
    print(f"✅ Using {len(sample_data)} sample entries")
    
    # Convert to training format
    print("\n🔄 Converting to training format...")
    training_examples = convert_to_training_format(sample_data)
    
    print(f"✅ Generated {len(training_examples)} training examples")
    
    # Show statistics
    summarization_count = len([ex for ex in training_examples if ex['task_type'] == 'summarization'])
    question_count = len([ex for ex in training_examples if ex['task_type'] == 'question_generation'])
    
    print(f"   • Summarization examples: {summarization_count}")
    print(f"   • Question generation examples: {question_count}")
    
    # Format and show samples
    print("\n📝 Sample Formatted Training Examples:")
    print("=" * 50)
    
    # Show one of each type
    for task_type in ['summarization', 'question_generation']:
        examples = [ex for ex in training_examples if ex['task_type'] == task_type]
        if examples:
            sample_ex = examples[0]
            formatted = format_phi4_chat_template(sample_ex)
            
            print(f"\n🔹 {task_type.title()} Example:")
            print("Input:", sample_ex['input'][:100] + "..." if len(sample_ex['input']) > 100 else sample_ex['input'])
            print("Output:", sample_ex['output'][:100] + "..." if len(sample_ex['output']) > 100 else sample_ex['output'])
            print("Formatted length:", len(formatted['text']), "characters")
            print("Formatted preview:", formatted['text'][:200] + "..." if len(formatted['text']) > 200 else formatted['text'])
    
    # Verify Phi-4 format
    print("\n🔍 Verifying Phi-4 Format:")
    print("=" * 30)
    
    formatted_example = format_phi4_chat_template(training_examples[0])
    text = formatted_example['text']
    
    required_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    for token in required_tokens:
        if token in text:
            print(f"✅ {token} found")
        else:
            print(f"❌ {token} missing")
    
    # Check format structure
    parts = text.split("<|end|>")
    print(f"✅ Format has {len(parts)} parts (should be 4: system, user, assistant, empty)")
    
    # Show token counts
    print(f"\n📊 Sample Lengths:")
    lengths = [len(format_phi4_chat_template(ex)['text']) for ex in training_examples[:10]]
    print(f"   • Average: {sum(lengths)/len(lengths):.1f} characters")
    print(f"   • Min: {min(lengths)} characters") 
    print(f"   • Max: {max(lengths)} characters")
    
    print("\n✅ Data conversion test completed!")


def verify_dataset_integrity():
    """Verify the clean dataset has the required fields."""
    print("\n🔍 Verifying Dataset Integrity")
    print("=" * 30)
    
    data = load_clean_dataset("plsql_analysis_combined/clean.jsonl")
    
    required_fields = ['procedure_name', 'signature', 'summary', 'module', 'file', 'synthetic_queries']
    
    missing_fields = {}
    valid_entries = 0
    
    for i, entry in enumerate(data[:100]):  # Check first 100 entries
        entry_valid = True
        for field in required_fields:
            if field not in entry or not entry[field]:
                if field not in missing_fields:
                    missing_fields[field] = 0
                missing_fields[field] += 1
                entry_valid = False
        
        if entry_valid:
            valid_entries += 1
    
    print(f"✅ Valid entries: {valid_entries}/100")
    
    if missing_fields:
        print("⚠️  Missing fields detected:")
        for field, count in missing_fields.items():
            print(f"   • {field}: missing in {count} entries")
    else:
        print("✅ All required fields present")
    
    # Check synthetic queries
    entries_with_queries = sum(1 for entry in data[:100] if entry.get('synthetic_queries') and len(entry['synthetic_queries']) >= 2)
    print(f"✅ Entries with 2+ synthetic queries: {entries_with_queries}/100")


if __name__ == "__main__":
    random.seed(42)
    
    verify_dataset_integrity()
    test_data_conversion()
    
    print("\n🎯 Ready for fine-tuning!")
    print("Run: python finetune_phi4.py")
