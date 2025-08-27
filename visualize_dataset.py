#!/usr/bin/env python3
"""
Dataset Visualization Script for IFS Cloud Alpaca Dataset
Creates word clouds and distribution analysis to identify potential biases or skews.
"""

import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter, defaultdict
import re
import numpy as np
import seaborn as sns
from pathlib import Path

def load_dataset(file_path):
    """Load the Alpaca dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_instructions(data):
    """Analyze instruction patterns and create word cloud."""
    instructions = [example['instruction'] for example in data]
    
    # Clean instructions for word cloud
    instruction_text = ' '.join(instructions)
    # Remove common stop words but keep important terms
    stop_words = {'this', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about'}
    
    # Create word cloud
    wordcloud = WordCloud(
        width=1200, height=800, 
        background_color='white',
        max_words=100,
        stopwords=stop_words,
        colormap='viridis'
    ).generate(instruction_text)
    
    # Plot
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Instruction Words Distribution', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_instructions_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return instructions

def analyze_outputs(data):
    """Analyze output patterns and create word cloud."""
    outputs = [example['output'] for example in data]
    
    # Clean outputs for word cloud
    output_text = ' '.join(outputs)
    # Remove common stop words
    stop_words = {'this', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    # Create word cloud
    wordcloud = WordCloud(
        width=1200, height=800,
        background_color='white', 
        max_words=150,
        stopwords=stop_words,
        colormap='plasma'
    ).generate(output_text)
    
    # Plot
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Output Content Distribution', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_outputs_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return outputs

def analyze_technical_terms(data):
    """Analyze technical terms and IFS Cloud specific vocabulary."""
    # Extract all text
    all_text = []
    for example in data:
        all_text.extend([example['instruction'], example['input'], example['output']])
    
    combined_text = ' '.join(all_text)
    
    # Focus on technical terms
    technical_pattern = r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b|\b[a-z]+_[a-z]+\b|\b[A-Z]{2,}\b'
    technical_terms = re.findall(technical_pattern, combined_text)
    
    # Filter for likely technical terms (procedural, database, business terms)
    tech_counter = Counter(technical_terms)
    
    # Create technical terms word cloud
    tech_text = ' '.join([term for term, count in tech_counter.most_common(200)])
    
    wordcloud = WordCloud(
        width=1200, height=800,
        background_color='navy',
        max_words=100,
        colormap='cool',
        collocations=False
    ).generate(tech_text)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Technical Terms & IFS Cloud Vocabulary', fontsize=20, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('dataset_technical_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tech_counter

def analyze_instruction_distribution(instructions):
    """Analyze instruction type distribution."""
    # Categorize instructions by first word
    first_words = [inst.split()[0].lower() for inst in instructions if inst.split()]
    first_word_counts = Counter(first_words)
    
    # Create distribution plot
    plt.figure(figsize=(12, 8))
    top_words = dict(first_word_counts.most_common(15))
    plt.bar(top_words.keys(), top_words.values(), color='skyblue', edgecolor='navy')
    plt.title('Distribution of Instruction Types (First Word)', fontsize=16, fontweight='bold')
    plt.xlabel('Instruction Start Word', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (word, count) in enumerate(top_words.items()):
        plt.text(i, count + 1000, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('instruction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return first_word_counts

def analyze_modules_and_files(data):
    """Analyze module and file distribution in inputs."""
    modules = []
    files = []
    
    for example in data:
        input_text = example['input']
        
        # Extract module
        module_match = re.search(r'Module:\s*(\w+)', input_text)
        if module_match:
            modules.append(module_match.group(1).lower())
        
        # Extract file
        file_match = re.search(r'File:\s*([^\\n]+)', input_text)
        if file_match:
            file_name = file_match.group(1).strip()
            # Get just the base name without extension
            base_name = Path(file_name).stem
            files.append(base_name.lower())
    
    # Module distribution
    module_counts = Counter(modules)
    plt.figure(figsize=(14, 8))
    top_modules = dict(module_counts.most_common(20))
    plt.bar(range(len(top_modules)), list(top_modules.values()), color='lightcoral', edgecolor='darkred')
    plt.title('Distribution of IFS Cloud Modules', fontsize=16, fontweight='bold')
    plt.xlabel('Module', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(top_modules)), list(top_modules.keys()), rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('module_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # File types word cloud
    file_text = ' '.join(files[:1000])  # Sample to avoid overwhelming
    if file_text:
        wordcloud = WordCloud(
            width=1200, height=800,
            background_color='white',
            max_words=80,
            colormap='Set3'
        ).generate(file_text)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('File Names Distribution', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig('files_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return module_counts, Counter(files)

def analyze_dataset_balance(data):
    """Analyze overall dataset balance and potential skews."""
    
    # Instruction length distribution
    instruction_lengths = [len(ex['instruction'].split()) for ex in data]
    
    plt.figure(figsize=(12, 6))
    plt.hist(instruction_lengths, bins=30, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    plt.title('Distribution of Instruction Lengths', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.axvline(np.mean(instruction_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(instruction_lengths):.1f} words')
    plt.legend()
    plt.tight_layout()
    plt.savefig('instruction_lengths.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output length distribution  
    output_lengths = [len(ex['output'].split()) for ex in data]
    
    plt.figure(figsize=(12, 6))
    plt.hist(output_lengths, bins=50, color='lightblue', edgecolor='darkblue', alpha=0.7)
    plt.title('Distribution of Output Lengths', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.axvline(np.mean(output_lengths), color='red', linestyle='--',
                label=f'Mean: {np.mean(output_lengths):.1f} words')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_lengths.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'instruction_lengths': instruction_lengths,
        'output_lengths': output_lengths,
        'mean_instruction_length': np.mean(instruction_lengths),
        'mean_output_length': np.mean(output_lengths)
    }

def main():
    """Main visualization function."""
    print("🎨 DATASET VISUALIZATION AND ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    data = load_dataset('ifs_cloud_alpaca_dataset.json')
    print(f"📊 Loaded {len(data):,} examples")
    
    # 1. Analyze instructions
    print("\\n📝 Analyzing instructions...")
    instructions = analyze_instructions(data)
    
    # 2. Analyze outputs  
    print("\\n📄 Analyzing outputs...")
    outputs = analyze_outputs(data)
    
    # 3. Analyze technical terms
    print("\\n🔧 Analyzing technical terms...")
    tech_terms = analyze_technical_terms(data)
    
    # 4. Analyze instruction distribution
    print("\\n📊 Analyzing instruction distribution...")
    inst_dist = analyze_instruction_distribution(instructions)
    
    # 5. Analyze modules and files
    print("\\n🗂️  Analyzing modules and files...")
    modules, files = analyze_modules_and_files(data)
    
    # 6. Analyze dataset balance
    print("\\n⚖️  Analyzing dataset balance...")
    balance_stats = analyze_dataset_balance(data)
    
    # Print summary statistics
    print("\\n📈 SUMMARY STATISTICS:")
    print(f"  Total examples: {len(data):,}")
    print(f"  Unique modules: {len(modules)}")
    print(f"  Most common module: {modules.most_common(1)[0] if modules else 'N/A'}")
    print(f"  Top instruction types: {list(inst_dist.most_common(3))}")
    print(f"  Mean instruction length: {balance_stats['mean_instruction_length']:.1f} words")
    print(f"  Mean output length: {balance_stats['mean_output_length']:.1f} words")
    print(f"  Top technical terms: {list(tech_terms.most_common(5))}")
    
    print("\\n✅ Visualization complete! Check the generated PNG files:")
    print("  - dataset_instructions_wordcloud.png")
    print("  - dataset_outputs_wordcloud.png") 
    print("  - dataset_technical_wordcloud.png")
    print("  - instruction_distribution.png")
    print("  - module_distribution.png")
    print("  - files_wordcloud.png")
    print("  - instruction_lengths.png")
    print("  - output_lengths.png")

if __name__ == "__main__":
    main()
