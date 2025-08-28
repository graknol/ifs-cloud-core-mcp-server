#!/usr/bin/env python3
"""
Repository cleanup script - removes all training, dataset, and testing files
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_repository():
    """Remove all training, dataset, and testing related files."""
    
    print("🧹 REPOSITORY CLEANUP - REMOVING TRAINING/DATASET FILES")
    print("=" * 60)
    
    # Files to remove (exact matches)
    files_to_remove = [
        # Dataset files
        "ifs_cloud_alpaca_dataset.json",
        "ifs_cloud_alpaca_dataset.jsonl", 
        
        # Visualization files
        "dataset_instructions_wordcloud.png",
        "dataset_outputs_wordcloud.png",
        "dataset_technical_wordcloud.png",
        "files_wordcloud.png",
        "instruction_distribution.png",
        "instruction_lengths.png",
        "module_distribution.png",
        "output_lengths.png",
        
        # Training configs
        "axolotl_config.yml",
        "finetune_config.ini",
        "requirements_finetune.txt",
        "train_axolotl.sh",
        
        # CSV files
        "final_optimizer_keywords.csv",
        "final_optimizer_keywords_original.csv",
        
        # Dataset generation scripts
        "analyze_clean.py",
        "analyze_combined.py", 
        "analyze_plsql_procedures.py",
        "analyze_token_counts.py",
        "clean_dataset.py",
        "clean_keywords.py",
        "clean_synthetic_questions.py",
        "combine_summaries.py",
        "convert_json_to_jsonl.py",
        "convert_to_alpaca.py",
        "convert_to_alpaca_fixed.py", 
        "convert_to_axolotl.py",
        "correlate_summaries_with_prompts.py",
        "extract_top_10_per_module.py",
        "filter_ellipsis_entries.py",
        "fix_comprehensive_corruption.py",
        "fix_corrupted_data.py",
        "fix_final_corruption.py",
        "fix_final_json.py",
        "visualize_dataset.py",
        
        # Generation scripts
        "generate_batch_summaries.py",
        "generate_clean_simple.py", 
        "generate_direct_simple.py",
        "generate_diversified_batch.py",
        "generate_optimized_summaries.py",
        "generate_simple_optimized.py",
        "generate_simple_summaries.py",
        "generate_smart_optimized.py",
        "generate_synthetic_questions.py",
        
        # Training scripts
        "finetune_phi4.py",
        "inference_phi4.py",
        "launch_finetuning.py",
        "launch_training.py", 
        "setup_axolotl.py",
        "setup_finetune.py",
        "supervised_training_loop.py",
        "training_validator.py",
        
        # GUI review scripts
        "launch_final_gui_review.py",
        "launch_gui_review.py",
        
        # Test scripts
        "test_copilot_api.py",
        "test_data_format.py",
        "test_extraction.py", 
        "test_new_extraction.py",
        "test_training_data.py",
        "test_triton.py",
        
        # Other utilities
        "check_dependencies.py",
        "run_plsql_analysis.py",
        "ifs_parser_integration.py"
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "axolotl_data",
        "batch_planning", 
        "batch_summaries",
        "plsql_analysis_combined",
        "tests"  # Test directory
    ]
    
    # Documentation files to remove (training-related)
    docs_to_remove = [
        "DATASET_VISUALIZATION_REPORT.md",
        "FINETUNE_README.md", 
        "SUPERVISED_TRAINING.md",
        "SUMMARY_CORRELATION_COMPLETE.md",
        "TOKEN_COUNT_ENHANCEMENT_COMPLETE.md",
        "TWO_PHASE_TRAINING_SUMMARY.md"
    ]
    
    removed_count = 0
    
    # Remove files
    print("📁 Removing files...")
    for filename in files_to_remove + docs_to_remove:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"   ✅ Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {filename}: {e}")
        else:
            print(f"   ⚪ Not found: {filename}")
    
    # Remove directories
    print("\\n📂 Removing directories...")
    for dirname in dirs_to_remove:
        if os.path.exists(dirname):
            try:
                shutil.rmtree(dirname)
                print(f"   ✅ Removed directory: {dirname}/")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove directory {dirname}: {e}")
        else:
            print(f"   ⚪ Not found: {dirname}/")
    
    print(f"\\n✅ CLEANUP COMPLETE")
    print(f"   📊 Removed {removed_count} files/directories")
    print(f"   🎯 Repository cleaned - training files moved to separate project")
    
    # Show remaining structure
    print(f"\\n📋 REMAINING REPOSITORY STRUCTURE:")
    remaining_files = []
    for item in os.listdir('.'):
        if not item.startswith('.') and not item.startswith('__'):
            remaining_files.append(item)
    
    for item in sorted(remaining_files)[:20]:  # Show first 20
        if os.path.isdir(item):
            print(f"   📂 {item}/")
        else:
            print(f"   📄 {item}")
    
    if len(remaining_files) > 20:
        print(f"   ... and {len(remaining_files) - 20} more items")

if __name__ == "__main__":
    cleanup_repository()
