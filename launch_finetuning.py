#!/usr/bin/env python3
"""
Fine-tuning Summary and Launch Script
====================================

This script provides a summary of the fine-tuning setup and launches the training process.
"""

import json
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "trl",
        "unsloth"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: python setup_finetune.py")
        return False
    
    print("✅ All dependencies available")
    return True


def check_dataset():
    """Check if dataset is available."""
    print("\n📊 Dataset Status")
    print("=" * 20)
    
    dataset_path = Path("plsql_analysis_combined/clean.jsonl")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Count entries
    with open(dataset_path, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    
    print(f"✅ Dataset found: {count:,} entries")
    
    # Estimate training examples
    estimated_examples = count * 20  # 10 prompts × 2 tasks
    print(f"📈 Estimated training examples: {estimated_examples:,}")
    
    return True


def show_config():
    """Show training configuration."""
    print("\n⚙️  Training Configuration")
    print("=" * 30)
    
    config_path = Path("finetune_config.ini")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Extract key settings
        for line in config_content.split('\n'):
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in ['model_name', 'num_epochs', 'batch_size', 'learning_rate', 'lora_rank']:
                    print(f"• {key}: {value}")
    else:
        print("• Using default configuration")
    
    print("\n🎯 Training Features:")
    print("• Model: Microsoft Phi-4 Mini Instruct")
    print("• Method: QLoRA (4-bit quantization)")
    print("• Tasks: Summarization + Question Generation")
    print("• Prompts: 10 variations per task type")
    print("• Format: Phi-4 chat template")
    print("• Memory: Optimized for 16GB VRAM")


def estimate_training_time():
    """Estimate training time."""
    print("\n⏱️  Training Estimates")
    print("=" * 25)
    
    print("• Setup time: ~5-10 minutes")
    print("• Training time: ~2-4 hours (3 epochs)")
    print("• Total time: ~2.5-4.5 hours")
    print("• GPU memory: ~12-14GB VRAM")
    print("• Disk space: ~10-15GB for model files")


def show_next_steps():
    """Show what to do after training."""
    print("\n🚀 After Training")
    print("=" * 20)
    
    print("1. Test the fine-tuned model:")
    print("   python inference_phi4.py")
    print()
    print("2. Batch processing:")
    print("   python inference_phi4.py --batch procedure_names.txt")
    print()
    print("3. Model files will be saved in:")
    print("   ./phi4_procedure_summarizer/")


def main():
    """Main summary and launch function."""
    print("🔬 Phi-4 Fine-tuning Summary")
    print("=" * 40)
    
    # Check all prerequisites
    deps_ok = check_dependencies()
    dataset_ok = check_dataset()
    
    if not (deps_ok and dataset_ok):
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    show_config()
    estimate_training_time()
    show_next_steps()
    
    print("\n" + "=" * 40)
    print("✅ Ready for Fine-tuning!")
    
    # Ask if user wants to start training
    response = input("\nStart fine-tuning now? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting fine-tuning...")
        subprocess.run([sys.executable, "finetune_phi4.py"])
    else:
        print("\n📝 To start training later, run:")
        print("   python finetune_phi4.py")


if __name__ == "__main__":
    main()
