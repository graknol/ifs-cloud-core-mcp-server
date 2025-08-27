#!/usr/bin/env python3
"""
Setup and run script for Phi-4 fine-tuning with Unsloth.
Handles dependency installation and model fine-tuning.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_gpu():
    """Check GPU availability and VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🔧 GPU Detected: {gpu_name}")
            print(f"🔧 GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 12:
                print("⚠️  Warning: Less than 12GB VRAM detected. Consider using smaller batch sizes.")
            elif gpu_memory >= 16:
                print("✅ Sufficient VRAM for optimal training (16GB+)")
            else:
                print("✅ Sufficient VRAM for training (12-16GB)")
            
            return True
        else:
            print("❌ No CUDA GPU detected. Fine-tuning will be very slow on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed. Will install with dependencies.")
        return None


def install_dependencies():
    """Install required dependencies for fine-tuning."""
    print("📦 Installing fine-tuning dependencies...")
    print("   This may take several minutes...")
    
    # Install PyTorch with CUDA support first
    print("\n🔥 Installing PyTorch with CUDA support...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], check=True)
    
    # Install Unsloth
    print("\n🦥 Installing Unsloth...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "unsloth[colab-new]", "@", "git+https://github.com/unslothai/unsloth.git"
    ], check=True)
    
    # Install other dependencies
    print("\n📚 Installing additional dependencies...")
    dependencies = [
        "transformers>=4.36.0",
        "datasets>=2.14.0", 
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "scipy",
        "scikit-learn",
        "matplotlib"
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
    
    print("✅ Dependencies installed successfully!")


def verify_dataset():
    """Verify the clean dataset exists."""
    dataset_path = Path("plsql_analysis_combined/clean.jsonl")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please run the dataset cleaning script first:")
        print("   python clean_dataset.py")
        return False
    
    # Count entries
    with open(dataset_path, 'r', encoding='utf-8') as f:
        count = sum(1 for line in f)
    
    print(f"✅ Dataset verified: {count:,} entries found")
    return True


def run_fine_tuning():
    """Execute the fine-tuning script."""
    print("\n🚀 Starting Phi-4 fine-tuning...")
    print("   This process will take several hours depending on your hardware.")
    print("   Monitor GPU memory usage and training progress.")
    
    try:
        subprocess.run([sys.executable, "finetune_phi4.py"], check=True)
        print("✅ Fine-tuning completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fine-tuning failed with error: {e}")
        return False


def main():
    """Main setup and execution function."""
    print("🔥 Phi-4 Mini Fine-tuning Setup for IFS Cloud Procedure Summarization")
    print("=" * 80)
    
    # Check if files exist
    if not Path("finetune_phi4.py").exists():
        print("❌ Fine-tuning script not found: finetune_phi4.py")
        sys.exit(1)
    
    # Verify dataset
    if not verify_dataset():
        sys.exit(1)
    
    # Check GPU
    gpu_available = check_gpu()
    if gpu_available is False:
        response = input("\n⚠️  No GPU detected. Continue with CPU training? (very slow) [y/N]: ")
        if response.lower() != 'y':
            print("Exiting. Please ensure CUDA is properly installed.")
            sys.exit(1)
    
    # Install dependencies
    try:
        # Quick check if unsloth is already installed
        import unsloth
        print("✅ Unsloth already installed")
    except ImportError:
        install_deps = input("\n📦 Install required dependencies? [Y/n]: ")
        if install_deps.lower() != 'n':
            try:
                install_dependencies()
            except subprocess.CalledProcessError as e:
                print(f"❌ Dependency installation failed: {e}")
                print("   Please install dependencies manually:")
                print("   pip install -r requirements_finetune.txt")
                sys.exit(1)
    
    # Confirm before starting training
    print(f"\n📋 Training Configuration Summary:")
    print(f"   • Model: Phi-4 Mini Instruct") 
    print(f"   • Quantization: 4-bit QLoRA")
    print(f"   • Max sequence length: 2048 tokens")
    print(f"   • Batch size: 1 (effective: 8 with gradient accumulation)")
    print(f"   • Epochs: 3")
    print(f"   • Learning rate: 2e-4")
    print(f"   • Expected training time: 2-6 hours (depending on hardware)")
    
    start_training = input(f"\n🚀 Start fine-tuning? [Y/n]: ")
    if start_training.lower() != 'n':
        success = run_fine_tuning()
        
        if success:
            print(f"\n🎉 Fine-tuning completed!")
            print(f"   • Model saved to: ./phi4_ifs_cloud_final")
            print(f"   • Test inference with: python inference_phi4.py")
            print(f"   • Use the model for procedure summarization in your IFS Cloud analysis")
        else:
            print(f"\n❌ Fine-tuning failed. Check the logs above for details.")
    else:
        print("Fine-tuning cancelled.")


if __name__ == "__main__":
    main()
