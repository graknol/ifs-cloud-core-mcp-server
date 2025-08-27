#!/usr/bin/env python3
"""
Setup script for Axolotl fine-tuning environment.
Installs Axolotl and its dependencies with Flash Attention 2 support.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("🔍 Checking CUDA availability...")
    
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
        else:
            print("⚠️  No NVIDIA GPU detected - will install CPU version")
            return False
    except:
        print("⚠️  nvidia-smi not found - assuming no CUDA")
        return False


def install_axolotl(cuda_available=True):
    """Install Axolotl with appropriate dependencies."""
    
    print("\n🚀 Installing Axolotl")
    print("=" * 30)
    
    # Base requirements
    base_packages = [
        "pyyaml",
        "tensorboard", 
        "wandb",
        "datasets",
        "evaluate",
        "scipy",
        "einops",
        "peft",
        "bitsandbytes",
    ]
    
    # Install base packages first
    for package in base_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Install Axolotl
    if cuda_available:
        print("🎯 Installing Axolotl with CUDA support...")
        axolotl_cmd = 'pip install "axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl"'
    else:
        print("🎯 Installing Axolotl without CUDA...")
        axolotl_cmd = 'pip install "axolotl @ git+https://github.com/OpenAccess-AI-Collective/axolotl"'
    
    if not run_command(axolotl_cmd, "Installing Axolotl"):
        print("❌ Failed to install Axolotl from git, trying PyPI...")
        run_command("pip install axolotl", "Installing Axolotl from PyPI")


def create_axolotl_requirements():
    """Create requirements file for Axolotl setup."""
    
    requirements_content = '''# Axolotl Fine-tuning Requirements
# Core Axolotl dependencies
axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl

# Training essentials
torch>=2.8.0
transformers>=4.44.0
datasets>=2.14.0
tokenizers>=0.15.0
accelerate>=0.20.0
peft>=0.8.0
bitsandbytes>=0.41.0

# Flash Attention (already installed)
# flash-attn>=2.6.3  # We already have this from previous setup

# Optimization
deepspeed>=0.12.0
ninja>=1.10.0
scipy>=1.11.0
einops>=0.7.0

# Monitoring and logging  
wandb>=0.15.0
tensorboard>=2.14.0
mlflow>=2.7.0

# Data processing
pyyaml>=6.0
evaluate>=0.4.0
scikit-learn>=1.3.0

# Optional but recommended
xformers>=0.0.22  # Alternative attention implementation
'''
    
    requirements_file = Path("requirements_axolotl.txt")
    with open(requirements_file, 'w') as f:
        f.write(requirements_content)
    
    print(f"📝 Created requirements file: {requirements_file}")
    return requirements_file


def setup_environment():
    """Set up environment variables and configuration."""
    
    print("\n⚙️  Setting up environment")
    print("=" * 30)
    
    # Create environment setup script
    env_script_content = '''#!/bin/bash
# Environment setup for Axolotl training on RTX 5070 Ti

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1

# Tokenizer optimization
export TOKENIZERS_PARALLELISM=false

# Distributed training (even for single GPU - helps with memory)
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0

# Flash Attention (we have Flash Attention 2)
export FLASH_ATTENTION=1

# Axolotl specific
export AXOLOTL_CONFIG_PATH="./axolotl_data/axolotl_config.yml"

echo "✅ Environment configured for Axolotl training"
echo "🎯 CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "🚀 Flash Attention: $FLASH_ATTENTION"
'''
    
    env_script = Path("setup_axolotl_env.sh")
    with open(env_script, 'w') as f:
        f.write(env_script_content)
    
    env_script.chmod(0o755)
    print(f"🔧 Created environment script: {env_script}")


def verify_installation():
    """Verify Axolotl installation."""
    
    print("\n🔍 Verifying installation")
    print("=" * 30)
    
    # Test imports
    test_imports = [
        "axolotl",
        "transformers",
        "datasets", 
        "peft",
        "bitsandbytes",
        "flash_attn"  # We already have this
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Some imports failed: {failed_imports}")
        print("   You may need to install these manually")
    else:
        print("\n🎉 All imports successful!")
    
    # Test Axolotl command
    try:
        result = subprocess.run("axolotl --help", shell=True, capture_output=True)
        if result.returncode == 0:
            print("✅ Axolotl CLI available")
        else:
            print("⚠️  Axolotl CLI not found in PATH")
    except:
        print("⚠️  Could not test Axolotl CLI")


def create_quick_start_guide():
    """Create a quick start guide."""
    
    guide_content = '''# IFS Cloud Axolotl Fine-tuning Quick Start

## 🚀 Getting Started

1. **Activate Environment**:
   ```bash
   source setup_axolotl_env.sh
   ```

2. **Convert Dataset**:
   ```bash
   python convert_to_axolotl.py
   ```

3. **Start Training**:
   ```bash
   cd axolotl_data
   ./train_with_axolotl.sh
   ```

## 📊 Monitoring Training

- **TensorBoard**: `tensorboard --logdir outputs/`
- **Weights & Biases**: Check your W&B dashboard
- **Console logs**: Watch for training progress

## ⚙️ Configuration

The Axolotl config (`axolotl_config.yml`) is optimized for:
- RTX 5070 Ti (16GB VRAM)
- Flash Attention 2
- LoRA fine-tuning
- Conversation format training

## 🎯 Key Features

- **Memory Efficient**: LoRA + gradient checkpointing
- **Fast Training**: Flash Attention 2 + SDPA
- **Professional**: Advanced logging and metrics
- **Flexible**: Easy to modify for different models

## 🔧 Customization

Edit `axolotl_config.yml` to:
- Change model (`base_model`)
- Adjust batch size (`micro_batch_size`)
- Modify LoRA parameters (`lora_r`, `lora_alpha`)
- Set training epochs (`num_epochs`)

## 📈 Expected Performance

With RTX 5070 Ti:
- Training speed: ~2-3 samples/sec
- Memory usage: ~12-14GB VRAM
- Training time: ~2-4 hours for 3 epochs

## 🆘 Troubleshooting

- **OOM Error**: Reduce `micro_batch_size` to 1
- **Slow Training**: Check Flash Attention is enabled
- **Import Errors**: Run `pip install -r requirements_axolotl.txt`
'''
    
    guide_file = Path("AXOLOTL_QUICKSTART.md")
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"📖 Created quick start guide: {guide_file}")


def main():
    """Main setup function."""
    
    print("🔧 IFS Cloud Axolotl Setup")
    print("=" * 40)
    print("Setting up Axolotl for advanced fine-tuning")
    print("Optimized for RTX 5070 Ti with Flash Attention 2")
    print()
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Create requirements file
    requirements_file = create_axolotl_requirements()
    
    # Install Axolotl
    install_axolotl(cuda_available)
    
    # Setup environment
    setup_environment()
    
    # Verify installation
    verify_installation()
    
    # Create quick start guide
    create_quick_start_guide()
    
    print("\n🎉 Axolotl Setup Complete!")
    print("=" * 40)
    print("Next steps:")
    print("1. Run: python convert_to_axolotl.py")
    print("2. Review: axolotl_data/axolotl_config.yml") 
    print("3. Start training: cd axolotl_data && ./train_with_axolotl.sh")
    print("4. Check: AXOLOTL_QUICKSTART.md for details")


if __name__ == "__main__":
    main()
