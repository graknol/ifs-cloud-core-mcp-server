#!/usr/bin/env python3
"""
Simple launcher for the supervised training loop.
This script provides an easy way to start the training process.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from supervised_training_loop import SupervisedTrainingLoop


def main():
    # Set CUDA memory management environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error reporting

    print("🚀 Starting IFS Cloud Supervised Training Loop")
    print("=" * 50)

    # Configuration for two-phase approach:
    # Phase 1: Generate all summaries with larger model
    # Phase 2: Fine-tune smaller model with multiple epochs + augmentation
    config = {
        "summary_model_name": "unsloth/Qwen2.5-7B-Instruct",  # Large model for quality generation
        "training_model_name": "unsloth/Qwen2.5-1.5B-Instruct",  # Small model for fine-tuning
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "batch_size": 8,  # Can be larger for generation phase
        "max_length": 4096,  # Generous context for detailed procedures
        "save_dir": "./training_checkpoints",
        "target_summaries": 200,  # Generate all 200 summaries first
        "training_epochs": 15,  # Multiple epochs for limited dataset
        "data_augmentation": True,  # Enable randomization strategies
        "two_phase_training": True,  # Enable two-phase approach
    }

    print(f"📊 Two-Phase Training Strategy:")
    print(f"  Phase 1 - Summary Generation: {config['summary_model_name']}")
    print(f"  Phase 2 - Fine-tuning: {config['training_model_name']}")
    print(f"🎯 Target: {config['target_summaries']} summaries")
    print(f"🔄 Training Epochs: {config['training_epochs']}")
    print(f"🎲 Data Augmentation: {config['data_augmentation']}")
    print(f"📁 IFS Source: {config['ifs_source_path']}")
    print(f"💾 Save Directory: {config['save_dir']}")
    print()

    # Check if IFS source exists
    if not Path(config["ifs_source_path"]).exists():
        print("⚠️  IFS source path not found. Please update the path in the script or:")
        print("   - Ensure IFS source is available at the configured location")
        print("   - Update the ifs_source_path in this script")
        return

    try:
        # Create training loop
        training_loop = SupervisedTrainingLoop(**config)

        print("✅ Training loop initialized successfully")
        print("📋 Starting procedure extraction...")

        # Run the training loop
        training_loop.run_training_loop()

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
