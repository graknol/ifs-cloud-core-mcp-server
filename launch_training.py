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
    print("🚀 Starting IFS Cloud Supervised Training Loop")
    print("=" * 50)

    # Default configuration
    config = {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "batch_size": 10,
        "save_dir": "./training_checkpoints",
    }

    print(f"Model: {config['model_name']}")
    print(f"IFS Source: {config['ifs_source_path']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Save Directory: {config['save_dir']}")
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
