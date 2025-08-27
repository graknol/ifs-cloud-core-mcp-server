#!/usr/bin/env python3
"""
Convert the Alpaca JSON dataset to JSONL format.
JSONL is more memory-efficient for large datasets and easier to stream during training.
"""

import json
from pathlib import Path


def convert_json_to_jsonl(input_file: str, output_file: str) -> None:
    """Convert JSON array to JSONL format."""
    
    print(f"🔄 Converting {input_file} to JSONL format")
    print(f"📁 Output: {output_file}")
    
    # Load JSON data
    print("📖 Loading JSON data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data):,} examples")
    
    # Write to JSONL format
    print("💾 Writing JSONL format...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(data):
            if i > 0 and i % 10000 == 0:
                print(f"   Written {i:,} examples...")
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully converted to JSONL format!")
    print(f"📊 {len(data):,} examples written to {output_file}")
    
    # Show file sizes
    input_size = Path(input_file).stat().st_size / (1024*1024)  # MB
    output_size = Path(output_file).stat().st_size / (1024*1024)  # MB
    
    print(f"📏 File sizes:")
    print(f"   JSON:  {input_size:.1f} MB")
    print(f"   JSONL: {output_size:.1f} MB")


def main():
    """Main function."""
    
    print("🔄 JSON → JSONL Conversion")
    print("=" * 30)
    
    # Set paths
    input_file = "ifs_cloud_alpaca_dataset.json"
    output_file = "ifs_cloud_alpaca_dataset.jsonl"
    
    # Check if input exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Convert
    convert_json_to_jsonl(input_file, output_file)
    
    print(f"\n🎉 Conversion complete!")
    print(f"📁 JSONL dataset: {output_file}")
    print(f"🚀 Ready for efficient streaming during training!")


if __name__ == "__main__":
    main()
