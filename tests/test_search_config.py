#!/usr/bin/env python3
"""Test the SearchConfig implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.hybrid_search import SearchConfig


def test_search_config():
    """Test SearchConfig presets."""

    print("🧪 Testing SearchConfig implementation")

    # Test hardware presets
    fast_config = SearchConfig.fast_hardware()
    print(f"✅ Fast hardware config: {fast_config}")
    assert fast_config.enable_faiss == True
    assert fast_config.enable_flashrank == True
    assert fast_config.fetch_multiplier == 3
    assert fast_config.preset_name == "fast_hardware"

    medium_config = SearchConfig.medium_hardware()
    print(f"✅ Medium hardware config: {medium_config}")
    assert medium_config.enable_faiss == True
    assert medium_config.enable_flashrank == False
    assert medium_config.fetch_multiplier == 2
    assert medium_config.preset_name == "medium_hardware"

    slow_config = SearchConfig.slow_hardware()
    print(f"✅ Slow hardware config: {slow_config}")
    assert slow_config.enable_faiss == False
    assert slow_config.enable_flashrank == False
    assert slow_config.fetch_multiplier == 1
    assert slow_config.preset_name == "slow_hardware"

    # Test custom config
    custom_config = SearchConfig(
        enable_faiss=True,
        enable_flashrank=True,
        fetch_multiplier=5,
        min_score_threshold=0.8,
        preset_name="custom_high_precision",
    )
    print(f"✅ Custom config: {custom_config}")
    assert custom_config.min_score_threshold == 0.8
    assert custom_config.preset_name == "custom_high_precision"

    print("🎉 All SearchConfig tests passed!")


if __name__ == "__main__":
    test_search_config()
