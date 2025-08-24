# High-Performance Parallel Processing Setup Guide

## Overview

This guide covers setting up the multi-stage parallel processing pipeline for optimal performance when processing 10,000+ PL/SQL files.

## Architecture

### 3-Stage Pipeline Design

```
Stage 1: Parallel AST Parsing (I/O Bound)
├── ThreadPoolExecutor with 50 workers
├── High concurrency for file reading
└── AST parser subprocess calls

Stage 2: Parallel Function Processing (CPU Bound)
├── ProcessPoolExecutor with 8 workers
├── Function quality filtering
├── Smart truncation for UnixCoder
└── Training sample creation

Stage 3: Batch Summarization (Model Bound)
├── vLLM or HuggingFace batching
├── GPU-optimized batch sizes
└── Parallel summary generation
```

## Performance Optimization

### 1. vLLM Setup (Recommended for Best Performance)

vLLM provides 10-20x faster inference than standard transformers:

```bash
# Install vLLM (requires CUDA)
pip install vllm

# Or via UV
uv add vllm
```

**System Requirements:**

- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8 or 12.1
- 32GB+ system RAM for large models

**Configuration:**

```python
# Optimal vLLM settings for batch processing
llm = LLM(
    model="microsoft/unixcoder-base",
    tensor_parallel_size=1,        # Single GPU
    max_model_len=2048,           # UnixCoder context
    gpu_memory_utilization=0.8,   # 80% GPU memory
    swap_space=4,                 # 4GB swap space
    max_num_batched_tokens=8192   # Large batches
)
```

### 2. Alternative: HuggingFace Transformers

For systems without GPU or vLLM support:

```bash
# CPU-optimized transformers
pip install torch transformers --index-url https://download.pytorch.org/whl/cpu

# Or GPU version
pip install torch transformers
```

### 3. System Configuration

**Optimal Hardware:**

- CPU: 8+ cores (16+ threads)
- RAM: 32GB+ for large-scale processing
- Storage: NVMe SSD for fast file I/O
- GPU: 8GB+ VRAM (optional but recommended)

**OS Tuning:**

```bash
# Increase file descriptor limits
ulimit -n 65536

# Optimize for high-concurrency I/O
echo 'net.core.rmem_max = 67108864' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 67108864' >> /etc/sysctl.conf
```

## Usage Examples

### 1. Basic Parallel Processing

```bash
# Run with defaults (10,000 samples)
python parallel_extraction_pipeline.py

# Specify source directory and sample count
python parallel_extraction_pipeline.py --source ../ifs-cloud --samples 5000

# Test run with small sample
python parallel_extraction_pipeline.py --test
```

### 2. With AST Parser

```bash
# Ensure AST parser executable is available
./plsql_parser.exe --help

# Run with AST parser
python parallel_extraction_pipeline.py --parser ./plsql_parser.exe
```

### 3. With vLLM Acceleration

```bash
# Use vLLM for fastest summarization
python parallel_extraction_pipeline.py --vllm

# Combined: AST parser + vLLM
python parallel_extraction_pipeline.py --parser ./plsql_parser.exe --vllm
```

### 4. Claude Summarization (Highest Quality)

```bash
# Extract with UnixCoder, then post-process with Claude
python parallel_extraction_pipeline.py
python parallel_extraction_pipeline.py --claude

# Or do both in one command
python parallel_extraction_pipeline.py --claude
```

## Performance Benchmarks

### Expected Throughput (10,000 files)

| Configuration | Time | Files/sec | Functions/sec |
| ------------- | ---- | --------- | ------------- |
| Regex + CPU   | 45m  | 3.7       | 15-20         |
| AST + CPU     | 25m  | 6.7       | 25-30         |
| AST + vLLM    | 15m  | 11.1      | 40-50         |
| AST + Claude  | 60m  | 2.8       | 10-15         |

### Memory Usage

| Stage           | RAM Usage | GPU Usage |
| --------------- | --------- | --------- |
| Parsing         | 2-4GB     | None      |
| Processing      | 4-8GB     | None      |
| vLLM            | 8-12GB    | 6-8GB     |
| HF Transformers | 6-10GB    | 4-6GB     |

## Optimization Tips

### 1. Worker Configuration

```python
# For I/O bound parsing (Stage 1)
max_parse_workers = min(50, cpu_count() * 4)

# For CPU bound processing (Stage 2)
max_process_workers = cpu_count()

# For GPU bound summarization (Stage 3)
batch_size = 32 if gpu_available else 8
```

### 2. Memory Management

```python
# Monitor memory usage
import psutil
memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
if memory_usage > 16000:  # 16GB
    # Reduce batch sizes or worker counts
    batch_size = batch_size // 2
```

### 3. Error Handling

```python
# Robust error handling for large-scale processing
try:
    stats = await processor.process_files_pipeline(files, scores)
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    # Save partial results before crashing
    save_partial_results()
```

## Monitoring and Debugging

### 1. Real-time Progress

The pipeline provides comprehensive logging:

```
📊 Parsed 1000/10000 files (2500 functions)
⚙️ Processed 800/2500 functions
🧠 Summarized 600/800 samples
```

### 2. Performance Metrics

Check `parallel_pipeline_report.json` for detailed metrics:

```json
{
  "pipeline_performance": {
    "total_time": 900.5,
    "throughput_files_per_sec": 11.1,
    "success_rate": 0.95
  },
  "stage_performance": {
    "parsing_time": 180.2,
    "processing_time": 240.8,
    "summarization_time": 479.5
  }
}
```

### 3. Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce worker counts or batch sizes
2. **GPU OOM**: Lower `gpu_memory_utilization` or batch size
3. **Slow Parsing**: Check if AST parser is hanging on specific files
4. **API Rate Limits**: Reduce Claude batch size or add delays

**Debug Mode:**

```bash
# Enable debug logging
export PYTHONPATH=.
python -m logging --level DEBUG parallel_extraction_pipeline.py
```

## Integration with Existing Pipeline

The parallel processor integrates seamlessly:

```python
# Use existing PageRank scores
from extract_training_samples import load_pagerank_scores
pagerank_scores = load_pagerank_scores("comprehensive_plsql_analysis.json")

# Use existing stratified sampling
from extract_training_samples import create_stratified_sample
selected_files = create_stratified_sample(all_files, pagerank_scores, 10000)

# Run parallel processing
processor = HighPerformancePipelineProcessor()
stats = await processor.process_files_pipeline(selected_files, pagerank_scores)
```

## Next Steps

1. **Test Setup**: Run `--test` mode first
2. **Benchmark**: Compare performance with your hardware
3. **Scale Up**: Gradually increase sample sizes
4. **Monitor**: Watch resource usage and adjust accordingly
5. **Fine-tune**: Adjust worker counts and batch sizes for optimal performance

The parallel pipeline can process your entire IFS Cloud codebase efficiently, generating high-quality training samples for UnixCoder fine-tuning.
