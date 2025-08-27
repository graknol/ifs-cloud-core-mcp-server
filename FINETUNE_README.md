# Phi-4 Mini Fine-tuning for IFS Cloud Procedure Summarization

This directory contains scripts and configuration for fine-tuning Microsoft's Phi-4 Mini Instruct model using Unsloth and QLoRA for IFS Cloud procedure summarization.

## 🎯 Overview

The fine-tuning process trains Phi-4 Mini to generate concise, business-focused summaries of IFS Cloud PL/SQL procedures based on context (module, file, procedure name, signature) and user queries/statements.

## 📋 Prerequisites

- **GPU**: NVIDIA GPU with at least 12GB VRAM (16GB+ recommended)
- **Python**: 3.8+
- **CUDA**: 11.8 or 12.1+
- **Dataset**: Clean synthetic questions dataset (`plsql_analysis_combined/clean.jsonl`)

## 🚀 Quick Start

1. **Verify dataset exists**:
   ```bash
   # Should exist from previous cleaning step
   ls plsql_analysis_combined/clean.jsonl
   ```

2. **Run the setup script** (handles dependency installation and training):
   ```bash
   python setup_finetune.py
   ```

3. **Or run manually**:
   ```bash
   # Install dependencies
   pip install -r requirements_finetune.txt
   
   # Start fine-tuning
   python finetune_phi4.py
   ```

4. **Test the fine-tuned model**:
   ```bash
   python inference_phi4.py
   ```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `finetune_phi4.py` | Main fine-tuning script with Unsloth + QLoRA |
| `inference_phi4.py` | Inference script to test the fine-tuned model |
| `setup_finetune.py` | Automated setup and training launcher |
| `requirements_finetune.txt` | Python dependencies |
| `finetune_config.ini` | Configuration parameters |
| `FINETUNE_README.md` | This file |

## ⚙️ Configuration

### Model Settings
- **Model**: `microsoft/Phi-4` (Mini Instruct)
- **Quantization**: 4-bit (QLoRA) for memory efficiency
- **Max Sequence Length**: 2048 tokens
- **LoRA Rank**: 16 (balanced quality/memory)

### Training Settings (Optimized for 16GB VRAM)
- **Batch Size**: 1 (effective: 8 with gradient accumulation)
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW 8-bit
- **Scheduler**: Cosine
- **Warmup Steps**: 100

### Memory Optimizations
- 4-bit quantization with bitsandbytes
- Gradient checkpointing
- 8-bit optimizer
- Small batch sizes with gradient accumulation
- Efficient attention mechanisms

## 📊 Dataset Format

The training uses the cleaned dataset with the following format conversion:

**Input Format** (Context + Query/Statement):
```
Context: Module: accrul, File: AccountingCodePartValue.plsql, Procedure: Check_Exist___, Signature: Check_Exist___(company_ IN VARCHAR2, code_part_value_ IN VARCHAR2) RETURN BOOLEAN
Query: How to verify if a record exists in the database?
```

**Output Format** (Summary):
```
Checks if a record exists in the accounting_code_part_value_tab table for the specified company and code part value. Returns TRUE if found, otherwise FALSE.
```

## 🔧 Hardware Requirements

| VRAM | Batch Size | Sequence Length | Status |
|------|------------|-----------------|--------|
| 12GB | 1 | 1024 | ✅ Supported |
| 16GB | 1 | 2048 | ✅ Optimal |
| 24GB+ | 2-4 | 2048+ | ✅ Fast training |
| <12GB | 1 | 512 | ⚠️ Limited |

## 📈 Training Progress

Expected training time on different hardware:
- **RTX 4090 (24GB)**: ~2-3 hours
- **RTX 3090 (24GB)**: ~3-4 hours  
- **RTX 4080 (16GB)**: ~4-5 hours
- **RTX 3080 (16GB)**: ~5-6 hours

## 🧪 Model Evaluation

The fine-tuned model will be evaluated on:
1. **Validation Loss**: Convergence monitoring
2. **Sample Outputs**: Quality of generated summaries
3. **Business Relevance**: Accuracy for IFS Cloud procedures

## 💾 Output Models

After training, you'll have:
1. **HuggingFace Format**: `./phi4_ifs_cloud_final/` (for further training)
2. **GGUF Format**: `./phi4_ifs_cloud_final/*.gguf` (for fast inference)

## 🔍 Usage Examples

### Basic Inference
```python
from inference_phi4 import IFSProcedureSummarizer

summarizer = IFSProcedureSummarizer()
summary = summarizer.summarize_procedure(
    procedure_name="Check_Insert___",
    signature="Check_Insert___(newrec_ IN OUT NOCOPY account_tab%ROWTYPE)",
    module="accrul",
    file_name="Account.plsql",
    query="What validation does this procedure perform?"
)
print(summary)
```

### Batch Processing
```python
procedures = [
    {"procedure_name": "Calculate_Tax", "module": "invoic", "query": "How is tax calculated?"},
    {"procedure_name": "Update_Status", "module": "orders", "query": "What status updates occur?"}
]

results = summarizer.batch_summarize(procedures)
for result in results:
    print(f"{result['procedure_name']}: {result['generated_summary']}")
```

## 🛠️ Troubleshooting

### Common Issues

**OOM (Out of Memory) Errors**:
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 1024 or 1536
- Reduce `gradient_accumulation_steps`

**Slow Training**:
- Ensure CUDA is properly installed
- Check GPU utilization with `nvidia-smi`
- Verify using GPU with `torch.cuda.is_available()`

**Poor Quality Outputs**:
- Increase training epochs
- Increase LoRA rank (`r` parameter)
- Adjust learning rate
- Check dataset quality

**Installation Issues**:
- Install PyTorch with CUDA first: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Install Unsloth from git: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`

## 📝 Configuration Tuning

Edit `finetune_config.ini` to adjust:
- **Memory usage**: Reduce batch size, sequence length, LoRA rank
- **Quality**: Increase LoRA rank, learning rate, epochs
- **Speed**: Increase batch size (if you have VRAM), reduce epochs

## 🎯 Next Steps

After successful fine-tuning:
1. **Test on new procedures** not in the training set
2. **Integrate with IFS analysis pipeline** 
3. **Compare with original GPT-4 summaries** for quality assessment
4. **Deploy for production procedure summarization**

## 📚 References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Phi-4 Model Card](https://huggingface.co/microsoft/Phi-4)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
