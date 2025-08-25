# Supervised Training Loop for IFS Cloud Procedure Summaries

A comprehensive system for iteratively fine-tuning language models on IFS Cloud procedure summaries with human supervision and overfitting prevention.

## 🎯 Overview

This system implements a complete supervised training loop that:

1. **Extracts procedures** from IFS Cloud source code with rich context
2. **Generates initial summaries** using a pre-trained model (Qwen2.5-Coder-7B-Instruct)
3. **Presents summaries in a GUI** for human review and editing
4. **Fine-tunes the model** on validated summaries using LoRA (Low-Rank Adaptation)
5. **Prevents overfitting** through validation monitoring and early stopping
6. **Iterates continuously** with increasing training data

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (RTX 4090, RTX 5070 Ti, etc.)
- IFS Cloud source code (tested with 25.1.0)
- UV package manager installed

### Installation

```bash
# Clone and setup
cd your-project-directory
uv sync --extra gpu129  # or gpu128/gpu126 depending on your CUDA version

# Test the system
python test_training_system.py
```

### Launch Training

```bash
# Simple launch with defaults
python launch_training.py

# Or directly with custom settings
python supervised_training_loop.py --model "Qwen/Qwen2.5-Coder-7B-Instruct" --ifs-path "C:/path/to/ifs/source" --batch-size 10
```

## 🖥️ GUI Interface

The training GUI provides an intuitive interface for reviewing procedure summaries:

### Layout
- **Left Panel**: Context information (module, file, parameters, code snippets)
- **Right Panel**: Generated summary editor with keyboard shortcuts

### Keyboard Shortcuts
- **Enter**: Accept current summary as-is
- **E**: Edit summary (focus text area)
- **S**: Skip this procedure
- **←/→**: Navigate between procedures
- **Ctrl+S**: Save and continue to training
- **Escape**: Cancel and exit

### Context Information
Each procedure is presented with:
- Module name and file location
- Procedure name and filtered parameters (CRUD noise removed)
- File header comments for business context
- Extracted business-relevant code with control structures
- AST analysis results (loops, conditions, SQL operations)

## 🧠 Model Architecture

### Base Model
- **Qwen2.5-Coder-7B-Instruct**: High-quality code understanding model
- **Memory Usage**: ~14.2GB VRAM with fp16 precision
- **Context Window**: 32k tokens for comprehensive code analysis

### Fine-tuning Strategy
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with <1% trainable parameters
- **Conservative Training**: 2 epochs, reduced learning rate, weight decay
- **Gradient Clipping**: Prevents unstable training
- **Mixed Precision**: fp16 for memory efficiency

### Training Parameters
```python
r=16,                    # LoRA rank
lora_alpha=32,          # LoRA scaling
lora_dropout=0.1,       # Regularization
learning_rate=5e-5,     # Conservative rate
weight_decay=0.01,      # L2 regularization
max_grad_norm=1.0       # Gradient clipping
```

## 📊 Overfitting Prevention

### Validation Strategy
- **Automatic Split**: 20% of labeled data held out for validation
- **Early Stopping**: Monitors validation loss with patience=3
- **Overfitting Detection**: Tracks validation loss trends
- **Training Curves**: Visual monitoring of training progress

### Warning System
- **Real-time Monitoring**: Validation loss calculated after each iteration
- **User Alerts**: GUI warnings when overfitting is detected
- **Recommendations**: Intelligent suggestions for training continuation
- **Manual Override**: User can continue training despite warnings

### Conservative Defaults
The system uses conservative settings to minimize overfitting risk:
- Small learning rates
- Limited epochs per iteration
- Strong regularization
- Validation monitoring

## 📁 File Structure

```
supervised_training_loop.py    # Main training loop implementation
launch_training.py            # Simple launcher script
training_validator.py         # Overfitting prevention system
test_training_system.py      # Test suite for components
training_checkpoints/         # Model checkpoints and progress
├── checkpoint_iteration_N/   # LoRA weights for iteration N
├── training_state.json      # Training progress and summaries
├── validation_history.json  # Validation metrics history
├── summaries_iteration_N.csv # CSV backup of summaries
└── training_curves_N.png    # Training progress visualization
```

## 🔄 Training Loop Details

### Iteration Process
1. **Random Sampling**: Select 10 procedures not yet processed
2. **Summary Generation**: Generate initial summaries with current model
3. **Human Review**: GUI-based review with rich context
4. **Data Collection**: Collect accepted/edited summaries
5. **Model Training**: Fine-tune on all collected data
6. **Validation**: Evaluate on held-out validation set
7. **Overfitting Check**: Monitor validation metrics
8. **Checkpoint**: Save model and training state
9. **Repeat**: Continue with expanded training data

### Progressive Learning
- **Cumulative Training**: Each iteration includes all previous data
- **Validation Monitoring**: Prevents overfitting across iterations
- **Quality Control**: Only high-quality human-validated summaries used
- **Checkpoint Recovery**: Resume from any iteration

## 🛡️ Addressing Overfitting Concerns

### Your Question: "Will this cause overfitting issues?"

**Short Answer**: The system is designed to prevent overfitting through multiple mechanisms.

### Overfitting Prevention Mechanisms

1. **Validation Split**: 20% of data held out for monitoring
2. **Early Stopping**: Training halts when validation degrades
3. **LoRA Fine-tuning**: Only trains 0.5% of parameters
4. **Conservative Settings**: Small learning rates, limited epochs
5. **Regularization**: Weight decay and gradient clipping
6. **Progressive Validation**: Monitoring across all iterations

### Why Cumulative Training Works
- **LoRA Efficiency**: Fine-tuning adapter weights, not full model
- **Small Parameter Count**: <100M trainable vs 7B total parameters
- **Domain Specificity**: Training on consistent IFS procedure patterns
- **Human Validation**: High-quality, consistent training data
- **Validation Monitoring**: Early detection of overfitting

### Recommended Practice
- **Start Small**: Begin with 10 samples, validate, then expand
- **Monitor Carefully**: Watch validation curves after each iteration
- **Trust the System**: Early stopping will prevent overfitting
- **Quality Over Quantity**: Better to have fewer high-quality samples

## 🧪 Testing

### Test Components
```bash
# Test GUI interface with mock data
python test_training_system.py

# Test specific components
python -c "from training_validator import TrainingValidator; TrainingValidator().create_training_report()"
```

### Mock Data Testing
The test system includes realistic IFS procedure examples:
- Customer discount calculations
- Inventory validation logic
- Work order processing
- Complete with AST analysis and business context

## 📈 Performance Monitoring

### Metrics Tracked
- **Training Loss**: Model learning progress
- **Validation Loss**: Overfitting monitoring  
- **Acceptance Rate**: Human validation quality
- **Processing Speed**: Procedures per hour
- **Memory Usage**: GPU utilization

### Visual Monitoring
- **Training Curves**: Loss progression over iterations
- **Overfitting Signals**: Validation trend analysis
- **Progress Reports**: Comprehensive training summaries

## 🔧 Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0    # GPU selection
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Memory optimization
```

### Custom Configuration
```python
# In launch_training.py or custom script
config = {
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "ifs_source_path": "C:/repos/_ifs/25.1.0", 
    "batch_size": 10,
    "save_dir": "./training_checkpoints"
}
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Memory**: Reduce batch size or use gradient checkpointing
2. **PEFT Import**: Ensure `uv sync` completed successfully
3. **IFS Path**: Verify IFS source code directory exists
4. **Tkinter**: GUI requires tkinter (usually included with Python)

### Performance Optimization
- **Use fp16**: Reduces memory usage by ~50%
- **Gradient Accumulation**: Effective larger batch sizes
- **LoRA Configuration**: Adjust rank (r) for memory/quality tradeoff

## 📚 Architecture Deep Dive

### Code Context Extraction
The system extracts rich context for each procedure:

```python
def extract_business_code(self, procedure: Dict) -> str:
    # Control structures (IF, WHILE, FOR)
    # Business logic (non-CRUD operations) 
    # SQL operations and API calls
    # Exception handling patterns
    # Intelligent truncation with "..." markers
```

### Prompt Engineering
Optimized prompts focus on business intent:

```python
prompt = f"""Analyze this IFS Cloud procedure:

Module: {module}
Procedure: {name}
Parameters: {filtered_params}  # CRUD noise removed

Code Logic:
{business_code}  # AST-prioritized content

Provide a business-focused summary."""
```

### Training Data Quality
- **Parameter Filtering**: Removes info_, objid_, objversion_ noise
- **Code Prioritization**: Control structures over declarations
- **Business Focus**: Intent over implementation details
- **Human Validation**: Expert review ensures quality

## 🎯 Expected Results

After several iterations, the model should generate summaries like:

- **Before**: "This is a PL/SQL procedure that processes some data"
- **After**: "Validates customer credit limits against outstanding orders and applies hold status when limits are exceeded"

The system produces:
- **Concise**: Single sentence business summaries
- **Accurate**: Human-validated and corrected
- **Consistent**: Standardized format across procedures
- **Contextual**: Incorporates module and business domain knowledge

---

## 🤝 Contributing

The system is designed for extensibility:
- **Custom Models**: Easy model swapping
- **Enhanced Context**: Additional AST analysis
- **GUI Improvements**: Extended keyboard shortcuts
- **Validation Metrics**: Custom overfitting detection

## 📄 License

This project follows the same license as the parent IFS Cloud MCP Server project.

---

*Happy Training! 🚀*
