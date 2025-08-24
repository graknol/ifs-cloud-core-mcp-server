# PL/SQL Code Summarization Fine-tuning Pipeline 🚀

This directory contains a complete pipeline for fine-tuning local language models to summarize PL/SQL code from your IFS Cloud codebase.

## 📋 Quick Start

### 1. Prerequisites

- Python 3.8+ with UV package manager
- IFS Cloud codebase indexed with PageRank scores (`ranked.jsonl`)
- Claude API key (for generating training summaries)

### 2. Install Dependencies

```bash
uv sync
```

### 3. Set Claude API Key

```bash
# Windows
$env:CLAUDE_API_KEY = "your-claude-api-key"

# Or pass directly to script
```

### 4. Run Complete Pipeline

```bash
# Extract 200 samples, generate summaries, prepare training
python fine_tuning_workflow.py --claude-api-key your-key
```

## 🔧 Individual Steps

### Extract Training Samples

```bash
python extract_training_samples.py --num-samples 200
```

### Evaluate Sample Quality

```bash
python evaluate_training_pipeline.py
```

### Generate Summaries with Claude

```bash
python generate_summaries_with_claude.py --input extracted_samples.jsonl
```

### Start Training

```bash
cd training_YYYYMMDD_HHMMSS/
python train_model.py
```

## 📊 What Each Script Does

### `extract_training_samples.py`

- **Purpose**: Intelligently extracts PL/SQL code samples for training
- **Strategy**: Stratified sampling across PageRank tiers
- **Features**:
  - ✅ Complexity analysis (cyclomatic complexity, line counts)
  - ✅ Rich context (API names, modules, business domains)
  - ✅ Quality filtering (skip trivial getters/setters)
  - ✅ Cross-module diversity

### `generate_summaries_with_claude.py`

- **Purpose**: Uses Claude API to generate high-quality summaries
- **Features**:
  - ✅ Optimized prompts for code summarization
  - ✅ Rate limiting and error handling
  - ✅ Quality validation
  - ✅ Fine-tuning dataset formatting

### `evaluate_training_pipeline.py`

- **Purpose**: Validates training data quality
- **Metrics**:
  - ✅ Sample distribution analysis
  - ✅ Diversity metrics (API, module, complexity)
  - ✅ Quality issue detection
  - ✅ Overall dataset scoring

### `fine_tuning_workflow.py`

- **Purpose**: Orchestrates the complete pipeline
- **Features**:
  - ✅ Step-by-step execution
  - ✅ Error handling and validation
  - ✅ Training setup generation
  - ✅ Complete automation

## 🎯 Enhanced Strategy

Your original approach (top 200 by PageRank) has been improved with:

### 1. **Stratified Sampling** 🎪

- **High-tier (25%)**: Most critical functions by PageRank
- **Mid-tier (50%)**: Moderately referenced functions
- **Low-tier (25%)**: Diverse patterns and edge cases
- **Result**: Better model generalization

### 2. **Rich Context** 🧠

```json
{
  "context": {
    "api_name": "CustomerOrder",
    "module": "order",
    "function_name": "Validate_Order_Lines___",
    "complexity_metrics": {
      "cyclomatic_complexity": 12,
      "code_lines": 85
    },
    "pagerank_score": 0.0234,
    "business_domain": "order_management"
  }
}
```

### 3. **Quality Controls** ✅

- Skip trivial functions (simple getters/setters)
- Filter by complexity (10-200 lines)
- Ensure module diversity
- Validate sample quality

### 4. **Claude Prompt Engineering** 📝

Specialized prompts for technical PL/SQL summaries:

- State main purpose with active verbs
- Identify key operations and validations
- Note side effects and exceptions
- Use proper technical terminology

## 📈 Expected Results

- **200 high-quality samples** with rich context
- **Balanced representation** across complexity tiers
- **Diverse coverage** of APIs and modules
- **Consistent summary style** from Claude
- **Ready-to-train dataset** for CodeT5+ or similar models

## 🏆 Model Recommendations

Instead of base CodeT5+, consider these specialized models:

1. **CodeT5+ 220M** - Good balance of size/performance
2. **CodeBERT** - Strong code understanding
3. **GraphCodeBERT** - Understands code structure
4. **UnixCoder** - Excellent for summarization

## 🔄 Production Pipeline

After fine-tuning:

```python
# Use fine-tuned model for bulk summarization
for chunk in all_plsql_functions:
    summary = fine_tuned_model.generate(chunk)
    if confidence_score(summary) < threshold:
        summary = fallback_to_claude(chunk)  # Hybrid approach
    save_summary(chunk, summary)
```

## 📊 Quality Metrics

The evaluation script provides:

- **Distribution analysis** (APIs, modules, complexity)
- **Diversity scores** (entropy-based metrics)
- **Quality indicators** (issue detection)
- **Overall dataset scoring** (0-100 scale)

## 🎯 Success Tips

1. **Quality > Quantity** - 200 excellent samples beats 1000 mediocre ones
2. **Rich Context** - Include business domains and complexity metrics
3. **Consistent Style** - Use well-engineered Claude prompts
4. **Iterative Improvement** - Start small, measure, improve
5. **Hybrid Approach** - Fine-tuned model + Claude fallback

## 📁 File Structure

```
├── extract_training_samples.py    # Sample extraction with stratification
├── generate_summaries_with_claude.py  # Claude-powered summary generation
├── evaluate_training_pipeline.py  # Quality evaluation and metrics
├── fine_tuning_workflow.py       # Complete pipeline orchestration
├── SAMPLES.md                     # Detailed approach documentation
└── training_YYYYMMDD_HHMMSS/     # Generated training setup
    ├── dataset.jsonl             # Fine-tuning dataset
    ├── training_config.json      # Model configuration
    └── train_model.py            # Training script
```

## 🚀 Next Steps

1. **Run the pipeline**: `python fine_tuning_workflow.py`
2. **Review quality report**: Check evaluation metrics
3. **Start training**: Use generated training setup
4. **Evaluate results**: Test on held-out samples
5. **Deploy**: Use for bulk summarization with Claude fallback

Your approach is excellent - these enhancements will make it even more effective! 🎯
