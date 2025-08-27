#!/usr/bin/env python3
"""
Simple converter to create Alpaca format dataset from IFS Cloud data.
Generates diverse training examples with multiple prompt variations.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_summarization_prompts() -> List[str]:
    """Get diverse summarization prompt templates."""
    return [
        "Analyze this IFS Cloud procedure and provide a concise business summary",
        "Summarize the business purpose of this procedure", 
        "Describe what this IFS Cloud procedure accomplishes",
        "Provide a brief overview of this procedure's functionality",
        "Explain the business value of this procedure",
        "What does this IFS Cloud procedure do?",
        "Give a concise description of this procedure",
        "Outline the business logic of this procedure",
        "Describe the main purpose of this procedure",
        "Summarize this procedure's core functionality"
    ]


def get_question_generation_prompts() -> List[str]:
    """Get diverse question generation prompt templates."""  
    return [
        "Generate a relevant question about this IFS Cloud procedure",
        "What question would a developer ask about this procedure?",
        "Create a search query for this procedure",
        "Generate a question that this procedure would answer", 
        "What would someone search for to find this procedure?",
        "Create a natural language query for this functionality",
        "Generate a question about this procedure's purpose",
        "What question does this procedure solve?",
        "Create a search term for this IFS Cloud functionality",
        "Generate a developer question about this procedure"
    ]


def load_clean_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the clean JSONL dataset."""
    data = []
    dataset_file = Path(dataset_path)
    
    if not dataset_file.exists():
        print(f"⚠️  Dataset file not found: {dataset_path}")
        return data
        
    print(f"📖 Loading dataset from {dataset_path}")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error on line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(data)} entries")
    return data


def create_example_dataset() -> List[Dict[str, Any]]:
    """Create example dataset for testing."""
    print("🔧 Creating example dataset for testing...")
    
    return [
        {
            "procedure_name": "Create_Purchase_Order",
            "signature": "Create_Purchase_Order(supplier_id_ IN VARCHAR2, order_date_ IN DATE)",
            "summary": "Creates a new purchase order for the specified supplier with validation and workflow processing",
            "module": "PURCHASE",
            "file": "PurchaseOrder.plsql",
            "synthetic_queries": [
                "How do I create a purchase order in IFS Cloud?",
                "What is the procedure for creating purchase orders?"
            ]
        },
        {
            "procedure_name": "Validate_Customer_Credit", 
            "signature": "Validate_Customer_Credit(customer_id_ IN VARCHAR2, credit_limit_ IN NUMBER)",
            "summary": "Validates customer credit limit and checks for any credit holds or restrictions",
            "module": "CUSTOMER",
            "file": "CustomerOrder.plsql", 
            "synthetic_queries": [
                "How is customer credit validated in IFS?",
                "What procedure checks customer credit limits?"
            ]
        },
        {
            "procedure_name": "Process_Invoice",
            "signature": "Process_Invoice(invoice_id_ IN VARCHAR2, approval_required_ IN BOOLEAN)",
            "summary": "Processes incoming invoices with automatic matching and approval workflow",
            "module": "FINANCE", 
            "file": "InvoiceProcessing.plsql",
            "synthetic_queries": [
                "How are invoices processed automatically in IFS?",
                "What is the invoice processing procedure?"
            ]
        }
    ]


def convert_to_alpaca_format(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert to simple Alpaca format with diverse examples."""
    print("🔄 Converting to Alpaca format...")
    
    alpaca_examples = []
    summarization_prompts = get_summarization_prompts()
    question_prompts = get_question_generation_prompts()
    
    for entry in data:
        procedure_name = entry.get('procedure_name', '')
        signature = entry.get('signature', '')
        summary = entry.get('summary', '')
        module = entry.get('module', '')
        file_name = entry.get('file', '')
        synthetic_queries = entry.get('synthetic_queries', [])
        
        # Skip if missing essential data
        if not all([procedure_name, signature, summary, module, file_name]):
            continue
        
        # Create procedure context as input
        procedure_input = f"Module: {module}\nFile: {file_name}\nProcedure: {procedure_name}\nSignature: {signature}"
        
        # 1. Summarization tasks - use all 10 prompts
        for instruction_template in summarization_prompts:
            alpaca_examples.append({
                "instruction": instruction_template,
                "input": procedure_input,
                "output": summary
            })
        
        # 2. Question generation tasks - use all 10 prompts with available queries
        for query in synthetic_queries:
            if query.strip():
                # Use different instruction templates for variety
                instruction_template = random.choice(question_prompts)
                alpaca_examples.append({
                    "instruction": instruction_template,
                    "input": procedure_input, 
                    "output": query
                })
    
    print(f"✅ Created {len(alpaca_examples)} Alpaca examples")
    return alpaca_examples


def save_alpaca_dataset(examples: List[Dict[str, str]], output_file: str = "alpaca_dataset.jsonl"):
    """Save dataset in Alpaca JSONL format."""
    
    # Shuffle for better training
    random.shuffle(examples)
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved {len(examples)} examples to {output_path}")
    
    # Show statistics
    instructions = set(ex['instruction'] for ex in examples)
    print(f"📊 Dataset statistics:")
    print(f"   - Total examples: {len(examples)}")
    print(f"   - Unique instructions: {len(instructions)}")
    print(f"   - Average examples per instruction: {len(examples) / len(instructions):.1f}")


def main():
    """Simple main function."""
    
    print("🔄 IFS Cloud → Alpaca Dataset Conversion")
    print("=" * 45)
    
    # Look for dataset
    dataset_candidates = [
        "plsql_analysis_combined/clean.jsonl",
        "clean.jsonl",
        "dataset.jsonl", 
        "training_data.jsonl"
    ]
    
    dataset_path = None
    for candidate in dataset_candidates:
        if Path(candidate).exists():
            dataset_path = candidate
            break
    
    # Load data
    if dataset_path:
        data = load_clean_dataset(dataset_path)
    else:
        print("⚠️  No dataset found, creating example dataset")
        data = create_example_dataset()
    
    if not data:
        print("❌ No data available for conversion")
        return
    
    # Convert to Alpaca format
    alpaca_examples = convert_to_alpaca_format(data)
    
    # Save dataset
    save_alpaca_dataset(alpaca_examples, "ifs_cloud_alpaca.jsonl")
    
    print("\n🎉 Conversion Complete!")
    print(f"📁 Output file: ifs_cloud_alpaca.jsonl")
    print(f"📊 Total examples: {len(alpaca_examples)}")
    print("\n📝 Example entry:")
    if alpaca_examples:
        example = alpaca_examples[0]
        print(json.dumps(example, indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()
            "Analyze this IFS Cloud procedure and provide a concise business summary",
            "Summarize the business purpose of this procedure",
            "Describe what this IFS Cloud procedure accomplishes",
            "Provide a brief overview of this procedure's functionality", 
            "Explain the business value of this procedure",
            "What does this IFS Cloud procedure do?",
            "Give a concise description of this procedure",
            "Outline the business logic of this procedure",
            "Describe the main purpose of this procedure",
            "Summarize this procedure's core functionality"
        ]
    
    def get_question_generation_prompts(self) -> List[str]:
        """Get diverse question generation prompt templates."""
        return [
            "Generate a relevant question about this IFS Cloud procedure",
            "What question would a developer ask about this procedure?",
            "Create a search query for this procedure",
            "Generate a question that this procedure would answer",
            "What would someone search for to find this procedure?",
            "Create a natural language query for this functionality",
            "Generate a question about this procedure's purpose",
            "What question does this procedure solve?",
            "Create a search term for this IFS Cloud functionality",
            "Generate a developer question about this procedure"
        ]
    
    def load_clean_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load the clean JSONL dataset."""
        data = []
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            print(f"⚠️  Dataset file not found: {dataset_path}")
            return data
            
        print(f"📖 Loading dataset from {dataset_path}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error on line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(data)} entries")
        return data
    
    def convert_to_axolotl_conversations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert to Axolotl conversation format."""
        print("🔄 Converting to Axolotl conversation format...")
        
        conversations = []
        summarization_prompts = self.get_summarization_prompts()
        question_prompts = self.get_question_generation_prompts()
        
        system_message = ("You are an expert IFS Cloud developer and analyst. You understand PL/SQL "
                         "procedures, business logic, and can provide accurate, concise explanations "
                         "about IFS Cloud functionality.")
        
        for entry in data:
            procedure_name = entry.get('procedure_name', '')
            signature = entry.get('signature', '')
            summary = entry.get('summary', '')
            module = entry.get('module', '')
            file_name = entry.get('file', '')
            synthetic_queries = entry.get('synthetic_queries', [])
            
            # Skip if missing essential data
            if not all([procedure_name, signature, summary, module, file_name]):
                continue
            
            # Create procedure context
            procedure_context = (
                f"Module: {module}\n"
                f"File: {file_name}\n"
                f"Procedure: {procedure_name}\n"
                f"Signature: {signature}"
            )
            
            # 1. Summarization conversations
            for prompt_template in summarization_prompts[:3]:  # Limit to 3 variations
                conversation = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": system_message
                        },
                        {
                            "from": "human", 
                            "value": f"{prompt_template}:\n\n{procedure_context}"
                        },
                        {
                            "from": "gpt",
                            "value": summary
                        }
                    ],
                    "source": "ifs_cloud_summaries",
                    "metadata": {
                        "module": module,
                        "file": file_name,
                        "procedure": procedure_name,
                        "task_type": "summarization"
                    }
                }
                conversations.append(conversation)
            
            # 2. Question generation conversations  
            for query in synthetic_queries[:2]:  # Limit to 2 queries per procedure
                if query.strip():
                    prompt_template = random.choice(question_prompts)
                    conversation = {
                        "conversations": [
                            {
                                "from": "system",
                                "value": system_message
                            },
                            {
                                "from": "human",
                                "value": f"{prompt_template}:\n\n{procedure_context}"
                            },
                            {
                                "from": "gpt", 
                                "value": query
                            }
                        ],
                        "source": "ifs_cloud_questions",
                        "metadata": {
                            "module": module,
                            "file": file_name, 
                            "procedure": procedure_name,
                            "task_type": "question_generation"
                        }
                    }
                    conversations.append(conversation)
        
        print(f"✅ Created {len(conversations)} conversation examples")
        
        # Statistics
        summarization_count = len([c for c in conversations if c['metadata']['task_type'] == 'summarization'])
        question_count = len([c for c in conversations if c['metadata']['task_type'] == 'question_generation'])
        
        print(f"   - Summarization conversations: {summarization_count}")
        print(f"   - Question generation conversations: {question_count}")
        
        return conversations
    
    def create_axolotl_config(self, model_name: str = "microsoft/Phi-4", 
                             output_name: str = "ifs-cloud-phi4") -> Dict[str, Any]:
        """Create Axolotl training configuration."""
        
        config = {
            # Base model configuration
            "base_model": model_name,
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "is_llama_derived_model": False,
            
            # Data configuration
            "datasets": [
                {
                    "path": str(self.output_dir / "train_conversations.jsonl"),
                    "type": "sharegpt",  # Axolotl's conversation format
                    "conversation_template": "phi4" if "phi" in model_name.lower() else "chatml"
                }
            ],
            
            # Training configuration optimized for RTX 5070 Ti (16GB VRAM)
            "sequence_len": 4096,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            
            # LoRA configuration for memory efficiency
            "adapter": "lora",
            "lora_model_dir": f"./lora_models/{output_name}",
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "lora_target_linear": True,
            "lora_fan_in_fan_out": False,
            "lora_target_modules": [
                "q_proj",
                "v_proj", 
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
            
            # Training parameters
            "output_dir": f"./outputs/{output_name}",
            "num_epochs": 3,
            "micro_batch_size": 2,  # Conservative for 16GB VRAM
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "lr_scheduler": "cosine",
            "warmup_steps": 100,
            
            # Optimization settings
            "optimizer": "adamw_torch_fused",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 4,
            
            # Flash Attention (since we have it installed)
            "flash_attention": True,
            "s2_attention": False,
            
            # Evaluation and saving
            "val_set_size": 0.1,
            "eval_steps": 100,
            "save_steps": 500,
            "logging_steps": 10,
            "save_safetensors": True,
            
            # Early stopping
            "early_stopping_patience": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            
            # Misc settings
            "wandb_project": f"ifs-cloud-{output_name}",
            "wandb_watch": "gradients",
            "hub_model_id": f"ifs-cloud/{output_name}",
            "push_dataset_to_hub": False,
            "resume_from_checkpoint": True,
            
            # Chat template (for Phi-4)
            "chat_template": "phi4" if "phi" in model_name.lower() else "chatml",
            
            # Special tokens
            "special_tokens": {
                "pad_token": "<|pad|>",
                "eos_token": "<|end|>",
                "bos_token": "<|begin|>",
                "unk_token": "<|unk|>"
            } if "phi" in model_name.lower() else None
        }
        
        return config
    
    def save_axolotl_dataset(self, conversations: List[Dict[str, Any]], 
                           split_ratio: float = 0.9) -> None:
        """Save dataset in Axolotl format with train/val split."""
        
        # Shuffle conversations
        random.shuffle(conversations)
        
        # Split into train/validation
        split_idx = int(len(conversations) * split_ratio)
        train_conversations = conversations[:split_idx]
        val_conversations = conversations[split_idx:]
        
        # Save training set
        train_file = self.output_dir / "train_conversations.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for conv in train_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        # Save validation set
        val_file = self.output_dir / "val_conversations.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for conv in val_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved datasets:")
        print(f"   - Training: {len(train_conversations)} conversations → {train_file}")
        print(f"   - Validation: {len(val_conversations)} conversations → {val_file}")
    
    def create_example_dataset(self) -> List[Dict[str, Any]]:
        """Create example dataset if no real data is available."""
        print("🔧 Creating example dataset for testing...")
        
        example_data = [
            {
                "procedure_name": "Create_Purchase_Order",
                "signature": "Create_Purchase_Order(supplier_id_ IN VARCHAR2, order_date_ IN DATE)",
                "summary": "Creates a new purchase order for the specified supplier with validation and workflow processing",
                "module": "PURCHASE",
                "file": "PurchaseOrder.plsql",
                "synthetic_queries": [
                    "How do I create a purchase order in IFS Cloud?",
                    "What is the procedure for creating purchase orders?"
                ]
            },
            {
                "procedure_name": "Validate_Customer_Credit",
                "signature": "Validate_Customer_Credit(customer_id_ IN VARCHAR2, credit_limit_ IN NUMBER)", 
                "summary": "Validates customer credit limit and checks for any credit holds or restrictions",
                "module": "CUSTOMER",
                "file": "CustomerOrder.plsql",
                "synthetic_queries": [
                    "How is customer credit validated in IFS?", 
                    "What procedure checks customer credit limits?"
                ]
            }
        ]
        
        return example_data
    
    def convert_dataset(self, dataset_path: Optional[str] = None, 
                       model_name: str = "microsoft/Phi-4",
                       output_name: str = "ifs-cloud-phi4") -> None:
        """Main conversion process."""
        
        print("🔄 IFS Cloud → Axolotl Dataset Conversion")
        print("=" * 50)
        
        # Load dataset
        if dataset_path and Path(dataset_path).exists():
            data = self.load_clean_dataset(dataset_path)
        else:
            print("⚠️  No dataset provided, creating example dataset")
            data = self.create_example_dataset()
        
        if not data:
            print("❌ No data available for conversion")
            return
        
        # Convert to Axolotl format
        conversations = self.convert_to_axolotl_conversations(data)
        
        # Save dataset
        self.save_axolotl_dataset(conversations)
        
        # Create Axolotl config
        config = self.create_axolotl_config(model_name, output_name)
        
        # Save config
        config_file = self.output_dir / "axolotl_config.yml" 
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"⚙️  Saved Axolotl config: {config_file}")
        
        # Create training script
        self.create_training_script()
        
        print("\n🎉 Conversion Complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔧 Config file: axolotl_config.yml")
        print(f"📊 Training data: train_conversations.jsonl")
        print(f"🧪 Validation data: val_conversations.jsonl")
        print(f"🚀 Training script: train_with_axolotl.sh")
        
    def create_training_script(self) -> None:
        """Create training script for Axolotl."""
        
        script_content = '''#!/bin/bash
# IFS Cloud Fine-tuning with Axolotl
# Optimized for RTX 5070 Ti with Flash Attention 2

echo "🚀 Starting IFS Cloud fine-tuning with Axolotl"
echo "=============================================="

# Check if axolotl is installed
if ! command -v axolotl &> /dev/null; then
    echo "❌ Axolotl not found. Installing..."
    pip install "axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl"
fi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# Training command
echo "🏃 Starting training..."
axolotl train axolotl_config.yml

echo "✅ Training completed!"
echo "📁 Model saved in: outputs/"
echo "🔍 Check logs for training metrics"
'''
        
        script_file = self.output_dir / "train_with_axolotl.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make executable
        script_file.chmod(0o755)
        
        print(f"🚀 Created training script: {script_file}")


def main():
    """Main function to run the conversion."""
    
    converter = AxolotlDatasetConverter()
    
    # Look for common dataset locations
    dataset_candidates = [
        "plsql_analysis_combined/clean.jsonl",
        "clean.jsonl", 
        "dataset.jsonl",
        "training_data.jsonl"
    ]
    
    dataset_path = None
    for candidate in dataset_candidates:
        if Path(candidate).exists():
            dataset_path = candidate
            break
    
    # Convert with different model options
    models = [
        ("microsoft/Phi-4", "ifs-cloud-phi4"),
        ("Qwen/Qwen2.5-7B-Instruct", "ifs-cloud-qwen25-7b"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "ifs-cloud-qwen25-1.5b")
    ]
    
    print("🎯 Available model configurations:")
    for i, (model, name) in enumerate(models, 1):
        print(f"  {i}. {model} → {name}")
    
    try:
        choice = input("\nChoose model (1-3, default=1): ").strip()
        choice = int(choice) if choice else 1
        model_name, output_name = models[choice - 1]
    except (ValueError, IndexError):
        model_name, output_name = models[0]
    
    print(f"\n🎯 Using model: {model_name}")
    print(f"📦 Output name: {output_name}")
    
    # Run conversion
    converter.convert_dataset(dataset_path, model_name, output_name)


if __name__ == "__main__":
    main()
