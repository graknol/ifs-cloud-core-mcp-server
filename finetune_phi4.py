#!/usr/bin/env python3
"""
Fine-tune Phi-4 Mini Instruct model using Unsloth and QLoRA for IFS Cloud procedure summarization.
Optimized for 16GB VRAM with 4-bit quantization and efficient memory management.
"""

import json
import torch
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
import gc
import os

# Model configuration
MODEL_NAME = "microsoft/Phi-4"  # Phi-4 Mini Instruct
MAX_SEQ_LENGTH = 2048  # Optimized for 16GB VRAM
DTYPE = None  # Auto-detect based on hardware
LOAD_IN_4BIT = True  # 4-bit quantization for memory efficiency

# LoRA configuration optimized for 16GB VRAM
LORA_CONFIG = {
    "r": 16,              # LoRA rank - balanced for quality/memory
    "target_modules": [   # Phi-4 attention modules
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_alpha": 16,     # LoRA scaling factor
    "lora_dropout": 0.1,  # Dropout for regularization
    "bias": "none",       # No bias adaptation
    "use_gradient_checkpointing": "unsloth",  # Memory optimization
    "random_state": 42,
    "use_rslora": False,  # Disable rank-stabilized LoRA for simplicity
    "loftq_config": None, # No LoftQ quantization
}

# Training hyperparameters optimized for 16GB VRAM
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,      # Small batch size for memory
    "gradient_accumulation_steps": 8,       # Effective batch size = 8
    "warmup_steps": 100,                    # Warmup for stable training
    "num_train_epochs": 3,                  # Multiple epochs for good convergence
    "max_steps": -1,                        # Use epochs instead
    "learning_rate": 2e-4,                  # Standard learning rate for QLoRA
    "fp16": not is_bfloat16_supported(),   # Use fp16 if bfloat16 not supported
    "bf16": is_bfloat16_supported(),       # Use bfloat16 if supported
    "logging_steps": 50,                    # Log every 50 steps
    "optim": "adamw_8bit",                 # 8-bit optimizer for memory savings
    "weight_decay": 0.01,                  # Light weight decay
    "lr_scheduler_type": "cosine",         # Cosine learning rate schedule
    "seed": 42,                            # Reproducibility
    "dataloader_num_workers": 0,           # Avoid multiprocessing issues
}


def load_clean_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the cleaned synthetic questions dataset."""
    print(f"📖 Loading clean dataset from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error on line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(data):,} entries")
    return data


def get_summarization_prompts() -> List[str]:
    """Get diverse summarization prompt templates to reduce overfitting."""
    return [
        "Summarize this procedure:",
        "Provide a summary of this procedure:",
        "Describe what this procedure does:",
        "Explain the functionality of this procedure:",
        "What is the purpose of this procedure?",
        "Give me a brief description of this procedure:",
        "Outline what this procedure accomplishes:",
        "Detail the function of this procedure:",
        "Can you summarize this procedure?",
        "Please describe this procedure's purpose:"
    ]


def get_question_generation_prompts() -> List[str]:
    """Get diverse question generation prompt templates to reduce overfitting."""
    return [
        "Write a question that is answered by the following procedure:",
        "Generate a question that this procedure addresses:",
        "Create a query that this procedure would solve:",
        "What question does this procedure answer?",
        "Formulate a question for which this procedure provides the solution:",
        "Compose a question that relates to this procedure:",
        "What inquiry would this procedure respond to?",
        "Generate an appropriate question for this procedure:",
        "What would you ask to get this procedure as an answer?",
        "Create a relevant question about this procedure:"
    ]


def convert_to_training_format(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert clean dataset to training format with proper Phi-4 chat template.
    Creates diverse training examples per procedure:
    1. Summarization tasks: Module+File+Signature -> Summary
    2. Question generation tasks: Module+File+Signature -> Synthetic Query
    """
    print("🔄 Converting dataset to training format with diverse prompts...")
    
    training_examples = []
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
        
        # Create standardized procedure context (what we'll use during inference)
        procedure_context = f"Module: {module}\nFile: {file_name}\nSignature: {signature}"
        
        # 1. Summarization tasks: Context -> Summary
        for prompt_template in summarization_prompts:
            training_examples.append({
                "input": f"{prompt_template}\n\n{procedure_context}",
                "output": summary,
                "task_type": "summarization"
            })
        
        # 2. Question generation tasks: Context -> Synthetic Query
        for query in synthetic_queries:
            # Use different prompt templates for variety
            for prompt_template in question_prompts:
                training_examples.append({
                    "input": f"{prompt_template}\n\n{procedure_context}",
                    "output": query,
                    "task_type": "question_generation"
                })
    
    print(f"✅ Created {len(training_examples):,} training examples")
    print(f"   - Summarization examples: {len([ex for ex in training_examples if ex['task_type'] == 'summarization']):,}")
    print(f"   - Question generation examples: {len([ex for ex in training_examples if ex['task_type'] == 'question_generation']):,}")
    
    return training_examples


def format_phi4_chat_template(example: Dict[str, str]) -> str:
    """
    Format training example using correct Phi-4 chat template.
    Format: <|system|>System Message<|end|><|user|>User Message<|end|><|assistant|>Assistant Response<|end|>
    """
    system_message = "You are an expert IFS Cloud developer. You analyze PL/SQL procedures and provide accurate, concise responses."
    
    # Format according to Phi-4's expected template
    formatted_text = (
        f"<|system|>{system_message}<|end|>"
        f"<|user|>{example['input']}<|end|>"
        f"<|assistant|>{example['output']}<|end|>"
    )
    
    return {"text": formatted_text}


def create_training_dataset(data: List[Dict[str, Any]], test_split: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Create training and validation datasets with proper formatting."""
    print("📊 Creating training datasets...")
    
    # Convert to training format
    training_examples = convert_to_training_format(data)
    
    # Shuffle and split
    random.shuffle(training_examples)
    split_idx = int(len(training_examples) * (1 - test_split))
    
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    print(f"📈 Training examples: {len(train_data):,}")
    print(f"📉 Validation examples: {len(val_data):,}")
    
    # Format with Phi-4 chat template
    train_formatted = [format_phi4_chat_template(ex) for ex in train_data]
    val_formatted = [format_phi4_chat_template(ex) for ex in val_data]
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    return train_dataset, val_dataset


def setup_model_and_tokenizer():
    """Load and configure the model and tokenizer with QLoRA."""
    print("🚀 Loading Phi-4 Mini Instruct model with 4-bit quantization...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,  # Required for Phi-4
    )
    
    # Configure LoRA
    print("⚙️  Applying LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        use_gradient_checkpointing=LORA_CONFIG["use_gradient_checkpointing"],
        random_state=LORA_CONFIG["random_state"],
        use_rslora=LORA_CONFIG["use_rslora"],
        loftq_config=LORA_CONFIG["loftq_config"],
    )
    
    # Add special tokens for Phi-4 chat format
    special_tokens = {
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded successfully")
    print(f"   • Parameters: {model.num_parameters():,}")
    print(f"   • Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   • Vocabulary size: {len(tokenizer)}")
    
    return model, tokenizer


def setup_trainer(model, tokenizer, train_dataset, val_dataset):
    """Setup the trainer with optimized configuration."""
    print("🎯 Setting up trainer...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./phi4_ifs_cloud_finetuned",
        overwrite_output_dir=True,
        **TRAINING_CONFIG,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps", 
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard for simplicity
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    # Create trainer with text field for Phi-4 format
    from trl import SFTTrainer
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",  # Use "text" field for Phi-4 formatted strings
        packing=False,  # Disable packing for simplicity
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    return trainer


def fine_tune_model():
    """Main fine-tuning function."""
    print("🔥 Starting Phi-4 Mini Fine-tuning for IFS Cloud Procedure Summarization")
    print("=" * 80)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔧 GPU Memory Available: {gpu_memory:.1f} GB")
        if gpu_memory < 15:
            print("⚠️  Warning: Less than 16GB VRAM detected. Consider reducing batch size or sequence length.")
    
    # Load dataset
    data = load_clean_dataset("plsql_analysis_combined/clean.jsonl")
    
    # Create training datasets
    train_dataset, val_dataset = create_training_dataset(data)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)
    
    # Clear cache before training
    torch.cuda.empty_cache()
    gc.collect()
    
    # Start training
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    model.save_pretrained("phi4_ifs_cloud_final")
    tokenizer.save_pretrained("phi4_ifs_cloud_final")
    
    # Save as GGUF for inference
    print("📦 Saving as GGUF format...")
    model.save_pretrained_gguf(
        "phi4_ifs_cloud_final",
        tokenizer,
        quantization_method="q4_k_m"  # 4-bit quantization
    )
    
    print("✅ Fine-tuning completed successfully!")
    print(f"   • Model saved to: ./phi4_ifs_cloud_final")
    print(f"   • GGUF model saved for efficient inference")


def test_model_inference():
    """Test the fine-tuned model with a sample inference."""
    print("\n🧪 Testing model inference...")
    
    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="phi4_ifs_cloud_final",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    # Test summarization
    test_input_summary = """Summarize this procedure:

Module: accrul
File: AccountingCodePartValue.plsql
Signature: Check_Exist___(company_ IN VARCHAR2, code_part_value_ IN VARCHAR2) RETURN BOOLEAN"""
    
    # Test question generation
    test_input_question = """Write a question that is answered by the following procedure:

Module: accrul
File: AccountingCodePartValue.plsql
Signature: Check_Exist___(company_ IN VARCHAR2, code_part_value_ IN VARCHAR2) RETURN BOOLEAN"""
    
    # Format for Phi-4
    system_message = "You are an expert IFS Cloud developer. You analyze PL/SQL procedures and provide accurate, concise responses."
    
    for test_name, test_input in [("Summarization", test_input_summary), ("Question Generation", test_input_question)]:
        formatted_input = f"<|system|>{system_message}<|end|><|user|>{test_input}<|end|><|assistant|>"
        
        inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True).to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("<|end|>")[0] if "<|end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id,
        )
        
        # Decode and extract response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1]
            if "<|end|>" in response:
                response = response.split("<|end|>")[0]
            response = response.strip()
        else:
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
        
        print(f"🎯 {test_name} Test Response: {response}")


if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        
        # Run fine-tuning
        fine_tune_model()
        
        # Test inference
        test_model_inference()
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
