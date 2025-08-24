#!/usr/bin/env python3
"""
Complete workflow for fine-tuning PL/SQL summarization model.
This script orchestrates the entire pipeline from sample extraction to model training.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

class PLSQLFineTuningWorkflow:
    """Orchestrates the complete fine-tuning workflow."""
    
    def __init__(self, source_dir: str = "_work/source"):
        self.source_dir = Path(source_dir)
        self.work_dir = Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def check_prerequisites(self) -> bool:
        """Check if all required files and dependencies are available."""
        print("🔍 Checking prerequisites...")
        
        # Check for source directory
        if not self.source_dir.exists():
            print(f"❌ Source directory not found: {self.source_dir}")
            return False
        
        # Check for ranked.jsonl (PageRank scores)
        if not os.path.exists("ranked.jsonl"):
            print("❌ PageRank scores not found (ranked.jsonl)")
            print("   Run your indexing script first to generate PageRank scores")
            return False
        
        # Check for required scripts
        required_scripts = [
            "extract_training_samples.py",
            "generate_summaries_with_claude.py",
            "evaluate_training_pipeline.py"
        ]
        
        for script in required_scripts:
            if not os.path.exists(script):
                print(f"❌ Required script missing: {script}")
                return False
        
        print("✅ All prerequisites found!")
        return True
    
    def extract_samples(self, num_samples: int = 200) -> bool:
        """Extract training samples using stratified approach."""
        print(f"\n📊 Extracting {num_samples} training samples...")
        
        try:
            cmd = [
                sys.executable,
                "extract_training_samples.py",
                "--num-samples", str(num_samples),
                "--output", f"extracted_samples_{self.timestamp}.jsonl"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Sample extraction completed successfully!")
                return True
            else:
                print(f"❌ Sample extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error during sample extraction: {e}")
            return False
    
    def evaluate_samples(self, samples_file: str) -> bool:
        """Evaluate the quality of extracted samples."""
        print(f"\n🔍 Evaluating sample quality...")
        
        try:
            # Create a symlink or copy for the evaluator
            if os.path.exists("training_samples.jsonl"):
                os.remove("training_samples.jsonl")
            os.symlink(samples_file, "training_samples.jsonl")
            
            cmd = [sys.executable, "evaluate_training_pipeline.py"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Sample evaluation completed!")
                print(result.stdout)
                return True
            else:
                print(f"❌ Sample evaluation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error during sample evaluation: {e}")
            return False
    
    def generate_summaries(self, samples_file: str, method: str = "unixcoder", claude_api_key: str = None) -> bool:
        """Generate summaries using either UnixCoder or Claude API."""
        
        if method == "unixcoder":
            print(f"\n� Generating summaries with UnixCoder (FREE & LOCAL)...")
            return self._generate_summaries_unixcoder(samples_file)
        elif method == "claude":
            print(f"\n�🧠 Generating summaries with Claude API...")
            return self._generate_summaries_claude(samples_file, claude_api_key)
        else:
            print(f"❌ Unknown summarization method: {method}")
            return False
    
    def _generate_summaries_unixcoder(self, samples_file: str) -> bool:
        """Generate summaries using UnixCoder/CodeT5+ (local, free)."""
        try:
            cmd = [
                sys.executable,
                "generate_summaries_with_unixcoder.py",
                "--input", samples_file,
                "--output", f"training_dataset_{self.timestamp}.jsonl",
                "--validate"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ UnixCoder summary generation completed!")
                print(result.stdout)
                return True
            else:
                print(f"❌ UnixCoder summary generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error during UnixCoder summary generation: {e}")
            return False
    
    def _generate_summaries_claude(self, samples_file: str, claude_api_key: str = None) -> bool:
        """Generate summaries using Claude API."""
        if not claude_api_key:
            claude_api_key = os.getenv("CLAUDE_API_KEY")
            if not claude_api_key:
                print("❌ Claude API key not provided")
                print("   Set CLAUDE_API_KEY environment variable or pass --claude-api-key")
                return False
        
        try:
            env = os.environ.copy()
            env["CLAUDE_API_KEY"] = claude_api_key
            
            cmd = [
                sys.executable,
                "generate_summaries_with_claude.py",
                "--input", samples_file,
                "--output", f"training_dataset_{self.timestamp}.jsonl"
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Claude summary generation completed!")
                return True
            else:
                print(f"❌ Claude summary generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error during Claude summary generation: {e}")
            return False
    
    def prepare_model_training(self, dataset_file: str) -> bool:
        """Prepare the dataset for model training."""
        print(f"\n🚀 Preparing model training setup...")
        
        try:
            # Create training directory
            training_dir = Path(f"training_{self.timestamp}")
            training_dir.mkdir(exist_ok=True)
            
            # Copy dataset to training directory
            import shutil
            shutil.copy(dataset_file, training_dir / "dataset.jsonl")
            
            # Create training configuration
            config = {
                "model_name": "Salesforce/codet5p-220m",
                "dataset_file": "dataset.jsonl",
                "max_source_length": 512,
                "max_target_length": 128,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 100,
                "logging_steps": 50,
                "output_dir": "fine_tuned_model",
                "save_total_limit": 3,
                "evaluation_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False
            }
            
            with open(training_dir / "training_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create training script
            training_script = '''#!/usr/bin/env python3
"""
Fine-tuning script for PL/SQL code summarization using CodeT5+.
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import evaluate

def load_dataset(file_path):
    """Load training dataset from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'input': item['input'],
                'target': item['target']
            })
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    """Preprocess training examples."""
    inputs = examples['input']
    targets = examples['target']
    
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # Load configuration
    with open('training_config.json') as f:
        config = json.load(f)
    
    print(f"🚀 Starting fine-tuning with {config['model_name']}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])
    
    # Load and preprocess dataset
    dataset = load_dataset(config['dataset_file'])
    
    # Split dataset (80% train, 20% eval)
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Preprocess
    def preprocess(examples):
        return preprocess_function(examples, tokenizer, config['max_source_length'], config['max_target_length'])
    
    tokenized_datasets = dataset.map(preprocess, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_train_epochs=config['num_epochs'],
        warmup_steps=config['warmup_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        logging_steps=config['logging_steps'],
        save_total_limit=config['save_total_limit'],
        evaluation_strategy=config['evaluation_strategy'],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],
        report_to=None,  # Disable wandb
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(config['output_dir'])
    
    print(f"✅ Training completed! Model saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
'''
            
            with open(training_dir / "train_model.py", 'w') as f:
                f.write(training_script)
            
            print(f"✅ Training setup created in {training_dir}/")
            print(f"   📁 Dataset: {training_dir}/dataset.jsonl")
            print(f"   ⚙️ Config: {training_dir}/training_config.json")
            print(f"   🚀 Script: {training_dir}/train_model.py")
            print(f"\nTo start training:")
            print(f"   cd {training_dir}")
            print(f"   python train_model.py")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preparing training setup: {e}")
            return False
    
    def run_complete_workflow(self, num_samples: int = 200, method: str = "unixcoder", claude_api_key: str = None) -> bool:
        """Run the complete workflow from sample extraction to training preparation."""
        print("🎯 PL/SQL Fine-tuning Complete Workflow")
        print("=" * 50)
        
        if method == "unixcoder":
            print("🤖 Using UnixCoder for summarization (FREE & LOCAL)")
        else:
            print("🧠 Using Claude API for summarization")
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Step 2: Extract samples
        samples_file = f"extracted_samples_{self.timestamp}.jsonl"
        if not self.extract_samples(num_samples):
            return False
        
        # Step 3: Evaluate samples
        if not self.evaluate_samples(samples_file):
            print("⚠️ Sample evaluation failed, but continuing...")
        
        # Step 4: Generate summaries (UnixCoder or Claude)
        dataset_file = f"training_dataset_{self.timestamp}.jsonl"
        if not self.generate_summaries(samples_file, method, claude_api_key):
            return False
        
        # Step 5: Prepare training
        if not self.prepare_model_training(dataset_file):
            return False
        
        print("\n🎉 Complete workflow finished successfully!")
        print("\n📋 Summary:")
        print(f"   📊 Samples extracted: {samples_file}")
        print(f"   🧠 Training dataset: {dataset_file}")
        print(f"   🚀 Training setup: training_{self.timestamp}/")
        print(f"   💰 Cost: {'$0 (FREE!)' if method == 'unixcoder' else '$50-100 (Claude API)'}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="PL/SQL Fine-tuning Workflow")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to extract")
    parser.add_argument("--method", choices=["unixcoder", "claude"], default="unixcoder", 
                       help="Summarization method: unixcoder (free, local) or claude (paid, API)")
    parser.add_argument("--claude-api-key", help="Claude API key for summary generation (if using claude method)")
    parser.add_argument("--source-dir", default="_work/source", help="Source directory path")
    parser.add_argument("--step", choices=["extract", "evaluate", "summarize", "prepare", "complete"], 
                       default="complete", help="Which step to run")
    
    args = parser.parse_args()
    
    workflow = PLSQLFineTuningWorkflow(args.source_dir)
    
    if args.step == "extract":
        workflow.extract_samples(args.num_samples)
    elif args.step == "evaluate":
        # Find most recent extracted samples
        samples_files = [f for f in os.listdir(".") if f.startswith("extracted_samples_") and f.endswith(".jsonl")]
        if samples_files:
            latest_samples = sorted(samples_files)[-1]
            workflow.evaluate_samples(latest_samples)
        else:
            print("❌ No extracted samples found")
    elif args.step == "summarize":
        # Find most recent extracted samples
        samples_files = [f for f in os.listdir(".") if f.startswith("extracted_samples_") and f.endswith(".jsonl")]
        if samples_files:
            latest_samples = sorted(samples_files)[-1]
            workflow.generate_summaries(latest_samples, args.method, args.claude_api_key)
        else:
            print("❌ No extracted samples found")
    elif args.step == "prepare":
        # Find most recent training dataset
        dataset_files = [f for f in os.listdir(".") if f.startswith("training_dataset_") and f.endswith(".jsonl")]
        if dataset_files:
            latest_dataset = sorted(dataset_files)[-1]
            workflow.prepare_model_training(latest_dataset)
        else:
            print("❌ No training dataset found")
    else:  # complete
        workflow.run_complete_workflow(args.num_samples, args.method, args.claude_api_key)

if __name__ == "__main__":
    main()
