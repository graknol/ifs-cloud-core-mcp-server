#!/usr/bin/env python3
"""
Training Validation and Overfitting Prevention

This module provides utilities to detect and prevent overfitting during
the supervised training process. It includes:
- Validation set creation from held-out procedures
- Overfitting detection through validation loss monitoring
- Early stopping mechanisms
- Training curve visualization
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset

logger = logging.getLogger(__name__)


class TrainingValidator:
    """Handles validation and overfitting detection during training."""
    
    def __init__(self, 
                 save_dir: str = "./training_checkpoints",
                 validation_split: float = 0.2,
                 patience: int = 3,
                 min_delta: float = 0.001):
        
        self.save_dir = Path(save_dir)
        self.validation_split = validation_split
        self.patience = patience
        self.min_delta = min_delta
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'iterations': [],
            'best_val_loss': float('inf'),
            'patience_counter': 0
        }
        
        self.validation_procedures = []
        self.load_validation_history()

    def load_validation_history(self):
        """Load validation history if it exists."""
        history_file = self.save_dir / "validation_history.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)
            logger.info(f"Loaded validation history: {len(self.training_history['iterations'])} iterations")

    def save_validation_history(self):
        """Save validation history."""
        history_file = self.save_dir / "validation_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)

    def create_validation_set(self, all_procedures: List[Dict], current_summaries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Create training and validation sets from available procedures.
        
        Args:
            all_procedures: All available procedures
            current_summaries: Currently labeled summaries
            
        Returns:
            Tuple of (training_summaries, validation_summaries)
        """
        if len(current_summaries) < 10:
            # Not enough data for validation split yet
            logger.info("Not enough labeled data for validation split")
            return current_summaries, []
        
        # Shuffle and split
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(current_summaries))
        
        split_idx = int(len(current_summaries) * (1 - self.validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_summaries = [current_summaries[i] for i in train_indices]
        val_summaries = [current_summaries[i] for i in val_indices]
        
        # Store validation set for consistency
        self.validation_procedures = val_summaries
        
        logger.info(f"Created train/val split: {len(train_summaries)} train, {len(val_summaries)} val")
        
        return train_summaries, val_summaries

    def should_continue_training(self, current_iteration: int, val_loss: float) -> Tuple[bool, str]:
        """
        Determine if training should continue based on validation performance.
        
        Args:
            current_iteration: Current training iteration
            val_loss: Current validation loss
            
        Returns:
            Tuple of (should_continue, reason)
        """
        if not self.validation_procedures:
            return True, "No validation set available yet"
        
        # Update history
        self.training_history['iterations'].append(current_iteration)
        self.training_history['val_loss'].append(val_loss)
        
        # Check for improvement
        if val_loss < self.training_history['best_val_loss'] - self.min_delta:
            self.training_history['best_val_loss'] = val_loss
            self.training_history['patience_counter'] = 0
            return True, f"Validation improved to {val_loss:.4f}"
        else:
            self.training_history['patience_counter'] += 1
            
            if self.training_history['patience_counter'] >= self.patience:
                return False, f"Early stopping: no improvement for {self.patience} iterations"
            
            return True, f"No improvement ({self.training_history['patience_counter']}/{self.patience})"

    def evaluate_model(self, model, tokenizer, validation_summaries: List[Dict]) -> float:
        """
        Evaluate the model on validation set.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            validation_summaries: Validation data
            
        Returns:
            Validation loss
        """
        if not validation_summaries:
            return 0.0
        
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for summary_data in validation_summaries:
                # Create the same prompt format as training
                prompt = self.create_validation_prompt(summary_data)
                target = summary_data['human_summary']
                
                messages = [
                    {"role": "system", "content": "You are an expert at analyzing IFS Cloud business procedures."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
                
                # Calculate loss
                labels = inputs.input_ids.clone()
                
                # Mask prompt tokens (only calculate loss on assistant response)
                prompt_only = tokenizer.apply_chat_template(
                    messages[:2],
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_tokens = tokenizer(prompt_only, return_tensors="pt").input_ids.shape[1]
                labels[:, :prompt_tokens] = -100
                
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss

    def create_validation_prompt(self, procedure: Dict) -> str:
        """Create validation prompt (same as training prompt)."""
        name = procedure['name']
        params = procedure.get('parameters', [])
        business_code = procedure.get('business_code', '')
        module = procedure.get('module_name', 'unknown')
        
        # Filter out common CRUD parameters
        filtered_params = [p for p in params 
                          if not any(crud in p.lower() for crud in 
                                   ['info_', 'objid_', 'objversion_', 'attr_', 'action_'])]
        
        param_list = ', '.join(filtered_params) if filtered_params else 'no parameters'
        
        prompt = f"""Analyze this IFS Cloud procedure and provide a concise business summary:

Module: {module}
Procedure: {name}
Parameters: {param_list}

Code Logic:
{business_code}

Provide a single sentence summary that describes the business purpose of this procedure."""
        
        return prompt

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves."""
        if len(self.training_history['iterations']) < 2:
            logger.info("Not enough data points for plotting")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        iterations = self.training_history['iterations']
        
        if 'train_loss' in self.training_history and self.training_history['train_loss']:
            plt.plot(iterations, self.training_history['train_loss'], 'b-', label='Training Loss', marker='o')
        
        if 'val_loss' in self.training_history and self.training_history['val_loss']:
            plt.plot(iterations, self.training_history['val_loss'], 'r-', label='Validation Loss', marker='s')
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting indicators
        plt.subplot(1, 2, 2)
        if len(self.training_history['val_loss']) > 1:
            val_losses = self.training_history['val_loss']
            overfitting_signal = []
            
            for i in range(1, len(val_losses)):
                if val_losses[i] > val_losses[i-1]:
                    overfitting_signal.append(1)  # Potential overfitting
                else:
                    overfitting_signal.append(0)  # Improving
            
            plt.plot(iterations[1:], overfitting_signal, 'g-', marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Overfitting Signal')
            plt.title('Overfitting Detection (1=Potential, 0=Improving)')
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()

    def get_overfitting_recommendation(self) -> str:
        """Get recommendation on whether to continue training."""
        if not self.training_history['val_loss']:
            return "Continue: No validation data available yet"
        
        if len(self.training_history['val_loss']) < 3:
            return "Continue: Need more data points to assess overfitting"
        
        # Check recent trend
        recent_losses = self.training_history['val_loss'][-3:]
        
        if all(recent_losses[i] <= recent_losses[i-1] for i in range(1, len(recent_losses))):
            return "Continue: Validation loss is decreasing"
        
        if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
            return "⚠️  Warning: Validation loss is increasing (potential overfitting)"
        
        return "Continue: Mixed signals, more data needed"

    def create_training_report(self) -> str:
        """Create a comprehensive training report."""
        report = []
        report.append("=" * 60)
        report.append("TRAINING VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic stats
        report.append("TRAINING STATISTICS:")
        report.append(f"  • Total iterations: {len(self.training_history['iterations'])}")
        report.append(f"  • Validation set size: {len(self.validation_procedures)}")
        report.append(f"  • Best validation loss: {self.training_history['best_val_loss']:.4f}")
        report.append(f"  • Patience counter: {self.training_history['patience_counter']}/{self.patience}")
        report.append("")
        
        # Recent performance
        if self.training_history['val_loss']:
            recent_val = self.training_history['val_loss'][-1]
            report.append(f"RECENT PERFORMANCE:")
            report.append(f"  • Last validation loss: {recent_val:.4f}")
            
            if len(self.training_history['val_loss']) > 1:
                prev_val = self.training_history['val_loss'][-2]
                change = recent_val - prev_val
                direction = "📈" if change > 0 else "📉"
                report.append(f"  • Change from previous: {change:+.4f} {direction}")
        
        report.append("")
        report.append(f"RECOMMENDATION: {self.get_overfitting_recommendation()}")
        
        return "\n".join(report)


def main():
    """Demo of the validation system."""
    validator = TrainingValidator()
    
    # Create some dummy data for demonstration
    dummy_summaries = [
        {
            'name': f'Procedure_{i}',
            'human_summary': f'This is a test summary for procedure {i}',
            'parameters': [f'param_{j}' for j in range(3)],
            'business_code': f'BEGIN\n  -- Code for procedure {i}\nEND;',
            'module_name': 'TEST_MODULE'
        }
        for i in range(20)
    ]
    
    # Create validation split
    train_data, val_data = validator.create_validation_set([], dummy_summaries)
    print(f"Training set: {len(train_data)} procedures")
    print(f"Validation set: {len(val_data)} procedures")
    
    # Generate training report
    print(validator.create_training_report())


if __name__ == "__main__":
    main()
