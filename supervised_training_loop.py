#!/usr/bin/env python3
"""
Supervised Training Loop for IFS Cloud Procedure Summaries

This module implements an iterative training loop where:
1. Procedures are extracted and prepared with context
2. Random batches are run through the model for initial summaries
3. A GUI allows human verification and editing of summaries
4. The model is fine-tuned on verified summaries
5. Process repeats with increasing training data

Key Features:
- Cross-platform tkinter GUI with keyboard shortcuts
- Model checkpointing and progress saving
- Iterative fine-tuning with LoRA for efficiency
- Context-rich procedure presentation for human review
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# PyTorch SDPA optimization (no need for flash_attn)
SDPA_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
print(f"✅ PyTorch SDPA available: {SDPA_AVAILABLE}")

# Enable optimized attention backends
torch.backends.cuda.enable_flash_sdp(True)
if hasattr(torch.backends.cuda, 'enable_math_sdp'):
    torch.backends.cuda.enable_math_sdp(True)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(True)
from datasets import Dataset
import numpy as np

# Modern IFS Cloud parser using tree-sitter
try:
    from ifs_parser_integration import IFSCloudParserIntegration
    IFS_PARSER_AVAILABLE = True
    print("✅ IFS Cloud Tree-sitter Parser available")
except ImportError as e:
    IFSCloudParserIntegration = None
    IFS_PARSER_AVAILABLE = False
    print(f"⚠️ IFS Cloud Parser not available: {e}")

# Training validator
from training_validator import TrainingValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupervisedTrainingLoop:
    """Main class for the supervised training loop."""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                 ifs_source_path: str = "C:/repos/_ifs/25.1.0",
                 batch_size: int = 10,
                 save_dir: str = "./training_checkpoints"):
        
        self.model_name = model_name
        self.ifs_source_path = ifs_source_path
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training state
        self.current_iteration = 0
        self.all_summaries = []  # All accepted/edited summaries
        self.current_batch = []  # Current batch being reviewed
        self.procedures_pool = []  # All available procedures
        
        # Performance optimizations
        if torch.cuda.is_available():
            # Enable optimized CUDA operations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"🎯 CUDA optimizations enabled on {torch.cuda.get_device_name()}")
        
        # Validation system
        self.validator = TrainingValidator(save_dir=save_dir)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # GUI components
        self.root = None
        self.gui_queue = queue.Queue()
        
        # Load or initialize state
        self.load_training_state()
        self.initialize_model()
        self.extract_procedures()

    def load_training_state(self):
        """Load previous training state if it exists."""
        state_file = self.save_dir / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.current_iteration = state.get('current_iteration', 0)
                self.all_summaries = state.get('all_summaries', [])
            logger.info(f"Loaded training state: iteration {self.current_iteration}, "
                       f"{len(self.all_summaries)} summaries")
        else:
            logger.info("Starting fresh training session")

    def save_training_state(self):
        """Save current training state."""
        state = {
            'current_iteration': self.current_iteration,
            'all_summaries': self.all_summaries,
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = self.save_dir / "training_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        # Also save CSV backup
        csv_file = self.save_dir / f"summaries_iteration_{self.current_iteration}.csv"
        if self.all_summaries:
            import pandas as pd
            df = pd.DataFrame(self.all_summaries)
            df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"Saved training state: {len(self.all_summaries)} summaries")

    def initialize_model(self):
        """Initialize the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model loading with Flash Attention 2 if available
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
            "use_cache": True  # Enable KV caching for inference speedup
        }
        
        # Add Flash Attention 2 if available, otherwise use SDPA
        if SDPA_AVAILABLE:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("🚀 Using Flash Attention 2 for maximum performance!")
        else:
            model_kwargs["attn_implementation"] = "sdpa"  # Use PyTorch's optimized SDPA
            print("⚡ Using PyTorch SDPA (Scaled Dot Product Attention) for optimized performance")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Check for existing checkpoint
        checkpoint_dir = self.save_dir / f"checkpoint_iteration_{self.current_iteration}"
        if checkpoint_dir.exists():
            logger.info(f"Loading checkpoint from iteration {self.current_iteration}")
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(self.model, checkpoint_dir)
        else:
            logger.info("No checkpoint found, using base model")

    def extract_procedures(self):
        """Extract all procedures from IFS source code using tree-sitter parser."""
        if hasattr(self, '_procedures_extracted'):
            return
        
        logger.info("Extracting procedures from IFS source code...")
        
        self.procedures_pool = []
        
        # Use tree-sitter parser if available, otherwise fall back to mock data
        if not IFS_PARSER_AVAILABLE:
            logger.info("Using mock procedure data for demonstration...")
            self.procedures_pool = self.generate_mock_procedures()
            self._procedures_extracted = True
            return
            
        # Real implementation with tree-sitter parser
        parser = IFSCloudParserIntegration()
        
        # Find all source directories 
        try:
            from src.ifs_cloud_mcp_server.directory_utils import find_ifs_source_directories
            source_dirs = find_ifs_source_directories(self.ifs_source_path)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not find source directories: {e}")
            # Fall back to direct path search
            source_dirs = []
            if Path(self.ifs_source_path).exists():
                source_dirs = [p for p in Path(self.ifs_source_path).rglob("*") if p.is_dir() and "source" in str(p)]
        
        if not source_dirs:
            logger.warning("No source directories found, using mock data")
            self.procedures_pool = self.generate_mock_procedures()
            self._procedures_extracted = True
            return
        
        logger.info(f"Found {len(source_dirs)} source directories")
        
        for source_dir in source_dirs[:5]:  # Limit for demo
            logger.info(f"Processing directory: {source_dir}")
            
            # Look for .plsql files
            for file_path in Path(source_dir).rglob("*.plsql"):
                if file_path.stat().st_size > 1000000:  # Skip very large files
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Parse with tree-sitter
                    parsed_result = parser.parse_code(content)
                    
                    # Extract procedures and functions from parsed result
                    procedures = parsed_result.get('procedures', [])
                    functions = parsed_result.get('functions', [])
                    
                    # Process procedures
                    for proc_name in procedures:
                        procedure_info = self.extract_procedure_info(content, proc_name, file_path, parsed_result)
                        if procedure_info and len(procedure_info.get('body', '')) >= 50:
                            self.procedures_pool.append(procedure_info)
                    
                    # Process functions
                    for func_name in functions:
                        function_info = self.extract_procedure_info(content, func_name, file_path, parsed_result, is_function=True)
                        if function_info and len(function_info.get('body', '')) >= 50:
                            self.procedures_pool.append(function_info)
                        
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        if not self.procedures_pool:
            logger.warning("No procedures found, using mock data")
            self.procedures_pool = self.generate_mock_procedures()
        
        logger.info(f"Extracted {len(self.procedures_pool)} procedures using tree-sitter parser")
        self._procedures_extracted = True

    def extract_procedure_info(self, content: str, proc_name: str, file_path: Path, parsed_result: dict, is_function: bool = False) -> Optional[dict]:
        """Extract detailed information about a procedure or function."""
        try:
            # Find the procedure/function in the content
            lines = content.split('\n')
            proc_type = "FUNCTION" if is_function else "PROCEDURE"
            
            # Look for the procedure/function declaration
            start_line = None
            end_line = None
            
            for i, line in enumerate(lines):
                line_upper = line.upper().strip()
                if f"{proc_type} {proc_name.upper()}" in line_upper or f"{proc_type}  {proc_name.upper()}" in line_upper:
                    start_line = i
                    break
            
            if start_line is None:
                return None
            
            # Find the end of the procedure (END procedure_name;)
            for i in range(start_line + 1, len(lines)):
                line_upper = lines[i].upper().strip()
                if f"END {proc_name.upper()}" in line_upper:
                    end_line = i
                    break
            
            if end_line is None:
                end_line = min(start_line + 100, len(lines) - 1)  # Fallback
            
            # Extract the procedure body
            procedure_body = '\n'.join(lines[start_line:end_line + 1])
            
            # Extract business logic (remove comments and declarations)
            business_code = self.extract_business_code_from_body(procedure_body)
            
            # Get module name from file path
            module_name = self.extract_module_name(file_path)
            
            # Get file header
            file_header = self.extract_file_header(content)
            
            # Extract parameters from parsed result
            parameters = parsed_result.get('parameters', [])
            
            procedure_data = {
                'name': proc_name,
                'procedure_name': proc_name,  # For compatibility
                'parameters': parameters,
                'body': procedure_body,
                'business_code': business_code,
                'file_path': str(file_path),
                'module_name': module_name,
                'file_header': file_header,
                'line_number': start_line + 1,
                'parsed_info': parsed_result,
                'context': f"Module: {module_name}\nFile: {file_path.name}\n{file_header[:200]}...",
                'type': 'function' if is_function else 'procedure'
            }
            
            return procedure_data
            
        except Exception as e:
            logger.warning(f"Error extracting procedure info for {proc_name}: {e}")
            return None

    def extract_business_code_from_body(self, body: str) -> str:
        """Extract the core business logic from procedure body."""
        lines = body.split('\n')
        business_lines = []
        
        in_declaration = False
        found_begin = False
        
        for line in lines:
            line_stripped = line.strip().upper()
            
            # Skip empty lines and pure comments
            if not line_stripped or line_stripped.startswith('--'):
                continue
            
            # Skip declarations (between PROCEDURE/FUNCTION and BEGIN)
            if not found_begin:
                if 'BEGIN' in line_stripped:
                    found_begin = True
                continue
            
            # Skip END statement
            if line_stripped.startswith('END '):
                break
            
            # Add business logic lines
            if found_begin:
                business_lines.append(line)
        
        return '\n'.join(business_lines) if business_lines else body

    def extract_module_name(self, file_path: Path) -> str:
        """Extract module name from file path."""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part == "source" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"

    def extract_file_header(self, content: str) -> str:
        """Extract header comments from file."""
        lines = content.split('\n')
        header_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('--') or stripped.startswith('/*') or stripped.startswith('*'):
                header_lines.append(line)
            elif stripped and not stripped.startswith('--'):
                break
        
        return '\n'.join(header_lines[:20])  # First 20 header lines

    def generate_mock_procedures(self) -> List[Dict]:
        """Generate mock procedure data for testing and demonstration."""
        mock_procedures = [
            {
                'name': 'Process_Customer_Order___',
                'procedure_name': 'Process_Customer_Order___',
                'parameters': ['customer_id_', 'order_info_', 'result_'],
                'body': '''PROCEDURE Process_Customer_Order___ (
   customer_id_    IN VARCHAR2,
   order_info_     IN VARCHAR2,
   result_         OUT VARCHAR2) IS
   
   cursor_id_      NUMBER;
   order_status_   VARCHAR2(20);
BEGIN
   -- Validate customer
   IF customer_id_ IS NULL THEN
      Error_SYS.Record_General('CUSTOMER_NULL', 'Customer ID cannot be null');
   END IF;
   
   -- Process order
   SELECT status INTO order_status_ 
   FROM customer_order 
   WHERE customer_id = customer_id_;
   
   IF order_status_ = 'ACTIVE' THEN
      -- Update order processing
      UPDATE customer_order 
      SET process_date = SYSDATE 
      WHERE customer_id = customer_id_;
      
      result_ := 'ORDER_PROCESSED';
   END IF;
END Process_Customer_Order___;''',
                'business_code': '''-- Validate customer
IF customer_id_ IS NULL THEN
   Error_SYS.Record_General('CUSTOMER_NULL', 'Customer ID cannot be null');
END IF;

-- Process order
SELECT status INTO order_status_ 
FROM customer_order 
WHERE customer_id = customer_id_;

IF order_status_ = 'ACTIVE' THEN
   -- Update order processing
   UPDATE customer_order 
   SET process_date = SYSDATE 
   WHERE customer_id = customer_id_;
   
   result_ := 'ORDER_PROCESSED';
END IF;''',
                'file_path': 'mock/customer_order_api.plsql',
                'module_name': 'orderprocessing',
                'file_header': '''-- IFS Cloud Customer Order API
-- Purpose: Handle customer order processing and validation
-- Created: 2024
-- Module: Order Processing''',
                'line_number': 120,
                'context': 'Module: orderprocessing\nFile: customer_order_api.plsql\n-- IFS Cloud Customer Order API\n-- Purpose: Handle customer order processing...',
                'type': 'procedure'
            },
            {
                'name': 'Validate_Purchase_Req___',
                'procedure_name': 'Validate_Purchase_Req___',
                'parameters': ['req_no_', 'validation_result_'],
                'body': '''PROCEDURE Validate_Purchase_Req___ (
   req_no_            IN VARCHAR2,
   validation_result_ OUT VARCHAR2) IS
   
   req_status_        VARCHAR2(20);
   authorized_        VARCHAR2(5);
BEGIN
   -- Check requisition status
   SELECT status, authorized 
   INTO req_status_, authorized_
   FROM purchase_req_header
   WHERE req_no = req_no_;
   
   IF req_status_ != 'RELEASED' THEN
      validation_result_ := 'REQ_NOT_RELEASED';
      RETURN;
   END IF;
   
   IF authorized_ != 'TRUE' THEN
      validation_result_ := 'REQ_NOT_AUTHORIZED';
      RETURN;
   END IF;
   
   validation_result_ := 'VALIDATION_OK';
END Validate_Purchase_Req___;''',
                'business_code': '''-- Check requisition status
SELECT status, authorized 
INTO req_status_, authorized_
FROM purchase_req_header
WHERE req_no = req_no_;

IF req_status_ != 'RELEASED' THEN
   validation_result_ := 'REQ_NOT_RELEASED';
   RETURN;
END IF;

IF authorized_ != 'TRUE' THEN
   validation_result_ := 'REQ_NOT_AUTHORIZED';
   RETURN;
END IF;

validation_result_ := 'VALIDATION_OK';''',
                'file_path': 'mock/purchase_req_api.plsql',
                'module_name': 'purchasing',
                'file_header': '''-- IFS Cloud Purchase Requisition API
-- Purpose: Handle purchase requisition validation and processing
-- Created: 2024
-- Module: Purchasing''',
                'line_number': 85,
                'context': 'Module: purchasing\nFile: purchase_req_api.plsql\n-- IFS Cloud Purchase Requisition API...',
                'type': 'procedure'
            }
        ]
        
        # Add more varied procedures to reach desired count
        base_procedures = ['Invoice_Processing___', 'Asset_Validation___', 'Project_Setup___', 
                          'Person_Management___', 'Financial_Posting___']
        modules = ['accounting', 'assets', 'projects', 'hrm', 'finance']
        
        for i, (proc_name, module) in enumerate(zip(base_procedures, modules)):
            mock_procedures.append({
                'name': proc_name,
                'procedure_name': proc_name,
                'parameters': ['input_param_', 'result_param_'],
                'body': f'''PROCEDURE {proc_name} IS
   temp_var_   VARCHAR2(100);
   status_     VARCHAR2(20);
BEGIN
   -- Business logic for {proc_name.replace('_', ' ').title()}
   SELECT status INTO status_ FROM {module}_table WHERE id = input_param_;
   
   IF status_ = 'ACTIVE' THEN
      -- Process the {module} operation
      temp_var_ := 'PROCESSED_' || input_param_;
      result_param_ := temp_var_;
   END IF;
END {proc_name};''',
                'business_code': f'''SELECT status INTO status_ FROM {module}_table WHERE id = input_param_;

IF status_ = 'ACTIVE' THEN
   -- Process the {module} operation
   temp_var_ := 'PROCESSED_' || input_param_;
   result_param_ := temp_var_;
END IF;''',
                'file_path': f'mock/{module}_api.plsql',
                'module_name': module,
                'file_header': f'''-- IFS Cloud {module.title()} API
-- Purpose: Handle {module} operations
-- Module: {module.title()}''',
                'line_number': 50 + i * 10,
                'context': f'Module: {module}\nFile: {module}_api.plsql\n-- IFS Cloud {module.title()} API...',
                'type': 'procedure'
            })
        
        return mock_procedures

    def generate_summary_for_procedure(self, procedure: Dict) -> str:
        """Generate initial summary for a procedure using the model."""
        prompt = self.create_prompt(procedure)
        
        # Use the active model (PEFT if available, otherwise base)
        model_to_use = self.peft_model if self.peft_model else self.model
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing IFS Cloud business procedures. Generate concise, business-focused summaries that describe what the procedure does from a functional perspective."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(model_to_use.device)
        
        with torch.no_grad():
            # Optimized generation parameters for Flash Attention 2
            generated_ids = model_to_use.generate(
                model_inputs.input_ids,
                max_new_tokens=80,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                use_cache=True,  # Leverage KV caching for speed
                attention_mask=model_inputs.attention_mask if hasattr(model_inputs, 'attention_mask') else None
            )
        
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up the response
        summary = response.strip()
        if summary.startswith('**') and summary.endswith('**'):
            summary = summary[2:-2]
        
        # Remove markdown artifacts
        summary = summary.replace('```', '').replace('**', '').replace('*', '')
        
        return summary.strip()

    def cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"🧹 GPU memory cleaned: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")

    def create_prompt(self, procedure: Dict) -> str:
        """Create a prompt for the model based on procedure context."""
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

    def run_training_loop(self):
        """Main training loop."""
        logger.info("Starting supervised training loop")
        
        while True:
            # Get random batch
            self.current_batch = self.get_random_batch()
            
            if not self.current_batch:
                logger.info("No more procedures to process")
                break
            
            # Generate initial summaries
            logger.info(f"Generating summaries for batch {self.current_iteration + 1}")
            for proc in self.current_batch:
                proc['generated_summary'] = self.generate_summary_for_procedure(proc)
                proc['human_summary'] = proc['generated_summary']  # Initialize with generated
                proc['status'] = 'pending'  # pending, accepted, edited, skipped
            
            # Launch GUI for human review
            self.launch_review_gui()
            
            # Collect accepted summaries
            accepted = [p for p in self.current_batch if p['status'] in ['accepted', 'edited']]
            self.all_summaries.extend(accepted)
            
            if len(accepted) == 0:
                logger.info("No summaries accepted in this batch")
                break
            
            logger.info(f"Accepted {len(accepted)} summaries, total: {len(self.all_summaries)}")
            
            # Fine-tune model
            should_continue_training = True
            if len(self.all_summaries) >= 5:  # Need minimum for training
                should_continue_training = self.fine_tune_model()
                
                if not should_continue_training:
                    logger.warning("Training stopped due to overfitting concerns!")
                    # Show final report and ask user
                    report = self.validator.create_training_report()
                    continue_anyway = messagebox.askyesno(
                        "Overfitting Detected",
                        f"{report}\n\nOverfitting may be occurring. Continue anyway?",
                        default='no'
                    )
                    if not continue_anyway:
                        break
            
            self.current_iteration += 1
            self.save_training_state()
            
            # Show training summary
            if hasattr(self.validator, 'training_history') and self.validator.training_history['val_loss']:
                summary_msg = f"""Iteration {self.current_iteration} Summary:
• Accepted summaries: {len(accepted)}
• Total summaries: {len(self.all_summaries)}
• Last validation loss: {self.validator.training_history['val_loss'][-1]:.4f}
• Overfitting status: {self.validator.get_overfitting_recommendation()}"""
                logger.info(summary_msg)
            
            # Ask if user wants to continue
            if not self.ask_continue():
                break
        
        logger.info("Training loop completed")

    def get_random_batch(self) -> List[Dict]:
        """Get a random batch of procedures with module prioritization."""
        # Get procedures not yet processed
        processed_names = {s['name'] for s in self.all_summaries}
        current_batch_names = {p['name'] for p in self.current_batch}
        
        available = [p for p in self.procedures_pool 
                    if p['name'] not in processed_names 
                    and p['name'] not in current_batch_names]
        
        if not available:
            return []
        
        # Prioritize specific modules: proj, purch, accrul, person, prjdel
        priority_modules = {'PROJ', 'PURCH', 'ACCRUL', 'PERSON', 'PRJDEL'}
        
        priority_procedures = [p for p in available 
                             if p.get('module_name', '').upper() in priority_modules]
        other_procedures = [p for p in available 
                          if p.get('module_name', '').upper() not in priority_modules]
        
        # Calculate batch composition
        if len(available) < self.batch_size:
            batch_size = len(available)
        else:
            batch_size = self.batch_size
        
        # Prefer priority modules (75% of batch if available)
        priority_count = min(len(priority_procedures), int(batch_size * 0.75))
        other_count = batch_size - priority_count
        
        batch = []
        
        # Add priority procedures first
        if priority_procedures and priority_count > 0:
            batch.extend(random.sample(priority_procedures, priority_count))
        
        # Fill remaining slots with other procedures
        if other_procedures and other_count > 0:
            remaining_needed = other_count
            if len(batch) < batch_size:
                remaining_needed = batch_size - len(batch)
            
            if remaining_needed > 0:
                available_others = min(len(other_procedures), remaining_needed)
                batch.extend(random.sample(other_procedures, available_others))
        
        logger.info(f"Selected batch: {len(batch)} procedures "
                   f"({sum(1 for p in batch if p.get('module_name', '').upper() in priority_modules)} priority)")
        
        return batch

    def ask_continue(self) -> bool:
        """Ask user if they want to continue with another batch."""
        result = messagebox.askyesno(
            "Continue Training",
            f"Iteration {self.current_iteration} completed.\n"
            f"Total summaries: {len(self.all_summaries)}\n"
            f"Continue with next batch?",
            default='yes'
        )
        return result

    def launch_review_gui(self):
        """Launch the GUI for reviewing summaries."""
        self.root = tk.Tk()
        gui = SummaryReviewGUI(self.root, self.current_batch)
        gui.run()
        self.root = None

    def fine_tune_model(self):
        """Fine-tune the model on accepted summaries with validation."""
        logger.info(f"Fine-tuning model on {len(self.all_summaries)} summaries")
        
        # Create train/validation split to prevent overfitting
        train_summaries, val_summaries = self.validator.create_validation_set(
            self.procedures_pool, self.all_summaries
        )
        
        if len(train_summaries) < 5:
            logger.warning("Not enough training data for fine-tuning")
            return
        
        # Prepare training data
        train_data = []
        for summary_data in train_summaries:
            prompt = self.create_prompt(summary_data)
            target = summary_data['human_summary']
            
            messages = [
                {"role": "system", "content": "You are an expert at analyzing IFS Cloud business procedures. Generate concise, business-focused summaries that describe what the procedure does from a functional perspective."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            train_data.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(train_data)
        
        # Prepare model for training if not already done
        if self.peft_model is None:
            self.setup_peft_model()
        
        # Training arguments - Optimized for Flash Attention 2
        training_args = TrainingArguments(
            output_dir=str(self.save_dir / f"training_iteration_{self.current_iteration}"),
            num_train_epochs=2,  # Reduced from 3 to prevent overfitting
            per_device_train_batch_size=2 if SDPA_AVAILABLE else 1,  # Larger batch with SDPA
            gradient_accumulation_steps=4,
            warmup_steps=max(5, len(train_data) // 10),  # Dynamic warmup
            learning_rate=5e-5,  # Reduced learning rate
            fp16=True,
            logging_steps=5,
            save_strategy="epoch",
            evaluation_strategy="no",
            remove_unused_columns=False,
            weight_decay=0.01,  # Add weight decay for regularization
            max_grad_norm=1.0,  # Gradient clipping
            dataloader_num_workers=4 if SDPA_AVAILABLE else 2,  # More workers with SDPA
            dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
            ddp_find_unused_parameters=False,  # Optimization for single GPU
            report_to=[],  # Disable wandb/tensorboard for speed
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=1024
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        self.cleanup_gpu_memory()  # Clean memory before training
        trainer.train()
        self.cleanup_gpu_memory()  # Clean memory after training
        
        # Evaluate on validation set if available
        val_loss = 0.0
        if val_summaries:
            logger.info("Evaluating on validation set...")
            val_loss = self.validator.evaluate_model(self.peft_model, self.tokenizer, val_summaries)
            self.cleanup_gpu_memory()  # Clean memory after validation
        
        # Check if we should continue training
        should_continue, reason = self.validator.should_continue_training(
            self.current_iteration + 1, val_loss
        )
        
        logger.info(f"Validation result: {reason}")
        
        if not should_continue:
            logger.warning("Early stopping triggered - potential overfitting detected!")
            self.validator.plot_training_curves(
                str(self.save_dir / f"training_curves_iteration_{self.current_iteration}.png")
            )
            
            # Show training report
            report = self.validator.create_training_report()
            logger.info(f"\n{report}")
        
        # Save checkpoint
        checkpoint_dir = self.save_dir / f"checkpoint_iteration_{self.current_iteration + 1}"
        trainer.save_model(str(checkpoint_dir))
        
        # Save validation history
        self.validator.save_validation_history()
        
        logger.info(f"Fine-tuning completed, model saved to {checkpoint_dir}")
        
        return should_continue

    def setup_peft_model(self):
        """Setup PEFT (LoRA) configuration for efficient fine-tuning."""
        logger.info("Setting up PEFT model for efficient fine-tuning")
        
        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Get PEFT model
        self.peft_model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")


class SummaryReviewGUI:
    """GUI for reviewing and editing procedure summaries."""
    
    def __init__(self, root: tk.Tk, batch_data: List[Dict]):
        self.root = root
        self.batch_data = batch_data
        self.current_index = 0
        
        self.setup_gui()
        self.setup_bindings()
        self.display_current_procedure()

    def setup_gui(self):
        """Setup the GUI layout for wide screens with three columns."""
        self.root.title("IFS Cloud Procedure Summary Review")
        self.root.geometry("2400x1000")  # Wide screen optimized
        
        # Modern dark theme colors
        bg_dark = '#1e1e1e'
        bg_medium = '#2d2d30' 
        bg_light = '#3e3e42'
        text_primary = '#ffffff'
        text_secondary = '#cccccc'
        accent_blue = '#0078d4'
        accent_green = '#16c60c'
        accent_yellow = '#ffb900'
        accent_red = '#d13438'
        
        self.root.configure(bg=bg_dark)
        
        # Configure modern style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Modern.TLabel', 
                       font=('Segoe UI', 11),
                       background=bg_dark, 
                       foreground=text_primary)
        
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 14, 'bold'),
                       background=bg_dark, 
                       foreground=accent_blue)
        
        style.configure('Info.TLabel', 
                       font=('Segoe UI', 9),
                       background=bg_dark, 
                       foreground=text_secondary)
        
        # Main container
        main_container = tk.Frame(self.root, bg=bg_dark)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header with progress
        header_frame = tk.Frame(main_container, bg=bg_dark)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.progress_label = ttk.Label(header_frame, text="", style='Title.TLabel')
        self.progress_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_indicator = tk.Label(header_frame, 
                                       text="●", 
                                       font=('Segoe UI', 16),
                                       fg=accent_yellow,
                                       bg=bg_dark)
        self.status_indicator.pack(side=tk.RIGHT)
        
        # Main three-column content area
        content_frame = tk.Frame(main_container, bg=bg_dark)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # === LEFT COLUMN - Full File Contents ===
        left_panel = tk.Frame(content_frame, bg=bg_medium, relief='solid', bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # File contents header
        file_header = tk.Frame(left_panel, bg=bg_medium, height=40)
        file_header.pack(fill=tk.X, padx=15, pady=(10, 5))
        file_header.pack_propagate(False)
        
        ttk.Label(file_header, text="� Full File Contents", style='Title.TLabel').pack(anchor=tk.W)
        
        # File contents display
        file_container = tk.Frame(left_panel, bg=bg_medium)
        file_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.file_contents_text = scrolledtext.ScrolledText(
            file_container,
            height=30,
            width=60,
            bg=bg_light,
            fg=text_primary,
            font=('Cascadia Code', 9),
            insertbackground=text_primary,
            selectbackground=accent_blue,
            selectforeground='white',
            state=tk.DISABLED,
            relief='flat',
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.file_contents_text.pack(fill=tk.BOTH, expand=True)
        
        # === MIDDLE COLUMN - Context & Prompt ===
        middle_panel = tk.Frame(content_frame, bg=bg_medium, relief='solid', bd=1)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Context header
        context_header = tk.Frame(middle_panel, bg=bg_medium, height=40)
        context_header.pack(fill=tk.X, padx=15, pady=(10, 5))
        context_header.pack_propagate(False)
        
        ttk.Label(context_header, text="📋 Context & Model Prompt", style='Title.TLabel').pack(anchor=tk.W)
        
        # Context information
        context_container = tk.Frame(middle_panel, bg=bg_medium)
        context_container.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.context_text = scrolledtext.ScrolledText(
            context_container,
            height=15,
            width=50,
            bg=bg_light,
            fg=text_primary,
            font=('Segoe UI', 10),
            insertbackground=text_primary,
            selectbackground=accent_blue,
            selectforeground='white',
            state=tk.DISABLED,
            relief='flat',
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.context_text.pack(fill=tk.X)
        
        # Prompt preview section
        prompt_container = tk.Frame(middle_panel, bg=bg_medium)
        prompt_container.pack(fill=tk.X, padx=15, pady=(10, 15))
        
        ttk.Label(prompt_container, text="🤖 Model Prompt:", style='Info.TLabel').pack(anchor=tk.W)
        
        self.prompt_text = scrolledtext.ScrolledText(
            prompt_container,
            height=15,
            width=50,
            bg=bg_light,
            fg=text_secondary,
            font=('Cascadia Code', 9),
            state=tk.DISABLED,
            relief='flat',
            borderwidth=0,
            padx=8,
            pady=8
        )
        self.prompt_text.pack(fill=tk.X, pady=(3, 0))
        
        # === RIGHT COLUMN - Summary Editing ===
        right_panel = tk.Frame(content_frame, bg=bg_medium, relief='solid', bd=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Summary header
        summary_header = tk.Frame(right_panel, bg=bg_medium, height=40)
        summary_header.pack(fill=tk.X, padx=15, pady=(10, 5))
        summary_header.pack_propagate(False)
        
        ttk.Label(summary_header, text="✏️ Edit Generated Summary", style='Title.TLabel').pack(anchor=tk.W)
        
        # Edit instruction
        edit_container = tk.Frame(right_panel, bg=bg_medium)
        edit_container.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        edit_hint = ttk.Label(edit_container, 
                            text="📝 Edit the summary below (Ctrl+E to focus):", 
                            style='Info.TLabel')
        edit_hint.pack(anchor=tk.W)
        
        # Summary editing area
        summary_container = tk.Frame(right_panel, bg=bg_medium)
        summary_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
        
        self.summary_text = scrolledtext.ScrolledText(
            summary_container,
            height=20,
            width=60,
            bg='#ffffff',  # White background to clearly show it's editable
            fg='#000000',  # Black text for contrast
            font=('Segoe UI', 11),
            insertbackground='#000000',  # Black cursor
            selectbackground=accent_blue,
            selectforeground='white',
            wrap=tk.WORD,
            relief='solid',
            borderwidth=2,
            padx=10,
            pady=8
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Instructions section below summary
        instructions_container = tk.Frame(right_panel, bg=bg_medium)
        instructions_container.pack(fill=tk.X, padx=15, pady=(10, 15))
        
        ttk.Label(instructions_container, text="⌨️ Keyboard Shortcuts", style='Title.TLabel').pack(anchor=tk.W)
        
        instructions = """Ctrl+Enter → Accept current summary
Ctrl+S → Skip this procedure
Ctrl+E → Focus summary editor
Ctrl+← / Ctrl+→ → Navigate procedures
Ctrl+Q → Save and continue
Escape → Unfocus editor (move focus away)

Review context, check the model prompt, and edit 
the summary as needed. All changes are auto-saved."""
        
        instruction_text = scrolledtext.ScrolledText(
            instructions_container,
            height=9,
            width=60,
            bg=bg_light,
            fg=text_secondary,
            font=('Segoe UI', 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
            relief='flat',
            borderwidth=0,
            padx=10,
            pady=5
        )
        instruction_text.pack(fill=tk.X)
        instruction_text.config(state=tk.NORMAL)
        instruction_text.insert(tk.END, instructions)
        instruction_text.config(state=tk.DISABLED)
        
        # Bottom action bar spanning all columns
        bottom_frame = tk.Frame(main_container, bg=bg_dark)
        bottom_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Action buttons with modern styling
        button_frame = tk.Frame(bottom_frame, bg=bg_dark)
        button_frame.pack()
        
        # Custom button styling
        button_style = {
            'font': ('Segoe UI', 11, 'bold'),
            'padx': 20,
            'pady': 10,
            'relief': 'flat',
            'cursor': 'hand2'
        }
        
        self.accept_btn = tk.Button(button_frame, 
                                  text="✓ Accept (Ctrl+Enter)",
                                  command=self.accept_summary,
                                  bg=accent_green,
                                  fg='white',
                                  **button_style)
        self.accept_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.skip_btn = tk.Button(button_frame,
                                text="⏭ Skip (Ctrl+S)",
                                command=self.skip_summary,
                                bg=accent_yellow,
                                fg='black',
                                **button_style)
        self.skip_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.prev_btn = tk.Button(button_frame,
                                text="← Previous (Ctrl+←)",
                                command=self.previous_procedure,
                                bg=bg_light,
                                fg=text_primary,
                                **button_style)
        self.prev_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.next_btn = tk.Button(button_frame,
                                text="Next → (Ctrl+→)",
                                command=self.next_procedure,
                                bg=accent_blue,
                                fg='white',
                                **button_style)
        self.next_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_continue_btn = tk.Button(button_frame,
                                         text="💾 Save & Continue (Ctrl+Q)",
                                         command=self.save_and_continue,
                                         bg=accent_red,
                                         fg='white',
                                         **button_style)
        self.save_continue_btn.pack(side=tk.LEFT)
        
        # Status bar
        status_frame = tk.Frame(bottom_frame, bg=bg_dark)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="", style='Info.TLabel')
        self.status_label.pack()

    def setup_bindings(self):
        """Setup keyboard bindings that work globally."""
        # Global bindings that work regardless of focus
        self.root.bind_all('<Control-Return>', lambda e: self.accept_summary())
        self.root.bind_all('<Control-s>', lambda e: self.skip_summary())
        self.root.bind_all('<Control-e>', lambda e: self.focus_summary_editor())
        self.root.bind_all('<Control-Left>', lambda e: self.previous_procedure())
        self.root.bind_all('<Control-Right>', lambda e: self.next_procedure())
        self.root.bind_all('<Control-q>', lambda e: self.save_and_continue())
        
        # Escape should just remove focus from editor, not exit
        self.root.bind_all('<Escape>', lambda e: self.root.focus())
        
        # Allow Tab navigation between panels
        self.root.bind_all('<Tab>', lambda e: self.cycle_focus())
        self.root.bind_all('<Shift-Tab>', lambda e: self.cycle_focus(reverse=True))
        
        # Make sure clicking in text areas still allows editing
        self.summary_text.bind('<Button-1>', lambda e: self.summary_text.focus())
        if hasattr(self, 'file_text'):
            self.file_text.bind('<Button-1>', lambda e: self.file_text.focus())
        if hasattr(self, 'context_text'):
            self.context_text.bind('<Button-1>', lambda e: self.context_text.focus())
        if hasattr(self, 'prompt_text'):
            self.prompt_text.bind('<Button-1>', lambda e: self.prompt_text.focus())
    
    def focus_summary_editor(self):
        """Focus the summary editor for editing."""
        self.summary_text.focus()
        # Place cursor at end of text
        self.summary_text.mark_set(tk.INSERT, tk.END)
        self.summary_text.see(tk.INSERT)
        return 'break'  # Prevent default behavior
    
    def cycle_focus(self, reverse=False):
        """Cycle focus between the main text areas."""
        focused_widget = self.root.focus_get()
        
        # Define focus order based on available widgets
        focus_order = []
        if hasattr(self, 'file_text'):
            focus_order.append(self.file_text)
        if hasattr(self, 'context_text'):
            focus_order.append(self.context_text)
        if hasattr(self, 'prompt_text'):
            focus_order.append(self.prompt_text)
        focus_order.append(self.summary_text)
        
        if reverse:
            focus_order.reverse()
        
        try:
            current_index = focus_order.index(focused_widget)
            next_index = (current_index + 1) % len(focus_order)
        except ValueError:
            # If current widget not in order, start with first
            next_index = 0
        
        focus_order[next_index].focus()
        return 'break'

    def display_current_procedure(self):
        """Display the current procedure in the GUI."""
        if not self.batch_data or self.current_index >= len(self.batch_data):
            return
        
        proc = self.batch_data[self.current_index]
        
        # Update progress
        total = len(self.batch_data)
        self.progress_label.config(text=f"Procedure {self.current_index + 1} of {total}")
        
        # Update status indicator color
        status = proc.get('status', 'pending')
        status_colors = {
            'pending': '#ffb900',    # Yellow
            'accepted': '#16c60c',   # Green  
            'edited': '#0078d4',     # Blue
            'skipped': '#d13438'     # Red
        }
        self.status_indicator.config(fg=status_colors.get(status, '#ffb900'))
        
        # Update context information (left panel)
        context_info = f"""📁 MODULE: {proc.get('module_name', 'unknown').upper()}
📄 FILE: {Path(proc['file_path']).name}
⚙️ PROCEDURE: {proc['name']}
📝 PARAMETERS: {', '.join(proc.get('parameters', []))}
📍 LINE: {proc.get('line_number', '?')}
🔄 STATUS: {status.upper()}

📋 FILE HEADER:
{proc.get('file_header', '(no header)')}

💻 BUSINESS CODE:
{proc.get('business_code', proc.get('body', '')[:500] + '...' if len(proc.get('body', '')) > 500 else proc.get('body', ''))}
"""
        
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete(1.0, tk.END)
        self.context_text.insert(tk.END, context_info)
        self.context_text.config(state=tk.DISABLED)
        
        # Update prompt display (middle panel)
        prompt = self.create_prompt(proc)
        self.prompt_text.config(state=tk.NORMAL)
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(tk.END, prompt)
        self.prompt_text.config(state=tk.DISABLED)
        
        # Update summary - make sure it's editable (middle panel)
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, proc.get('human_summary', ''))
        # Keep it editable - don't disable!
        
        # Update full file contents (right panel)
        file_contents = self.load_file_contents(proc.get('file_path', ''))
        self.file_contents_text.config(state=tk.NORMAL)
        self.file_contents_text.delete(1.0, tk.END)
        self.file_contents_text.insert(tk.END, file_contents)
        self.file_contents_text.config(state=tk.DISABLED)
        
        # Update status display
        self.status_label.config(text=f"Status: {status.upper()} • Module: {proc.get('module_name', 'unknown').upper()}")

    def load_file_contents(self, file_path: str) -> str:
        """Load full file contents with line numbers."""
        try:
            if file_path and Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Add line numbers
                lines = content.split('\n')
                numbered_lines = [f"{i+1:3d}  {line}" for i, line in enumerate(lines)]
                return '\n'.join(numbered_lines)
            else:
                # Return mock content for testing
                return self.get_mock_file_content(file_path)
        except Exception as e:
            return f"Error loading file: {e}\n\nFile path: {file_path}"
    
    def get_mock_file_content(self, file_path: str) -> str:
        """Generate mock file content for testing purposes."""
        return f"""  1  -- IFS Cloud Package Body
  2  -- File: {Path(file_path).name if file_path else 'unknown.plsql'}
  3  -- Generated for demonstration
  4  
  5  PACKAGE BODY Test_API IS
  6  
  7  lu_name_ CONSTANT VARCHAR2(30) := 'TestAPI';
  8  
  9  PROCEDURE Example_Procedure___ IS
 10  BEGIN
 11     NULL;
 12  END Example_Procedure___;
 13  
 14  -- More procedures would appear here...
 15  
 16  END Test_API;"""

    def create_prompt(self, procedure: Dict) -> str:
        """Create a prompt for the model based on procedure context."""
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

    def accept_summary(self):
        """Accept the current summary."""
        if not self.batch_data or self.current_index >= len(self.batch_data):
            return
        
        proc = self.batch_data[self.current_index]
        current_summary = self.summary_text.get(1.0, tk.END).strip()
        
        if current_summary != proc.get('generated_summary', ''):
            proc['status'] = 'edited'
        else:
            proc['status'] = 'accepted'
        
        proc['human_summary'] = current_summary
        
        self.next_procedure()

    def skip_summary(self):
        """Skip the current summary."""
        if not self.batch_data or self.current_index >= len(self.batch_data):
            return
        
        proc = self.batch_data[self.current_index]
        proc['status'] = 'skipped'
        
        self.next_procedure()

    def previous_procedure(self):
        """Go to previous procedure."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_procedure()

    def next_procedure(self):
        """Go to next procedure."""
        if self.current_index < len(self.batch_data) - 1:
            self.current_index += 1
            self.display_current_procedure()
        else:
            # All procedures reviewed
            self.save_and_continue()

    def save_and_continue(self):
        """Save current work and continue."""
        # Update current procedure with any unsaved changes
        if self.batch_data and self.current_index < len(self.batch_data):
            proc = self.batch_data[self.current_index]
            current_summary = self.summary_text.get(1.0, tk.END).strip()
            
            if proc['status'] == 'pending':
                if current_summary != proc.get('generated_summary', ''):
                    proc['status'] = 'edited'
                    proc['human_summary'] = current_summary
        
        self.root.quit()

    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()


def main():
    """Main function to run the supervised training loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Supervised Training Loop for IFS Cloud Procedures")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model name")
    parser.add_argument("--ifs-path", default="C:/repos/_ifs/25.1.0", help="IFS source path")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--save-dir", default="./training_checkpoints", help="Save directory")
    
    args = parser.parse_args()
    
    # Create and run training loop
    training_loop = SupervisedTrainingLoop(
        model_name=args.model,
        ifs_source_path=args.ifs_path,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    training_loop.run_training_loop()


if __name__ == "__main__":
    main()
