#!/usr/bin/env python3
"""
Inference script for the fine-tuned Phi-4 model.
Use this to test your fine-tuned model on new procedure summarization tasks.
"""

import torch
from unsloth import FastLanguageModel
from typing import List, Dict, Any
import json


class IFSProcedureSummarizer:
    """Fine-tuned Phi-4 model for IFS Cloud procedure summarization."""
    
    def __init__(self, model_path: str = "phi4_ifs_cloud_final", max_seq_length: int = 2048):
        """Initialize the summarizer with the fine-tuned model."""
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"🚀 Loading fine-tuned model from {self.model_path}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        print("✅ Model loaded successfully")
    
    def summarize_procedure(
        self, 
        procedure_name: str,
        signature: str = "",
        module: str = "",
        file_name: str = "",
        task_type: str = "summarize",  # "summarize" or "generate_question"
        temperature: float = 0.7,
        max_new_tokens: int = 128
    ) -> str:
        """
        Generate a summary or question for a given procedure.
        
        Args:
            procedure_name: Name of the procedure
            signature: Procedure signature with parameters
            module: IFS Cloud module name
            file_name: Source file name
            task_type: Either "summarize" or "generate_question"
            temperature: Generation temperature (0.0 = deterministic, 1.0 = creative)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated summary or question
        """
        # Build standardized procedure context (matches training format)
        procedure_context = f"Module: {module}\nFile: {file_name}\nSignature: {signature}"
        
        # Choose prompt based on task type
        if task_type == "summarize":
            user_input = f"Summarize this procedure:\n\n{procedure_context}"
        elif task_type == "generate_question":
            user_input = f"Write a question that is answered by the following procedure:\n\n{procedure_context}"
        else:
            raise ValueError("task_type must be 'summarize' or 'generate_question'")
        
        # Format using Phi-4 chat template
        system_message = "You are an expert IFS Cloud developer. You analyze PL/SQL procedures and provide accurate, concise responses."
        formatted_input = f"<|system|>{system_message}<|end|><|user|>{user_input}<|end|><|assistant|>"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("<|end|>")[0] if "<|end|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1]
            if "<|end|>" in response:
                response = response.split("<|end|>")[0]
            response = response.strip()
        else:
            # Fallback: decode only the new tokens
            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
        
        return response
    
    def batch_summarize(self, procedures: List[Dict[str, Any]], task_type: str = "summarize") -> List[Dict[str, Any]]:
        """
        Process multiple procedures in batch.
        
        Args:
            procedures: List of procedure dictionaries with keys:
                       - procedure_name (required)
                       - signature (optional)
                       - module (optional)
                       - file_name (optional)
            task_type: Either "summarize" or "generate_question"
        
        Returns:
            List of procedures with added result field
        """
        task_name = "summaries" if task_type == "summarize" else "questions"
        result_field = "generated_summary" if task_type == "summarize" else "generated_question"
        
        print(f"📝 Generating {task_name} for {len(procedures)} procedures...")
        
        results = []
        for i, proc in enumerate(procedures, 1):
            try:
                result = self.summarize_procedure(
                    procedure_name=proc.get('procedure_name', ''),
                    signature=proc.get('signature', ''),
                    module=proc.get('module', ''),
                    file_name=proc.get('file_name', ''),
                    task_type=task_type
                )
                
                proc_result = proc.copy()
                proc_result[result_field] = result
                results.append(proc_result)
                
                if i % 10 == 0:
                    print(f"   Progress: {i}/{len(procedures)} ({i/len(procedures)*100:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️  Error processing procedure {proc.get('procedure_name', 'Unknown')}: {e}")
                proc_result = proc.copy()
                proc_result[result_field] = f"Error: {str(e)}"
                results.append(proc_result)
        
        print(f"✅ Batch processing completed")
        return results


def main():
    """Demo the fine-tuned model."""
    print("🧪 IFS Cloud Procedure Analyzer Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = IFSProcedureSummarizer()
    
    # Test examples
    test_procedures = [
        {
            "procedure_name": "Check_Exist___",
            "signature": "Check_Exist___(company_ IN VARCHAR2, code_part_value_ IN VARCHAR2) RETURN BOOLEAN",
            "module": "accrul",
            "file_name": "AccountingCodePartValue.plsql"
        },
        {
            "procedure_name": "Delete___",
            "signature": "Delete___(objid_ IN VARCHAR2, remrec_ IN ACCOUNTING_CODE_PART_VALUE_TAB%ROWTYPE)",
            "module": "accrul", 
            "file_name": "AccountingCodePartValue.plsql"
        },
        {
            "procedure_name": "Calculate_Tax",
            "signature": "Calculate_Tax(amount_ IN NUMBER, tax_rate_ IN NUMBER) RETURN NUMBER",
            "module": "invoic",
            "file_name": "TaxCalculation.plsql"
        }
    ]
    
    # Test both summarization and question generation
    for task_type, task_name in [("summarize", "Summarization"), ("generate_question", "Question Generation")]:
        print(f"\n📋 {task_name} Results:")
        print("=" * 50)
        
        results = analyzer.batch_summarize(test_procedures, task_type=task_type)
        result_field = "generated_summary" if task_type == "summarize" else "generated_question"
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Procedure: {result['procedure_name']}")
            print(f"   Module: {result.get('module', 'N/A')}")
            print(f"   {task_name}: {result[result_field]}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
