#!/usr/bin/env python3
"""
Simple Optimized Batch Generator for IFS Cloud Procedure Summaries
Uses direct extraction without SupervisedTrainingLoop initialization issues
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direct imports for extraction functions
from ifs_parser_integration import IFSCloudParserIntegration


class SimpleBatchGenerator:
    """Simple batch generator without complex training loop dependencies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.parser = IFSCloudParserIntegration()
        
        # Load IFS Cloud keywords
        self.important_keywords = self.load_important_keywords()
        
        # Output directory
        self.output_dir = Path("batch_summaries")
        self.output_dir.mkdir(exist_ok=True)
        
        # Find or create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"simple_optimized_summaries_{timestamp}.json"
        
    def load_important_keywords(self) -> List[Dict[str, Any]]:
        """Load IFS Cloud important keywords."""
        keywords_file = Path("final_optimizer_keywords.csv")
        keywords = []
        
        if keywords_file.exists():
            import csv
            with open(keywords_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keywords.append({
                        'keyword': row['keyword'],
                        'variants': row.get('variants', '').split(',') if row.get('variants') else []
                    })
            logger.info(f"Loaded {len(keywords)} important keywords")
        
        return keywords
        
    def load_model_full_gpu(self):
        """Load model with aggressive GPU optimization."""
        logger.info(f"Loading model with full GPU optimization: {self.config['model_name']}")
        
        # Set aggressive CUDA memory settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
        # Set memory fraction to use more GPU
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            
            # Ensure we have proper tokens
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            print(f"✅ Tokenizer loaded with pad_token: {self.tokenizer.pad_token}")
            
            # Load model with full GPU allocation
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "cuda:0",  # Force GPU 0
                "trust_remote_code": True,
                "use_cache": True,
                "low_cpu_mem_usage": False,  # Keep on GPU
                "attn_implementation": "sdpa"
            }
            
            print("🚀 Loading model with full GPU allocation...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"], **model_kwargs
            )
            
            # Verify model is on GPU
            device = next(self.model.parameters()).device
            print(f"✅ Model loaded on device: {device}")
            
            # Print memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def extract_procedures(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract procedures from a PLSQL file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Use parser to extract procedures
            parsed_info = self.parser.parse_content(content, str(file_path))
            
            procedures = []
            if parsed_info and 'procedures' in parsed_info:
                for proc in parsed_info['procedures']:
                    procedures.append({
                        'name': proc.get('name', 'Unknown'),
                        'content': proc.get('content', ''),
                        'line_start': proc.get('line_start', 0),
                        'line_end': proc.get('line_end', 0),
                        'file_path': file_path,
                        'parsed_info': parsed_info
                    })
                    
            return procedures
            
        except Exception as e:
            logger.warning(f"Error extracting from {file_path}: {e}")
            return []
            
    def enhance_context_with_keywords(self, context: str) -> str:
        """Enhance context with relevant IFS Cloud keywords."""
        if not self.important_keywords:
            return context
            
        # Find relevant keywords in content
        relevant_keywords = []
        context_lower = context.lower()
        
        for kw_info in self.important_keywords:
            keyword = kw_info['keyword'].lower()
            variants = [v.strip().lower() for v in kw_info.get('variants', []) if v.strip()]
            
            # Check if keyword or variants appear in context
            if keyword in context_lower:
                relevant_keywords.append(kw_info['keyword'])
            else:
                for variant in variants:
                    if variant in context_lower:
                        relevant_keywords.append(kw_info['keyword'])
                        break
        
        # Add keyword enhancement if relevant keywords found
        if relevant_keywords:
            keyword_note = "\n\n=== IFS Cloud Keywords Found ===\n"
            keyword_note += ", ".join(relevant_keywords[:10])  # Limit to top 10
            keyword_note += "\nThese keywords indicate specific IFS Cloud business patterns and should be emphasized in the summary.\n"
            return context + keyword_note
            
        return context
        
    def generate_summary(self, procedure_info: Dict[str, Any]) -> Optional[str]:
        """Generate summary for a single procedure."""
        try:
            # Prepare context
            content = procedure_info['content']
            if len(content) > 3000:  # Limit context size
                content = content[:3000] + "\n... [content truncated] ..."
                
            # Enhance with keywords
            enhanced_content = self.enhance_context_with_keywords(content)
            
            # Create prompt
            prompt = f'''You are an expert IFS Cloud developer analyzing PL/SQL database procedures.

**Procedure to analyze:**
```plsql
{enhanced_content}
```

**Task:** Create a comprehensive technical summary for this IFS Cloud procedure.

**Required format:**
- **Purpose**: Brief description of what this procedure does
- **Key Features**: Main functionality and capabilities  
- **Parameters**: Important parameters and their purposes
- **Business Logic**: Core business rules and logic
- **Integration Points**: How it connects with other IFS components
- **Technical Notes**: Important implementation details

**Guidelines:**
- Focus on IFS Cloud business context and patterns
- Highlight important business logic and validation rules
- Explain parameter purposes and data flow
- Note any security checks or authorization logic
- Keep technical but accessible to other developers
- Aim for 150-300 words

**Summary:**'''

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=3500)
            
            # Move to GPU
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the summary part
            if "**Summary:**" in generated_text:
                summary = generated_text.split("**Summary:**")[-1].strip()
                return summary
            else:
                return generated_text[len(prompt):].strip()
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
            
    def find_plsql_files(self) -> List[str]:
        """Find all PLSQL files."""
        source_path = Path(self.config["ifs_source_path"])
        plsql_files = list(source_path.rglob("*.plsql"))
        logger.info(f"Found {len(plsql_files)} PLSQL files")
        return [str(f) for f in plsql_files]
        
    def run_batch_generation(self):
        """Run the batch generation process."""
        logger.info("🚀 Starting simple optimized batch generation...")
        
        # Load model
        self.load_model_full_gpu()
        
        # Find files
        plsql_files = self.find_plsql_files()
        
        # Load existing results if any
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                completed_files = set(item['file_path'] for item in existing_data)
        else:
            existing_data = []
            completed_files = set()
            
        # Process files
        target_summaries = min(self.config["target_summaries"], len(plsql_files))
        processed_count = len(existing_data)
        
        print(f"\n🎯 Target: {target_summaries} summaries")
        print(f"📊 Already completed: {processed_count}")
        print(f"🔄 Remaining: {target_summaries - processed_count}")
        
        start_time = time.time()
        
        for file_path in tqdm(plsql_files[:target_summaries], desc="Processing files"):
            if file_path in completed_files:
                continue
                
            try:
                # Extract procedures
                procedures = self.extract_procedures(file_path)
                
                # Take the first substantial procedure
                for proc in procedures:
                    if len(proc['content']) > 100:  # Skip tiny procedures
                        
                        # Generate summary
                        proc_start_time = time.time()
                        summary = self.generate_summary(proc)
                        proc_time = time.time() - proc_start_time
                        
                        if summary:
                            # Save result
                            result = {
                                'file_path': file_path,
                                'procedure_name': proc['name'],
                                'summary': summary,
                                'generation_time': proc_time,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            existing_data.append(result)
                            processed_count += 1
                            
                            # Save incrementally
                            with open(self.output_file, 'w', encoding='utf-8') as f:
                                json.dump(existing_data, f, indent=2)
                            
                            # Print progress
                            total_time = time.time() - start_time
                            avg_time = total_time / processed_count if processed_count > 0 else 0
                            remaining = target_summaries - processed_count
                            eta_minutes = (remaining * avg_time) / 60
                            
                            print(f"\n✅ {processed_count}/{target_summaries} - {proc['name']}")
                            print(f"⏱️  This: {proc_time:.1f}s, Avg: {avg_time:.1f}s, ETA: {eta_minutes:.1f}min")
                            
                            if processed_count >= target_summaries:
                                break
                                
                        break  # Only process first good procedure per file
                        
                if processed_count >= target_summaries:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
        total_time = time.time() - start_time
        print(f"\n🎉 Completed {processed_count} summaries in {total_time/60:.1f} minutes")
        print(f"💾 Results saved to: {self.output_file}")


def main():
    """Main entry point."""
    # Check optimizations
    print("✅ PyTorch SDPA available:", torch.backends.cuda.sdp_kernel())
    
    try:
        import triton
        print(f"✅ Triton available: {triton.__version__} (for custom kernel optimizations)")
    except ImportError:
        print("⚠️ Triton not available")
    
    try:
        import optimum
        print("✅ Optimum available (for inference optimization)")
    except ImportError:
        print("⚠️ Optimum not available")
    
    print("✅ IFS Cloud Tree-sitter Parser available")
    
    print("\n🎯 Simple Optimized IFS Cloud Batch Summary Generator")
    print("=" * 60)
    
    # Configuration
    config = {
        "model_name": "unsloth/Qwen2.5-7B-Instruct",
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "target_summaries": 200,
        "max_length": 4096,
        "batch_size": 8
    }
    
    print(f"📊 Optimization Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Mode: Full GPU (no CPU offloading)")
    print(f"  Target: {config['target_summaries']} summaries")
    print(f"  Context: 3500 tokens + 400 generation")
    print(f"  Source: {config['ifs_source_path']}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 RTX optimizations enabled on {gpu_name}")
        print(f"💾 Available VRAM: {total_memory:.1f}GB")
    else:
        print("❌ CUDA not available")
        return
    
    # Run generation
    generator = SimpleBatchGenerator(config)
    generator.run_batch_generation()


if __name__ == "__main__":
    main()
