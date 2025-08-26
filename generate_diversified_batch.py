#!/usr/bin/env python3
"""
Diversified Batch Generator - Smart Selection from Target Modules
Focuses on 6 core modules: purch, order, enterp, proj, accru, prjdell
Pre-processes and randomizes selection before inference
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target modules for IFS Cloud analysis
TARGET_MODULES = ['purch', 'order', 'enterp', 'proj', 'accru', 'prjdell']


class DiversifiedBatchGenerator:
    """Diversified batch generator that focuses on target modules with smart selection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Load IFS Cloud keywords
        self.important_keywords = self.load_important_keywords()
        
        # Output directories
        self.output_dir = Path("batch_summaries")
        self.output_dir.mkdir(exist_ok=True)
        
        self.planning_dir = Path("batch_planning")
        self.planning_dir.mkdir(exist_ok=True)
        
        # Planning file for pre-processed contexts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.planning_file = self.planning_dir / f"diversified_plan_{timestamp}.json"
        self.output_file = self.output_dir / f"diversified_summaries_{timestamp}.json"
        
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
        
    def extract_procedures_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract all procedures from a PLSQL file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Find procedure/function definitions
            procedure_pattern = r'(?:PROCEDURE|FUNCTION)\s+([A-Za-z_][A-Za-z0-9_]*)'
            matches = re.finditer(procedure_pattern, content, re.IGNORECASE)
            
            procedures = []
            content_lines = content.split('\n')
            
            for match in matches:
                proc_name = match.group(1)
                start_pos = match.start()
                
                # Find line number
                line_start = content[:start_pos].count('\n') + 1
                
                # Extract procedure context (try to get full procedure or reasonable chunk)
                # Start from procedure declaration
                start_line_idx = line_start - 1
                
                # Look for END statement or next procedure (limit to reasonable size)
                end_line_idx = min(start_line_idx + 200, len(content_lines))  # Max 200 lines per procedure
                
                # Try to find proper END statement
                for i in range(start_line_idx + 10, min(start_line_idx + 200, len(content_lines))):
                    line = content_lines[i].strip().upper()
                    if (line.startswith('END ') and proc_name.upper() in line) or line == 'END;':
                        end_line_idx = i + 1
                        break
                
                procedure_content = '\n'.join(content_lines[start_line_idx:end_line_idx])
                
                # Calculate complexity score
                complexity = self.calculate_complexity(procedure_content)
                
                procedures.append({
                    'name': proc_name,
                    'content': procedure_content,
                    'file_path': file_path,
                    'line_start': line_start,
                    'line_end': line_start + (end_line_idx - start_line_idx),
                    'complexity_score': complexity,
                    'content_length': len(procedure_content)
                })
            
            return procedures
            
        except Exception as e:
            logger.warning(f"Error extracting procedures from {file_path}: {e}")
            return []
    
    def calculate_complexity(self, content: str) -> float:
        """Calculate complexity score for a procedure."""
        content_upper = content.upper()
        
        # Count various complexity indicators
        indicators = {
            'IF': content_upper.count('IF '),
            'CASE': content_upper.count('CASE '),
            'LOOP': content_upper.count('LOOP'),
            'WHILE': content_upper.count('WHILE'),
            'FOR': content_upper.count('FOR '),
            'CURSOR': content_upper.count('CURSOR'),
            'EXCEPTION': content_upper.count('EXCEPTION'),
            'RAISE': content_upper.count('RAISE'),
            'SELECT': content_upper.count('SELECT'),
            'UPDATE': content_upper.count('UPDATE'),
            'INSERT': content_upper.count('INSERT'),
            'DELETE': content_upper.count('DELETE'),
        }
        
        # Weight different indicators
        weights = {
            'IF': 1.5, 'CASE': 2.0, 'LOOP': 2.5, 'WHILE': 2.5, 'FOR': 2.0,
            'CURSOR': 3.0, 'EXCEPTION': 2.0, 'RAISE': 1.5,
            'SELECT': 1.0, 'UPDATE': 1.5, 'INSERT': 1.5, 'DELETE': 1.5
        }
        
        complexity = sum(count * weights[indicator] for indicator, count in indicators.items())
        
        # Normalize by content length (favor longer, more complex procedures)
        if len(content) > 0:
            complexity = complexity * (len(content) / 1000)  # Per 1000 characters
            
        return complexity
    
    def find_module_files(self) -> Dict[str, List[str]]:
        """Find PLSQL files organized by target modules."""
        source_path = Path(self.config["ifs_source_path"])
        all_files = list(source_path.rglob("*.plsql"))
        
        module_files = {module: [] for module in TARGET_MODULES}
        
        for file_path in all_files:
            file_path_lower = str(file_path).lower()
            
            # Check if file belongs to any target module
            for module in TARGET_MODULES:
                if module in file_path_lower:
                    module_files[module].append(str(file_path))
                    break
        
        # Log statistics
        total_found = sum(len(files) for files in module_files.values())
        print(f"\n📂 Module File Distribution:")
        for module, files in module_files.items():
            print(f"  {module}: {len(files)} files")
        print(f"  Total: {total_found} files from target modules")
        
        return module_files
    
    def create_diversified_plan(self) -> List[Dict[str, Any]]:
        """Create a diversified selection plan with pre-computed contexts."""
        print("\n🎯 Creating diversified selection plan...")
        
        # Find files by module
        module_files = self.find_module_files()
        
        # Extract procedures from all target files
        all_procedures = []
        
        for module, files in module_files.items():
            print(f"\n📝 Processing {module} module ({len(files)} files)...")
            
            # Randomize file order within module
            random.shuffle(files)
            
            for file_path in tqdm(files, desc=f"Extracting from {module}"):
                procedures = self.extract_procedures_from_file(file_path)
                
                # Add module info to each procedure
                for proc in procedures:
                    proc['module'] = module
                    all_procedures.append(proc)
        
        print(f"\n📊 Extracted {len(all_procedures)} procedures from target modules")
        
        # Filter by complexity and quality
        quality_procedures = [p for p in all_procedures if 
                            p['complexity_score'] > 5.0 and  # Minimum complexity
                            p['content_length'] > 500 and    # Minimum size
                            p['content_length'] < 12000]     # Maximum size for context window
        
        print(f"📊 Quality filtered: {len(quality_procedures)} procedures")
        
        # Sort by complexity (descending) for diversity
        quality_procedures.sort(key=lambda x: x['complexity_score'], reverse=True)
        
        # Create diversified selection
        target_count = self.config["target_summaries"]
        selected_procedures = []
        
        # Ensure module diversity - get procedures from each module
        module_quotas = {module: target_count // len(TARGET_MODULES) for module in TARGET_MODULES}
        remainder = target_count % len(TARGET_MODULES)
        
        # Distribute remainder
        for i, module in enumerate(TARGET_MODULES[:remainder]):
            module_quotas[module] += 1
        
        print(f"\n📋 Module quotas: {module_quotas}")
        
        # Select procedures per module
        for module in TARGET_MODULES:
            module_procs = [p for p in quality_procedures if p['module'] == module]
            quota = module_quotas[module]
            
            if len(module_procs) >= quota:
                # Randomly sample from top complexity procedures
                top_complex = module_procs[:min(quota * 3, len(module_procs))]  # Top candidates
                selected = random.sample(top_complex, quota)
                selected_procedures.extend(selected)
                print(f"  {module}: Selected {len(selected)}/{len(module_procs)} procedures")
            else:
                # Take all available
                selected_procedures.extend(module_procs)
                print(f"  {module}: Selected {len(module_procs)}/{len(module_procs)} procedures (all available)")
        
        # Final shuffle for processing order
        random.shuffle(selected_procedures)
        
        print(f"\n🎯 Final selection: {len(selected_procedures)} diverse procedures")
        
        # Pre-compute enhanced contexts and prompts
        planning_data = []
        
        for i, proc in enumerate(tqdm(selected_procedures, desc="Pre-computing contexts")):
            enhanced_context = self.enhance_context_with_keywords(proc['content'])
            
            prompt = f'''You are an expert IFS Cloud developer analyzing PL/SQL database code.

**IFS Cloud {proc['module'].upper()} module procedure to analyze:**
```plsql
{enhanced_context}
```

**Procedure:** {proc['name']}
**Module:** {proc['module'].upper()}
**Complexity Score:** {proc['complexity_score']:.1f}

**Task:** Create a comprehensive technical summary for this IFS Cloud procedure.

**Required format:**
- **Purpose**: Brief description of what this procedure does
- **Module Context**: How it fits within the {proc['module'].upper()} module
- **Key Features**: Main functionality and capabilities  
- **Database Objects**: Important tables, procedures, functions used
- **Business Logic**: Core business rules and validation logic
- **Integration Points**: How it connects with other IFS components
- **Technical Notes**: Important implementation details and patterns

**Guidelines:**
- Focus on IFS Cloud {proc['module'].upper()} business context and patterns
- Highlight important business logic and validation rules
- Explain the procedure's role in the overall {proc['module']} workflow
- Note any security checks or authorization logic
- Keep technical but accessible to other developers
- Aim for 300-500 words

**Summary:**'''

            planning_item = {
                'id': i + 1,
                'procedure_name': proc['name'],
                'module': proc['module'],
                'file_path': proc['file_path'],
                'line_start': proc['line_start'],
                'line_end': proc['line_end'],
                'complexity_score': proc['complexity_score'],
                'content_length': proc['content_length'],
                'prompt': prompt,
                'enhanced_context': enhanced_context,
                'created_at': datetime.now().isoformat()
            }
            
            planning_data.append(planning_item)
        
        # Save planning file
        with open(self.planning_file, 'w', encoding='utf-8') as f:
            json.dump(planning_data, f, indent=2)
        
        print(f"💾 Planning saved to: {self.planning_file}")
        
        return planning_data
    
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
            keyword_note += f"\nThese keywords indicate specific IFS Cloud business patterns and should be emphasized in the summary.\n"
            return context + keyword_note
            
        return context
    
    def clear_gpu_memory(self):
        """Aggressively clear GPU memory to free fragmented memory."""
        if torch.cuda.is_available():
            # Multiple clearing passes
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()  # Inter-process cleanup
            
            import gc
            gc.collect()  # Python garbage collection
            
            torch.cuda.empty_cache()  # Second pass
            torch.cuda.synchronize()
            
            # Print memory status
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"🧹 GPU Memory cleared - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    def load_model_smart_gpu(self):
        """Load model with smart GPU/CPU distribution."""
        logger.info(f"Loading model with smart memory management: {self.config['model_name']}")
        
        # Clear GPU memory once at startup
        self.clear_gpu_memory()
        
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
            
            # Load model with 4-bit quantization and allow CPU offloading if needed
            # Configure BitsAndBytes quantization for memory efficiency
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True  # This goes in BitsAndBytesConfig
            )
            
            model_kwargs = {
                "quantization_config": bnb_config,
                "torch_dtype": torch.bfloat16,  # Use bfloat16 for RTX cards
                "device_map": "auto",  # Let transformers decide optimal placement
                "trust_remote_code": True,
                "use_cache": True,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa"
            }
            
            print("🚀 Loading quantized 14B model with auto device mapping...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"], **model_kwargs
            )
            
            # Check where the model ended up
            device_info = {}
            for name, param in self.model.named_parameters():
                device = str(param.device)
                if device not in device_info:
                    device_info[device] = 0
                device_info[device] += param.numel()
                
            print("📊 Model distribution:")
            for device, param_count in device_info.items():
                print(f"  {device}: {param_count:,} parameters")
            
            # Print memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_summary_from_plan(self, planning_item: Dict[str, Any]) -> Optional[str]:
        """Generate summary from pre-computed planning item."""
        try:
            prompt = planning_item['prompt']
            
            # Tokenize with manageable context window (balance quality vs memory)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=3500)
            
            # Move inputs to model's primary device
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Generate with optimized settings
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=500,  # Further reduced to stay within 16GB VRAM
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
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
            logger.error(f"Error generating summary for {planning_item['procedure_name']}: {e}")
            return None
    
    def run_diversified_generation(self):
        """Run the diversified batch generation process."""
        logger.info("🚀 Starting diversified batch generation...")
        
        # Check for existing planning file
        existing_plans = list(self.planning_dir.glob("diversified_plan_*.json"))
        
        if existing_plans:
            # Use most recent planning file
            latest_plan = max(existing_plans, key=lambda p: p.stat().st_mtime)
            print(f"📂 Found existing planning file: {latest_plan.name}")
            print("🔄 Loading existing plan to save time...")
            
            with open(latest_plan, 'r', encoding='utf-8') as f:
                planning_data = json.load(f)
                
            print(f"✅ Loaded {len(planning_data)} pre-planned procedures")
            
        else:
            # Step 1: Create diversified plan
            planning_data = self.create_diversified_plan()
        
        # Step 2: Load model
        print("\n🤖 Loading model for inference...")
        self.clear_gpu_memory()
        self.load_model_smart_gpu()
        
        # Step 3: Process planned items
        print(f"\n⚡ Starting inference on {len(planning_data)} diversified procedures...")
        print("📊 Memory-optimized settings: 3500 tokens + 500 generation")
        
        results = []
        start_time = time.time()
        
        # Use no_grad context for entire batch processing
        with torch.no_grad():
            for i, planning_item in enumerate(tqdm(planning_data, desc="Processing diversified procedures")):
                
                # More aggressive cache clearing every 10 procedures to prevent memory buildup
                if i > 0 and i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Print memory status after clearing
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"🧹 Memory cleared at {i}: Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
                try:
                    # Generate summary
                    proc_start_time = time.time()
                    summary = self.generate_summary_from_plan(planning_item)
                    proc_time = time.time() - proc_start_time
                    
                    if summary:
                        # Save result
                        result = {
                            'id': planning_item['id'],
                            'procedure_name': planning_item['procedure_name'],
                            'module': planning_item['module'],
                            'file_path': planning_item['file_path'],
                            'line_start': planning_item['line_start'],
                            'line_end': planning_item['line_end'],
                            'complexity_score': planning_item['complexity_score'],
                            'content_length': planning_item['content_length'],
                            'summary': summary,
                            'generation_time': proc_time,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results.append(result)
                        
                        # Save incrementally
                        with open(self.output_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2)
                        
                        # Print progress
                        total_time = time.time() - start_time
                        avg_time = total_time / len(results) if results else 0
                        remaining = len(planning_data) - len(results)
                        eta_minutes = (remaining * avg_time) / 60
                        tokens_per_sec = 500 / proc_time if proc_time > 0 else 0  # Updated token rate
                        
                        print(f"\n✅ {len(results)}/{len(planning_data)} - [{planning_item['module']}] {planning_item['procedure_name']}")
                        print(f"⏱️  This: {proc_time:.1f}s ({tokens_per_sec:.1f} t/s), Avg: {avg_time:.1f}s, ETA: {eta_minutes:.1f}min")
                        print(f"🎯 Complexity: {planning_item['complexity_score']:.1f}")
                        
                        # Show GPU memory usage periodically
                        if len(results) % 5 == 0 and torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                        
                except Exception as e:
                    logger.error(f"Error processing {planning_item['procedure_name']}: {e}")
                    # Aggressive cache clearing after errors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_tokens_per_sec = (len(results) * 500) / total_time if total_time > 0 else 0
        
        # Final statistics
        print(f"\n🎉 Completed {len(results)} diversified summaries in {total_time/60:.1f} minutes")
        print(f"🚀 Average performance: {avg_tokens_per_sec:.1f} tokens/second")
        print(f"💾 Results saved to: {self.output_file}")
        print(f"📋 Planning saved to: {self.planning_file}")
        
        # Module distribution statistics
        module_stats = {}
        for result in results:
            module = result['module']
            if module not in module_stats:
                module_stats[module] = 0
            module_stats[module] += 1
        
        print(f"\n📊 Module diversity achieved:")
        for module, count in module_stats.items():
            print(f"  {module}: {count} procedures")


def main():
    """Main entry point."""
    # Check optimizations
    try:
        print("✅ PyTorch SDPA available:", hasattr(torch.backends.cuda, 'sdp_kernel'))
    except:
        print("⚠️ PyTorch SDPA check failed")
    
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
    
    print("\n🎯 Diversified IFS Cloud Batch Summary Generator")
    print("=" * 60)
    
    # Configuration
    config = {
        "model_name": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "ifs_source_path": "C:/repos/_ifs/25.1.0",
        "target_summaries": 200,
        "max_length": 4000,  # Reduced context window for 16GB VRAM
        "batch_size": 8
    }
    
    print(f"📊 Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Mode: 4-bit quantized with diversified selection")
    print(f"  Target: {config['target_summaries']} summaries")
    print(f"  Context: 3500 tokens + 500 generation (memory optimized)")
    print(f"  Modules: {', '.join(TARGET_MODULES)}")
    print(f"  Source: {config['ifs_source_path']}")
    print(f"  Memory: ~8GB VRAM (vs ~24GB for FP16 14B)")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 RTX optimizations enabled on {gpu_name}")
        print(f"💾 Available VRAM: {total_memory:.1f}GB")
    else:
        print("❌ CUDA not available")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run generation
    generator = DiversifiedBatchGenerator(config)
    generator.run_diversified_generation()


if __name__ == "__main__":
    main()
