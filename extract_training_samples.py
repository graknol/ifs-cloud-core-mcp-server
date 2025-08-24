#!/usr/bin/env python3
"""
Smart Sample Extraction for Fine-tuning Code Summarization Models

This script extracts high-quality PL/SQL function samples for training
a code summarization model, using PageRank and complexity analysis.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PLSQLSampleExtractor:
    """Extract high-quality PL/SQL samples for fine-tuning."""
    
    def __init__(self, version_path: Path):
        self.version_path = Path(version_path)
        self.pagerank_scores = self._load_pagerank_scores()
        
    def _load_pagerank_scores(self) -> Dict[str, float]:
        """Load PageRank scores from ranked.jsonl"""
        scores = {}
        ranked_file = self.version_path / "ranked.jsonl"
        
        if ranked_file.exists():
            with open(ranked_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    scores[item['file_path']] = item.get('pagerank_score', 0.0)
        
        return scores
    
    def analyze_function_complexity(self, function_text: str) -> Dict:
        """Analyze function complexity metrics."""
        lines = function_text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Count decision points for cyclomatic complexity
        decision_keywords = ['IF', 'CASE', 'WHEN', 'LOOP', 'FOR', 'WHILE', 'EXCEPTION']
        complexity = 1  # Base complexity
        
        for line in non_empty_lines:
            line_upper = line.upper()
            for keyword in decision_keywords:
                complexity += line_upper.count(keyword)
        
        return {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'cyclomatic_complexity': complexity,
            'estimated_tokens': len(function_text.split()) * 1.3  # Rough estimate
        }
    
    def extract_functions_from_file(self, file_path: Path) -> List[Dict]:
        """Extract individual functions/procedures from a PL/SQL file using proper indentation rules."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []

        functions = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts a function/procedure at position 0 (no indentation)
            if (line.startswith('FUNCTION ') or line.startswith('PROCEDURE ')) and not line.startswith('   '):
                function_lines = [line]
                function_name = self._extract_function_name(line)
                
                i += 1
                
                # Extract the complete function until we find END at position 0
                while i < len(lines):
                    current_line = lines[i]
                    function_lines.append(current_line)
                    
                    # Check if this is the END of our function (at position 0)
                    if (current_line.startswith('END ') or current_line.strip() == 'END;' or 
                        (current_line.startswith('END') and current_line[3:4] in [' ', ';', '\n', ''])):
                        # This is the END of our main function
                        break
                    
                    i += 1
                
                func_text = '\n'.join(function_lines)
                
                # Skip very small or very large functions
                complexity = self.analyze_function_complexity(func_text)
                if complexity['code_lines'] < 10 or complexity['code_lines'] > 500:
                    i += 1
                    continue
                
                # Skip trivial functions (simple getters/setters)
                if self._is_trivial_function(func_text, function_name):
                    i += 1
                    continue
                
                functions.append({
                    'function_name': function_name,
                    'function_text': func_text,
                    'complexity': complexity,
                    'file_path': file_path,
                    'position': len(functions)
                })
            
            i += 1
        
        # Sort by complexity and return best candidates
        functions.sort(key=lambda x: x['complexity']['cyclomatic_complexity'], reverse=True)
        return functions[:10]  # Top 10 most complex functions per file
    
    def _extract_function_name(self, declaration_line: str) -> str:
        """Extract function/procedure name from declaration line."""
        # Remove FUNCTION/PROCEDURE keyword and extract name
        cleaned = re.sub(r'^(FUNCTION|PROCEDURE)\s+', '', declaration_line, flags=re.IGNORECASE)
        # Extract name before parameters or spaces
        name_match = re.match(r'(\w+)', cleaned)
        return name_match.group(1) if name_match else 'unknown'
    
    def _is_trivial_function(self, function_text: str, function_name: str) -> bool:
        """Check if this is a trivial function (getter/setter) that should be skipped."""
        
        # Skip obvious getters/setters
        if any(pattern in function_name.upper() for pattern in ['GET_', 'SET_', 'IS_', 'HAS_']):
            # If it's short and simple, skip it
            if len(function_text.split('\n')) < 20:
                return True
        
        # Skip functions that are mostly just assignments
        lines = [line.strip() for line in function_text.split('\n') if line.strip()]
        assignment_lines = sum(1 for line in lines if ':=' in line or 'RETURN ' in line.upper())
        
        if assignment_lines > 0 and assignment_lines / len(lines) > 0.7:
            return True  # More than 70% assignments - likely trivial
        
        return False
    
    def get_file_summary(self, file_path: Path) -> str:
        """Generate a brief summary of what the file does."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return "File summary not available"
        
        # Extract package/file name
        file_stem = file_path.stem
        
        # Look for common patterns to infer purpose
        content_upper = content.upper()
        
        if 'CUSTOMER' in content_upper and 'ORDER' in content_upper:
            return f"Customer order management and processing - {file_stem}"
        elif 'INVOICE' in content_upper:
            return f"Invoice processing and management - {file_stem}"
        elif 'FINANCIAL' in content_upper or 'ACCOUNT' in content_upper:
            return f"Financial operations and accounting - {file_stem}"
        elif 'INVENTORY' in content_upper:
            return f"Inventory management and tracking - {file_stem}"
        else:
            return f"Business logic and data operations - {file_stem}"
    
    def create_training_sample(self, function_info: Dict, all_functions: List[Dict]) -> Dict:
        """Create a rich training sample with context."""
        
        # Find previous and next functions for context
        position = function_info['position']
        file_functions = [f for f in all_functions if f['file_path'] == function_info['file_path']]
        
        prev_func = None
        next_func = None
        
        for func in file_functions:
            if func['position'] == position - 1:
                prev_func = func
            elif func['position'] == position + 1:
                next_func = func
        
        # Create context
        file_path = Path(function_info['file_path'])
        
        # Apply smart truncation for UnixCoder compatibility
        original_code = function_info['function_text']
        truncated_code, truncation_metadata = self._smart_truncate_for_unixcoder(original_code)
        
        context = {
            'api_name': file_path.stem,
            'module': file_path.parent.name if file_path.parent.name != 'database' else file_path.parent.parent.name,
            'file_summary': self.get_file_summary(file_path),
            'function_name': function_info['function_name'],
            'previous_function': prev_func['function_name'] if prev_func else None,
            'next_function': next_func['function_name'] if next_func else None,
            'complexity_metrics': function_info['complexity'],
            'pagerank_score': self.pagerank_scores.get(str(file_path), 0.0),
            'truncation_metadata': truncation_metadata
        }
        
        return {
            'id': f"{file_path.stem}_{function_info['function_name']}",
            'context': context,
            'code': truncated_code,
            'original_code_length': len(original_code),
            'summary': None  # To be filled by Claude
        }
    
    def _smart_truncate_for_unixcoder(self, code: str, max_chars: int = 1800) -> tuple[str, dict]:
        """
        Intelligent truncation preserving PL/SQL structure for UnixCoder.
        Reserve 200 chars for context metadata.
        
        Returns:
            tuple: (truncated_code, truncation_metadata)
        """
        original_length = len(code)
        
        if original_length <= max_chars:
            return code, {
                "original_length": original_length,
                "truncated_length": original_length,
                "truncation_method": "no_truncation"
            }
        
        lines = code.split('\n')
        essential_lines = []
        
        # Strategy 1: Always include function/procedure declaration
        declaration_found = False
        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            if any(keyword in line_upper for keyword in ['FUNCTION', 'PROCEDURE']) and not declaration_found:
                # Include 2 lines before (comments) and 10 lines after declaration
                start_idx = max(0, i-2)
                end_idx = min(len(lines), i+12)
                essential_lines.extend(lines[start_idx:end_idx])
                declaration_found = True
                break
        
        # Strategy 2: Add key business logic from middle section
        current_length = len('\n'.join(essential_lines))
        remaining_chars = max_chars - current_length
        
        if remaining_chars > 500 and len(lines) > 20:
            # Find middle section with business logic
            middle_start = len(lines) // 3
            middle_end = min(len(lines), len(lines) * 2 // 3)
            
            # Look for important keywords in middle section
            important_lines = []
            for line in lines[middle_start:middle_end]:
                line_upper = line.strip().upper()
                if any(keyword in line_upper for keyword in [
                    'IF', 'THEN', 'ELSE', 'ELSIF', 'WHEN', 'LOOP', 
                    'CURSOR', 'SELECT', 'UPDATE', 'INSERT', 'DELETE',
                    'VALIDATE', 'CHECK', 'RAISE', 'EXCEPTION'
                ]):
                    important_lines.append(line)
                
                # Stop if we're approaching the limit
                if len('\n'.join(important_lines)) > remaining_chars - 200:
                    break
            
            if important_lines:
                essential_lines.append("-- ... key business logic ...")
                essential_lines.extend(important_lines)
        
        # Strategy 3: Include exception handling (usually at end)
        current_length = len('\n'.join(essential_lines))
        remaining_chars = max_chars - current_length
        
        if remaining_chars > 200:
            for i in range(len(lines)-1, max(0, len(lines)-20), -1):
                if 'EXCEPTION' in lines[i].upper():
                    exception_section = lines[i:min(len(lines), i+8)]
                    exception_text = '\n'.join(exception_section)
                    if len(exception_text) <= remaining_chars:
                        essential_lines.append("-- ... exception handling ...")
                        essential_lines.extend(exception_section)
                    break
        
        # Final truncation if still too long
        truncated_code = '\n'.join(essential_lines)
        if len(truncated_code) > max_chars:
            truncated_code = truncated_code[:max_chars-3] + "..."
        
        return truncated_code, {
            "original_length": original_length,
            "truncated_length": len(truncated_code),
            "truncation_method": "smart_structure_preserve",
            "truncation_ratio": len(truncated_code) / original_length
        }
    
    def extract_stratified_samples(self, target_count: int = 200) -> List[Dict]:
        """Extract samples using stratified sampling approach."""
        
        # Get all PL/SQL files with PageRank scores
        source_dir = self.version_path / "source"
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return []
        
        plsql_files = list(source_dir.rglob("*.plsql"))
        logger.info(f"Found {len(plsql_files)} PL/SQL files")
        
        # Sort by PageRank score
        scored_files = [
            (f, self.pagerank_scores.get(str(f), 0.0)) 
            for f in plsql_files
        ]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        # Stratified sampling
        high_tier = scored_files[:50]      # Top 50 by PageRank
        mid_tier = scored_files[50:150]    # Middle tier
        low_tier = scored_files[150:300]   # Lower tier (if available)
        
        all_samples = []
        
        for tier, tier_name in [(high_tier, "high"), (mid_tier, "mid"), (low_tier, "low")]:
            tier_samples = []
            
            for file_path, score in tier:
                functions = self.extract_functions_from_file(file_path)
                tier_samples.extend(functions)
            
            # Sort by complexity and diversity
            tier_samples.sort(key=lambda x: x['complexity']['cyclomatic_complexity'], reverse=True)
            
            # Take best samples from this tier
            samples_per_tier = min(target_count // 3, len(tier_samples))
            selected = tier_samples[:samples_per_tier]
            
            logger.info(f"Selected {len(selected)} samples from {tier_name} tier")
            all_samples.extend(selected)
        
        # Create training samples with rich context
        training_samples = []
        for func_info in all_samples:
            sample = self.create_training_sample(func_info, all_samples)
            training_samples.append(sample)
        
        return training_samples[:target_count]


def main():
    """Generate training samples for fine-tuning."""
    from src.ifs_cloud_mcp_server.directory_utils import get_data_directory
    
    data_dir = get_data_directory()
    version_dir = data_dir / "versions" / "25.1.0"
    
    if not version_dir.exists():
        logger.error(f"Version directory not found: {version_dir}")
        return
    
    extractor = PLSQLSampleExtractor(version_dir)
    samples = extractor.extract_stratified_samples(200)
    
    # Save samples for Claude processing
    output_file = Path("training_samples_for_claude.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, indent=None) + '\n')
    
    logger.info(f"✅ Generated {len(samples)} training samples in {output_file}")
    
    # Print statistics
    complexities = [s['context']['complexity_metrics']['cyclomatic_complexity'] for s in samples]
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    
    # Truncation statistics
    truncated_samples = [s for s in samples if s['context']['truncation_metadata']['truncation_method'] != 'no_truncation']
    if truncated_samples:
        avg_truncation_ratio = sum(s['context']['truncation_metadata']['truncation_ratio'] for s in truncated_samples) / len(truncated_samples)
        truncation_stats = f"""
� Truncation Statistics:
- Samples truncated: {len(truncated_samples)}/{len(samples)} ({len(truncated_samples)/len(samples)*100:.1f}%)
- Average truncation ratio: {avg_truncation_ratio:.2f}
- Average original length: {sum(s['original_code_length'] for s in truncated_samples) / len(truncated_samples):.0f} chars
- Average truncated length: {sum(s['context']['truncation_metadata']['truncated_length'] for s in truncated_samples) / len(truncated_samples):.0f} chars"""
    else:
        truncation_stats = "\n📝 No samples required truncation (all under 1800 chars)"
    
    print(f"""
�📊 Sample Statistics:
- Total samples: {len(samples)}
- Average cyclomatic complexity: {avg_complexity:.2f}
- Complexity range: {min(complexities)} - {max(complexities)}
- Files covered: {len(set(s['context']['api_name'] for s in samples))}{truncation_stats}
""")

if __name__ == "__main__":
    main()
