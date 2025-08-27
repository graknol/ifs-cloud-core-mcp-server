#!/usr/bin/env python3
"""
Comprehensive fix for all corruption patterns in the dataset.
"""

import json
import re
from pathlib import Path

def fix_corrupted_entry(data):
    """
    Fix all types of corruption in an entry.
    """
    signature = data['signature']
    summary = data['summary']
    parameters = data['parameters']
    procedure_name = data['procedure_name']
    
    original_signature = signature
    
    # Detect corruption types
    corruption_types = []
    
    # Type 1: Incomplete signature ending with '('
    if signature.endswith(' (') or signature.endswith('('):
        corruption_types.append('incomplete_signature')
    
    # Type 2: Summary starts with RETURN
    if summary.strip().upper().startswith('RETURN '):
        corruption_types.append('return_in_summary')
    
    # Type 3: Parameters field is '(' and summary has parameters
    if parameters.strip() == '(' and ('IN ' in summary[:100] or 'OUT ' in summary[:100] or 'RETURN ' in summary[:100]):
        corruption_types.append('params_in_summary')
    
    # Type 4: Summary starts with parameter-like content
    summary_start = summary.strip()[:50].upper()
    if (summary_start.startswith('CALC_') or 
        summary_start.startswith('REC_') or
        'IN VARCHAR2' in summary[:100] or
        'IN NUMBER' in summary[:100] or
        'IN BOOLEAN' in summary[:100]):
        corruption_types.append('param_like_summary')
    
    if not corruption_types:
        return data  # Not corrupted
    
    print(f"🔧 Fixing: {procedure_name} - Types: {corruption_types}")
    
    # Fix based on corruption type
    new_signature = signature
    new_summary = summary
    new_parameters = parameters
    
    if 'incomplete_signature' in corruption_types or 'params_in_summary' in corruption_types or 'param_like_summary' in corruption_types:
        # Extract parameters from summary
        if ')' in summary:
            # Find the closing parenthesis that ends the parameter list
            paren_pos = summary.find(')')
            param_part = summary[:paren_pos + 1].strip()
            summary_part = summary[paren_pos + 1:].strip()
            
            # Clean parameter part
            if param_part.startswith('('):
                param_part = param_part[1:]  # Remove leading (
            if param_part.endswith(')'):
                param_part = param_part[:-1]  # Remove trailing )
            
            # Build new signature
            if signature.endswith('(') or signature.endswith(' ('):
                # Remove trailing (
                base_sig = signature.rstrip(' (')
                if param_part:
                    new_signature = f"{base_sig}({param_part})"
                else:
                    new_signature = f"{base_sig}()"
            else:
                # Signature might be complete already
                if param_part and '(' not in signature:
                    new_signature = f"{signature}({param_part})"
            
            # Extract any return type from parameter part or summary
            return_match = re.search(r'(RETURN [A-Z_%.]+(?:\s*PIPELINED)?)', param_part + ' ' + summary_part, re.IGNORECASE)
            if return_match:
                return_type = return_match.group(1)
                # Remove return type from parameter part
                param_part = re.sub(r'RETURN [A-Z_%.]+(?:\s*PIPELINED)?', '', param_part, flags=re.IGNORECASE).strip().rstrip(',').strip()
                # Add return type to signature
                new_signature = f"{new_signature.rstrip(')')})" + f" {return_type}"
                # Remove return type from summary
                summary_part = re.sub(r'RETURN [A-Z_%.]+(?:\s*PIPELINED)?', '', summary_part, flags=re.IGNORECASE).strip()
            
            # Update parameters field
            new_parameters = f"({param_part})" if param_part else "()"
            
            # Clean up summary - remove parameter remnants
            new_summary = summary_part.strip()
            
            # If summary is empty or too short, create a default one
            if not new_summary or len(new_summary) < 20:
                if 'RETURN' in new_signature.upper():
                    new_summary = "This function performs the specified operation and returns a value."
                else:
                    new_summary = "This procedure performs the specified operation."
    
    elif 'return_in_summary' in corruption_types:
        # Handle return type in summary
        words = summary.split()
        if len(words) >= 2 and words[0].upper() == 'RETURN':
            return_type = f"{words[0]} {words[1]}"
            new_signature = f"{signature} {return_type}"
            new_summary = ' '.join(words[2:]).strip()
            
            if not new_summary or len(new_summary) < 20:
                new_summary = "This function performs the specified operation and returns a value."
    
    # Final cleanup
    new_signature = re.sub(r'\s+', ' ', new_signature).strip()
    new_summary = re.sub(r'\s+', ' ', new_summary).strip()
    
    # Update data
    data['signature'] = new_signature
    data['summary'] = new_summary
    data['parameters'] = new_parameters
    
    return data

def main():
    input_file = Path("plsql_analysis_combined/clean.jsonl")
    output_file = Path("plsql_analysis_combined/clean_comprehensive_fix.jsonl")
    
    print("🔧 COMPREHENSIVE CORRUPTION FIX")
    print("=" * 50)
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # Check if entry needs fixing
                original_signature = data['signature']
                fixed_data = fix_corrupted_entry(data)
                
                if fixed_data['signature'] != original_signature or fixed_data['summary'] != data['summary'] or fixed_data['parameters'] != data['parameters']:
                    fixed_count += 1
                
                # Write the (potentially fixed) data
                outfile.write(json.dumps(fixed_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                # Write the original line if it can't be parsed
                outfile.write(line)
    
    print(f"\n✅ COMPREHENSIVE FIX COMPLETE:")
    print(f"   Total entries: {total_count}")
    print(f"   Fixed entries: {fixed_count}")
    print(f"   Output file: {output_file}")
    
    if fixed_count > 0:
        print(f"\n🔄 To use the comprehensively fixed dataset, run:")
        print(f"   mv {output_file} {input_file}")

if __name__ == "__main__":
    main()
