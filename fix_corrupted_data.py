#!/usr/bin/env python3
"""
Fix corrupted entries in the JSONL dataset where signature and summary fields have been mixed up.
"""

import json
import re
from pathlib import Path

def fix_corrupted_entry(data):
    """
    Fix corrupted entries where:
    1. Signature ends with '(' and parameters are in summary
    2. Summary starts with 'RETURN <datatype>'
    """
    signature = data['signature']
    summary = data['summary']
    procedure_name = data['procedure_name']
    
    # Check for Type 1 corruption: signature ends with '(' 
    type1_corruption = (signature.endswith(' (') or signature.endswith('(')) and (
        summary.startswith('rec_') or 'IN VARCHAR2' in summary[:50] or 
        'IN NUMBER' in summary[:50] or 'IN OUT' in summary[:50] or
        'RETURN' in summary[:100]
    )
    
    # Check for Type 2 corruption: summary starts with RETURN
    type2_corruption = summary.strip().upper().startswith('RETURN ')
    
    if not (type1_corruption or type2_corruption):
        return data  # Not corrupted, return as-is
    
    print(f"🔧 Fixing: {procedure_name}")
    
    if type2_corruption:
        # Handle Type 2: Summary starts with RETURN
        print(f"  Type 2: Return type in summary")
        
        # Find where the actual summary starts after the return type
        summary_upper = summary.upper()
        
        # Look for patterns that indicate where summary actually begins
        patterns = [
            ' CALCULATES ',
            ' DETERMINES ',
            ' CHECKS ',
            ' GENERATES ',
            ' FETCHES ',
            ' RETRIEVES ',
            ' RETURNS ',
            ' CREATES ',
            ' UPDATES ',
            ' DELETES ',
            ' VALIDATES ',
            ' PROCESSES ',
            ' HANDLES ',
        ]
        
        return_part = ""
        actual_summary = summary
        
        for pattern in patterns:
            pos = summary_upper.find(pattern)
            if pos > 0:
                # Found a likely start of actual summary
                return_part = summary[:pos].strip()
                actual_summary = summary[pos:].strip()
                break
        
        # If no pattern found, try to split on the first word after RETURN TYPE
        if return_part == "":
            words = summary.split()
            if len(words) >= 3 and words[0].upper() == 'RETURN':
                # Assume format is "RETURN TYPE Description..."
                return_part = f"{words[0]} {words[1]}"
                actual_summary = ' '.join(words[2:])
        
        # Build complete signature with return type
        if return_part and 'RETURN' in return_part.upper():
            # Add return type to signature
            if not signature.endswith(')'):
                new_signature = f"{signature}) {return_part}"
            else:
                new_signature = f"{signature} {return_part}"
        else:
            new_signature = signature
            
        # Clean up the actual summary
        if not actual_summary or len(actual_summary.strip()) < 10:
            actual_summary = "This function performs the specified operation and returns a value."
        
        data['signature'] = new_signature
        data['summary'] = actual_summary.strip()
        
    elif type1_corruption:
        # Handle Type 1: Parameters in summary (existing logic)
        print(f"  Type 1: Parameters in summary")
        
        # Split the summary to separate parameters from actual summary
        lines = summary.split('\n', 1)
        if len(lines) < 2:
            # Try to split on ') ' pattern to separate params from summary
            parts = summary.split(') ', 1)
            if len(parts) >= 2:
                params_part = parts[0] + ')'
                summary_part = parts[1]
            else:
                # If we can't cleanly split, try to find where parameters end
                # Look for common summary starting patterns
                summary_patterns = [
                    r'\) (Removes|Generates|Updates|Inserts|Deletes|Creates|Validates|Sets|Gets|Checks|Processes|Handles|Imports|Determines|Identifies)',
                    r'\) [A-Z][a-z]',  # Capital letter starting actual summary
                ]
                
                params_part = ""
                summary_part = summary
                for pattern in summary_patterns:
                    match = re.search(pattern, summary)
                    if match:
                        split_pos = match.start() + 1  # Include the closing paren
                        params_part = summary[:split_pos]
                        summary_part = summary[split_pos:].strip()
                        break
                
                # If still no split found, look for return type patterns
                if not params_part and 'RETURN' in summary:
                    return_match = re.search(r'RETURN [A-Z_]+(?:%[A-Z_]+)?', summary)
                    if return_match:
                        split_pos = return_match.end()
                        params_part = summary[:split_pos]
                        # Look for summary after return type
                        remaining = summary[split_pos:].strip()
                        if remaining and len(remaining) > 10:
                            summary_part = remaining
                        else:
                            summary_part = "This procedure performs the specified operation."
        else:
            params_part = lines[0]
            summary_part = lines[1].strip()
        
        # Reconstruct the proper signature
        if params_part.strip():
            # Clean up the parameters
            params_part = params_part.strip()
            if not params_part.endswith(')'):
                params_part += ')'
            
            # Build complete signature
            new_signature = f"{procedure_name}({params_part[params_part.find('_'):] if '_' in params_part else params_part}"
            if not new_signature.endswith(')'):
                new_signature += ')'
                
            # Handle return types
            if 'RETURN' in params_part:
                new_signature = new_signature.replace(') RETURN', ') RETURN')
        else:
            # No parameters found, use empty params
            new_signature = f"{procedure_name}()"
        
        # Clean up the summary
        if not summary_part or len(summary_part.strip()) < 10:
            summary_part = "This procedure performs the specified operation."
        
        # Update the data
        data['signature'] = new_signature
        data['summary'] = summary_part.strip()
    
    # Update parameters field to match the signature
    # Update parameters field to match the signature
    new_signature = data['signature']
    if '(' in new_signature and ')' in new_signature:
        param_start = new_signature.find('(')
        param_end = new_signature.rfind(')')
        param_content = new_signature[param_start+1:param_end].strip()
        
        # Extract just the parameters part (before any RETURN)
        if 'RETURN' in param_content:
            param_content = param_content.split('RETURN')[0].strip().rstrip(',').strip()
        
        if param_content:
            data['parameters'] = f"({param_content})"
        else:
            data['parameters'] = "()"
    else:
        data['parameters'] = "()"
    
    return data

def main():
    input_file = Path("plsql_analysis_combined/clean.jsonl")
    output_file = Path("plsql_analysis_combined/clean_fixed.jsonl")
    
    print("🔧 FIXING CORRUPTED DATA IN DATASET")
    print("=" * 50)
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # Check if entry is corrupted and fix it
                original_signature = data['signature']
                fixed_data = fix_corrupted_entry(data)
                
                if fixed_data['signature'] != original_signature:
                    fixed_count += 1
                    print(f"  Line {line_num}: {data['procedure_name']}")
                    print(f"    Old: {original_signature}")
                    print(f"    New: {fixed_data['signature']}")
                    print()
                
                # Write the (potentially fixed) data
                outfile.write(json.dumps(fixed_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                # Write the original line if it can't be parsed
                outfile.write(line)
    
    print(f"✅ PROCESSING COMPLETE:")
    print(f"   Total entries: {total_count}")
    print(f"   Fixed entries: {fixed_count}")
    print(f"   Output file: {output_file}")
    
    if fixed_count > 0:
        print(f"\n🔄 To use the fixed dataset, run:")
        print(f"   mv {output_file} {input_file}")

if __name__ == "__main__":
    main()
