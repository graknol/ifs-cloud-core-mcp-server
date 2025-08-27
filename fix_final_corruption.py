#!/usr/bin/env python3
"""
Final fix for the remaining 2 corrupted entries.
"""

import json
import re
from pathlib import Path

def fix_final_corruption(data):
    """
    Fix the remaining specific corruption cases.
    """
    signature = data['signature']
    summary = data['summary']
    parameters = data['parameters']
    procedure_name = data['procedure_name']
    
    # Check if this is one of the problematic entries
    if procedure_name == "Handle_Emission_Insert___":
        # This entry has parameters in summary and empty signature
        param_text = summary.strip()
        
        # Build proper signature
        new_signature = f"Handle_Emission_Insert___({param_text})"
        new_parameters = f"({param_text})"
        new_summary = "This procedure handles emission insert operations with the specified parameters."
        
        data['signature'] = new_signature
        data['parameters'] = new_parameters  
        data['summary'] = new_summary
        
        print(f"🔧 Fixed: {procedure_name}")
        return data
        
    elif procedure_name == "Modify_Period" and summary.startswith("emp_no_"):
        # This entry has incomplete signature and parameters in summary
        param_text = parameters.strip()
        if param_text.startswith('(') and param_text.endswith(')'):
            param_text = param_text[1:-1]  # Remove parentheses
        
        summary_param_text = summary.strip()
        
        # Combine parameters
        full_params = f"{param_text}, {summary_param_text}".strip()
        if full_params.startswith(','):
            full_params = full_params[1:].strip()
        if full_params.endswith(','):
            full_params = full_params[:-1].strip()
            
        new_signature = f"Modify_Period({full_params})"
        new_parameters = f"({full_params})"
        new_summary = "This procedure modifies period information with the specified parameters."
        
        data['signature'] = new_signature
        data['parameters'] = new_parameters
        data['summary'] = new_summary
        
        print(f"🔧 Fixed: {procedure_name}")
        return data
    
    return data

def main():
    input_file = Path("plsql_analysis_combined/clean_comprehensive_fix.jsonl")
    output_file = Path("plsql_analysis_combined/clean_final_fix.jsonl")
    
    print("🔧 FINAL CORRUPTION FIX")
    print("=" * 30)
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # Check if entry needs fixing
                original_summary = data['summary']
                fixed_data = fix_final_corruption(data)
                
                if fixed_data['summary'] != original_summary:
                    fixed_count += 1
                
                # Write the (potentially fixed) data
                outfile.write(json.dumps(fixed_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                # Write the original line if it can't be parsed
                outfile.write(line)
    
    print(f"\n✅ FINAL FIX COMPLETE:")
    print(f"   Total entries: {total_count}")
    print(f"   Fixed entries: {fixed_count}")
    print(f"   Output file: {output_file}")

if __name__ == "__main__":
    main()
