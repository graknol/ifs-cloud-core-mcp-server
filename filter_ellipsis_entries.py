#!/usr/bin/env python3
"""
Filter out entries with ellipsis (...) from the dataset as they indicate incomplete/truncated data.
"""

import json
from pathlib import Path

def has_ellipsis(data):
    """
    Check if an entry contains ellipsis in signature or summary fields.
    """
    signature = data.get('signature', '')
    summary = data.get('summary', '')
    
    return '...' in signature or '...' in summary

def main():
    input_file = Path("plsql_analysis_combined/clean.jsonl")
    output_file = Path("plsql_analysis_combined/clean_filtered.jsonl")
    
    print("🧹 FILTERING OUT ENTRIES WITH ELLIPSIS")
    print("=" * 50)
    
    filtered_count = 0
    kept_count = 0
    total_count = 0
    
    filtered_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                if has_ellipsis(data):
                    filtered_count += 1
                    filtered_entries.append({
                        'line': line_num,
                        'procedure_name': data.get('procedure_name', ''),
                        'signature': data.get('signature', ''),
                        'summary': data.get('summary', '')[:100] + ('...' if len(data.get('summary', '')) > 100 else ''),
                        'reason': 'Contains ellipsis (...)'
                    })
                    print(f"🚫 Filtered line {line_num}: {data.get('procedure_name', '')}")
                    if '...' in data.get('signature', ''):
                        print(f"   Reason: Ellipsis in signature")
                    if '...' in data.get('summary', ''):
                        print(f"   Reason: Ellipsis in summary")
                else:
                    # Keep this entry
                    kept_count += 1
                    outfile.write(line)
                    
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                # Skip malformed lines
                filtered_count += 1
    
    print(f"\n✅ FILTERING COMPLETE:")
    print(f"   Total entries processed: {total_count:,}")
    print(f"   Entries kept: {kept_count:,}")
    print(f"   Entries filtered out: {filtered_count}")
    print(f"   Filter rate: {filtered_count/total_count*100:.3f}%")
    print(f"   Output file: {output_file}")
    
    print(f"\n📋 FILTERED ENTRIES:")
    for entry in filtered_entries:
        print(f"   • {entry['procedure_name']} (Line {entry['line']}) - {entry['reason']}")
    
    if filtered_count > 0:
        print(f"\n🔄 To use the filtered dataset, run:")
        print(f"   mv {output_file} {input_file}")
        print(f"\n💡 This will remove {filtered_count} incomplete entries, improving dataset quality.")

if __name__ == "__main__":
    main()
