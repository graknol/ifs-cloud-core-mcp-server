#!/usr/bin/env python3
"""
Clean synthetic questions dataset by:
1. Filter out entries with null original_summary
2. Correlate with procedures_analysis.jsonl to get accurate procedure info
3. Use summaries as correlation keys (module + file + summary should be 1-to-1)
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_procedures_analysis(file_path: Path) -> dict:
    """Load procedures analysis and create lookup by summary."""
    print(f"📖 Loading procedures analysis from {file_path}")
    
    procedures_by_summary = {}
    procedures_by_name_file = {}
    total_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1
                
                # Create lookup by summary (primary key for correlation)
                summary_key = data.get('summary', '').strip()
                if summary_key:
                    procedures_by_summary[summary_key] = data
                
                # Create secondary lookup by procedure name + file
                name_file_key = f"{data.get('procedure_name', '')}|{data.get('file', '')}"
                procedures_by_name_file[name_file_key] = data
                
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error on line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {total_count} procedure entries")
    print(f"   - {len(procedures_by_summary)} unique summaries")
    print(f"   - {len(procedures_by_name_file)} unique name+file combinations")
    
    return procedures_by_summary, procedures_by_name_file


def clean_and_correlate_synthetic_questions(
    synthetic_file: Path,
    procedures_by_summary: dict,
    procedures_by_name_file: dict,
    output_file: Path
) -> None:
    """Clean synthetic questions and correlate with procedures analysis."""
    
    print(f"\n🧹 Cleaning synthetic questions from {synthetic_file}")
    
    stats = {
        'total_entries': 0,
        'null_summary_filtered': 0,
        'summary_matched': 0,
        'name_file_matched': 0,
        'no_match': 0,
        'clean_entries': 0
    }
    
    clean_entries = []
    
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                stats['total_entries'] += 1
                
                # Filter out entries with null original_summary
                original_summary = data.get('original_summary')
                if original_summary is None:
                    stats['null_summary_filtered'] += 1
                    continue
                
                # Try to correlate with procedures analysis
                procedure_data = None
                match_type = None
                
                # Primary correlation: by summary
                summary_key = original_summary.strip()
                if summary_key in procedures_by_summary:
                    procedure_data = procedures_by_summary[summary_key]
                    match_type = 'summary'
                    stats['summary_matched'] += 1
                else:
                    # Secondary correlation: by procedure name + file
                    name_file_key = f"{data.get('procedure_name', '')}|{data.get('file', '')}"
                    if name_file_key in procedures_by_name_file:
                        procedure_data = procedures_by_name_file[name_file_key]
                        match_type = 'name_file'
                        stats['name_file_matched'] += 1
                    else:
                        stats['no_match'] += 1
                        # Keep the original data but mark it as unmatched
                        procedure_data = data
                        match_type = 'no_match'
                
                # Create clean entry with enhanced data
                clean_entry = {
                    'procedure_name': procedure_data.get('procedure_name', data.get('procedure_name')),
                    'parameters': procedure_data.get('parameters', ''),
                    'signature': procedure_data.get('signature', ''),
                    'summary': procedure_data.get('summary', original_summary),
                    'synthetic_queries': data.get('synthetic_queries', []),
                    'synthetic_statements': data.get('synthetic_statements', []),
                    'module': procedure_data.get('module', data.get('module')),
                    'file': procedure_data.get('file', data.get('file')),
                    'file_path': procedure_data.get('file_path', data.get('source_file')),
                    'match_type': match_type,
                    'timestamp': data.get('timestamp', datetime.now().isoformat())
                }
                
                clean_entries.append(clean_entry)
                stats['clean_entries'] += 1
                
                if line_num % 1000 == 0:
                    print(f"   Processed {line_num:,} entries...")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Error processing line {line_num}: {e}")
                continue
    
    # Write clean entries to output file
    print(f"\n💾 Writing clean entries to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in clean_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\n📊 Cleaning Statistics:")
    print(f"{'='*50}")
    print(f"Total entries processed:     {stats['total_entries']:,}")
    print(f"Null summary filtered out:   {stats['null_summary_filtered']:,}")
    print(f"Summary-based matches:       {stats['summary_matched']:,}")
    print(f"Name+file-based matches:     {stats['name_file_matched']:,}")
    print(f"No correlation match:        {stats['no_match']:,}")
    print(f"Clean entries written:       {stats['clean_entries']:,}")
    print(f"")
    print(f"Cleaning efficiency:         {(stats['clean_entries']/stats['total_entries']*100):.1f}%")
    print(f"Correlation success rate:    {((stats['summary_matched']+stats['name_file_matched'])/stats['clean_entries']*100):.1f}%")


def main():
    """Main execution function."""
    base_dir = Path("plsql_analysis_combined")
    
    # Input files
    synthetic_file = base_dir / "synthetic_questions.jsonl"
    procedures_file = base_dir / "procedures_analysis.jsonl"
    
    # Output file
    clean_file = base_dir / "clean.jsonl"
    
    # Validate input files
    if not synthetic_file.exists():
        print(f"❌ Synthetic questions file not found: {synthetic_file}")
        sys.exit(1)
    
    if not procedures_file.exists():
        print(f"❌ Procedures analysis file not found: {procedures_file}")
        sys.exit(1)
    
    # Load procedures analysis for correlation
    procedures_by_summary, procedures_by_name_file = load_procedures_analysis(procedures_file)
    
    # Clean and correlate synthetic questions
    clean_and_correlate_synthetic_questions(
        synthetic_file,
        procedures_by_summary,
        procedures_by_name_file,
        clean_file
    )
    
    print(f"\n✅ Dataset cleaning completed!")
    print(f"   Clean dataset: {clean_file}")
    print(f"   Ready for embedding fine-tuning with accurate procedure correlation.")


if __name__ == "__main__":
    main()
                if clean_proc_name == clean_question_name:
                    matched_procedure = proc
                    break
        
        if matched_procedure:
            # Create enhanced question entry with procedure correlation
            clean_entry = {
                'procedure_name': matched_procedure['procedure_name'],
                'procedure_signature': matched_procedure.get('signature', ''),
                'procedure_parameters': matched_procedure.get('parameters', ''),
                'original_summary': matched_procedure['summary'],
                'synthetic_queries': question_entry['synthetic_queries'],
                'synthetic_statements': question_entry['synthetic_statements'],
                'module': matched_procedure['module'],
                'file': matched_procedure['file'],
                'file_path': matched_procedure['file_path'],
                'source_file': question_entry['source_file'],
                'timestamp': question_entry['timestamp']
            }
            clean_questions.append(clean_entry)
            stats['procedure_matched'] += 1
        else:
            # Keep entry but mark as unmatched
            stats['procedure_not_matched'] += 1
            # Still include it if it has valid content
            if question_entry.get('original_summary'):
                clean_questions.append(question_entry)
    
    stats['final_clean'] = len(clean_questions)
    
    # Save cleaned dataset
    output_file = Path("plsql_analysis_combined/clean.jsonl")
    save_jsonl(clean_questions, output_file)
    
    print(f"\n📊 Cleaning Statistics:")
    print(f"   • Total original entries: {stats['total_original']:,}")
    print(f"   • Null summary filtered: {stats['null_summary_filtered']:,}")
    print(f"   • Procedure matched: {stats['procedure_matched']:,}")
    print(f"   • Procedure not matched: {stats['procedure_not_matched']:,}")
    print(f"   • Final clean entries: {stats['final_clean']:,}")
    print(f"   • Improvement: {((stats['final_clean']/stats['total_original'])*100):.1f}% retained after cleaning")
    
    print(f"\n💾 Cleaned dataset saved to: {output_file}")
    
    # Show some sample entries
    print(f"\n🔍 Sample cleaned entries:")
    for i, entry in enumerate(clean_questions[:3]):
        print(f"\n   Sample {i+1}:")
        print(f"   • Procedure: {entry['procedure_name']}")
        print(f"   • Module: {entry['module']}")
        print(f"   • Summary: {entry['original_summary'][:100]}...")
        print(f"   • Query 1: {entry['synthetic_queries'][0]}")
        print(f"   • Statement 1: {entry['synthetic_statements'][0]}")
    
    return clean_questions

def analyze_cleaning_results():
    """Analyze the cleaning results"""
    print("\n📈 Detailed Analysis of Cleaned Dataset")
    print("=" * 50)
    
    clean_file = Path("plsql_analysis_combined/clean.jsonl")
    if not clean_file.exists():
        print("❌ Clean dataset not found. Run cleaning first.")
        return
    
    clean_data = load_jsonl(clean_file)
    
    # Analysis by module
    modules = defaultdict(int)
    for entry in clean_data:
        modules[entry.get('module', 'unknown')] += 1
    
    print(f"📊 Clean dataset contains {len(clean_data):,} entries across {len(modules)} modules")
    
    # Top modules by question count
    sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔝 Top 10 modules by question count:")
    for i, (module, count) in enumerate(sorted_modules[:10]):
        print(f"   {i+1:2d}. {module:12s}: {count:,} questions")
    
    # Quality check
    valid_summaries = sum(1 for entry in clean_data if entry.get('original_summary') and len(entry.get('original_summary', '')) > 20)
    valid_queries = sum(1 for entry in clean_data if entry.get('synthetic_queries') and len(entry.get('synthetic_queries', [])) >= 2)
    valid_statements = sum(1 for entry in clean_data if entry.get('synthetic_statements') and len(entry.get('synthetic_statements', [])) >= 2)
    
    print(f"\n✅ Quality metrics:")
    print(f"   • Valid summaries (>20 chars): {valid_summaries:,} ({(valid_summaries/len(clean_data)*100):.1f}%)")
    print(f"   • Valid queries (>=2): {valid_queries:,} ({(valid_queries/len(clean_data)*100):.1f}%)")
    print(f"   • Valid statements (>=2): {valid_statements:,} ({(valid_statements/len(clean_data)*100):.1f}%)")

if __name__ == "__main__":
    try:
        clean_questions = clean_synthetic_questions()
        analyze_cleaning_results()
        print(f"\n🎉 Dataset cleaning complete!")
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
