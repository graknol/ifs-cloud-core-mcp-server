#!/usr/bin/env python3
import json

with open(
    "batch_summaries/correlated_summaries_with_prompts_20250826_101247.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

print("🔤 TOKEN COUNT ANALYSIS")
print("=" * 50)
print(f"📄 Total entries: {len(data)}")

# Overall statistics
total_summary_tokens = sum(entry["summary_token_count"] for entry in data)
total_prompt_tokens = sum(entry["prompt_token_count"] for entry in data)
avg_summary_tokens = total_summary_tokens / len(data)
avg_prompt_tokens = total_prompt_tokens / len(data)

print()
print("📊 OVERALL STATISTICS:")
print(f"   • Total summary tokens: {total_summary_tokens:,}")
print(f"   • Total prompt tokens: {total_prompt_tokens:,}")
print(f"   • Average summary tokens: {avg_summary_tokens:.0f}")
print(f"   • Average prompt tokens: {avg_prompt_tokens:.0f}")
print(f"   • Prompt/Summary ratio: {(avg_prompt_tokens/avg_summary_tokens):.1f}x")

# Module breakdown
modules = {}
for entry in data:
    module = entry["module"]
    if module not in modules:
        modules[module] = {"count": 0, "summary_tokens": 0, "prompt_tokens": 0}
    modules[module]["count"] += 1
    modules[module]["summary_tokens"] += entry["summary_token_count"]
    modules[module]["prompt_tokens"] += entry["prompt_token_count"]

print()
print("📈 MODULE BREAKDOWN:")
for module in sorted(modules.keys()):
    stats = modules[module]
    avg_summary = stats["summary_tokens"] / stats["count"]
    avg_prompt = stats["prompt_tokens"] / stats["count"]
    print(f"   {module.upper()}:")
    print(f'     - Procedures: {stats["count"]}')
    print(f"     - Avg summary tokens: {avg_summary:.0f}")
    print(f"     - Avg prompt tokens: {avg_prompt:.0f}")
    print(
        f'     - Total tokens: {(stats["summary_tokens"] + stats["prompt_tokens"]):,}'
    )

# Find extremes
max_summary = max(data, key=lambda x: x["summary_token_count"])
min_summary = min(data, key=lambda x: x["summary_token_count"])
max_prompt = max(data, key=lambda x: x["prompt_token_count"])
min_prompt = min(data, key=lambda x: x["prompt_token_count"])

print()
print("🔍 EXTREMES:")
print(
    f'   Largest summary: {max_summary["summary_token_count"]} tokens ({max_summary["module"]}.{max_summary["procedure_name"]})'
)
print(
    f'   Smallest summary: {min_summary["summary_token_count"]} tokens ({min_summary["module"]}.{min_summary["procedure_name"]})'
)
print(
    f'   Largest prompt: {max_prompt["prompt_token_count"]} tokens ({max_prompt["module"]}.{max_prompt["procedure_name"]})'
)
print(
    f'   Smallest prompt: {min_prompt["prompt_token_count"]} tokens ({min_prompt["module"]}.{min_prompt["procedure_name"]})'
)

# Token distribution
summary_ranges = {"<500": 0, "500-750": 0, "750-1000": 0, "1000+": 0}
prompt_ranges = {"<500": 0, "500-750": 0, "750-1000": 0, "1000+": 0}

for entry in data:
    s_tokens = entry["summary_token_count"]
    p_tokens = entry["prompt_token_count"]

    # Summary ranges
    if s_tokens < 500:
        summary_ranges["<500"] += 1
    elif s_tokens < 750:
        summary_ranges["500-750"] += 1
    elif s_tokens < 1000:
        summary_ranges["750-1000"] += 1
    else:
        summary_ranges["1000+"] += 1

    # Prompt ranges
    if p_tokens < 500:
        prompt_ranges["<500"] += 1
    elif p_tokens < 750:
        prompt_ranges["500-750"] += 1
    elif p_tokens < 1000:
        prompt_ranges["750-1000"] += 1
    else:
        prompt_ranges["1000+"] += 1

print()
print("📊 TOKEN DISTRIBUTION:")
print("   Summary tokens:")
for range_name, count in summary_ranges.items():
    percentage = (count / len(data)) * 100
    print(f"     - {range_name}: {count} ({percentage:.1f}%)")
print("   Prompt tokens:")
for range_name, count in prompt_ranges.items():
    percentage = (count / len(data)) * 100
    print(f"     - {range_name}: {count} ({percentage:.1f}%)")
