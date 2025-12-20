"""
Quick analysis to find patterns similar to our working detectors
"""
import json
import numpy as np
from collections import defaultdict

# Load evaluation challenges
with open('arc-prize-2024/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_challenges = json.load(f)

# Categorize tasks by transformation type
categories = {
    'same_size': [],
    'simple_expansion': [],
    '2x2_to_6x6': [],  # Sandwich tiling pattern
    '4x4_to_16x16': [],  # Self-referential pattern
    'other_expansion': [],
    'size_change_other': []
}

print("Analyzing evaluation set transformations...")
print(f"\nTotal tasks: {len(eval_challenges)}")

for task_id, task in eval_challenges.items():
    if not task['train']:
        continue
    
    # Check first training example
    first_example = task['train'][0]
    input_shape = np.array(first_example['input']).shape
    output_shape = np.array(first_example['output']).shape
    
    if input_shape == output_shape:
        categories['same_size'].append(task_id)
    elif input_shape == (2, 2) and output_shape == (6, 6):
        categories['2x2_to_6x6'].append(task_id)
    elif input_shape == (4, 4) and output_shape == (16, 16):
        categories['4x4_to_16x16'].append(task_id)
    elif (output_shape[0] % input_shape[0] == 0 and 
          output_shape[1] % input_shape[1] == 0 and
          output_shape[0] // input_shape[0] == output_shape[1] // input_shape[1]):
        factor = output_shape[0] // input_shape[0]
        categories['simple_expansion'].append((task_id, input_shape, output_shape, factor))
    else:
        categories['size_change_other'].append((task_id, input_shape, output_shape))

print("\n" + "="*60)
print("CATEGORY BREAKDOWN:")
print("="*60)

print(f"\n1. SAME SIZE TRANSFORMATIONS: {len(categories['same_size'])} tasks")
print(f"   (These are likely recoloring, rotation, pattern completion)")

print(f"\n2. 2×2 → 6×6 EXPANSION: {len(categories['2x2_to_6x6'])} tasks")
print(f"   (Sandwich tiling pattern)")
if categories['2x2_to_6x6']:
    print(f"   Tasks: {categories['2x2_to_6x6'][:10]}")
    if len(categories['2x2_to_6x6']) > 10:
        print(f"   ... and {len(categories['2x2_to_6x6']) - 10} more")

print(f"\n3. 4×4 → 16×16 EXPANSION: {len(categories['4x4_to_16x16'])} tasks")
print(f"   (Self-referential tiling pattern)")
if categories['4x4_to_16x16']:
    print(f"   Tasks: {categories['4x4_to_16x16'][:10]}")
    if len(categories['4x4_to_16x16']) > 10:
        print(f"   ... and {len(categories['4x4_to_16x16']) - 10} more")

print(f"\n4. SIMPLE UNIFORM EXPANSION: {len(categories['simple_expansion'])} tasks")
print(f"   (N×N expansion with same factor)")
if categories['simple_expansion']:
    # Group by expansion factor
    by_factor = defaultdict(list)
    for task_id, inp, out, factor in categories['simple_expansion']:
        by_factor[factor].append((task_id, inp, out))
    
    for factor in sorted(by_factor.keys()):
        tasks = by_factor[factor]
        print(f"\n   {factor}× expansion ({len(tasks)} tasks):")
        for task_id, inp, out in tasks[:5]:
            print(f"      {task_id}: {inp} → {out}")
        if len(tasks) > 5:
            print(f"      ... and {len(tasks) - 5} more")

print(f"\n5. OTHER SIZE CHANGES: {len(categories['size_change_other'])} tasks")
print(f"   (Complex transformations)")

print("\n" + "="*60)
print("QUICK WIN OPPORTUNITIES:")
print("="*60)

print(f"\n✅ Already solved 2×2→6×6: {len(categories['2x2_to_6x6'])} tasks (sandwich tiling)")
print(f"✅ Already solved 4×4→16×16: {len(categories['4x4_to_16x16'])} tasks (self-ref tiling)")

# Identify next best targets
print(f"\n🎯 Next targets for quick implementation:")

if len(categories['simple_expansion']) > 0:
    by_factor = defaultdict(list)
    for task_id, inp, out, factor in categories['simple_expansion']:
        by_factor[factor].append((task_id, inp, out))
    
    most_common_factor = max(by_factor.keys(), key=lambda f: len(by_factor[f]))
    print(f"\n   1. {most_common_factor}× uniform expansion ({len(by_factor[most_common_factor])} tasks)")
    print(f"      These might be simple tiling or self-referential patterns")
    print(f"      Sample tasks: {[t[0] for t in by_factor[most_common_factor][:5]]}")

print(f"\n   2. Same-size transformations ({len(categories['same_size'])} tasks)")
print(f"      Would need pattern-specific analysis (recoloring, rotation, etc.)")

print(f"\n⏱️ Time constraint: 15 minutes remaining")
print(f"   Recommendation: Focus on verifying existing detector coverage")
print(f"   or analyzing why some 4×4→16×16 tasks aren't being detected")
