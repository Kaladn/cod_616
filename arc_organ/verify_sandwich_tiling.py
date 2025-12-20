#!/usr/bin/env python3
"""Verify the sandwich tiling pattern."""

import json
import numpy as np
from pathlib import Path

eval_challenges = json.loads(Path('arc-prize-2024/arc-agi_evaluation_challenges.json').read_text())
task = eval_challenges['00576224']

print("SANDWICH TILING VERIFICATION")
print("="*70)

for ex_num, ex in enumerate(task['train'], 1):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    print(f"\nExample {ex_num}:")
    print(f"Input:\n{inp}")
    
    # Build expected output
    # Rows 1-2: tile input 3x
    rows_1_2 = np.tile(inp, (1, 3))
    
    # Rows 3-4: flip columns, then tile 3x
    inp_flipped = np.flip(inp, axis=1)  # flip horizontally
    rows_3_4 = np.tile(inp_flipped, (1, 3))
    
    # Rows 5-6: repeat rows 1-2
    rows_5_6 = rows_1_2
    
    # Stack them
    expected = np.vstack([rows_1_2, rows_3_4, rows_5_6])
    
    print(f"\nExpected output:\n{expected}")
    print(f"\nActual output:\n{out}")
    print(f"\nMatch: {np.array_equal(expected, out)}")
    
    if np.array_equal(expected, out):
        print("✓ SANDWICH TILING CONFIRMED!")

# Apply to test case
print("\n" + "="*70)
print("TEST CASE PREDICTION")
print("="*70)

test_inp = np.array(task['test'][0]['input'])
print(f"\nTest input:\n{test_inp}")

# Apply sandwich tiling
rows_1_2 = np.tile(test_inp, (1, 3))
inp_flipped = np.flip(test_inp, axis=1)
rows_3_4 = np.tile(inp_flipped, (1, 3))
rows_5_6 = rows_1_2

prediction = np.vstack([rows_1_2, rows_3_4, rows_5_6])

print(f"\nPredicted output (6×6):")
print(prediction)

print("\n" + "="*70)
print("This is the answer for 00576224!")
print("="*70)
