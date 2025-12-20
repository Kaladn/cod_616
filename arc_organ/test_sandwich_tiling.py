#!/usr/bin/env python3
"""Test sandwich tiling detector on 00576224."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks, ArcGridPair, ARCTask
from cod_616.arc_organ.arc_two_attempt_sampler import ARCTwoAttemptSampler

# Load evaluation set
eval_challenges = json.loads(Path('arc-prize-2024/arc-agi_evaluation_challenges.json').read_text())
eval_solutions = json.loads(Path('arc-prize-2024/arc-agi_evaluation_solutions.json').read_text())

# Convert to ARCTask format
task_data = eval_challenges['00576224']
train_examples = [
    ArcGridPair(np.array(ex['input']), np.array(ex['output']))
    for ex in task_data['train']
]
test_input = np.array(task_data['test'][0]['input'])
expected_output = eval_solutions['00576224'][0]

print("="*70)
print("TESTING SANDWICH TILING ON 00576224")
print("="*70)

# Generate attempts
sampler = ARCTwoAttemptSampler()
attempt_1, attempt_2 = sampler.generate_attempts(train_examples, test_input)

print(f"\nTest input (2×2):")
print(test_input)

print(f"\nAttempt 1 (6×6):")
print(attempt_1)

print(f"\nAttempt 2 (6×6):")
print(attempt_2)

print(f"\nExpected output (6×6):")
print(np.array(expected_output))

# Check matches
match_1 = np.array_equal(attempt_1, expected_output)
match_2 = np.array_equal(attempt_2, expected_output)

print(f"\n{'='*70}")
print(f"Attempt 1 matches: {match_1}")
print(f"Attempt 2 matches: {match_2}")

if match_1 or match_2:
    print(f"\n🏆 00576224 SOLVED!")
    print(f"{'='*70}")
else:
    print(f"\n✗ No match")
    print(f"{'='*70}")
