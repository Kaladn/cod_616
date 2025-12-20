#!/usr/bin/env python3
"""Evaluate ARC submission accuracy on sample tasks."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cod_616.arc_organ.arc_task_runner import load_arc_tasks

print("=" * 70)
print("ARC SUBMISSION ACCURACY EVALUATION")
print("=" * 70)

# Load submission
submission_path = Path('submission/arc_sample_5tasks.json')
print(f"\n📂 Loading submission: {submission_path}")
with open(submission_path) as f:
    submission = json.load(f)
print(f"✓ Loaded predictions for {len(submission)} tasks")

# Load solutions
print("\n📂 Loading solutions...")
challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')
tasks = load_arc_tasks(challenges, solutions)
print(f"✓ Loaded {len(tasks)} tasks")

# Evaluate
print(f"\n{'=' * 70}")
print("EVALUATION RESULTS")
print("=" * 70)

total_test_cases = 0
correct_attempts = 0
correct_test_cases = 0

for task_id in submission.keys():
    task = tasks[task_id]
    predictions = submission[task_id]
    
    print(f"\nTask {task_id}:")
    
    for test_idx, (pred, test_case) in enumerate(zip(predictions, task.test)):
        total_test_cases += 1
        
        # Get ground truth
        gt = test_case.output_grid
        
        # Get predictions (convert back to numpy)
        attempt_1 = np.array(pred[0])
        attempt_2 = np.array(pred[1])
        
        # Check if either attempt matches
        match_1 = np.array_equal(attempt_1, gt)
        match_2 = np.array_equal(attempt_2, gt)
        
        if match_1 or match_2:
            correct_test_cases += 1
            status = "✓ CORRECT"
            if match_1:
                correct_attempts += 1
                which = "(attempt 1)"
            if match_2:
                correct_attempts += 1
                which = "(attempt 2)" if not match_1 else "(both)"
        else:
            status = "✗ WRONG"
            which = ""
        
        print(f"  Test {test_idx + 1}: {status} {which}")
        print(f"    GT shape: {gt.shape}, Pred shapes: {attempt_1.shape}, {attempt_2.shape}")
        
        # Show first few values if wrong
        if not (match_1 or match_2) and gt.size <= 100:
            print(f"    GT sample:\n{gt[:3, :3] if len(gt.shape) == 2 else gt}")
            print(f"    Attempt 1 sample:\n{attempt_1[:3, :3] if len(attempt_1.shape) == 2 else attempt_1}")

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"Total test cases: {total_test_cases}")
print(f"Correct test cases: {correct_test_cases}/{total_test_cases} ({100*correct_test_cases/total_test_cases:.1f}%)")
print(f"Total attempts: {total_test_cases * 2}")
print(f"Correct attempts: {correct_attempts}/{total_test_cases * 2} ({100*correct_attempts/(total_test_cases*2):.1f}%)")

if correct_test_cases > 0:
    print(f"\n✓ System is working! {correct_test_cases} correct predictions.")
else:
    print(f"\n⚠ No correct predictions yet. System needs tuning.")
    print("   This is expected with pure threshold-based heuristics.")
    print("   Next step: Analyze patterns and refine recognition field.")
