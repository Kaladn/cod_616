#!/usr/bin/env python3
"""Test current system on evaluation set."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import ArcGridPair
from cod_616.arc_organ.arc_two_attempt_sampler import ARCTwoAttemptSampler

# Load evaluation challenges and solutions
project_root = Path(__file__).parent.parent.parent
eval_challenges = json.loads((project_root / 'arc-prize-2024/arc-agi_evaluation_challenges.json').read_text())
eval_solutions = json.loads((project_root / 'arc-prize-2024/arc-agi_evaluation_solutions.json').read_text())

print("="*70)
print(f"TESTING ON EVALUATION SET ({len(eval_challenges)} tasks)")
print("="*70)

sampler = ARCTwoAttemptSampler()
correct = 0
total = 0
solved_tasks = []

for task_id in sorted(eval_challenges.keys()):
    task_data = eval_challenges[task_id]
    expected_outputs = eval_solutions[task_id]
    
    # Convert to ArcGridPair format
    train_examples = [
        ArcGridPair(np.array(ex['input']), np.array(ex['output']))
        for ex in task_data['train']
    ]
    
    # Test each test case
    task_correct = 0
    task_total = 0
    
    for test_idx, test_case in enumerate(task_data['test']):
        test_input = np.array(test_case['input'])
        expected = expected_outputs[test_idx]
        
        # Generate attempts
        attempt_1, attempt_2 = sampler.generate_attempts(train_examples, test_input)
        
        # Check if either matches
        match_1 = np.array_equal(attempt_1, expected)
        match_2 = np.array_equal(attempt_2, expected)
        
        task_total += 1
        total += 1
        if match_1 or match_2:
            task_correct += 1
            correct += 1
    
    if task_correct == task_total:
        print(f"✓ {task_id} ({task_correct}/{task_total})")
        solved_tasks.append(task_id)
    else:
        print(f"✗ {task_id} ({task_correct}/{task_total})")

print(f"\n{'='*70}")
accuracy = (correct / total * 100) if total > 0 else 0
print(f"ACCURACY: {correct}/{total} = {accuracy:.1f}%")
print(f"{'='*70}")

if solved_tasks:
    print(f"\n🏆 SOLVED TASKS ({len(solved_tasks)}):")
    for task_id in solved_tasks:
        print(f"  - {task_id}")

print(f"\n📊 Coverage: {len(solved_tasks)}/{len(eval_challenges)} tasks fully solved")
