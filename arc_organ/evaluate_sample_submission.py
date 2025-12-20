#!/usr/bin/env python3
"""Evaluate accuracy of sample 5-task submission."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cod_616.arc_organ.arc_task_runner import load_arc_tasks

# Load solutions
solutions_path = Path('arc-prize-2024/arc-agi_training_solutions.json')
solutions = json.loads(solutions_path.read_text())

# Load submission
submission_path = Path('submission/arc_sample_5tasks.json')
submission = json.loads(submission_path.read_text())

sample_tasks = ['007bbfb7', '00d62c1b', '017c7c7b', '025d127b', '045e512c']

print('='*60)
print('EVALUATING SAMPLE SUBMISSION')
print('='*60)
print()

total_correct = 0
total_attempts = 0

for task_id in sample_tasks:
    if task_id not in submission:
        print(f'{task_id}: MISSING')
        continue
    
    expected = solutions[task_id]
    attempts = submission[task_id]
    
    # For each test case
    for test_idx, expected_output in enumerate(expected):
        test_attempts = attempts[test_idx]
        
        # Check if either attempt matches
        attempt_1_correct = (test_attempts[0] == expected_output)
        attempt_2_correct = (test_attempts[1] == expected_output)
        
        correct = attempt_1_correct or attempt_2_correct
        total_attempts += 1
        if correct:
            total_correct += 1
            status = '✓ CORRECT'
            which = '(attempt 1)' if attempt_1_correct else '(attempt 2)'
        else:
            status = '✗ INCORRECT'
            which = ''
        
        print(f'{task_id} test {test_idx+1}: {status} {which}')

print()
print('='*60)
accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
print(f'ACCURACY: {total_correct}/{total_attempts} = {accuracy:.1f}%')
print('='*60)

if accuracy > 0:
    print(f'\n🎯 First non-zero accuracy achieved!')
if accuracy >= 20:
    print(f'\n🏆 TARGET REACHED: {accuracy:.1f}% >= 20%')
