#!/usr/bin/env python3
"""Generate ARC submission on sample tasks."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cod_616.arc_organ.arc_task_runner import load_arc_tasks
from cod_616.arc_organ.arc_two_attempt_sampler import ARCTwoAttemptSampler

print("=" * 70)
print("ARC PRIZE SUBMISSION GENERATOR - SAMPLE TEST")
print("=" * 70)

# Load tasks
print("\n📂 Loading tasks...")
challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')

tasks = load_arc_tasks(challenges, solutions)
print(f"✓ Loaded {len(tasks)} total tasks\n")

# Test on first 5 tasks
sample_size = 5
sample_ids = list(tasks.keys())[:sample_size]
print(f"🎯 Generating submission for {sample_size} sample tasks:")
for task_id in sample_ids:
    task = tasks[task_id]
    print(f"   {task_id}: {len(task.train)} train, {len(task.test)} test")

# Initialize sampler
sampler = ARCTwoAttemptSampler()

# Build submission
print(f"\n{'=' * 70}")
print("PROCESSING TASKS")
print("=" * 70)

submission = {}

for i, task_id in enumerate(sample_ids, 1):
    task = tasks[task_id]
    print(f"\n[{i}/{sample_size}] Task {task_id}:")
    
    task_predictions = []
    
    for test_idx, test_case in enumerate(task.test):
        print(f"  Test case {test_idx + 1}/{len(task.test)}...", end=" ")
        
        # Generate attempts
        attempt_1, attempt_2 = sampler.generate_attempts(
            train_examples=task.train,
            test_input=test_case.input_grid,
        )
        
        # Convert to lists
        task_predictions.append([attempt_1.tolist(), attempt_2.tolist()])
        print(f"✓ ({attempt_1.shape} → {attempt_1.shape}, {attempt_2.shape})")
    
    submission[task_id] = task_predictions

# Save submission
output_path = Path('submission/arc_sample_5tasks.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(submission, f, indent=2)

print(f"\n{'=' * 70}")
print("✓ SUBMISSION COMPLETE")
print("=" * 70)
print(f"\nSubmission saved to: {output_path}")
print(f"Tasks: {len(submission)}")
print(f"Total test cases: {sum(len(preds) for preds in submission.values())}")
print(f"Total attempts: {sum(len(preds) * 2 for preds in submission.values())}")

# Validate format
print("\n📋 Validating submission format...")
try:
    with open(output_path) as f:
        data = json.load(f)
    
    assert isinstance(data, dict), "Root must be dict"
    
    for task_id, predictions in data.items():
        assert isinstance(predictions, list), f"{task_id}: predictions must be list"
        for pred in predictions:
            assert isinstance(pred, list) and len(pred) == 2, f"{task_id}: must have 2 attempts"
            for attempt in pred:
                assert isinstance(attempt, list), f"{task_id}: attempt must be list"
                assert all(isinstance(row, list) for row in attempt), f"{task_id}: all rows must be lists"
    
    print("✓ Format validation PASSED")
    print("\n✓ READY FOR SUBMISSION")
except Exception as e:
    print(f"✗ Format validation FAILED: {e}")
    exit(1)
