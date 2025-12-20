#!/usr/bin/env python3
"""Simple debug test for ARC pipeline."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Step 1: Import modules...")
try:
    from cod_616.arc_organ.arc_task_runner import load_arc_tasks
    print("✓ Imported arc_task_runner")
except Exception as e:
    print(f"✗ Failed to import arc_task_runner: {e}")
    exit(1)

try:
    from cod_616.arc_organ.arc_two_attempt_sampler import ARCTwoAttemptSampler
    print("✓ Imported arc_two_attempt_sampler")
except Exception as e:
    print(f"✗ Failed to import arc_two_attempt_sampler: {e}")
    exit(1)

print("\nStep 2: Load tasks...")
try:
    challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
    solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')
    
    # Check files exist
    if not challenges.exists():
        print(f"✗ Challenges file not found: {challenges}")
        exit(1)
    if not solutions.exists():
        print(f"✗ Solutions file not found: {solutions}")
        exit(1)
    
    print(f"  Challenges: {challenges}")
    print(f"  Solutions: {solutions}")
    
    tasks = load_arc_tasks(challenges, solutions)
    print(f"✓ Loaded {len(tasks)} tasks")
except Exception as e:
    print(f"✗ Failed to load tasks: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nStep 3: Get first task...")
try:
    first_id = list(tasks.keys())[0]
    first_task = tasks[first_id]
    print(f"✓ Task ID: {first_id}")
    print(f"  Train examples: {len(first_task.train)}")
    print(f"  Test cases: {len(first_task.test)}")
except Exception as e:
    print(f"✗ Failed to get first task: {e}")
    exit(1)

print("\nStep 4: Initialize sampler...")
try:
    sampler = ARCTwoAttemptSampler()
    print("✓ Sampler initialized")
except Exception as e:
    print(f"✗ Failed to initialize sampler: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nStep 5: Generate attempts for first test case...")
try:
    test_input = first_task.test[0].input_grid
    print(f"  Input shape: {test_input.shape}")
    print(f"  Train examples: {len(first_task.train)}")
    
    print("  Calling generate_attempts()...")
    attempt_1, attempt_2 = sampler.generate_attempts(first_task.train, test_input)
    
    print(f"✓ Generated attempts")
    print(f"  Attempt 1 shape: {attempt_1.shape}")
    print(f"  Attempt 2 shape: {attempt_2.shape}")
except Exception as e:
    print(f"✗ Failed to generate attempts: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ ALL TESTS PASSED")
