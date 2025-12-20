#!/usr/bin/env python3
"""Test rolling tiling detector on 007bbfb7."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks
from cod_616.arc_organ.arc_example_fuser import ARCExampleFuser
from cod_616.arc_organ.arc_recognition_field_v2 import ARCRecognitionFieldV2
from cod_616.arc_organ.arc_rule_applicator import RuleApplicator

# Load task
challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')
tasks = load_arc_tasks(challenges, solutions)

task = tasks["007bbfb7"]

print("=== Testing Rolling Tiling on 007bbfb7 ===\n")

# Fuse training examples
fused = ARCExampleFuser.fuse_examples(task.train)

print("Task Signature:")
print(f"  mean_tiling_strength: {fused.mean_tiling_strength:.3f}")
print(f"  Training examples: {len(task.train)}")
print()

# Recognize rules
recognizer = ARCRecognitionFieldV2()
hypotheses = recognizer.recognize_task(task.train, fused)

print(f"Detected hypotheses: {len(hypotheses)}")
for i, hyp in enumerate(hypotheses[:3], 1):
    print(f"\n{i}. {hyp.family.name}")
    print(f"   Confidence: {hyp.confidence:.3f}")
    print(f"   Params: {hyp.params}")
    print(f"   Reasoning: {hyp.reasoning}")

print("\n" + "="*60)
print("Testing on first training example:")
print("="*60 + "\n")

ex = task.train[0]
print(f"Input shape: {ex.input_grid.shape}")
print(f"Expected output shape: {ex.output_grid.shape}")
print()

# Apply top hypothesis
if hypotheses:
    top_hypothesis = hypotheses[0]
    print(f"Applying: {top_hypothesis.family.name}")
    
    applicator = RuleApplicator()
    attempt_1, attempt_2 = applicator.apply(top_hypothesis, ex.input_grid)
    
    print(f"\nAttempt 1 shape: {attempt_1.shape}")
    print(f"Attempt 2 shape: {attempt_2.shape}")
    
    if ex.output_grid is not None:
        # Check match
        match_1 = np.array_equal(attempt_1, ex.output_grid)
        match_2 = np.array_equal(attempt_2, ex.output_grid)
        
        print(f"\n✓ Attempt 1 matches expected: {match_1}")
        print(f"✓ Attempt 2 matches expected: {match_2}")
        
        if not match_1 and not match_2:
            diff_1 = np.sum(attempt_1 != ex.output_grid)
            diff_2 = np.sum(attempt_2 != ex.output_grid)
            print(f"\nDifferences: {diff_1} pixels (attempt 1), {diff_2} pixels (attempt 2)")
            
            # Show a small sample
            print("\nExpected output (top-left 5×5):")
            print(ex.output_grid[:5, :5])
            print("\nAttempt 1 (top-left 5×5):")
            print(attempt_1[:5, :5])
        else:
            print("\n🎯 SUCCESS! Rolling tiling works perfectly!")

print("\n" + "="*60)
print("Testing on test case:")
print("="*60 + "\n")

test_input = task.test[0].input_grid
test_output = task.test[0].output_grid

print(f"Test input shape: {test_input.shape}")
print(f"Test output shape: {test_output.shape if test_output is not None else 'Unknown'}")

if hypotheses:
    attempt_1, attempt_2 = applicator.apply(hypotheses[0], test_input)
    
    print(f"\nGenerated shape: {attempt_1.shape}")
    
    if test_output is not None:
        match_1 = np.array_equal(attempt_1, test_output)
        match_2 = np.array_equal(attempt_2, test_output)
        
        print(f"✓ Attempt 1 matches: {match_1}")
        print(f"✓ Attempt 2 matches: {match_2}")
        
        if match_1 or match_2:
            print("\n🏆 TEST CASE SOLVED! 007bbfb7 COMPLETE!")
