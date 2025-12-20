#!/usr/bin/env python3
"""Debug Phase E rule detection on specific tasks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np

from cod_616.arc_organ.arc_task_runner import load_arc_tasks
from cod_616.arc_organ.arc_example_fuser import ARCExampleFuser
from cod_616.arc_organ.arc_recognition_field_v2 import ARCRecognitionFieldV2

print("=" * 70)
print("PHASE E DEBUG - Rule Detection Analysis")
print("=" * 70)

# Load tasks
challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')
tasks = load_arc_tasks(challenges, solutions)

# Test on 007bbfb7 (tiling task)
task_id = "007bbfb7"
task = tasks[task_id]

print(f"\n📋 Task {task_id}")
print(f"   Training examples: {len(task.train)}")
print(f"   Test cases: {len(task.test)}")

# Show training examples
print("\n🔍 Training Example Analysis:")
for i, ex in enumerate(task.train):
    print(f"\n  Example {i+1}:")
    print(f"    Input shape: {ex.input_grid.shape}")
    print(f"    Output shape: {ex.output_grid.shape if ex.output_grid is not None else 'None'}")
    if ex.output_grid is not None:
        h_in, w_in = ex.input_grid.shape
        h_out, w_out = ex.output_grid.shape
        if h_in > 0 and w_in > 0:
            factor_h = h_out / h_in
            factor_w = w_out / w_in
            print(f"    Size ratio: {factor_h:.1f}×{factor_w:.1f}")
        
        # Check if output is tiled input
        if h_out % h_in == 0 and w_out % w_in == 0:
            factor_h_int = h_out // h_in
            factor_w_int = w_out // w_in
            tiled = np.tile(ex.input_grid, (factor_h_int, factor_w_int))
            is_tiled = np.array_equal(tiled, ex.output_grid)
            print(f"    Perfect tiling: {is_tiled} (factor {factor_h_int}×{factor_w_int})")

# Fuse examples
print("\n📊 Fusing training examples...")
fused = ARCExampleFuser.fuse_examples(task.train)

print(f"\n   Task Signature:")
print(f"     mean_tiling_strength (D11): {fused.mean_tiling_strength:.3f}")
print(f"     mean_vertical_symmetry (D8): {fused.mean_vertical_symmetry_strength:.3f}")
print(f"     mean_horizontal_symmetry (D9): {fused.mean_horizontal_symmetry_strength:.3f}")
print(f"     mean_change_localization (D16): {fused.mean_change_localization:.3f}")

# Run recognition
print("\n🧠 Running Recognition Field V2...")
recognizer = ARCRecognitionFieldV2()
hypotheses = recognizer.recognize_task(task.train, fused)

print(f"\n   Detected {len(hypotheses)} hypotheses:")
for i, hyp in enumerate(hypotheses, 1):
    print(f"\n   {i}. {hyp.family.name}")
    print(f"      Confidence: {hyp.confidence:.3f}")
    print(f"      Params: {hyp.params}")
    print(f"      Reasoning: {hyp.reasoning}")

# Test application on first test case
if len(hypotheses) > 0 and len(task.test) > 0:
    print(f"\n🎯 Testing top hypothesis on test case...")
    from cod_616.arc_organ.arc_rule_applicator import RuleApplicator
    
    applicator = RuleApplicator()
    test_input = task.test[0].input_grid
    test_output = task.test[0].output_grid
    
    print(f"   Test input shape: {test_input.shape}")
    print(f"   Expected output shape: {test_output.shape if test_output is not None else 'Unknown'}")
    
    attempt_1, attempt_2 = applicator.apply(hypotheses[0], test_input)
    
    print(f"   Generated output shapes: {attempt_1.shape}, {attempt_2.shape}")
    
    if test_output is not None:
        correct_1 = np.array_equal(attempt_1, test_output)
        correct_2 = np.array_equal(attempt_2, test_output)
        print(f"   Attempt 1 correct: {correct_1}")
        print(f"   Attempt 2 correct: {correct_2}")
        
        if not correct_1:
            print(f"\n   Attempt 1 sample (top-left 3×3):")
            print(attempt_1[:3, :3])
            print(f"   Expected sample (top-left 3×3):")
            print(test_output[:3, :3])

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
