#!/usr/bin/env python3
"""Debug why rolling tiling detector not firing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from cod_616.arc_organ.arc_task_runner import load_arc_tasks
from cod_616.arc_organ.arc_example_fuser import ARCExampleFuser
from cod_616.arc_organ.arc_rule_hypothesis import RollingTilingDetector

# Load task
challenges = Path('arc-prize-2024/arc-agi_training_challenges.json')
solutions = Path('arc-prize-2024/arc-agi_training_solutions.json')
tasks = load_arc_tasks(challenges, solutions)

task = tasks["007bbfb7"]
fused = ARCExampleFuser.fuse_examples(task.train)

print("Direct detector test:\n")
print(f"mean_tiling_strength: {fused.mean_tiling_strength}")

detector = RollingTilingDetector()
result = detector.detect(task.train, fused)

if result is None:
    print("\n❌ Detector returned None")
    print("\nManual checks:")
    
    # Check threshold
    print(f"  Tiling strength check: {fused.mean_tiling_strength} >= 0.8 ? {fused.mean_tiling_strength >= 0.8}")
    
    # Check integer expansion
    print("\n  Size checks:")
    for i, ex in enumerate(task.train):
        in_h, in_w = ex.input_grid.shape
        out_h, out_w = ex.output_grid.shape
        print(f"    Ex {i+1}: {in_h}×{in_w} → {out_h}×{out_w}")
        print(f"      Integer multiples? h: {out_h % in_h == 0}, w: {out_w % in_w == 0}")
        print(f"      Factors: {out_h // in_h}×{out_w // in_w}")
    
    # Try to detect shift pattern manually
    print("\n  Testing shift detection on first example:")
    ex = task.train[0]
    tile = ex.input_grid
    output = ex.output_grid
    h, w = tile.shape
    factor_h = output.shape[0] // h
    factor_w = output.shape[1] // w
    
    print(f"    Tile: {h}×{w}, Factors: {factor_h}×{factor_w}")
    
    for i_block in range(factor_h):
        block_start = i_block * h
        block_end = (i_block + 1) * h
        
        print(f"\n    Row block {i_block} (rows {block_start}-{block_end-1}):")
        
        for shift in range(w):
            shifted_tile = np.roll(tile, shift, axis=1)
            
            matches = 0
            total = 0
            for j_block in range(factor_w):
                col_start = j_block * w
                col_end = (j_block + 1) * w
                block = output[block_start:block_end, col_start:col_end]
                matches += np.sum(block == shifted_tile)
                total += block.size
            
            match_rate = matches / total if total > 0 else 0
            if match_rate > 0.9:
                print(f"      Shift {shift}: {match_rate:.3f} match ✓")
else:
    print(f"✓ Detector successful!")
    print(f"  Family: {result.family.name}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Params: {result.params}")
