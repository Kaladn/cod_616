"""
Analyze why 310f3251 isn't being detected by sandwich tiling
"""
import json
import numpy as np

# Load evaluation challenges and solutions
with open('arc-prize-2024/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_challenges = json.load(f)

with open('arc-prize-2024/arc-agi_evaluation_solutions.json', 'r') as f:
    eval_solutions = json.load(f)

task_id = '310f3251'
task = eval_challenges[task_id]
solution = eval_solutions[task_id]

print(f"\n{'='*60}")
print(f"Task: {task_id} (2×2 → 6×6 but not detected)")
print(f"{'='*60}")

print(f"\n📚 TRAINING EXAMPLES ({len(task['train'])}):\n")

for i, example in enumerate(task['train'], 1):
    input_grid = np.array(example['input'])
    output_grid = np.array(example['output'])
    
    print(f"Example {i}:")
    print(f"  Input ({input_grid.shape}):")
    print(input_grid)
    print(f"\n  Output ({output_grid.shape}):")
    print(output_grid)
    
    # Test sandwich tiling pattern
    if input_grid.shape == (2, 2) and output_grid.shape == (6, 6):
        print(f"\n  Testing sandwich tiling pattern:")
        
        # Build expected sandwich pattern
        rows_1_2 = np.tile(input_grid, (1, 3))
        rows_3_4 = np.tile(np.flip(input_grid, axis=1), (1, 3))  # flip columns
        rows_5_6 = rows_1_2
        expected = np.vstack([rows_1_2, rows_3_4, rows_5_6])
        
        print(f"\n  Expected sandwich (attempt 1):")
        print(expected)
        
        match = np.array_equal(output_grid, expected)
        print(f"\n  Match: {match}")
        
        if not match:
            # Try vertical flip variant
            rows_1_2 = np.tile(input_grid, (1, 3))
            rows_3_4 = np.tile(np.flip(input_grid, axis=0), (1, 3))  # flip rows
            rows_5_6 = rows_1_2
            expected_v2 = np.vstack([rows_1_2, rows_3_4, rows_5_6])
            
            print(f"\n  Expected sandwich (attempt 2 - row flip):")
            print(expected_v2)
            
            match_v2 = np.array_equal(output_grid, expected_v2)
            print(f"\n  Match v2: {match_v2}")
            
            if not match_v2:
                print(f"\n  ❌ Neither sandwich pattern matches!")
                print(f"\n  Let me analyze the actual pattern...")
                
                # Analyze row by row
                print(f"\n  Row-by-row comparison:")
                for row_idx in range(6):
                    print(f"    Row {row_idx}: {output_grid[row_idx]}")
                
                # Check if it's simple tiling
                simple_tile = np.tile(input_grid, (3, 3))
                if np.array_equal(output_grid, simple_tile):
                    print(f"\n  ✅ This is SIMPLE TILING (3× both directions)!")
    else:
        print(f"  ⚠️ Not 2×2 → 6×6!")
    
    print("\n" + "-"*60 + "\n")
