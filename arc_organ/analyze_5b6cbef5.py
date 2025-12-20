"""
Quick analysis of task 5b6cbef5 to understand what pattern matched
"""
import json
import numpy as np

# Load evaluation challenges and solutions
with open('arc-prize-2024/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_challenges = json.load(f)

with open('arc-prize-2024/arc-agi_evaluation_solutions.json', 'r') as f:
    eval_solutions = json.load(f)

task_id = '5b6cbef5'
task = eval_challenges[task_id]
solution = eval_solutions[task_id]

print(f"\n{'='*60}")
print(f"Task: {task_id}")
print(f"{'='*60}")

print(f"\n📚 TRAINING EXAMPLES ({len(task['train'])}):\n")

for i, example in enumerate(task['train'], 1):
    input_grid = np.array(example['input'])
    output_grid = np.array(example['output'])
    
    print(f"Example {i}:")
    print(f"  Input shape:  {input_grid.shape}")
    print(f"  Output shape: {output_grid.shape}")
    
    print(f"\n  Input:")
    print(input_grid)
    
    print(f"\n  Output:")
    print(output_grid)
    
    # Check if it's a tiling pattern
    if output_grid.shape[0] % input_grid.shape[0] == 0 and output_grid.shape[1] % input_grid.shape[1] == 0:
        vertical_factor = output_grid.shape[0] // input_grid.shape[0]
        horizontal_factor = output_grid.shape[1] // input_grid.shape[1]
        print(f"\n  📐 Size expansion: {vertical_factor}× vertical, {horizontal_factor}× horizontal")
        
        # Check if it's simple tiling
        expected_tile = np.tile(input_grid, (vertical_factor, horizontal_factor))
        if np.array_equal(output_grid, expected_tile):
            print(f"  ✓ Simple tiling pattern!")
        else:
            print(f"  ✗ Not simple tiling (may be self-referential or other pattern)")
    else:
        print(f"  ⚠️ Not a simple size expansion")
    
    print()

print(f"\n🎯 TEST CASE:\n")
test_input = np.array(task['test'][0]['input'])
expected_output = np.array(solution[0])

print(f"Test input shape:  {test_input.shape}")
print(f"Expected output shape: {expected_output.shape}")

print(f"\nTest input:")
print(test_input)

print(f"\nExpected output:")
print(expected_output)

# Check pattern
if expected_output.shape[0] % test_input.shape[0] == 0 and expected_output.shape[1] % test_input.shape[1] == 0:
    vertical_factor = expected_output.shape[0] // test_input.shape[0]
    horizontal_factor = expected_output.shape[1] // test_input.shape[1]
    print(f"\n📐 Size expansion: {vertical_factor}× vertical, {horizontal_factor}× horizontal")
    
    # Check if self-referential tiling
    expected_tile = np.tile(test_input, (vertical_factor, horizontal_factor))
    
    # Mask: zero out blocks where input has 0
    mask = np.tile(test_input != 0, (vertical_factor, horizontal_factor))
    expected_self_ref = expected_tile * mask
    
    if np.array_equal(expected_output, expected_self_ref):
        print(f"✓ SELF-REFERENTIAL TILING PATTERN!")
    elif np.array_equal(expected_output, expected_tile):
        print(f"✓ Simple tiling pattern!")
    else:
        print(f"✗ Different pattern")
