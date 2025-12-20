# cod_616/arc_organ/arc_submission_builder.py

"""
ARC Submission Builder - Generate submission.json in ARC Prize format.

Takes trained CompuCog system and generates predictions for all test tasks.
Output format matches ARC Prize requirements exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from .arc_task_runner import load_arc_tasks, ARCTask
from .arc_two_attempt_sampler import ARCTwoAttemptSampler


class ARCSubmissionBuilder:
    """
    Builds submission.json for ARC Prize evaluation.
    
    Format:
    {
        "task_id_1": [
            [[attempt_1_grid]],  # First attempt
            [[attempt_2_grid]]   # Second attempt
        ],
        "task_id_2": ...
    }
    """
    
    def __init__(self):
        self.sampler = ARCTwoAttemptSampler()
    
    def build_submission(
        self,
        tasks: Dict[str, ARCTask],
        output_path: Path,
        verbose: bool = True,
    ) -> None:
        """
        Generate submission.json for all tasks.
        
        Args:
            tasks: Dictionary of task_id -> ARCTask
            output_path: Where to save submission.json
            verbose: Print progress
        """
        submission = {}
        
        for task_id, task in tasks.items():
            if verbose:
                print(f"Processing {task_id}... ({len(task.test)} test cases)")
            
            task_predictions = []
            
            for test_case in task.test:
                # Generate two attempts
                attempt_1, attempt_2 = self.sampler.generate_attempts(
                    train_examples=task.train,
                    test_input=test_case.input_grid,
                )
                
                # Convert to list format
                attempt_1_list = attempt_1.tolist()
                attempt_2_list = attempt_2.tolist()
                
                # ARC format: each test case gets [attempt_1, attempt_2]
                task_predictions.append([attempt_1_list, attempt_2_list])
            
            submission[task_id] = task_predictions
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)
        
        if verbose:
            print(f"\n✓ Submission saved to: {output_path}")
            print(f"  Tasks: {len(submission)}")
            total_test_cases = sum(len(preds) for preds in submission.values())
            print(f"  Test cases: {total_test_cases}")
            print(f"  Total attempts: {total_test_cases * 2}")
    
    @staticmethod
    def validate_submission(submission_path: Path) -> bool:
        """
        Validate submission.json format.
        
        Checks:
        - Valid JSON
        - All task IDs present
        - Each test case has exactly 2 attempts
        - All grids are valid (2D lists of integers)
        """
        try:
            with open(submission_path, 'r') as f:
                submission = json.load(f)
            
            if not isinstance(submission, dict):
                print("❌ Submission must be a dictionary")
                return False
            
            for task_id, predictions in submission.items():
                if not isinstance(predictions, list):
                    print(f"❌ Task {task_id}: predictions must be a list")
                    return False
                
                for test_idx, attempts in enumerate(predictions):
                    if not isinstance(attempts, list) or len(attempts) != 2:
                        print(f"❌ Task {task_id} test {test_idx}: must have exactly 2 attempts")
                        return False
                    
                    for attempt_idx, grid in enumerate(attempts):
                        if not isinstance(grid, list):
                            print(f"❌ Task {task_id} test {test_idx} attempt {attempt_idx}: grid must be a list")
                            return False
                        
                        # Check all rows are lists
                        for row_idx, row in enumerate(grid):
                            if not isinstance(row, list):
                                print(f"❌ Task {task_id} test {test_idx} attempt {attempt_idx} row {row_idx}: must be a list")
                                return False
                            
                            # Check all values are integers
                            for col_idx, val in enumerate(row):
                                if not isinstance(val, int):
                                    print(f"❌ Task {task_id} test {test_idx} attempt {attempt_idx} [{row_idx},{col_idx}]: value must be integer")
                                    return False
            
            print("✓ Submission format valid")
            return True
        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return False
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False


def main():
    """Build submission for ARC Prize evaluation set."""
    print("=" * 60)
    print("ARC SUBMISSION BUILDER")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    arc_dir = project_root / "arc-prize-2024"
    
    # Choose dataset
    dataset = "evaluation"  # or "training" for testing
    
    if dataset == "training":
        challenges_path = arc_dir / "arc-agi_training_challenges.json"
        solutions_path = arc_dir / "arc-agi_training_solutions.json"
        output_path = project_root / "submission" / "arc_submission_training.json"
    else:
        challenges_path = arc_dir / "arc-agi_evaluation_challenges.json"
        solutions_path = arc_dir / "arc-agi_evaluation_solutions.json"
        output_path = project_root / "submission" / "arc_submission_evaluation.json"
    
    print(f"Dataset: {dataset}")
    print(f"Loading: {challenges_path.name}")
    
    # Load tasks
    tasks = load_arc_tasks(challenges_path, solutions_path if solutions_path.exists() else None)
    print(f"Tasks loaded: {len(tasks)}")
    
    # Build submission
    builder = ARCSubmissionBuilder()
    builder.build_submission(tasks, output_path, verbose=True)
    
    # Validate
    print(f"\nValidating submission...")
    if builder.validate_submission(output_path):
        print(f"\n{'=' * 60}")
        print(f"✓ SUBMISSION READY")
        print(f"{'=' * 60}")
        print(f"File: {output_path}")
        print(f"\nTo submit to ARC Prize 2025:")
        print(f"1. Review submission format")
        print(f"2. Test on training set first")
        print(f"3. Upload to competition platform")
        print(f"{'=' * 60}")
    else:
        print(f"\n❌ Submission validation failed")


if __name__ == "__main__":
    main()
