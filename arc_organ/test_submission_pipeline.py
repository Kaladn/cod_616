#!/usr/bin/env python3
"""
Test ARC Submission Pipeline - Validate full system on sample tasks.
"""

from pathlib import Path
import sys
import os

# Set up paths
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from cod_616.arc_organ.arc_submission_builder import ARCSubmissionBuilder
from cod_616.arc_organ.arc_task_runner import load_arc_tasks

def test_sample_submission():
    """Test submission builder on first 3 training tasks."""
    
    print("=" * 60)
    print("ARC SUBMISSION PIPELINE TEST")
    print("=" * 60)
    
    # Load training tasks
    challenges_path = Path('arc-prize-2024/arc-agi_training_challenges.json')
    solutions_path = Path('arc-prize-2024/arc-agi_training_solutions.json')
    
    print(f"\n📂 Loading tasks from:")
    print(f"   Challenges: {challenges_path}")
    print(f"   Solutions: {solutions_path}")
    
    tasks = load_arc_tasks(challenges_path, solutions_path)
    print(f"\n✓ Loaded {len(tasks)} total tasks")
    
    # Test on first 3 tasks
    sample_ids = list(tasks.keys())[:3]
    sample_tasks = {task_id: tasks[task_id] for task_id in sample_ids}
    
    print(f"\n🎯 Testing on sample: {sample_ids}")
    
    # Build submission
    output_path = Path('submission/arc_test_sample.json')
    output_path.parent.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("BUILDING SUBMISSION")
    print("=" * 60)
    
    builder = ARCSubmissionBuilder()
    builder.build_submission(sample_tasks, output_path, verbose=True)
    
    print("\n" + "=" * 60)
    print("✓ TEST COMPLETE")
    print("=" * 60)
    print(f"\nSubmission saved to: {output_path}")
    
    # Validate format
    print("\n📋 Validating submission format...")
    is_valid = ARCSubmissionBuilder.validate_submission(output_path)
    
    if is_valid:
        print("✓ Format validation PASSED")
        return True
    else:
        print("✗ Format validation FAILED")
        return False


if __name__ == "__main__":
    success = test_sample_submission()
    exit(0 if success else 1)
