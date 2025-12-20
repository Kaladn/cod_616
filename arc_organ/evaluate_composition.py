"""
Evaluation Harness for Composition Engine

Runs the new composition solver on the full ARC dataset
and compares against baseline single-operator performance.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from solve_task_composition import CompositionSolver


def load_arc_tasks(dataset_path: str) -> Dict:
    """
    Load ARC dataset from JSON file.
    
    Args:
        dataset_path: Path to challenges.json or evaluation.json
    
    Returns:
        Dict mapping task_id to task dict
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Convert to standard format if needed
    tasks = {}
    for task_id, task_data in data.items():
        tasks[task_id] = {
            "train": task_data.get("train", []),
            "test": task_data.get("test", [])
        }
    
    return tasks


def evaluate_on_dataset(dataset_path: str,
                        rule_db_path: str = None,
                        use_prior: bool = True,
                        max_tasks: int = None) -> Dict:
    """
    Evaluate composition solver on full dataset.
    
    Args:
        dataset_path: Path to ARC challenges JSON
        rule_db_path: Path to Math-DSL rules
        use_prior: Whether to use prior selector
        max_tasks: Max tasks to evaluate (None = all)
    
    Returns:
        Results dict with solve counts and solved task IDs
    """
    print("=" * 80)
    print("COMPOSITION ENGINE EVALUATION")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Math-DSL rules: {rule_db_path or 'None'}")
    print(f"Prior selector: {'enabled' if use_prior else 'disabled'}")
    print()
    
    # Load tasks
    tasks = load_arc_tasks(dataset_path)
    task_ids = list(tasks.keys())
    
    if max_tasks:
        task_ids = task_ids[:max_tasks]
    
    print(f"Evaluating {len(task_ids)} tasks...")
    print()
    
    # Initialize solver
    solver = CompositionSolver(rule_db_path=rule_db_path, use_prior=use_prior)
    
    # Track results
    results = {
        "total_tasks": len(task_ids),
        "solved_single": [],
        "solved_composition": [],
        "failed": [],
        "solve_times": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Evaluate each task
    start_time = time.time()
    
    for i, task_id in enumerate(task_ids):
        task = tasks[task_id]
        
        print(f"[{i+1}/{len(task_ids)}] {task_id}...", end=" ")
        
        task_start = time.time()
        solution = solver.solve(task)
        task_time = time.time() - task_start
        
        results["solve_times"][task_id] = task_time
        
        if solution:
            rule_type = solution["rule_type"]
            operators = solution["operators"]
            
            if rule_type == "composition":
                results["solved_composition"].append(task_id)
                print(f"✓ COMPOSITION: {' → '.join(operators)} ({task_time:.2f}s)")
            else:
                results["solved_single"].append(task_id)
                print(f"✓ SINGLE: {operators[0]} ({task_time:.2f}s)")
        else:
            results["failed"].append(task_id)
            print(f"✗ ({task_time:.2f}s)")
    
    total_time = time.time() - start_time
    
    # Summary
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Solved by composition: {len(results['solved_composition'])} ({len(results['solved_composition'])/len(task_ids)*100:.1f}%)")
    print(f"Solved by single op: {len(results['solved_single'])} ({len(results['solved_single'])/len(task_ids)*100:.1f}%)")
    print(f"Total solved: {len(results['solved_composition']) + len(results['solved_single'])} ({(len(results['solved_composition']) + len(results['solved_single']))/len(task_ids)*100:.1f}%)")
    print(f"Failed: {len(results['failed'])} ({len(results['failed'])/len(task_ids)*100:.1f}%)")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per task: {total_time/len(task_ids):.2f}s")
    print()
    
    # Show new solves from composition
    if results['solved_composition']:
        print("NEW SOLVES FROM COMPOSITION:")
        for task_id in results['solved_composition'][:10]:
            print(f"  - {task_id}")
        if len(results['solved_composition']) > 10:
            print(f"  ... and {len(results['solved_composition']) - 10} more")
    
    return results


def compare_with_baseline(baseline_path: str, composition_results: Dict):
    """
    Compare composition results with baseline single-operator performance.
    
    Args:
        baseline_path: Path to solutions_with_new_operators.json
        composition_results: Results from evaluate_on_dataset
    """
    print()
    print("=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    
    # Load baseline
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    baseline_solved = set(baseline.keys())
    
    composition_solved = set(composition_results['solved_composition'])
    single_solved = set(composition_results['solved_single'])
    all_solved = composition_solved | single_solved
    
    # Calculate gains
    new_solves = all_solved - baseline_solved
    lost_solves = baseline_solved - all_solved
    maintained = baseline_solved & all_solved
    
    print(f"Baseline solves: {len(baseline_solved)}")
    print(f"Composition engine solves: {len(all_solved)}")
    print(f"Gain: +{len(new_solves)} tasks")
    print(f"Lost: {len(lost_solves)} tasks (regression)")
    print(f"Maintained: {len(maintained)} tasks")
    print()
    
    if new_solves:
        print("NEW TASKS SOLVED:")
        for task_id in sorted(new_solves)[:15]:
            print(f"  ✓ {task_id}")
        if len(new_solves) > 15:
            print(f"  ... and {len(new_solves) - 15} more")
    
    if lost_solves:
        print()
        print("TASKS LOST (REGRESSION):")
        for task_id in sorted(lost_solves)[:10]:
            print(f"  ✗ {task_id}")
        if len(lost_solves) > 10:
            print(f"  ... and {len(lost_solves) - 10} more")
    
    print()


def main():
    """Main evaluation entry point."""
    import sys
    
    # Default paths (adjust for your setup)
    dataset_path = "../../arc-prize-2025/arc-agi_training_challenges.json"
    rule_db_path = "../../arc_transformation_rules_math_dsl.jsonl"
    baseline_path = "../../solutions_with_new_operators.json"
    
    # Check if paths exist, adjust if needed
    if not Path(dataset_path).exists():
        # Try alternative path
        dataset_path = "../../data/training/challenges.json"
    
    if not Path(rule_db_path).exists():
        print(f"Warning: Math-DSL rules not found at {rule_db_path}")
        print("Running without prior selector...")
        rule_db_path = None
    
    # Run evaluation
    results = evaluate_on_dataset(
        dataset_path=dataset_path,
        rule_db_path=rule_db_path,
        use_prior=True,
        max_tasks=None  # Evaluate all tasks
    )
    
    # Save results
    output_path = "composition_engine_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    
    # Compare with baseline if available
    if Path(baseline_path).exists():
        compare_with_baseline(baseline_path, results)


if __name__ == "__main__":
    main()
