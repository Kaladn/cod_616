"""
✅ GOLD STANDARD SOLVER PIPELINE

Architecture:
1. ⬤ BASELINE FIRST — Try proven single-operator solver (56 solves)
2. ⬤ COMPOSITION SECOND — Try 2-step chains (adds new solves)
3. ⬤ RETURN BEST — Never lose previous solves

This is the correct AGI research solver layering.
Sequential. Stable. Additive.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any

# Import baseline solver (NEVER MODIFY THIS)
try:
    from arc_operators import OPERATORS, infer_rule_for_task, solve_task as baseline_solve_task
except ImportError:
    from .arc_operators import OPERATORS, infer_rule_for_task, solve_task as baseline_solve_task

# Import composition engine (AUGMENTATION ONLY)
try:
    from composition_engine import CompositionSearch
    from prior_selector import PriorSelector
except ImportError:
    from .composition_engine import CompositionSearch
    from .prior_selector import PriorSelector


def solve_task_with_baseline(task: Dict) -> Optional[Dict]:
    """
    ⬤ BASELINE SOLVER — Uses proven arc_operators.solve_task
    
    This is your 56/1000 solves. NEVER breaks.
    
    Args:
        task: ARC task dict with 'train' and 'test' keys
    
    Returns:
        Solution dict with 'operator', 'config', 'test_outputs' or None
    """
    # Convert to format baseline expects
    train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) 
                   for ex in task["train"]]
    test_inputs = [np.array(ex["input"]) for ex in task["test"]]
    
    # Call proven baseline solver
    test_outputs = baseline_solve_task(train_pairs, test_inputs)
    
    # Check if it worked
    if all(out is None for out in test_outputs):
        return None
    
    # Get the operator that worked
    op, config = infer_rule_for_task(train_pairs)
    if op is None:
        return None
    
    return {
        "rule_type": "single",
        "operator": op.name,
        "config": config,
        "test_outputs": [out.tolist() if out is not None else None 
                        for out in test_outputs]
    }


def solve_task_with_composition(task: Dict, 
                                operators: List = None,
                                use_prior: bool = False,
                                prior_selector: PriorSelector = None) -> Optional[Dict]:
    """
    ⬤ COMPOSITION SOLVER — Tries 2-step operator chains
    
    Only called if baseline fails. Adds new solves without breaking old ones.
    
    Args:
        task: ARC task dict
        operators: List of operators (uses OPERATORS if None)
        use_prior: Whether to use prior ranking
        prior_selector: PriorSelector instance for ranking
    
    Returns:
        Solution dict with composition chain or None
    """
    operators = operators if operators is not None else OPERATORS
    search = CompositionSearch(operators)
    
    # Get prior ranking if requested
    prior_ranked = None
    if use_prior and prior_selector:
        try:
            prior_ranked = prior_selector.rank_compositions(task, operators)[:50]
        except Exception:
            pass
    
    # Search for valid 2-step chain
    result = search.find_best_2_step(task, prior_ranked=prior_ranked)
    
    if result is None:
        return None
    
    op1, op2, cfg1, cfg2 = result
    
    # Apply chain to test inputs
    test_outputs = []
    for test_ex in task["test"]:
        inp = np.array(test_ex["input"])
        
        try:
            # Apply op1 → op2
            intermediate = op1.apply(inp, cfg1)
            if intermediate is None:
                test_outputs.append(None)
                continue
            
            final = op2.apply(intermediate, cfg2)
            test_outputs.append(final.tolist() if final is not None else None)
        except Exception:
            test_outputs.append(None)
    
    # Only return if at least one test succeeded
    if all(out is None for out in test_outputs):
        return None
    
    return {
        "rule_type": "composition",
        "operators": [op1.name, op2.name],
        "configs": {
            op1.name: str(cfg1),
            op2.name: str(cfg2)
        },
        "test_outputs": test_outputs
    }


def solve_task(task: Dict,
               use_composition: bool = True,
               use_prior: bool = False,
               prior_selector: PriorSelector = None) -> Optional[Dict]:
    """
    🎯 MASTER SOLVER — The complete pipeline
    
    Execution order (NEVER CHANGE THIS):
    1. Try baseline single-operator solver (56 solves guaranteed)
    2. If that fails, try 2-step composition (adds new solves)
    3. Return best result
    
    This architecture ensures:
    - Never lose previous solves
    - Composition only adds value
    - Clean separation of concerns
    
    Args:
        task: ARC task dict with 'train' and 'test' keys
        use_composition: Whether to try composition if baseline fails
        use_prior: Whether to use prior ranking in composition search
        prior_selector: PriorSelector instance (optional)
    
    Returns:
        Solution dict or None if no solution found
    """
    
    # ⬤ STEP 1: BASELINE FIRST (your 56 solves — SACRED)
    baseline_result = solve_task_with_baseline(task)
    if baseline_result is not None:
        return baseline_result
    
    # ⬤ STEP 2: COMPOSITION SECOND (only if baseline failed)
    if use_composition:
        composition_result = solve_task_with_composition(
            task, 
            use_prior=use_prior,
            prior_selector=prior_selector
        )
        if composition_result is not None:
            return composition_result
    
    # ⬤ STEP 3: Nothing worked
    return None


class CompositionSolver:
    """
    Convenience wrapper for solve_task with persistent state.
    
    Use this if you want to reuse prior_selector across multiple tasks.
    """
    
    def __init__(self, 
                 operators: List = None,
                 rule_db_path: str = None,
                 use_prior: bool = True,
                 use_composition: bool = True):
        """
        Args:
            operators: List of operators (uses OPERATORS if None)
            rule_db_path: Path to Math-DSL rule database
            use_prior: Whether to use prior ranking
            use_composition: Whether to try composition
        """
        self.operators = operators if operators is not None else OPERATORS
        self.use_composition = use_composition
        self.use_prior = use_prior
        
        # Initialize prior selector if requested
        self.prior_selector = None
        if use_prior and rule_db_path:
            try:
                self.prior_selector = PriorSelector(rule_db_path)
                print(f"✓ Prior selector loaded: {len(self.prior_selector.rules)} rules")
            except Exception as e:
                print(f"⚠ Prior selector failed to load: {e}")
        
        print(f"CompositionSolver initialized:")
        print(f"  Operators: {len(self.operators)}")
        print(f"  Composition: {'enabled' if self.use_composition else 'disabled'}")
        print(f"  Prior ranking: {'enabled' if self.prior_selector else 'disabled'}")
    
    def solve(self, task: Dict) -> Optional[Dict]:
        """
        Solve a task using the master pipeline.
        
        Args:
            task: ARC task dict
        
        Returns:
            Solution dict or None
        """
        return solve_task(
            task,
            use_composition=self.use_composition,
            use_prior=self.use_prior,
            prior_selector=self.prior_selector
        )
