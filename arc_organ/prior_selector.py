"""
Prior Selector — The Brain

Uses the Math-DSL knowledge base to rank operators and operator pairs
based on semantic feature matching with historical successes.

This is the intelligence layer that prevents brute-force search by
predicting which operators are most likely to work.
"""

import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

try:
    from feature_extractor import FeatureExtractor
except ImportError:
    from .feature_extractor import FeatureExtractor


class PriorSelector:
    """
    Rank operators using Math-DSL knowledge base.
    
    Matches task features against historical rule database to predict
    which operators are most likely to solve the current task.
    """
    
    def __init__(self, rule_db_path: str = None):
        """
        Args:
            rule_db_path: Path to JSONL file containing Math-DSL rules
        """
        self.extractor = FeatureExtractor()
        self.rules = []
        
        if rule_db_path:
            self.load_rules(rule_db_path)
    
    def load_rules(self, rule_db_path: str):
        """
        Load Math-DSL rule database from JSONL file.
        
        Args:
            rule_db_path: Path to arc_transformation_rules_math_dsl.jsonl
        """
        rule_file = Path(rule_db_path)
        
        if not rule_file.exists():
            print(f"Warning: Rule DB not found at {rule_db_path}")
            return
        
        self.rules = []
        with open(rule_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rule = json.loads(line)
                        self.rules.append(rule)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.rules)} rules from Math-DSL database")
    
    def rank_single_ops(self, task: Dict, operators: List) -> List[Tuple[Any, float]]:
        """
        Rank operators by likelihood of solving the task.
        
        Args:
            task: ARC task dict
            operators: List of ArcOperator instances
        
        Returns:
            List of (operator, score) tuples, sorted by descending score
        """
        features = self.extractor.extract(task)
        
        scores = []
        for op in operators:
            score = self._score_operator(op, features)
            scores.append((op, score))
        
        # Sort by score (descending)
        return sorted(scores, key=lambda x: -x[1])
    
    def rank_compositions(self, task: Dict, operators: List) -> List[Tuple[Tuple, float]]:
        """
        Rank operator pairs by likelihood of solving the task.
        
        Args:
            task: ARC task dict
            operators: List of ArcOperator instances
        
        Returns:
            List of ((op1, op2), score) tuples, sorted by descending score
        """
        features = self.extractor.extract(task)
        
        scores = []
        for op1 in operators:
            for op2 in operators:
                score = self._score_composition(op1, op2, features)
                if score > 0:  # Only include plausible pairs
                    scores.append(((op1, op2), score))
        
        # Sort by score (descending)
        return sorted(scores, key=lambda x: -x[1])
    
    def _score_operator(self, op, features: Dict) -> float:
        """
        Score a single operator based on feature matching.
        
        Args:
            op: ArcOperator instance
            features: Feature dict from FeatureExtractor
        
        Returns:
            Score (higher = more likely)
        """
        if not self.rules:
            return 1.0  # No prior knowledge, all equal
        
        score = 0.0
        
        for rule in self.rules:
            # Check if this rule uses the operator
            rule_ops = rule.get("operators", [])
            if not isinstance(rule_ops, list):
                rule_ops = [rule_ops]
            
            if op.name not in rule_ops:
                continue
            
            # Match features
            rule_features = rule.get("semantic_features", {})
            similarity = self._feature_similarity(features, rule_features)
            score += similarity
        
        return score
    
    def _score_composition(self, op1, op2, features: Dict) -> float:
        """
        Score an operator pair based on feature matching.
        
        Args:
            op1: First operator
            op2: Second operator
            features: Feature dict from FeatureExtractor
        
        Returns:
            Score (higher = more likely)
        """
        # Base score from individual operators
        score1 = self._score_operator(op1, features)
        score2 = self._score_operator(op2, features)
        
        # Bonus for known compositions
        composition_bonus = 0.0
        for rule in self.rules:
            rule_ops = rule.get("operators", [])
            if isinstance(rule_ops, list) and len(rule_ops) >= 2:
                if op1.name in rule_ops and op2.name in rule_ops:
                    # Check if they appear in order
                    idx1 = rule_ops.index(op1.name)
                    idx2 = rule_ops.index(op2.name) if op2.name in rule_ops[idx1+1:] else -1
                    if idx2 > idx1:
                        rule_features = rule.get("semantic_features", {})
                        similarity = self._feature_similarity(features, rule_features)
                        composition_bonus += similarity * 2.0  # Double weight for known composition
        
        return score1 + score2 + composition_bonus
    
    def _feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute similarity between two feature dicts.
        
        Args:
            features1: First feature dict
            features2: Second feature dict
        
        Returns:
            Similarity score (0 to 1)
        """
        if not features2:
            return 0.0
        
        matches = 0
        total = 0
        
        for key in features2:
            if key not in features1:
                continue
            
            total += 1
            val1 = features1[key]
            val2 = features2[key]
            
            # Boolean features: exact match
            if isinstance(val1, bool) and isinstance(val2, bool):
                if val1 == val2:
                    matches += 1
            
            # Numeric features: close match (within 20%)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) < 0.2 * abs(val2 + 1e-6):
                    matches += 1
            
            # Other: exact match
            elif val1 == val2:
                matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def get_top_n_ops(self, task: Dict, operators: List, n: int = 10) -> List:
        """
        Get top N operators for a task.
        
        Args:
            task: ARC task dict
            operators: List of all operators
            n: Number of top operators to return
        
        Returns:
            List of top N operators
        """
        ranked = self.rank_single_ops(task, operators)
        return [op for op, score in ranked[:n]]
    
    def get_top_n_pairs(self, task: Dict, operators: List, n: int = 20) -> List[Tuple]:
        """
        Get top N operator pairs for a task.
        
        Args:
            task: ARC task dict
            operators: List of all operators
            n: Number of top pairs to return
        
        Returns:
            List of top N (op1, op2) tuples
        """
        ranked = self.rank_compositions(task, operators)
        return [pair for pair, score in ranked[:n]]


def test_prior_selector():
    """Quick test of prior selector."""
    print("Prior Selector loaded successfully!")
    
    # Test without rules first
    selector = PriorSelector()
    print(f"Rules loaded: {len(selector.rules)}")
    
    return True


if __name__ == "__main__":
    test_prior_selector()
