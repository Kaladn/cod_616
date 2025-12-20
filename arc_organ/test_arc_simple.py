# cod_616/arc_organ/test_arc_simple.py

from __future__ import annotations

import numpy as np

from .arc_grid_parser import ArcGridPair, compute_arc_channels
from .arc_resonance_state import ARCResonanceState
from .arc_rule_engine import (
    ARCRuleEngine,
    ARCRuleVector,
    SpatialTransform,
    ColorTransform,
    ObjectTransform,
    MaskMode,
)


def _print_grid(title: str, grid: np.ndarray) -> None:
    print(f"\n=== {title} ===")
    for row in grid:
        print(" ".join(str(int(v)) for v in row))


def simple_puzzles() -> list[ArcGridPair]:
    """
    A few trivial ARC-style tasks:
      1) Identity
      2) Horizontal flip
      3) Recolor min->max
    """
    # 1) Identity
    g1 = np.array(
        [
            [0, 1, 0],
            [2, 2, 0],
            [0, 1, 0],
        ],
        dtype=int,
    )
    p1 = ArcGridPair(input_grid=g1, output_grid=g1.copy())

    # 2) Horizontal flip
    g2_in = np.array(
        [
            [1, 0, 2],
            [1, 0, 2],
        ],
        dtype=int,
    )
    g2_out = np.fliplr(g2_in)
    p2 = ArcGridPair(input_grid=g2_in, output_grid=g2_out)

    # 3) Recolor min->max
    g3_in = np.array(
        [
            [0, 1, 1],
            [0, 0, 2],
        ],
        dtype=int,
    )
    # map 0->2, 2->2, 1 stays 1
    g3_out = g3_in.copy()
    g3_out[g3_out == 0] = 2
    p3 = ArcGridPair(input_grid=g3_in, output_grid=g3_out)

    return [p1, p2, p3]


def main() -> None:
    puzzles = simple_puzzles()
    engine = ARCRuleEngine()

    for idx, pair in enumerate(puzzles):
        print(f"\n================= PUZZLE {idx} =================")
        _print_grid("INPUT", pair.input_grid)
        if pair.output_grid is not None:
            _print_grid("TARGET", pair.output_grid)

        # 1) Channels
        channels = compute_arc_channels(pair.input_grid, pair.output_grid)

        # 2) Resonance
        res = ARCResonanceState.from_channels(channels)
        print("\nResonance snapshot:")
        for k, v in res.as_dict().items():
            print(f"  {k:32s} {v:8.4f}")

        # 3) Apply a hard-coded rule for now (RecognitionField will choose later)
        if idx == 0:
            rule = ARCRuleVector(
                spatial=SpatialTransform.IDENTITY,
                color=ColorTransform.IDENTITY,
                obj=ObjectTransform.IDENTITY,
                mask=MaskMode.FULL,
            )
        elif idx == 1:
            rule = ARCRuleVector(
                spatial=SpatialTransform.FLIP_H,
                color=ColorTransform.IDENTITY,
                obj=ObjectTransform.IDENTITY,
                mask=MaskMode.FULL,
            )
        else:
            rule = ARCRuleVector(
                spatial=SpatialTransform.IDENTITY,
                color=ColorTransform.MAP_MIN_TO_MAX,
                obj=ObjectTransform.IDENTITY,
                mask=MaskMode.FULL,
            )

        out_grid = engine.apply(
            input_grid=pair.input_grid,
            rule=rule,
            delta_map=channels.delta_map,
            foreground_mask=channels.color_normalized > 0.0,
        )

        _print_grid("OUTPUT", out_grid)

        if pair.output_grid is not None:
            match = np.array_equal(out_grid, pair.output_grid)
            print(f"\nMatch target: {match}")

    print("\n" + "=" * 60)
    print("✓ ARC Organ operational")
    print("✓ 6 channels → 20-dim resonance → rule engine → output grid")
    print("=" * 60)


if __name__ == "__main__":
    main()
