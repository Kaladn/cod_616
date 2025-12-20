import numpy as np
import importlib
from pathlib import Path

# ensure package import
repo_root = Path(__file__).resolve().parents[2]
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

mod = importlib.import_module('CompuCog_Visual_v2.gaming.screen_vector_engine')
ScreenVectorEngine = mod.ScreenVectorEngine


def test_global_and_sector_entropy_bounds():
    sve = ScreenVectorEngine(short_axis_cells=8)
    H = int(sve.screen_info.screen_height_px)
    W = int(sve.screen_info.screen_width_px)
    frame = np.ones((H, W), dtype=float) * 50.0

    state = sve.process_frame(frame)
    am = state.anomaly_metrics
    assert am.global_entropy == 0.0
    for v in am.sector_entropies.values():
        assert v == 0.0


def test_symmetry_scores_and_flags_determinism():
    sve = ScreenVectorEngine(short_axis_cells=8)
    H = int(sve.screen_info.screen_height_px)
    W = int(sve.screen_info.screen_width_px)
    # create a perfectly horizontally symmetric image
    left = np.tile(np.linspace(0, 255, W // 2), (H, 1))
    right = np.fliplr(left)
    frame = np.concatenate([left, right], axis=1)

    state1 = sve.process_frame(frame)
    state2 = sve.process_frame(frame)

    am1 = state1.anomaly_metrics
    am2 = state2.anomaly_metrics

    # symmetry scores near 1
    assert am1.horizontal_symmetry_score >= 0.99
    assert am1.vertical_symmetry_score >= 0.0

    # deterministic flags
    assert am1.anomaly_flags == am2.anomaly_flags


def test_diagonal_symmetry_on_square_region():
    sve = ScreenVectorEngine(short_axis_cells=8)
    # create a symmetric matrix about diagonal using a square central area
    H = int(sve.screen_info.screen_height_px)
    W = int(sve.screen_info.screen_width_px)
    # produce an image where central square is symmetric
    k = min(H, W)
    base = np.zeros((H, W), dtype=float)
    sq = np.arange(k).reshape(k, 1) + np.arange(k).reshape(1, k)
    # place sq centrally
    r0 = (H - k) // 2
    c0 = (W - k) // 2
    base[r0:r0 + k, c0:c0 + k] = sq

    state = sve.process_frame(base)
    # Allow small deviations due to pooling / boundary effects
    assert state.anomaly_metrics.diagonal_symmetry_score >= 0.85
