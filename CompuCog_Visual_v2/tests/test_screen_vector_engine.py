import sys
from pathlib import Path
import numpy as np
# Ensure repo root is on path for imports
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import importlib
mod = importlib.import_module('CompuCog_Visual_v2.gaming.screen_vector_engine')
ScreenVectorEngine = mod.ScreenVectorEngine

# Adjust import paths if necessary


def test_grid_config_even():
    sve = ScreenVectorEngine(short_axis_cells=16, bloom_alpha=0.2)
    assert sve.grid_config.cells_x % 2 == 0
    assert sve.grid_config.cells_y % 2 == 0


def test_pooling_and_core():
    sve = ScreenVectorEngine(short_axis_cells=8)
    # Create a synthetic frame with gradients so pooling is nontrivial
    H = int(sve.screen_info.screen_height_px)
    W = int(sve.screen_info.screen_width_px)
    frame = np.tile(np.linspace(0, 255, W, dtype=float), (H, 1))

    state = sve.process_frame(frame, prev_grid=None)
    assert state.core_block.avg_intensity >= 0.0
    assert isinstance(state.sectors, dict)


def test_sectors_exclude_core():
    sve = ScreenVectorEngine(short_axis_cells=8)
    H = int(sve.screen_info.screen_height_px)
    W = int(sve.screen_info.screen_width_px)
    frame = np.ones((H, W), dtype=float) * 128.0
    state = sve.process_frame(frame)
    # Ensure core cells are not present in UP sector indices
    core = set(sve.core_cells)
    up_indices = set(state.sectors.get('UP').cell_indices)
    assert core.isdisjoint(up_indices)


def test_weights_matrix_shape():
    sve = ScreenVectorEngine(short_axis_cells=8)
    w = sve.weights_matrix
    assert w.shape == (sve.grid_config.cells_y, sve.grid_config.cells_x)
