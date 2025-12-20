import tempfile
import yaml
from pathlib import Path
import importlib

# Ensure package imports work
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from CompuCog_Visual_v2.operators.crosshair_lock import CrosshairLockOperator, FrameSequence
from CompuCog_Visual_v2.gaming.screen_vector_state import (
    ScreenVectorState, ScreenInfo, GridConfig, CoreBlock, SectorDefinition, DirectionalVector, AnomalyMetrics
)


def make_state(core_val, sector_vals=None):
    screen_info = ScreenInfo(screen_width_px=1920, screen_height_px=1080, aspect_ratio=1920/1080)
    grid_config = GridConfig(cells_x=32, cells_y=32, cell_width_px=60.0, cell_height_px=33.75)
    core_block = CoreBlock(core_cells=[(15,15),(15,16),(16,15),(16,16)], core_bounds_px=(0,0,120,120), avg_intensity=core_val, motion_delta=0.0)
    sectors = {}
    if sector_vals is None:
        sector_vals = {'UP': 10.0, 'DOWN': 10.0, 'LEFT': 10.0, 'RIGHT': 10.0}
    for name, v in sector_vals.items():
        sectors[name] = SectorDefinition(name=name, cell_indices=[(0,0)], pixel_bounds=(0,0,10,10), weighted_avg_intensity=v, weighted_motion_delta=0.0)
    directional_vectors = [DirectionalVector(direction=d, gradient_change=0.0, entropy=0.0) for d in ['N','NE','E','SE','S','SW','W','NW']]
    anomaly_metrics = AnomalyMetrics(global_entropy=0.0, sector_entropies={}, horizontal_symmetry_score=1.0, vertical_symmetry_score=1.0, diagonal_symmetry_score=1.0, anomaly_flags=[])
    return ScreenVectorState(screen_info=screen_info, grid_config=grid_config, core_block=core_block, sectors=sectors, directional_vectors=directional_vectors, anomaly_metrics=anomaly_metrics, weights_matrix=[[1.0]])


def test_crosshair_lock_with_sve(tmp_path):
    # Create config file
    cfg = {
        'operators': {
            'crosshair_lock': {
                'core_over_sector_threshold': 0.1,
                'hit_marker_delta': 50.0,
                'velocity_window': 5
            }
        }
    }
    cfg_path = tmp_path / 'crosshair.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    op = CrosshairLockOperator(str(cfg_path))

    # Create frames: alternating low/high core intensity
    frames = [make_state(val) for val in [10, 200, 200, 200, 10]]
    seq = FrameSequence(frames=frames, t_start=0.0, t_end=1.0, src='test')

    res = op.analyze(seq)
    assert res is not None
    assert 'on_target_frames' in res.metrics
    assert res.metrics['on_target_frames'] >= 2
    assert res.metrics['hit_marker_frames'] >= 1

    # Determinism: repeated analyze yields same flags
    res2 = op.analyze(seq)
    assert res.flags == res2.flags
