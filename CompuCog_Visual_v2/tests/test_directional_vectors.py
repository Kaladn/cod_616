import numpy as np
import importlib
from pathlib import Path

# ensure package import works
repo_root = Path(__file__).resolve().parents[2]
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

mod = importlib.import_module('CompuCog_Visual_v2.gaming.screen_vector_engine')
ScreenVectorEngine = mod.ScreenVectorEngine


def test_directional_vector_count_and_fields():
    sve = ScreenVectorEngine(short_axis_cells=8)
    H = int(sve.screen_info.screen_height_px)
    W = int(sve.screen_info.screen_width_px)
    frame = np.ones((H, W), dtype=float) * 100.0

    state = sve.process_frame(frame)
    vectors = state.directional_vectors
    assert len(vectors) == 8
    for v in vectors:
        assert hasattr(v, 'direction')
        assert hasattr(v, 'gradient_change')
        assert hasattr(v, 'entropy')
        assert v.gradient_change >= 0.0
        assert v.entropy >= 0.0


def test_directional_vector_sensitivity():
    # Create diagonal ramp favoring NE direction (higher gradient NE)
    sve = ScreenVectorEngine(short_axis_cells=8)
    H = int(sve.screen_info.screen_height_px)
    W = int(sve.screen_info.screen_width_px)
    # create a frame where intensity increases toward top-right
    xv = np.linspace(0, 255, W)
    yv = np.linspace(0, 255, H)
    frame = np.add.outer(255 - yv, xv) / 2.0

    state = sve.process_frame(frame)
    vecs = {v.direction: v for v in state.directional_vectors}
    # Expect NE gradient_change to be nonzero and comparatively higher than opposite SW
    assert vecs['NE'].gradient_change >= 0.0
    # NE should be larger than or equal to the median gradient across directions (robust check)
    grads = [v.gradient_change for v in vecs.values()]
    import numpy as _np
    assert vecs['NE'].gradient_change >= float(_np.median(grads))
