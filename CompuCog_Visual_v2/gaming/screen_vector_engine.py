import ctypes
import math
import numpy as np
from typing import List, Tuple, Dict, Optional

from .screen_vector_state import (
    ScreenInfo,
    GridConfig,
    CoreBlock,
    SectorDefinition,
    ScreenVectorState,
    DirectionalVector,
    AnomalyMetrics,
)


class ScreenVectorEngine:
    """Resolution-aware engine implementing the SVE spec v4.0."""

    def __init__(self, short_axis_cells: int = 32, bloom_alpha: float = 0.1):
        self.short_axis_cells = short_axis_cells
        self.bloom_alpha = bloom_alpha

        s_width, s_height = self._get_screen_resolution()
        self.screen_info = ScreenInfo(screen_width_px=s_width, screen_height_px=s_height, aspect_ratio=s_width / s_height)

        # derive grid dimensions (ensure even numbers for 2x2 core)
        if self.screen_info.aspect_ratio >= 1:
            cells_y = self.short_axis_cells
            cells_x = int(round(cells_y * self.screen_info.aspect_ratio))
        else:
            cells_x = self.short_axis_cells
            cells_y = int(round(cells_x / self.screen_info.aspect_ratio))

        if cells_x % 2 != 0:
            cells_x += 1
        if cells_y % 2 != 0:
            cells_y += 1

        self.grid_config = GridConfig(
            cells_x=cells_x,
            cells_y=cells_y,
            cell_width_px=float(s_width) / cells_x,
            cell_height_px=float(s_height) / cells_y,
        )

        # center indices (column, row)
        self.center_x = self.grid_config.cells_x // 2
        self.center_y = self.grid_config.cells_y // 2

        # core cells expressed as (row, col) for numpy indexing
        self.core_cells = [
            (self.center_y - 1, self.center_x - 1),
            (self.center_y - 1, self.center_x),
            (self.center_y,     self.center_x - 1),
            (self.center_y,     self.center_x),
        ]

        self.weights_matrix = self._compute_weights_matrix()

    def _get_screen_resolution(self) -> Tuple[int, int]:
        try:
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return 1920, 1080

    def _compute_weights_matrix(self) -> np.ndarray:
        rows = self.grid_config.cells_y
        cols = self.grid_config.cells_x
        weights = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            for c in range(cols):
                dx = max(0, abs(c - self.center_x) + 0.5 - 1)
                dy = max(0, abs(r - self.center_y) + 0.5 - 1)
                dist = max(dx, dy)
                weights[r, c] = 1.0 / (1.0 + self.bloom_alpha * dist)
        return weights

    def process_frame(self, frame_px: np.ndarray, prev_grid: Optional[np.ndarray] = None) -> ScreenVectorState:
        # 1. Create grid using average pooling
        grid = self._create_grid_from_frame(frame_px)

        # 2. Delta
        delta_grid = grid - prev_grid if prev_grid is not None else np.zeros_like(grid)

        # 3. Components
        core_block = self._process_core_block(grid, delta_grid)
        sectors = self._process_sectors(grid, delta_grid)
        directional_vectors = self._compute_directional_vectors(grid)
        anomaly_metrics = self._compute_anomaly_metrics(grid, sectors)

        return ScreenVectorState(
            screen_info=self.screen_info,
            grid_config=self.grid_config,
            core_block=core_block,
            sectors=sectors,
            directional_vectors=directional_vectors,
            anomaly_metrics=anomaly_metrics,
            weights_matrix=self.weights_matrix.tolist(),
        )

    def _create_grid_from_frame(self, frame_px: np.ndarray) -> np.ndarray:
        # Robust average pooling: compute pixel bounds per cell and average
        H, W = frame_px.shape
        rows = self.grid_config.cells_y
        cols = self.grid_config.cells_x
        ch = self.grid_config.cell_height_px
        cw = self.grid_config.cell_width_px

        grid = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            y0 = int(round(r * ch))
            y1 = int(round((r + 1) * ch))
            y1 = min(y1, H)
            if y1 <= y0:
                y1 = min(y0 + 1, H)
            for c in range(cols):
                x0 = int(round(c * cw))
                x1 = int(round((c + 1) * cw))
                x1 = min(x1, W)
                if x1 <= x0:
                    x1 = min(x0 + 1, W)
                patch = frame_px[y0:y1, x0:x1]
                if patch.size == 0:
                    grid[r, c] = 0.0
                else:
                    grid[r, c] = float(np.mean(patch))
        return grid

    def _process_core_block(self, grid: np.ndarray, delta_grid: np.ndarray) -> CoreBlock:
        intensities = [grid[r, c] for (r, c) in self.core_cells]
        deltas = [delta_grid[r, c] for (r, c) in self.core_cells]

        x_min_px = (self.center_x - 1) * self.grid_config.cell_width_px
        y_min_px = (self.center_y - 1) * self.grid_config.cell_height_px
        x_max_px = (self.center_x + 1) * self.grid_config.cell_width_px
        y_max_px = (self.center_y + 1) * self.grid_config.cell_height_px

        return CoreBlock(
            core_cells=self.core_cells,
            core_bounds_px=(x_min_px, y_min_px, x_max_px, y_max_px),
            avg_intensity=float(np.mean(intensities)),
            motion_delta=float(np.mean(np.abs(deltas)))
        )

    def _process_sectors(self, grid: np.ndarray, delta_grid: np.ndarray) -> Dict[str, SectorDefinition]:
        rows = self.grid_config.cells_y
        cols = self.grid_config.cells_x

        # Sectors exclude the 2x2 core block
        up_indices = [(r, c) for r in range(0, self.center_y - 1) for c in range(cols)]
        down_indices = [(r, c) for r in range(self.center_y + 1, rows) for c in range(cols)]
        left_indices = [(r, c) for r in range(rows) for c in range(0, self.center_x - 1)]
        right_indices = [(r, c) for r in range(rows) for c in range(self.center_x + 1, cols)]

        sector_map = {
            "UP": up_indices,
            "DOWN": down_indices,
            "LEFT": left_indices,
            "RIGHT": right_indices,
        }

        sectors: Dict[str, SectorDefinition] = {}
        for name, indices in sector_map.items():
            if not indices:
                continue
            intensities = [grid[r, c] for (r, c) in indices]
            deltas = [delta_grid[r, c] for (r, c) in indices]
            weights = [self.weights_matrix[r, c] for (r, c) in indices]

            total_weight = float(sum(weights)) if weights else 0.0
            if total_weight > 0:
                weighted_intensity = sum(i * w for i, w in zip(intensities, weights)) / total_weight
                weighted_delta = sum(abs(d) * w for d, w in zip(deltas, weights)) / total_weight
            else:
                weighted_intensity = 0.0
                weighted_delta = 0.0

            min_r = min(r for r, c in indices)
            min_c = min(c for r, c in indices)
            max_r = max(r for r, c in indices)
            max_c = max(c for r, c in indices)

            bounds = (
                min_c * self.grid_config.cell_width_px,
                min_r * self.grid_config.cell_height_px,
                (max_c + 1) * self.grid_config.cell_width_px,
                (max_r + 1) * self.grid_config.cell_height_px,
            )

            sectors[name] = SectorDefinition(
                name=name,
                cell_indices=indices,
                pixel_bounds=bounds,
                weighted_avg_intensity=float(weighted_intensity),
                weighted_motion_delta=float(weighted_delta),
            )

        return sectors

    def _compute_directional_vectors(self, grid: np.ndarray) -> List[DirectionalVector]:
        """Compute 8 fixed directional vectors (N, NE, E, SE, S, SW, W, NW).

        Algorithm (deterministic):
        - For each direction, start at the cell adjacent to the 2x2 core and step cell-by-cell outward
        - Collect intensity samples along the ray until the grid edge
        - Compute gradient_change as the mean absolute difference between successive samples
        - Compute entropy as the Shannon entropy of the normalized absolute sample magnitudes
        """
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        vectors: List[DirectionalVector] = []

        rows = self.grid_config.cells_y
        cols = self.grid_config.cells_x

        # Direction unit steps in (dr, dc)
        dir_steps = {
            "N": (-1, 0),
            "NE": (-1, 1),
            "E": (0, 1),
            "SE": (1, 1),
            "S": (1, 0),
            "SW": (1, -1),
            "W": (0, -1),
            "NW": (-1, -1),
        }

        # Start positions: the first cell beyond the core in each direction
        # For axis-aligned directions we pick the row(s)/col(s) just outside the 2x2 core
        core_rows = {r for (r, c) in self.core_cells}
        core_cols = {c for (r, c) in self.core_cells}

        for d in dirs:
            dr, dc = dir_steps[d]

            # Determine a deterministic starting cell (row, col) outside the core
            start_r = self.center_y + (dr * 2) if dr != 0 else self.center_y - 1
            start_c = self.center_x + (dc * 2) if dc != 0 else self.center_x - 1

            # Adjust for axis-aligned directions to begin adjacent to core
            if d == "N":
                start_r = self.center_y - 2
                start_c = self.center_x - 1
            elif d == "S":
                start_r = self.center_y + 2
                start_c = self.center_x - 1
            elif d == "W":
                start_r = self.center_y - 1
                start_c = self.center_x - 2
            elif d == "E":
                start_r = self.center_y - 1
                start_c = self.center_x + 2
            elif d == "NE":
                start_r = self.center_y - 2
                start_c = self.center_x + 2
            elif d == "NW":
                start_r = self.center_y - 2
                start_c = self.center_x - 2
            elif d == "SE":
                start_r = self.center_y + 2
                start_c = self.center_x + 2
            elif d == "SW":
                start_r = self.center_y + 2
                start_c = self.center_x - 2

            # Clamp start within grid
            r, c = int(start_r), int(start_c)
            samples: List[float] = []

            while 0 <= r < rows and 0 <= c < cols:
                # Exclude core cells explicitly
                if (r, c) not in self.core_cells:
                    samples.append(float(grid[r, c]))
                r += dr
                c += dc

            if len(samples) < 2:
                gradient_change = 0.0
                entropy = 0.0
            else:
                samples_arr = np.asarray(samples, dtype=float)
                diffs = np.abs(np.diff(samples_arr))
                gradient_change = float(np.mean(diffs))

                # entropy on normalized magnitudes
                mags = np.abs(samples_arr)
                s = np.sum(mags)
                if s <= 0:
                    entropy = 0.0
                else:
                    p = mags / s
                    # numerical stability
                    p = p[p > 0]
                    entropy = float(-np.sum(p * np.log2(p)))

            vectors.append(DirectionalVector(direction=d, gradient_change=gradient_change, entropy=entropy))

        return vectors

    def _compute_anomaly_metrics(self, grid: np.ndarray, sectors: Dict[str, SectorDefinition]) -> AnomalyMetrics:
        """Compute global and sector entropies and symmetry scores.

        Rules:
        - Entropy: Shannon entropy of absolute intensities normalized (no heuristics)
        - Symmetry: 1 - normalized mean absolute difference (range 0..1)
        - Diagonal symmetry is computed on the central square region when grid is rectangular
        - Anomaly flags are binary and derived from principled thresholds (fractions of max entropy or fixed symmetry cutoff)
        """
        rows, cols = grid.shape

        # Helper: histogram-based entropy over intensity values (bins in 0..255)
        def hist_entropy(values: np.ndarray, bins: int = 16) -> float:
            if values.size == 0:
                return 0.0
            counts, _ = np.histogram(values, bins=bins, range=(0.0, 255.0))
            total = counts.sum()
            if total <= 0:
                return 0.0
            p = counts.astype(float) / float(total)
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))

        # GLOBAL ENTROPY (histogram-based)
        global_entropy = hist_entropy(np.abs(grid).flatten(), bins=16)

        # SECTOR ENTROPIES
        sector_entropies: Dict[str, float] = {}
        for name, sec in sectors.items():
            vals = np.array([grid[r, c] for (r, c) in sec.cell_indices], dtype=float)
            sector_entropies[name] = hist_entropy(np.abs(vals), bins=8)

        # SYMMETRY SCORES (1.0 = perfect symmetry, 0.0 = maximal difference)
        # horizontal (left/right)
        left = grid[:, :cols // 2]
        right = grid[:, cols - (cols // 2):]
        # flip right for comparison
        right_flipped = np.fliplr(right)
        # If shapes differ (odd cols), trim to min width
        minw = min(left.shape[1], right_flipped.shape[1])
        if minw == 0:
            horizontal_symmetry = 0.0
        else:
            left_cut = left[:, :minw]
            right_cut = right_flipped[:, :minw]
            diff = np.abs(left_cut - right_cut)
            horizontal_symmetry = 1.0 - (float(np.mean(diff)) / 255.0)
            horizontal_symmetry = max(0.0, min(1.0, horizontal_symmetry))

        # vertical (up/down)
        top = grid[:rows // 2, :]
        bottom = grid[rows - (rows // 2):, :]
        bottom_flipped = np.flipud(bottom)
        minh = min(top.shape[0], bottom_flipped.shape[0])
        if minh == 0:
            vertical_symmetry = 0.0
        else:
            top_cut = top[:minh, :]
            bottom_cut = bottom_flipped[:minh, :]
            diff = np.abs(top_cut - bottom_cut)
            vertical_symmetry = 1.0 - (float(np.mean(diff)) / 255.0)
            vertical_symmetry = max(0.0, min(1.0, vertical_symmetry))

        # diagonal symmetry: compare central square region against its transpose
        k = min(rows, cols)
        # take central k x k square
        r0 = max(0, (rows - k) // 2)
        c0 = max(0, (cols - k) // 2)
        block = grid[r0:r0 + k, c0:c0 + k]
        if block.size == 0 or k < 2:
            diagonal_symmetry = 0.0
        else:
            diff = np.abs(block - block.T)
            diagonal_symmetry = 1.0 - (float(np.mean(diff)) / 255.0)
            diagonal_symmetry = max(0.0, min(1.0, diagonal_symmetry))

        # Binary anomaly flags derived from thresholds:
        flags: List[str] = []
        # Entropy threshold: flag if global_entropy >= 90% of max entropy for the histogram bins
        GLOBAL_BINS = 16
        max_entropy_global = math.log2(GLOBAL_BINS) if GLOBAL_BINS > 0 else 0.0
        if max_entropy_global > 0 and global_entropy >= 0.9 * max_entropy_global:
            flags.append('high_entropy')

        # Sector high entropy flags (bins used for sector entropy)
        SECTOR_BINS = 8
        for name, ent in sector_entropies.items():
            max_e = math.log2(SECTOR_BINS) if SECTOR_BINS > 0 else 0.0
            if max_e > 0 and ent >= 0.9 * max_e:
                flags.append(f'sector_high_entropy_{name.lower()}')

        # Symmetry thresholds (fixed cutoffs)
        SYM_THRESHOLD = 0.5
        if horizontal_symmetry < SYM_THRESHOLD:
            flags.append('low_horizontal_symmetry')
        if vertical_symmetry < SYM_THRESHOLD:
            flags.append('low_vertical_symmetry')
        if diagonal_symmetry < SYM_THRESHOLD:
            flags.append('low_diagonal_symmetry')

        return AnomalyMetrics(
            global_entropy=float(global_entropy),
            sector_entropies=sector_entropies,
            horizontal_symmetry_score=float(horizontal_symmetry),
            vertical_symmetry_score=float(vertical_symmetry),
            diagonal_symmetry_score=float(diagonal_symmetry),
            anomaly_flags=flags,
        )