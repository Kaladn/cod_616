"""
6-1-6 ARC Operator Architecture

Each operator implements:
  - analyze(input, output) -> params or None
    (6: extract structure, 1: symbolic rule parameters)
  - apply(input, params) -> output
    (6: reconstruct output from symbol)
"""
import numpy as np
from scipy import ndimage
from typing import Any, Dict, List, Optional, Tuple


class ArcOperator:
    """
    6-1-6 ARC operator:
      - analyze(input, output) -> params or None
        (6: extract structure, 1: symbolic rule parameters)
      - apply(input, params) -> output
        (6: reconstruct output from symbol)
    """
    name: str = "base"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


# ======================================================================
#  Operator 1: HorizontalReplicateOperator
# ======================================================================

class HorizontalReplicateOperator(ArcOperator):
    """
    Detects and applies horizontal replication:
      output = input repeated horizontally N times.
    Constraints:
      - height is preserved
      - width_out is an integer multiple of width_in
      - tiles are exact copies (no recolor / rotation / flip)
    
    Solves: a416b8f3
    """
    name = "horizontal_replicate"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        # 6: structure constraints
        if H_out != H_in:
            return None
        if W_out % W_in != 0:
            return None

        repeat_factor = W_out // W_in
        
        # Must be at least 2× (reject identity)
        if repeat_factor < 2:
            return None

        expected = np.tile(input_grid, (1, repeat_factor))

        if not np.array_equal(expected, output_grid):
            return None

        # 1: symbolic parameters
        return {"repeat_factor": repeat_factor}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        # 6: reconstruct
        r = params["repeat_factor"]
        return np.tile(input_grid, (1, r))


# ======================================================================
#  Operator 2: RecolorMappingOperator
# ======================================================================

class RecolorMappingOperator(ArcOperator):
    """
    Pure recoloring:
      - input and output have the same shape
      - geometry is unchanged
      - there is a one-to-one mapping of input colors to output colors
      - at least one color is actually changed (non-identity)
    
    Solves: b1948b0a, c8f0f002, d511f180
    """
    name = "recolor_mapping"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        # 6: structure alignment
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape
        mapping: Dict[int, int] = {}
        reverse: Dict[int, int] = {}

        for r in range(H):
            for c in range(W):
                a = int(input_grid[r, c])
                b = int(output_grid[r, c])

                # input -> output consistency
                if a not in mapping:
                    mapping[a] = b
                elif mapping[a] != b:
                    return None

                # ensure one-to-one
                if b not in reverse:
                    reverse[b] = a
                elif reverse[b] != a:
                    return None

        # reject identity recolor (no actual rule)
        if all(k == v for k, v in mapping.items()):
            return None

        # 1: symbolic rule = mapping dictionary
        return {"mapping": mapping}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        # 6: reconstruct via mapping
        mapping: Dict[int, int] = params["mapping"]
        out = np.copy(input_grid)
        H, W = input_grid.shape

        for r in range(H):
            for c in range(W):
                v = int(out[r, c])
                out[r, c] = mapping.get(v, v)

        return out


# ======================================================================
#  Operator 3: TilingExpandOperator (general 2D tiling)
# ======================================================================

class TilingExpandOperator(ArcOperator):
    """
    General tiling:
      output = input tiled in a full 2D grid (Y×X).
    No recolor, no rotation.

    This is the generic "expand by tiling" operator.
    Horizontal replication is a special case (Y=1).
    """
    name = "tiling_expand"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        if H_out % H_in != 0 or W_out % W_in != 0:
            return None

        reps_y = H_out // H_in
        reps_x = W_out // W_in

        # Disallow trivial 1x1 (no change)
        if reps_y == 1 and reps_x == 1:
            return None

        expected = np.tile(input_grid, (reps_y, reps_x))

        if not np.array_equal(expected, output_grid):
            return None

        return {"reps_y": reps_y, "reps_x": reps_x}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        reps_y = params["reps_y"]
        reps_x = params["reps_x"]
        return np.tile(input_grid, (reps_y, reps_x))


# ======================================================================
#  Operator 4: SelfReferentialTilingOperator
# ======================================================================

class SelfReferentialTilingOperator(ArcOperator):
    """
    Self-referential tiling / fractal-ish cases (e.g. 007bbfb7).
    For now this is a constrained wrapper over TilingExpand:
      - enforce square replication (Y == X)
      - optionally enforce small fixed factors later if needed.
    
    Solves: 007bbfb7
    """
    name = "self_referential_tiling"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        if H_out % H_in != 0 or W_out % W_in != 0:
            return None

        reps_y = H_out // H_in
        reps_x = W_out // W_in

        # require square replication (same factor in both dims)
        if reps_y != reps_x or reps_y <= 1:
            return None

        expected = np.tile(input_grid, (reps_y, reps_x))
        if not np.array_equal(expected, output_grid):
            return None

        return {"factor": reps_y}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        f = params["factor"]
        return np.tile(input_grid, (f, f))


# ======================================================================
#  Operator 5: SandwichTilingOperator (00576224-style)
# ======================================================================

class SandwichTilingOperator(ArcOperator):
    """
    00576224 pattern:
      - input is a small tile (typically 2x2)
      - output is a 3x3 grid of tiles:
          row 0 & 2: original tile
          row 1:     90° clockwise rotation of tile
      - each tile row is repeated across all tile-columns.

    Generalized:
      - H_out = 3 * H_in
      - W_out = 3 * W_in
    
    Solves: 00576224
    """
    name = "sandwich_tiling"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        if H_out != 3 * H_in or W_out != 3 * W_in:
            return None

        base = input_grid
        rot = np.rot90(base, -1)  # 90° clockwise

        # Construct expected 3x3 tiling
        try:
            row0 = np.hstack([base, base, base])
            row1 = np.hstack([rot, rot, rot])
            row2 = np.hstack([base, base, base])
            expected = np.vstack([row0, row1, row2])
        except ValueError:
            # Size mismatch during stacking
            return None

        if not np.array_equal(expected, output_grid):
            return None

        return {
            "pattern": "orig-rot-orig",
            "grid_size": (3, 3),
            "rotation": "90cw",
        }

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        base = input_grid
        rot = np.rot90(base, -1)  # 90° clockwise

        row0 = np.hstack([base, base, base])
        row1 = np.hstack([rot,  rot,  rot])
        row2 = np.hstack([base, base, base])

        return np.vstack([row0, row1, row2])


# ======================================================================
#  Operator 6: DiagonalMarkerOperator (310f3251-style)
# ======================================================================

class DiagonalMarkerOperator(ArcOperator):
    """
    310f3251 pattern (Rule Set 2):
      - For each non-zero input cell at (r, c),
        compute (r-1 mod H, c-1 mod W)  [toroidal up-left diagonal]
      - If that target cell is 0 in the input, mark it with marker_color in output.
      - All other cells remain identical to input.

    This operator:
      - infers marker_color as the color present in output but not in input
      - validates that:
          * all marker_color pixels lie at exactly those diagonal targets
          * no other changes exist
    
    Solves: 310f3251
    """
    name = "diagonal_marker_toroidal"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape
        colors_in = set(int(x) for x in np.unique(input_grid))
        colors_out = set(int(x) for x in np.unique(output_grid))

        # marker color must be present in output but not in input
        extra = colors_out - colors_in
        if len(extra) != 1:
            return None
        marker_color = extra.pop()

        # build set of expected marker positions from rule
        expected_positions = set()
        for r in range(H):
            for c in range(W):
                if int(input_grid[r, c]) != 0:
                    dr = (r - 1) % H
                    dc = (c - 1) % W
                    if int(input_grid[dr, dc]) == 0:
                        expected_positions.add((dr, dc))

        # now verify output
        actual_positions = set()
        for r in range(H):
            for c in range(W):
                inp = int(input_grid[r, c])
                out = int(output_grid[r, c])

                if out == marker_color:
                    # must be a zero in the input at a valid position
                    if inp != 0:
                        return None
                    actual_positions.add((r, c))
                else:
                    # for non-marker cells, require equality
                    if out != inp:
                        return None

        if actual_positions != expected_positions:
            return None

        return {
            "marker_color": marker_color,
            "offset": (-1, -1),
        }

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        marker_color = int(params["marker_color"])
        H, W = input_grid.shape
        out = np.copy(input_grid)

        for r in range(H):
            for c in range(W):
                if int(input_grid[r, c]) != 0:
                    dr = (r - 1) % H
                    dc = (c - 1) % W
                    if int(input_grid[dr, dc]) == 0:
                        out[dr, dc] = marker_color

        return out


# ======================================================================
#  Operator 7: SwapMappingOperator (bidirectional color swap)
# ======================================================================

class SwapMappingOperator(ArcOperator):
    """
    Bidirectional color swap:
      - Same shape input/output.
      - Exactly two colors A and B are swapped:
          A -> B, B -> A
        All other colors remain unchanged (identity).
    
    Solves: d511f180
    """

    name = "swap_mapping"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape
        mapping: Dict[int, int] = {}

        for r in range(H):
            for c in range(W):
                a = int(input_grid[r, c])
                b = int(output_grid[r, c])

                if a not in mapping:
                    mapping[a] = b
                elif mapping[a] != b:
                    return None

        # Find which colors are actually changed
        changed = [k for k, v in mapping.items() if k != v]

        # For a pure swap, there must be exactly two changed colors
        if len(changed) != 2:
            return None

        a, b = changed
        # Must be a perfect swap
        if mapping[a] != b or mapping[b] != a:
            return None

        # All others must map to themselves
        for k, v in mapping.items():
            if k not in (a, b) and k != v:
                return None

        return {"a": a, "b": b}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        a = int(params["a"])
        b = int(params["b"])
        out = np.copy(input_grid)

        # swap a <-> b
        mask_a = (out == a)
        mask_b = (out == b)

        out[mask_a] = b
        out[mask_b] = a

        return out


# ======================================================================
#  Operator 8: SelfMaskingTilingOperator (007bbfb7-style)
# ======================================================================

class SelfMaskingTilingOperator(ArcOperator):
    """
    Self-masking tiling (007bbfb7 style):

      - Input is N x N (in the ARC task, N=3).
      - Output is (factor * N) x (factor * N), with factor = N.
      - Start from full tiling: tile = np.tile(input, (factor, factor))
      - For each tile-block (br, bc), look at input[br, bc]:
          if input[br, bc] == mask_color (usually 0):
              zero-out that entire block in the output.
          else:
              keep the tiled pattern in that block.

    This operator currently assumes:
      factor = H_out / H_in = W_out / W_in
      factor == H_in == W_in  (square 3x3 mask grid).
    
    Solves: 007bbfb7
    """

    name = "self_masking_tiling"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        # Must be square and scalable by a single factor
        if H_out % H_in != 0 or W_out % W_in != 0:
            return None

        factor_y = H_out // H_in
        factor_x = W_out // W_in
        if factor_y != factor_x:
            return None
        factor = factor_y

        # Require factor == H_in == W_in (3x3 → 9x9 in 007bbfb7)
        if H_in != W_in or H_in != factor:
            return None

        # Assume mask color is 0 (ARC-style background)
        mask_color = 0

        # Reconstruct via candidate rule and compare
        cand = self._build_self_masked_tiling(input_grid, factor, mask_color)

        if not np.array_equal(cand, output_grid):
            return None

        return {
            "factor": factor,
            "mask_color": mask_color,
        }

    def _build_self_masked_tiling(self, input_grid: np.ndarray,
                                  factor: int,
                                  mask_color: int) -> np.ndarray:
        H_in, W_in = input_grid.shape
        # Start from full tiling
        tiled = np.tile(input_grid, (factor, factor))
        out = np.zeros_like(tiled)

        # For each tile block, either keep or zero based on input[br, bc]
        for br in range(factor):
            for bc in range(factor):
                val = int(input_grid[br, bc])
                r0 = br * H_in
                c0 = bc * W_in
                block = tiled[r0:r0 + H_in, c0:c0 + W_in]
                if val == mask_color:
                    out[r0:r0 + H_in, c0:c0 + W_in] = mask_color
                else:
                    out[r0:r0 + H_in, c0:c0 + W_in] = block

        return out

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        factor = int(params["factor"])
        mask_color = int(params["mask_color"])
        return self._build_self_masked_tiling(input_grid, factor, mask_color)


# ======================================================================
#  Operator 9: LargestBlobExtractOperator
# ======================================================================

class LargestBlobExtractOperator(ArcOperator):
    """
    Keep only the largest connected component (4-connected) of non-zero cells.
    No recolor, no movement, same shape:
      - output[r, c] == input[r, c] for cells in the largest blob
      - output[r, c] == 0 for all other cells
    """

    name = "largest_blob_extract"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape

        # output must never introduce new non-zero or recolor
        for r in range(H):
            for c in range(W):
                a = int(input_grid[r, c])
                b = int(output_grid[r, c])
                if b != 0 and b != a:
                    return None

        # Compute largest blob in input (non-zero)
        largest_mask = self._largest_blob_mask(input_grid)
        if largest_mask is None:
            return None

        # Validate that output = largest blob only
        for r in range(H):
            for c in range(W):
                a = int(input_grid[r, c])
                b = int(output_grid[r, c])
                in_blob = largest_mask[r, c]

                if in_blob:
                    if b != a:
                        return None
                else:
                    if b != 0:
                        return None

        # No specific params needed other than "largest blob" rule
        return {"mode": "largest"}

    def _largest_blob_mask(self, grid: np.ndarray) -> Optional[np.ndarray]:
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        blobs: List[List[Tuple[int, int]]] = []

        def neighbors(r, c):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    yield nr, nc

        for r in range(H):
            for c in range(W):
                if visited[r, c]:
                    continue
                if grid[r, c] == 0:
                    visited[r, c] = True
                    continue

                # BFS this blob
                color = grid[r, c]
                stack = [(r, c)]
                visited[r, c] = True
                blob = [(r, c)]

                while stack:
                    rr, cc = stack.pop()
                    for nr, nc in neighbors(rr, cc):
                        if visited[nr, nc]:
                            continue
                        if grid[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                            blob.append((nr, nc))

                blobs.append(blob)

        if not blobs:
            return None

        # Choose largest blob (tie-break: first)
        largest_blob = max(blobs, key=len)
        mask = np.zeros((H, W), dtype=bool)
        for r, c in largest_blob:
            mask[r, c] = True

        return mask

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        mask = self._largest_blob_mask(input_grid)
        if mask is None:
            # If no non-zero, just return zeros
            return np.zeros_like(input_grid)
        out = np.zeros_like(input_grid)
        out[mask] = input_grid[mask]
        return out


# ======================================================================
#  Operator 10: MirrorOperator
# ======================================================================

class MirrorOperator(ArcOperator):
    """
    Simple mirror:
      - Same shape.
      - Output is either a horizontal or vertical mirror of input:
          horizontal: np.fliplr(input)
          vertical:   np.flipud(input)
    """

    name = "mirror"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        horiz = np.fliplr(input_grid)
        if np.array_equal(horiz, output_grid):
            return {"axis": "horizontal"}

        vert = np.flipud(input_grid)
        if np.array_equal(vert, output_grid):
            return {"axis": "vertical"}

        return None

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        axis = params["axis"]
        if axis == "horizontal":
            return np.fliplr(input_grid)
        elif axis == "vertical":
            return np.flipud(input_grid)
        else:
            # Fallback: no-op if somehow mis-specified
            return np.copy(input_grid)


# ======================================================================
#  Operator 11: CropToBoundingBoxOperator
# ======================================================================

class CropToBoundingBoxOperator(ArcOperator):
    """
    Crop to the minimal bounding box of non-zero cells.
    Rule:
      - Find min/max row/col indices where input != 0.
      - Output must equal input[rmin:rmax+1, cmin:cmax+1].
    """

    name = "crop_to_bounding_box"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        # Find bounding box of non-zero in input
        coords = np.argwhere(input_grid != 0)
        if coords.size == 0:
            return None  # no obvious rule: nothing to crop

        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)

        box = input_grid[rmin:rmax+1, cmin:cmax+1]

        if box.shape != (H_out, W_out):
            return None

        if not np.array_equal(box, output_grid):
            return None

        return {
            "rmin": int(rmin),
            "rmax": int(rmax),
            "cmin": int(cmin),
            "cmax": int(cmax),
        }

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        # We can recompute from scratch for new input, ignoring stored params,
        # but we keep them for consistency with the analyze result.
        coords = np.argwhere(input_grid != 0)
        if coords.size == 0:
            # If nothing non-zero, return a 1x1 zero grid (or zeros like input).
            return np.zeros((1, 1), dtype=input_grid.dtype)

        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)

        return input_grid[rmin:rmax+1, cmin:cmax+1]


# ======================================================================
#  Operator Pack 3 (Tier 2): Structural & Pattern Operators
# ======================================================================

class PositionBasedRecolorOperator(ArcOperator):
    """
    Position-based periodic recolor:
      - Same shape.
      - Output color at (r, c) depends only on (r % k, c % m),
        not on the input color.
      - Good for stripes, checkerboards, small repeating motifs.
    """

    name = "position_based_recolor"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape
        max_k = min(4, H)
        max_m = min(4, W)

        # Try small periods
        best_params = None

        for k in range(1, max_k + 1):
            for m in range(1, max_m + 1):
                # Skip trivial (1,1): that's just constant fill (handled elsewhere)
                if k == 1 and m == 1:
                    continue

                pattern = {}
                ok = True
                for r in range(H):
                    for c in range(W):
                        key = (r % k, c % m)
                        val = int(output_grid[r, c])
                        if key not in pattern:
                            pattern[key] = val
                        elif pattern[key] != val:
                            ok = False
                            break
                    if not ok:
                        break

                if not ok:
                    continue

                # This k, m works. Prefer smallest pattern.
                best_params = {
                    "k": k,
                    "m": m,
                    "pattern": pattern,  # mapping (rk, ck) -> color
                }
                break
            if best_params is not None:
                break

        return best_params

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        H, W = input_grid.shape
        k = int(params["k"])
        m = int(params["m"])
        pattern: Dict[Tuple[int, int], int] = params["pattern"]

        out = np.zeros_like(input_grid)
        for r in range(H):
            for c in range(W):
                key = (r % k, c % m)
                out[r, c] = pattern.get(key, 0)
        return out


class StructuredRecolorOperator(ArcOperator):
    """
    Structured position-based recolor with per-example color remapping.

    Idea:
      - All training outputs share the same (k, m) period.
      - Their per-cell pattern differs only by a one-to-one recoloring
        of some canonical base pattern.

    This captures cases where the *lattice* (3x3, 2x2, etc.) is the
    real invariant and colors are arbitrary per-example.

    For now, at apply-time we just use the canonical pattern from the
    first example. This will solve tasks where test output uses the
    same canonical pattern (which is surprisingly common).
    """

    name = "structured_position_recolor"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        # This operator can't infer rule from a single example in isolation,
        # so analyze() here is only meaningful in multi-example merging
        # via merge_operator_params. To keep the interface consistent,
        # we just delegate to PositionBasedRecolorOperator for the
        # single pair.
        base_op = PositionBasedRecolorOperator()
        return base_op.analyze(input_grid, output_grid)

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        # Same apply as PositionBasedRecolor, using canonical pattern.
        H, W = input_grid.shape
        k = int(params["k"])
        m = int(params["m"])
        pattern: Dict[Tuple[int, int], int] = params["pattern"]

        out = np.zeros_like(input_grid)
        for r in range(H):
            for c in range(W):
                key = (r % k, c % m)
                out[r, c] = pattern.get(key, 0)
        return out


class LatticeRoleRecolorOperator(ArcOperator):
    """
    Lattice-based role recolor with input-conditioned color semantics.
    
    THE TIER 3 UNLOCK:
    
    - Detects periodic lattice structure (k, m) across training examples
    - Extracts ROLES: spatial equivalence classes of lattice positions
    - Learns INPUT→ROLE mapping: which input colors drive which roles
    - Learns ROLE→OUTPUT mapping: canonical output color per role
    - At test time: reads test input colors, maps to roles, outputs canonical colors
    
    This solves tasks where:
    - Spatial structure is consistent (same k, m lattice)
    - Output pattern shape is consistent (same roles)
    - But COLORS vary per example (including test)
    
    Unlocks: 05269061, 0d3d703e, 3c9b0459, 6150a2bd, 74dd1130, caa06a1f,
             d037b0a7, e9afcf9a, ed36ccf7, and more
    """

    name = "lattice_role_recolor"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Single-example analysis: detect lattice structure.
        Full role inference happens in merge_operator_params.
        """
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape
        max_k = min(4, H)
        max_m = min(4, W)

        # Try small periods (same logic as PositionBasedRecolor)
        for k in range(1, max_k + 1):
            for m in range(1, max_m + 1):
                if k == 1 and m == 1:
                    continue  # Trivial constant fill

                # Check if output is periodic with (k, m)
                output_pattern = {}
                ok = True
                for r in range(H):
                    for c in range(W):
                        key = (r % k, c % m)
                        val = int(output_grid[r, c])
                        if key not in output_pattern:
                            output_pattern[key] = val
                        elif output_pattern[key] != val:
                            ok = False
                            break
                    if not ok:
                        break

                if not ok:
                    continue

                # Build input pattern at same lattice positions
                input_pattern = {}
                for r in range(H):
                    for c in range(W):
                        key = (r % k, c % m)
                        val = int(input_grid[r, c])
                        if key not in input_pattern:
                            input_pattern[key] = []
                        input_pattern[key].append(val)

                # Simplify: use most common input color per lattice position
                input_dominant = {}
                for key, colors in input_pattern.items():
                    colors_arr = np.array(colors)
                    vals, counts = np.unique(colors_arr, return_counts=True)
                    input_dominant[key] = int(vals[np.argmax(counts)])

                return {
                    "k": k,
                    "m": m,
                    "output_pattern": output_pattern,
                    "input_pattern": input_dominant,
                }

        return None

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        """
        Apply learned role mapping to test input.
        
        CRITICAL INSIGHT:
        Test input may use DIFFERENT color palette than training.
        We need to build a test-specific color mapping by analyzing
        which test colors appear at which lattice positions, then
        map those to the canonical output colors.
        
        Strategy:
        1. For each test input color, collect which lattice positions it appears at
        2. Match those positions to training's input_pattern
        3. Build test_color -> role -> output_color mapping
        """
        H, W = input_grid.shape
        k = int(params["k"])
        m = int(params["m"])
        
        input_pattern: Dict[Tuple[int, int], int] = params["input_pattern"]
        output_pattern: Dict[Tuple[int, int], int] = params["output_pattern"]
        
        # Step 1: Analyze test input - which colors appear at which lattice positions?
        test_color_at_pos = {}
        for r in range(H):
            for c in range(W):
                lattice_pos = (r % k, c % m)
                test_color = int(input_grid[r, c])
                if lattice_pos not in test_color_at_pos:
                    test_color_at_pos[lattice_pos] = []
                test_color_at_pos[lattice_pos].append(test_color)
        
        # Step 2: Find dominant test color per lattice position
        test_dominant = {}
        for pos, colors in test_color_at_pos.items():
            colors_arr = np.array(colors)
            vals, counts = np.unique(colors_arr, return_counts=True)
            test_dominant[pos] = int(vals[np.argmax(counts)])
        
        # Step 3: Build mapping from test_color -> output_color
        # by matching test color distribution to training input pattern
        test_color_to_output = {}
        for lattice_pos in input_pattern.keys():
            train_input_color = input_pattern[lattice_pos]
            test_color = test_dominant.get(lattice_pos, 0)
            output_color = output_pattern[lattice_pos]
            
            # Map test color to output color via this lattice position
            if test_color not in test_color_to_output:
                test_color_to_output[test_color] = output_color
        
        # Step 4: Apply the mapping
        out = np.zeros_like(input_grid)
        for r in range(H):
            for c in range(W):
                test_color = int(input_grid[r, c])
                lattice_pos = (r % k, c % m)
                
                # Use mapped color if available, else use output pattern default
                if test_color in test_color_to_output:
                    out[r, c] = test_color_to_output[test_color]
                else:
                    out[r, c] = output_pattern.get(lattice_pos, 0)
        
        return out


# ======================================================================
#  TIER 3.A: PATTERN GENERATOR - Latin Square From Diagonal
# ======================================================================

class LatinSquareFromDiagonalOperator(ArcOperator):
    """
    Pattern generator operator for tasks with diagonal color inputs.
    
    Rule (discovered from 05269061):
    1. Extract 3 unique non-zero colors from input (appear on diagonal)
    2. Determine their ordering by scanning the input
    3. Generate a 3x3 Latin square rotation pattern:
       [c0, c1, c2]
       [c1, c2, c0]  
       [c2, c0, c1]
    4. Tile to output dimensions
    
    This is a GENERATIVE operator (Tier 3.A):
    - Not a recolor (no 1:1 mapping)
    - Not a transform (creates new structure)
    - Synthesizes a periodic pattern from input color palette
    
    Key insight: The diagonal arrangement encodes the color ordering
    for the Latin square pattern.
    
    Solves: 05269061 (and likely similar diagonal→pattern tasks)
    """
    name = "latin_square_from_diagonal"
    
    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect if output is a 3x3 Latin square tiled pattern
        derived from input's non-zero colors.
        """
        # Check if output is periodic with period 3x3
        H_out, W_out = output_grid.shape
        
        if H_out < 3 or W_out < 3:
            return None
        
        # Extract the 3x3 pattern from output
        pattern_3x3 = output_grid[:3, :3]
        
        # Verify it tiles correctly
        for r in range(H_out):
            for c in range(W_out):
                expected = pattern_3x3[r % 3, c % 3]
                if output_grid[r, c] != expected:
                    return None
        
        # Check if it's a Latin square rotation pattern
        # Latin square rotation: each row is a cyclic shift
        c0, c1, c2 = pattern_3x3[0, 0], pattern_3x3[0, 1], pattern_3x3[0, 2]
        
        # Verify second row is cyclic shift: [c1, c2, c0]
        if (pattern_3x3[1, 0] != c1 or 
            pattern_3x3[1, 1] != c2 or 
            pattern_3x3[1, 2] != c0):
            return None
        
        # Verify third row is cyclic shift: [c2, c0, c1]
        if (pattern_3x3[2, 0] != c2 or 
            pattern_3x3[2, 1] != c0 or 
            pattern_3x3[2, 2] != c1):
            return None
        
        # Extract unique non-zero colors from input
        unique_input_colors = []
        for r in range(input_grid.shape[0]):
            for c in range(input_grid.shape[1]):
                color = int(input_grid[r, c])
                if color != 0 and color not in unique_input_colors:
                    unique_input_colors.append(color)
        
        # Must have exactly 3 non-zero colors
        if len(unique_input_colors) != 3:
            return None
        
        # Output colors must match input colors (no new colors)
        output_colors = set([int(c0), int(c1), int(c2)])
        if set(unique_input_colors) != output_colors:
            return None
        
        # Success: found Latin square pattern
        # Store the color ordering [c0, c1, c2]
        return {
            "colors": [int(c0), int(c1), int(c2)],
            "output_shape": output_grid.shape
        }
    
    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        """
        Generate Latin square pattern using LINEAR SEQUENCE formula.
        
        KEY INSIGHT (from user): Pattern cycles continuously across rows.
        Example: 1231231, 2312312, 3123123
        Row 0 ends at position 6, Row 1 starts at position 7 (continues sequence).
        
        Formula: output[r,c] = palette[(r*WIDTH + c) % 3]
        
        Strategy:
        1. Extract 3 colors from input diagonals (r+c = constant)
        2. Determine palette ordering by trying permutations
        3. Use linear row-major indexing, not diagonal indexing
        """
        output_shape = params["output_shape"]
        H_out, W_out = output_shape
        
        # Extract colors from diagonals in input
        diagonal_map = {}
        H_in, W_in = input_grid.shape
        for r in range(H_in):
            for c in range(W_in):
                color = int(input_grid[r, c])
                if color != 0:
                    diag = r + c
                    if diag not in diagonal_map:
                        diagonal_map[diag] = set()
                    diagonal_map[diag].add(color)
        
        if len(diagonal_map) != 3:
            return np.zeros(output_shape, dtype=input_grid.dtype)
        
        # Get colors from diagonals (one per diagonal)
        diagonal_colors = []
        for diag in sorted(diagonal_map.keys()):
            colors_in_diag = sorted(diagonal_map[diag])
            diagonal_colors.append(colors_in_diag[0])
        
        if len(diagonal_colors) != 3:
            return np.zeros(output_shape, dtype=input_grid.dtype)
        
        # BREAKTHROUGH: Palette ordering is determined by (diagonal % 3)
        # Position (0, k) in output corresponds to diagonal where (d % 3) == k
        # Example: diagonal 8 has (8%3)=2, so its color goes to position 2
        
        palette = [None, None, None]
        diag_indices = sorted(diagonal_map.keys())
        for j, diag in enumerate(diag_indices):
            target_position = diag % 3
            palette[target_position] = diagonal_colors[j]
        
        # Generate output using linear row-major indexing
        # Formula: output[r,c] = palette[(r*WIDTH + c) % 3]
        result = np.array([[palette[(r*W_out + c) % 3] for c in range(W_out)] 
                          for r in range(H_out)], dtype=input_grid.dtype)
        
        return result


class GridSubdivisionOperator(ArcOperator):
    """
    Block-wise majority recolor:
      - Same shape.
      - Partition into blocks of size (bh, bw).
      - Each output block is a single color, which must be the
        most frequent color in the corresponding input block.
    """

    name = "grid_subdivision_majority"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape
        # Try small block sizes that divide H, W
        candidates = []
        for bh in range(1, H + 1):
            if H % bh != 0:
                continue
            for bw in range(1, W + 1):
                if W % bw != 0:
                    continue
                # Skip trivial whole-grid block
                if bh == H and bw == W:
                    continue
                candidates.append((bh, bw))

        for bh, bw in candidates:
            if self._matches_blocks(input_grid, output_grid, bh, bw):
                return {"bh": bh, "bw": bw}

        return None

    def _matches_blocks(self, inp: np.ndarray, out: np.ndarray,
                        bh: int, bw: int) -> bool:
        H, W = inp.shape
        for r0 in range(0, H, bh):
            for c0 in range(0, W, bw):
                block_in = inp[r0:r0 + bh, c0:c0 + bw]
                block_out = out[r0:r0 + bh, c0:c0 + bw]

                # Output block must be uniform
                vals = np.unique(block_out)
                if len(vals) != 1:
                    return False
                out_col = int(vals[0])

                # Find majority color in input block
                flat = block_in.flatten()
                if flat.size == 0:
                    return False
                # Count frequencies
                vals_in, counts = np.unique(flat, return_counts=True)
                majority_color = int(vals_in[np.argmax(counts)])

                if out_col != majority_color:
                    return False
        return True

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        bh = int(params["bh"])
        bw = int(params["bw"])
        inp = input_grid
        H, W = inp.shape
        out = np.zeros_like(inp)

        for r0 in range(0, H, bh):
            for c0 in range(0, W, bw):
                block_in = inp[r0:r0 + bh, c0:c0 + bw]
                flat = block_in.flatten()
                vals_in, counts = np.unique(flat, return_counts=True)
                majority_color = int(vals_in[np.argmax(counts)])
                out[r0:r0 + bh, c0:c0 + bw] = majority_color

        return out


class SymmetryCompletionOperator(ArcOperator):
    """
    Complete a vertical or horizontal symmetry:

      - output is exactly symmetric (flip == output).
      - input is a partial version:
          * where input != 0, it must equal output
          * input may have zeros where output has non-zeros
      - No recolor, no movement beyond symmetry completion.
    """

    name = "symmetry_completion"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        H, W = input_grid.shape

        # Check vertical symmetry
        vertical_sym = np.array_equal(output_grid, np.flipud(output_grid))
        horizontal_sym = np.array_equal(output_grid, np.fliplr(output_grid))

        axis = None
        if vertical_sym:
            axis = "vertical"
        elif horizontal_sym:
            axis = "horizontal"
        else:
            return None

        # Check consistency with input: input's non-zero cells must match output
        for r in range(H):
            for c in range(W):
                a = int(input_grid[r, c])
                b = int(output_grid[r, c])
                if a != 0 and a != b:
                    return None

        # Require that output actually adds something (not identical to input)
        if np.array_equal(input_grid, output_grid):
            return None

        return {"axis": axis}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        axis = params["axis"]
        H, W = input_grid.shape
        out = np.copy(input_grid)

        if axis == "horizontal":
            # mirror left-right
            for r in range(H):
                for c in range(W):
                    c2 = W - 1 - c
                    v1 = out[r, c]
                    v2 = out[r, c2]
                    # If one side has color and the other is 0, propagate color
                    if v1 == 0 and v2 != 0:
                        out[r, c] = v2
                    elif v2 == 0 and v1 != 0:
                        out[r, c2] = v1
                    elif v1 != 0 and v2 != 0 and v1 != v2:
                        # conflict; keep original, but this shouldn't happen if
                        # training examples were consistent
                        pass
        elif axis == "vertical":
            # mirror up-down
            for r in range(H):
                for c in range(W):
                    r2 = H - 1 - r
                    v1 = out[r, c]
                    v2 = out[r2, c]
                    if v1 == 0 and v2 != 0:
                        out[r, c] = v2
                    elif v2 == 0 and v1 != 0:
                        out[r2, c] = v1
                    elif v1 != 0 and v2 != 0 and v1 != v2:
                        pass

        return out


class ColorHistogramFillOperator(ArcOperator):
    """
    Uniform fill based on color histogram:

      - Output is a constant color grid (all cells same).
      - That color is either:
          * the most common non-zero color in input, or
          * the least common non-zero color in input.

      The output shape is assumed equal to input for now.
    """

    name = "color_histogram_fill"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        # Output must be uniform
        vals_out = np.unique(output_grid)
        if len(vals_out) != 1:
            return None
        if input_grid.shape != output_grid.shape:
            return None

        c_out = int(vals_out[0])

        # Count non-zero colors in input
        flat = input_grid.flatten()
        flat_nz = flat[flat != 0]
        if flat_nz.size == 0:
            return None

        colors, counts = np.unique(flat_nz, return_counts=True)
        most_color = int(colors[np.argmax(counts)])
        least_color = int(colors[np.argmin(counts)])

        if c_out == most_color:
            mode = "most"
        elif c_out == least_color:
            mode = "least"
        else:
            return None

        return {"mode": mode}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        mode = params["mode"]
        flat = input_grid.flatten()
        flat_nz = flat[flat != 0]
        if flat_nz.size == 0:
            # fallback: just zeros
            fill_color = 0
        else:
            colors, counts = np.unique(flat_nz, return_counts=True)
            if mode == "most":
                fill_color = int(colors[np.argmax(counts)])
            else:
                fill_color = int(colors[np.argmin(counts)])

        H, W = input_grid.shape
        out = np.full((H, W), fill_color, dtype=input_grid.dtype)
        return out


class GrowShrinkOperator(ArcOperator):
    """
    Morphological grow/shrink of a single foreground color via 4-neighborhood:

      - Same shape.
      - Exactly one non-zero color in both input & output (foreground).
      - Either:
          * 'dilate': output foreground = dilation of input foreground
          * 'erode' : output foreground = erosion of input foreground
    """

    name = "grow_shrink"

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        if input_grid.shape != output_grid.shape:
            return None

        # Extract non-zero colors
        in_nz = input_grid[input_grid != 0]
        out_nz = output_grid[output_grid != 0]

        # Require a single foreground color in input & output
        if in_nz.size == 0 or out_nz.size == 0:
            return None

        in_colors = np.unique(in_nz)
        out_colors = np.unique(out_nz)
        if len(in_colors) != 1 or len(out_colors) != 1:
            return None

        fg = int(in_colors[0])
        if fg != int(out_colors[0]):
            return None

        inp_mask = (input_grid == fg)
        out_mask = (output_grid == fg)

        # Try dilation check
        dil = self._dilate(inp_mask)
        if np.array_equal(dil, out_mask) and not np.array_equal(inp_mask, out_mask):
            return {"mode": "dilate", "fg": fg}

        # Try erosion check
        ero = self._erode(inp_mask)
        if np.array_equal(ero, out_mask) and not np.array_equal(inp_mask, out_mask):
            return {"mode": "erode", "fg": fg}

        return None

    def _neighbors4(self, H: int, W: int, r: int, c: int):
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                yield nr, nc

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape
        out = np.zeros_like(mask)
        for r in range(H):
            for c in range(W):
                # If any neighbor is foreground, this becomes foreground
                for nr, nc in self._neighbors4(H, W, r, c):
                    if mask[nr, nc]:
                        out[r, c] = True
                        break
        return out

    def _erode(self, mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape
        out = np.zeros_like(mask)
        for r in range(H):
            for c in range(W):
                if not mask[r, c]:
                    continue
                keep = True
                for nr, nc in self._neighbors4(H, W, r, c):
                    if not mask[nr, nc]:
                        keep = False
                        break
                out[r, c] = keep
        return out

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        mode = params["mode"]
        fg = int(params["fg"])
        H, W = input_grid.shape
        mask = (input_grid == fg)
        if mode == "dilate":
            new_mask = self._dilate(mask)
        else:
            new_mask = self._erode(mask)

        out = np.zeros_like(input_grid)
        out[new_mask] = fg
        return out


# ======================================================================
#  Operator: CenterHaloExpansionOperator
# ======================================================================

from .arc_centering import (
    extract_center_object, 
    make_centered_canvas,
    centered_diff,
    centered_growth_detector,
    centered_color_roles,
)

class CenterHaloExpansionOperator(ArcOperator):
    """
    Center-anchor + halo expansion pattern (3befdf3e):
    
    Uses Lee's centering primitive to:
    1. Extract center object (removes noise/padding)
    2. Identify ring (majority color) and core (minority color)
    3. Double the size: N×N → 2N×2N
    4. Apply expansion rule:
       - Corners of output: stay background (0)
       - Outer edges: ring color
       - Interior: core color expands
       - Original core positions: marked with ring color
    
    Example:
      Input 3×3:  4 4 4        Output 5×5:  0 4 4 4 0
                  4 6 4    →                4 6 6 6 4
                  4 4 4                     4 6 4 6 4
                                            4 6 6 6 4
                                            0 4 4 4 0
    
    Solves: 3befdf3e
    """
    name = "center_halo_expansion"

    def _find_halo_structure(self, grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Find center-anchor + halo structure in grid.
        
        Universal center detection:
        - Odd-sized core: single center cell
        - Even-sized core: center 2×2 (or larger) block treated as one anchor
        
        Returns: {
            'bbox': (min_r, min_c, max_r, max_c),
            'core_color': int,
            'ring_color': int,
            'core_region': set of (r, c) positions that are core
        }
        """
        # Find non-zero bounding box
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return None
        
        min_r, min_c = non_zero.min(axis=0)
        max_r, max_c = non_zero.max(axis=0)
        max_r += 1
        max_c += 1
        
        # Extract region
        region = grid[min_r:max_r, min_c:max_c]
        h, w = region.shape
        
        # Must be at least 3×3
        if h < 3 or w < 3:
            return None
        
        # Find unique non-zero colors
        unique_colors = set(region.flatten()) - {0}
        if len(unique_colors) != 2:
            return None
        
        # Identify core and ring by counting: core is minority
        from collections import Counter
        color_counts = Counter(region.flatten())
        if 0 in color_counts:
            del color_counts[0]
        
        colors_sorted = sorted(color_counts.items(), key=lambda x: x[1])
        core_color = colors_sorted[0][0]
        ring_color = colors_sorted[1][0]
        
        # Find all core positions (the minority color)
        core_positions = set()
        for r in range(h):
            for c in range(w):
                if region[r, c] == core_color:
                    core_positions.add((r, c))
        
        if len(core_positions) == 0:
            return None
        
        # Check if center contains core color (validates structure)
        center_r = h // 2
        center_c = w // 2
        
        # For even dimensions, check center 2×2 block
        has_core_at_center = False
        if h % 2 == 0 and w % 2 == 0:
            # Check center 4 cells
            for dr in [-1, 0]:
                for dc in [-1, 0]:
                    if (center_r + dr, center_c + dc) in core_positions:
                        has_core_at_center = True
                        break
        else:
            # Check single center or center row/col
            if (center_r, center_c) in core_positions:
                has_core_at_center = True
        
        if not has_core_at_center:
            # Try swapping
            core_color, ring_color = ring_color, core_color
            core_positions = set()
            for r in range(h):
                for c in range(w):
                    if region[r, c] == core_color:
                        core_positions.add((r, c))
        
        return {
            'bbox': (min_r, min_c, max_r, max_c),
            'core_color': core_color,
            'ring_color': ring_color,
            'core_positions': core_positions
        }

    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect center-halo expansion pattern.
        
        Note: The colors will swap roles in the output (core expands, ring marks center),
        so we only check that the same two colors are present and the bbox grows correctly.
        """
        # Must be same size
        if input_grid.shape != output_grid.shape:
            return None
        
        # Find input structure
        input_struct = self._find_halo_structure(input_grid)
        if input_struct is None:
            return None
        
        # Verify output has some structure (might have swapped colors)
        output_struct = self._find_halo_structure(output_grid)
        if output_struct is None:
            return None
        
        # Check that output uses the same two colors (in either role)
        input_colors = {input_struct['core_color'], input_struct['ring_color']}
        output_colors = {output_struct['core_color'], output_struct['ring_color']}
        
        if input_colors != output_colors:
            return None
        
        # Output bbox should be larger (expansion amount varies by core size)
        in_bbox = input_struct['bbox']
        out_bbox = output_struct['bbox']
        
        in_h = in_bbox[2] - in_bbox[0]
        in_w = in_bbox[3] - in_bbox[1]
        out_h = out_bbox[2] - out_bbox[0]
        out_w = out_bbox[3] - out_bbox[1]
        
        # Should grow (any positive amount, typically 2-4 cells per dimension)
        if out_h <= in_h or out_w <= in_w:
            return None
        
        # Growth should be even (same on both sides)
        growth_h = out_h - in_h
        growth_w = out_w - in_w
        if growth_h % 2 != 0 or growth_w % 2 != 0:
            return None
        
        # Verify transformation by reconstruction
        # Don't pass colors - apply() will auto-detect from input
        reconstructed = self.apply(input_grid, {'pattern': 'center_halo_expansion'})
        
        if not np.array_equal(reconstructed, output_grid):
            return None
        
        # Return color-agnostic params (pattern type only)
        return {'pattern': 'center_halo_expansion'}

    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        """
        Apply center-halo expansion using ratio-based centering engine.
        
        RATIO THINKING:
        - 3×3 → 5×5: ratio 1.67× (add 2 total, +1 each side)
        - 4×4 → 8×8: ratio 2.0× (doubled)
        - Pattern: RING DILATES OUTWARD, CORE MARKS CENTER
        
        COLOR-AGNOSTIC: Auto-detects ring/core colors from input structure.
        """
        # Step 1: Extract + analyze with new engine
        center_obj = extract_center_object(input_grid, treat_zero_as_background=True)
        color_info = centered_color_roles(input_grid, treat_zero_as_background=True)
        
        if center_obj is None or center_obj.size == 0 or len(color_info['colors']) != 2:
            return input_grid.copy()
        
        # Auto-detect colors from input (majority=ring, minority=core)
        ring_color = color_info['majority_color']
        core_color = color_info['minority_color']
        
        # Step 2: Get core positions
        core_positions = set(map(tuple, np.argwhere(center_obj == core_color).tolist()))
        obj_h, obj_w = center_obj.shape
        
        # Step 3: Calculate expansion (ratio-based)
        if obj_h % 2 == 1:
            out_h = obj_h + 2
            edge_thickness = 1
        else:
            out_h = obj_h * 2
            edge_thickness = 2
        
        if obj_w % 2 == 1:
            out_w = obj_w + 2
        else:
            out_w = obj_w * 2
        
        # Step 4: Generate expanded canvas
        expanded = np.zeros((out_h, out_w), dtype=input_grid.dtype)
        
        for r in range(out_h):
            for c in range(out_w):
                # Corners (stay 0)
                if obj_h % 2 == 0:
                    is_corner = ((r < 2 or r >= out_h - 2) and (c < 2 or c >= out_w - 2))
                else:
                    is_corner = ((r == 0 or r == out_h-1) and (c == 0 or c == out_w-1))
                
                if is_corner:
                    expanded[r, c] = 0
                    continue
                
                # Edge layer (ring color)
                is_edge = (r < edge_thickness or r >= out_h - edge_thickness or
                          c < edge_thickness or c >= out_w - edge_thickness)
                
                if is_edge:
                    expanded[r, c] = ring_color
                else:
                    expanded[r, c] = core_color
        
        # Step 5: Mark center (role inversion)
        center_r = out_h // 2
        center_c = out_w // 2
        
        if len(core_positions) > 1:
            # 2×2 core
            expanded[center_r-1:center_r+1, center_c-1:center_c+1] = ring_color
        else:
            # 1×1 core
            expanded[center_r, center_c] = ring_color
        
        # Step 6: Place back on canvas preserving original offset
        # Growth pattern: N×N → (N+k)×(N+k) by adding k/2 on each side
        # Original top-left should shift by -(growth/2)
        
        nz = np.argwhere(input_grid != 0)
        if len(nz) > 0:
            orig_min_r, orig_min_c = nz.min(axis=0)
        else:
            # Fallback: center on grid
            return make_centered_canvas(expanded, canvas_size=max(input_grid.shape),
                                       pad_value=0, force_odd=False)
        
        # Calculate growth offset (how much to shift original top-left)
        growth_h = out_h - obj_h
        growth_w = out_w - obj_w
        
        # New top-left position (original shifts by growth/2 in each direction)
        place_r = orig_min_r - growth_h // 2
        place_c = orig_min_c - growth_w // 2
        
        # Place expanded object
        result = np.zeros_like(input_grid)
        for r in range(out_h):
            for c in range(out_w):
                target_r = place_r + r
                target_c = place_c + c
                if 0 <= target_r < input_grid.shape[0] and 0 <= target_c < input_grid.shape[1]:
                    result[target_r, target_c] = expanded[r, c]
        
        return result


class HollowFrameExtractionOperator(ArcOperator):
    """
    Hollow Frame Extraction: Detects rectangular frames and fills their interiors.
    
    Pattern:
      - Input: Contains one or more rectangular frames (hollow borders) in frame_color
      - Output: Interior of each frame filled with fill_color, frame itself removed
      - Solid rectangles (no hollow interior) are ignored
    
    Algorithm:
      1. Find connected components of frame_color
      2. For each component, detect if it's a hollow rectangle
      3. Extract the interior coordinates
      4. Fill interior with fill_color in output
    
    Solves: d5d6de2d
    """
    name = "hollow_frame_extraction"
    
    def _find_connected_components(self, grid: np.ndarray, color: int) -> List[np.ndarray]:
        """Find all connected components of a specific color."""
        mask = (grid == color)
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        components = []
        
        def flood_fill(start_r, start_c):
            """Flood fill to find connected component."""
            stack = [(start_r, start_c)]
            coords = []
            
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= H or c < 0 or c >= W:
                    continue
                if visited[r, c] or not mask[r, c]:
                    continue
                    
                visited[r, c] = True
                coords.append((r, c))
                
                # 4-connectivity
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            
            return coords
        
        for r in range(H):
            for c in range(W):
                if mask[r, c] and not visited[r, c]:
                    coords = flood_fill(r, c)
                    if coords:
                        components.append(np.array(coords))
        
        return components
    
    def _is_hollow_rectangle(self, coords: np.ndarray, grid: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Check if coordinates form a hollow rectangle.
        Returns (min_r, max_r, min_c, max_c) of the frame if hollow, None otherwise.
        """
        if len(coords) < 8:  # Minimum for hollow: 3×3 frame = 8 cells
            return None
        
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        # Check if it forms a frame (edges only)
        expected_frame = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                # Add edge cells
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    expected_frame.add((r, c))
        
        actual_coords = set(map(tuple, coords.tolist()))
        
        # Must match frame exactly
        if expected_frame != actual_coords:
            return None
        
        # Must have interior (at least 1×1)
        interior_height = height - 2
        interior_width = width - 2
        
        if interior_height < 1 or interior_width < 1:
            return None
        
        # Check that interior is NOT filled with frame color (must be hollow)
        interior_r_start = min_r + 1
        interior_r_end = max_r
        interior_c_start = min_c + 1
        interior_c_end = max_c
        
        for r in range(interior_r_start, interior_r_end):
            for c in range(interior_c_start, interior_c_end):
                if (r, c) in actual_coords:
                    # Interior has frame color = not hollow
                    return None
        
        return (min_r, max_r, min_c, max_c)
    
    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze if pattern matches: hollow frames → filled interiors.
        """
        # Must be same shape
        if input_grid.shape != output_grid.shape:
            return None
        
        # Find frame color (most common non-zero color in input)
        input_nz = input_grid[input_grid != 0]
        if len(input_nz) == 0:
            return None
        
        colors, counts = np.unique(input_nz, return_counts=True)
        frame_color = int(colors[np.argmax(counts)])
        
        # Find fill color (most common non-zero color in output)
        output_nz = output_grid[output_grid != 0]
        if len(output_nz) == 0:
            return None
        
        fill_colors, fill_counts = np.unique(output_nz, return_counts=True)
        fill_color = int(fill_colors[np.argmax(fill_counts)])
        
        # Find all frames in input
        components = self._find_connected_components(input_grid, frame_color)
        
        if len(components) == 0:
            return None
        
        # Collect all hollow frames
        hollow_frames = []
        for comp in components:
            bounds = self._is_hollow_rectangle(comp, input_grid)
            if bounds is not None:
                hollow_frames.append(bounds)
        
        if len(hollow_frames) == 0:
            return None
        
        # Verify output: interiors filled, frames removed
        expected_output = np.zeros_like(output_grid)
        
        for (min_r, max_r, min_c, max_c) in hollow_frames:
            # Fill interior
            for r in range(min_r + 1, max_r):
                for c in range(min_c + 1, max_c):
                    expected_output[r, c] = fill_color
        
        if not np.array_equal(expected_output, output_grid):
            return None
        
        return {
            "frame_color": frame_color,
            "fill_color": fill_color
        }
    
    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        """
        Apply hollow frame extraction transformation.
        """
        frame_color = params["frame_color"]
        fill_color = params["fill_color"]
        
        # Find all frames
        components = self._find_connected_components(input_grid, frame_color)
        
        # Create output
        output = np.zeros_like(input_grid)
        
        # For each hollow frame, fill its interior
        for comp in components:
            bounds = self._is_hollow_rectangle(comp, input_grid)
            if bounds is not None:
                min_r, max_r, min_c, max_c = bounds
                # Fill interior
                for r in range(min_r + 1, max_r):
                    for c in range(min_c + 1, max_c):
                        output[r, c] = fill_color
        
        return output


# ======================================================================
#  Operator 17: BlockExpansionOperator
# ======================================================================

class BlockExpansionOperator(ArcOperator):
    """
    Position-based block expansion:
    - Each cell in input grid expands to a solid block_size × block_size region
    - Non-zero cell at (i,j) → block at (i*block_size, j*block_size) in output
    - Output size = (H*block_size, W*block_size)
    - Each block is filled with the original cell's color
    
    Examples:
    - 3×3 input → 9×9 output (block_size=3)
    - 3×3 input → 6×6 output (block_size=2)
    - 3×3 input → 15×15 output (block_size=5)
    
    Solves: ac0a08a4
    """
    name = "block_expansion"
    
    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect block expansion pattern.
        Two modes:
        1. Fixed block_size (constant across examples)
        2. Variable block_size (derived from non-zero count)
        """
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape
        
        # Output must be evenly divisible by input dimensions
        if H_out % H_in != 0 or W_out % W_in != 0:
            return None
        
        block_h = H_out // H_in
        block_w = W_out // W_in
        
        # Must be square blocks
        if block_h != block_w:
            return None
        
        block_size = block_h
        
        # Verify: each input cell expands to uniform block
        for i in range(H_in):
            for j in range(W_in):
                input_color = input_grid[i, j]
                
                # Check the corresponding block in output
                r_start = i * block_size
                r_end = (i + 1) * block_size
                c_start = j * block_size
                c_end = (j + 1) * block_size
                
                output_block = output_grid[r_start:r_end, c_start:c_end]
                
                # Block must be uniform with input color
                if not np.all(output_block == input_color):
                    return None
        
        # Check if block_size equals number of non-zero cells (variable mode)
        non_zero_count = np.count_nonzero(input_grid)
        if block_size == non_zero_count:
            return {"rule": "block_size_from_nonzero_count"}
        else:
            return {"rule": "fixed", "block_size": block_size}
    
    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        """
        Apply block expansion transformation.
        Two modes:
        1. Fixed: use provided block_size
        2. Variable: compute block_size from non-zero count
        """
        rule = params.get("rule")
        
        if rule == "block_size_from_nonzero_count":
            block_size = np.count_nonzero(input_grid)
        elif rule == "fixed":
            block_size = params["block_size"]
        else:
            return input_grid.copy()
        
        if block_size == 0:
            return input_grid.copy()
        
        H_in, W_in = input_grid.shape
        
        H_out = H_in * block_size
        W_out = W_in * block_size
        
        output = np.zeros((H_out, W_out), dtype=input_grid.dtype)
        
        # Expand each cell to a block
        for i in range(H_in):
            for j in range(W_in):
                color = input_grid[i, j]
                
                r_start = i * block_size
                r_end = (i + 1) * block_size
                c_start = j * block_size
                c_end = (j + 1) * block_size
                
                output[r_start:r_end, c_start:c_end] = color
        
        return output


# ======================================================================
#  Operator 18: CenterRepulsionOperator
# ======================================================================

class CenterRepulsionOperator(ArcOperator):
    """
    Shape-type classification and band placement:
    - Extracts all shapes and classifies by GEOMETRIC TYPE
    - HOLLOW_SQUARE (8-cell hollow 3x3) → TOP BAND (rows 0-2)
    - THICK shapes (4x4 ring, 3x4 filled) → MIDDLE BAND (rows 3-5)
    - PLUS shapes (5-cell cross, wide cross) → BOTTOM BAND (rows 10-12)
    - Within each band, sorts left-to-right by original column position
    - Rows 6-9 stay empty (intentional gap)
    
    NOT color-based! It's SHAPE GEOMETRY classification.
    
    Solves: ac2e8ecf
    """
    name = "center_repulsion"
    
    def _classify_shape_type(self, grid: np.ndarray) -> str:
        """Classify shape by geometric pattern"""
        h, w = grid.shape
        filled_cells = np.sum(grid > 0)
        
        # HOLLOW_SQUARE: 3x3 with 8 cells (XXX / X0X / XXX)
        if h == 3 and w == 3 and filled_cells == 8:
            pattern = (grid > 0).astype(int)
            hollow_pattern = np.array([[1,1,1], [1,0,1], [1,1,1]])
            if np.array_equal(pattern, hollow_pattern):
                return 'hollow_square'
        
        # PLUS: 3x3 with 5 cells (0X0 / XXX / 0X0)
        if h == 3 and w == 3 and filled_cells == 5:
            pattern = (grid > 0).astype(int)
            plus_pattern = np.array([[0,1,0], [1,1,1], [0,1,0]])
            if np.array_equal(pattern, plus_pattern):
                return 'plus'
        
        # THICK_RING: 4x4 hollow (XXXX / X00X / X00X / XXXX)
        if h == 4 and w == 4 and filled_cells == 12:
            pattern = (grid > 0).astype(int)
            ring_pattern = np.array([[1,1,1,1], [1,0,0,1], [1,0,0,1], [1,1,1,1]])
            if np.array_equal(pattern, ring_pattern):
                return 'thick_ring'
        
        # WIDE shapes (3x4 or 4x3) - check pattern
        if (h == 3 and w == 4) or (h == 4 and w == 3):
            if filled_cells >= 10:
                return 'thick_ring'  # Wide ring/thick shape → middle band
            elif filled_cells >= 6:
                return 'plus'  # Wide plus → bottom band
        
        # Other crosses and tall shapes - default to plus (bottom band)
        if filled_cells >= 6:
            return 'plus'
        
        return 'unknown'
    
    def _extract_shapes(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Extract shapes as locked units - each connected component is one shape"""
        from scipy import ndimage
        
        shapes = []
        colors = set(grid.flatten()) - {0}
        
        for color in colors:
            # Find connected components for this color
            mask = (grid == color)
            labeled, num_features = ndimage.label(mask)
            
            for label_id in range(1, num_features + 1):
                # Get all cells belonging to this shape
                obj_mask = (labeled == label_id)
                positions = np.argwhere(obj_mask)
                
                if len(positions) == 0:
                    continue
                
                # Get bounding box
                min_row, min_col = positions.min(axis=0)
                max_row, max_col = positions.max(axis=0)
                
                # Extract COMPLETE shape grid (includes background 0s within bbox)
                shape_grid = grid[min_row:max_row+1, min_col:max_col+1].copy()
                
                # Classify by geometric type
                shape_type = self._classify_shape_type(shape_grid)
                
                # Center of mass
                center_row = positions[:, 0].mean()
                center_col = positions[:, 1].mean()
                
                shapes.append({
                    'grid': shape_grid,  # Complete grid of shape
                    'bbox': (int(min_row), int(min_col), int(max_row), int(max_col)),
                    'center_row': center_row,
                    'center_col': center_col,
                    'height': int(max_row - min_row + 1),
                    'width': int(max_col - min_col + 1),
                    'color': int(color),
                    'type': shape_type
                })
        
        return shapes
    
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Check if output is center-repulsion version of input"""
        H, W = input_grid.shape
        
        if input_grid.shape != output_grid.shape:
            return None
        
        midpoint = H / 2.0
        
        # Extract shapes from input
        input_shapes = self._extract_shapes(input_grid)
        if len(input_shapes) == 0:
            return None
        
        # Extract shapes from output
        output_shapes = self._extract_shapes(output_grid)
        
        # Must have same number of shapes
        if len(input_shapes) != len(output_shapes):
            return None
        
        # Classify by hemisphere
        upper_in = [s for s in input_shapes if s['center_row'] < midpoint]
        lower_in = [s for s in input_shapes if s['center_row'] >= midpoint]
        
        upper_out = [s for s in output_shapes if s['center_row'] < midpoint]
        lower_out = [s for s in output_shapes if s['center_row'] >= midpoint]
        
        # Check for repulsion (upper shapes moved up, lower moved down)
        if len(upper_out) > 0:
            upper_max_row = max(s['bbox'][2] for s in upper_out)
        else:
            upper_max_row = -1
        
        if len(lower_out) > 0:
            lower_min_row = min(s['bbox'][0] for s in lower_out)
        else:
            lower_min_row = H
        
        # Must have gap
        if upper_max_row >= 0 and lower_min_row < H:
            gap = lower_min_row - upper_max_row - 1
            if gap < 2:
                return None
        
        # Return empty dict - no parameters needed, all logic is in apply
        return {}
    
    def apply(self, input_grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply hollow/filled segregation: hollow shapes pack UP, filled shapes pack DOWN"""
        H, W = input_grid.shape
        output = np.zeros_like(input_grid)
        
        # Extract all shapes and classify by center pixel (hollow vs filled)
        hollow_shapes = []
        filled_shapes = []
        
        for color in range(1, 10):
            mask = (input_grid == color)
            if not mask.any():
                continue
            
            labeled, num = ndimage.label(mask)
            for shape_id in range(1, num + 1):
                shape_mask = (labeled == shape_id)
                rows, cols = np.where(shape_mask)
                
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
                height = max_row - min_row + 1
                
                # Check center pixel to determine shape type
                center_r = (min_row + max_row) // 2
                center_c = (min_col + max_col) // 2
                has_center = input_grid[center_r, center_c] == color
                
                shape_data = {
                    'color': color,
                    'min_row': min_row,
                    'max_row': max_row,
                    'min_col': min_col,
                    'max_col': max_col,
                    'height': height,
                    'pixels': list(zip(rows.tolist(), cols.tolist()))
                }
                
                if has_center:
                    filled_shapes.append(shape_data)
                else:
                    hollow_shapes.append(shape_data)
        
        # Pack HOLLOW shapes upward from row 0 with 2D bin packing
        hollow_shapes.sort(key=lambda s: s['min_row'])
        
        for shape in hollow_shapes:
            # Find earliest row where shape fits (can share rows with non-overlapping columns)
            placed = False
            for try_row in range(H):
                fits = True
                for r_offset in range(shape['height']):
                    row = try_row + r_offset
                    if row >= H:
                        fits = False
                        break
                    for col in range(shape['min_col'], shape['max_col'] + 1):
                        if output[row, col] != 0:
                            fits = False
                            break
                    if not fits:
                        break
                
                if fits:
                    row_offset = try_row - shape['min_row']
                    for r, c in shape['pixels']:
                        output[r + row_offset, c] = shape['color']
                    placed = True
                    break
        
        # Pack FILLED shapes downward from row H-1 with 2D bin packing
        filled_shapes.sort(key=lambda s: s['max_row'], reverse=True)
        
        for shape in filled_shapes:
            placed = False
            for try_row in range(H - 1, -1, -1):
                start_row = try_row - shape['height'] + 1
                if start_row < 0:
                    continue
                
                fits = True
                for r_offset in range(shape['height']):
                    row = start_row + r_offset
                    for col in range(shape['min_col'], shape['max_col'] + 1):
                        if output[row, col] != 0:
                            fits = False
                            break
                    if not fits:
                        break
                
                if fits:
                    row_offset = start_row - shape['min_row']
                    for r, c in shape['pixels']:
                        output[r + row_offset, c] = shape['color']
                    placed = True
                    break
        
        return output


# ======================================================================
#  Operator 19: DiagonalReflectionOperator
# ======================================================================

class DiagonalReflectionOperator(ArcOperator):
    """
    Diagonal reflection/propagation pattern:
    - Finds bottom structure with frame_color + inner_color
    - Inner color propagates upward in expanding diagonal rays
    - Each row up, diagonals expand one step outward (left and right)
    - Creates inverted pyramid/V-shape above the base structure
    
    Pattern:
    - Bottom: frame surrounds center fill color
    - Fill color "reflects" upward in diagonal expansion
    - Stops at grid edges
    
    Solves: b8cdaf2b
    """
    name = "diagonal_reflection"
    
    def analyze(self, input_grid: np.ndarray,
                output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect diagonal reflection pattern.
        """
        if input_grid.shape != output_grid.shape:
            return None
        
        H, W = input_grid.shape
        
        # Find all colors in input (excluding 0)
        all_colors = set(input_grid[input_grid != 0])
        
        if len(all_colors) < 2:
            return None
        
        # Find bottom-most row with content
        bottom_row = None
        for r in range(H - 1, -1, -1):
            if np.any(input_grid[r] != 0):
                bottom_row = r
                break
        
        if bottom_row is None:
            return None
        
        # Get all colors present in input (sorted for consistency)
        colors = sorted(list(all_colors))
        color_counts = [np.sum(input_grid == c) for c in colors]
        
        # Try each color as potential inner_color
        for inner_color in colors:
            expected_output = input_grid.copy()
            
            # Find all positions of inner_color
            inner_positions = np.where(input_grid == inner_color)
            
            if len(inner_positions[0]) == 0:
                continue
            
            # Find center column (average of all inner_color positions)
            center_col = int(np.mean(inner_positions[1]))
            
            # Find BOTTOM-most row with inner_color (base row)
            base_row = np.max(inner_positions[0])
            
            # Find where to start placing (first row above base that should get color)
            # This is the last contiguous all-zero row just above any structure
            start_row = base_row - 1
            # Skip past any structure/obstacles to find first all-zero row
            while start_row >= 0 and not np.all(expected_output[start_row] == 0):
                start_row -= 1
            
            if start_row < 0:
                continue  # No room to place anything
            
            # Starting distance based on width of inner_color at base row
            inner_cols_at_base = inner_positions[1][inner_positions[0] == base_row]
            width = len(inner_cols_at_base)
            distance = (width + 1) // 2
            for r in range(start_row, -1, -1):
                left_col = center_col - distance
                right_col = center_col + distance
                
                if 0 <= left_col < W and expected_output[r, left_col] == 0:
                    expected_output[r, left_col] = inner_color
                
                if 0 <= right_col < W and expected_output[r, right_col] == 0:
                    expected_output[r, right_col] = inner_color
                
                distance += 1
            
            if np.array_equal(expected_output, output_grid):
                return {"inner_color": int(inner_color)}
        
        return None
    
    def apply(self, input_grid: np.ndarray,
              params: Dict[str, Any]) -> np.ndarray:
        """
        Apply diagonal reflection transformation.
        Purely additive - only adds to blank (0) cells.
        """
        output = input_grid.copy()
        H, W = output.shape
        
        # Auto-detect inner_color if not provided
        if "inner_color" not in params or params["inner_color"] is None:
            # Find the least common non-zero color at bottom row
            bottom_row = H - 1
            while bottom_row >= 0 and np.all(input_grid[bottom_row] == 0):
                bottom_row -= 1
            
            if bottom_row < 0:
                return output
            
            colors = set(input_grid[bottom_row]) - {0}
            if not colors:
                return output
            
            # Inner color is at center of bottom row
            center_col = W // 2
            inner_color = input_grid[bottom_row, center_col]
            if inner_color == 0 or inner_color not in colors:
                # Fallback: try colors sorted by value
                inner_color = sorted(list(colors))[0]
        else:
            inner_color = params["inner_color"]
        
        # Find all positions of inner_color
        inner_positions = np.where(output == inner_color)
        
        if len(inner_positions[0]) == 0:
            return output
        
        # Find center column (average of all inner_color positions)
        center_col = int(np.mean(inner_positions[1]))
        
        # Find BOTTOM-most row with inner_color (base row)
        base_row = np.max(inner_positions[0])
        
        # Find where to start placing (first row above base that should get color)
        # This is the last contiguous all-zero row just above any structure
        start_row = base_row - 1
        # Skip past any structure/obstacles to find first all-zero row
        while start_row >= 0 and not np.all(output[start_row] == 0):
            start_row -= 1
        
        if start_row < 0:
            return output  # No room to place anything
        
        # Starting distance based on width of inner_color at base row
        inner_cols_at_base = inner_positions[1][inner_positions[0] == base_row]
        width = len(inner_cols_at_base)
        distance = (width + 1) // 2
        for r in range(start_row, -1, -1):
            left_col = center_col - distance
            right_col = center_col + distance
            
            if 0 <= left_col < W and output[r, left_col] == 0:
                output[r, left_col] = inner_color
            
            if 0 <= right_col < W and output[r, right_col] == 0:
                output[r, right_col] = inner_color
            
            distance += 1
        
        return output


# ======================================================================
#  Operator registry & rule inference
# ======================================================================

OPERATORS: List[ArcOperator] = [
    # Simple / primitive recolor/replicate
    SwapMappingOperator(),
    RecolorMappingOperator(),
    ColorHistogramFillOperator(),
    HorizontalReplicateOperator(),

    # Positional & block-wise patterns
    PositionBasedRecolorOperator(),
    StructuredRecolorOperator(),
    LatticeRoleRecolorOperator(),  # TIER 3 UNLOCK: input-conditioned color semantics
    
    # TIER 3.A: Pattern generators
    LatinSquareFromDiagonalOperator(),  # 05269061: diagonal colors → Latin square tiling
    
    GridSubdivisionOperator(),

    # Structural / geometric
    MirrorOperator(),
    CropToBoundingBoxOperator(),
    LargestBlobExtractOperator(),
    GrowShrinkOperator(),
    
    # Spatial reasoning
    CenterRepulsionOperator(),  # ac2e8ecf: anti-gravity push to edges
    
    # Center-anchor patterns
    DiagonalReflectionOperator(),  # b8cdaf2b: diagonal propagation upward
    CenterHaloExpansionOperator(),  # 3befdf3e: expand anchor+ring with inversion
    HollowFrameExtractionOperator(),  # d5d6de2d: hollow frames → filled interiors
    BlockExpansionOperator(),  # ac0a08a4: sparse grid → block expansion
    SymmetryCompletionOperator(),

    # Tiling families
    TilingExpandOperator(),
    SelfMaskingTilingOperator(),
    SelfReferentialTilingOperator(),
    SandwichTilingOperator(),

    # Marker / directional
    DiagonalMarkerOperator(),
]

# ======================================================================
# Import new operators from separate modules
# ======================================================================
import sys
from pathlib import Path

# Add operators directory to path
operators_dir = Path(__file__).parent / "operators"
if operators_dir.exists():
    # Import new operators
    try:
        sys.path.insert(0, str(operators_dir))
        from enclosed_region_fill import EnclosedRegionFillOperator
        from checkerboard_tiling_alternating import CheckerboardTilingAlternatingOperator
        from symmetry_from_fragment import SymmetryFromFragmentOperator
        from shape_reference_recolor import ShapeReferenceRecolorOperator
        from pattern_completion_prototype import PatternCompletionPrototypeOperator
        
        # Add to operators list
        OPERATORS.extend([
            EnclosedRegionFillOperator(),
            CheckerboardTilingAlternatingOperator(),
            SymmetryFromFragmentOperator(),
            ShapeReferenceRecolorOperator(),
            PatternCompletionPrototypeOperator(),
        ])
        
        print(f"✓ Loaded 5 new operators from {operators_dir}")
    except Exception as e:
        print(f"Warning: Could not load new operators: {e}")
else:
    print(f"Warning: Operators directory not found: {operators_dir}")


def merge_operator_params(op: ArcOperator,
                          params_list: List[Dict[str, Any]]
                          ) -> Optional[Dict[str, Any]]:
    """
    Merge parameters from multiple training pairs for a given operator.

    Operator-specific logic:
      - swap_mapping:
          consider (a, b) equivalent to (b, a)
      - position_based_recolor:
          require consistent (k, m), allow per-example pattern variation;
          store canonical pattern from first example
      - structured_position_recolor:
          same as position_based_recolor (k, m consistency)
      - symmetry_completion:
          require consistent axis, ignore anything else
      - default:
          require dict equality
    """
    if not params_list:
        return None

    name = getattr(op, "name", "")

    # ---- swap_mapping: treat (a, b) and (b, a) as the same swap ----
    if name == "swap_mapping":
        # normalize to sorted pair
        def norm(p):
            a, b = int(p["a"]), int(p["b"])
            if a < b:
                return {"a": a, "b": b}
            else:
                return {"a": b, "b": a}

        base = norm(params_list[0])
        for p in params_list[1:]:
            if norm(p) != base:
                return None
        return base

    # ---- position_based_recolor: same (k, m), pattern may differ ----
    if name == "position_based_recolor":
        base = params_list[0]
        k0, m0 = int(base["k"]), int(base["m"])
        # All examples must share same period
        for p in params_list[1:]:
            if int(p["k"]) != k0 or int(p["m"]) != m0:
                return None
        # We keep the first pattern as canonical.
        # This assumes structure (period) is the key rule, and pattern
        # is allowed to vary between examples.
        return {"k": k0, "m": m0, "pattern": base["pattern"]}

    # ---- structured_position_recolor: same (k, m), allow pattern recolor ----
    if name == "structured_position_recolor":
        # We treat it similarly to position_based_recolor for now:
        # same (k, m), keep first pattern as canonical.
        base = params_list[0]
        k0, m0 = int(base["k"]), int(base["m"])
        for p in params_list[1:]:
            if int(p["k"]) != k0 or int(p["m"]) != m0:
                return None
            # Optional: we *could* check whether each p["pattern"] is a
            # recolor of base["pattern"]. For now, we skip that to keep
            # it simple and broad.
        return {"k": k0, "m": m0, "pattern": base["pattern"]}

    # ---- lattice_role_recolor: unify roles across examples ----
    if name == "lattice_role_recolor":
        # Verify consistent lattice
        base = params_list[0]
        k0, m0 = int(base["k"]), int(base["m"])
        
        for p in params_list[1:]:
            if int(p["k"]) != k0 or int(p["m"]) != m0:
                return None
        
        # Build unified role mapping
        # Use first example as canonical structure
        output_pattern = base["output_pattern"]
        input_pattern = base["input_pattern"]
        
        # Optionally: verify consistency across examples
        # For now, use first example's mapping as canonical
        
        return {
            "k": k0,
            "m": m0,
            "output_pattern": output_pattern,
            "input_pattern": input_pattern,
        }

    # ---- latin_square_from_diagonal: verify consistent color ordering ----
    if name == "latin_square_from_diagonal":
        # All examples should produce the same color ordering
        # (This is deterministic from the generative rule)
        base = params_list[0]
        colors0 = base["colors"]
        output_shape0 = base["output_shape"]
        
        for p in params_list[1:]:
            # Colors must match exactly (same Latin square pattern)
            if p["colors"] != colors0:
                return None
            # Output shape must match
            if p["output_shape"] != output_shape0:
                return None
        
        return {"colors": colors0, "output_shape": output_shape0}
    
    # ---- symmetry_completion: all axes must agree ----
    if name == "symmetry_completion":
        axis0 = params_list[0].get("axis")
        for p in params_list[1:]:
            if p.get("axis") != axis0:
                return None
        return {"axis": axis0}

    # ---- block_expansion: two modes (fixed or variable) ----
    if name == "block_expansion":
        # Check if all examples use the same rule
        rule0 = params_list[0].get("rule")
        
        if rule0 == "block_size_from_nonzero_count":
            # Variable mode: all must use same rule
            for p in params_list[1:]:
                if p.get("rule") != "block_size_from_nonzero_count":
                    return None
            return {"rule": "block_size_from_nonzero_count"}
        
        elif rule0 == "fixed":
            # Fixed mode: all must use same block_size
            block_size0 = params_list[0]["block_size"]
            for p in params_list[1:]:
                if p.get("rule") != "fixed" or p.get("block_size") != block_size0:
                    return None
            return {"rule": "fixed", "block_size": block_size0}
        
        return None

    # ---- diagonal_reflection: per-example inner_color ----
    if name == "diagonal_reflection":
        # Each example has its own inner_color - this is expected
        # Return empty dict to signal operator works but needs per-example params
        return {}

    # ---- default: strict equality ----
    base = params_list[0]
    for p in params_list[1:]:
        if p != base:
            return None
    return base


def infer_rule_for_task(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[Optional[ArcOperator], Optional[Dict[str, Any]]]:
    """
    Try all operators against the training examples.
    Returns:
      (chosen_operator, merged_params)
      or (None, None) if nothing fits.
    """
    for op in OPERATORS:
        params_per_example: List[Dict[str, Any]] = []
        ok = True

        for inp, out in train_pairs:
            params = op.analyze(inp, out)
            if params is None:
                ok = False
                break
            params_per_example.append(params)

        if not ok:
            continue

        merged = merge_operator_params(op, params_per_example)
        if merged is None:
            continue

        # Validate: apply() must produce correct output for all training examples
        valid = True
        for inp, expected_out in train_pairs:
            try:
                result = op.apply(inp, merged)
                if not np.array_equal(result, expected_out):
                    valid = False
                    break
            except:
                valid = False
                break
        
        if not valid:
            continue

        return op, merged

    return None, None


def solve_task(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_inputs: List[np.ndarray],
) -> List[Optional[np.ndarray]]:
    """
    High-level solver for a single ARC task:
      - learn a rule from training pairs using the operator library
      - apply it to each test input
    """
    op, params = infer_rule_for_task(train_pairs)
    if op is None or params is None:
        return [None for _ in test_inputs]

    outputs: List[Optional[np.ndarray]] = []
    for test_inp in test_inputs:
        try:
            out = op.apply(test_inp, params)
        except Exception:
            out = None
        outputs.append(out)
    return outputs
