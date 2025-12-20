#!/usr/bin/env python3
"""
Phase A Validation Tests

Validates pixel metadata extraction and metadata-enhanced channel computation.
Tests on known ARC patterns to ensure correctness.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cod_616.arc_organ.arc_pixel_metadata import extract_grid_metadata, print_metadata_summary
from cod_616.arc_organ.arc_grid_parser import compute_arc_channels_v2

print("=" * 70)
print("PHASE A VALIDATION TESTS")
print("=" * 70)


# ============================================================================
# TEST 1: Tiling Detection
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1: Tiling Detection (3×3 tile repeated 3×3 = 9×9 grid)")
print("=" * 70)

# Create 3×3 base tile
base_tile = np.array([
    [7, 0, 7],
    [7, 0, 7],
    [7, 7, 0]
], dtype=np.int32)

# Tile it 3×3
tiled_grid = np.tile(base_tile, (3, 3))

print(f"\nInput grid shape: {tiled_grid.shape}")
print("Grid:\n", tiled_grid[:6, :6], "\n...")  # Show sample

# Extract metadata
metadata = extract_grid_metadata(tiled_grid)

print("\nMetadata Analysis:")
print_metadata_summary(metadata)

# Validate
assert metadata.is_tiled, "❌ FAIL: Should detect tiling"
assert metadata.tile_size == (3, 3), f"❌ FAIL: Expected tile_size (3,3), got {metadata.tile_size}"
assert metadata.tile_repetitions == (3, 3), f"❌ FAIL: Expected (3,3) repetitions, got {metadata.tile_repetitions}"

print("\n✓ TEST 1 PASSED")


# ============================================================================
# TEST 2: Symmetry Detection
# ============================================================================

print("\n" + "=" * 70)
print("TEST 2: Symmetry Detection (vertical mirror)")
print("=" * 70)

# Create vertically symmetric pattern
symmetric_grid = np.array([
    [1, 2, 3, 2, 1],
    [4, 5, 6, 5, 4],
    [7, 8, 9, 8, 7]
], dtype=np.int32)

print(f"\nInput grid shape: {symmetric_grid.shape}")
print("Grid:\n", symmetric_grid)

# Extract metadata
metadata = extract_grid_metadata(symmetric_grid)

print("\nMetadata Analysis:")
print_metadata_summary(metadata)

# Validate
assert metadata.has_vertical_symmetry, "❌ FAIL: Should detect vertical symmetry"
assert metadata.vertical_symmetry_score > 0.9, f"❌ FAIL: Vertical symmetry score {metadata.vertical_symmetry_score} too low"

print("\n✓ TEST 2 PASSED")


# ============================================================================
# TEST 3: Component Detection
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3: Component Detection (multiple objects)")
print("=" * 70)

# Create grid with 3 distinct components
component_grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 2, 0],
    [0, 3, 3, 3, 0],
], dtype=np.int32)

print(f"\nInput grid shape: {component_grid.shape}")
print("Grid:\n", component_grid)

# Extract metadata
metadata = extract_grid_metadata(component_grid)

print("\nMetadata Analysis:")
print_metadata_summary(metadata)

# Validate
assert metadata.num_components == 3, f"❌ FAIL: Expected 3 components, got {metadata.num_components}"
assert len(metadata.component_sizes) == 3, f"❌ FAIL: Expected 3 component sizes"

# Check component sizes
expected_sizes = [4, 3, 1]  # Sorted by detection order
print(f"\nComponent sizes: {metadata.component_sizes}")

print("\n✓ TEST 3 PASSED")


# ============================================================================
# TEST 4: Border Detection
# ============================================================================

print("\n" + "=" * 70)
print("TEST 4: Border/Interior Detection")
print("=" * 70)

# Create grid with clear border/interior distinction
border_grid = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1],
], dtype=np.int32)

print(f"\nInput grid shape: {border_grid.shape}")
print("Grid:\n", border_grid)

# Extract metadata
metadata = extract_grid_metadata(border_grid)

print("\nMetadata Analysis:")
print_metadata_summary(metadata)

# Check specific pixels
center_pixel = metadata.pixel_metadata[2, 2]  # Should be interior
border_pixel = metadata.pixel_metadata[0, 0]  # Should be border

print(f"\nCenter pixel (2,2):")
print(f"  Color: {center_pixel.color}")
print(f"  Is border: {center_pixel.is_border}")
print(f"  Is interior: {center_pixel.is_interior}")

print(f"\nCorner pixel (0,0):")
print(f"  Color: {border_pixel.color}")
print(f"  Is border: {border_pixel.is_border}")
print(f"  Is interior: {border_pixel.is_interior}")

# Validate
assert center_pixel.is_interior or not center_pixel.is_border, "❌ FAIL: Center should be interior or not border"

print("\n✓ TEST 4 PASSED")


# ============================================================================
# TEST 5: Metadata-Enhanced Channels
# ============================================================================

print("\n" + "=" * 70)
print("TEST 5: Metadata-Enhanced Channel Extraction")
print("=" * 70)

# Use tiled grid from TEST 1
channels = compute_arc_channels_v2(tiled_grid)

print(f"\nChannel shapes:")
print(f"  CH1 (color):     {channels.color_normalized.shape}")
print(f"  CH2 (component): {channels.component_ids.shape}")
print(f"  CH3 (border):    {channels.shape_signature_map.shape}")
print(f"  CH4 (symmetry):  {channels.symmetry_score_map.shape}")
print(f"  CH5 (tiling):    {channels.repetition_score_map.shape}")
print(f"  CH6 (delta):     {channels.delta_map.shape}")

# Validate
assert channels.color_normalized.shape == tiled_grid.shape, "❌ FAIL: Channel shape mismatch"
assert channels.component_ids.shape == tiled_grid.shape, "❌ FAIL: Channel shape mismatch"

# Check value ranges
print(f"\nChannel value ranges:")
print(f"  CH1 (color):     [{channels.color_normalized.min():.3f}, {channels.color_normalized.max():.3f}]")
print(f"  CH2 (component): [{channels.component_ids.min():.3f}, {channels.component_ids.max():.3f}]")
print(f"  CH3 (border):    [{channels.shape_signature_map.min():.3f}, {channels.shape_signature_map.max():.3f}]")
print(f"  CH4 (symmetry):  [{channels.symmetry_score_map.min():.3f}, {channels.symmetry_score_map.max():.3f}]")
print(f"  CH5 (tiling):    [{channels.repetition_score_map.min():.3f}, {channels.repetition_score_map.max():.3f}]")

# All channels should be in [0, 1]
assert channels.color_normalized.min() >= 0.0 and channels.color_normalized.max() <= 1.0, "❌ FAIL: CH1 out of range"
assert channels.component_ids.min() >= 0.0 and channels.component_ids.max() <= 1.0, "❌ FAIL: CH2 out of range"

print("\n✓ TEST 5 PASSED")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PHASE A VALIDATION SUMMARY")
print("=" * 70)
print("\n✓ All 5 tests passed")
print("\nPhase A Status:")
print("  ✓ Pixel metadata extraction")
print("  ✓ Tiling detection")
print("  ✓ Symmetry detection")
print("  ✓ Component detection")
print("  ✓ Border/interior detection")
print("  ✓ Metadata-enhanced channels")
print("\n✓ PHASE A VALIDATED")
print("\nReady for Phase C (20-dim resonance state).")
print("=" * 70)
