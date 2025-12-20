"""
616 Screen Grid Mapper + Vision Resonance
CompuCog Multimodal Game Intelligence Engine

Captures screen, divides into 100×100 grid, compresses to 10×10 blocks.
Extracts 20-dimensional visual resonance features for 616 fusion.

CompuCogVision Phase 1 - The True Eye.

Built: November 25, 2025
Updated: November 26, 2025 - YOLO Purged, Resonance Integrated
"""

import sys
from pathlib import Path

# Add parent directory to path for screen_resonance_state import
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import mss
import time
from typing import Tuple, Dict, List, Optional

from screen_resonance_state import ScreenResonanceState


class ScreenGridMapper:
    """
    Real-time screen capture and grid-based feature extraction.
    
    Divides screen into NxM grid, tracks per-cell changes,
    compresses to blocks for 616 resonance analysis.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        block_size: Tuple[int, int] = (10, 10),
        monitor: int = 0,
        capture_fps: int = 60,
        capture_resolution: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            grid_size: (rows, cols) - fine grid resolution (e.g., 100×100)
            block_size: (rows, cols) - compressed block size (e.g., 10×10)
            monitor: Monitor index (0 = primary)
            capture_fps: Target capture rate
            capture_resolution: Optional (width, height) to downscale to (None = native)
        """
        self.grid_size = grid_size
        self.block_size = block_size
        self.monitor = monitor
        self.capture_fps = capture_fps
        self.frame_delay = 1.0 / capture_fps
        self.capture_resolution = capture_resolution
        self.native_resolution = None  # Detected on first capture
        
        # MSS screen capture
        self.sct = mss.mss()
        self.monitor_config = self.sct.monitors[monitor + 1]  # 0 = all monitors
        
        # Grid configuration
        self.grid_rows, self.grid_cols = grid_size
        self.block_rows, self.block_cols = block_size
        self.cells_per_block_row = self.grid_rows // self.block_rows
        self.cells_per_block_col = self.grid_cols // self.block_cols
        
        # Screen dimensions
        self.screen_width = self.monitor_config["width"]
        self.screen_height = self.monitor_config["height"]
        
        # Cell dimensions (pixels per cell)
        self.cell_width = self.screen_width / self.grid_cols
        self.cell_height = self.screen_height / self.grid_rows
        
        # Previous frame for delta calculation
        self.prev_grid = None
        self.prev_frame_time = None
        
        # CompuCogVision: Resonance State (20-dim visual feature extractor)
        self.resonance = ScreenResonanceState(
            grid_size=self.block_rows,  # Operates on 10×10 blocks
            ema_alpha_fast=0.3,
            ema_alpha_slow=0.1
        )
        
        # Statistics
        self.frames_captured = 0
        self.total_capture_time = 0.0
        
        print(f"[616 Screen Grid Mapper + Vision Resonance]")
        print(f"  Screen: {self.screen_width}×{self.screen_height}")
        print(f"  Grid: {self.grid_rows}×{self.grid_cols} ({self.grid_rows * self.grid_cols} cells)")
        print(f"  Blocks: {self.block_rows}×{self.block_cols} ({self.block_rows * self.block_cols} blocks)")
        print(f"  Cell size: {self.cell_width:.1f}×{self.cell_height:.1f} pixels")
        print(f"  Target FPS: {self.capture_fps}")
        print(f"  Vision Resonance: 20-dim features (YOLO-free, pure math)")
    
    def capture_frame(self) -> np.ndarray:
        """
        Capture current screen frame.
        
        Returns:
            np.ndarray: RGB image (H, W, 3)
        """
        screenshot = self.sct.grab(self.monitor_config)
        frame = np.array(screenshot)[:, :, :3]  # Drop alpha channel (BGRA → BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB
        
        # Detect native resolution on first capture
        if self.native_resolution is None:
            self.native_resolution = (frame.shape[1], frame.shape[0])  # (width, height)
        
        # Downscale if target resolution specified
        if self.capture_resolution is not None:
            frame = cv2.resize(frame, self.capture_resolution, interpolation=cv2.INTER_AREA)
        
        return frame
    
    def frame_to_grid(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to grid of average pixel intensities.
        
        Args:
            frame: RGB image (H, W, 3)
        
        Returns:
            np.ndarray: Grid of average intensities (grid_rows, grid_cols)
        """
        # Convert to grayscale for simplicity
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Divide into grid cells and compute mean intensity
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y1 = int(i * self.cell_height)
                y2 = int((i + 1) * self.cell_height)
                x1 = int(j * self.cell_width)
                x2 = int((j + 1) * self.cell_width)
                
                cell = gray[y1:y2, x1:x2]
                grid[i, j] = cell.mean()
        
        return grid
    
    def compute_delta(self, current_grid: np.ndarray) -> np.ndarray:
        """
        Compute frame-to-frame change (delta).
        
        Args:
            current_grid: Current frame grid
        
        Returns:
            np.ndarray: Delta grid (absolute difference)
        """
        if self.prev_grid is None:
            return np.zeros_like(current_grid)
        
        delta = np.abs(current_grid - self.prev_grid)
        return delta
    
    def compress_to_blocks(self, grid: np.ndarray) -> np.ndarray:
        """
        Compress fine grid to coarser blocks.
        
        Args:
            grid: Fine grid (grid_rows, grid_cols)
        
        Returns:
            np.ndarray: Compressed blocks (block_rows, block_cols)
        """
        blocks = np.zeros((self.block_rows, self.block_cols), dtype=np.float32)
        
        for i in range(self.block_rows):
            for j in range(self.block_cols):
                i_start = i * self.cells_per_block_row
                i_end = (i + 1) * self.cells_per_block_row
                j_start = j * self.cells_per_block_col
                j_end = (j + 1) * self.cells_per_block_col
                
                block_region = grid[i_start:i_end, j_start:j_end]
                blocks[i, j] = block_region.mean()
        
        return blocks
    
    def extract_features(self) -> Dict[str, np.ndarray]:
        """
        Capture frame and extract grid-based features + 20-dim vision resonance.
        
        Returns:
            Dict with:
                - 'frame': Raw frame (H, W, 3)
                - 'grid': Fine grid (grid_rows, grid_cols)
                - 'delta': Frame delta (grid_rows, grid_cols)
                - 'blocks': Compressed blocks (block_rows, block_cols)
                - 'block_deltas': Compressed delta blocks (block_rows, block_cols)
                - 'block_vector': Flattened feature vector (block_rows * block_cols,)
                - 'visual_resonance': Dict with 20 visual features (CompuCogVision)
                - 'timestamp': Capture timestamp
                - 'fps': Actual FPS
                - 'frame_number': Sequential frame counter
        """
        start_time = time.perf_counter()
        
        # Capture frame
        frame = self.capture_frame()
        
        # Convert to grid
        grid = self.frame_to_grid(frame)
        
        # Compute delta
        delta = self.compute_delta(grid)
        
        # Compress to blocks
        blocks = self.compress_to_blocks(grid)
        block_deltas = self.compress_to_blocks(delta)
        
        # Flatten to feature vector
        block_vector = block_deltas.flatten()
        
        # === CompuCogVision: Extract 20-dimensional resonance features ===
        # Normalize blocks to 0-1 range for resonance state
        blocks_normalized = (blocks - blocks.min()) / (blocks.max() - blocks.min() + 1e-8)
        visual_resonance = self.resonance.update(blocks_normalized)
        
        # Compute FPS
        current_time = time.perf_counter()
        if self.prev_frame_time is not None:
            fps = 1.0 / (current_time - self.prev_frame_time)
        else:
            fps = 0.0
        
        # Update state
        self.prev_grid = grid
        self.prev_frame_time = current_time
        self.frames_captured += 1
        self.total_capture_time += (current_time - start_time)
        
        return {
            'frame': frame,
            'grid': grid,
            'delta': delta,
            'blocks': blocks,
            'block_deltas': block_deltas,
            'block_vector': block_vector,  # 100 features (10×10 blocks)
            'visual_resonance': visual_resonance,  # 20 features (CompuCogVision Phase 1)
            'timestamp': current_time,
            'fps': fps,
            'frame_number': self.frames_captured
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get capture statistics.
        
        Returns:
            Dict with frame count, average FPS, average capture time
        """
        if self.frames_captured == 0:
            return {
                'frames_captured': 0,
                'avg_fps': 0.0,
                'avg_capture_time_ms': 0.0
            }
        
        avg_capture_time = self.total_capture_time / self.frames_captured
        avg_fps = 1.0 / avg_capture_time if avg_capture_time > 0 else 0.0
        
        return {
            'frames_captured': self.frames_captured,
            'avg_fps': avg_fps,
            'avg_capture_time_ms': avg_capture_time * 1000
        }
    
    def visualize_grid(self, features: Dict[str, np.ndarray], mode: str = 'delta') -> np.ndarray:
        """
        Visualize grid overlay on frame.
        
        Args:
            features: Features from extract_features()
            mode: 'delta' or 'blocks'
        
        Returns:
            np.ndarray: Frame with grid overlay
        """
        frame = features['frame'].copy()
        
        if mode == 'delta':
            data = features['delta']
            rows, cols = self.grid_rows, self.grid_cols
            cell_h, cell_w = self.cell_height, self.cell_width
        elif mode == 'blocks':
            data = features['block_deltas']
            rows, cols = self.block_rows, self.block_cols
            cell_h = self.screen_height / rows
            cell_w = self.screen_width / cols
        else:
            return frame
        
        # Normalize data for visualization
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-6)
        
        # Overlay grid
        for i in range(rows):
            for j in range(cols):
                intensity = data_norm[i, j]
                color = int(255 * intensity)
                
                y1 = int(i * cell_h)
                y2 = int((i + 1) * cell_h)
                x1 = int(j * cell_w)
                x2 = int((j + 1) * cell_w)
                
                # Draw semi-transparent rectangle
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (color, 0, 255 - color), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return frame
    
    def close(self):
        """Release resources."""
        self.sct.close()


if __name__ == "__main__":
    """Test screen grid mapper."""
    
    print("Testing 616 Screen Grid Mapper...")
    print("Press 'q' to quit\n")
    
    # Initialize mapper
    mapper = ScreenGridMapper(
        grid_size=(100, 100),
        block_size=(10, 10),
        monitor=0,
        capture_fps=60
    )
    
    try:
        while True:
            # Extract features
            features = mapper.extract_features()
            
            # Print stats every 60 frames
            if mapper.frames_captured % 60 == 0:
                stats = mapper.get_statistics()
                print(f"[Frame {stats['frames_captured']:>5}] "
                      f"FPS: {features['fps']:>6.1f} | "
                      f"Avg: {stats['avg_fps']:>6.1f} FPS | "
                      f"Capture: {stats['avg_capture_time_ms']:>5.2f}ms | "
                      f"Block sum: {features['block_vector'].sum():>8.1f}")
            
            # Visualize (optional - slows down capture)
            # vis_frame = mapper.visualize_grid(features, mode='blocks')
            # cv2.imshow('616 Screen Grid', vis_frame)
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # Rate limiting
            time.sleep(max(0, mapper.frame_delay - 0.001))
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        stats = mapper.get_statistics()
        print(f"\n[Final Stats]")
        print(f"  Frames captured: {stats['frames_captured']}")
        print(f"  Average FPS: {stats['avg_fps']:.1f}")
        print(f"  Average capture time: {stats['avg_capture_time_ms']:.2f}ms")
        
        mapper.close()
        cv2.destroyAllWindows()
