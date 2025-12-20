"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     CompuCog — Sovereign Cognitive Defense System                           ║
║     Intellectual Property of Cortex Evolved / L.A. Mercey                   ║
║                                                                              ║
║     Copyright © 2025 Cortex Evolved. All Rights Reserved.                   ║
║                                                                              ║
║     "We use unconventional digital wisdom —                                  ║
║        because conventional digital wisdom doesn't protect anyone."         ║
║                                                                              ║
║     This software is proprietary and confidential.                           ║
║     Unauthorized access, copying, modification, or distribution             ║
║     is strictly prohibited and may violate applicable laws.                  ║
║                                                                              ║
║     File automatically watermarked on: 2025-11-29 19:21:12                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""

"""
CompuCog Visual Sensor — Frame-Grid Converter (Module 1)

Purpose:
  Convert raw frames into ARC-style grids for symbolic reasoning.

Responsibilities:
  1. Capture frames from screen or game window
  2. Downsample to small grid (32×32 default)
  3. Quantize colors to palette (0-9 like ARC)
  4. Output FrameGrid objects with metadata
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np

try:
    import mss
    HAVE_MSS = True
except ImportError:
    HAVE_MSS = False
    print("[WARN] mss not installed; using stub FrameCapture for tests")

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    print("[WARN] PIL not installed; using stub image for tests")


@dataclass
class FrameGrid:
    """ARC-style grid representation of a captured frame"""
    frame_id: int
    t_sec: float
    grid: List[List[int]]  # H×W, values 0-9 (or configured palette size)
    source: str
    capture_region: str
    h: int
    w: int


class FrameCapture:
    """
    Captures raw frames from screen using mss (fast screen capture).
    Provides a graceful stub when `mss` is unavailable so test-imports don't fail.
    """

    def __init__(self, config: dict):
        self.config = config
        self.source = config.get("capture", {}).get("source", "auto")
        self.region_type = config.get("capture", {}).get("region", "full")

        if HAVE_MSS:
            self.sct = mss.mss()
            self.monitor = self._get_monitor()
            print(f"[+] FrameCapture initialized")
            print(f"    Source: {self.source}")
            print(f"    Region: {self.region_type}")
            print(f"    Monitor: {self.monitor['width']}×{self.monitor['height']}")
        else:
            # Minimal stub for tests; capture() will return None
            self.sct = None
            self.monitor = {"left": 0, "top": 0, "width": 640, "height": 480}
            print("[!] FrameCapture running in STUB mode (mss unavailable)")

    def _get_monitor(self) -> dict:
        """Get monitor/region configuration for capture"""
        # For now, use primary monitor
        # Future: support window-specific capture by process name
        monitor = self.sct.monitors[1]  # Primary monitor

        if self.region_type == "full":
            return monitor
        elif self.region_type == "center_720p":
            # Capture 1280×720 from center
            center_x = monitor["left"] + monitor["width"] // 2
            center_y = monitor["top"] + monitor["height"] // 2
            return {
                "left": center_x - 640,
                "top": center_y - 360,
                "width": 1280,
                "height": 720
            }
        else:
            # Default to full
            return monitor

    def capture(self) -> Optional[Image.Image]:
        """Capture current frame from configured source"""
        if not HAVE_MSS:
            return None

        try:
            # Capture screen
            sct_img = self.sct.grab(self.monitor)

            # Convert to PIL Image (RGB)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            return img
        except Exception as e:
            print(f"[ERROR] Frame capture failed: {e}")
            return None


class FrameToGrid:
    """
    Converts raw frame to ARC-style grid.
    
    Process:
      1. Downsample to grid dimensions (e.g., 32×32)
      2. Convert to grayscale
      3. Quantize to palette (0-9)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.grid_h = config.get("grid_h", 32)
        self.grid_w = config.get("grid_w", 32)
        self.palette_size = config.get("palette_size", 10)
        
        print(f"[+] FrameToGrid initialized")
        print(f"    Grid: {self.grid_h}×{self.grid_w}")
        print(f"    Palette: 0-{self.palette_size - 1}")
    
    def convert(self, frame: Image.Image, frame_id: int, t_sec: float, source: str) -> FrameGrid:
        """Convert PIL Image to ARC-style grid"""
        # Downsample
        downsampled = self._downsample(frame)
        
        # Quantize
        grid = self._quantize(downsampled)
        
        return FrameGrid(
            frame_id=frame_id,
            t_sec=t_sec,
            grid=grid,
            source=source,
            capture_region=self.config.get("capture", {}).get("region", "full"),
            h=self.grid_h,
            w=self.grid_w
        )
    
    def _downsample(self, frame: Image.Image) -> Image.Image:
        """Resize frame to grid dimensions"""
        return frame.resize((self.grid_w, self.grid_h), Image.Resampling.LANCZOS)
    
    def _quantize(self, frame: Image.Image) -> List[List[int]]:
        """
        Quantize pixel values to palette.
        
        Process:
          1. Convert to grayscale
          2. Map 0-255 intensity to 0-(palette_size-1)
          3. Return 2D grid
        """
        # Convert to grayscale
        gray = frame.convert("L")
        
        # Convert to numpy array
        arr = np.array(gray)
        
        # Quantize: intensity // bin_size
        bin_size = 256 // self.palette_size
        quantized = arr // bin_size
        
        # Clamp to palette range
        quantized = np.clip(quantized, 0, self.palette_size - 1)
        
        # Convert to list of lists
        grid = quantized.tolist()
        
        return grid


# Simple test harness
if __name__ == "__main__":
    import time
    import json
    
    # Test config
    config = {
        "capture": {
            "source": "auto",
            "region": "full"
        },
        "grid_h": 32,
        "grid_w": 32,
        "palette_size": 10
    }
    
    print("[*] Testing Frame-Grid Converter...")
    
    capturer = FrameCapture(config)
    converter = FrameToGrid(config)
    
    # Capture 5 frames
    for i in range(5):
        frame = capturer.capture()
        if frame is None:
            print(f"[WARN] Frame {i} capture failed")
            continue
        
        t_sec = time.time()
        grid = converter.convert(frame, i, t_sec, "Test")
        
        print(f"[{i}] Captured {grid.h}×{grid.w} grid, palette range: {min(min(row) for row in grid.grid)}-{max(max(row) for row in grid.grid)}")
        
        time.sleep(0.5)
    
    print("[+] Test complete")
