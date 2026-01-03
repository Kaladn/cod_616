"""
Screen Vector Engine — Native Resolution GPU Processing

NO 32×32 DOWNSAMPLING. NATIVE GPU RESOLUTION ONLY.

This module processes frames at their original capture resolution using GPU
acceleration via CuPy. It extracts geometric representations without any
lossy downsampling that destroys edge/pixel information.

Architecture:
    GPU Frame Buffer → Screen Vector Engine → ScreenVector (geometric abstraction)
    
    ScreenVector contains:
    - Core block metrics (center region analysis at native res)
    - Sector analysis (8 directional sectors at native res)
    - Directional gradients (Sobel edge detection at native res)
    - Motion vectors (frame-to-frame optical flow approximation)
    - Anomaly metrics (entropy, symmetry, stability at native res)

Performance Targets:
    - 1920×1080 @ 60 FPS = 16.67ms/frame budget
    - GPU memory: ~50MB working set for dual-frame analysis
    - Latency: <10ms for full ScreenVector extraction
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
import time

try:
    import cupy as cp
    # Test if CUDA actually works - this triggers full initialization
    _test = cp.zeros(1, dtype=cp.float32)
    _test = _test + 1  # Force kernel compilation
    del _test
    GPU_AVAILABLE = True
except Exception as e:
    cp = None
    GPU_AVAILABLE = False
    print(f"[WARN] GPU not available ({type(e).__name__}), using NumPy (CPU)")

try:
    import mss
except ImportError:
    mss = None
    print("[ERROR] mss not installed. Run: pip install mss")


@dataclass
class ScreenVector:
    """
    Geometric abstraction of a native-resolution frame.
    
    This is NOT a downsampled grid. This is a structured representation
    of the full-resolution frame's geometric properties.
    """
    timestamp: float
    frame_id: int
    resolution: Tuple[int, int]  # (height, width) - NATIVE, not 32×32
    
    # Core block (center 20% of screen - primary focus area)
    core_intensity_mean: float
    core_intensity_std: float
    core_gradient_magnitude: float
    core_dominant_direction: float  # radians, 0 = right, pi/2 = down
    core_entropy: float
    
    # 8 Directional sectors (N, NE, E, SE, S, SW, W, NW)
    sector_intensities: List[float] = field(default_factory=list)  # 8 values
    sector_gradients: List[float] = field(default_factory=list)    # 8 values
    sector_entropies: List[float] = field(default_factory=list)    # 8 values
    
    # Motion (requires previous frame)
    motion_magnitude: float = 0.0
    motion_direction: float = 0.0  # radians
    motion_uniformity: float = 0.0  # 0 = chaotic, 1 = uniform motion
    
    # Anomaly metrics
    symmetry_score: float = 0.0  # 0 = asymmetric, 1 = perfectly symmetric
    stability_score: float = 0.0  # vs previous frame, 0 = chaotic, 1 = static
    flash_intensity: float = 0.0  # sudden brightness spike detection
    
    # Raw stats for operators
    histogram: List[int] = field(default_factory=list)  # 256 bins
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "resolution": self.resolution,
            "core": {
                "intensity_mean": self.core_intensity_mean,
                "intensity_std": self.core_intensity_std,
                "gradient_magnitude": self.core_gradient_magnitude,
                "dominant_direction": self.core_dominant_direction,
                "entropy": self.core_entropy,
            },
            "sectors": {
                "intensities": self.sector_intensities,
                "gradients": self.sector_gradients,
                "entropies": self.sector_entropies,
            },
            "motion": {
                "magnitude": self.motion_magnitude,
                "direction": self.motion_direction,
                "uniformity": self.motion_uniformity,
            },
            "anomaly": {
                "symmetry": self.symmetry_score,
                "stability": self.stability_score,
                "flash": self.flash_intensity,
            },
        }


class NativeFrameCapture:
    """
    Captures frames at NATIVE GPU resolution.
    NO DOWNSAMPLING. Full resolution preserved.
    """
    
    def __init__(self, monitor_index: int = 1):
        if mss is None:
            raise ImportError("mss required for screen capture")
        
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]
        self.width = self.monitor["width"]
        self.height = self.monitor["height"]
        
        print(f"[SVE] NativeFrameCapture initialized")
        print(f"      Resolution: {self.width}×{self.height} (NATIVE)")
        print(f"      Pixels per frame: {self.width * self.height:,}")
        print(f"      Memory per frame: {self.width * self.height * 3 / 1024 / 1024:.1f} MB (RGB)")
    
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture frame at native resolution.
        Returns: numpy array (H, W, 3) RGB uint8 at FULL RESOLUTION
        """
        try:
            sct_img = self.sct.grab(self.monitor)
            # Convert BGRA to RGB numpy array - NO RESIZE
            frame = np.frombuffer(sct_img.raw, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 4))
            frame = frame[:, :, :3]  # Drop alpha, keep BGR
            frame = frame[:, :, ::-1]  # BGR to RGB
            return frame.copy()  # Contiguous array for GPU transfer
        except Exception as e:
            print(f"[SVE] Capture failed: {e}")
            return None
    
    def capture_to_gpu(self) -> Optional["cp.ndarray"]:
        """Capture directly to GPU memory."""
        frame = self.capture()
        if frame is None:
            return None
        if GPU_AVAILABLE:
            return cp.asarray(frame)
        return frame


class ScreenVectorEngine:
    """
    Extracts ScreenVector from native resolution frames using GPU.
    
    NO 32×32. NO DOWNSAMPLING. FULL RESOLUTION ANALYSIS.
    """
    
    def __init__(self):
        self.xp = cp if GPU_AVAILABLE else np  # Array library (GPU or CPU)
        self.frame_counter = 0
        self.previous_frame = None
        self.previous_gray = None
        
        # Precompute Sobel kernels on GPU
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        if GPU_AVAILABLE:
            self._sobel_x = cp.asarray(sobel_x)
            self._sobel_y = cp.asarray(sobel_y)
        else:
            self._sobel_x = sobel_x
            self._sobel_y = sobel_y
        
        print(f"[SVE] ScreenVectorEngine initialized")
        print(f"      Backend: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (NumPy)'}")
    
    def process(self, frame: np.ndarray, timestamp: float = None) -> ScreenVector:
        """
        Process native resolution frame into ScreenVector.
        
        Args:
            frame: (H, W, 3) RGB uint8 at NATIVE RESOLUTION
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            ScreenVector with geometric analysis at full resolution
        """
        if timestamp is None:
            timestamp = time.time()
        
        h, w = frame.shape[:2]
        
        # Transfer to GPU if available
        if GPU_AVAILABLE and not isinstance(frame, cp.ndarray):
            gpu_frame = cp.asarray(frame)
        else:
            gpu_frame = frame
        
        # Convert to grayscale on GPU: Y = 0.299*R + 0.587*G + 0.114*B
        xp = self.xp
        gray = (0.299 * gpu_frame[:, :, 0].astype(xp.float32) +
                0.587 * gpu_frame[:, :, 1].astype(xp.float32) +
                0.114 * gpu_frame[:, :, 2].astype(xp.float32))
        
        # Core block analysis (center 20% of screen)
        core_metrics = self._analyze_core(gray, h, w)
        
        # Sector analysis (8 directional sectors)
        sector_metrics = self._analyze_sectors(gray, h, w)
        
        # Motion analysis (requires previous frame)
        motion_metrics = self._analyze_motion(gray)
        
        # Anomaly detection
        anomaly_metrics = self._analyze_anomalies(gray, gpu_frame)
        
        # Histogram (full frame)
        histogram = self._compute_histogram(gray)
        
        # Update state for next frame
        self.previous_frame = gpu_frame
        self.previous_gray = gray
        self.frame_counter += 1
        
        return ScreenVector(
            timestamp=timestamp,
            frame_id=self.frame_counter,
            resolution=(h, w),
            core_intensity_mean=core_metrics["intensity_mean"],
            core_intensity_std=core_metrics["intensity_std"],
            core_gradient_magnitude=core_metrics["gradient_magnitude"],
            core_dominant_direction=core_metrics["dominant_direction"],
            core_entropy=core_metrics["entropy"],
            sector_intensities=sector_metrics["intensities"],
            sector_gradients=sector_metrics["gradients"],
            sector_entropies=sector_metrics["entropies"],
            motion_magnitude=motion_metrics["magnitude"],
            motion_direction=motion_metrics["direction"],
            motion_uniformity=motion_metrics["uniformity"],
            symmetry_score=anomaly_metrics["symmetry"],
            stability_score=anomaly_metrics["stability"],
            flash_intensity=anomaly_metrics["flash"],
            histogram=histogram,
        )
    
    def _analyze_core(self, gray: "cp.ndarray", h: int, w: int) -> Dict:
        """Analyze center 20% of screen at native resolution."""
        xp = self.xp
        
        # Core region bounds (center 20%)
        margin_y = int(h * 0.4)
        margin_x = int(w * 0.4)
        core = gray[margin_y:h-margin_y, margin_x:w-margin_x]
        
        # Intensity statistics
        intensity_mean = float(xp.mean(core))
        intensity_std = float(xp.std(core))
        
        # Gradient via Sobel (native resolution)
        grad_x = self._convolve2d(core, self._sobel_x)
        grad_y = self._convolve2d(core, self._sobel_y)
        gradient_magnitude = float(xp.mean(xp.sqrt(grad_x**2 + grad_y**2)))
        
        # Dominant gradient direction
        dominant_direction = float(xp.arctan2(xp.mean(grad_y), xp.mean(grad_x)))
        
        # Entropy (information content)
        entropy = self._compute_entropy(core)
        
        return {
            "intensity_mean": intensity_mean,
            "intensity_std": intensity_std,
            "gradient_magnitude": gradient_magnitude,
            "dominant_direction": dominant_direction,
            "entropy": entropy,
        }
    
    def _analyze_sectors(self, gray: "cp.ndarray", h: int, w: int) -> Dict:
        """Analyze 8 directional sectors at native resolution."""
        xp = self.xp
        
        # Divide frame into 3×3 grid, use 8 outer cells as sectors
        # Order: N, NE, E, SE, S, SW, W, NW
        h3, w3 = h // 3, w // 3
        
        sector_coords = [
            (0, h3, w3, 2*w3),       # N (top center)
            (0, h3, 2*w3, w),        # NE (top right)
            (h3, 2*h3, 2*w3, w),     # E (middle right)
            (2*h3, h, 2*w3, w),      # SE (bottom right)
            (2*h3, h, w3, 2*w3),     # S (bottom center)
            (2*h3, h, 0, w3),        # SW (bottom left)
            (h3, 2*h3, 0, w3),       # W (middle left)
            (0, h3, 0, w3),          # NW (top left)
        ]
        
        intensities = []
        gradients = []
        entropies = []
        
        for y1, y2, x1, x2 in sector_coords:
            sector = gray[y1:y2, x1:x2]
            
            intensities.append(float(xp.mean(sector)))
            
            grad_x = self._convolve2d(sector, self._sobel_x)
            grad_y = self._convolve2d(sector, self._sobel_y)
            gradients.append(float(xp.mean(xp.sqrt(grad_x**2 + grad_y**2))))
            
            entropies.append(self._compute_entropy(sector))
        
        return {
            "intensities": intensities,
            "gradients": gradients,
            "entropies": entropies,
        }
    
    def _analyze_motion(self, gray: "cp.ndarray") -> Dict:
        """Analyze motion between current and previous frame."""
        xp = self.xp
        
        if self.previous_gray is None:
            return {"magnitude": 0.0, "direction": 0.0, "uniformity": 0.0}
        
        # Frame difference (absolute)
        diff = xp.abs(gray - self.previous_gray)
        
        # Motion magnitude (average pixel change)
        magnitude = float(xp.mean(diff))
        
        # Motion direction via gradient of difference
        grad_x = self._convolve2d(diff, self._sobel_x)
        grad_y = self._convolve2d(diff, self._sobel_y)
        direction = float(xp.arctan2(xp.mean(grad_y), xp.mean(grad_x)))
        
        # Motion uniformity (std of diff normalized)
        diff_std = float(xp.std(diff))
        uniformity = 1.0 - min(1.0, diff_std / (magnitude + 1e-6))
        
        return {
            "magnitude": magnitude,
            "direction": direction,
            "uniformity": uniformity,
        }
    
    def _analyze_anomalies(self, gray: "cp.ndarray", rgb: "cp.ndarray") -> Dict:
        """Detect anomalies: symmetry, stability, flash."""
        xp = self.xp
        h, w = gray.shape
        
        # Horizontal symmetry (left vs right half)
        left = gray[:, :w//2]
        right = xp.flip(gray[:, w//2:w//2*2], axis=1)
        if left.shape == right.shape:
            symmetry = 1.0 - float(xp.mean(xp.abs(left - right)) / 255.0)
        else:
            symmetry = 0.5
        
        # Stability (vs previous frame)
        if self.previous_gray is not None:
            diff = xp.abs(gray - self.previous_gray)
            stability = 1.0 - float(xp.mean(diff) / 255.0)
        else:
            stability = 1.0
        
        # Flash detection (sudden brightness spike)
        current_brightness = float(xp.mean(gray))
        if self.previous_gray is not None:
            prev_brightness = float(xp.mean(self.previous_gray))
            flash = abs(current_brightness - prev_brightness) / 255.0
        else:
            flash = 0.0
        
        return {
            "symmetry": symmetry,
            "stability": stability,
            "flash": flash,
        }
    
    def _compute_histogram(self, gray: "cp.ndarray") -> List[int]:
        """Compute 256-bin histogram of grayscale frame."""
        xp = self.xp
        hist, _ = xp.histogram(gray.astype(xp.uint8), bins=256, range=(0, 256))
        if GPU_AVAILABLE:
            return hist.get().tolist()
        return hist.tolist()
    
    def _compute_entropy(self, region: "cp.ndarray") -> float:
        """Compute Shannon entropy of region."""
        xp = self.xp
        
        # Quantize to 64 bins for faster entropy calc
        quantized = (region / 4).astype(xp.uint8)
        hist, _ = xp.histogram(quantized, bins=64, range=(0, 64))
        
        # Normalize to probability
        total = xp.sum(hist)
        if total == 0:
            return 0.0
        
        probs = hist.astype(xp.float32) / total
        probs = probs[probs > 0]  # Avoid log(0)
        
        entropy = -float(xp.sum(probs * xp.log2(probs)))
        return entropy
    
    def _convolve2d(self, image: "cp.ndarray", kernel: "cp.ndarray") -> "cp.ndarray":
        """
        Fast 2D convolution using GPU.
        Simplified version for 3×3 kernels.
        """
        xp = self.xp
        h, w = image.shape
        kh, kw = kernel.shape
        
        # Pad image
        pad_h, pad_w = kh // 2, kw // 2
        padded = xp.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # For 3×3 kernel, use direct computation (faster than generic convolution)
        if kh == 3 and kw == 3:
            result = (
                kernel[0, 0] * padded[0:h, 0:w] +
                kernel[0, 1] * padded[0:h, 1:w+1] +
                kernel[0, 2] * padded[0:h, 2:w+2] +
                kernel[1, 0] * padded[1:h+1, 0:w] +
                kernel[1, 1] * padded[1:h+1, 1:w+1] +
                kernel[1, 2] * padded[1:h+1, 2:w+2] +
                kernel[2, 0] * padded[2:h+2, 0:w] +
                kernel[2, 1] * padded[2:h+2, 1:w+1] +
                kernel[2, 2] * padded[2:h+2, 2:w+2]
            )
            return result
        
        # Fallback for other kernel sizes
        result = xp.zeros((h, w), dtype=xp.float32)
        for ky in range(kh):
            for kx in range(kw):
                result += kernel[ky, kx] * padded[ky:ky+h, kx:kx+w]
        return result
    
    def reset(self):
        """Reset engine state (clear previous frame)."""
        self.previous_frame = None
        self.previous_gray = None
        self.frame_counter = 0


class ScreenVectorBuffer:
    """
    Temporal buffer for ScreenVectors (sliding window analysis).
    """
    
    def __init__(self, max_size: int = 60):
        self.max_size = max_size
        self.vectors: List[ScreenVector] = []
    
    def push(self, sv: ScreenVector):
        """Add ScreenVector, evict oldest if at capacity."""
        self.vectors.append(sv)
        if len(self.vectors) > self.max_size:
            self.vectors.pop(0)
    
    def get_window(self, seconds: float) -> List[ScreenVector]:
        """Get ScreenVectors from last N seconds."""
        if not self.vectors:
            return []
        
        cutoff = self.vectors[-1].timestamp - seconds
        return [sv for sv in self.vectors if sv.timestamp >= cutoff]
    
    def get_motion_trend(self, seconds: float = 1.0) -> Dict:
        """Compute motion statistics over time window."""
        window = self.get_window(seconds)
        if len(window) < 2:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        
        magnitudes = [sv.motion_magnitude for sv in window]
        return {
            "mean": sum(magnitudes) / len(magnitudes),
            "std": (sum((m - sum(magnitudes)/len(magnitudes))**2 for m in magnitudes) / len(magnitudes)) ** 0.5,
            "max": max(magnitudes),
        }
    
    def get_flash_events(self, threshold: float = 0.3) -> List[ScreenVector]:
        """Get ScreenVectors with significant flash events."""
        return [sv for sv in self.vectors if sv.flash_intensity > threshold]


# Self-test harness
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SCREEN VECTOR ENGINE — Native Resolution GPU Test")
    print("="*70)
    
    # Initialize
    capture = NativeFrameCapture()
    engine = ScreenVectorEngine()
    buffer = ScreenVectorBuffer(max_size=30)
    
    print("\nCapturing 10 frames at NATIVE resolution...")
    print("-"*70)
    
    timings = []
    
    for i in range(10):
        t0 = time.time()
        
        # Capture at native resolution
        frame = capture.capture()
        if frame is None:
            print(f"[{i}] Capture failed")
            continue
        
        t_capture = time.time() - t0
        
        # Process to ScreenVector
        t1 = time.time()
        sv = engine.process(frame)
        t_process = time.time() - t1
        
        buffer.push(sv)
        
        total_ms = (time.time() - t0) * 1000
        timings.append(total_ms)
        
        print(f"[{i}] {sv.resolution[1]}×{sv.resolution[0]} | "
              f"Core: μ={sv.core_intensity_mean:.1f} σ={sv.core_intensity_std:.1f} | "
              f"Motion: {sv.motion_magnitude:.2f} | "
              f"Flash: {sv.flash_intensity:.3f} | "
              f"Time: {total_ms:.1f}ms")
        
        time.sleep(0.05)  # ~20 FPS
    
    print("-"*70)
    print(f"Average processing time: {sum(timings)/len(timings):.1f}ms")
    print(f"Budget for 60 FPS: 16.67ms")
    print(f"Status: {'✅ WITHIN BUDGET' if sum(timings)/len(timings) < 16.67 else '⚠️ OVER BUDGET'}")
    
    # Motion trend
    motion = buffer.get_motion_trend(1.0)
    print(f"\nMotion trend (1s): mean={motion['mean']:.2f}, max={motion['max']:.2f}")
    
    # Flash events
    flashes = buffer.get_flash_events(0.1)
    print(f"Flash events (>10%): {len(flashes)}")
    
    print("\n✅ Screen Vector Engine test complete")
    print("   NO 32×32. NATIVE RESOLUTION. GPU ACCELERATED.")
