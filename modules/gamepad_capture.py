"""
Gamepad Capture Module
CompuCog Multimodal Game Intelligence Engine

Real-time controller input capture:
- Button states
- Stick positions
- Trigger values
- Timing patterns

Built: November 25, 2025
"""

import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not installed. Gamepad capture disabled.")


class GamepadCapture:
    """
    Real-time gamepad/controller input capture and feature extraction.
    
    Captures button presses, stick movements, and timing for 616 fusion.
    """
    
    def __init__(
        self,
        poll_rate_hz: int = 120,
        deadzone: float = 0.1,
        history_size: int = 60
    ):
        """
        Args:
            poll_rate_hz: Polling rate (Hz)
            deadzone: Stick deadzone threshold
            history_size: Number of frames to keep in history
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame not installed. Run: pip install pygame")
        
        self.poll_rate_hz = poll_rate_hz
        self.deadzone = deadzone
        self.history_size = history_size
        self.poll_delay = 1.0 / poll_rate_hz
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Get first available controller
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[616 Gamepad Capture]")
            print(f"  Controller: {self.joystick.get_name()}")
            print(f"  Buttons: {self.joystick.get_numbuttons()}")
            print(f"  Axes: {self.joystick.get_numaxes()}")
            print(f"  Poll rate: {poll_rate_hz} Hz")
        else:
            print("[616 Gamepad Capture]")
            print("  [WARNING] No controller detected")
            print("  Gamepad features will be zero")
        
        # Input history
        self.button_history = deque(maxlen=history_size)
        self.stick_history = deque(maxlen=history_size)
        
        # Statistics
        self.frames_captured = 0
        self.total_capture_time = 0.0
        
        # Previous state for delta
        self.prev_buttons = None
        self.prev_sticks = None
    
    def apply_deadzone(self, value: float) -> float:
        """Apply deadzone to stick value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value
    
    def capture(self) -> Dict[str, np.ndarray]:
        """
        Capture current controller state.
        
        Returns:
            Dict with:
                - 'buttons': np.ndarray of button states (0 or 1)
                - 'sticks': np.ndarray of stick positions (float)
                - 'timestamp': Capture timestamp
        """
        import time
        start_time = time.perf_counter()
        
        # Process pygame events
        pygame.event.pump()
        
        if self.joystick is None:
            # No controller, return zeros
            return {
                'buttons': np.zeros(16, dtype=np.float32),
                'sticks': np.zeros(6, dtype=np.float32),
                'timestamp': start_time
            }
        
        # Capture buttons
        num_buttons = self.joystick.get_numbuttons()
        buttons = np.zeros(16, dtype=np.float32)  # Fixed size for consistency
        for i in range(min(num_buttons, 16)):
            buttons[i] = float(self.joystick.get_button(i))
        
        # Capture sticks/axes
        num_axes = self.joystick.get_numaxes()
        sticks = np.zeros(6, dtype=np.float32)  # Fixed size (left_x, left_y, right_x, right_y, LT, RT)
        for i in range(min(num_axes, 6)):
            value = self.joystick.get_axis(i)
            sticks[i] = self.apply_deadzone(value)
        
        # Update statistics
        capture_time = time.perf_counter() - start_time
        self.frames_captured += 1
        self.total_capture_time += capture_time
        
        return {
            'buttons': buttons,
            'sticks': sticks,
            'timestamp': start_time
        }
    
    def extract_features(self) -> Dict[str, np.ndarray]:
        """
        Capture controller state and extract features.
        
        Returns:
            Dict with:
                - 'buttons': Current button states (16,)
                - 'sticks': Current stick positions (6,)
                - 'button_deltas': Button changes from previous frame (16,)
                - 'stick_deltas': Stick changes from previous frame (6,)
                - 'button_press_count': Number of buttons pressed
                - 'stick_magnitude': Combined stick magnitude
                - 'feature_vector': Flattened feature vector (47 features)
                - 'timestamp': Capture timestamp
        """
        # Capture current state
        state = self.capture()
        buttons = state['buttons']
        sticks = state['sticks']
        timestamp = state['timestamp']
        
        # Compute deltas
        if self.prev_buttons is not None:
            button_deltas = np.abs(buttons - self.prev_buttons)
            stick_deltas = np.abs(sticks - self.prev_sticks)
        else:
            button_deltas = np.zeros_like(buttons)
            stick_deltas = np.zeros_like(sticks)
        
        # Compute summary features
        button_press_count = np.sum(buttons)
        stick_magnitude = np.sqrt(np.sum(sticks**2))
        
        # Stick velocity (if history available)
        if len(self.stick_history) > 0:
            prev_sticks = self.stick_history[-1]
            stick_velocity = np.sqrt(np.sum((sticks - prev_sticks)**2))
        else:
            stick_velocity = 0.0
        
        # Button press rate (last 60 frames)
        if len(self.button_history) > 0:
            button_press_rate = np.mean([np.sum(b) for b in self.button_history])
        else:
            button_press_rate = 0.0
        
        # Stick stats
        left_stick_angle = np.arctan2(sticks[1], sticks[0]) if np.sum(sticks[:2]**2) > 0.01 else 0.0
        right_stick_angle = np.arctan2(sticks[3], sticks[2]) if np.sum(sticks[2:4]**2) > 0.01 else 0.0
        left_stick_mag = np.sqrt(sticks[0]**2 + sticks[1]**2)
        right_stick_mag = np.sqrt(sticks[2]**2 + sticks[3]**2)
        
        # Update history
        self.button_history.append(buttons.copy())
        self.stick_history.append(sticks.copy())
        self.prev_buttons = buttons
        self.prev_sticks = sticks
        
        # Build feature vector
        feature_vector = np.concatenate([
            buttons,  # 16
            sticks,  # 6
            button_deltas,  # 16
            stick_deltas,  # 6
            [button_press_count],  # 1
            [stick_magnitude],  # 1
            [stick_velocity],  # 1
            [button_press_rate],  # 1
            [left_stick_angle, left_stick_mag],  # 2
            [right_stick_angle, right_stick_mag],  # 2
            sticks[4:6]  # LT, RT triggers (2)
        ])  # Total: 54 features
        
        return {
            'buttons': buttons,
            'sticks': sticks,
            'button_deltas': button_deltas,
            'stick_deltas': stick_deltas,
            'button_press_count': button_press_count,
            'stick_magnitude': stick_magnitude,
            'feature_vector': feature_vector,
            'timestamp': timestamp
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get capture statistics.
        
        Returns:
            Dict with frame count, average poll rate, average capture time
        """
        if self.frames_captured == 0:
            return {
                'frames_captured': 0,
                'avg_poll_rate': 0.0,
                'avg_capture_time_ms': 0.0
            }
        
        avg_capture_time = self.total_capture_time / self.frames_captured
        avg_poll_rate = 1.0 / avg_capture_time if avg_capture_time > 0 else 0.0
        
        return {
            'frames_captured': self.frames_captured,
            'avg_poll_rate': avg_poll_rate,
            'avg_capture_time_ms': avg_capture_time * 1000
        }
    
    def close(self):
        """Release resources."""
        if self.joystick is not None:
            self.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    """Test gamepad capture."""
    
    if not PYGAME_AVAILABLE:
        print("pygame not installed. Exiting.")
        exit(1)
    
    print("Testing 616 Gamepad Capture...")
    print("Press Ctrl+C to quit\n")
    
    # Initialize capture
    capture = GamepadCapture(
        poll_rate_hz=120,
        deadzone=0.1
    )
    
    try:
        while True:
            # Extract features
            features = capture.extract_features()
            
            # Print stats every 120 frames
            if capture.frames_captured % 120 == 0:
                stats = capture.get_statistics()
                print(f"[Frame {stats['frames_captured']:>5}] "
                      f"Poll rate: {stats['avg_poll_rate']:>6.1f} Hz | "
                      f"Capture: {stats['avg_capture_time_ms']:>5.2f}ms | "
                      f"Buttons: {int(features['button_press_count'])} | "
                      f"Stick mag: {features['stick_magnitude']:.3f}")
            
            # Rate limiting
            time.sleep(max(0, capture.poll_delay - 0.0001))
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        stats = capture.get_statistics()
        print(f"\n[Final Stats]")
        print(f"  Frames captured: {stats['frames_captured']}")
        print(f"  Average poll rate: {stats['avg_poll_rate']:.1f} Hz")
        print(f"  Average capture time: {stats['avg_capture_time_ms']:.2f}ms")
        
        capture.close()
