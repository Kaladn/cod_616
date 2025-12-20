#!/usr/bin/env python3

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
╚══════════════════════════════════════════════════════════════════════════════╝

CompuCog Gamepad Logger - Continuous Event Stream

Records EVERY controller input event with precise timestamps.
Continuous recording with NO gaps - logs raw event stream for temporal correlation.

Output Format (JSONL):
  {"timestamp": "2025-12-02T16:52:23.001234Z", "event": "button_press", "button": 0, "state": 1}
  {"timestamp": "2025-12-02T16:52:23.045678Z", "event": "axis_move", "axis": 0, "value": 0.523}
  {"timestamp": "2025-12-02T16:52:23.089012Z", "event": "trigger", "trigger": "RT", "value": 0.812}
  {"timestamp": "2025-12-02T16:52:23.134567Z", "event": "hat_change", "hat": 0, "value": [1, 0]}

Usage can aggregate these events by timestamp range to match TrueVision windows.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pygame
from pygame import joystick

# Initialize pygame and joystick subsystem
pygame.init()
joystick.init()


def get_project_root() -> str:
    """Get CompuCog root directory"""
    logger_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(logger_dir, os.pardir))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class ContinuousGamepadLogger:
    """
    Continuous event stream logger for gamepad input.
    Records every state change with microsecond-precision timestamps.
    """
    
    def __init__(self, log_path: str, poll_rate: int = 60):
        self.log_path = log_path
        self.poll_rate = poll_rate
        self.poll_interval = 1.0 / poll_rate
        
        self.controllers: List[pygame.joystick.Joystick] = []
        # Track previous state to detect changes (initialized in _init_controllers)
        self.prev_button_states = {}
        self.prev_axis_values = {}
        self.prev_hat_values = {}
        
        # Event counter for CLI display
        self.event_count = 0
        
        self._init_controllers()
        
        # Open log file in append mode with line buffering
        self.log_file = open(self.log_path, 'a', buffering=1, encoding='utf-8')
        
    def _init_controllers(self):
        """Initialize all connected controllers"""
        count = joystick.get_count()
        self.controllers = []
        
        # Initialize tracking dicts first
        self.prev_button_states = {}
        self.prev_axis_values = {}
        self.prev_hat_values = {}
        
        for i in range(count):
            try:
                controller = joystick.Joystick(i)
                controller.init()
                self.controllers.append(controller)
                print(f"[Gamepad] Detected: {controller.get_name()} (ID: {i})")
                
                # Initialize state tracking for this controller
                self.prev_button_states[i] = [False] * controller.get_numbuttons()
                self.prev_axis_values[i] = [0.0] * controller.get_numaxes()
                self.prev_hat_values[i] = [(0, 0)] * controller.get_numhats()
                
            except Exception as e:
                print(f"[Gamepad] Failed to init controller {i}: {e}")
    
    def log_event(self, event_type: str, **kwargs):
        """Write event to JSONL log with microsecond timestamp"""
        event = {
            "timestamp": iso_now(),
            "event": event_type,
            **kwargs
        }
        self.log_file.write(json.dumps(event) + '\n')
        self.event_count += 1
    
    def poll_and_log(self):
        """
        Poll all controllers and log state changes.
        Call this continuously at poll_rate (e.g., 60Hz).
        """
        if not self.controllers:
            return
        
        # Process pending events (required for pygame joystick updates)
        pygame.event.pump()
        
        for controller_id, controller in enumerate(self.controllers):
            try:
                # Check button changes
                num_buttons = controller.get_numbuttons()
                for button_id in range(num_buttons):
                    current_state = controller.get_button(button_id)
                    prev_state = self.prev_button_states[controller_id][button_id]
                    
                    if current_state != prev_state:
                        self.log_event(
                            "button",
                            controller=controller_id,
                            button=button_id,
                            state=1 if current_state else 0
                        )
                        self.prev_button_states[controller_id][button_id] = current_state
                
                # Check axis changes (sticks, triggers)
                num_axes = controller.get_numaxes()
                for axis_id in range(num_axes):
                    current_value = controller.get_axis(axis_id)
                    prev_value = self.prev_axis_values[controller_id][axis_id]
                    
                    # Log if change > 0.01 (1% deadzone)
                    if abs(current_value - prev_value) > 0.01:
                        # Determine axis type for semantic labeling
                        axis_name = self._get_axis_name(axis_id)
                        
                        self.log_event(
                            "axis",
                            controller=controller_id,
                            axis=axis_id,
                            axis_name=axis_name,
                            value=round(current_value, 3)
                        )
                        self.prev_axis_values[controller_id][axis_id] = current_value
                
                # Check hat (D-pad) changes
                num_hats = controller.get_numhats()
                for hat_id in range(num_hats):
                    current_value = controller.get_hat(hat_id)
                    prev_value = self.prev_hat_values[controller_id][hat_id]
                    
                    if current_value != prev_value:
                        self.log_event(
                            "hat",
                            controller=controller_id,
                            hat=hat_id,
                            value=list(current_value)  # (x, y) tuple
                        )
                        self.prev_hat_values[controller_id][hat_id] = current_value
                
            except pygame.error as e:
                # Controller disconnected or hardware error - silent skip
                pass
            except Exception:
                # Other errors - silent skip to avoid spam
                pass
    
    def _get_axis_name(self, axis_id: int) -> str:
        """Map axis ID to semantic name (Xbox controller layout)"""
        axis_map = {
            0: "left_stick_x",
            1: "left_stick_y",
            2: "right_stick_x",
            3: "right_stick_y",
            4: "left_trigger",
            5: "right_trigger"
        }
        return axis_map.get(axis_id, f"axis_{axis_id}")
    
    def run(self):
        """Main continuous logging loop"""
        print(f"[Gamepad] Logging to {self.log_path}")
        print(f"[Gamepad] Poll rate: {self.poll_rate}Hz")
        print(f"[Gamepad] Continuous event stream (Ctrl+C to stop)")
        
        # Log initial connection state
        for i, controller in enumerate(self.controllers):
            self.log_event(
                "connected",
                controller=i,
                name=controller.get_name(),
                num_buttons=controller.get_numbuttons(),
                num_axes=controller.get_numaxes(),
                num_hats=controller.get_numhats()
            )
        
        # Event counter for CLI display
        last_display_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                self.poll_and_log()
                
                # Update counter display every second
                if time.time() - last_display_time >= 1.0:
                    print(f"\rEvents logged: {self.event_count}", end='', flush=True)
                    last_display_time = time.time()
                
                # Sleep to maintain poll rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.poll_interval - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\n[Gamepad] Stopped - Total events: {self.event_count}")
        finally:
            self.log_file.close()
            for controller in self.controllers:
                controller.quit()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CompuCog Continuous Gamepad Logger"
    )
    parser.add_argument(
        "--poll-rate",
        type=int,
        default=60,
        help="Controller polling frequency in Hz (default: 60)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    
    # Setup log path
    root = get_project_root()
    log_dir = os.path.join(root, "logs", "gamepad")
    ensure_dir(log_dir)
    
    today = dt.datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"gamepad_stream_{today}.jsonl")
    
    # Run continuous logger
    logger = ContinuousGamepadLogger(log_path, poll_rate=args.poll_rate)
    logger.run()
