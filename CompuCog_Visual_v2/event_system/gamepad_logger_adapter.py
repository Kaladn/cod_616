"""
GamepadLoggerAdapter - Direct Controller Input Monitoring
Monitors gamepad/controller input and converts to Event objects per CONTRACT_ATLAS.md

Per Phase 2 Architecture:
- NO subprocess launches
- NO JSONL file writes
- Direct data collection via pygame joystick
- Returns Event objects → EventManager → Forge
"""

from typing import Dict, Any, Optional, List
import logging

from event_system.sensor_registry import SensorAdapter, SensorConfig
from event_system.event_manager import Event
from event_system.chronos_manager import ChronosManager


class GamepadLoggerAdapter(SensorAdapter):
    """
    Direct gamepad/controller input monitoring adapter.
    
    Per CONTRACT_ATLAS.md:
    - Monitors controller button presses, axis movements, triggers
    - Detects state changes (not continuous polling)
    - Converts to Event objects (NEVER dicts or ForgeRecords)
    - EventManager writes Events → Forge via BinaryLog
    - source_id: "gamepad_monitor"
    - tags: ["gamepad", "controller", "input"]
    """
    
    def __init__(self, config: SensorConfig, chronos: ChronosManager, event_mgr):
        super().__init__(config, chronos, event_mgr)
        self.pygame = None
        self.joystick = None
        self.controllers: List = []
        self.prev_button_states: Dict = {}
        self.prev_axis_values: Dict = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"GamepadAdapter[{config.source_id}]")
        
    def initialize(self) -> bool:
        """Initialize pygame and detect controllers."""
        try:
            # Import pygame (optional dependency)
            try:
                import pygame
                from pygame import joystick
                self.pygame = pygame
                self.joystick = joystick
            except ImportError:
                self.logger.error("pygame not installed - required for gamepad monitoring")
                self.logger.error("Install with: pip install pygame")
                return False
            
            # Initialize pygame subsystems
            self.pygame.init()
            self.joystick.init()
            
            # Detect controllers
            count = self.joystick.get_count()
            self.controllers = []
            
            for i in range(count):
                try:
                    controller = self.joystick.Joystick(i)
                    controller.init()
                    self.controllers.append(controller)
                    
                    # Initialize state tracking for this controller
                    self.prev_button_states[i] = [False] * controller.get_numbuttons()
                    self.prev_axis_values[i] = [0.0] * controller.get_numaxes()
                    
                    self.logger.info(f"Controller {i}: {controller.get_name()}")
                except Exception as e:
                    self.logger.warning(f"Failed to init controller {i}: {e}")
            
            if not self.controllers:
                self.logger.warning("No gamepads detected")
                return False
            
            self.logger.info(f"GamepadLoggerAdapter initialized ({len(self.controllers)} controllers)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GamepadLoggerAdapter: {e}")
            return False
    
    def start(self) -> bool:
        """Start monitoring (no subprocess needed)."""
        try:
            self.running = True
            self.logger.info("Gamepad monitoring started (direct pygame mode)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start gamepad monitoring: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop monitoring and cleanup pygame."""
        try:
            self.running = False
            
            # Quit controllers
            for controller in self.controllers:
                try:
                    controller.quit()
                except:
                    pass
            
            # Quit pygame
            if self.pygame:
                self.pygame.quit()
            
            self.logger.info("Gamepad monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop gamepad monitoring: {e}")
            return False
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Poll controllers for state changes.
        
        Returns:
            Dict with gamepad event data, or None if no changes
        """
        try:
            if not self.controllers:
                return None
            
            # Process pygame events (required for joystick updates)
            if self.pygame:
                self.pygame.event.pump()
            
            # Check each controller for changes
            for controller_id, controller in enumerate(self.controllers):
                # Check buttons
                num_buttons = controller.get_numbuttons()
                for button_id in range(num_buttons):
                    button_pressed = controller.get_button(button_id)
                    prev_state = self.prev_button_states[controller_id][button_id]
                    
                    if button_pressed != prev_state:
                        self.prev_button_states[controller_id][button_id] = button_pressed
                        
                        return {
                            'event_type': 'button_press' if button_pressed else 'button_release',
                            'controller_id': controller_id,
                            'controller_name': controller.get_name(),
                            'button': button_id,
                            'state': 1 if button_pressed else 0
                        }
                
                # Check axes (sticks/triggers) - only report significant changes
                num_axes = controller.get_numaxes()
                for axis_id in range(num_axes):
                    axis_value = controller.get_axis(axis_id)
                    prev_value = self.prev_axis_values[controller_id][axis_id]
                    
                    # Only report if change > 0.1 (reduce noise)
                    if abs(axis_value - prev_value) > 0.1:
                        self.prev_axis_values[controller_id][axis_id] = axis_value
                        
                        # Map axis to semantic name
                        axis_name = self._get_axis_name(axis_id)
                        
                        return {
                            'event_type': 'axis_move',
                            'controller_id': controller_id,
                            'controller_name': controller.get_name(),
                            'axis': axis_id,
                            'axis_name': axis_name,
                            'value': round(axis_value, 3)
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error polling gamepad: {e}")
            return None
    
    def _get_axis_name(self, axis_id: int) -> str:
        """Map axis ID to semantic name (Xbox controller layout)."""
        axis_names = {
            0: "left_stick_x",
            1: "left_stick_y",
            2: "left_trigger",
            3: "right_stick_x",
            4: "right_stick_y",
            5: "right_trigger"
        }
        return axis_names.get(axis_id, f"axis_{axis_id}")
    
    def convert_to_event(self, data: Dict[str, Any]) -> Event:
        """
        Convert gamepad data to Event object per CONTRACT_ATLAS.md.
        
        Per CONTRACT_ATLAS.md Contract #2:
        - MUST return Event object (NEVER dict or ForgeRecord)
        - timestamp from ChronosManager (deterministic)
        - source_id from config
        - tags include sensor type
        - metadata is JSON-serializable
        
        Args:
            data: Dict from get_latest_data()
        
        Returns:
            Event object (validated by SensorAdapter.record_event)
        """
        # Get deterministic timestamp from ChronosManager
        timestamp = self.chronos.now()
        
        event_type = data.get('event_type', 'unknown')
        
        # Build Event object per CONTRACT_ATLAS.md
        return Event(
            event_id=f"gamepad_{event_type}_{int(timestamp * 1000)}",
            timestamp=timestamp,
            source_id=self.config.source_id,
            tags=self.config.tags + ["gamepad", "controller", event_type],
            metadata={
                "event_type": event_type,
                "controller_id": data.get("controller_id", 0),
                "controller_name": data.get("controller_name", "unknown"),
                "button": data.get("button"),
                "button_state": data.get("state"),
                "axis": data.get("axis"),
                "axis_name": data.get("axis_name"),
                "axis_value": data.get("value")
            }
        )
