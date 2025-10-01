#!/usr/bin/env python3
"""
Advanced MIDI Controller Integration for Hardcore Music Production
Support for Novation Launchpad, MIDI Fighter, LaunchKey, and LaunchControl
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import threading
import queue

try:
    import mido
    import rtmidi
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    mido = None
    rtmidi = None

import numpy as np
from collections import defaultdict

from ..interfaces.synthesizer import AbstractSynthesizer
from ..models.hardcore_models import HardcorePattern, SynthParams
from ..performance.live_performance_engine import LivePerformanceEngine, TransitionType


logger = logging.getLogger(__name__)


class ControllerType(Enum):
    LAUNCHPAD_MK1 = "launchpad_mk1"
    LAUNCHPAD_MK2 = "launchpad_mk2" 
    MIDI_FIGHTER_3D = "midi_fighter_3d"
    LAUNCHKEY_49 = "launchkey_49"
    LAUNCH_CONTROL = "launch_control"
    GENERIC_MIDI = "generic_midi"


class MIDIMessageType(Enum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    CONTROL_CHANGE = "control_change"
    PROGRAM_CHANGE = "program_change"
    PITCH_BEND = "pitch_bend"
    AFTERTOUCH = "aftertouch"


@dataclass 
class MIDIMapping:
    """MIDI control mapping definition"""
    controller_type: ControllerType
    midi_type: MIDIMessageType
    midi_channel: int
    midi_note_or_cc: int
    function_name: str
    parameter_name: Optional[str] = None
    value_range: Tuple[float, float] = (0.0, 1.0)
    curve_type: str = "linear"  # linear, exponential, logarithmic
    description: str = ""


@dataclass
class ControllerState:
    """Current state of a MIDI controller"""
    controller_type: ControllerType
    is_connected: bool = False
    port_name: str = ""
    button_states: Dict[int, bool] = field(default_factory=dict)
    fader_values: Dict[int, float] = field(default_factory=dict)
    knob_values: Dict[int, float] = field(default_factory=dict)
    last_activity: float = 0.0
    total_messages: int = 0


class LEDPattern(Enum):
    """LED patterns for visual feedback"""
    OFF = "off"
    SOLID = "solid"
    PULSE = "pulse"
    FLASH = "flash"
    RAINBOW = "rainbow"
    VU_METER = "vu_meter"
    PATTERN_GRID = "pattern_grid"


@dataclass
class LEDCommand:
    """Command to control LED feedback"""
    controller_type: ControllerType
    led_id: Union[int, str]
    pattern: LEDPattern
    color: Optional[Tuple[int, int, int]] = None  # RGB
    intensity: float = 1.0
    speed: float = 1.0  # For animated patterns


class AbstractMIDIController(ABC):
    """Abstract base class for MIDI controllers"""
    
    def __init__(self, controller_type: ControllerType):
        self.controller_type = controller_type
        self.state = ControllerState(controller_type)
        self.mappings: Dict[Tuple[MIDIMessageType, int, int], MIDIMapping] = {}
        self.callback_handlers: Dict[str, Callable] = {}
        self.led_commands: List[LEDCommand] = []
        
    @abstractmethod
    async def connect(self, port_name: Optional[str] = None) -> bool:
        """Connect to the MIDI controller"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the MIDI controller"""
        pass
    
    @abstractmethod
    async def send_led_update(self, command: LEDCommand):
        """Send LED update to controller"""
        pass
    
    @abstractmethod
    def get_default_mappings(self) -> List[MIDIMapping]:
        """Get default MIDI mappings for this controller"""
        pass
    
    def add_mapping(self, mapping: MIDIMapping):
        """Add a MIDI mapping"""
        key = (mapping.midi_type, mapping.midi_channel, mapping.midi_note_or_cc)
        self.mappings[key] = mapping
        logger.debug(f"Added mapping: {mapping.function_name} -> {key}")
    
    def add_callback_handler(self, function_name: str, handler: Callable):
        """Add callback handler for controller functions"""
        self.callback_handlers[function_name] = handler
        logger.debug(f"Added callback handler: {function_name}")
    
    async def handle_midi_message(self, message):
        """Handle incoming MIDI message"""
        try:
            self.state.last_activity = time.time()
            self.state.total_messages += 1
            
            # Parse message
            msg_type = MIDIMessageType.NOTE_ON if message.type == 'note_on' else \
                      MIDIMessageType.NOTE_OFF if message.type == 'note_off' else \
                      MIDIMessageType.CONTROL_CHANGE if message.type == 'control_change' else None
            
            if not msg_type:
                return
            
            # Find mapping
            key = (msg_type, message.channel, getattr(message, 'note', getattr(message, 'control', 0)))
            mapping = self.mappings.get(key)
            
            if not mapping:
                logger.debug(f"No mapping found for {key}")
                return
            
            # Update controller state
            if msg_type in [MIDIMessageType.NOTE_ON, MIDIMessageType.NOTE_OFF]:
                note = message.note
                self.state.button_states[note] = (msg_type == MIDIMessageType.NOTE_ON and message.velocity > 0)
            elif msg_type == MIDIMessageType.CONTROL_CHANGE:
                cc = message.control
                normalized_value = message.value / 127.0
                
                # Apply curve
                if mapping.curve_type == "exponential":
                    normalized_value = normalized_value ** 2
                elif mapping.curve_type == "logarithmic":
                    normalized_value = np.sqrt(normalized_value)
                
                # Scale to parameter range
                param_value = mapping.value_range[0] + (normalized_value * (mapping.value_range[1] - mapping.value_range[0]))
                
                if "fader" in mapping.description.lower():
                    self.state.fader_values[cc] = param_value
                else:
                    self.state.knob_values[cc] = param_value
            
            # Call handler if available
            handler = self.callback_handlers.get(mapping.function_name)
            if handler:
                await self._call_handler_safely(handler, mapping, message)
            
        except Exception as e:
            logger.error(f"Error handling MIDI message: {e}")
    
    async def _call_handler_safely(self, handler: Callable, mapping: MIDIMapping, message):
        """Safely call a handler function"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(mapping, message)
            else:
                handler(mapping, message)
        except Exception as e:
            logger.error(f"Handler {mapping.function_name} failed: {e}")


class LaunchpadMK1Controller(AbstractMIDIController):
    """Novation Launchpad MK1 controller implementation"""
    
    def __init__(self):
        super().__init__(ControllerType.LAUNCHPAD_MK1)
        self.grid_size = (8, 8)  # 8x8 grid
        self.side_buttons = 8
        self.top_buttons = 8
        
    async def connect(self, port_name: Optional[str] = None) -> bool:
        """Connect to Launchpad"""
        if not MIDI_AVAILABLE:
            logger.error("MIDI not available - install mido and rtmidi")
            return False
        
        try:
            # Find Launchpad port
            available_ports = mido.get_input_names()
            launchpad_port = None
            
            for port in available_ports:
                if "launchpad" in port.lower():
                    launchpad_port = port
                    break
            
            if not launchpad_port and port_name:
                launchpad_port = port_name
            
            if not launchpad_port:
                logger.warning("No Launchpad found in available ports")
                return False
            
            # Open MIDI ports
            self.input_port = mido.open_input(launchpad_port, callback=self._midi_callback)
            self.output_port = mido.open_output(launchpad_port)
            
            self.state.is_connected = True
            self.state.port_name = launchpad_port
            
            # Initialize LED display
            await self._initialize_display()
            
            logger.info(f"Connected to Launchpad: {launchpad_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Launchpad: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Launchpad"""
        try:
            if hasattr(self, 'input_port'):
                self.input_port.close()
            if hasattr(self, 'output_port'):
                self.output_port.close()
            
            self.state.is_connected = False
            logger.info("Disconnected from Launchpad")
            
        except Exception as e:
            logger.error(f"Error disconnecting Launchpad: {e}")
    
    def _midi_callback(self, message):
        """MIDI message callback"""
        asyncio.create_task(self.handle_midi_message(message))
    
    async def _initialize_display(self):
        """Initialize Launchpad LED display"""
        try:
            # Clear all LEDs
            for x in range(8):
                for y in range(8):
                    note = self._xy_to_note(x, y)
                    msg = mido.Message('note_on', note=note, velocity=0)
                    self.output_port.send(msg)
            
            # Set up initial pattern
            await self._display_startup_pattern()
            
        except Exception as e:
            logger.error(f"Failed to initialize Launchpad display: {e}")
    
    async def _display_startup_pattern(self):
        """Display startup pattern on Launchpad"""
        try:
            # Flash red pattern
            for i in range(3):
                # All red
                for x in range(8):
                    for y in range(8):
                        note = self._xy_to_note(x, y)
                        msg = mido.Message('note_on', note=note, velocity=3)  # Red
                        self.output_port.send(msg)
                
                await asyncio.sleep(0.2)
                
                # All off
                for x in range(8):
                    for y in range(8):
                        note = self._xy_to_note(x, y)
                        msg = mido.Message('note_on', note=note, velocity=0)
                        self.output_port.send(msg)
                
                await asyncio.sleep(0.2)
            
            # Set up pattern slot indicators
            await self._display_slot_pattern()
            
        except Exception as e:
            logger.error(f"Failed to display startup pattern: {e}")
    
    async def _display_slot_pattern(self):
        """Display pattern slots on grid"""
        try:
            # Light up first 8 pads as pattern slots
            for i in range(8):
                x = i % 8
                y = 0
                note = self._xy_to_note(x, y)
                msg = mido.Message('note_on', note=note, velocity=1)  # Dim green
                self.output_port.send(msg)
            
        except Exception as e:
            logger.error(f"Failed to display slot pattern: {e}")
    
    def _xy_to_note(self, x: int, y: int) -> int:
        """Convert XY coordinates to MIDI note number"""
        return (y * 16) + x
    
    def _note_to_xy(self, note: int) -> Tuple[int, int]:
        """Convert MIDI note to XY coordinates"""
        y = note // 16
        x = note % 16
        return (x, y)
    
    async def send_led_update(self, command: LEDCommand):
        """Send LED update to Launchpad"""
        try:
            if not self.state.is_connected:
                return
            
            # Convert LED command to MIDI
            if isinstance(command.led_id, int):
                note = command.led_id
            elif isinstance(command.led_id, str) and 'x' in command.led_id:
                # Parse "x,y" format
                x, y = map(int, command.led_id.split(','))
                note = self._xy_to_note(x, y)
            else:
                return
            
            # Convert pattern to velocity
            if command.pattern == LEDPattern.OFF:
                velocity = 0
            elif command.pattern == LEDPattern.SOLID:
                velocity = int(command.intensity * 3)  # Launchpad has 4 levels (0-3)
            elif command.pattern == LEDPattern.FLASH:
                velocity = 3 if int(time.time() * command.speed * 2) % 2 else 0
            else:
                velocity = 1  # Default
            
            msg = mido.Message('note_on', note=note, velocity=velocity)
            self.output_port.send(msg)
            
        except Exception as e:
            logger.error(f"Failed to send LED update: {e}")
    
    def get_default_mappings(self) -> List[MIDIMapping]:
        """Get default mappings for Launchpad MK1"""
        mappings = []
        
        # Grid pads for pattern triggering (first row)
        for i in range(8):
            note = self._xy_to_note(i, 0)
            mappings.append(MIDIMapping(
                controller_type=self.controller_type,
                midi_type=MIDIMessageType.NOTE_ON,
                midi_channel=0,
                midi_note_or_cc=note,
                function_name=f"trigger_slot_{i:02d}",
                description=f"Trigger pattern slot {i}"
            ))
        
        # Side buttons for transport control
        side_buttons = [104, 105, 106, 107, 108, 109, 110, 111]
        functions = ["play", "stop", "record", "overdub", "solo", "mute", "arm", "select"]
        
        for button, function in zip(side_buttons, functions):
            mappings.append(MIDIMapping(
                controller_type=self.controller_type,
                midi_type=MIDIMessageType.NOTE_ON,
                midi_channel=0,
                midi_note_or_cc=button,
                function_name=function,
                description=f"{function.title()} button"
            ))
        
        # Top buttons for scene triggering
        top_buttons = [8, 24, 40, 56, 72, 88, 104, 120]
        for i, button in enumerate(top_buttons):
            mappings.append(MIDIMapping(
                controller_type=self.controller_type,
                midi_type=MIDIMessageType.NOTE_ON,
                midi_channel=0,
                midi_note_or_cc=button,
                function_name=f"scene_{i}",
                description=f"Scene {i} button"
            ))
        
        return mappings


class MIDIFighter3DController(AbstractMIDIController):
    """DJ TechTools MIDI Fighter 3D controller implementation"""
    
    def __init__(self):
        super().__init__(ControllerType.MIDI_FIGHTER_3D)
        self.arcade_buttons = 16
        self.motion_sensors = 4  # X, Y, Z axes + pressure
        
    async def connect(self, port_name: Optional[str] = None) -> bool:
        """Connect to MIDI Fighter 3D"""
        if not MIDI_AVAILABLE:
            return False
        
        try:
            available_ports = mido.get_input_names()
            fighter_port = None
            
            for port in available_ports:
                if "midi fighter" in port.lower() or "fighter" in port.lower():
                    fighter_port = port
                    break
            
            if not fighter_port and port_name:
                fighter_port = port_name
            
            if not fighter_port:
                logger.warning("No MIDI Fighter 3D found")
                return False
            
            self.input_port = mido.open_input(fighter_port, callback=self._midi_callback)
            self.output_port = mido.open_output(fighter_port) if fighter_port else None
            
            self.state.is_connected = True
            self.state.port_name = fighter_port
            
            logger.info(f"Connected to MIDI Fighter 3D: {fighter_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MIDI Fighter 3D: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MIDI Fighter 3D"""
        try:
            if hasattr(self, 'input_port'):
                self.input_port.close()
            if hasattr(self, 'output_port') and self.output_port:
                self.output_port.close()
            
            self.state.is_connected = False
            logger.info("Disconnected from MIDI Fighter 3D")
            
        except Exception as e:
            logger.error(f"Error disconnecting MIDI Fighter 3D: {e}")
    
    def _midi_callback(self, message):
        """MIDI message callback"""
        asyncio.create_task(self.handle_midi_message(message))
    
    async def send_led_update(self, command: LEDCommand):
        """Send LED update to MIDI Fighter 3D"""
        # MIDI Fighter 3D uses velocity to control LED brightness
        if not self.state.is_connected or not hasattr(self, 'output_port'):
            return
        
        try:
            if isinstance(command.led_id, int) and command.led_id < 16:
                note = 36 + command.led_id  # Base note + button offset
                
                if command.pattern == LEDPattern.OFF:
                    velocity = 0
                elif command.pattern == LEDPattern.SOLID:
                    velocity = int(command.intensity * 127)
                else:
                    velocity = 64  # Medium brightness
                
                msg = mido.Message('note_on', note=note, velocity=velocity)
                self.output_port.send(msg)
                
        except Exception as e:
            logger.error(f"Failed to send LED update to MIDI Fighter 3D: {e}")
    
    def get_default_mappings(self) -> List[MIDIMapping]:
        """Get default mappings for MIDI Fighter 3D"""
        mappings = []
        
        # 16 arcade buttons for various functions
        button_functions = [
            "trigger_slot_00", "trigger_slot_01", "trigger_slot_02", "trigger_slot_03",
            "trigger_slot_04", "trigger_slot_05", "trigger_slot_06", "trigger_slot_07",
            "effect_filter", "effect_delay", "effect_reverb", "effect_distortion",
            "transport_play", "transport_stop", "record_arm", "loop_toggle"
        ]
        
        for i, function in enumerate(button_functions):
            note = 36 + i  # MIDI Fighter 3D typically starts at note 36
            mappings.append(MIDIMapping(
                controller_type=self.controller_type,
                midi_type=MIDIMessageType.NOTE_ON,
                midi_channel=0,
                midi_note_or_cc=note,
                function_name=function,
                description=f"Button {i+1}: {function}"
            ))
        
        # Motion sensor controls
        motion_ccs = [12, 13, 14, 15]  # X, Y, Z, Pressure
        motion_functions = ["crossfader", "filter_cutoff", "effect_send", "master_volume"]
        
        for cc, function in zip(motion_ccs, motion_functions):
            mappings.append(MIDIMapping(
                controller_type=self.controller_type,
                midi_type=MIDIMessageType.CONTROL_CHANGE,
                midi_channel=0,
                midi_note_or_cc=cc,
                function_name=function,
                parameter_name=function,
                value_range=(0.0, 1.0),
                description=f"Motion sensor: {function}"
            ))
        
        return mappings


class HardwareMIDIIntegration:
    """Main hardware integration manager"""
    
    def __init__(self, 
                 synthesizer: AbstractSynthesizer,
                 performance_engine: Optional[LivePerformanceEngine] = None):
        self.synthesizer = synthesizer
        self.performance_engine = performance_engine
        
        # Controller management
        self.controllers: Dict[ControllerType, AbstractMIDIController] = {}
        self.active_controllers: List[AbstractMIDIController] = []
        
        # MIDI port scanning
        self.available_ports: List[str] = []
        self.scan_interval = 5.0  # Scan every 5 seconds
        self.is_scanning = False
        
        # LED feedback system
        self.led_update_queue = asyncio.Queue()
        self.led_patterns = {}
        
        # Statistics
        self.stats = {
            "total_midi_messages": 0,
            "messages_per_controller": defaultdict(int),
            "last_activity": {},
            "connection_attempts": 0,
            "successful_connections": 0
        }
    
    def register_controller_type(self, controller_class):
        """Register a controller type"""
        controller = controller_class()
        self.controllers[controller.controller_type] = controller
        logger.info(f"Registered controller type: {controller.controller_type.value}")
    
    async def scan_for_controllers(self) -> List[str]:
        """Scan for available MIDI controllers"""
        if not MIDI_AVAILABLE:
            logger.warning("MIDI not available for controller scanning")
            return []
        
        try:
            self.available_ports = mido.get_input_names()
            logger.debug(f"Found MIDI ports: {self.available_ports}")
            return self.available_ports
            
        except Exception as e:
            logger.error(f"Failed to scan MIDI ports: {e}")
            return []
    
    async def auto_connect_controllers(self) -> Dict[ControllerType, bool]:
        """Automatically connect to available controllers"""
        await self.scan_for_controllers()
        connection_results = {}
        
        for controller_type, controller in self.controllers.items():
            self.stats["connection_attempts"] += 1
            
            try:
                success = await controller.connect()
                connection_results[controller_type] = success
                
                if success:
                    self.active_controllers.append(controller)
                    self.stats["successful_connections"] += 1
                    
                    # Set up default mappings and handlers
                    await self._setup_controller(controller)
                    
                    logger.info(f"Successfully connected: {controller_type.value}")
                else:
                    logger.debug(f"Failed to connect: {controller_type.value}")
                    
            except Exception as e:
                logger.error(f"Error connecting {controller_type.value}: {e}")
                connection_results[controller_type] = False
        
        return connection_results
    
    async def _setup_controller(self, controller: AbstractMIDIController):
        """Set up a controller with default mappings and handlers"""
        try:
            # Add default mappings
            default_mappings = controller.get_default_mappings()
            for mapping in default_mappings:
                controller.add_mapping(mapping)
            
            # Add callback handlers
            await self._setup_callback_handlers(controller)
            
            # Start LED feedback for this controller
            asyncio.create_task(self._controller_led_feedback_loop(controller))
            
        except Exception as e:
            logger.error(f"Failed to setup controller {controller.controller_type.value}: {e}")
    
    async def _setup_callback_handlers(self, controller: AbstractMIDIController):
        """Set up callback handlers for a controller"""
        
        # Pattern slot triggers
        for i in range(8):
            slot_function = f"trigger_slot_{i:02d}"
            controller.add_callback_handler(
                slot_function,
                lambda mapping, msg, slot_id=f"slot_{i:02d}": 
                    asyncio.create_task(self._handle_slot_trigger(slot_id, mapping, msg))
            )
        
        # Transport controls
        transport_handlers = {
            "play": self._handle_play,
            "stop": self._handle_stop,
            "record": self._handle_record,
            "overdub": self._handle_overdub
        }
        
        for function, handler in transport_handlers.items():
            controller.add_callback_handler(function, handler)
        
        # Effect controls
        effect_handlers = {
            "crossfader": self._handle_crossfader,
            "filter_cutoff": self._handle_filter,
            "effect_send": self._handle_effect_send,
            "master_volume": self._handle_master_volume
        }
        
        for function, handler in effect_handlers.items():
            controller.add_callback_handler(function, handler)
    
    async def _handle_slot_trigger(self, slot_id: str, mapping: MIDIMapping, message):
        """Handle pattern slot trigger"""
        if not self.performance_engine:
            logger.warning("No performance engine available for slot trigger")
            return
        
        try:
            # Only trigger on note on
            if message.type == 'note_on' and message.velocity > 0:
                await self.performance_engine.trigger_slot(slot_id)
                
                # Send LED feedback
                await self._send_led_feedback(
                    mapping.controller_type,
                    mapping.midi_note_or_cc,
                    LEDPattern.FLASH,
                    intensity=1.0
                )
                
                logger.debug(f"Triggered slot {slot_id}")
                
        except Exception as e:
            logger.error(f"Failed to handle slot trigger: {e}")
    
    async def _handle_play(self, mapping: MIDIMapping, message):
        """Handle play button"""
        if message.type == 'note_on' and message.velocity > 0:
            if self.performance_engine:
                # Toggle play state
                stats = self.performance_engine.get_performance_stats()
                if stats["state"] == "playing":
                    await self.performance_engine.stop_performance()
                else:
                    await self.performance_engine.start_performance()
    
    async def _handle_stop(self, mapping: MIDIMapping, message):
        """Handle stop button"""
        if message.type == 'note_on' and message.velocity > 0:
            if self.performance_engine:
                await self.performance_engine.stop_performance()
    
    async def _handle_record(self, mapping: MIDIMapping, message):
        """Handle record button"""
        if message.type == 'note_on' and message.velocity > 0:
            # Toggle recording (implementation specific)
            logger.info("Record button pressed")
    
    async def _handle_overdub(self, mapping: MIDIMapping, message):
        """Handle overdub button"""
        if message.type == 'note_on' and message.velocity > 0:
            logger.info("Overdub button pressed")
    
    async def _handle_crossfader(self, mapping: MIDIMapping, message):
        """Handle crossfader control"""
        if message.type == 'control_change':
            # Convert MIDI value to crossfader position (-1 to 1)
            position = (message.value / 127.0 * 2.0) - 1.0
            
            if self.performance_engine:
                await self.performance_engine.set_crossfader(position)
    
    async def _handle_filter(self, mapping: MIDIMapping, message):
        """Handle filter cutoff control"""
        if message.type == 'control_change':
            cutoff = message.value / 127.0
            # Apply to active slots (implementation specific)
            logger.debug(f"Filter cutoff: {cutoff:.2f}")
    
    async def _handle_effect_send(self, mapping: MIDIMapping, message):
        """Handle effect send control"""
        if message.type == 'control_change':
            send_level = message.value / 127.0
            logger.debug(f"Effect send: {send_level:.2f}")
    
    async def _handle_master_volume(self, mapping: MIDIMapping, message):
        """Handle master volume control"""
        if message.type == 'control_change':
            volume = message.value / 127.0
            
            if self.performance_engine:
                await self.performance_engine.set_master_volume(volume)
    
    async def _send_led_feedback(self, 
                                controller_type: ControllerType,
                                led_id: Union[int, str],
                                pattern: LEDPattern,
                                **kwargs):
        """Send LED feedback to controller"""
        command = LEDCommand(
            controller_type=controller_type,
            led_id=led_id,
            pattern=pattern,
            **kwargs
        )
        
        await self.led_update_queue.put(command)
    
    async def _controller_led_feedback_loop(self, controller: AbstractMIDIController):
        """LED feedback loop for a controller"""
        try:
            while controller.state.is_connected:
                try:
                    # Get LED command from queue (with timeout)
                    command = await asyncio.wait_for(
                        self.led_update_queue.get(),
                        timeout=0.1
                    )
                    
                    if command.controller_type == controller.controller_type:
                        await controller.send_led_update(command)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"LED feedback error: {e}")
                    
        except Exception as e:
            logger.error(f"LED feedback loop failed: {e}")
    
    async def start_monitoring(self):
        """Start monitoring controllers"""
        self.is_scanning = True
        
        # Start port scanning task
        asyncio.create_task(self._port_scan_loop())
        
        # Start statistics update task
        asyncio.create_task(self._stats_update_loop())
        
        logger.info("Started hardware monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring controllers"""
        self.is_scanning = False
        
        # Disconnect all controllers
        for controller in self.active_controllers:
            await controller.disconnect()
        
        self.active_controllers.clear()
        logger.info("Stopped hardware monitoring")
    
    async def _port_scan_loop(self):
        """Periodic port scanning loop"""
        while self.is_scanning:
            try:
                await self.scan_for_controllers()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Port scan loop error: {e}")
    
    async def _stats_update_loop(self):
        """Update statistics periodically"""
        while self.is_scanning:
            try:
                # Update controller statistics
                for controller in self.active_controllers:
                    controller_type = controller.controller_type.value
                    self.stats["messages_per_controller"][controller_type] = controller.state.total_messages
                    self.stats["last_activity"][controller_type] = controller.state.last_activity
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Stats update error: {e}")
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get current hardware status"""
        return {
            "midi_available": MIDI_AVAILABLE,
            "available_ports": self.available_ports,
            "active_controllers": [
                {
                    "type": controller.controller_type.value,
                    "connected": controller.state.is_connected,
                    "port": controller.state.port_name,
                    "messages": controller.state.total_messages,
                    "last_activity": controller.state.last_activity
                }
                for controller in self.active_controllers
            ],
            "statistics": dict(self.stats),
            "registered_types": list(self.controllers.keys())
        }


# Factory function
def create_hardware_integration(synthesizer: AbstractSynthesizer,
                               performance_engine: Optional[LivePerformanceEngine] = None) -> HardwareMIDIIntegration:
    """Create hardware integration with default controllers"""
    integration = HardwareMIDIIntegration(synthesizer, performance_engine)
    
    # Register default controller types
    integration.register_controller_type(LaunchpadMK1Controller)
    integration.register_controller_type(MIDIFighter3DController)
    
    return integration


if __name__ == "__main__":
    # Demo the hardware integration
    async def demo_hardware():
        from ..interfaces.synthesizer import MockSynthesizer
        from ..performance.live_performance_engine import create_live_performance_engine
        
        # Create dependencies
        synth = MockSynthesizer()
        performance_engine = create_live_performance_engine(synth)
        
        # Create hardware integration
        hardware = create_hardware_integration(synth, performance_engine)
        
        print("=== Hardware MIDI Integration Demo ===")
        
        # Start monitoring
        await hardware.start_monitoring()
        
        # Scan for controllers
        await hardware.scan_for_controllers()
        
        # Try to connect
        connection_results = await hardware.auto_connect_controllers()
        print(f"Connection results: {connection_results}")
        
        # Get status
        status = hardware.get_hardware_status()
        print(f"Hardware status: {status}")
        
        # Run for a bit to see MIDI messages
        print("Listening for MIDI messages for 10 seconds...")
        await asyncio.sleep(10)
        
        # Stop monitoring
        await hardware.stop_monitoring()
        print("Demo completed")
    
    # Run demo
    asyncio.run(demo_hardware())