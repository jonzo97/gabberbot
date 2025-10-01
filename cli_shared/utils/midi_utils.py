#!/usr/bin/env python3
"""
MIDI Utilities for Hardware Controller Integration

Supports hardcore producers' hardware setups:
- Novation Launchpad MK1 (8x8 grid)  
- DJ TechTools MIDI Fighter 3D (16 buttons + motion sensors)
- Novation Launchkey 49 (keys + pads + knobs)
- Novation LaunchControl (16 knobs + 8 pads)
"""

import time
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading

# MIDI support
try:
    import rtmidi
    RTMIDI_AVAILABLE = True
except ImportError:
    RTMIDI_AVAILABLE = False
    print("⚠️  python-rtmidi not installed. Hardware integration disabled.")

class MIDIControllerType(Enum):
    """Supported MIDI controller types"""
    LAUNCHPAD_MK1 = "launchpad_mk1"
    MIDI_FIGHTER_3D = "midi_fighter_3d"  
    LAUNCHKEY_49 = "launchkey_49"
    LAUNCHCONTROL = "launchcontrol"
    GENERIC = "generic"

class MIDIMessageType(Enum):
    """MIDI message types"""
    NOTE_ON = 0x90
    NOTE_OFF = 0x80
    CONTROL_CHANGE = 0xB0
    PITCH_BEND = 0xE0
    AFTERTOUCH = 0xA0

@dataclass
class MIDIMessage:
    """MIDI message data"""
    message_type: MIDIMessageType
    channel: int
    note_or_cc: int
    velocity_or_value: int
    timestamp: float
    
    @classmethod
    def from_rtmidi(cls, message: List[int], timestamp: float) -> 'MIDIMessage':
        """Create from rtmidi message format"""
        status_byte = message[0]
        message_type = MIDIMessageType(status_byte & 0xF0)
        channel = status_byte & 0x0F
        
        note_or_cc = message[1] if len(message) > 1 else 0
        velocity_or_value = message[2] if len(message) > 2 else 0
        
        return cls(message_type, channel, note_or_cc, velocity_or_value, timestamp)

@dataclass
class ControllerMapping:
    """Mapping between MIDI controller and hardcore functions"""
    controller_type: MIDIControllerType
    mappings: Dict[int, str]  # MIDI note/CC -> function name
    led_mappings: Dict[str, int]  # function name -> LED note/CC
    colors: Dict[str, int]  # function -> color code (for RGB controllers)

class LaunchpadMK1:
    """Novation Launchpad MK1 specific implementation"""
    
    # Grid layout (8x8)
    GRID_SIZE = 8
    
    # Color codes for Launchpad MK1
    COLORS = {
        "off": 0,
        "red_low": 1,
        "red_med": 2, 
        "red_high": 3,
        "green_low": 16,
        "green_med": 32,
        "green_high": 48,
        "orange_low": 17,
        "orange_med": 34,
        "orange_high": 51,
        "yellow": 35
    }
    
    @staticmethod
    def xy_to_note(x: int, y: int) -> int:
        """Convert X,Y grid position to MIDI note"""
        if 0 <= x < 8 and 0 <= y < 8:
            return y * 16 + x
        return -1
    
    @staticmethod
    def note_to_xy(note: int) -> Tuple[int, int]:
        """Convert MIDI note to X,Y grid position"""
        y = note // 16
        x = note % 16
        if 0 <= x < 8 and 0 <= y < 8:
            return (x, y)
        return (-1, -1)
    
    @staticmethod
    def create_hardcore_mapping() -> ControllerMapping:
        """Create hardcore-specific mapping for Launchpad MK1"""
        mappings = {}
        led_mappings = {}
        colors = {}
        
        # Top row (Y=0): Transport controls
        mappings[LaunchpadMK1.xy_to_note(0, 0)] = "play_stop"
        mappings[LaunchpadMK1.xy_to_note(1, 0)] = "record"
        mappings[LaunchpadMK1.xy_to_note(2, 0)] = "loop"
        mappings[LaunchpadMK1.xy_to_note(3, 0)] = "metronome"
        
        # Kick drum row (Y=1)
        for x in range(8):
            note = LaunchpadMK1.xy_to_note(x, 1)
            mappings[note] = f"kick_step_{x}"
            led_mappings[f"kick_step_{x}"] = note
            colors[f"kick_step_{x}"] = LaunchpadMK1.COLORS["red_high"]
        
        # Bass row (Y=2)
        for x in range(8):
            note = LaunchpadMK1.xy_to_note(x, 2)
            mappings[note] = f"bass_step_{x}"
            led_mappings[f"bass_step_{x}"] = note
            colors[f"bass_step_{x}"] = LaunchpadMK1.COLORS["green_med"]
        
        # Synth row (Y=3)
        for x in range(8):
            note = LaunchpadMK1.xy_to_note(x, 3)
            mappings[note] = f"synth_step_{x}"
            led_mappings[f"synth_step_{x}"] = note  
            colors[f"synth_step_{x}"] = LaunchpadMK1.COLORS["orange_high"]
        
        # Pattern selection (Y=4-7)
        for y in range(4, 8):
            for x in range(8):
                pattern_num = (y - 4) * 8 + x
                note = LaunchpadMK1.xy_to_note(x, y)
                mappings[note] = f"pattern_{pattern_num}"
                led_mappings[f"pattern_{pattern_num}"] = note
                colors[f"pattern_{pattern_num}"] = LaunchpadMK1.COLORS["yellow"]
        
        return ControllerMapping(
            MIDIControllerType.LAUNCHPAD_MK1,
            mappings,
            led_mappings,
            colors
        )

class MIDIFighter3D:
    """DJ TechTools MIDI Fighter 3D implementation"""
    
    # 4x4 button grid + side buttons
    GRID_SIZE = 4
    
    @staticmethod
    def create_hardcore_mapping() -> ControllerMapping:
        """Create hardcore mapping for MIDI Fighter 3D"""
        mappings = {}
        led_mappings = {}
        colors = {}
        
        # Main 4x4 grid (notes 36-51)
        kick_notes = [36, 37, 38, 39]  # Top row
        for i, note in enumerate(kick_notes):
            mappings[note] = f"kick_variant_{i}"
            led_mappings[f"kick_variant_{i}"] = note
            colors[f"kick_variant_{i}"] = 1  # Red
        
        bass_notes = [40, 41, 42, 43]  # Second row
        for i, note in enumerate(bass_notes):
            mappings[note] = f"bass_variant_{i}" 
            led_mappings[f"bass_variant_{i}"] = note
            colors[f"bass_variant_{i}"] = 2  # Green
        
        # Side buttons for transport
        mappings[48] = "play_stop"
        mappings[49] = "record"
        mappings[50] = "loop"
        mappings[51] = "metronome"
        
        return ControllerMapping(
            MIDIControllerType.MIDI_FIGHTER_3D,
            mappings,
            led_mappings,
            colors
        )

class HardcoreMIDIController:
    """
    MIDI Controller integration for hardcore music production
    
    Features:
    - Multiple controller support
    - Real-time LED feedback
    - Pattern step triggering
    - Transport control
    - Parameter automation
    """
    
    def __init__(self):
        self.midi_in = None
        self.midi_out = None
        self.controller_type = MIDIControllerType.GENERIC
        self.mapping: Optional[ControllerMapping] = None
        self.is_connected = False
        self.input_thread = None
        self.running = False
        
        # Callbacks
        self.on_step_triggered: Optional[Callable[[str, int], None]] = None
        self.on_transport_changed: Optional[Callable[[str], None]] = None
        self.on_pattern_selected: Optional[Callable[[int], None]] = None
        
        # State tracking
        self.active_steps = set()
        self.current_pattern = 0
        self.is_playing = False
        
    def connect(self, device_name: Optional[str] = None) -> bool:
        """Connect to MIDI controller"""
        if not RTMIDI_AVAILABLE:
            print("❌ RTMIDI not available")
            return False
        
        try:
            # Setup MIDI input
            self.midi_in = rtmidi.MidiIn()
            available_ports = self.midi_in.get_ports()
            
            print(f"🎹 Available MIDI ports: {available_ports}")
            
            # Auto-detect or connect to specific device
            port_index = -1
            detected_type = MIDIControllerType.GENERIC
            
            for i, port in enumerate(available_ports):
                port_lower = port.lower()
                if device_name and device_name.lower() in port_lower:
                    port_index = i
                    break
                elif "launchpad" in port_lower:
                    port_index = i
                    detected_type = MIDIControllerType.LAUNCHPAD_MK1
                    break
                elif "fighter" in port_lower:
                    port_index = i
                    detected_type = MIDIControllerType.MIDI_FIGHTER_3D
                    break
                elif "launchkey" in port_lower:
                    port_index = i
                    detected_type = MIDIControllerType.LAUNCHKEY_49
                    break
            
            if port_index >= 0:
                self.midi_in.open_port(port_index)
                self.controller_type = detected_type
                
                # Setup MIDI output for LED control
                self.midi_out = rtmidi.MidiOut()
                out_ports = self.midi_out.get_ports()
                
                # Find corresponding output port
                for i, port in enumerate(out_ports):
                    if available_ports[port_index].split()[0] in port:
                        self.midi_out.open_port(i)
                        break
                
                # Setup controller mapping
                self._setup_controller_mapping()
                
                # Start input monitoring
                self.running = True
                self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
                self.input_thread.start()
                
                self.is_connected = True
                print(f"✅ Connected to {self.controller_type.value}: {available_ports[port_index]}")
                
                # Initialize controller display
                self._initialize_controller()
                
                return True
            
            else:
                print("❌ No supported MIDI controller found")
                return False
                
        except Exception as e:
            print(f"❌ MIDI connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MIDI controller"""
        self.running = False
        self.is_connected = False
        
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
        
        if self.midi_in:
            self.midi_in.close_port()
        
        if self.midi_out:
            self.midi_out.close_port()
        
        print("🔌 MIDI controller disconnected")
    
    def _setup_controller_mapping(self):
        """Setup controller-specific mapping"""
        if self.controller_type == MIDIControllerType.LAUNCHPAD_MK1:
            self.mapping = LaunchpadMK1.create_hardcore_mapping()
        elif self.controller_type == MIDIControllerType.MIDI_FIGHTER_3D:
            self.mapping = MIDIFighter3D.create_hardcore_mapping()
        else:
            # Generic mapping
            self.mapping = ControllerMapping(
                MIDIControllerType.GENERIC,
                {},
                {},
                {}
            )
    
    def _initialize_controller(self):
        """Initialize controller display"""
        if not self.midi_out or not self.mapping:
            return
        
        # Clear all LEDs first
        self._clear_all_leds()
        
        # Light up default pattern
        if self.controller_type == MIDIControllerType.LAUNCHPAD_MK1:
            # Show step grid
            for step in range(8):
                self._set_led(f"kick_step_{step}", False)
                self._set_led(f"bass_step_{step}", False)
                self._set_led(f"synth_step_{step}", False)
    
    def _input_loop(self):
        """MIDI input monitoring loop"""
        while self.running and self.midi_in:
            try:
                message = self.midi_in.get_message()
                if message:
                    midi_msg, timestamp = message
                    self._handle_midi_message(midi_msg, timestamp)
                time.sleep(0.001)  # 1ms polling
            except Exception as e:
                print(f"❌ MIDI input error: {e}")
                break
    
    def _handle_midi_message(self, midi_msg: List[int], timestamp: float):
        """Handle incoming MIDI message"""
        if len(midi_msg) < 2 or not self.mapping:
            return
        
        message = MIDIMessage.from_rtmidi(midi_msg, timestamp)
        
        # Only handle note on messages for now
        if message.message_type != MIDIMessageType.NOTE_ON:
            return
        
        # Look up function mapping
        function = self.mapping.mappings.get(message.note_or_cc)
        if not function:
            return
        
        velocity = message.velocity_or_value
        is_pressed = velocity > 0
        
        print(f"🎹 MIDI: {function} {'pressed' if is_pressed else 'released'} (vel: {velocity})")
        
        # Handle different function types
        if function.startswith("kick_step_"):
            step = int(function.split("_")[-1])
            self._handle_step_trigger("kick", step, is_pressed)
        
        elif function.startswith("bass_step_"):
            step = int(function.split("_")[-1])
            self._handle_step_trigger("bass", step, is_pressed)
        
        elif function.startswith("synth_step_"):
            step = int(function.split("_")[-1])
            self._handle_step_trigger("synth", step, is_pressed)
        
        elif function.startswith("pattern_"):
            pattern_num = int(function.split("_")[-1])
            if is_pressed:
                self._handle_pattern_select(pattern_num)
        
        elif function in ["play_stop", "record", "loop", "metronome"]:
            if is_pressed:
                self._handle_transport(function)
    
    def _handle_step_trigger(self, track: str, step: int, is_pressed: bool):
        """Handle step trigger from controller"""
        step_key = f"{track}_{step}"
        
        if is_pressed:
            if step_key in self.active_steps:
                # Turn off step
                self.active_steps.remove(step_key)
                self._set_led(f"{track}_step_{step}", False)
            else:
                # Turn on step  
                self.active_steps.add(step_key)
                self._set_led(f"{track}_step_{step}", True)
            
            # Trigger callback
            if self.on_step_triggered:
                self.on_step_triggered(track, step)
    
    def _handle_pattern_select(self, pattern_num: int):
        """Handle pattern selection"""
        # Update pattern LEDs
        if self.controller_type == MIDIControllerType.LAUNCHPAD_MK1:
            # Turn off old pattern LED
            old_led = self.mapping.led_mappings.get(f"pattern_{self.current_pattern}")
            if old_led:
                self._send_led_message(old_led, LaunchpadMK1.COLORS["off"])
            
            # Turn on new pattern LED
            new_led = self.mapping.led_mappings.get(f"pattern_{pattern_num}")
            if new_led:
                self._send_led_message(new_led, LaunchpadMK1.COLORS["yellow"])
        
        self.current_pattern = pattern_num
        
        # Trigger callback
        if self.on_pattern_selected:
            self.on_pattern_selected(pattern_num)
    
    def _handle_transport(self, command: str):
        """Handle transport control"""
        if command == "play_stop":
            self.is_playing = not self.is_playing
        
        # Trigger callback
        if self.on_transport_changed:
            self.on_transport_changed(command)
    
    def _set_led(self, function: str, active: bool):
        """Set LED state for a function"""
        if not self.midi_out or not self.mapping:
            return
        
        led_note = self.mapping.led_mappings.get(function)
        if led_note is None:
            return
        
        if active:
            color = self.mapping.colors.get(function, 1)
        else:
            color = 0  # Off
        
        self._send_led_message(led_note, color)
    
    def _send_led_message(self, note: int, color: int):
        """Send LED control message"""
        if self.midi_out:
            message = [0x90, note, color]  # Note on with color as velocity
            self.midi_out.send_message(message)
    
    def _clear_all_leds(self):
        """Clear all controller LEDs"""
        if not self.midi_out:
            return
        
        if self.controller_type == MIDIControllerType.LAUNCHPAD_MK1:
            # Clear main grid (0-127)
            for note in range(128):
                self._send_led_message(note, 0)
    
    def update_step_display(self, track: str, step: int, active: bool):
        """Update step LED display"""
        function = f"{track}_step_{step}"
        self._set_led(function, active)
    
    def update_transport_display(self, is_playing: bool, is_recording: bool = False):
        """Update transport LEDs"""
        if self.controller_type == MIDIControllerType.LAUNCHPAD_MK1:
            play_color = LaunchpadMK1.COLORS["green_high"] if is_playing else LaunchpadMK1.COLORS["green_low"]
            record_color = LaunchpadMK1.COLORS["red_high"] if is_recording else LaunchpadMK1.COLORS["off"]
            
            play_led = self.mapping.led_mappings.get("play_stop")
            record_led = self.mapping.led_mappings.get("record")
            
            if play_led:
                self._send_led_message(play_led, play_color)
            if record_led:
                self._send_led_message(record_led, record_color)
    
    def set_step_callback(self, callback: Callable[[str, int], None]):
        """Set callback for step triggers"""
        self.on_step_triggered = callback
    
    def set_transport_callback(self, callback: Callable[[str], None]):
        """Set callback for transport changes"""
        self.on_transport_changed = callback
    
    def set_pattern_callback(self, callback: Callable[[int], None]):
        """Set callback for pattern selection"""
        self.on_pattern_selected = callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            "connected": self.is_connected,
            "controller_type": self.controller_type.value if self.controller_type else "none",
            "active_steps": len(self.active_steps),
            "current_pattern": self.current_pattern,
            "is_playing": self.is_playing
        }

# Utility functions
def scan_midi_devices() -> List[Dict[str, Any]]:
    """Scan for available MIDI devices"""
    devices = []
    
    if not RTMIDI_AVAILABLE:
        return devices
    
    try:
        midi_in = rtmidi.MidiIn()
        input_ports = midi_in.get_ports()
        
        midi_out = rtmidi.MidiOut()  
        output_ports = midi_out.get_ports()
        
        for i, port in enumerate(input_ports):
            device_info = {
                "name": port,
                "input_port": i,
                "output_port": None,
                "type": "generic"
            }
            
            # Try to find matching output port
            for j, out_port in enumerate(output_ports):
                if port.split()[0] in out_port:
                    device_info["output_port"] = j
                    break
            
            # Detect controller type
            port_lower = port.lower()
            if "launchpad" in port_lower:
                device_info["type"] = "launchpad_mk1"
            elif "fighter" in port_lower:
                device_info["type"] = "midi_fighter_3d"
            elif "launchkey" in port_lower:
                device_info["type"] = "launchkey_49"
            elif "launchcontrol" in port_lower:
                device_info["type"] = "launchcontrol"
            
            devices.append(device_info)
        
        midi_in.close()
        midi_out.close()
        
    except Exception as e:
        print(f"❌ Error scanning MIDI devices: {e}")
    
    return devices

def test_midi_controller():
    """Test MIDI controller functionality"""
    print("🎹 Testing MIDI Controller Integration")
    print("=" * 50)
    
    # Scan for devices
    devices = scan_midi_devices()
    print(f"Found {len(devices)} MIDI devices:")
    for device in devices:
        print(f"  📱 {device['name']} ({device['type']})")
    
    if not devices:
        print("❌ No MIDI devices found")
        return
    
    # Connect to first device
    controller = HardcoreMIDIController()
    
    def on_step(track: str, step: int):
        print(f"🥁 Step triggered: {track} step {step}")
    
    def on_transport(command: str):
        print(f"⏯️ Transport: {command}")
    
    def on_pattern(pattern: int):
        print(f"🎵 Pattern selected: {pattern}")
    
    controller.set_step_callback(on_step)
    controller.set_transport_callback(on_transport)
    controller.set_pattern_callback(on_pattern)
    
    success = controller.connect()
    
    if success:
        print("✅ Controller connected! Press buttons to test...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(0.1)
                
                # Update some LEDs periodically for testing
                if controller.controller_type == MIDIControllerType.LAUNCHPAD_MK1:
                    step = int(time.time() * 2) % 8
                    controller.update_step_display("kick", step, True)
                    if step > 0:
                        controller.update_step_display("kick", step - 1, False)
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopping test")
        
        finally:
            controller.disconnect()
    
    else:
        print("❌ Failed to connect to controller")

if __name__ == "__main__":
    test_midi_controller()