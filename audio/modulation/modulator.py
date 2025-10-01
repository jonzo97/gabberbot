#!/usr/bin/env python3
"""
Modulation System - Flexible parameter control with LFOs, envelopes, and automation
Designed for reusability across synthesizers and effects
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class WaveformType(Enum):
    """Waveform types for oscillators and LFOs"""
    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    SAW = "sawtooth"
    REVERSE_SAW = "reverse_sawtooth"
    NOISE = "noise"
    SAMPLE_HOLD = "sample_hold"
    STEP = "step"
    CUSTOM = "custom"


class SyncMode(Enum):
    """Synchronization modes for tempo-synced modulation"""
    FREE = "free"          # Free-running, not synced
    SYNC = "sync"          # Synced to tempo
    ONE_SHOT = "one_shot"  # Triggered once, then free-running


class InterpolationMode(Enum):
    """Interpolation modes for automation"""
    LINEAR = "linear"
    SMOOTH = "smooth"     # Cubic spline
    STEP = "step"         # No interpolation
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"


@dataclass
class ModulationPoint:
    """Single point in automation curve"""
    time: float      # Time in beats or seconds
    value: float     # Parameter value (0.0-1.0)
    curve: float = 0.0  # Curve amount (-1.0 to 1.0)


class ModulationSource(ABC):
    """Abstract base class for all modulation sources"""
    
    def __init__(self, name: str = "mod_source"):
        self.name = name
        self.enabled = True
        self.output_range = (0.0, 1.0)  # Min, max output
        self.bipolar = False  # True for -1.0 to 1.0, False for 0.0 to 1.0
        
    @abstractmethod
    def get_value(self, time: float, sample_rate: float) -> float:
        """Get modulation value at given time"""
        pass
    
    def get_scaled_value(self, time: float, sample_rate: float) -> float:
        """Get value scaled to output range"""
        if not self.enabled:
            return 0.0
            
        value = self.get_value(time, sample_rate)
        
        # Scale from internal range to output range
        if self.bipolar:
            # -1.0 to 1.0 -> output_range
            scaled = ((value + 1.0) / 2.0) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        else:
            # 0.0 to 1.0 -> output_range
            scaled = value * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        
        return scaled
    
    def set_range(self, min_val: float, max_val: float):
        """Set output range"""
        self.output_range = (min_val, max_val)


class LFO(ModulationSource):
    """Low Frequency Oscillator for cyclic modulation"""
    
    def __init__(self, 
                 rate: float = 1.0,
                 waveform: WaveformType = WaveformType.SINE,
                 phase: float = 0.0,
                 sync_mode: SyncMode = SyncMode.FREE,
                 name: str = "lfo"):
        super().__init__(name)
        
        self.rate = rate  # Hz or beat division
        self.waveform = waveform
        self.phase = phase  # 0.0-1.0
        self.sync_mode = sync_mode
        self.depth = 1.0  # Modulation depth
        
        # For sample & hold
        self.last_random_value = 0.0
        self.last_sample_time = 0.0
        
        # For custom waveforms
        self.custom_waveform: Optional[np.ndarray] = None
        
        # Tempo sync
        self.tempo_bpm = 120.0
        self.beat_divisions = {
            "1/32": 32, "1/16": 16, "1/8": 8, "1/4": 4,
            "1/2": 2, "1": 1, "2": 0.5, "4": 0.25
        }
    
    def set_tempo(self, bpm: float):
        """Set tempo for sync mode"""
        self.tempo_bpm = bpm
    
    def set_beat_division(self, division: str):
        """Set rate as beat division (e.g., '1/4' for quarter note)"""
        if division in self.beat_divisions:
            beats_per_second = self.tempo_bpm / 60.0
            self.rate = beats_per_second * self.beat_divisions[division]
            self.sync_mode = SyncMode.SYNC
    
    def get_value(self, time: float, sample_rate: float) -> float:
        """Generate LFO value"""
        # Calculate phase position
        if self.sync_mode == SyncMode.SYNC:
            # Use musical time (beats)
            beat_time = time * (self.tempo_bpm / 60.0)
            phase_pos = (beat_time * self.rate + self.phase) % 1.0
        else:
            # Free running time
            phase_pos = (time * self.rate + self.phase) % 1.0
        
        # Generate waveform
        value = self._generate_waveform(phase_pos, time)
        
        return value * self.depth
    
    def _generate_waveform(self, phase: float, time: float) -> float:
        """Generate waveform at given phase (0.0-1.0)"""
        if self.waveform == WaveformType.SINE:
            return np.sin(2 * np.pi * phase)
        
        elif self.waveform == WaveformType.TRIANGLE:
            if phase < 0.5:
                return (4 * phase) - 1  # -1 to 1
            else:
                return 3 - (4 * phase)  # 1 to -1
        
        elif self.waveform == WaveformType.SQUARE:
            return 1.0 if phase < 0.5 else -1.0
        
        elif self.waveform == WaveformType.SAW:
            return (2 * phase) - 1  # -1 to 1
        
        elif self.waveform == WaveformType.REVERSE_SAW:
            return 1 - (2 * phase)  # 1 to -1
        
        elif self.waveform == WaveformType.NOISE:
            return np.random.uniform(-1, 1)
        
        elif self.waveform == WaveformType.SAMPLE_HOLD:
            # Sample and hold - update value at each cycle
            if time - self.last_sample_time >= (1.0 / self.rate):
                self.last_random_value = np.random.uniform(-1, 1)
                self.last_sample_time = time
            return self.last_random_value
        
        elif self.waveform == WaveformType.STEP:
            # 8-step sequencer-style
            step = int(phase * 8) % 8
            step_values = [-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5]
            return step_values[step]
        
        elif self.waveform == WaveformType.CUSTOM and self.custom_waveform is not None:
            # Interpolate through custom waveform
            index = phase * (len(self.custom_waveform) - 1)
            idx = int(index)
            frac = index - idx
            
            if idx >= len(self.custom_waveform) - 1:
                return self.custom_waveform[-1]
            
            # Linear interpolation
            return self.custom_waveform[idx] * (1 - frac) + self.custom_waveform[idx + 1] * frac
        
        return 0.0


class Envelope(ModulationSource):
    """ADSR Envelope generator"""
    
    def __init__(self,
                 attack: float = 0.1,
                 decay: float = 0.2,
                 sustain: float = 0.7,
                 release: float = 0.5,
                 name: str = "envelope"):
        super().__init__(name)
        
        self.attack = attack    # Attack time in seconds
        self.decay = decay      # Decay time in seconds
        self.sustain = sustain  # Sustain level (0.0-1.0)
        self.release = release  # Release time in seconds
        
        self.trigger_time = 0.0
        self.release_time = None
        self.triggered = False
        
        # Envelope curve types
        self.attack_curve = 0.5   # 0=linear, <0.5=exponential, >0.5=logarithmic
        self.decay_curve = 0.5
        self.release_curve = 0.5
    
    def trigger(self, time: float):
        """Trigger envelope"""
        self.trigger_time = time
        self.release_time = None
        self.triggered = True
    
    def release(self, time: float):
        """Release envelope"""
        if self.triggered:
            self.release_time = time
    
    def get_value(self, time: float, sample_rate: float) -> float:
        """Calculate envelope value"""
        if not self.triggered:
            return 0.0
        
        elapsed = time - self.trigger_time
        
        if self.release_time is None:
            # Attack/Decay/Sustain phase
            if elapsed < self.attack:
                # Attack phase
                progress = elapsed / self.attack if self.attack > 0 else 1.0
                return self._apply_curve(progress, self.attack_curve)
            
            elif elapsed < self.attack + self.decay:
                # Decay phase
                decay_elapsed = elapsed - self.attack
                progress = decay_elapsed / self.decay if self.decay > 0 else 1.0
                decay_amount = (1.0 - self.sustain) * self._apply_curve(progress, self.decay_curve)
                return 1.0 - decay_amount
            
            else:
                # Sustain phase
                return self.sustain
        
        else:
            # Release phase
            release_elapsed = time - self.release_time
            if release_elapsed >= self.release:
                self.triggered = False
                return 0.0
            
            progress = release_elapsed / self.release if self.release > 0 else 1.0
            release_amount = self.sustain * self._apply_curve(progress, self.release_curve)
            return self.sustain - release_amount
    
    def _apply_curve(self, progress: float, curve: float) -> float:
        """Apply curve to linear progress"""
        if curve == 0.5:
            return progress  # Linear
        elif curve < 0.5:
            # Exponential (fast start)
            exp_factor = (0.5 - curve) * 10
            return 1 - np.exp(-exp_factor * progress)
        else:
            # Logarithmic (slow start)
            log_factor = (curve - 0.5) * 10
            return np.log(1 + log_factor * progress) / np.log(1 + log_factor)


class AutomationLane(ModulationSource):
    """Automation lane with keyframes"""
    
    def __init__(self, name: str = "automation"):
        super().__init__(name)
        
        self.points: List[ModulationPoint] = []
        self.interpolation = InterpolationMode.LINEAR
        self.loop_enabled = False
        self.loop_start = 0.0
        self.loop_end = 4.0  # 4 beats
        
        # Default points (flat line at 0.5)
        self.points = [
            ModulationPoint(0.0, 0.5),
            ModulationPoint(4.0, 0.5)
        ]
    
    def add_point(self, time: float, value: float, curve: float = 0.0):
        """Add automation point"""
        point = ModulationPoint(time, value, curve)
        
        # Insert in correct position
        inserted = False
        for i, existing in enumerate(self.points):
            if existing.time > time:
                self.points.insert(i, point)
                inserted = True
                break
        
        if not inserted:
            self.points.append(point)
    
    def remove_point(self, index: int):
        """Remove automation point"""
        if 0 <= index < len(self.points):
            self.points.pop(index)
    
    def get_value(self, time: float, sample_rate: float) -> float:
        """Get interpolated value at time"""
        if not self.points:
            return 0.0
        
        # Handle looping
        if self.loop_enabled:
            loop_duration = self.loop_end - self.loop_start
            if loop_duration > 0:
                if time >= self.loop_end:
                    time = self.loop_start + ((time - self.loop_start) % loop_duration)
        
        # Find surrounding points
        if time <= self.points[0].time:
            return self.points[0].value
        
        if time >= self.points[-1].time:
            return self.points[-1].value
        
        # Find interpolation range
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            if p1.time <= time <= p2.time:
                return self._interpolate(p1, p2, time)
        
        return 0.0
    
    def _interpolate(self, p1: ModulationPoint, p2: ModulationPoint, time: float) -> float:
        """Interpolate between two points"""
        if p2.time - p1.time <= 0:
            return p1.value
        
        # Normalize time to 0.0-1.0
        progress = (time - p1.time) / (p2.time - p1.time)
        
        if self.interpolation == InterpolationMode.STEP:
            return p1.value
        
        elif self.interpolation == InterpolationMode.LINEAR:
            return p1.value + (p2.value - p1.value) * progress
        
        elif self.interpolation == InterpolationMode.SMOOTH:
            # Cubic interpolation with curve control
            curve = (p1.curve + p2.curve) / 2.0
            t2 = progress * progress
            t3 = t2 * progress
            
            # Hermite interpolation
            return (2*t3 - 3*t2 + 1) * p1.value + (t3 - 2*t2 + progress) * curve + \
                   (-2*t3 + 3*t2) * p2.value + (t3 - t2) * curve
        
        elif self.interpolation == InterpolationMode.EXPONENTIAL:
            exp_progress = 1 - np.exp(-5 * progress)
            return p1.value + (p2.value - p1.value) * exp_progress
        
        elif self.interpolation == InterpolationMode.LOGARITHMIC:
            log_progress = np.log(1 + 4 * progress) / np.log(5)
            return p1.value + (p2.value - p1.value) * log_progress
        
        return p1.value


@dataclass
class ModulationConnection:
    """Connection between modulation source and parameter"""
    source: ModulationSource
    parameter_name: str
    amount: float = 1.0          # Modulation amount (-1.0 to 1.0)
    offset: float = 0.0          # DC offset
    enabled: bool = True


class ModulationRouter:
    """Routes modulation sources to parameters"""
    
    def __init__(self):
        self.connections: List[ModulationConnection] = []
        self.parameters: Dict[str, float] = {}  # Current parameter values
        
    def connect(self, 
               source: ModulationSource,
               parameter: str,
               amount: float = 1.0,
               offset: float = 0.0) -> ModulationConnection:
        """Connect modulation source to parameter"""
        connection = ModulationConnection(source, parameter, amount, offset)
        self.connections.append(connection)
        return connection
    
    def disconnect(self, connection: ModulationConnection):
        """Remove modulation connection"""
        if connection in self.connections:
            self.connections.remove(connection)
    
    def update(self, time: float, sample_rate: float) -> Dict[str, float]:
        """Update all modulated parameters"""
        # Start with base parameter values
        modulated_params = self.parameters.copy()
        
        # Apply modulation from all connections
        for connection in self.connections:
            if not connection.enabled:
                continue
                
            # Get modulation value
            mod_value = connection.source.get_scaled_value(time, sample_rate)
            
            # Apply amount and offset
            final_mod = (mod_value * connection.amount) + connection.offset
            
            # Add to parameter (or replace if parameter doesn't exist yet)
            if connection.parameter_name in modulated_params:
                modulated_params[connection.parameter_name] += final_mod
            else:
                modulated_params[connection.parameter_name] = final_mod
            
            # Clamp to 0-1 range (or allow bipolar if needed)
            modulated_params[connection.parameter_name] = np.clip(
                modulated_params[connection.parameter_name], 0.0, 1.0
            )
        
        return modulated_params
    
    def set_parameter(self, name: str, value: float):
        """Set base parameter value"""
        self.parameters[name] = value
    
    def get_connections_for_parameter(self, parameter: str) -> List[ModulationConnection]:
        """Get all connections for a parameter"""
        return [conn for conn in self.connections if conn.parameter_name == parameter]


# Preset modulation patterns for common uses
class SidechainCompressor(ModulationSource):
    """Specialized modulation source for sidechain pumping"""
    
    def __init__(self, 
                 pump_rate: str = "1/4",  # Beat division
                 pump_amount: float = 0.8,
                 attack: float = 0.01,
                 release: float = 0.2,
                 name: str = "sidechain"):
        super().__init__(name)
        
        self.pump_rate = pump_rate
        self.pump_amount = pump_amount
        self.attack = attack
        self.release = release
        self.tempo_bpm = 120.0
        
        # Beat divisions
        self.beat_divisions = {
            "1/32": 32, "1/16": 16, "1/8": 8, "1/4": 4,
            "1/2": 2, "1": 1
        }
    
    def set_tempo(self, bpm: float):
        """Set tempo"""
        self.tempo_bpm = bpm
    
    def get_value(self, time: float, sample_rate: float) -> float:
        """Generate sidechain pump pattern"""
        # Calculate beat time
        beats_per_second = self.tempo_bpm / 60.0
        beat_time = time * beats_per_second
        
        # Get pump frequency
        if self.pump_rate in self.beat_divisions:
            pump_freq = beats_per_second * self.beat_divisions[self.pump_rate]
        else:
            pump_freq = beats_per_second * 4  # Default to 1/4 note
        
        # Calculate phase
        phase = (beat_time * pump_freq) % 1.0
        
        # Generate pump envelope
        if phase < self.attack:
            # Attack phase (duck down)
            progress = phase / self.attack if self.attack > 0 else 1.0
            return 1.0 - (self.pump_amount * progress)
        else:
            # Release phase (come back up)
            release_phase = (phase - self.attack) / (1.0 - self.attack)
            release_progress = min(1.0, release_phase / self.release) if self.release > 0 else 1.0
            ducked_amount = self.pump_amount * (1.0 - release_progress)
            return 1.0 - ducked_amount


def create_hardcore_sidechain(tempo_bpm: float = 180.0) -> SidechainCompressor:
    """Create sidechain modulator optimized for hardcore"""
    sidechain = SidechainCompressor(
        pump_rate="1/4",
        pump_amount=0.6,
        attack=0.005,
        release=0.15,
        name="hardcore_pump"
    )
    sidechain.set_tempo(tempo_bpm)
    return sidechain


def create_acid_filter_lfo(tempo_bpm: float = 180.0) -> LFO:
    """Create LFO optimized for acid filter sweeps"""
    lfo = LFO(
        waveform=WaveformType.SAW,
        sync_mode=SyncMode.SYNC,
        name="acid_filter"
    )
    lfo.set_tempo(tempo_bpm)
    lfo.set_beat_division("1/16")
    lfo.bipolar = False
    lfo.depth = 0.8
    return lfo


def create_tremolo_lfo(rate: float = 8.0) -> LFO:
    """Create tremolo LFO"""
    return LFO(
        rate=rate,
        waveform=WaveformType.TRIANGLE,
        depth=0.5,
        name="tremolo"
    )


# Testing
if __name__ == "__main__":
    print("🌊 Testing Modulation System")
    print("=" * 40)
    
    # Create LFO
    lfo = LFO(rate=2.0, waveform=WaveformType.SINE)
    lfo.set_range(0.2, 0.8)
    
    # Create envelope
    env = Envelope(attack=0.1, decay=0.2, sustain=0.6, release=0.3)
    env.trigger(0.0)
    
    # Create automation
    auto = AutomationLane()
    auto.add_point(0.0, 0.0)
    auto.add_point(2.0, 1.0)
    auto.add_point(4.0, 0.5)
    
    # Create router
    router = ModulationRouter()
    router.set_parameter("filter_cutoff", 0.5)
    router.set_parameter("volume", 0.8)
    
    # Connect modulations
    router.connect(lfo, "filter_cutoff", amount=0.3)
    router.connect(env, "volume", amount=1.0)
    
    # Test over time
    sample_rate = 44100
    for i in range(5):
        time_sec = i * 0.5
        params = router.update(time_sec, sample_rate)
        
        print(f"\nTime {time_sec:.1f}s:")
        print(f"  LFO: {lfo.get_scaled_value(time_sec, sample_rate):.3f}")
        print(f"  Env: {env.get_value(time_sec, sample_rate):.3f}")
        print(f"  Auto: {auto.get_value(time_sec, sample_rate):.3f}")
        print(f"  Filter: {params.get('filter_cutoff', 0):.3f}")
        print(f"  Volume: {params.get('volume', 0):.3f}")
    
    # Test sidechain
    print(f"\n🔄 Sidechain Test:")
    sidechain = create_hardcore_sidechain(180.0)
    for i in range(8):
        beat = i * 0.25  # Quarter note steps
        value = sidechain.get_value(beat * (60/180), sample_rate)
        print(f"  Beat {beat}: {value:.3f}")
    
    print("\n✅ Modulation system test complete!")