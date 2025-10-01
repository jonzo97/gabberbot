#!/usr/bin/env python3
"""
Shared Data Models for Hardcore Music Production

Common data structures used by both Strudel and SuperCollider implementations.
Ensures consistency and interoperability between backends.
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

class SynthType(Enum):
    """Types of hardcore synthesizers"""
    GABBER_KICK = "gabber_kick"
    INDUSTRIAL_KICK = "industrial_kick" 
    RAWSTYLE_KICK = "rawstyle_kick"
    PVC_KICK = "pvc_kick"
    ZAAG_KICK = "zaag_kick"
    EARTHQUAKE_KICK = "earthquake_kick"
    PIEP_KICK = "piep_kick"
    HOOVER_SYNTH = "hoover_synth"
    ACID_BASS = "acid_bass"
    SCREECH_LEAD = "screech_lead"
    HARDCORE_STAB = "hardcore_stab"
    INDUSTRIAL_NOISE = "industrial_noise"
    DISTORTED_BELL = "distorted_bell"
    SUPERSAW = "supersaw"
    SCREECHE = "screeche"

class PatternEventType(Enum):
    """Types of pattern events"""
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    PARAMETER_CHANGE = "parameter_change"
    EFFECT_TRIGGER = "effect_trigger"
    TEMPO_CHANGE = "tempo_change"

@dataclass
class SynthParams:
    """Parameters for hardcore synthesizers"""
    # Core synthesis parameters
    freq: float = 220.0                # Fundamental frequency (Hz)
    amp: float = 0.8                   # Amplitude (0-1)
    
    # Envelope parameters
    attack: float = 0.01               # Attack time (seconds)
    decay: float = 0.1                 # Decay time (seconds)
    sustain: float = 0.7               # Sustain level (0-1)
    release: float = 0.3               # Release time (seconds)
    
    # Hardcore-specific parameters
    crunch: float = 0.5                # Crunch/distortion factor (0-1)
    drive: float = 1.0                 # Overdrive amount (1-10)
    
    # Filter parameters
    cutoff: float = 8000.0             # Filter cutoff frequency (Hz)
    resonance: float = 0.3             # Filter resonance (0-1)
    filter_type: str = "lowpass"       # Filter type
    
    # Modulation parameters
    mod_index: float = 1.0             # FM modulation index
    mod_freq: float = 1.0              # Modulation frequency ratio
    
    # Hardcore processing chain parameters
    doorlussen: float = 0.0            # Serial distortion chain intensity (0-1)
    rumble: float = 0.0                # Industrial rumble amount (0-1)
    click: float = 0.0                 # High-frequency click amount (0-1)
    
    # Additional parameters (backend-specific)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynthParams':
        """Create from dictionary"""
        extra = data.pop('extra_params', {})
        params = cls(**data)
        params.extra_params = extra
        return params
    
    def copy(self) -> 'SynthParams':
        """Create a copy of the parameters"""
        return SynthParams.from_dict(self.to_dict())
    
    def scale_hardcore_params(self, intensity: float):
        """Scale hardcore-specific parameters by intensity (0-1)"""
        self.crunch = min(1.0, self.crunch * (1 + intensity))
        self.drive = min(10.0, self.drive * (1 + intensity * 2))
        self.doorlussen = min(1.0, self.doorlussen * (1 + intensity))

@dataclass
class PatternStep:
    """Single step in a hardcore pattern"""
    synth_type: SynthType
    params: SynthParams
    velocity: float = 1.0              # Note velocity (0-1)
    duration: float = 0.25             # Step duration in beats
    probability: float = 1.0           # Trigger probability (0-1)
    swing: float = 0.0                 # Swing offset (-1 to 1)
    
    # Automation
    param_automation: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "synth_type": self.synth_type.value,
            "params": self.params.to_dict(),
            "velocity": self.velocity,
            "duration": self.duration,
            "probability": self.probability,
            "swing": self.swing,
            "param_automation": self.param_automation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternStep':
        """Create from dictionary"""
        data["synth_type"] = SynthType(data["synth_type"])
        data["params"] = SynthParams.from_dict(data["params"])
        return cls(**data)

@dataclass
class HardcoreTrack:
    """A single track in a hardcore pattern"""
    name: str
    steps: List[Optional[PatternStep]]
    muted: bool = False
    solo: bool = False
    volume: float = 1.0
    pan: float = 0.0                   # -1 (left) to 1 (right)
    
    # Effects chain
    effects: List[Dict[str, Any]] = field(default_factory=list)
    
    def set_step(self, step_index: int, pattern_step: Optional[PatternStep]):
        """Set a step in the track"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index] = pattern_step
    
    def get_step(self, step_index: int) -> Optional[PatternStep]:
        """Get a step from the track"""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def clear_step(self, step_index: int):
        """Clear a step in the track"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index] = None
    
    def get_active_steps(self) -> List[Tuple[int, PatternStep]]:
        """Get all active (non-None) steps"""
        return [(i, step) for i, step in enumerate(self.steps) if step is not None]

@dataclass
class HardcorePattern:
    """Complete hardcore music pattern with multiple tracks"""
    name: str
    bpm: float = 170.0                 # Typical hardcore BPM
    steps: int = 16                    # Pattern length in steps
    bars: int = 1                      # Number of bars
    time_signature: Tuple[int, int] = (4, 4)  # Time signature
    
    # Pattern tracks
    tracks: Dict[str, HardcoreTrack] = field(default_factory=dict)
    
    # Pattern code/data for synthesis backends
    pattern_data: Optional[str] = None  # Strudel/SuperCollider code
    synth_type: Optional['SynthType'] = None  # Primary synth type for pattern
    
    # Global pattern settings
    swing: float = 0.0                 # Global swing (-1 to 1)
    shuffle: float = 0.0               # Shuffle amount (0-1)
    
    # Arrangement
    loop: bool = True
    loop_count: int = -1               # -1 = infinite loop
    
    # Metadata
    genre: str = "hardcore"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    def add_track(self, track_name: str) -> HardcoreTrack:
        """Add a new track to the pattern"""
        track = HardcoreTrack(
            name=track_name,
            steps=[None] * self.steps
        )
        self.tracks[track_name] = track
        self.modified_at = time.time()
        return track
    
    def remove_track(self, track_name: str) -> bool:
        """Remove a track from the pattern"""
        if track_name in self.tracks:
            del self.tracks[track_name]
            self.modified_at = time.time()
            return True
        return False
    
    def get_track(self, track_name: str) -> Optional[HardcoreTrack]:
        """Get a track by name"""
        return self.tracks.get(track_name)
    
    def set_step(self, track_name: str, step_index: int, pattern_step: Optional[PatternStep]):
        """Set a step in a specific track"""
        if track_name not in self.tracks:
            self.add_track(track_name)
        self.tracks[track_name].set_step(step_index, pattern_step)
        self.modified_at = time.time()
    
    def get_step(self, track_name: str, step_index: int) -> Optional[PatternStep]:
        """Get a step from a specific track"""
        track = self.get_track(track_name)
        if track:
            return track.get_step(step_index)
        return None
    
    def get_step_events(self, step_index: int) -> List[Tuple[str, PatternStep]]:
        """Get all events for a specific step across all tracks"""
        events = []
        for track_name, track in self.tracks.items():
            if not track.muted:
                step = track.get_step(step_index)
                if step:
                    events.append((track_name, step))
        return events
    
    def get_pattern_duration(self) -> float:
        """Get total pattern duration in seconds"""
        beats_per_bar = self.time_signature[0]
        total_beats = (self.steps / 4) * self.bars  # Assuming 16th note steps
        return (total_beats / self.bpm) * 60
    
    def clone(self, new_name: str) -> 'HardcorePattern':
        """Create a copy of the pattern with a new name"""
        pattern_dict = self.to_dict()
        pattern_dict['name'] = new_name
        pattern_dict['created_at'] = time.time()
        pattern_dict['modified_at'] = time.time()
        return HardcorePattern.from_dict(pattern_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization"""
        return {
            "name": self.name,
            "bpm": self.bpm,
            "steps": self.steps,
            "bars": self.bars,
            "time_signature": self.time_signature,
            "tracks": {
                name: {
                    "name": track.name,
                    "steps": [step.to_dict() if step else None for step in track.steps],
                    "muted": track.muted,
                    "solo": track.solo,
                    "volume": track.volume,
                    "pan": track.pan,
                    "effects": track.effects
                }
                for name, track in self.tracks.items()
            },
            "swing": self.swing,
            "shuffle": self.shuffle,
            "loop": self.loop,
            "loop_count": self.loop_count,
            "genre": self.genre,
            "tags": self.tags,
            "created_at": self.created_at,
            "modified_at": self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardcorePattern':
        """Create pattern from dictionary"""
        tracks_data = data.pop('tracks', {})
        pattern = cls(**data)
        
        # Reconstruct tracks
        for track_name, track_data in tracks_data.items():
            steps_data = track_data.pop('steps', [])
            track = HardcoreTrack(**track_data)
            
            # Reconstruct steps
            track.steps = [
                PatternStep.from_dict(step_data) if step_data else None
                for step_data in steps_data
            ]
            
            pattern.tracks[track_name] = track
        
        return pattern
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save pattern to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'HardcorePattern':
        """Load pattern from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

@dataclass
class AudioAnalysisResult:
    """Results from audio analysis"""
    timestamp: float = field(default_factory=time.time)
    
    # Basic metrics
    rms: float = 0.0                   # RMS level
    peak: float = 0.0                  # Peak level
    lufs: float = -23.0                # Loudness (LUFS)
    
    # Frequency analysis
    spectral_centroid: float = 0.0     # Brightness measure
    spectral_rolloff: float = 0.0      # 85% energy point
    spectral_flux: float = 0.0         # Rate of change
    
    # Kick drum analysis (hardcore-specific)
    kick_detected: bool = False
    kick_frequency: float = 0.0        # Fundamental frequency
    kick_punch: float = 0.0            # Attack/body ratio
    kick_weight: float = 0.0           # Low-end weight
    
    # Frequency band energies
    band_energies: Dict[str, float] = field(default_factory=dict)
    
    # Distortion analysis
    thd: float = 0.0                   # Total Harmonic Distortion
    crunch_factor: float = 0.0         # Digital artifacts measure
    
    # Dynamic range
    dynamic_range: float = 0.0         # Peak to RMS ratio
    crest_factor: float = 0.0          # Peak to average ratio

@dataclass
class SessionState:
    """Current session state"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    
    # Current playback state
    is_playing: bool = False
    current_pattern: Optional[str] = None  # Pattern name
    current_step: int = 0
    current_bpm: float = 170.0
    
    # Loaded patterns
    patterns: Dict[str, HardcorePattern] = field(default_factory=dict)
    
    # Session settings
    master_volume: float = 0.8
    metronome_enabled: bool = False
    recording: bool = False
    
    # Performance stats
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    audio_dropouts: int = 0
    
    def add_pattern(self, pattern: HardcorePattern):
        """Add a pattern to the session"""
        self.patterns[pattern.name] = pattern
    
    def get_pattern(self, name: str) -> Optional[HardcorePattern]:
        """Get a pattern by name"""
        return self.patterns.get(name)
    
    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern from the session"""
        if name in self.patterns:
            del self.patterns[name]
            if self.current_pattern == name:
                self.current_pattern = None
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "is_playing": self.is_playing,
            "current_pattern": self.current_pattern,
            "current_step": self.current_step,
            "current_bpm": self.current_bpm,
            "patterns": {name: pattern.to_dict() for name, pattern in self.patterns.items()},
            "master_volume": self.master_volume,
            "metronome_enabled": self.metronome_enabled,
            "recording": self.recording,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "audio_dropouts": self.audio_dropouts
        }

# Utility functions for creating common hardcore patterns

def create_gabber_kick_pattern(name: str = "Gabber Kick", bpm: float = 180.0) -> HardcorePattern:
    """Create a classic gabber kick pattern"""
    pattern = HardcorePattern(name=name, bpm=bpm, steps=16)
    
    # Add kick track with 4/4 pattern
    kick_track = pattern.add_track("kick")
    kick_params = SynthParams(
        freq=60.0,
        amp=0.9,
        crunch=0.8,
        drive=2.5,
        doorlussen=0.7,
        attack=0.001,
        decay=0.08,
        sustain=0.3,
        release=0.2
    )
    
    # Classic 4/4 gabber kick
    kick_step = PatternStep(SynthType.GABBER_KICK, kick_params)
    for step in [0, 4, 8, 12]:
        pattern.set_step("kick", step, kick_step)
    
    return pattern

def create_industrial_pattern(name: str = "Industrial", bpm: float = 140.0) -> HardcorePattern:
    """Create an industrial techno pattern"""
    pattern = HardcorePattern(name=name, bpm=bpm, steps=16)
    
    # Industrial kick
    kick_track = pattern.add_track("kick")
    kick_params = SynthParams(
        freq=45.0,
        amp=0.85,
        crunch=0.6,
        drive=1.8,
        rumble=0.8,
        attack=0.002,
        decay=0.15,
        sustain=0.2,
        release=0.4
    )
    kick_step = PatternStep(SynthType.INDUSTRIAL_KICK, kick_params)
    pattern.set_step("kick", 0, kick_step)
    pattern.set_step("kick", 8, kick_step)
    
    # Industrial noise hits
    noise_track = pattern.add_track("noise")
    noise_params = SynthParams(
        freq=1000.0,
        amp=0.4,
        crunch=0.9,
        drive=3.0,
        attack=0.001,
        decay=0.05,
        sustain=0.1,
        release=0.1
    )
    noise_step = PatternStep(SynthType.INDUSTRIAL_NOISE, noise_params, duration=0.125)
    pattern.set_step("noise", 4, noise_step)
    pattern.set_step("noise", 12, noise_step)
    
    return pattern

def create_acid_pattern(name: str = "Acid Hardcore", bpm: float = 175.0) -> HardcorePattern:
    """Create an acid hardcore pattern"""
    pattern = HardcorePattern(name=name, bpm=bpm, steps=16)
    
    # Kick
    kick_track = pattern.add_track("kick")
    kick_params = SynthParams(freq=65.0, amp=0.8, crunch=0.7, drive=2.0)
    kick_step = PatternStep(SynthType.GABBER_KICK, kick_params)
    for step in [0, 4, 8, 12]:
        pattern.set_step("kick", step, kick_step)
    
    # Acid bassline
    bass_track = pattern.add_track("acid_bass")
    bass_params = SynthParams(
        freq=110.0,
        amp=0.6,
        crunch=0.4,
        cutoff=1200.0,
        resonance=0.8,
        attack=0.01,
        decay=0.3,
        sustain=0.3,
        release=0.2
    )
    
    # Create acid pattern with different frequencies
    acid_notes = [110, 130, 140, 110, 165, 110, 130, 140]
    for i, freq in enumerate(acid_notes):
        bass_params_copy = bass_params.copy()
        bass_params_copy.freq = freq
        acid_step = PatternStep(SynthType.ACID_BASS, bass_params_copy, duration=0.5)
        pattern.set_step("acid_bass", i * 2, acid_step)
    
    return pattern