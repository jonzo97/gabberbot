#!/usr/bin/env python3
"""
Track Architecture - Professional DAW-Style Audio Track

Clean, modular Track class that follows DAW principles:
Control Source → Audio Source → FX Chain → Mixer

Uses existing interfaces from cli_shared and extracted audio modules.
NO spaghetti inheritance - composition-based design.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Use existing interfaces from cli_shared
from cli_shared.interfaces.synthesizer import AbstractSynthesizer, SynthesizerState
from cli_shared.models.hardcore_models import HardcorePattern, SynthParams, SynthType

# Use extracted audio modules
from ..effects import (
    apply_compression, apply_hardcore_limiter, 
    rotterdam_doorlussen, warehouse_reverb,
    apply_kick_space_highpass, apply_harshness_lowpass
)
from ..synthesis import synthesize_simple_kick
from ..parameters.synthesis_constants import HardcoreConstants, SynthesisParams

# Import modulation system
from ..modulation.modulator import ModulationRouter, LFO, Envelope


# Control Sources (what triggers the track)
class ControlSource(ABC):
    """Abstract control source for track triggering"""
    
    @abstractmethod
    def should_trigger(self, step: int, bpm: float) -> bool:
        """Check if track should trigger on this step"""
        pass
    
    @abstractmethod
    def get_velocity(self, step: int) -> float:
        """Get trigger velocity (0.0-1.0)"""
        pass


class PatternControlSource(ControlSource):
    """Pattern-based control source"""
    
    def __init__(self, pattern: str = "x ~ x ~ x ~ x ~"):
        self.pattern = pattern.replace(" ", "")  # Remove spaces
        self.length = len(self.pattern)
    
    def should_trigger(self, step: int, bpm: float) -> bool:
        pattern_step = step % self.length
        return self.pattern[pattern_step] == 'x'
    
    def get_velocity(self, step: int) -> float:
        return 1.0 if self.should_trigger(step, 0) else 0.0


class MidiControlSource(ControlSource):
    """MIDI-based control source (uses existing cli_shared MIDI utils)"""
    
    def __init__(self, midi_note: int = 36):  # C1 kick
        self.midi_note = midi_note
        self.active_notes: Dict[int, bool] = {}
    
    def should_trigger(self, step: int, bpm: float) -> bool:
        return self.active_notes.get(self.midi_note, False)
    
    def get_velocity(self, step: int) -> float:
        return 1.0 if self.should_trigger(step, bpm) else 0.0


# Audio Sources (what generates the sound)  
class AudioSource(ABC):
    """Abstract audio source for track synthesis"""
    
    @abstractmethod
    def generate_audio(self, velocity: float, params: SynthesisParams, 
                      sample_rate: int) -> np.ndarray:
        """Generate audio for this trigger"""
        pass


class KickAudioSource(AudioSource):
    """Kick drum audio source using extracted synthesis"""
    
    def __init__(self, frequency: float = 60.0, duration_ms: int = 400):
        self.frequency = frequency
        self.duration_ms = duration_ms
    
    def generate_audio(self, velocity: float, params: SynthesisParams, 
                      sample_rate: int) -> np.ndarray:
        """Generate kick drum using extracted oscillator functions"""
        if velocity <= 0:
            return np.array([])
            
        # Use extracted synthesis function
        kick = synthesize_simple_kick(
            frequency=self.frequency * velocity,  # Velocity affects pitch
            duration_ms=self.duration_ms,
            sample_rate=sample_rate
        )
        
        return kick * velocity


class SynthesizerAudioSource(AudioSource):
    """Audio source using existing AbstractSynthesizer interface"""
    
    def __init__(self, synthesizer: AbstractSynthesizer = None, synth_type: Optional[SynthType] = None):
        self.synthesizer = synthesizer
        self.synth_type = synth_type
        self._engine_router = None
    
    def generate_audio(self, velocity: float, params: SynthesisParams,
                      sample_rate: int) -> np.ndarray:
        """Generate audio using existing synthesizer interface"""
        if velocity <= 0:
            return np.array([])
        
        # If no synthesizer provided but synth_type is specified, use engine router
        if not self.synthesizer and self.synth_type:
            if not self._engine_router:
                from .engine_router import get_engine_router
                self._engine_router = get_engine_router()
            
            # Route to best available backend
            backend = self._engine_router.route_synth_request(self.synth_type, params)
            if backend:
                try:
                    import asyncio
                    engine = asyncio.run(self._engine_router.get_engine(backend))
                    if engine and hasattr(engine, 'render_pattern_step'):
                        return engine.render_pattern_step(velocity, params)
                except Exception as e:
                    print(f"Engine router error: {e}")
                    # Fall back to basic synthesis
                    pass
        
        # Use existing synthesizer interface
        if self.synthesizer and hasattr(self.synthesizer, 'render_pattern_step'):
            return self.synthesizer.render_pattern_step(velocity, params)
        
        # Fallback: return silence
        return np.array([])


# Effects Chain (what processes the sound)
@dataclass
class EffectSettings:
    """Settings for individual effects"""
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class EffectsChain:
    """Chain of audio effects using extracted modules"""
    
    def __init__(self):
        self.effects: List[Callable] = []
        self.settings: Dict[str, EffectSettings] = {}
    
    def add_distortion(self, style: str = "gabber", drive: float = 2.5):
        """Add distortion using extracted modules"""
        self.settings["distortion"] = EffectSettings(
            enabled=True,
            parameters={"style": style, "drive": drive}
        )
    
    def add_compression(self, ratio: float = 8.0, threshold_db: float = -10):
        """Add compression using extracted modules"""  
        self.settings["compression"] = EffectSettings(
            enabled=True,
            parameters={"ratio": ratio, "threshold_db": threshold_db}
        )
    
    def add_reverb(self, style: str = "warehouse", wet_level: float = 0.3):
        """Add reverb using extracted modules"""
        self.settings["reverb"] = EffectSettings(
            enabled=True, 
            parameters={"style": style, "wet_level": wet_level}
        )
    
    def process_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through effects chain"""
        if len(audio) == 0:
            return audio
            
        processed = audio.copy()
        
        # Apply distortion if enabled
        if "distortion" in self.settings and self.settings["distortion"].enabled:
            params = self.settings["distortion"].parameters
            if params["style"] == "gabber":
                processed = rotterdam_doorlussen(
                    processed, 
                    drive_per_stage=params["drive"],
                    sample_rate=sample_rate
                )
        
        # Apply compression if enabled
        if "compression" in self.settings and self.settings["compression"].enabled:
            params = self.settings["compression"].parameters
            processed = apply_compression(
                processed,
                ratio=params["ratio"],
                threshold_db=params["threshold_db"],
                sample_rate=sample_rate
            )
        
        # Apply reverb if enabled
        if "reverb" in self.settings and self.settings["reverb"].enabled:
            params = self.settings["reverb"].parameters
            if params["style"] == "warehouse":
                processed = warehouse_reverb(
                    processed,
                    sample_rate=sample_rate,
                    wet_level=params["wet_level"]
                )
        
        return processed


# Main Track Class
class Track:
    """
    Professional DAW-style audio track.
    
    Follows composition pattern instead of spaghetti inheritance:
    - Control Source: What triggers the track (pattern, MIDI, etc.)
    - Audio Source: What generates the sound (synth, kick, etc.) 
    - Effects Chain: What processes the sound (distortion, compression, etc.)
    - Mixer: Volume, pan, mute, solo controls
    
    Uses existing interfaces and extracted modules - NO reinventing wheels.
    """
    
    def __init__(self, name: str = "Track"):
        self.name = name
        
        # Track components (composition not inheritance)
        self.control_source: Optional[ControlSource] = None
        self.audio_source: Optional[AudioSource] = None
        self.effects_chain = EffectsChain()
        
        # Modulation system integration
        self.modulation_router = ModulationRouter()
        self.modulated_params: Dict[str, float] = {
            "volume": 1.0,
            "pan": 0.0,
            "filter_cutoff": 1000.0,
            "distortion_drive": 2.5,
            "reverb_wet": 0.3
        }
        
        # Mixer controls  
        self.volume = 1.0
        self.pan = 0.0  # -1.0 (left) to 1.0 (right)
        self.muted = False
        self.soloed = False
        
        # Track state
        self.enabled = True
        self.sample_rate = HardcoreConstants.SAMPLE_RATE_44K
    
    def set_control_source(self, control_source: ControlSource):
        """Set how this track is triggered"""
        self.control_source = control_source
    
    def set_audio_source(self, audio_source: AudioSource):
        """Set what generates the audio"""
        self.audio_source = audio_source
    
    def add_modulation(self, param_name: str, modulator, amount: float = 1.0):
        """Add modulation to a track parameter"""
        if param_name in self.modulated_params:
            self.modulation_router.connect(modulator, param_name, amount)
    
    def add_lfo_modulation(self, param_name: str, rate: float = 0.5, 
                          amplitude: float = 0.5, waveform: str = "sine"):
        """Quick setup: add LFO modulation to a parameter"""
        from ..modulation.modulator import WaveformType
        
        # Convert string waveform to enum
        waveform_map = {
            "sine": WaveformType.SINE,
            "triangle": WaveformType.TRIANGLE,
            "square": WaveformType.SQUARE,
            "sawtooth": WaveformType.SAW
        }
        waveform_enum = waveform_map.get(waveform, WaveformType.SINE)
        
        lfo = LFO(rate=rate, waveform=waveform_enum)
        self.add_modulation(param_name, lfo, amplitude)
        return lfo
    
    def update_modulation(self, current_time: float):
        """Update all modulation values"""
        # Update modulation router with current time
        modulation_values = self.modulation_router.update(current_time, self.sample_rate)
        
        # Apply modulation to parameters
        for param_name, base_value in self.modulated_params.items():
            if param_name in modulation_values:
                modulated_value = base_value + modulation_values[param_name]
                # Apply the modulated value to the actual parameter
                if param_name == "volume":
                    self.volume = max(0.0, min(2.0, modulated_value))
                elif param_name == "pan":
                    self.pan = max(-1.0, min(1.0, modulated_value))
                # Effects parameters will be handled in the effects chain
    
    def render_step(self, step: int, bpm: float, params: SynthesisParams) -> np.ndarray:
        """
        Render one step of this track.
        
        This is the main rendering pipeline:
        Control Source → Audio Source → Effects Chain → Modulation → Mixer
        """
        if not self.enabled or self.muted or not self.control_source or not self.audio_source:
            return np.array([])
        
        # Calculate current time for modulation
        current_time = step * (60.0 / bpm / 4.0)  # Assuming 16th note steps
        
        # Update modulation before processing
        self.update_modulation(current_time)
        
        # 1. Control Source - should we trigger?
        should_trigger = self.control_source.should_trigger(step, bpm)
        if not should_trigger:
            return np.array([])
        
        velocity = self.control_source.get_velocity(step)
        
        # 2. Audio Source - generate the sound
        audio = self.audio_source.generate_audio(velocity, params, self.sample_rate)
        if len(audio) == 0:
            return audio
        
        # 3. Effects Chain - process the sound with modulated parameters
        audio = self.effects_chain.process_audio(audio, self.sample_rate)
        
        # 4. Mixer - apply modulated volume/pan
        audio = audio * self.volume
        
        return audio
    
    def add_kick_pattern(self, pattern: str = "x ~ x ~ x ~ x ~", 
                        frequency: float = 60.0, duration_ms: int = 400):
        """Quick setup: kick drum with pattern"""
        self.set_control_source(PatternControlSource(pattern))
        self.set_audio_source(KickAudioSource(frequency, duration_ms))
    
    def add_gabber_effects(self):
        """Quick setup: Rotterdam gabber effects chain"""
        self.effects_chain.add_distortion("gabber", drive=2.5)
        self.effects_chain.add_compression(ratio=8.0, threshold_db=-10)
        self.effects_chain.add_reverb("warehouse", wet_level=0.2)
    
    def add_synthesizer_source(self, synthesizer: AbstractSynthesizer):
        """Use existing synthesizer interface (cli_strudel/cli_sc)"""
        self.set_audio_source(SynthesizerAudioSource(synthesizer))
    
    def add_smart_synth_source(self, synth_type: SynthType):
        """Use engine router to automatically select best backend for synth type"""
        self.set_audio_source(SynthesizerAudioSource(synth_type=synth_type))
    
    def add_gabber_synth(self):
        """Quick setup: gabber kick using best available backend"""
        self.add_smart_synth_source(SynthType.GABBER_KICK)
    
    def add_hoover_synth(self):
        """Quick setup: hoover synth using best available backend"""
        self.add_smart_synth_source(SynthType.HOOVER_SYNTH)
    
    def add_modulated_gabber_effects(self):
        """Quick setup: gabber effects with LFO modulation"""
        self.add_gabber_effects()
        
        # Add LFO modulation for dynamic effects
        self.add_lfo_modulation("distortion_drive", rate=0.25, amplitude=0.5)
        self.add_lfo_modulation("filter_cutoff", rate=0.125, amplitude=300.0)
        self.add_lfo_modulation("reverb_wet", rate=0.1, amplitude=0.1)


# Track Collection for multi-track arrangements
class TrackCollection:
    """Collection of tracks for complete arrangements"""
    
    def __init__(self, name: str = "Session"):
        self.name = name
        self.tracks: List[Track] = []
        self.master_volume = 1.0
        self.master_limiter_enabled = True
    
    def add_track(self, track: Track):
        """Add track to collection"""
        self.tracks.append(track)
    
    def render_step(self, step: int, bpm: float, params: SynthesisParams) -> np.ndarray:
        """Render all tracks for one step"""
        if not self.tracks:
            return np.array([])
        
        # Render all tracks
        track_outputs = []
        for track in self.tracks:
            output = track.render_step(step, bpm, params)
            if len(output) > 0:
                track_outputs.append(output)
        
        if not track_outputs:
            return np.array([])
        
        # Mix tracks together (sum)
        # Ensure all tracks have same length
        max_length = max(len(output) for output in track_outputs)
        mixed = np.zeros(max_length)
        
        for output in track_outputs:
            if len(output) < max_length:
                # Pad shorter tracks with silence
                padded = np.pad(output, (0, max_length - len(output)))
                mixed += padded
            else:
                mixed += output[:max_length]
        
        # Apply master volume
        mixed *= self.master_volume
        
        # Apply master limiter
        if self.master_limiter_enabled:
            mixed = apply_hardcore_limiter(mixed, sample_rate=HardcoreConstants.SAMPLE_RATE_44K)
        
        return mixed