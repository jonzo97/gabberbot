"""
Hardcore Music Synthesis Modules.

Implements authentic hardcore, gabber, and industrial synthesis using
the parameters and aesthetic defined in CLAUDE.md.
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging

from ..models.core import MIDINote


class HardcoreSynthesizer:
    """
    General-purpose hardcore synthesizer for riffs, leads, and melodic content.
    
    Implements aggressive, industrial-style synthesis with proper hardcore aesthetics.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize hardcore synthesizer."""
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
    
    def generate_note(
        self, 
        pitch: int, 
        duration: float, 
        velocity: int,
        waveform: str = "sawtooth"
    ) -> np.ndarray:
        """
        Generate a single hardcore note with aggressive characteristics.
        
        Args:
            pitch: MIDI note number (0-127)
            duration: Note duration in seconds
            velocity: MIDI velocity (0-127)
            waveform: Base waveform type
            
        Returns:
            Audio samples for the note
        """
        # Convert MIDI to frequency
        frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        
        # Generate base waveform
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        if waveform == "sawtooth":
            audio = self._generate_sawtooth(frequency, t)
        elif waveform == "square":
            audio = self._generate_square(frequency, t)
        else:  # sine fallback
            audio = np.sin(2 * np.pi * frequency * t)
        
        # Apply hardcore processing
        audio = self._apply_hardcore_characteristics(audio, velocity, duration)
        
        return audio.astype(np.float32)
    
    def _generate_sawtooth(self, frequency: float, t: np.ndarray) -> np.ndarray:
        """Generate sawtooth wave (good for hardcore leads)."""
        phase = 2 * np.pi * frequency * t
        return 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
    
    def _generate_square(self, frequency: float, t: np.ndarray) -> np.ndarray:
        """Generate square wave (good for harsh industrial sounds)."""
        return np.sign(np.sin(2 * np.pi * frequency * t))
    
    def _apply_hardcore_characteristics(
        self, audio: np.ndarray, velocity: int, duration: float
    ) -> np.ndarray:
        """Apply hardcore-specific processing to make it aggressive."""
        # Velocity scaling
        amplitude = velocity / 127.0
        audio *= amplitude
        
        # Envelope (quick attack, moderate decay)
        envelope = self._create_hardcore_envelope(len(audio), duration)
        audio *= envelope
        
        # Add slight distortion for edge
        audio = np.tanh(audio * 1.5)
        
        return audio
    
    def _create_hardcore_envelope(self, samples: int, duration: float) -> np.ndarray:
        """Create aggressive envelope for hardcore sounds."""
        envelope = np.ones(samples)
        
        # Quick attack (10ms)
        attack_samples = int(0.01 * self.sample_rate)
        if attack_samples < samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Exponential decay
        decay_start = attack_samples
        decay_samples = samples - decay_start
        if decay_samples > 0:
            decay = np.exp(-3 * np.linspace(0, 1, decay_samples))
            envelope[decay_start:] *= decay
        
        return envelope


class KickSynthesizer:
    """
    Hardcore kick drum synthesizer implementing authentic gabber/hardcore kicks.
    
    Based on CLAUDE.md hardcore parameters and Rotterdam gabber techniques.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize kick synthesizer."""
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Hardcore kick parameters from CLAUDE.md
        self.kick_params = {
            'sub_freqs': [41.2, 82.4, 123.6],  # E1, E2, E2+fifth Hz
            'distortion_db': 15,  # Reduced from 18 per user feedback
            'compression_ratio': 8,
            'limiter_threshold': -0.5,
        }
    
    def generate_kick(
        self, 
        duration: float = 0.15, 
        velocity: int = 127,
        pitch: int = 36,
        style: str = "gabber"
    ) -> np.ndarray:
        """
        Generate a hardcore kick drum with CRUNCHY characteristics.
        
        Args:
            duration: Kick duration in seconds
            velocity: MIDI velocity (affects intensity)
            pitch: MIDI pitch (affects tuning)
            style: Kick style ("gabber", "hardcore", "industrial")
            
        Returns:
            Audio samples for the kick
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Base frequency from pitch
        base_freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        
        # Generate multi-layered kick
        kick_audio = self._generate_kick_layers(t, base_freq, style)
        
        # Apply hardcore processing
        kick_audio = self._apply_kick_processing(kick_audio, velocity, style)
        
        return kick_audio.astype(np.float32)
    
    def _generate_kick_layers(
        self, t: np.ndarray, base_freq: float, style: str
    ) -> np.ndarray:
        """Generate multiple layers for a full hardcore kick."""
        # Layer 1: Sub-bass thump
        sub_freq = max(base_freq, 60.0)  # Ensure it's in sub range
        sub_layer = self._generate_sub_thump(t, sub_freq)
        
        # Layer 2: Mid punch
        mid_layer = self._generate_mid_punch(t, base_freq * 2)
        
        # Layer 3: High click/transient
        click_layer = self._generate_click_transient(t)
        
        # Mix layers based on style
        if style == "gabber":
            # Gabber: Heavy on distortion and mid punch
            kick = sub_layer * 0.7 + mid_layer * 0.8 + click_layer * 0.5
        elif style == "industrial":
            # Industrial: More sub, less mid
            kick = sub_layer * 0.9 + mid_layer * 0.4 + click_layer * 0.6
        else:  # hardcore
            # Balanced but aggressive
            kick = sub_layer * 0.8 + mid_layer * 0.7 + click_layer * 0.6
        
        return kick
    
    def _generate_sub_thump(self, t: np.ndarray, freq: float) -> np.ndarray:
        """Generate the sub-bass thump of the kick."""
        # Sine wave with exponential decay
        audio = np.sin(2 * np.pi * freq * t)
        
        # Rapid pitch drop for thump effect
        pitch_envelope = np.exp(-8 * t)
        modulated_freq = freq * (0.3 + 0.7 * pitch_envelope)
        audio = np.sin(2 * np.pi * modulated_freq * t)
        
        # Amplitude envelope
        amp_envelope = np.exp(-5 * t)
        audio *= amp_envelope
        
        return audio
    
    def _generate_mid_punch(self, t: np.ndarray, freq: float) -> np.ndarray:
        """Generate the mid-frequency punch."""
        # Start with triangle wave for warmth
        phase = 2 * np.pi * freq * t
        audio = 2 * np.arcsin(np.sin(phase)) / np.pi
        
        # Quick decay for punch
        envelope = np.exp(-12 * t)
        audio *= envelope
        
        return audio
    
    def _generate_click_transient(self, t: np.ndarray) -> np.ndarray:
        """Generate the high-frequency click for attack."""
        # High-frequency burst
        freq = 4000.0
        audio = np.sin(2 * np.pi * freq * t)
        
        # Very quick decay (just the transient)
        envelope = np.exp(-100 * t)
        audio *= envelope
        
        return audio
    
    def _apply_kick_processing(
        self, audio: np.ndarray, velocity: int, style: str
    ) -> np.ndarray:
        """Apply hardcore kick processing (distortion, compression)."""
        # Velocity scaling
        amplitude = velocity / 127.0
        audio *= amplitude
        
        # Hardcore distortion (the key to gabber sound)
        distortion_amount = self.kick_params['distortion_db'] / 20.0
        if style == "gabber":
            distortion_amount *= 1.5  # Extra crunch for gabber
        
        # Apply distortion
        audio = np.tanh(audio * distortion_amount)
        
        # Compression simulation
        compressed = self._apply_compression(audio)
        
        # Hard limiting for maximum impact
        limited = np.clip(compressed, -0.95, 0.95)
        
        return limited
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply hardcore-style compression."""
        threshold = 0.3
        ratio = self.kick_params['compression_ratio']
        
        # Simple compression
        compressed = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
            audio
        )
        
        return compressed


class AcidSynthesizer:
    """
    TB-303 style acid synthesizer for authentic acid basslines.
    
    Implements the classic squelchy, sliding basslines characteristic of acid house
    and its hardcore derivatives.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize acid synthesizer."""
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
    
    def generate_sequence(
        self, 
        notes: List[MIDINote], 
        bpm: float,
        total_duration: float
    ) -> np.ndarray:
        """
        Generate an acid bassline sequence with slides and filter automation.
        
        Args:
            notes: List of MIDI notes for the sequence
            bpm: Beats per minute for timing
            total_duration: Total duration in seconds
            
        Returns:
            Audio samples for the acid sequence
        """
        total_samples = int(total_duration * self.sample_rate)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)
        
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        for i, note in enumerate(sorted_notes):
            # Convert beat time to seconds
            start_time = note.start_time * (60.0 / bpm)
            duration = note.duration * (60.0 / bpm)
            
            # Check if this note should slide to the next
            has_slide = self._should_slide(note, sorted_notes, i)
            
            # Generate acid note
            note_audio = self._generate_acid_note(
                note.pitch, duration, note.velocity, has_slide
            )
            
            # Add to buffer
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + len(note_audio)
            
            if end_sample <= total_samples:
                audio_buffer[start_sample:end_sample] += note_audio
        
        # Apply filter automation to entire sequence
        filtered_audio = self._apply_filter_automation(audio_buffer, total_duration)
        
        return filtered_audio
    
    def _generate_acid_note(
        self, pitch: int, duration: float, velocity: int, has_slide: bool
    ) -> np.ndarray:
        """Generate a single acid note with TB-303 characteristics."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Base frequency
        frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        
        # Generate sawtooth wave (TB-303 characteristic)
        audio = self._generate_acid_sawtooth(frequency, t, has_slide)
        
        # Apply acid envelope
        envelope = self._create_acid_envelope(samples, velocity)
        audio *= envelope
        
        return audio
    
    def _generate_acid_sawtooth(
        self, frequency: float, t: np.ndarray, has_slide: bool
    ) -> np.ndarray:
        """Generate TB-303 style sawtooth with optional slide."""
        if has_slide:
            # Slide up in pitch over time
            slide_amount = 0.1  # Slide up by 10%
            freq_modulation = frequency * (1 + slide_amount * t / t[-1])
            phase = 2 * np.pi * np.cumsum(freq_modulation) / self.sample_rate
        else:
            phase = 2 * np.pi * frequency * t
        
        # Generate sawtooth
        sawtooth = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        
        return sawtooth
    
    def _create_acid_envelope(self, samples: int, velocity: int) -> np.ndarray:
        """Create TB-303 style envelope."""
        envelope = np.ones(samples)
        
        # Quick attack
        attack_samples = int(0.005 * self.sample_rate)  # 5ms
        if attack_samples < samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Sustained decay (characteristic of TB-303)
        decay_start = attack_samples
        decay_samples = samples - decay_start
        if decay_samples > 0:
            # Slower decay for acid sustain
            decay_rate = 1.5 if velocity > 100 else 2.5
            decay = np.exp(-decay_rate * np.linspace(0, 1, decay_samples))
            envelope[decay_start:] *= decay
        
        return envelope
    
    def _should_slide(
        self, current_note: MIDINote, all_notes: List[MIDINote], index: int
    ) -> bool:
        """Determine if this note should slide to the next note."""
        if index >= len(all_notes) - 1:
            return False
        
        next_note = all_notes[index + 1]
        
        # Slide if next note starts immediately after this one
        current_end = current_note.start_time + current_note.duration
        gap = next_note.start_time - current_end
        
        # Slide if gap is very small (less than 1/16 note)
        return gap < 0.25
    
    def _apply_filter_automation(
        self, audio: np.ndarray, duration: float
    ) -> np.ndarray:
        """Apply classic acid filter sweep automation."""
        # Simple low-pass filter simulation with cutoff automation
        # This is a simplified version - real TB-303 filter is more complex
        
        # Create cutoff automation (starts low, sweeps up)
        cutoff_envelope = 0.3 + 0.7 * np.sin(2 * np.pi * 0.5 * np.linspace(0, duration, len(audio)))
        
        # Apply simple filter effect (this is a placeholder for proper filtering)
        # In a full implementation, you'd use a proper resonant filter
        filtered = audio * cutoff_envelope
        
        # Add resonance simulation (slight amplification around cutoff)
        resonance = 1.2
        filtered *= resonance
        
        # Prevent clipping
        filtered = np.tanh(filtered)
        
        return filtered.astype(np.float32)