#!/usr/bin/env python3
"""
Tuned Kick Generator

Creates pitched kick drum sequences for frenchcore, hardstyle, and hardcore.
Unlike regular kicks, these have specific pitches and can play melodies.
"""

import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.midi_clips import MIDIClip, MIDINote, TimeSignature, note_name_to_midi


class KickTuning(Enum):
    """Common kick tuning approaches"""
    CHROMATIC = "chromatic"      # Chromatic scale (all semitones)
    PENTATONIC = "pentatonic"    # Pentatonic scale (more musical)
    MINOR = "minor"              # Natural minor scale
    HARMONIC_MINOR = "harmonic_minor"  # Harmonic minor (dramatic)
    FIFTHS = "fifths"            # Perfect fifths (powerful)
    OCTAVES = "octaves"          # Octave jumps only


# Scale intervals for different tuning types
TUNING_INTERVALS = {
    KickTuning.CHROMATIC: list(range(12)),  # All semitones
    KickTuning.PENTATONIC: [0, 3, 5, 7, 10],  # Minor pentatonic
    KickTuning.MINOR: [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    KickTuning.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],  # Harmonic minor
    KickTuning.FIFTHS: [0, 7],  # Root and fifth only
    KickTuning.OCTAVES: [0],  # Root only (octave jumps)
}


@dataclass
class TunedKickSettings:
    """Settings for tuned kick generation"""
    tuning: KickTuning = KickTuning.PENTATONIC
    octave_range: Tuple[int, int] = (1, 3)  # Octave range (C1-C3 typical)
    pitch_variation: float = 0.7  # How often pitch changes vs repeats
    velocity_variation: float = 0.3  # Velocity randomization amount
    
    # Velocity settings
    base_velocity: int = 110
    accent_velocity: int = 127
    ghost_velocity: int = 80
    
    # Pattern interpretation
    accent_char: str = 'X'  # Capital X = accent
    normal_char: str = 'x'  # Lowercase x = normal
    ghost_char: str = 'o'   # o = ghost note
    rest_char: str = '~'    # ~ = rest


class TunedKickGenerator:
    """
    Generate tuned kick drum patterns
    
    Creates pitched kick sequences typical of:
    - Frenchcore (melodic kick sequences)
    - Hardstyle (pitched kicks with melody)
    - Industrial hardcore (tuned impact sounds)
    - Gabber with pitch movement
    """
    
    def __init__(self,
                 root_note: str = "C1", 
                 pattern: str = "x ~ x ~ x ~ x ~",
                 tuning: str = "pentatonic",
                 pitch_sequence: Optional[List[int]] = None,
                 settings: Optional[TunedKickSettings] = None):
        
        self.root_note = note_name_to_midi(root_note)
        self.pattern = pattern.replace(" ", "")
        
        # Parse tuning string to enum
        try:
            self.tuning = KickTuning(tuning.lower())
        except ValueError:
            self.tuning = KickTuning.PENTATONIC
            
        self.pitch_sequence = pitch_sequence  # Optional predetermined pitch sequence
        self.settings = settings or TunedKickSettings(tuning=self.tuning)
        
        # Build available pitches
        self._build_pitch_palette()
        
    def _build_pitch_palette(self) -> None:
        """Build available pitches based on tuning and octave range"""
        intervals = TUNING_INTERVALS[self.tuning]
        min_octave, max_octave = self.settings.octave_range
        
        self.available_pitches = []
        
        for octave in range(min_octave, max_octave + 1):
            octave_offset = (octave - 1) * 12  # C1 is MIDI note 24
            for interval in intervals:
                pitch = self.root_note + octave_offset + interval
                if 0 <= pitch <= 127:  # Valid MIDI range
                    self.available_pitches.append(pitch)
        
        # For octaves-only tuning, add octave variations
        if self.tuning == KickTuning.OCTAVES:
            base_pitch = self.root_note
            for octave in range(min_octave, max_octave + 1):
                pitch = base_pitch + (octave - 1) * 12
                if 0 <= pitch <= 127 and pitch not in self.available_pitches:
                    self.available_pitches.append(pitch)
        
        self.available_pitches.sort()
    
    def _get_pattern_char_at_step(self, step: int) -> str:
        """Get pattern character at given step"""
        if not self.pattern:
            return self.settings.rest_char
        
        pattern_step = step % len(self.pattern)
        return self.pattern[pattern_step]
    
    def _choose_pitch_for_step(self, step: int, previous_pitch: Optional[int]) -> int:
        """Choose pitch for the current step"""
        
        # If we have a predetermined pitch sequence, use it
        if self.pitch_sequence:
            seq_index = step % len(self.pitch_sequence)
            semitone_offset = self.pitch_sequence[seq_index]
            return self.root_note + semitone_offset
        
        # If no previous pitch, start with root or random choice
        if previous_pitch is None:
            if self.tuning == KickTuning.OCTAVES:
                return self.root_note  # Always start on root for octave tuning
            else:
                return random.choice(self.available_pitches[:3])  # Lower pitches
        
        # Decide whether to change pitch
        if random.random() > self.settings.pitch_variation:
            return previous_pitch  # Repeat previous pitch
        
        # Choose new pitch based on tuning type
        if self.tuning == KickTuning.OCTAVES:
            # Octave jumps only
            octave_pitches = [p for p in self.available_pitches if p % 12 == self.root_note % 12]
            return random.choice(octave_pitches)
        
        elif self.tuning == KickTuning.CHROMATIC:
            # Any pitch within range
            return random.choice(self.available_pitches)
        
        else:
            # Scale-based movement - prefer step-wise motion
            try:
                current_index = self.available_pitches.index(previous_pitch)
            except ValueError:
                return random.choice(self.available_pitches)
            
            # Movement options with weights
            movement_options = []
            
            # Add stepwise movement (higher weight)
            if current_index > 0:
                movement_options.extend([self.available_pitches[current_index - 1]] * 3)
            if current_index < len(self.available_pitches) - 1:
                movement_options.extend([self.available_pitches[current_index + 1]] * 3)
            
            # Add small jumps (medium weight)
            for jump in [-2, 2]:
                target_index = current_index + jump
                if 0 <= target_index < len(self.available_pitches):
                    movement_options.extend([self.available_pitches[target_index]] * 2)
            
            # Add current pitch (stay put - medium weight)
            movement_options.extend([previous_pitch] * 2)
            
            # Add random jumps (low weight)
            movement_options.extend(self.available_pitches)
            
            return random.choice(movement_options)
    
    def _get_velocity_for_char(self, pattern_char: str, step: int) -> int:
        """Get velocity based on pattern character"""
        base_velocity = self.settings.base_velocity
        
        if pattern_char == self.settings.accent_char:  # X = accent
            base_velocity = self.settings.accent_velocity
        elif pattern_char == self.settings.ghost_char:  # o = ghost
            base_velocity = self.settings.ghost_velocity
        
        # Add variation
        variation_amount = int(self.settings.velocity_variation * 20)
        variation = random.randint(-variation_amount, variation_amount)
        
        return max(1, min(127, base_velocity + variation))
    
    def _get_note_duration(self, pattern_char: str) -> float:
        """Get note duration based on pattern character"""
        # Tuned kicks are usually short and punchy
        if pattern_char == self.settings.accent_char:
            return 0.3  # Slightly longer for accents
        elif pattern_char == self.settings.ghost_char:
            return 0.15  # Shorter for ghost notes
        else:
            return 0.25  # Standard kick length
    
    def generate(self, length_bars: float = 1.0, bpm: float = 180.0) -> MIDIClip:
        """Generate tuned kick pattern MIDI clip"""
        
        # Create clip
        clip = MIDIClip(
            name=f"tuned_kicks_{self.tuning.value}",
            length_bars=length_bars,
            time_signature=TimeSignature(4, 4),
            bpm=bpm
        )
        clip.tags = ["tuned_kick", "frenchcore", "hardstyle", "kick"]
        
        # Calculate timing
        beats_per_bar = 4.0
        total_beats = length_bars * beats_per_bar
        step_size = 0.25  # 16th notes (typical for kick patterns)
        num_steps = int(total_beats / step_size)
        
        # Generate sequence
        previous_pitch = None
        
        for step in range(num_steps):
            step_time = step * step_size
            pattern_char = self._get_pattern_char_at_step(step)
            
            # Skip rests
            if pattern_char == self.settings.rest_char:
                continue
            
            # Choose pitch for this kick
            pitch = self._choose_pitch_for_step(step, previous_pitch)
            previous_pitch = pitch
            
            # Get velocity and duration
            velocity = self._get_velocity_for_char(pattern_char, step)
            duration = self._get_note_duration(pattern_char)
            
            # Create MIDI note
            note = MIDINote(
                pitch=pitch,
                velocity=velocity,
                start_time=step_time,
                duration=duration,
                channel=9  # Drum channel
            )
            
            clip.add_note(note)
        
        return clip
    
    def create_melodic_sequence(self, melody_pitches: List[int], 
                               rhythm_pattern: str = "x x x x x x x x") -> MIDIClip:
        """Create a melodic kick sequence with specific pitches"""
        
        # Temporarily set pitch sequence
        original_sequence = self.pitch_sequence
        self.pitch_sequence = melody_pitches
        
        # Temporarily set pattern
        original_pattern = self.pattern
        self.pattern = rhythm_pattern.replace(" ", "")
        
        try:
            # Generate with melody
            clip = self.generate(length_bars=len(melody_pitches) * 0.25, bpm=180)
            clip.name = f"melodic_kicks_{len(melody_pitches)}_notes"
            return clip
        finally:
            # Restore original settings
            self.pitch_sequence = original_sequence
            self.pattern = original_pattern


# Convenience functions for common patterns
def create_frenchcore_kicks(root_note: str = "C1", length_bars: float = 2.0, bpm: float = 200.0) -> MIDIClip:
    """Create typical frenchcore melodic kick pattern"""
    generator = TunedKickGenerator(
        root_note=root_note,
        pattern="x x x ~ x x ~ x x ~ x x x ~ ~ x",  # Complex frenchcore pattern
        tuning="pentatonic"
    )
    return generator.generate(length_bars, bpm)


def create_hardstyle_kicks(root_note: str = "C1", length_bars: float = 1.0, bpm: float = 150.0) -> MIDIClip:
    """Create hardstyle pitched kick pattern"""
    generator = TunedKickGenerator(
        root_note=root_note,
        pattern="X ~ ~ ~ X ~ ~ ~",  # Classic hardstyle pattern with accents
        tuning="minor",
        pitch_sequence=[0, 0, 0, 0, 5, 5, 5, 5]  # Root then fifth
    )
    return generator.generate(length_bars, bpm)


def create_industrial_kicks(root_note: str = "C1", length_bars: float = 4.0, bpm: float = 140.0) -> MIDIClip:
    """Create industrial tuned kick pattern"""
    settings = TunedKickSettings(
        tuning=KickTuning.CHROMATIC,
        octave_range=(1, 2),  # Stay in lower register
        pitch_variation=0.4,  # Less pitch movement
        base_velocity=120
    )
    
    generator = TunedKickGenerator(
        root_note=root_note,
        pattern="x ~ x ~ ~ x ~ x ~ x ~ ~ x ~ x ~",  # Industrial rhythm
        tuning="chromatic",
        settings=settings
    )
    return generator.generate(length_bars, bpm)


def create_octave_kicks(root_note: str = "C1", length_bars: float = 1.0, bpm: float = 180.0) -> MIDIClip:
    """Create octave-jumping kick pattern (powerful and dramatic)"""
    generator = TunedKickGenerator(
        root_note=root_note,
        pattern="X ~ x ~ X ~ x ~",  # Alternating accents
        tuning="octaves"
    )
    return generator.generate(length_bars, bpm)