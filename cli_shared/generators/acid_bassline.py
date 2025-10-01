#!/usr/bin/env python3
"""
Acid Bassline Generator

Creates authentic acid/reverse basslines with:
- Scale-based note selection
- Accent patterns for velocity variation  
- Slide/glide effects
- Octave jumping
- Hardcore-appropriate note durations and timing
"""

import random
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.midi_clips import MIDIClip, MIDINote, TimeSignature, note_name_to_midi


class AcidScale(Enum):
    """Scales commonly used in acid/hardcore music"""
    A_MINOR = "A_minor"
    E_MINOR = "E_minor" 
    C_MINOR = "C_minor"
    D_MINOR = "D_minor"
    G_MINOR = "G_minor"
    A_MINOR_PENTATONIC = "A_minor_pentatonic"
    E_MINOR_PENTATONIC = "E_minor_pentatonic"
    HARMONIC_MINOR = "A_harmonic_minor"
    PHRYGIAN = "E_phrygian"


# Scale definitions (semitones from root)
SCALE_INTERVALS = {
    AcidScale.A_MINOR: [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    AcidScale.E_MINOR: [0, 2, 3, 5, 7, 8, 10],
    AcidScale.C_MINOR: [0, 2, 3, 5, 7, 8, 10],
    AcidScale.D_MINOR: [0, 2, 3, 5, 7, 8, 10],
    AcidScale.G_MINOR: [0, 2, 3, 5, 7, 8, 10],
    AcidScale.A_MINOR_PENTATONIC: [0, 3, 5, 7, 10],
    AcidScale.E_MINOR_PENTATONIC: [0, 3, 5, 7, 10],
    AcidScale.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],  # Raised 7th
    AcidScale.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],  # Minor with b2
}

# Root notes for different scales  
SCALE_ROOTS = {
    AcidScale.A_MINOR: note_name_to_midi("A2"),
    AcidScale.E_MINOR: note_name_to_midi("E2"), 
    AcidScale.C_MINOR: note_name_to_midi("C2"),
    AcidScale.D_MINOR: note_name_to_midi("D2"),
    AcidScale.G_MINOR: note_name_to_midi("G2"),
    AcidScale.A_MINOR_PENTATONIC: note_name_to_midi("A2"),
    AcidScale.E_MINOR_PENTATONIC: note_name_to_midi("E2"),
    AcidScale.HARMONIC_MINOR: note_name_to_midi("A2"),
    AcidScale.PHRYGIAN: note_name_to_midi("E2"),
}


@dataclass
class AcidSettings:
    """Settings for acid bassline generation"""
    scale: AcidScale = AcidScale.A_MINOR
    octave_range: Tuple[int, int] = (2, 4)  # Octave range for notes
    note_density: float = 0.75  # Probability of note on each 16th
    accent_probability: float = 0.3  # Probability of accented note
    slide_probability: float = 0.4  # Probability of slide between notes
    octave_jump_probability: float = 0.15  # Probability of octave jump
    rest_probability: float = 0.1  # Probability of rest instead of note
    
    # Velocity settings
    normal_velocity: int = 90
    accent_velocity: int = 127
    ghost_velocity: int = 60
    
    # Note duration settings (in beats)
    short_note: float = 0.125  # 32nd note
    normal_note: float = 0.25  # 16th note  
    long_note: float = 0.5  # 8th note


class AcidBasslineGenerator:
    """
    Generate authentic acid basslines
    
    Creates patterns typical of acid house/hardcore with:
    - Scale-based melodies
    - Velocity accents for 303-style filter sweeps
    - Slide effects between notes
    - Octave variations
    - Appropriate note lengths and spacing
    """
    
    def __init__(self, 
                 scale: str = "A_minor",
                 accent_pattern: str = "x ~ ~ x ~ ~ x ~",
                 settings: Optional[AcidSettings] = None):
        
        # Parse scale string to enum
        try:
            self.scale = AcidScale(scale.upper())
        except ValueError:
            self.scale = AcidScale.A_MINOR
            
        self.accent_pattern = accent_pattern.replace(" ", "")
        self.settings = settings or AcidSettings(scale=self.scale)
        
        # Build scale notes
        self._build_scale_notes()
        
    def _build_scale_notes(self) -> None:
        """Build list of available scale notes across octave range"""
        root = SCALE_ROOTS[self.scale]
        intervals = SCALE_INTERVALS[self.scale]
        
        self.scale_notes = []
        min_octave, max_octave = self.settings.octave_range
        
        for octave in range(min_octave, max_octave + 1):
            octave_offset = (octave - 2) * 12  # Offset from base octave
            for interval in intervals:
                note = root + octave_offset + interval
                if 0 <= note <= 127:  # Valid MIDI range
                    self.scale_notes.append(note)
        
        self.scale_notes.sort()
    
    def _get_accent_at_step(self, step: int) -> bool:
        """Check if step should be accented based on pattern"""
        if not self.accent_pattern:
            return False
        
        pattern_step = step % len(self.accent_pattern)
        return self.accent_pattern[pattern_step] == 'x'
    
    def _choose_next_note(self, current_note: Optional[int], step: int) -> Optional[int]:
        """Choose next note based on musical logic"""
        
        # Chance of rest
        if random.random() < self.settings.rest_probability:
            return None
            
        # If no current note, choose random scale note
        if current_note is None:
            return random.choice(self.scale_notes)
        
        # Find current note in scale
        try:
            current_index = self.scale_notes.index(current_note)
        except ValueError:
            # Current note not in scale, pick random
            return random.choice(self.scale_notes)
        
        # Choose movement pattern
        if random.random() < self.settings.octave_jump_probability:
            # Octave jump - same note different octave
            octave_notes = [n for n in self.scale_notes if n % 12 == current_note % 12]
            if len(octave_notes) > 1:
                return random.choice([n for n in octave_notes if n != current_note])
        
        # Step-wise movement (most common)
        movement_options = []
        
        # Add neighboring notes
        if current_index > 0:
            movement_options.extend([self.scale_notes[current_index - 1]] * 3)  # Down step
        if current_index < len(self.scale_notes) - 1:
            movement_options.extend([self.scale_notes[current_index + 1]] * 3)  # Up step
            
        # Add small jumps (2-3 scale steps)
        for jump in [-3, -2, 2, 3]:
            target_index = current_index + jump
            if 0 <= target_index < len(self.scale_notes):
                movement_options.append(self.scale_notes[target_index])
        
        # Add current note (repeat)
        movement_options.extend([current_note] * 2)
        
        return random.choice(movement_options) if movement_options else current_note
    
    def _get_note_velocity(self, step: int, is_accent: bool) -> int:
        """Get velocity for note based on accent pattern and randomization"""
        if is_accent:
            # Accented note - high velocity for filter sweep effect
            base_vel = self.settings.accent_velocity
            variation = random.randint(-10, 5)
        else:
            # Normal note
            base_vel = self.settings.normal_velocity  
            variation = random.randint(-15, 15)
            
        # Occasionally add ghost notes (very quiet)
        if not is_accent and random.random() < 0.1:
            base_vel = self.settings.ghost_velocity
            variation = random.randint(-5, 5)
        
        return max(1, min(127, base_vel + variation))
    
    def _get_note_duration(self, step: int, is_accent: bool) -> float:
        """Get note duration based on position and accent"""
        
        # Accented notes often longer for filter sweep
        if is_accent and random.random() < 0.6:
            return self.settings.long_note
            
        # Mostly 16th notes with some variation
        duration_weights = [
            (self.settings.short_note, 1),    # 32nd note
            (self.settings.normal_note, 4),   # 16th note (most common)
            (self.settings.long_note, 2),     # 8th note
        ]
        
        durations, weights = zip(*duration_weights)
        return random.choices(durations, weights=weights)[0]
    
    def generate(self, length_bars: float = 4.0, bpm: float = 180.0) -> MIDIClip:
        """Generate acid bassline MIDI clip"""
        
        # Create clip
        clip = MIDIClip(
            name=f"acid_bassline_{self.scale.value}",
            length_bars=length_bars,
            time_signature=TimeSignature(4, 4),
            key_signature=self.scale.value,
            bpm=bpm
        )
        clip.tags = ["acid", "bassline", "303", "hardcore"]
        
        # Calculate timing
        beats_per_bar = 4.0
        total_beats = length_bars * beats_per_bar
        step_size = 0.25  # 16th notes
        num_steps = int(total_beats / step_size)
        
        # Generate sequence
        current_note = None
        for step in range(num_steps):
            step_time = step * step_size
            
            # Check if this step should have a note
            if random.random() < self.settings.note_density:
                
                # Choose note
                note_pitch = self._choose_next_note(current_note, step)
                if note_pitch is None:
                    continue  # Rest
                    
                current_note = note_pitch
                
                # Determine accent and velocity
                is_accent = self._get_accent_at_step(step)
                velocity = self._get_note_velocity(step, is_accent)
                duration = self._get_note_duration(step, is_accent)
                
                # Create MIDI note
                note = MIDINote(
                    pitch=note_pitch,
                    velocity=velocity,
                    start_time=step_time,
                    duration=duration,
                    channel=0  # Bass channel
                )
                
                clip.add_note(note)
        
        return clip
    
    def generate_variations(self, base_clip: MIDIClip, num_variations: int = 3) -> List[MIDIClip]:
        """Generate variations of a base acid bassline"""
        variations = []
        
        for i in range(num_variations):
            # Create variation with different settings
            var_settings = AcidSettings(
                scale=self.settings.scale,
                note_density=self.settings.note_density + random.uniform(-0.1, 0.1),
                accent_probability=self.settings.accent_probability + random.uniform(-0.1, 0.1),
                slide_probability=self.settings.slide_probability + random.uniform(-0.1, 0.1),
                octave_jump_probability=self.settings.octave_jump_probability + random.uniform(-0.05, 0.05)
            )
            
            # Create new generator with variation settings
            var_generator = AcidBasslineGenerator(
                scale=self.scale.value,
                accent_pattern=self.accent_pattern,
                settings=var_settings
            )
            
            # Generate variation
            variation = var_generator.generate(base_clip.length_bars, base_clip.bpm)
            variation.name = f"{base_clip.name}_var_{i+1}"
            variations.append(variation)
        
        return variations


# Convenience functions for quick generation
def create_classic_acid_line(length_bars: float = 4.0, bpm: float = 140.0) -> MIDIClip:
    """Create classic acid house style bassline"""
    generator = AcidBasslineGenerator(
        scale="A_minor",
        accent_pattern="x ~ ~ x ~ ~ x ~"
    )
    return generator.generate(length_bars, bpm)


def create_hardcore_acid_line(length_bars: float = 2.0, bpm: float = 180.0) -> MIDIClip:
    """Create aggressive hardcore acid bassline"""
    settings = AcidSettings(
        scale=AcidScale.E_MINOR,
        note_density=0.85,  # More notes
        accent_probability=0.4,  # More accents
        octave_jump_probability=0.25,  # More jumps
        accent_velocity=127,  # Max velocity accents
    )
    
    generator = AcidBasslineGenerator(
        scale="E_minor",
        accent_pattern="x ~ x ~ x x ~ x",  # More aggressive pattern
        settings=settings
    )
    return generator.generate(length_bars, bpm)


def create_minimal_acid_line(length_bars: float = 8.0, bpm: float = 130.0) -> MIDIClip:
    """Create minimal/hypnotic acid bassline"""
    settings = AcidSettings(
        scale=AcidScale.A_MINOR_PENTATONIC,
        note_density=0.6,  # Fewer notes
        accent_probability=0.2,  # Fewer accents
        octave_jump_probability=0.1,  # Minimal jumps
        rest_probability=0.2,  # More space
    )
    
    generator = AcidBasslineGenerator(
        scale="A_minor_pentatonic", 
        accent_pattern="x ~ ~ ~ x ~ ~ ~",  # Minimal pattern
        settings=settings
    )
    return generator.generate(length_bars, bpm)