#!/usr/bin/env python3
"""
Musical Context System - Project-level musical awareness
Provides harmonic cohesion and key/scale awareness across all tracks
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import existing MIDI utilities
from .models.midi_clips import note_name_to_midi, midi_to_note_name


class ScaleType(Enum):
    """Common scales used in electronic music"""
    MAJOR = "major"
    NATURAL_MINOR = "natural_minor"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"
    MINOR_PENTATONIC = "minor_pentatonic"
    MAJOR_PENTATONIC = "major_pentatonic"
    BLUES = "blues"
    CHROMATIC = "chromatic"
    # Hardcore-specific scales
    PHRYGIAN_DOMINANT = "phrygian_dominant"  # Common in industrial
    HUNGARIAN_MINOR = "hungarian_minor"  # Dark, exotic


# Scale intervals from root (in semitones)
SCALE_INTERVALS = {
    ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
    ScaleType.NATURAL_MINOR: [0, 2, 3, 5, 7, 8, 10],
    ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
    ScaleType.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],  # Ascending
    ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
    ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
    ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
    ScaleType.MINOR_PENTATONIC: [0, 3, 5, 7, 10],
    ScaleType.MAJOR_PENTATONIC: [0, 2, 4, 7, 9],
    ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
    ScaleType.CHROMATIC: list(range(12)),
    ScaleType.PHRYGIAN_DOMINANT: [0, 1, 4, 5, 7, 8, 10],
    ScaleType.HUNGARIAN_MINOR: [0, 2, 3, 6, 7, 8, 11],
}


class ChordType(Enum):
    """Common chord types"""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    MAJOR_7 = "maj7"
    MINOR_7 = "m7"
    DOMINANT_7 = "7"
    HALF_DIM_7 = "m7b5"
    DIM_7 = "dim7"
    SUS2 = "sus2"
    SUS4 = "sus4"
    POWER = "5"  # Power chord (root + fifth)
    # Extended chords
    ADD9 = "add9"
    MINOR_ADD9 = "madd9"
    SIXTH = "6"
    MINOR_6 = "m6"


# Chord intervals from root (in semitones)
CHORD_INTERVALS = {
    ChordType.MAJOR: [0, 4, 7],
    ChordType.MINOR: [0, 3, 7],
    ChordType.DIMINISHED: [0, 3, 6],
    ChordType.AUGMENTED: [0, 4, 8],
    ChordType.MAJOR_7: [0, 4, 7, 11],
    ChordType.MINOR_7: [0, 3, 7, 10],
    ChordType.DOMINANT_7: [0, 4, 7, 10],
    ChordType.HALF_DIM_7: [0, 3, 6, 10],
    ChordType.DIM_7: [0, 3, 6, 9],
    ChordType.SUS2: [0, 2, 7],
    ChordType.SUS4: [0, 5, 7],
    ChordType.POWER: [0, 7],
    ChordType.ADD9: [0, 4, 7, 14],
    ChordType.MINOR_ADD9: [0, 3, 7, 14],
    ChordType.SIXTH: [0, 4, 7, 9],
    ChordType.MINOR_6: [0, 3, 7, 9],
}


# Note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_midi(note_name: str) -> int:
    """Convert note name to MIDI number (e.g., 'C4' -> 60)"""
    # Parse note and octave
    if len(note_name) < 2:
        raise ValueError(f"Invalid note name: {note_name}")
    
    base_note = note_name[:-1].upper()
    octave = int(note_name[-1])
    
    # Handle flats
    if 'B' in base_note and base_note != 'B':
        base_note = base_note.replace('B', '')
        base_idx = NOTE_NAMES.index(base_note) - 1
    else:
        base_idx = NOTE_NAMES.index(base_note.replace('#', '#'))
    
    return base_idx + (octave + 1) * 12


def midi_to_note(midi_num: int) -> str:
    """Convert MIDI number to note name (e.g., 60 -> 'C4')"""
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"


@dataclass
class Chord:
    """Represents a chord in the progression"""
    root: int  # MIDI note number
    chord_type: ChordType
    inversion: int = 0  # 0=root position, 1=first inversion, etc.
    duration_bars: float = 1.0
    
    def get_notes(self) -> List[int]:
        """Get MIDI notes for this chord"""
        intervals = CHORD_INTERVALS[self.chord_type]
        notes = [self.root + interval for interval in intervals]
        
        # Apply inversion
        if self.inversion > 0:
            for i in range(min(self.inversion, len(notes))):
                notes[i] += 12
            notes.sort()
        
        return notes
    
    def get_scale_degree(self, key_root: int) -> int:
        """Get scale degree of chord root relative to key"""
        return (self.root - key_root) % 12


@dataclass
class Scale:
    """Represents a musical scale"""
    root: int  # MIDI note number
    scale_type: ScaleType
    
    def get_notes(self, octave_range: Tuple[int, int] = (0, 8)) -> List[int]:
        """Get all scale notes within octave range"""
        intervals = SCALE_INTERVALS[self.scale_type]
        notes = []
        
        for octave in range(octave_range[0], octave_range[1]):
            octave_root = self.root + (octave * 12)
            for interval in intervals:
                note = octave_root + interval
                if 0 <= note <= 127:  # Valid MIDI range
                    notes.append(note)
        
        return sorted(notes)
    
    def contains_note(self, midi_note: int) -> bool:
        """Check if a note is in the scale"""
        note_class = midi_note % 12
        root_class = self.root % 12
        relative_note = (note_class - root_class) % 12
        return relative_note in SCALE_INTERVALS[self.scale_type]
    
    def get_degree(self, midi_note: int) -> Optional[int]:
        """Get scale degree of note (1-based), None if not in scale"""
        note_class = midi_note % 12
        root_class = self.root % 12
        relative_note = (note_class - root_class) % 12
        
        intervals = SCALE_INTERVALS[self.scale_type]
        if relative_note in intervals:
            return intervals.index(relative_note) + 1
        return None


@dataclass
class TimeSignature:
    """Time signature representation"""
    numerator: int = 4
    denominator: int = 4
    
    def beats_per_bar(self) -> float:
        return float(self.numerator)
    
    def beat_duration(self) -> float:
        """Duration of one beat in quarter notes"""
        return 4.0 / self.denominator


@dataclass
class KeyChange:
    """Represents a key change at a specific point"""
    bar: int
    new_key: str
    new_scale: Scale
    transition_type: str = "direct"  # "direct", "pivot", "chromatic"


@dataclass
class MusicalContext:
    """
    Project-level musical context for harmonic cohesion
    
    Provides the harmonic framework for all tracks in a project
    """
    key: str  # e.g., "Am", "C", "F#m"
    scale: Scale
    tempo: float = 180.0
    time_signature: TimeSignature = field(default_factory=TimeSignature)
    
    # Chord progression (optional)
    chord_progression: List[Chord] = field(default_factory=list)
    progression_length_bars: int = 4
    
    # Section changes (for key modulations)
    section_changes: Dict[str, KeyChange] = field(default_factory=dict)
    
    # Genre hint for style-appropriate choices
    genre: str = "hardcore"
    
    def __post_init__(self):
        """Initialize scale from key if not provided"""
        if isinstance(self.scale, str):
            # Parse key string
            root_note = self.key.rstrip('m')
            is_minor = self.key.endswith('m')
            
            try:
                root_midi = note_to_midi(root_note + "3")
            except:
                # Try without octave
                root_midi = NOTE_NAMES.index(root_note) + 48  # Default to octave 3
            
            scale_type = ScaleType.NATURAL_MINOR if is_minor else ScaleType.MAJOR
            self.scale = Scale(root_midi, scale_type)
    
    def get_current_chord(self, bar: float) -> Optional[Chord]:
        """Get the current chord at a given bar position"""
        if not self.chord_progression:
            return None
        
        position_in_progression = bar % self.progression_length_bars
        accumulated_duration = 0.0
        
        for chord in self.chord_progression:
            if accumulated_duration <= position_in_progression < accumulated_duration + chord.duration_bars:
                return chord
            accumulated_duration += chord.duration_bars
        
        return self.chord_progression[0] if self.chord_progression else None
    
    def get_current_key(self, bar: int) -> Scale:
        """Get the current key/scale at a given bar (accounts for key changes)"""
        current_scale = self.scale
        
        for section_name, key_change in self.section_changes.items():
            if bar >= key_change.bar:
                current_scale = key_change.new_scale
        
        return current_scale


class HarmonicGuide:
    """
    Ensures harmonic cohesion across tracks
    
    Provides note validation and suggestions based on musical context
    """
    
    def __init__(self, context: MusicalContext):
        self.context = context
    
    def get_valid_notes(self, bar: float, octave_range: Tuple[int, int] = (0, 8)) -> List[int]:
        """Get all valid notes for the current bar position"""
        current_scale = self.context.get_current_key(int(bar))
        return current_scale.get_notes(octave_range)
    
    def get_chord_tones(self, bar: float) -> List[int]:
        """Get chord tones for the current bar"""
        current_chord = self.context.get_current_chord(bar)
        if current_chord:
            return current_chord.get_notes()
        
        # If no chord progression, return root triad
        root = self.context.scale.root
        if self.context.scale.scale_type in [ScaleType.NATURAL_MINOR, ScaleType.HARMONIC_MINOR]:
            return [root, root + 3, root + 7]  # Minor triad
        else:
            return [root, root + 4, root + 7]  # Major triad
    
    def get_safe_notes(self, bar: float, octave_range: Tuple[int, int] = (0, 8)) -> List[int]:
        """Get 'safe' notes (chord tones + scale tones)"""
        chord_tones = set(self.get_chord_tones(bar))
        scale_notes = set(self.get_valid_notes(bar, octave_range))
        
        # Expand chord tones to all octaves
        expanded_chord_tones = set()
        for note in chord_tones:
            note_class = note % 12
            for octave in range(octave_range[0], octave_range[1]):
                expanded_note = note_class + (octave * 12)
                if 0 <= expanded_note <= 127:
                    expanded_chord_tones.add(expanded_note)
        
        # Chord tones are safest, then scale notes
        return sorted(list(expanded_chord_tones | scale_notes))
    
    def suggest_passing_notes(self, from_note: int, to_note: int) -> List[int]:
        """Suggest passing notes between two notes"""
        current_scale = self.context.scale
        scale_notes = current_scale.get_notes()
        
        # Find notes between from and to
        min_note = min(from_note, to_note)
        max_note = max(from_note, to_note)
        
        passing_notes = [n for n in scale_notes if min_note < n < max_note]
        
        # For chromatic genres, add chromatic passing notes
        if self.context.genre in ["industrial", "experimental"]:
            for n in range(min_note + 1, max_note):
                if n not in passing_notes:
                    passing_notes.append(n)
        
        return sorted(passing_notes)
    
    def validate_note_choice(self, note: int, bar: float, 
                           strict: bool = False) -> Tuple[bool, str]:
        """
        Validate if a note choice is appropriate
        
        Args:
            note: MIDI note number
            bar: Current bar position
            strict: If True, only allow scale notes. If False, allow chromatic
        
        Returns:
            (is_valid, explanation)
        """
        current_scale = self.context.get_current_key(int(bar))
        current_chord = self.context.get_current_chord(bar)
        
        # Check if in scale
        if current_scale.contains_note(note):
            # Check if it's a chord tone
            if current_chord and note % 12 in [n % 12 for n in current_chord.get_notes()]:
                return True, "Chord tone - perfect choice"
            return True, "Scale tone - good choice"
        
        # Not in scale
        if strict:
            return False, f"Note {midi_to_note(note)} not in {self.context.key} scale"
        
        # Check for common chromatic uses
        scale_notes = current_scale.get_notes((0, 8))
        note_class = note % 12
        
        # Check if it's a leading tone (semitone below scale note)
        for scale_note in scale_notes:
            if (scale_note % 12) == (note_class + 1) % 12:
                return True, "Chromatic leading tone - adds tension"
        
        # For hardcore/industrial, allow more chromaticism
        if self.context.genre in ["hardcore", "gabber", "industrial"]:
            return True, "Chromatic note - adds edge/dissonance"
        
        return False, f"Note {midi_to_note(note)} creates dissonance with {self.context.key}"
    
    def get_tension_notes(self, bar: float) -> List[int]:
        """Get notes that create tension (for builds/drops)"""
        current_scale = self.context.scale
        root = current_scale.root % 12
        
        # Tension intervals: b9, #11, b13
        tension_intervals = [1, 6, 8]  # Semitones from root
        
        tension_notes = []
        for octave in range(2, 7):  # Reasonable range
            for interval in tension_intervals:
                note = (root + interval) + (octave * 12)
                if 0 <= note <= 127:
                    tension_notes.append(note)
        
        return tension_notes
    
    def suggest_bass_note(self, bar: float) -> int:
        """Suggest appropriate bass note for current bar"""
        current_chord = self.context.get_current_chord(bar)
        
        if current_chord:
            # Use chord root in bass octave
            root_class = current_chord.root % 12
            return root_class + 24  # C1
        else:
            # Use scale root
            root_class = self.context.scale.root % 12
            return root_class + 24  # C1
    
    def analyze_harmonic_density(self, notes: List[int]) -> Dict[str, any]:
        """Analyze harmonic density of a set of notes"""
        if not notes:
            return {"density": 0, "dissonance": 0, "description": "Empty"}
        
        # Count unique pitch classes
        pitch_classes = set(n % 12 for n in notes)
        
        # Check intervals for dissonance
        dissonant_intervals = [1, 2, 6, 10, 11]  # Semitones
        dissonance_count = 0
        
        for i, pc1 in enumerate(pitch_classes):
            for pc2 in list(pitch_classes)[i+1:]:
                interval = abs(pc1 - pc2)
                if interval in dissonant_intervals:
                    dissonance_count += 1
        
        density = len(pitch_classes) / 12.0
        dissonance = dissonance_count / max(1, len(pitch_classes))
        
        # Description
        if density < 0.25:
            density_desc = "sparse"
        elif density < 0.5:
            density_desc = "moderate"
        else:
            density_desc = "dense"
        
        if dissonance < 0.2:
            dissonance_desc = "consonant"
        elif dissonance < 0.5:
            dissonance_desc = "mildly dissonant"
        else:
            dissonance_desc = "highly dissonant"
        
        return {
            "density": density,
            "dissonance": dissonance,
            "description": f"{density_desc}, {dissonance_desc}",
            "pitch_classes": len(pitch_classes),
            "suggested_action": self._suggest_harmonic_action(density, dissonance)
        }
    
    def _suggest_harmonic_action(self, density: float, dissonance: float) -> str:
        """Suggest action based on harmonic analysis"""
        if density > 0.7:
            return "Consider removing some notes for clarity"
        elif density < 0.2:
            return "Could add more harmonic content"
        elif dissonance > 0.7:
            return "High dissonance - ensure this is intentional"
        else:
            return "Harmonic balance is good"


# Example chord progressions for different genres
GENRE_PROGRESSIONS = {
    "hardcore": [
        # i - iv - v - i (minor)
        [
            Chord(0, ChordType.MINOR),
            Chord(5, ChordType.MINOR),
            Chord(7, ChordType.MINOR),
            Chord(0, ChordType.MINOR),
        ],
        # i - VII - VI - VII (epic/anthem)
        [
            Chord(0, ChordType.MINOR),
            Chord(-2, ChordType.MAJOR),
            Chord(-4, ChordType.MAJOR),
            Chord(-2, ChordType.MAJOR),
        ],
    ],
    "industrial": [
        # Single chord drone (i)
        [Chord(0, ChordType.POWER, duration_bars=4)],
        # i - bII (Phrygian motion)
        [
            Chord(0, ChordType.MINOR, duration_bars=2),
            Chord(1, ChordType.MAJOR, duration_bars=2),
        ],
    ],
    "gabber": [
        # Often just root note emphasis
        [Chord(0, ChordType.POWER, duration_bars=4)],
        # i - v - i - v (simple minor)
        [
            Chord(0, ChordType.MINOR),
            Chord(7, ChordType.MINOR),
            Chord(0, ChordType.MINOR),
            Chord(7, ChordType.MINOR),
        ],
    ],
}


def create_context_from_genre(genre: str, key: str = "Am", tempo: float = 180.0) -> MusicalContext:
    """Create a musical context preset for a genre"""
    progressions = GENRE_PROGRESSIONS.get(genre.lower(), GENRE_PROGRESSIONS["hardcore"])
    
    # Parse key
    root_note = key.rstrip('m')
    is_minor = key.endswith('m')
    
    try:
        root_midi = note_to_midi(root_note + "3")
    except:
        root_midi = 57  # Default to A3
    
    scale_type = ScaleType.NATURAL_MINOR if is_minor else ScaleType.MAJOR
    
    # Adjust progression to key
    progression = []
    for chord in progressions[0]:  # Use first progression
        adjusted_chord = Chord(
            root=root_midi + chord.root,
            chord_type=chord.chord_type,
            inversion=chord.inversion,
            duration_bars=chord.duration_bars
        )
        progression.append(adjusted_chord)
    
    return MusicalContext(
        key=key,
        scale=Scale(root_midi, scale_type),
        tempo=tempo,
        chord_progression=progression,
        progression_length_bars=4,
        genre=genre
    )


# Testing
if __name__ == "__main__":
    # Create a context for Am hardcore
    context = create_context_from_genre("hardcore", "Am", 180)
    guide = HarmonicGuide(context)
    
    print(f"Key: {context.key}")
    print(f"Scale notes: {[midi_to_note(n) for n in context.scale.get_notes((3, 5))]}")
    
    # Get valid notes for bar 0
    valid_notes = guide.get_valid_notes(0, (3, 5))
    print(f"\nValid notes at bar 0: {[midi_to_note(n) for n in valid_notes[:12]]}")
    
    # Get chord tones
    chord_tones = guide.get_chord_tones(0)
    print(f"Chord tones: {[midi_to_note(n) for n in chord_tones]}")
    
    # Validate some notes
    test_notes = [57, 60, 61, 62]  # A3, C4, C#4, D4
    for note in test_notes:
        valid, reason = guide.validate_note_choice(note, 0)
        print(f"{midi_to_note(note)}: {valid} - {reason}")
    
    # Analyze harmony
    analysis = guide.analyze_harmonic_density([57, 60, 64, 67])
    print(f"\nHarmonic analysis: {analysis}")