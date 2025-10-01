#!/usr/bin/env python3
"""
MIDI Clip System for Hardcore Music Production

Standard MIDI clip representation that can output to multiple formats:
- TidalCycles patterns for rhythmic complexity
- Standard MIDI files for DAW integration
- OSC messages for direct SuperCollider control
"""

import time
import json
import math
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    mido = None

import numpy as np
from .hardcore_models import SynthType


class NoteValue(Enum):
    """Standard musical note values"""
    WHOLE = 1.0
    HALF = 0.5
    QUARTER = 0.25
    EIGHTH = 0.125
    SIXTEENTH = 0.0625
    THIRTY_SECOND = 0.03125
    
    # Dotted notes
    DOTTED_HALF = 0.75
    DOTTED_QUARTER = 0.375
    DOTTED_EIGHTH = 0.1875
    
    # Triplets
    QUARTER_TRIPLET = 1/6
    EIGHTH_TRIPLET = 1/12


@dataclass
class MIDINote:
    """Single MIDI note with timing and velocity"""
    pitch: int                    # MIDI note number (0-127)
    velocity: int                 # MIDI velocity (0-127)  
    start_time: float            # Start time in beats from clip start
    duration: float              # Duration in beats
    channel: int = 0             # MIDI channel (0-15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MIDINote':
        """Create from dictionary"""
        return cls(**data)
    
    def transpose(self, semitones: int) -> 'MIDINote':
        """Transpose note by semitones"""
        new_note = MIDINote(
            pitch=max(0, min(127, self.pitch + semitones)),
            velocity=self.velocity,
            start_time=self.start_time,
            duration=self.duration,
            channel=self.channel
        )
        return new_note
    
    def to_frequency(self) -> float:
        """Convert MIDI note to frequency in Hz"""
        return 440.0 * (2 ** ((self.pitch - 69) / 12.0))


@dataclass  
class Trigger:
    """Single trigger for drums/samples without pitch"""
    sample_id: str               # Sample/drum identifier
    velocity: int                # Trigger velocity (0-127)
    start_time: float           # Start time in beats from clip start
    channel: int = 9            # Default to MIDI channel 10 (drums)
    probability: float = 1.0    # Trigger probability (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trigger':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TimeSignature:
    """Time signature representation"""
    numerator: int = 4           # Beats per bar
    denominator: int = 4         # Note value per beat
    
    def beats_per_bar(self) -> float:
        """Get beats per bar as float"""
        return float(self.numerator)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"numerator": self.numerator, "denominator": self.denominator}


class MIDIClip:
    """
    Standard MIDI clip for melodic/harmonic content
    
    Can export to multiple formats while maintaining MIDI compatibility.
    Designed for riffs, basslines, chord progressions, arpeggios, etc.
    """
    
    def __init__(self, 
                 name: str = "untitled_clip",
                 length_bars: float = 4.0,
                 time_signature: TimeSignature = None,
                 key_signature: str = "C",
                 bpm: float = 120.0):
        self.name = name
        self.length_bars = length_bars
        self.time_signature = time_signature or TimeSignature()
        self.key_signature = key_signature  # "C", "Am", "F#m", etc.
        self.bpm = bpm
        self.notes: List[MIDINote] = []
        
        # Metadata
        self.created_at = time.time()
        self.tags: List[str] = []
        self.genre = "hardcore"
        
    def add_note(self, note: MIDINote) -> None:
        """Add a note to the clip"""
        self.notes.append(note)
        
    def add_notes(self, notes: List[MIDINote]) -> None:
        """Add multiple notes to the clip"""
        self.notes.extend(notes)
    
    def clear_notes(self) -> None:
        """Clear all notes from clip"""
        self.notes.clear()
        
    def get_notes_at_time(self, beat_time: float, tolerance: float = 0.01) -> List[MIDINote]:
        """Get all notes that start at given beat time"""
        return [note for note in self.notes 
                if abs(note.start_time - beat_time) < tolerance]
    
    def get_notes_in_range(self, start_beat: float, end_beat: float) -> List[MIDINote]:
        """Get all notes in time range"""
        return [note for note in self.notes
                if start_beat <= note.start_time < end_beat]
    
    def transpose(self, semitones: int) -> 'MIDIClip':
        """Create transposed copy of clip"""
        new_clip = MIDIClip(
            name=f"{self.name}_transposed_{semitones:+d}",
            length_bars=self.length_bars,
            time_signature=self.time_signature,
            key_signature=self.key_signature,
            bpm=self.bpm
        )
        new_clip.notes = [note.transpose(semitones) for note in self.notes]
        new_clip.tags = self.tags.copy()
        new_clip.genre = self.genre
        return new_clip
    
    def quantize(self, grid: float = 0.25) -> None:
        """Quantize note start times to grid (in beats)"""
        for note in self.notes:
            note.start_time = round(note.start_time / grid) * grid
    
    def get_total_beats(self) -> float:
        """Get total length in beats"""
        return self.length_bars * self.time_signature.beats_per_bar()
    
    def to_tidal_pattern(self) -> str:
        """Convert to TidalCycles pattern string"""
        if not self.notes:
            return "silence"
        
        # Create a basic pattern representation
        # This is a simplified version - full implementation would be more complex
        beats_per_bar = self.time_signature.beats_per_bar()
        total_beats = self.get_total_beats()
        
        # Group notes by beat positions for pattern creation
        pattern_steps = []
        step_size = 0.25  # 16th notes
        num_steps = int(total_beats / step_size)
        
        for step in range(num_steps):
            step_time = step * step_size
            notes_at_step = self.get_notes_at_time(step_time)
            
            if notes_at_step:
                # Convert MIDI notes to frequencies or note names
                if len(notes_at_step) == 1:
                    # Single note
                    note = notes_at_step[0]
                    freq = note.to_frequency()
                    pattern_steps.append(f"{freq:.1f}")
                else:
                    # Chord - use bracket notation
                    freqs = [note.to_frequency() for note in notes_at_step]
                    freq_str = " ".join(f"{f:.1f}" for f in freqs)
                    pattern_steps.append(f"[{freq_str}]")
            else:
                pattern_steps.append("~")  # Rest
        
        # Group into bars and join
        pattern = " ".join(pattern_steps)
        return f"sound \"superpiano\" # note \"{pattern}\""
    
    def to_midi_file(self) -> 'mido.MidiFile':
        """Export as standard MIDI file"""
        if not MIDO_AVAILABLE:
            raise ImportError("mido library required for MIDI export")
        
        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo (microseconds per beat)
        tempo = int(60000000 / self.bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        
        # Set time signature
        track.append(mido.MetaMessage('time_signature',
                                    numerator=self.time_signature.numerator,
                                    denominator=self.time_signature.denominator,
                                    time=0))
        
        # Convert notes to MIDI messages
        # Sort notes by start time
        sorted_notes = sorted(self.notes, key=lambda n: n.start_time)
        
        # Track current time in ticks
        ticks_per_beat = mid.ticks_per_beat
        current_ticks = 0
        
        # Create note on/off events
        events = []
        for note in sorted_notes:
            start_ticks = int(note.start_time * ticks_per_beat)
            end_ticks = int((note.start_time + note.duration) * ticks_per_beat)
            
            events.append((start_ticks, 'note_on', note))
            events.append((end_ticks, 'note_off', note))
        
        # Sort events by time
        events.sort(key=lambda e: e[0])
        
        # Convert to MIDI messages with delta times
        for event_time, event_type, note in events:
            delta_time = event_time - current_ticks
            current_ticks = event_time
            
            if event_type == 'note_on':
                track.append(mido.Message('note_on', 
                                        channel=note.channel,
                                        note=note.pitch,
                                        velocity=note.velocity,
                                        time=delta_time))
            else:  # note_off
                track.append(mido.Message('note_off',
                                        channel=note.channel, 
                                        note=note.pitch,
                                        velocity=0,
                                        time=delta_time))
        
        return mid
    
    def save_midi_file(self, filepath: str) -> bool:
        """Save as MIDI file"""
        try:
            mid = self.to_midi_file()
            mid.save(filepath)
            return True
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "length_bars": self.length_bars,
            "time_signature": self.time_signature.to_dict(),
            "key_signature": self.key_signature,
            "bpm": self.bpm,
            "notes": [note.to_dict() for note in self.notes],
            "created_at": self.created_at,
            "tags": self.tags,
            "genre": self.genre
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MIDIClip':
        """Create from dictionary"""
        time_sig_data = data.pop("time_signature", {"numerator": 4, "denominator": 4})
        time_sig = TimeSignature(**time_sig_data)
        
        notes_data = data.pop("notes", [])
        clip = cls(time_signature=time_sig, **{k: v for k, v in data.items() 
                                              if k not in ["created_at", "tags", "genre"]})
        
        # Restore metadata
        clip.created_at = data.get("created_at", time.time())
        clip.tags = data.get("tags", [])
        clip.genre = data.get("genre", "hardcore")
        
        # Add notes
        for note_data in notes_data:
            clip.add_note(MIDINote.from_dict(note_data))
        
        return clip


class TriggerClip:
    """
    Trigger-based clip for drums/samples without pitch information
    
    Simpler than MIDIClip - just timing and velocity.
    Designed for kick patterns, hi-hats, percussion, etc.
    """
    
    def __init__(self,
                 name: str = "untitled_triggers", 
                 length_bars: float = 4.0,
                 time_signature: TimeSignature = None,
                 bpm: float = 120.0):
        self.name = name
        self.length_bars = length_bars
        self.time_signature = time_signature or TimeSignature()
        self.bpm = bpm
        self.triggers: List[Trigger] = []
        
        # Metadata
        self.created_at = time.time()
        self.tags: List[str] = []
        self.genre = "hardcore"
        
    def add_trigger(self, trigger: Trigger) -> None:
        """Add a trigger to the clip"""
        self.triggers.append(trigger)
        
    def add_triggers(self, triggers: List[Trigger]) -> None:
        """Add multiple triggers to the clip"""
        self.triggers.extend(triggers)
    
    def clear_triggers(self) -> None:
        """Clear all triggers from clip"""
        self.triggers.clear()
        
    def get_triggers_at_time(self, beat_time: float, tolerance: float = 0.01) -> List[Trigger]:
        """Get all triggers that start at given beat time"""
        return [trigger for trigger in self.triggers 
                if abs(trigger.start_time - beat_time) < tolerance]
    
    def get_total_beats(self) -> float:
        """Get total length in beats"""
        return self.length_bars * self.time_signature.beats_per_bar()
    
    def to_tidal_pattern(self, sample_map: Dict[str, str] = None) -> str:
        """Convert to TidalCycles pattern string"""
        if not self.triggers:
            return "silence"
        
        # Default sample mapping
        default_map = {
            "kick": "bd",
            "snare": "sn", 
            "hihat": "hh",
            "openhat": "oh",
            "crash": "cy"
        }
        sample_map = sample_map or default_map
        
        # Create pattern representation
        total_beats = self.get_total_beats()
        step_size = 0.25  # 16th notes
        num_steps = int(total_beats / step_size)
        
        pattern_steps = []
        for step in range(num_steps):
            step_time = step * step_size
            triggers_at_step = self.get_triggers_at_time(step_time)
            
            if triggers_at_step:
                # Convert trigger to sample name
                trigger = triggers_at_step[0]  # Take first if multiple
                sample_name = sample_map.get(trigger.sample_id, trigger.sample_id)
                pattern_steps.append(sample_name)
            else:
                pattern_steps.append("~")  # Rest
        
        pattern = " ".join(pattern_steps)
        return f"sound \"{pattern}\""
    
    def to_pattern_string(self) -> str:
        """Convert to simple pattern string (x ~ x ~ format)"""
        total_beats = self.get_total_beats()
        step_size = 0.25  # 16th notes
        num_steps = int(total_beats / step_size)
        
        pattern_chars = []
        for step in range(num_steps):
            step_time = step * step_size
            triggers_at_step = self.get_triggers_at_time(step_time)
            pattern_chars.append("x" if triggers_at_step else "~")
        
        return " ".join(pattern_chars)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "length_bars": self.length_bars,
            "time_signature": self.time_signature.to_dict(),
            "bpm": self.bpm,
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "created_at": self.created_at,
            "tags": self.tags,
            "genre": self.genre
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerClip':
        """Create from dictionary"""
        time_sig_data = data.pop("time_signature", {"numerator": 4, "denominator": 4})
        time_sig = TimeSignature(**time_sig_data)
        
        triggers_data = data.pop("triggers", [])
        clip = cls(time_signature=time_sig, **{k: v for k, v in data.items() 
                                              if k not in ["created_at", "tags", "genre"]})
        
        # Restore metadata
        clip.created_at = data.get("created_at", time.time())
        clip.tags = data.get("tags", [])
        clip.genre = data.get("genre", "hardcore")
        
        # Add triggers
        for trigger_data in triggers_data:
            clip.add_trigger(Trigger.from_dict(trigger_data))
        
        return clip


# Utility functions for working with clips
def create_empty_midi_clip(name: str = "new_clip", bars: float = 4.0, bpm: float = 180.0) -> MIDIClip:
    """Create an empty MIDI clip ready for notes"""
    return MIDIClip(name=name, length_bars=bars, bpm=bpm)


def create_empty_trigger_clip(name: str = "new_triggers", bars: float = 4.0, bpm: float = 180.0) -> TriggerClip:
    """Create an empty trigger clip ready for drum patterns"""
    return TriggerClip(name=name, length_bars=bars, bpm=bpm)


# MIDI note name to number conversion
NOTE_NAMES = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

def note_name_to_midi(note_name: str) -> int:
    """Convert note name like 'C4' to MIDI number"""
    if len(note_name) < 2:
        raise ValueError(f"Invalid note name: {note_name}")
    
    # Parse note and octave
    if note_name[1] in ['#', 'b']:
        note = note_name[:2]
        octave = int(note_name[2:])
    else:
        note = note_name[0]
        octave = int(note_name[1:])
    
    if note not in NOTE_NAMES:
        raise ValueError(f"Unknown note: {note}")
    
    return (octave + 1) * 12 + NOTE_NAMES[note]


def midi_to_note_name(midi_number: int) -> str:
    """Convert MIDI number to note name like 'C4'"""
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{note_names[note_index]}{octave}"