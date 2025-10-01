"""
Core musical data models for the Hardcore Music Production System.

These Pydantic models define the fundamental data structures for MIDI clips,
notes, and automation following the Architecture Specification.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class MIDINote(BaseModel):
    """
    A single MIDI note with pitch, velocity, timing, and duration.
    
    Represents a musical note event with validation for MIDI ranges
    and musical timing constraints.
    """
    
    pitch: int = Field(
        ..., 
        ge=0, 
        le=127, 
        description="MIDI note number (0-127, where 60 = Middle C)"
    )
    velocity: int = Field(
        ..., 
        ge=0, 
        le=127, 
        description="MIDI velocity (0-127, where 127 = maximum volume)"
    )
    start_time: float = Field(
        ..., 
        ge=0.0, 
        description="Start time in beats from clip beginning"
    )
    duration: float = Field(
        ..., 
        gt=0.0, 
        description="Note duration in beats"
    )
    channel: int = Field(
        default=0, 
        ge=0, 
        le=15, 
        description="MIDI channel (0-15)"
    )
    
    model_config = ConfigDict(
        json_encoders={float: lambda v: round(v, 6)},
        str_strip_whitespace=True
    )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = self.pitch // 12 - 1
        note = note_names[self.pitch % 12]
        return f"{note}{octave} (vel:{self.velocity}, t:{self.start_time}-{self.start_time + self.duration})"
    
    @property
    def end_time(self) -> float:
        """Calculate the end time of the note."""
        return self.start_time + self.duration
    
    @property
    def frequency_hz(self) -> float:
        """Calculate the frequency in Hz (A4 = 440Hz)."""
        return 440.0 * (2.0 ** ((self.pitch - 69) / 12.0))


class MIDIClip(BaseModel):
    """
    A collection of MIDI notes with timing and metadata.
    
    Represents a musical pattern or sequence that can be played,
    edited, and exported to various formats.
    """
    
    id: str = Field(..., description="Unique identifier for the clip")
    name: str = Field(..., description="Human-readable name")
    notes: List[MIDINote] = Field(
        default_factory=list, 
        description="Collection of MIDI notes in the clip"
    )
    length_bars: float = Field(
        default=4.0, 
        gt=0.0, 
        description="Length of the clip in bars/measures"
    )
    bpm: float = Field(
        default=120.0, 
        ge=60.0, 
        le=300.0, 
        description="Beats per minute (BPM) for the clip"
    )
    time_signature: tuple[int, int] = Field(
        default=(4, 4), 
        description="Time signature as (numerator, denominator)"
    )
    key: Optional[str] = Field(
        default=None, 
        description="Musical key (e.g., 'C', 'Am', 'F#m')"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            tuple: lambda v: list(v)
        },
        str_strip_whitespace=True
    )
    
    @field_validator('time_signature')
    @classmethod
    def validate_time_signature(cls, v: tuple) -> tuple:
        """Validate time signature format."""
        if len(v) != 2:
            raise ValueError("Time signature must be a tuple of (numerator, denominator)")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("Time signature values must be positive")
        if v[1] not in [1, 2, 4, 8, 16]:
            raise ValueError("Time signature denominator must be 1, 2, 4, 8, or 16")
        return v
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate musical key format."""
        if v is None:
            return v
        
        valid_keys = [
            'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
            'Cm', 'C#m', 'Dbm', 'Dm', 'D#m', 'Ebm', 'Em', 'Fm', 'F#m', 'Gbm', 'Gm', 'G#m', 'Abm', 'Am', 'A#m', 'Bbm', 'Bm'
        ]
        
        if v not in valid_keys:
            raise ValueError(f"Invalid key '{v}'. Must be one of: {', '.join(valid_keys)}")
        return v
    
    def add_note(self, note: MIDINote) -> None:
        """Add a MIDI note to the clip."""
        self.notes.append(note)
    
    def remove_note(self, note: MIDINote) -> bool:
        """Remove a MIDI note from the clip. Returns True if found and removed."""
        try:
            self.notes.remove(note)
            return True
        except ValueError:
            return False
    
    @property
    def length_beats(self) -> float:
        """Calculate total length in beats."""
        return self.length_bars * self.time_signature[0]
    
    @property
    def length_seconds(self) -> float:
        """Calculate total length in seconds based on BPM."""
        beats_per_second = self.bpm / 60.0
        return self.length_beats / beats_per_second
    
    @property
    def note_count(self) -> int:
        """Get the total number of notes in the clip."""
        return len(self.notes)
    
    def get_notes_at_time(self, time: float, tolerance: float = 0.001) -> List[MIDINote]:
        """Get all notes that start at or near the specified time."""
        return [
            note for note in self.notes 
            if abs(note.start_time - time) <= tolerance
        ]
    
    def get_notes_in_range(self, start_time: float, end_time: float) -> List[MIDINote]:
        """Get all notes that play within the specified time range."""
        return [
            note for note in self.notes
            if note.start_time < end_time and note.end_time > start_time
        ]
    
    def quantize(self, grid: float = 0.25) -> None:
        """Quantize all note start times to the nearest grid value."""
        for note in self.notes:
            note.start_time = round(note.start_time / grid) * grid
    
    def transpose(self, semitones: int) -> None:
        """Transpose all notes by the specified number of semitones."""
        for note in self.notes:
            new_pitch = note.pitch + semitones
            # Clamp to MIDI range
            note.pitch = max(0, min(127, new_pitch))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert clip to dictionary for JSON serialization."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert clip to JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> MIDIClip:
        """Create clip from JSON string."""
        return cls.model_validate_json(json_str)


class AutomationPoint(BaseModel):
    """
    A single automation point with time and value.
    
    Used for parameter automation over time (e.g., filter cutoff, volume).
    """
    
    time: float = Field(..., ge=0.0, description="Time in beats")
    value: float = Field(..., description="Parameter value at this time")
    
    class Config:
        """Pydantic configuration for AutomationPoint."""
        json_encoders = {
            float: lambda v: round(v, 6)
        }


class AutomationClip(BaseModel):
    """
    A collection of automation points for a specific parameter.
    
    Represents parameter changes over time for effects, synthesis, etc.
    """
    
    id: str = Field(..., description="Unique identifier for the automation clip")
    parameter_name: str = Field(..., description="Name of the parameter being automated")
    points: List[AutomationPoint] = Field(
        default_factory=list, 
        description="Automation points in chronological order"
    )
    min_value: Optional[float] = Field(
        default=None, 
        description="Minimum allowed value for the parameter"
    )
    max_value: Optional[float] = Field(
        default=None, 
        description="Maximum allowed value for the parameter"
    )
    
    @model_validator(mode='after')
    def validate_points_order_and_values(self) -> 'AutomationClip':
        """Ensure automation points are in chronological order and within value range."""
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                if self.points[i].time < self.points[i-1].time:
                    raise ValueError("Automation points must be in chronological order")
        
        # Validate value ranges if min/max specified
        if self.min_value is not None or self.max_value is not None:
            for point in self.points:
                if self.min_value is not None and point.value < self.min_value:
                    raise ValueError(f"Automation value {point.value} below minimum {self.min_value}")
                if self.max_value is not None and point.value > self.max_value:
                    raise ValueError(f"Automation value {point.value} above maximum {self.max_value}")
        
        return self
    
    @field_validator('points')
    @classmethod
    def validate_point_values(cls, v: List[AutomationPoint], info) -> List[AutomationPoint]:
        """Validate automation point values are within range."""
        # Note: Range validation moved to model_validator since it needs access to min/max_value
        return v
    
    def add_point(self, point: AutomationPoint) -> None:
        """Add an automation point, maintaining chronological order."""
        # Insert point in correct chronological position
        insert_index = 0
        for i, existing_point in enumerate(self.points):
            if existing_point.time > point.time:
                insert_index = i
                break
            insert_index = i + 1
        
        self.points.insert(insert_index, point)
    
    def get_value_at_time(self, time: float) -> float:
        """Get the interpolated parameter value at the specified time."""
        if not self.points:
            return 0.0
        
        if time <= self.points[0].time:
            return self.points[0].value
        
        if time >= self.points[-1].time:
            return self.points[-1].value
        
        # Find surrounding points and interpolate
        for i in range(len(self.points) - 1):
            if self.points[i].time <= time <= self.points[i + 1].time:
                t1, v1 = self.points[i].time, self.points[i].value
                t2, v2 = self.points[i + 1].time, self.points[i + 1].value
                
                # Linear interpolation
                factor = (time - t1) / (t2 - t1)
                return v1 + (v2 - v1) * factor
        
        return self.points[0].value