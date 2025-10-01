"""
Session State Models for TUI.

Pydantic models representing the complete state of a TUI session including
conversation history, track information, and playback state.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )


class TrackStatus(str, Enum):
    """Status of a track in the session."""
    PENDING = "pending"
    GENERATING = "generating"
    RENDERING = "rendering"
    READY = "ready"
    ERROR = "error"
    PLAYING = "playing"


class GenerationStage(str, Enum):
    """Stage of the generation process."""
    PARSING = "parsing"
    AI_GENERATION = "ai_generation"
    FALLBACK_GENERATION = "fallback_generation"
    SYNTHESIS = "synthesis"
    PROCESSING = "processing"
    EXPORT = "export"
    COMPLETE = "complete"


class TrackInfo(BaseModel):
    """
    Information about a generated track.

    Represents a single track in the session with metadata, status,
    and file paths.
    """

    track_id: str = Field(..., description="Unique track identifier")
    name: str = Field(..., description="Track name")
    prompt: str = Field(..., description="Original user prompt")
    status: TrackStatus = Field(default=TrackStatus.PENDING, description="Current status")

    # File paths
    midi_path: Optional[Path] = Field(default=None, description="Path to MIDI file")
    audio_path: Optional[Path] = Field(default=None, description="Path to WAV file")

    # Musical metadata
    bpm: Optional[float] = Field(default=None, description="Beats per minute")
    key: Optional[str] = Field(default=None, description="Musical key")
    length_bars: Optional[float] = Field(default=None, description="Length in bars")
    note_count: Optional[int] = Field(default=None, description="Number of MIDI notes")

    # Generation metadata
    generation_method: Optional[str] = Field(default=None, description="AI or algorithmic")
    duration_seconds: Optional[float] = Field(default=None, description="Audio duration")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }
    )

    @property
    def is_ready(self) -> bool:
        """Check if track is ready for playback."""
        return self.status == TrackStatus.READY and self.audio_path is not None

    @property
    def is_generating(self) -> bool:
        """Check if track is currently being generated."""
        return self.status in [TrackStatus.GENERATING, TrackStatus.RENDERING]

    @property
    def has_error(self) -> bool:
        """Check if track has an error."""
        return self.status == TrackStatus.ERROR


class ProgressInfo(BaseModel):
    """Progress information for current operation."""

    stage: GenerationStage = Field(..., description="Current generation stage")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress from 0.0 to 1.0")
    message: str = Field(..., description="Human-readable progress message")
    track_id: Optional[str] = Field(default=None, description="Track ID being processed")

    @property
    def percentage(self) -> int:
        """Get progress as percentage (0-100)."""
        return int(self.progress * 100)


class PlaybackState(BaseModel):
    """Current playback state."""

    is_playing: bool = Field(default=False, description="Whether audio is playing")
    current_track_id: Optional[str] = Field(default=None, description="ID of playing track")
    volume: float = Field(default=0.8, ge=0.0, le=1.0, description="Playback volume")
    position_seconds: float = Field(default=0.0, ge=0.0, description="Current playback position")


class SessionState(BaseModel):
    """
    Complete state of a TUI session.

    Contains all information needed to render the UI and handle user interactions.
    """

    session_id: str = Field(..., description="Unique session identifier")
    conversation: List[ConversationMessage] = Field(
        default_factory=list,
        description="Conversation history"
    )
    tracks: List[TrackInfo] = Field(
        default_factory=list,
        description="List of generated tracks"
    )
    progress: Optional[ProgressInfo] = Field(
        default=None,
        description="Current operation progress"
    )
    playback: PlaybackState = Field(
        default_factory=PlaybackState,
        description="Playback state"
    )
    selected_track_index: Optional[int] = Field(
        default=None,
        description="Index of currently selected track"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Session creation time"
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the conversation."""
        message = ConversationMessage(role=role, content=content)
        self.conversation.append(message)

    def add_track(self, track: TrackInfo) -> None:
        """Add a track to the session."""
        self.tracks.append(track)

    def get_track(self, track_id: str) -> Optional[TrackInfo]:
        """Get a track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def update_track_status(self, track_id: str, status: TrackStatus) -> None:
        """Update the status of a track."""
        track = self.get_track(track_id)
        if track:
            track.status = status

    def update_progress(
        self,
        stage: GenerationStage,
        progress: float,
        message: str,
        track_id: Optional[str] = None
    ) -> None:
        """Update current progress information."""
        self.progress = ProgressInfo(
            stage=stage,
            progress=progress,
            message=message,
            track_id=track_id
        )

    def clear_progress(self) -> None:
        """Clear progress information."""
        self.progress = None

    @property
    def selected_track(self) -> Optional[TrackInfo]:
        """Get the currently selected track."""
        if self.selected_track_index is not None and 0 <= self.selected_track_index < len(self.tracks):
            return self.tracks[self.selected_track_index]
        return None

    @property
    def ready_tracks(self) -> List[TrackInfo]:
        """Get all tracks ready for playback."""
        return [track for track in self.tracks if track.is_ready]

    @property
    def track_count(self) -> int:
        """Get total number of tracks."""
        return len(self.tracks)
