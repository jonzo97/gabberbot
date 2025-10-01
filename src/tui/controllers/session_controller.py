"""
Session Controller for TUI.

Orchestrates the generation workflow by coordinating GenerationService and
AudioService. Translates service callbacks into session state updates.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from ...models.config import Settings
from ...models.core import MIDIClip
from ...services.generation_service import GenerationService, GenerationError
from ...services.audio_service import AudioService, RenderProgress, AudioConfig
from ..models.session_state import (
    SessionState,
    TrackInfo,
    TrackStatus,
    GenerationStage,
    MessageRole
)


class SessionController:
    """
    Controller for managing generation sessions.

    Wraps GenerationService and AudioService to coordinate the full
    text → MIDI → audio pipeline while updating session state.
    """

    def __init__(
        self,
        settings: Settings,
        session_state: SessionState,
        on_state_change: Optional[Callable[[SessionState], None]] = None
    ):
        """
        Initialize session controller.

        Args:
            settings: Application settings
            session_state: Session state object to update
            on_state_change: Optional callback when state changes
        """
        self.settings = settings
        self.session_state = session_state
        self.on_state_change = on_state_change
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.generation_service = GenerationService(settings)

        # Create audio service with progress callback
        audio_config = AudioConfig(
            sample_rate=settings.audio.sample_rate,
            bit_depth=settings.audio.bit_depth,
            buffer_size=settings.audio.buffer_size
        )
        self.audio_service = AudioService(
            settings=settings,
            audio_config=audio_config,
            progress_callback=self._on_audio_progress
        )

        # Output directory setup
        self.output_dir = Path(settings.audio.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_track(self, prompt: str) -> Optional[TrackInfo]:
        """
        Generate a complete track from a text prompt.

        This method orchestrates the full pipeline:
        1. Add user message to conversation
        2. Generate MIDI from prompt
        3. Render MIDI to audio
        4. Update track info and state

        Args:
            prompt: User prompt describing desired music

        Returns:
            TrackInfo object if successful, None if failed
        """
        # Create track ID and info
        track_id = self._generate_track_id()

        self.logger.info(f"Starting track generation for prompt: '{prompt}'")

        # Add user message to conversation
        self.session_state.add_message(MessageRole.USER, prompt)
        self._notify_state_change()

        # Create initial track info
        track_info = TrackInfo(
            track_id=track_id,
            name=f"Track {len(self.session_state.tracks) + 1}",
            prompt=prompt,
            status=TrackStatus.GENERATING
        )

        self.session_state.add_track(track_info)
        self._notify_state_change()

        try:
            # Stage 1: Generate MIDI
            self.session_state.update_progress(
                stage=GenerationStage.PARSING,
                progress=0.1,
                message="Parsing prompt...",
                track_id=track_id
            )
            self._notify_state_change()

            self.session_state.update_progress(
                stage=GenerationStage.AI_GENERATION,
                progress=0.3,
                message="Generating MIDI with AI...",
                track_id=track_id
            )
            self._notify_state_change()

            # Generate MIDI clip
            midi_clip = await self.generation_service.text_to_midi(
                prompt=prompt,
                fallback_to_algorithmic=True
            )

            if not midi_clip:
                raise GenerationError("MIDI generation returned None")

            self.logger.info(f"MIDI generated: {midi_clip.note_count} notes")

            # Update track info with MIDI metadata
            track_info.bpm = midi_clip.bpm
            track_info.key = midi_clip.key
            track_info.length_bars = midi_clip.length_bars
            track_info.note_count = midi_clip.note_count
            track_info.generation_method = "AI" if "ai_generated" in midi_clip.id else "Algorithmic"

            # Stage 2: Render to audio
            track_info.status = TrackStatus.RENDERING
            self.session_state.update_progress(
                stage=GenerationStage.SYNTHESIS,
                progress=0.5,
                message="Rendering audio...",
                track_id=track_id
            )
            self._notify_state_change()

            # Generate output filename
            audio_filename = f"{track_id}.wav"
            audio_path = self.output_dir / audio_filename

            # Render MIDI to WAV
            output_path = self.audio_service.render_to_wav(
                clip=midi_clip,
                output_path=audio_path
            )

            track_info.audio_path = output_path
            track_info.duration_seconds = midi_clip.length_seconds

            # Stage 3: Complete
            track_info.status = TrackStatus.READY
            self.session_state.update_progress(
                stage=GenerationStage.COMPLETE,
                progress=1.0,
                message="Track ready!",
                track_id=track_id
            )
            self._notify_state_change()

            # Add assistant response
            response_msg = (
                f"Generated '{track_info.name}' - "
                f"{track_info.bpm:.0f} BPM, {track_info.key or 'N/A'}, "
                f"{track_info.note_count} notes"
            )
            self.session_state.add_message(MessageRole.ASSISTANT, response_msg)

            # Clear progress after short delay
            await asyncio.sleep(1.0)
            self.session_state.clear_progress()
            self._notify_state_change()

            self.logger.info(f"Track generation complete: {track_id}")
            return track_info

        except Exception as e:
            self.logger.error(f"Track generation failed: {e}", exc_info=True)

            # Update track status to error
            track_info.status = TrackStatus.ERROR
            track_info.error_message = str(e)

            # Add error message to conversation
            error_msg = f"Generation failed: {str(e)}"
            self.session_state.add_message(MessageRole.ASSISTANT, error_msg)

            # Clear progress
            self.session_state.clear_progress()
            self._notify_state_change()

            return None

    def _on_audio_progress(self, progress: RenderProgress) -> None:
        """
        Callback for audio rendering progress.

        Translates AudioService progress to session state updates.

        Args:
            progress: RenderProgress from AudioService
        """
        # Map audio stages to generation stages
        stage_mapping = {
            "initialization": GenerationStage.SYNTHESIS,
            "synthesis": GenerationStage.SYNTHESIS,
            "processing": GenerationStage.PROCESSING,
            "export": GenerationStage.EXPORT,
            "complete": GenerationStage.COMPLETE
        }

        stage = stage_mapping.get(progress.stage, GenerationStage.SYNTHESIS)

        # Update progress in session state
        self.session_state.update_progress(
            stage=stage,
            progress=0.5 + (progress.progress * 0.5),  # Map to 50-100% range
            message=progress.message,
            track_id=None  # Audio service doesn't know track ID
        )

        self._notify_state_change()

    def get_track(self, track_id: str) -> Optional[TrackInfo]:
        """
        Get track by ID.

        Args:
            track_id: Track identifier

        Returns:
            TrackInfo or None if not found
        """
        return self.session_state.get_track(track_id)

    def get_all_tracks(self) -> list[TrackInfo]:
        """
        Get all tracks in the session.

        Returns:
            List of TrackInfo objects
        """
        return self.session_state.tracks

    def get_ready_tracks(self) -> list[TrackInfo]:
        """
        Get all tracks ready for playback.

        Returns:
            List of ready TrackInfo objects
        """
        return self.session_state.ready_tracks

    def clear_session(self) -> None:
        """Clear the session (remove all tracks and conversation)."""
        self.logger.info("Clearing session")

        # Clear tracks and conversation
        self.session_state.tracks.clear()
        self.session_state.conversation.clear()
        self.session_state.selected_track_index = None
        self.session_state.clear_progress()

        # Add system message
        self.session_state.add_message(
            MessageRole.SYSTEM,
            "Session cleared. Ready for new prompts."
        )

        self._notify_state_change()

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation service statistics.

        Returns:
            Dictionary of generation statistics
        """
        return self.generation_service.get_stats()

    def get_audio_stats(self) -> Dict[str, Any]:
        """
        Get audio service statistics.

        Returns:
            Dictionary of audio rendering statistics
        """
        return self.audio_service.get_stats()

    def is_generating(self) -> bool:
        """
        Check if currently generating a track.

        Returns:
            True if generation in progress
        """
        return any(track.is_generating for track in self.session_state.tracks)

    def _generate_track_id(self) -> str:
        """
        Generate a unique track ID.

        Returns:
            Unique track identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"track_{timestamp}_{unique_id}"

    def _notify_state_change(self) -> None:
        """Notify listeners that state has changed."""
        if self.on_state_change:
            try:
                self.on_state_change(self.session_state)
            except Exception as e:
                self.logger.error(f"Error in state change callback: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.logger.info("SessionController cleanup")
