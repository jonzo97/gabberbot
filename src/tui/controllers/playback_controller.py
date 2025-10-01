"""
Playback Controller for TUI.

Manages audio playback state and coordinates with AudioPlayer utility.
Provides high-level playback control (play, stop, volume) and state management.
"""

import logging
from pathlib import Path
from typing import Optional, Callable

from ..models.session_state import SessionState, TrackInfo, TrackStatus
from ..utils.audio_player import AsyncAudioPlayer


class PlaybackController:
    """
    Controller for managing audio playback.

    Coordinates AudioPlayer with session state to provide high-level
    playback functionality.
    """

    def __init__(
        self,
        session_state: SessionState,
        on_state_change: Optional[Callable[[SessionState], None]] = None,
        initial_volume: float = 0.8
    ):
        """
        Initialize playback controller.

        Args:
            session_state: Session state to update
            on_state_change: Optional callback when state changes
            initial_volume: Initial volume level (0.0 to 1.0)
        """
        self.session_state = session_state
        self.on_state_change = on_state_change
        self.logger = logging.getLogger(__name__)

        # Initialize audio player
        self.player = AsyncAudioPlayer(volume=initial_volume)
        self.session_state.playback.volume = initial_volume

        # Check if player is available
        if not self.player.is_available():
            self.logger.warning("Audio player not available on this system")

    def play_track(self, track_id: str) -> bool:
        """
        Play a track by ID.

        Args:
            track_id: ID of track to play

        Returns:
            True if playback started successfully
        """
        # Get track info
        track = self.session_state.get_track(track_id)
        if not track:
            self.logger.error(f"Track not found: {track_id}")
            return False

        if not track.is_ready:
            self.logger.error(f"Track not ready for playback: {track_id}")
            return False

        if not track.audio_path or not track.audio_path.exists():
            self.logger.error(f"Audio file not found for track: {track_id}")
            return False

        # Stop current playback
        self.stop()

        # Start playback
        success = self.player.play(
            file_path=track.audio_path,
            on_complete=self._on_playback_complete
        )

        if success:
            self.logger.info(f"Started playback: {track_id}")

            # Update session state
            self.session_state.playback.is_playing = True
            self.session_state.playback.current_track_id = track_id
            self.session_state.playback.position_seconds = 0.0

            # Update track status
            track.status = TrackStatus.PLAYING

            self._notify_state_change()
            return True
        else:
            self.logger.error(f"Failed to start playback: {track_id}")
            return False

    def play_selected(self) -> bool:
        """
        Play the currently selected track.

        Returns:
            True if playback started successfully
        """
        track = self.session_state.selected_track
        if not track:
            self.logger.warning("No track selected")
            return False

        return self.play_track(track.track_id)

    def stop(self) -> None:
        """Stop current playback."""
        if not self.session_state.playback.is_playing:
            return

        self.logger.info("Stopping playback")

        # Stop player
        self.player.stop()

        # Reset track status
        if self.session_state.playback.current_track_id:
            track = self.session_state.get_track(
                self.session_state.playback.current_track_id
            )
            if track and track.status == TrackStatus.PLAYING:
                track.status = TrackStatus.READY

        # Update state
        self.session_state.playback.is_playing = False
        self.session_state.playback.current_track_id = None
        self.session_state.playback.position_seconds = 0.0

        self._notify_state_change()

    def toggle_playback(self) -> bool:
        """
        Toggle playback (play/stop).

        If playing, stops. If stopped, plays selected track.

        Returns:
            True if now playing, False if now stopped
        """
        if self.session_state.playback.is_playing:
            self.stop()
            return False
        else:
            return self.play_selected()

    def set_volume(self, volume: float) -> None:
        """
        Set playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))
        self.player.set_volume(volume)
        self.session_state.playback.volume = volume

        self.logger.debug(f"Volume set to {volume}")
        self._notify_state_change()

    def increase_volume(self, delta: float = 0.1) -> None:
        """
        Increase volume by delta.

        Args:
            delta: Amount to increase (default 0.1)
        """
        new_volume = self.session_state.playback.volume + delta
        self.set_volume(new_volume)

    def decrease_volume(self, delta: float = 0.1) -> None:
        """
        Decrease volume by delta.

        Args:
            delta: Amount to decrease (default 0.1)
        """
        new_volume = self.session_state.playback.volume - delta
        self.set_volume(new_volume)

    def is_playing(self) -> bool:
        """
        Check if currently playing and update state.

        Returns:
            True if playing
        """
        # Check actual player state
        actually_playing = self.player.is_playing()

        # Update state if it changed
        if self.session_state.playback.is_playing != actually_playing:
            self.session_state.playback.is_playing = actually_playing

            if not actually_playing:
                # Playback stopped
                if self.session_state.playback.current_track_id:
                    track = self.session_state.get_track(
                        self.session_state.playback.current_track_id
                    )
                    if track and track.status == TrackStatus.PLAYING:
                        track.status = TrackStatus.READY

                self.session_state.playback.current_track_id = None

            self._notify_state_change()

        return actually_playing

    def get_current_track(self) -> Optional[TrackInfo]:
        """
        Get currently playing track.

        Returns:
            TrackInfo or None if not playing
        """
        if not self.session_state.playback.current_track_id:
            return None

        return self.session_state.get_track(
            self.session_state.playback.current_track_id
        )

    def is_available(self) -> bool:
        """
        Check if audio playback is available.

        Returns:
            True if player is available
        """
        return self.player.is_available()

    def _on_playback_complete(self) -> None:
        """Callback when playback completes naturally."""
        self.logger.info("Playback completed")

        # Reset track status
        if self.session_state.playback.current_track_id:
            track = self.session_state.get_track(
                self.session_state.playback.current_track_id
            )
            if track and track.status == TrackStatus.PLAYING:
                track.status = TrackStatus.READY

        # Update state
        self.session_state.playback.is_playing = False
        self.session_state.playback.current_track_id = None
        self.session_state.playback.position_seconds = 0.0

        self._notify_state_change()

    def _notify_state_change(self) -> None:
        """Notify listeners that state has changed."""
        if self.on_state_change:
            try:
                self.on_state_change(self.session_state)
            except Exception as e:
                self.logger.error(f"Error in state change callback: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("PlaybackController cleanup")
        self.stop()
        self.player.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
