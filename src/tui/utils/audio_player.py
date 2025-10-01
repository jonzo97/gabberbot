"""
Cross-Platform Audio Playback Utility.

Simple wrapper for playing WAV files across different platforms using
subprocess calls to system audio players.
"""

import subprocess
import logging
import platform
import shutil
from pathlib import Path
from typing import Optional, Callable
from enum import Enum


class PlayerState(Enum):
    """Audio player state."""
    IDLE = "idle"
    PLAYING = "playing"
    STOPPED = "stopped"
    ERROR = "error"


class AudioPlayer:
    """
    Cross-platform audio player using system commands.

    Uses different audio players depending on the platform:
    - Linux: paplay (PulseAudio) or aplay (ALSA)
    - macOS: afplay
    - Windows: powershell Start-Process
    """

    def __init__(self, volume: float = 0.8):
        """
        Initialize audio player.

        Args:
            volume: Playback volume (0.0 to 1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.volume = max(0.0, min(1.0, volume))
        self.state = PlayerState.IDLE
        self._process: Optional[subprocess.Popen] = None
        self._current_file: Optional[Path] = None
        self._platform = platform.system().lower()

        # Find available audio player on system
        self._player_cmd = self._detect_audio_player()

    def _detect_audio_player(self) -> Optional[str]:
        """
        Detect which audio player is available on the system.

        Returns:
            Command name for the audio player, or None if not found
        """
        if self._platform == "linux":
            # Try paplay first (PulseAudio), then aplay (ALSA)
            if shutil.which("paplay"):
                self.logger.info("Using paplay for audio playback")
                return "paplay"
            elif shutil.which("aplay"):
                self.logger.info("Using aplay for audio playback")
                return "aplay"
            else:
                self.logger.warning("No audio player found (tried paplay, aplay)")
                return None

        elif self._platform == "darwin":  # macOS
            if shutil.which("afplay"):
                self.logger.info("Using afplay for audio playback")
                return "afplay"
            else:
                self.logger.warning("afplay not found on macOS")
                return None

        elif self._platform == "windows":
            # Windows uses PowerShell
            self.logger.info("Using PowerShell for audio playback on Windows")
            return "powershell"

        else:
            self.logger.warning(f"Unsupported platform: {self._platform}")
            return None

    def is_available(self) -> bool:
        """
        Check if audio playback is available.

        Returns:
            True if an audio player is available
        """
        return self._player_cmd is not None

    def play(self, file_path: Path) -> bool:
        """
        Play an audio file.

        Args:
            file_path: Path to WAV file

        Returns:
            True if playback started successfully
        """
        if not self.is_available():
            self.logger.error("No audio player available")
            self.state = PlayerState.ERROR
            return False

        if not file_path.exists():
            self.logger.error(f"Audio file not found: {file_path}")
            self.state = PlayerState.ERROR
            return False

        # Stop any currently playing audio
        self.stop()

        try:
            # Build command based on platform
            cmd = self._build_play_command(file_path)

            self.logger.info(f"Playing audio: {file_path}")
            self.logger.debug(f"Command: {' '.join(cmd)}")

            # Start playback process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            self._current_file = file_path
            self.state = PlayerState.PLAYING
            return True

        except Exception as e:
            self.logger.error(f"Failed to start playback: {e}")
            self.state = PlayerState.ERROR
            return False

    def _build_play_command(self, file_path: Path) -> list:
        """
        Build the playback command for the current platform.

        Args:
            file_path: Path to audio file

        Returns:
            Command list for subprocess
        """
        file_str = str(file_path.absolute())

        if self._player_cmd == "paplay":
            # PulseAudio on Linux
            volume_percent = int(self.volume * 65536)  # paplay uses 0-65536
            return ["paplay", f"--volume={volume_percent}", file_str]

        elif self._player_cmd == "aplay":
            # ALSA on Linux (no volume control via command line)
            return ["aplay", "-q", file_str]

        elif self._player_cmd == "afplay":
            # macOS afplay
            volume_str = str(self.volume)
            return ["afplay", "-v", volume_str, file_str]

        elif self._player_cmd == "powershell":
            # Windows PowerShell
            ps_script = f'(New-Object Media.SoundPlayer "{file_str}").PlaySync()'
            return ["powershell", "-Command", ps_script]

        else:
            # Fallback - should not reach here
            return [self._player_cmd, file_str]

    def stop(self) -> None:
        """Stop current playback."""
        if self._process and self._process.poll() is None:
            try:
                self.logger.info("Stopping playback")
                self._process.terminate()
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.logger.warning("Process did not terminate, forcing kill")
                self._process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping playback: {e}")

        self._process = None
        self._current_file = None
        self.state = PlayerState.STOPPED

    def is_playing(self) -> bool:
        """
        Check if audio is currently playing.

        Returns:
            True if audio is playing
        """
        if self._process is None:
            return False

        # Check if process is still running
        if self._process.poll() is not None:
            # Process finished
            self._process = None
            self._current_file = None
            self.state = PlayerState.IDLE
            return False

        return True

    def set_volume(self, volume: float) -> None:
        """
        Set playback volume.

        Note: This only affects future playback, not current playback.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))
        self.logger.debug(f"Volume set to {self.volume}")

    def get_current_file(self) -> Optional[Path]:
        """
        Get the currently playing file.

        Returns:
            Path to current file, or None if not playing
        """
        if self.is_playing():
            return self._current_file
        return None

    def cleanup(self) -> None:
        """Clean up resources and stop playback."""
        self.stop()
        self.logger.info("AudioPlayer cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# ============================================================================
# ASYNC AUDIO PLAYER (for non-blocking playback)
# ============================================================================


class AsyncAudioPlayer:
    """
    Asynchronous wrapper for AudioPlayer.

    Allows non-blocking playback with callbacks.
    """

    def __init__(self, volume: float = 0.8):
        """
        Initialize async audio player.

        Args:
            volume: Playback volume (0.0 to 1.0)
        """
        self.player = AudioPlayer(volume=volume)
        self._on_complete_callback: Optional[Callable] = None

    def play(
        self,
        file_path: Path,
        on_complete: Optional[Callable] = None
    ) -> bool:
        """
        Play audio file asynchronously.

        Args:
            file_path: Path to WAV file
            on_complete: Optional callback when playback completes

        Returns:
            True if playback started successfully
        """
        self._on_complete_callback = on_complete
        return self.player.play(file_path)

    def stop(self) -> None:
        """Stop playback."""
        self.player.stop()
        self._on_complete_callback = None

    def is_playing(self) -> bool:
        """
        Check if playing and trigger completion callback if finished.

        Returns:
            True if still playing
        """
        was_playing = self.player.state == PlayerState.PLAYING
        is_playing = self.player.is_playing()

        # If playback just finished, trigger callback
        if was_playing and not is_playing and self._on_complete_callback:
            try:
                self._on_complete_callback()
            except Exception as e:
                self.player.logger.error(f"Error in completion callback: {e}")
            finally:
                self._on_complete_callback = None

        return is_playing

    def is_available(self) -> bool:
        """Check if audio playback is available."""
        return self.player.is_available()

    def set_volume(self, volume: float) -> None:
        """Set playback volume."""
        self.player.set_volume(volume)

    def get_current_file(self) -> Optional[Path]:
        """Get currently playing file."""
        return self.player.get_current_file()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.player.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
