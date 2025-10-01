"""
Main TUI Application.

Textual application that coordinates all controllers, widgets, and screens
to provide the complete hardcore music generation interface.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from textual.app import App
from textual.message import Message

from ..models.config import Settings
from ..utils.env import load_settings
from .models.session_state import SessionState, MessageRole
from .controllers.session_controller import SessionController
from .controllers.playback_controller import PlaybackController
from .screens.main_screen import MainScreen
from .widgets.input_field import InputField
from .utils import theme


class HardcoreMusicTUI(App):
    """
    Main TUI application for hardcore music generation.

    Coordinates:
    - Session state management
    - Generation and audio services via controllers
    - UI updates and user interactions
    - Playback control
    """

    # Application metadata
    TITLE = "Hardcore Music Assistant"
    SUB_TITLE = "Conversational AI for Industrial Electronic Music Production"

    # CSS for global app styling
    CSS = f"""
    Screen {{
        background: {theme.COLOR_BG_DARK};
    }}
    """

    def __init__(self, settings: Optional[Settings] = None, **kwargs):
        """
        Initialize the TUI application.

        Args:
            settings: Application settings (will load from env if not provided)
            **kwargs: Additional app arguments
        """
        super().__init__(**kwargs)

        # Load settings
        self.settings = settings or load_settings()
        self.logger = logging.getLogger(__name__)

        # Initialize session state
        session_id = str(uuid.uuid4())[:8]
        self.session_state = SessionState(session_id=session_id)

        # Controllers (initialized in on_mount)
        self.session_controller: Optional[SessionController] = None
        self.playback_controller: Optional[PlaybackController] = None

        # Screen reference
        self.main_screen: Optional[MainScreen] = None

    async def on_mount(self) -> None:
        """Initialize application on mount."""
        self.logger.info("Initializing Hardcore Music TUI")

        # Initialize controllers
        self.session_controller = SessionController(
            settings=self.settings,
            session_state=self.session_state,
            on_state_change=self._on_state_change
        )

        self.playback_controller = PlaybackController(
            session_state=self.session_state,
            on_state_change=self._on_state_change
        )

        # Check if audio player is available
        if not self.playback_controller.is_available():
            self.logger.warning("Audio player not available - playback disabled")
            self.session_state.add_message(
                MessageRole.SYSTEM,
                "Warning: Audio playback not available on this system."
            )

        # Push main screen
        self.main_screen = MainScreen(self.session_state)
        await self.push_screen(self.main_screen)

        # Add welcome message
        self.session_state.add_message(
            MessageRole.SYSTEM,
            f"Welcome to {self.TITLE}. Enter a prompt to generate hardcore music."
        )

        # Update UI
        self.main_screen.update_ui()

        self.logger.info("TUI initialization complete")

    # ========================================================================
    # Event Handlers
    # ========================================================================

    async def on_input_field_submitted(self, event: InputField.Submitted) -> None:
        """
        Handle user prompt submission.

        Args:
            event: Input submitted event
        """
        prompt = event.value.strip()

        if not prompt:
            return

        self.logger.info(f"User submitted prompt: {prompt}")

        # Check if it's a command
        if prompt.startswith("/"):
            await self._handle_command(prompt)
            return

        # Generate track from prompt
        await self._generate_track(prompt)

    async def on_main_screen_toggle_playback_message(self, event: Message) -> None:
        """Handle toggle playback message from main screen."""
        if self.playback_controller:
            is_playing = self.playback_controller.toggle_playback()

            if is_playing:
                self.session_state.add_message(
                    MessageRole.SYSTEM,
                    theme.SUCCESS_PLAYBACK_STARTED
                )
            else:
                self.session_state.add_message(
                    MessageRole.SYSTEM,
                    "Playback stopped."
                )

            self._update_ui()

    async def on_main_screen_stop_playback_message(self, event: Message) -> None:
        """Handle stop playback message from main screen."""
        if self.playback_controller:
            self.playback_controller.stop()
            self.session_state.add_message(
                MessageRole.SYSTEM,
                "Playback stopped."
            )
            self._update_ui()

    async def on_main_screen_clear_session_message(self, event: Message) -> None:
        """Handle clear session message from main screen."""
        if self.session_controller:
            self.session_controller.clear_session()
            self._update_ui()

    # ========================================================================
    # Generation & Playback
    # ========================================================================

    async def _generate_track(self, prompt: str) -> None:
        """
        Generate a track from user prompt.

        Args:
            prompt: User prompt
        """
        if not self.session_controller:
            self.logger.error("Session controller not initialized")
            return

        # Check if already generating
        if self.session_controller.is_generating():
            self.session_state.add_message(
                MessageRole.SYSTEM,
                "Generation already in progress. Please wait..."
            )
            self._update_ui()
            return

        # Start generation
        self.logger.info(f"Starting track generation: {prompt}")

        try:
            track_info = await self.session_controller.generate_track(prompt)

            if track_info:
                self.logger.info(f"Track generated successfully: {track_info.track_id}")
            else:
                self.logger.error("Track generation returned None")

        except Exception as e:
            self.logger.error(f"Track generation failed: {e}", exc_info=True)
            self.session_state.add_message(
                MessageRole.SYSTEM,
                f"Error: {str(e)}"
            )

        finally:
            self._update_ui()

    async def _handle_command(self, command: str) -> None:
        """
        Handle slash commands.

        Args:
            command: Command string starting with /
        """
        cmd = command.lower().strip()

        if cmd == "/help" or cmd == "/?":
            self._show_help()
        elif cmd == "/clear":
            if self.session_controller:
                self.session_controller.clear_session()
        elif cmd == "/stats":
            self._show_stats()
        elif cmd == "/quit" or cmd == "/exit":
            self.exit()
        else:
            self.session_state.add_message(
                MessageRole.SYSTEM,
                f"Unknown command: {command}. Type /help for available commands."
            )

        self._update_ui()

    def _show_help(self) -> None:
        """Show help message."""
        help_text = """
Available commands:
  /help, /?    - Show this help message
  /clear       - Clear session (remove all tracks)
  /stats       - Show generation statistics
  /quit, /exit - Exit application

Keyboard shortcuts:
  q          - Quit
  space      - Play/Pause selected track
  j/k        - Navigate tracks (next/previous)
  s          - Stop playback
  i          - Focus input field
  c          - Clear session

To generate music, simply type a description and press Enter.
Examples:
  - "aggressive 180 bpm gabber kick with distortion"
  - "dark acid bassline in Am, 160 bpm"
  - "industrial techno kick with rumble"
        """
        self.session_state.add_message(MessageRole.SYSTEM, help_text.strip())

    def _show_stats(self) -> None:
        """Show generation statistics."""
        if not self.session_controller:
            return

        gen_stats = self.session_controller.get_generation_stats()
        audio_stats = self.session_controller.get_audio_stats()

        stats_text = f"""
Generation Statistics:
  Total Requests: {gen_stats['total_requests']}
  Successful (AI): {gen_stats['successful_ai']}
  Successful (Fallback): {gen_stats['successful_fallback']}
  Failures: {gen_stats['failures']}
  Avg Response Time: {gen_stats['avg_response_time']:.2f}s

Audio Rendering Statistics:
  Total Renders: {audio_stats['total_renders']}
  Successful: {audio_stats['successful_renders']}
  Avg Render Time: {audio_stats['avg_render_time']:.2f}s
  Total Duration: {audio_stats['total_duration_rendered']:.2f}s
        """
        self.session_state.add_message(MessageRole.SYSTEM, stats_text.strip())

    # ========================================================================
    # State Management
    # ========================================================================

    def _on_state_change(self, state: SessionState) -> None:
        """
        Callback when session state changes.

        Args:
            state: Updated session state
        """
        # Update UI on main thread
        self.call_from_thread(self._update_ui)

    def _update_ui(self) -> None:
        """Update all UI components."""
        if self.main_screen:
            self.main_screen.update_ui()

    # ========================================================================
    # Cleanup
    # ========================================================================

    async def on_unmount(self) -> None:
        """Cleanup on application exit."""
        self.logger.info("Shutting down TUI")

        # Stop playback
        if self.playback_controller:
            self.playback_controller.cleanup()

        # Cleanup controllers
        if self.session_controller:
            self.session_controller.__exit__(None, None, None)

        self.logger.info("TUI shutdown complete")


# ============================================================================
# Entry Point
# ============================================================================


def run_tui(settings: Optional[Settings] = None) -> None:
    """
    Run the TUI application.

    Args:
        settings: Optional settings object (will load from env if not provided)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("tui.log"),
            logging.StreamHandler()
        ]
    )

    # Create and run app
    app = HardcoreMusicTUI(settings=settings)
    app.run()


if __name__ == "__main__":
    run_tui()
