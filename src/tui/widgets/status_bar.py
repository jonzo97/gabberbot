"""
Status Bar Widget.

Footer displaying keyboard shortcuts and status information.
"""

from textual.app import ComposeResult
from textual.widgets import Static
from rich.text import Text

from ..models.session_state import SessionState
from ..utils import theme


class StatusBar(Static):
    """
    Status bar widget showing keyboard shortcuts and app status.

    Displays available keyboard shortcuts and current playback status.
    """

    DEFAULT_CSS = f"""
    StatusBar {{
        background: {theme.COLOR_BG_HIGHLIGHT};
        color: {theme.COLOR_TEXT_SECONDARY};
        height: {theme.HEIGHT_STATUS_BAR};
        padding: 0 2;
        dock: bottom;
    }}

    StatusBar .shortcut {{
        color: {theme.COLOR_ACCENT_PRIMARY};
    }}

    StatusBar .separator {{
        color: {theme.COLOR_TEXT_DIM};
    }}
    """

    def __init__(self, session_state: SessionState, **kwargs):
        """
        Initialize status bar.

        Args:
            session_state: Session state for status info
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.session_state = session_state

    def compose(self) -> ComposeResult:
        """Compose initial status bar content."""
        yield from []

    def on_mount(self) -> None:
        """Update status bar on mount."""
        self.update_status()

    def update_status(self) -> None:
        """Update the status bar content."""
        text = Text()

        # Build shortcuts display
        shortcuts = []
        for key, description in theme.SHORTCUTS.items():
            shortcuts.append(f"[{key}] {description}")

        shortcuts_str = " " + theme.SYMBOL_SEPARATOR + " ".join(shortcuts)

        # Add shortcuts
        self._format_shortcuts(text, shortcuts_str)

        # Add playback status if playing
        if self.session_state.playback.is_playing:
            current_track = self.session_state.get_track(
                self.session_state.playback.current_track_id
            )
            if current_track:
                text.append("  ", style=theme.COLOR_TEXT_DIM)
                text.append(f"{theme.SYMBOL_PLAYING} ", style=theme.COLOR_STATUS_PLAYING)
                text.append(
                    f"Playing: {current_track.name}",
                    style=theme.COLOR_TEXT_PRIMARY
                )

        # Add track count
        track_count = len(self.session_state.tracks)
        ready_count = len(self.session_state.ready_tracks)
        text.append("  ", style=theme.COLOR_TEXT_DIM)
        text.append(
            f"Tracks: {ready_count}/{track_count}",
            style=theme.COLOR_TEXT_SECONDARY
        )

        self.update(text)

    def _format_shortcuts(self, text: Text, shortcuts_str: str) -> None:
        """
        Format keyboard shortcuts with highlighting.

        Args:
            text: Text object to append to
            shortcuts_str: Raw shortcuts string
        """
        # Parse and format shortcuts
        import re

        # Pattern to find [key] description pairs
        pattern = r'\[([^\]]+)\]\s+([^─]+)'
        matches = re.findall(pattern, shortcuts_str)

        for i, (key, desc) in enumerate(matches):
            if i > 0:
                text.append(" " + theme.SYMBOL_SEPARATOR + " ", style=theme.COLOR_TEXT_DIM)

            # Key in brackets with accent color
            text.append("[", style=theme.COLOR_TEXT_DIM)
            text.append(key, style=f"bold {theme.COLOR_ACCENT_PRIMARY}")
            text.append("] ", style=theme.COLOR_TEXT_DIM)

            # Description in secondary color
            text.append(desc.strip(), style=theme.COLOR_TEXT_SECONDARY)
