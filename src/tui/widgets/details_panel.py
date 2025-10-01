"""
Details Panel Widget.

Displays detailed information about the currently selected track.
"""

from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import VerticalScroll
from rich.text import Text
from rich.table import Table

from ..models.session_state import SessionState, TrackInfo
from ..utils import theme


class DetailsPanel(VerticalScroll):
    """
    Panel displaying detailed track information.

    Shows all metadata about the selected track including musical
    parameters, file paths, and generation details.
    """

    DEFAULT_CSS = f"""
    DetailsPanel {{
        border: solid {theme.COLOR_BORDER};
        background: {theme.COLOR_BG_PANEL};
        height: {theme.HEIGHT_DETAILS_PANEL};
        padding: {theme.PADDING_PANEL};
    }}

    DetailsPanel:focus {{
        border: solid {theme.COLOR_BORDER_FOCUS};
    }}

    .details-content {{
        background: transparent;
        color: {theme.COLOR_TEXT_PRIMARY};
    }}

    .details-empty {{
        color: {theme.COLOR_TEXT_DIM};
        text-style: italic;
    }}
    """

    def __init__(self, session_state: SessionState, **kwargs):
        """
        Initialize details panel.

        Args:
            session_state: Session state containing track info
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.session_state = session_state
        self.border_title = theme.TITLE_DETAILS
        self._content_widget = None

    def compose(self) -> ComposeResult:
        """Compose the details panel layout."""
        self._content_widget = Static(
            Text(theme.MSG_NO_SELECTION, style=theme.COLOR_TEXT_DIM),
            classes="details-empty"
        )
        yield self._content_widget

    def update_details(self, track: TrackInfo) -> None:
        """
        Update details display for a track.

        Args:
            track: TrackInfo to display
        """
        # Create rich table for details
        table = Table(
            show_header=False,
            show_edge=False,
            padding=(0, 1),
            box=None
        )
        table.add_column("Key", style=f"bold {theme.COLOR_TEXT_SECONDARY}", width=15)
        table.add_column("Value", style=theme.COLOR_TEXT_PRIMARY)

        # Add track information rows
        table.add_row("Track ID", track.track_id)
        table.add_row("Name", track.name)

        # Status with colored indicator
        status_symbol = theme.get_status_symbol(track.status.value)
        status_color = theme.get_status_color(track.status.value)
        status_display = Text()
        status_display.append(f"{status_symbol} ", style=status_color)
        status_display.append(track.status.value.title(), style=status_color)
        table.add_row("Status", status_display)

        # Musical parameters
        table.add_row("", "")  # Separator
        table.add_row("BPM", f"{track.bpm:.1f}" if track.bpm else "—")
        table.add_row("Key", track.key if track.key else "—")
        table.add_row("Length", f"{track.length_bars:.1f} bars" if track.length_bars else "—")
        table.add_row("Duration", theme.format_duration(track.duration_seconds) if track.duration_seconds else "—")
        table.add_row("Notes", str(track.note_count) if track.note_count else "—")

        # Generation details
        table.add_row("", "")  # Separator
        table.add_row("Method", track.generation_method if track.generation_method else "—")
        table.add_row("Created", track.created_at.strftime(theme.TIMESTAMP_FORMAT))

        # File paths
        table.add_row("", "")  # Separator
        if track.audio_path:
            audio_path_short = track.audio_path.name
            table.add_row("Audio File", audio_path_short)
        else:
            table.add_row("Audio File", "—")

        # Original prompt (truncated)
        table.add_row("", "")  # Separator
        prompt_display = theme.truncate_text(track.prompt, 60)
        table.add_row("Prompt", prompt_display)

        # Error message if any
        if track.error_message:
            table.add_row("", "")  # Separator
            error_text = Text(track.error_message, style=theme.COLOR_TEXT_ERROR)
            table.add_row("Error", error_text)

        # Update widget
        if self._content_widget:
            self._content_widget.update(table)
            self._content_widget.remove_class("details-empty")
            self._content_widget.add_class("details-content")

    def clear_details(self) -> None:
        """Clear details and show empty state."""
        if self._content_widget:
            self._content_widget.update(
                Text(theme.MSG_NO_SELECTION, style=theme.COLOR_TEXT_DIM)
            )
            self._content_widget.remove_class("details-content")
            self._content_widget.add_class("details-empty")

    def show_track_details(self) -> None:
        """Update details for currently selected track."""
        selected_track = self.session_state.selected_track
        if selected_track:
            self.update_details(selected_track)
        else:
            self.clear_details()
