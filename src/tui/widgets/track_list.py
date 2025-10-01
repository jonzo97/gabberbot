"""
Track List Widget.

Table displaying all generated tracks with their metadata and status.
"""

from textual.app import ComposeResult
from textual.widgets import DataTable
from textual.message import Message
from rich.text import Text

from ..models.session_state import SessionState, TrackInfo, TrackStatus
from ..utils import theme


class TrackList(DataTable):
    """
    Table widget displaying tracks.

    Shows track status, name, BPM, key, duration, and note count.
    Supports selection and navigation.
    """

    class TrackSelected(Message):
        """Message emitted when a track is selected."""

        def __init__(self, track_id: str, track_index: int) -> None:
            """
            Initialize track selected message.

            Args:
                track_id: ID of selected track
                track_index: Index of selected track
            """
            super().__init__()
            self.track_id = track_id
            self.track_index = track_index

    DEFAULT_CSS = f"""
    TrackList {{
        border: solid {theme.COLOR_BORDER};
        background: {theme.COLOR_BG_PANEL};
        height: {theme.HEIGHT_TRACK_LIST};
    }}

    TrackList:focus {{
        border: solid {theme.COLOR_BORDER_FOCUS};
    }}

    TrackList > .datatable--header {{
        background: {theme.COLOR_BG_HIGHLIGHT};
        color: {theme.COLOR_TEXT_PRIMARY};
        text-style: bold;
    }}

    TrackList > .datatable--cursor {{
        background: {theme.COLOR_BG_SELECTED};
    }}
    """

    def __init__(self, session_state: SessionState, **kwargs):
        """
        Initialize track list.

        Args:
            session_state: Session state containing tracks
            **kwargs: Additional widget arguments
        """
        super().__init__(
            cursor_type="row",
            zebra_stripes=True,
            **kwargs
        )
        self.session_state = session_state
        self.border_title = theme.TITLE_TRACK_LIST

        # Setup columns
        self._setup_columns()

    def _setup_columns(self) -> None:
        """Setup table columns."""
        self.add_column(theme.TABLE_HEADER_STATUS, width=theme.TABLE_COL_WIDTH_STATUS)
        self.add_column(theme.TABLE_HEADER_NAME, width=theme.TABLE_COL_WIDTH_NAME)
        self.add_column(theme.TABLE_HEADER_BPM, width=theme.TABLE_COL_WIDTH_BPM)
        self.add_column(theme.TABLE_HEADER_KEY, width=theme.TABLE_COL_WIDTH_KEY)
        self.add_column(theme.TABLE_HEADER_DURATION, width=theme.TABLE_COL_WIDTH_DURATION)
        self.add_column(theme.TABLE_HEADER_NOTES, width=theme.TABLE_COL_WIDTH_NOTES)

    def update_tracks(self) -> None:
        """Update the track list with latest data."""
        # Clear existing rows
        self.clear()

        # Add all tracks
        for track in self.session_state.tracks:
            self._add_track_row(track)

        # Restore selection if it exists
        if self.session_state.selected_track_index is not None:
            if 0 <= self.session_state.selected_track_index < self.row_count:
                self.move_cursor(row=self.session_state.selected_track_index)

    def _add_track_row(self, track: TrackInfo) -> None:
        """
        Add a track row to the table.

        Args:
            track: TrackInfo to add
        """
        # Status column with symbol and color
        status_symbol = theme.get_status_symbol(track.status.value)
        status_color = theme.get_status_color(track.status.value)
        status_text = Text(status_symbol, style=status_color)

        # Name column
        name_text = Text(
            theme.truncate_text(track.name, theme.TABLE_COL_WIDTH_NAME - 2),
            style=theme.COLOR_TEXT_PRIMARY
        )

        # BPM column
        bpm_text = Text(
            f"{track.bpm:.0f}" if track.bpm else "—",
            style=theme.COLOR_TEXT_SECONDARY
        )

        # Key column
        key_text = Text(
            track.key if track.key else "—",
            style=theme.COLOR_TEXT_SECONDARY
        )

        # Duration column
        if track.duration_seconds:
            duration_str = theme.format_duration(track.duration_seconds)
        else:
            duration_str = "—"
        duration_text = Text(duration_str, style=theme.COLOR_TEXT_SECONDARY)

        # Notes column
        notes_text = Text(
            str(track.note_count) if track.note_count else "—",
            style=theme.COLOR_TEXT_SECONDARY
        )

        # Add row
        self.add_row(
            status_text,
            name_text,
            bpm_text,
            key_text,
            duration_text,
            notes_text
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """
        Handle row selection.

        Args:
            event: Row selected event
        """
        row_index = event.cursor_row

        if 0 <= row_index < len(self.session_state.tracks):
            track = self.session_state.tracks[row_index]

            # Update session state
            self.session_state.selected_track_index = row_index

            # Emit custom message
            self.post_message(self.TrackSelected(track.track_id, row_index))

    def clear_tracks(self) -> None:
        """Clear all tracks from the table."""
        self.clear()

    def add_track(self, track: TrackInfo) -> None:
        """
        Add a single track to the list.

        Args:
            track: TrackInfo to add
        """
        self._add_track_row(track)

    def update_track(self, track_id: str) -> None:
        """
        Update a specific track's display.

        Args:
            track_id: ID of track to update
        """
        # Find track index
        track_index = None
        for i, track in enumerate(self.session_state.tracks):
            if track.track_id == track_id:
                track_index = i
                break

        if track_index is not None and track_index < self.row_count:
            # Get track info
            track = self.session_state.tracks[track_index]

            # Update row cells
            self._update_row_cells(track_index, track)

    def _update_row_cells(self, row_index: int, track: TrackInfo) -> None:
        """
        Update cells in a specific row.

        Args:
            row_index: Row index
            track: TrackInfo with updated data
        """
        # Update status
        status_symbol = theme.get_status_symbol(track.status.value)
        status_color = theme.get_status_color(track.status.value)
        self.update_cell_at(
            (row_index, 0),
            Text(status_symbol, style=status_color)
        )

        # Update other cells if needed
        # (Name, BPM, Key typically don't change after creation)
        # Could update duration/notes if they change

    def show_empty_state(self) -> None:
        """Show empty state message."""
        self.clear()
        # DataTable doesn't support empty state well, so we just leave it empty
        # The main screen will handle showing a message
