"""
Main Screen Layout.

Primary screen composition that assembles all widgets into the TUI layout.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header

from ..models.session_state import SessionState
from ..widgets.conversation_panel import ConversationPanel
from ..widgets.input_field import InputField
from ..widgets.track_list import TrackList
from ..widgets.progress_display import ProgressDisplay
from ..widgets.details_panel import DetailsPanel
from ..widgets.status_bar import StatusBar


class MainScreen(Screen):
    """
    Main application screen.

    Layout structure:
    ┌─────────────────────────────────────────┐
    │ Header                                   │
    ├────────────────────┬────────────────────┤
    │                    │                    │
    │ Conversation       │ Track List         │
    │ Panel              │                    │
    │                    │                    │
    ├────────────────────┼────────────────────┤
    │ Progress Display   │ Details Panel      │
    ├────────────────────┴────────────────────┤
    │ Input Field                             │
    ├─────────────────────────────────────────┤
    │ Status Bar                              │
    └─────────────────────────────────────────┘
    """

    BINDINGS = [
        ("q", "quit_app", "Quit"),
        ("space", "toggle_playback", "Play/Pause"),
        ("j", "next_track", "Next Track"),
        ("k", "prev_track", "Previous Track"),
        ("s", "stop_playback", "Stop"),
        ("c", "clear_session", "Clear"),
        ("i", "focus_input", "Focus Input"),
    ]

    DEFAULT_CSS = """
    MainScreen {
        background: $background;
    }

    .main-container {
        width: 100%;
        height: 100%;
    }

    .top-row {
        height: auto;
    }

    .middle-row {
        height: auto;
    }

    .left-column {
        width: 60%;
    }

    .right-column {
        width: 40%;
    }
    """

    def __init__(self, session_state: SessionState, **kwargs):
        """
        Initialize main screen.

        Args:
            session_state: Session state object
            **kwargs: Additional screen arguments
        """
        super().__init__(**kwargs)
        self.session_state = session_state

        # Widget references (set during compose)
        self.conversation_panel = None
        self.input_field = None
        self.track_list = None
        self.progress_display = None
        self.details_panel = None
        self.status_bar = None

    def compose(self) -> ComposeResult:
        """Compose the main screen layout."""
        # Header
        yield Header(show_clock=True)

        # Main container
        with Container(classes="main-container"):
            # Top row: Conversation + Track List
            with Horizontal(classes="top-row"):
                with Vertical(classes="left-column"):
                    self.conversation_panel = ConversationPanel(
                        self.session_state,
                        id="conversation-panel"
                    )
                    yield self.conversation_panel

                with Vertical(classes="right-column"):
                    self.track_list = TrackList(
                        self.session_state,
                        id="track-list"
                    )
                    yield self.track_list

            # Middle row: Progress + Details
            with Horizontal(classes="middle-row"):
                with Vertical(classes="left-column"):
                    self.progress_display = ProgressDisplay(
                        self.session_state,
                        id="progress-display"
                    )
                    yield self.progress_display

                with Vertical(classes="right-column"):
                    self.details_panel = DetailsPanel(
                        self.session_state,
                        id="details-panel"
                    )
                    yield self.details_panel

            # Input field
            self.input_field = InputField(id="input-field")
            yield self.input_field

        # Status bar
        self.status_bar = StatusBar(self.session_state, id="status-bar")
        yield self.status_bar

    def on_mount(self) -> None:
        """Handle screen mount."""
        # Focus input field by default
        self.input_field.focus()

        # Hide progress initially
        if self.session_state.progress is None:
            self.progress_display.hide_progress()

    # ========================================================================
    # Event Handlers
    # ========================================================================

    async def on_input_field_submitted(self, event: InputField.Submitted) -> None:
        """
        Handle input submission.

        Args:
            event: Input submitted event
        """
        # This will be handled by the main app
        # The event will bubble up
        pass

    def on_track_list_track_selected(self, event: TrackList.TrackSelected) -> None:
        """
        Handle track selection.

        Args:
            event: Track selected event
        """
        # Update details panel
        if self.details_panel:
            self.details_panel.show_track_details()

        # Update status bar
        if self.status_bar:
            self.status_bar.update_status()

    # ========================================================================
    # Action Handlers (Key Bindings)
    # ========================================================================

    def action_quit_app(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_toggle_playback(self) -> None:
        """Toggle playback (play/pause)."""
        # Emit custom message for app to handle
        from textual.message import Message

        class TogglePlaybackMessage(Message):
            pass

        self.post_message(TogglePlaybackMessage())

    def action_next_track(self) -> None:
        """Select next track."""
        if self.track_list and self.track_list.row_count > 0:
            current_row = self.track_list.cursor_row
            next_row = (current_row + 1) % self.track_list.row_count
            self.track_list.move_cursor(row=next_row)

    def action_prev_track(self) -> None:
        """Select previous track."""
        if self.track_list and self.track_list.row_count > 0:
            current_row = self.track_list.cursor_row
            prev_row = (current_row - 1) % self.track_list.row_count
            self.track_list.move_cursor(row=prev_row)

    def action_stop_playback(self) -> None:
        """Stop playback."""
        from textual.message import Message

        class StopPlaybackMessage(Message):
            pass

        self.post_message(StopPlaybackMessage())

    def action_clear_session(self) -> None:
        """Clear session."""
        from textual.message import Message

        class ClearSessionMessage(Message):
            pass

        self.post_message(ClearSessionMessage())

    def action_focus_input(self) -> None:
        """Focus the input field."""
        if self.input_field:
            self.input_field.focus()

    # ========================================================================
    # Update Methods
    # ========================================================================

    def update_ui(self) -> None:
        """Update all UI components with latest state."""
        # Update conversation
        if self.conversation_panel:
            self.conversation_panel.update_conversation()

        # Update track list
        if self.track_list:
            self.track_list.update_tracks()

        # Update progress
        if self.progress_display:
            if self.session_state.progress:
                self.progress_display.update_progress(self.session_state.progress)
                self.progress_display.show_progress()
            else:
                self.progress_display.hide_progress()

        # Update details
        if self.details_panel:
            self.details_panel.show_track_details()

        # Update status bar
        if self.status_bar:
            self.status_bar.update_status()

    def add_conversation_message(self, message) -> None:
        """
        Add a message to conversation display.

        Args:
            message: ConversationMessage to add
        """
        if self.conversation_panel:
            self.conversation_panel.add_message(message)

    def add_track(self, track) -> None:
        """
        Add a track to the track list.

        Args:
            track: TrackInfo to add
        """
        if self.track_list:
            self.track_list.add_track(track)

        # Update status bar
        if self.status_bar:
            self.status_bar.update_status()

    def update_track(self, track_id: str) -> None:
        """
        Update a specific track display.

        Args:
            track_id: ID of track to update
        """
        if self.track_list:
            self.track_list.update_track(track_id)

        # Update details if this is the selected track
        selected = self.session_state.selected_track
        if selected and selected.track_id == track_id and self.details_panel:
            self.details_panel.show_track_details()

        # Update status bar
        if self.status_bar:
            self.status_bar.update_status()
