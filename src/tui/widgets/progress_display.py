"""
Progress Display Widget.

Real-time progress indicator for generation and rendering operations.
"""

from textual.app import ComposeResult
from textual.widgets import Static, ProgressBar
from textual.containers import Container
from rich.text import Text

from ..models.session_state import SessionState, ProgressInfo
from ..utils import theme


class ProgressDisplay(Container):
    """
    Widget displaying current operation progress.

    Shows progress bar, stage name, and status message.
    """

    DEFAULT_CSS = f"""
    ProgressDisplay {{
        border: solid {theme.COLOR_BORDER};
        background: {theme.COLOR_BG_PANEL};
        height: {theme.HEIGHT_PROGRESS_DISPLAY};
        padding: 1;
    }}

    ProgressDisplay.hidden {{
        display: none;
    }}

    .progress-stage {{
        color: {theme.COLOR_ACCENT_PRIMARY};
        text-style: bold;
        height: 1;
    }}

    .progress-message {{
        color: {theme.COLOR_TEXT_SECONDARY};
        height: 1;
    }}

    .progress-bar-container {{
        height: 1;
        padding: 0;
    }}
    """

    def __init__(self, session_state: SessionState, **kwargs):
        """
        Initialize progress display.

        Args:
            session_state: Session state containing progress info
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.session_state = session_state
        self.border_title = theme.TITLE_PROGRESS

    def compose(self) -> ComposeResult:
        """Compose the progress display layout."""
        yield Static("", classes="progress-stage", id="progress-stage")
        yield Static("", classes="progress-message", id="progress-message")
        yield ProgressBar(
            total=100,
            show_eta=False,
            classes="progress-bar-container",
            id="progress-bar"
        )

    def update_progress(self, progress_info: ProgressInfo) -> None:
        """
        Update progress display.

        Args:
            progress_info: Progress information
        """
        # Show the widget if hidden
        self.remove_class("hidden")

        # Update stage
        stage_widget = self.query_one("#progress-stage", Static)
        stage_text = Text()
        stage_text.append(f"{theme.SYMBOL_GENERATING} ", style=theme.COLOR_ACCENT_PRIMARY)
        stage_text.append(
            progress_info.stage.value.replace("_", " ").title(),
            style=f"bold {theme.COLOR_ACCENT_PRIMARY}"
        )
        stage_widget.update(stage_text)

        # Update message
        message_widget = self.query_one("#progress-message", Static)
        message_text = Text(
            f"{theme.SYMBOL_ARROW_RIGHT} {progress_info.message}",
            style=theme.COLOR_TEXT_SECONDARY
        )
        message_widget.update(message_text)

        # Update progress bar
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_bar.update(progress=progress_info.percentage)

    def hide_progress(self) -> None:
        """Hide the progress display."""
        self.add_class("hidden")

    def show_progress(self) -> None:
        """Show the progress display."""
        self.remove_class("hidden")

    def is_visible(self) -> bool:
        """
        Check if progress is visible.

        Returns:
            True if visible
        """
        return not self.has_class("hidden")
