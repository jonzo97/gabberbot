"""
Conversation Panel Widget.

Chat-style display of conversation history between user and AI assistant.
"""

from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import VerticalScroll
from rich.text import Text
from datetime import datetime

from ..models.session_state import SessionState, ConversationMessage, MessageRole
from ..utils import theme


class ConversationPanel(VerticalScroll):
    """
    Scrollable panel displaying conversation history.

    Shows messages in chat-style format with timestamps and role indicators.
    """

    DEFAULT_CSS = f"""
    ConversationPanel {{
        border: solid {theme.COLOR_BORDER};
        background: {theme.COLOR_BG_PANEL};
        padding: {theme.PADDING_PANEL};
        height: {theme.HEIGHT_CONVERSATION_PANEL};
    }}

    ConversationPanel:focus {{
        border: solid {theme.COLOR_BORDER_FOCUS};
    }}

    .conversation-message {{
        background: transparent;
        padding: 0 1;
        margin-bottom: 1;
    }}

    .message-user {{
        color: {theme.COLOR_ACCENT_PRIMARY};
    }}

    .message-assistant {{
        color: {theme.COLOR_ACCENT_INFO};
    }}

    .message-system {{
        color: {theme.COLOR_TEXT_DIM};
    }}

    .message-timestamp {{
        color: {theme.COLOR_TEXT_DIM};
    }}

    .message-content {{
        color: {theme.COLOR_TEXT_PRIMARY};
    }}
    """

    def __init__(self, session_state: SessionState, **kwargs):
        """
        Initialize conversation panel.

        Args:
            session_state: Session state containing conversation
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.session_state = session_state
        self.border_title = theme.TITLE_CONVERSATION

    def compose(self) -> ComposeResult:
        """Compose the conversation panel layout."""
        # Initially render all messages
        yield from self._render_messages()

    def _render_messages(self) -> list[Static]:
        """
        Render all messages in conversation.

        Returns:
            List of Static widgets for each message
        """
        widgets = []

        if not self.session_state.conversation:
            # Show empty state
            empty_msg = Static(
                Text("No conversation yet. Enter a prompt below to start.",
                     style=f"italic {theme.COLOR_TEXT_DIM}"),
                classes="conversation-message"
            )
            widgets.append(empty_msg)
        else:
            for message in self.session_state.conversation:
                widget = self._create_message_widget(message)
                widgets.append(widget)

        return widgets

    def _create_message_widget(self, message: ConversationMessage) -> Static:
        """
        Create a widget for a single message.

        Args:
            message: ConversationMessage to render

        Returns:
            Static widget containing formatted message
        """
        # Format timestamp
        timestamp_str = message.timestamp.strftime(theme.TIMESTAMP_FORMAT)

        # Get role prefix and color
        if message.role == MessageRole.USER:
            role_prefix = theme.SYMBOL_USER
            role_class = "message-user"
            role_color = theme.COLOR_ACCENT_PRIMARY
        elif message.role == MessageRole.ASSISTANT:
            role_prefix = theme.SYMBOL_ASSISTANT
            role_class = "message-assistant"
            role_color = theme.COLOR_ACCENT_INFO
        else:  # SYSTEM
            role_prefix = "SYS"
            role_class = "message-system"
            role_color = theme.COLOR_TEXT_DIM

        # Build rich text
        text = Text()
        text.append(f"[{timestamp_str}] ", style=theme.COLOR_TEXT_DIM)
        text.append(f"{role_prefix}", style=f"bold {role_color}")
        text.append(f" {theme.SYMBOL_PROMPT} ", style=theme.COLOR_TEXT_DIM)
        text.append(message.content, style=theme.COLOR_TEXT_PRIMARY)

        # Create widget
        widget = Static(text, classes=f"conversation-message {role_class}")
        return widget

    def update_conversation(self) -> None:
        """Update the conversation display with latest messages."""
        # Remove all existing message widgets
        self.remove_children()

        # Re-render all messages
        for widget in self._render_messages():
            self.mount(widget)

        # Auto-scroll to bottom
        self.scroll_end(animate=False)

    def add_message(self, message: ConversationMessage) -> None:
        """
        Add a new message to the display.

        Args:
            message: ConversationMessage to add
        """
        widget = self._create_message_widget(message)
        self.mount(widget)

        # Auto-scroll to bottom
        self.scroll_end(animate=True)

    def clear_messages(self) -> None:
        """Clear all messages from display."""
        self.remove_children()

        # Show empty state
        empty_msg = Static(
            Text("Conversation cleared. Enter a prompt to continue.",
                 style=f"italic {theme.COLOR_TEXT_DIM}"),
            classes="conversation-message"
        )
        self.mount(empty_msg)
