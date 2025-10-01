"""
Input Field Widget.

Text input for user prompts with validation and submission handling.
"""

from textual.app import ComposeResult
from textual.widgets import Input
from textual.message import Message

from ..utils import theme


class InputField(Input):
    """
    Enhanced input widget for user prompts.

    Features:
    - Prompt placeholder text
    - Maximum length validation
    - Enter to submit
    - Industrial styling
    """

    class Submitted(Message):
        """Message emitted when input is submitted."""

        def __init__(self, value: str) -> None:
            """
            Initialize submitted message.

            Args:
                value: The submitted text value
            """
            super().__init__()
            self.value = value

    DEFAULT_CSS = f"""
    InputField {{
        border: solid {theme.COLOR_BORDER};
        background: {theme.COLOR_BG_PANEL};
        color: {theme.COLOR_TEXT_PRIMARY};
        height: {theme.HEIGHT_INPUT_FIELD};
        padding: 1 2;
    }}

    InputField:focus {{
        border: solid {theme.COLOR_BORDER_FOCUS};
    }}

    InputField > .input--placeholder {{
        color: {theme.COLOR_TEXT_DIM};
    }}
    """

    def __init__(self, **kwargs):
        """Initialize input field."""
        super().__init__(
            placeholder=theme.PROMPT_PLACEHOLDER,
            max_length=theme.MAX_PROMPT_LENGTH,
            **kwargs
        )
        self.border_title = theme.PROMPT_ENTER_COMMAND

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Handle input submission (Enter key).

        Args:
            event: Input submitted event
        """
        value = event.value.strip()

        if value:
            # Emit our custom message
            self.post_message(self.Submitted(value))

            # Clear the input
            self.value = ""

    def clear_input(self) -> None:
        """Clear the input field."""
        self.value = ""

    def set_placeholder(self, text: str) -> None:
        """
        Set placeholder text.

        Args:
            text: Placeholder text
        """
        self.placeholder = text
