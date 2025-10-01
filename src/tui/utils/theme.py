"""
Industrial Dark Theme for Hardcore Music TUI.

Constants and styling definitions for the industrial/warehouse aesthetic.
All color values, symbols, and styling parameters in one place.
"""

from typing import Final

# ============================================================================
# COLOR PALETTE - Industrial/Warehouse Theme
# ============================================================================

# Primary colors
COLOR_BG_DARK: Final[str] = "#0a0a0a"  # Almost black background
COLOR_BG_PANEL: Final[str] = "#1a1a1a"  # Panel background
COLOR_BG_HIGHLIGHT: Final[str] = "#2a2a2a"  # Highlighted background
COLOR_BG_SELECTED: Final[str] = "#3a3a3a"  # Selected item background

# Accent colors
COLOR_ACCENT_PRIMARY: Final[str] = "#ff4444"  # Aggressive red
COLOR_ACCENT_SECONDARY: Final[str] = "#ff8800"  # Warning orange
COLOR_ACCENT_SUCCESS: Final[str] = "#44ff44"  # Success green
COLOR_ACCENT_INFO: Final[str] = "#4488ff"  # Info blue

# Text colors
COLOR_TEXT_PRIMARY: Final[str] = "#ffffff"  # White text
COLOR_TEXT_SECONDARY: Final[str] = "#aaaaaa"  # Gray text
COLOR_TEXT_DIM: Final[str] = "#666666"  # Dimmed text
COLOR_TEXT_ERROR: Final[str] = "#ff4444"  # Error red
COLOR_TEXT_SUCCESS: Final[str] = "#44ff44"  # Success green
COLOR_TEXT_WARNING: Final[str] = "#ff8800"  # Warning orange

# Border colors
COLOR_BORDER: Final[str] = "#444444"  # Standard border
COLOR_BORDER_FOCUS: Final[str] = "#ff4444"  # Focused border
COLOR_BORDER_DIM: Final[str] = "#333333"  # Dimmed border

# Progress colors
COLOR_PROGRESS_BAR: Final[str] = "#ff4444"  # Progress bar fill
COLOR_PROGRESS_BG: Final[str] = "#2a2a2a"  # Progress bar background

# Status colors
COLOR_STATUS_READY: Final[str] = "#44ff44"  # Ready status
COLOR_STATUS_PENDING: Final[str] = "#aaaaaa"  # Pending status
COLOR_STATUS_GENERATING: Final[str] = "#4488ff"  # Generating status
COLOR_STATUS_ERROR: Final[str] = "#ff4444"  # Error status
COLOR_STATUS_PLAYING: Final[str] = "#ff8800"  # Playing status

# ============================================================================
# SYMBOLS & ICONS - ASCII/Unicode
# ============================================================================

# Status symbols
SYMBOL_READY: Final[str] = "●"  # Ready to play
SYMBOL_PENDING: Final[str] = "○"  # Pending
SYMBOL_GENERATING: Final[str] = "⟳"  # Generating
SYMBOL_ERROR: Final[str] = "✗"  # Error
SYMBOL_PLAYING: Final[str] = "▶"  # Playing
SYMBOL_PAUSED: Final[str] = "⏸"  # Paused

# UI symbols
SYMBOL_PROMPT: Final[str] = "›"  # Input prompt
SYMBOL_USER: Final[str] = "YOU"  # User message prefix
SYMBOL_ASSISTANT: Final[str] = "AI"  # Assistant message prefix
SYMBOL_SEPARATOR: Final[str] = "─"  # Horizontal separator
SYMBOL_BULLET: Final[str] = "•"  # Bullet point
SYMBOL_ARROW_RIGHT: Final[str] = "→"  # Right arrow
SYMBOL_ARROW_UP: Final[str] = "↑"  # Up arrow
SYMBOL_ARROW_DOWN: Final[str] = "↓"  # Down arrow

# Progress symbols
SYMBOL_PROGRESS_FILLED: Final[str] = "█"  # Filled progress block
SYMBOL_PROGRESS_EMPTY: Final[str] = "░"  # Empty progress block

# ============================================================================
# DIMENSIONS & SPACING
# ============================================================================

# Panel heights (in rows)
HEIGHT_CONVERSATION_PANEL: Final[int] = 15
HEIGHT_TRACK_LIST: Final[int] = 12
HEIGHT_DETAILS_PANEL: Final[int] = 8
HEIGHT_PROGRESS_DISPLAY: Final[int] = 4
HEIGHT_INPUT_FIELD: Final[int] = 3
HEIGHT_STATUS_BAR: Final[int] = 1

# Widths (percentage or fixed)
WIDTH_CONVERSATION_PERCENT: Final[int] = 60
WIDTH_DETAILS_PERCENT: Final[int] = 40

# Padding & margins
PADDING_PANEL: Final[int] = 1
PADDING_CONTENT: Final[int] = 1
MARGIN_BETWEEN_PANELS: Final[int] = 1

# ============================================================================
# TEXT STYLES
# ============================================================================

# Font weights and styles
STYLE_BOLD: Final[str] = "bold"
STYLE_ITALIC: Final[str] = "italic"
STYLE_DIM: Final[str] = "dim"
STYLE_REVERSE: Final[str] = "reverse"

# ============================================================================
# UI TEXT & LABELS
# ============================================================================

# Panel titles
TITLE_CONVERSATION: Final[str] = "CONVERSATION"
TITLE_TRACK_LIST: Final[str] = "TRACKS"
TITLE_DETAILS: Final[str] = "DETAILS"
TITLE_PROGRESS: Final[str] = "PROGRESS"

# Status messages
MSG_READY: Final[str] = "Ready"
MSG_GENERATING: Final[str] = "Generating..."
MSG_RENDERING: Final[str] = "Rendering..."
MSG_PLAYING: Final[str] = "Playing"
MSG_ERROR: Final[str] = "Error"
MSG_NO_TRACKS: Final[str] = "No tracks yet. Enter a prompt to generate music."
MSG_NO_SELECTION: Final[str] = "Select a track to view details."

# Input prompts
PROMPT_ENTER_COMMAND: Final[str] = "Enter prompt or command (Ctrl+C to quit)"
PROMPT_PLACEHOLDER: Final[str] = "describe your hardcore track..."

# ============================================================================
# KEYBOARD SHORTCUTS
# ============================================================================

KEY_QUIT: Final[str] = "q"
KEY_PLAY: Final[str] = "space"
KEY_STOP: Final[str] = "s"
KEY_NEXT_TRACK: Final[str] = "j"
KEY_PREV_TRACK: Final[str] = "k"
KEY_SELECT_INPUT: Final[str] = "i"
KEY_CLEAR: Final[str] = "c"
KEY_HELP: Final[str] = "?"

# Keyboard shortcut display strings
SHORTCUTS: Final[dict] = {
    "q": "Quit",
    "space": "Play/Pause",
    "j/k": "Navigate",
    "i": "Focus Input",
    "s": "Stop",
    "c": "Clear",
    "?": "Help"
}

# ============================================================================
# ANIMATION & TIMING
# ============================================================================

# Animation speeds (in seconds)
ANIMATION_FADE_DURATION: Final[float] = 0.3
ANIMATION_PROGRESS_UPDATE: Final[float] = 0.1
ANIMATION_SPINNER_SPEED: Final[float] = 0.2

# Spinner frames for loading animations
SPINNER_FRAMES: Final[tuple] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# ============================================================================
# PROGRESS BAR SETTINGS
# ============================================================================

PROGRESS_BAR_WIDTH: Final[int] = 40
PROGRESS_BAR_STYLE: Final[str] = "bar"

# ============================================================================
# TABLE SETTINGS (for TrackList)
# ============================================================================

# Column headers
TABLE_HEADER_STATUS: Final[str] = "ST"
TABLE_HEADER_NAME: Final[str] = "NAME"
TABLE_HEADER_BPM: Final[str] = "BPM"
TABLE_HEADER_KEY: Final[str] = "KEY"
TABLE_HEADER_DURATION: Final[str] = "TIME"
TABLE_HEADER_NOTES: Final[str] = "NOTES"

# Column widths (characters)
TABLE_COL_WIDTH_STATUS: Final[int] = 3
TABLE_COL_WIDTH_NAME: Final[int] = 30
TABLE_COL_WIDTH_BPM: Final[int] = 6
TABLE_COL_WIDTH_KEY: Final[int] = 5
TABLE_COL_WIDTH_DURATION: Final[int] = 8
TABLE_COL_WIDTH_NOTES: Final[int] = 6

# ============================================================================
# MESSAGE FORMATTING
# ============================================================================

# Maximum message length before truncation
MAX_MESSAGE_LENGTH: Final[int] = 1000
MAX_PROMPT_LENGTH: Final[int] = 500

# Timestamp format
TIMESTAMP_FORMAT: Final[str] = "%H:%M:%S"
TIMESTAMP_FORMAT_FULL: Final[str] = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_GENERATION_FAILED: Final[str] = "Generation failed. Check logs for details."
ERROR_PLAYBACK_FAILED: Final[str] = "Playback failed. Audio file may be corrupted."
ERROR_NO_AUDIO_DEVICE: Final[str] = "No audio output device found."
ERROR_INVALID_PROMPT: Final[str] = "Invalid prompt. Please try again."

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

SUCCESS_TRACK_GENERATED: Final[str] = "Track generated successfully!"
SUCCESS_TRACK_RENDERED: Final[str] = "Audio rendered successfully!"
SUCCESS_PLAYBACK_STARTED: Final[str] = "Playback started."

# ============================================================================
# STYLE HELPERS
# ============================================================================

def get_status_color(status: str) -> str:
    """
    Get color for a given status.

    Args:
        status: Status string (ready, pending, generating, error, playing)

    Returns:
        Color hex code
    """
    status_lower = status.lower()
    if status_lower == "ready":
        return COLOR_STATUS_READY
    elif status_lower == "pending":
        return COLOR_STATUS_PENDING
    elif status_lower in ["generating", "rendering"]:
        return COLOR_STATUS_GENERATING
    elif status_lower == "error":
        return COLOR_STATUS_ERROR
    elif status_lower == "playing":
        return COLOR_STATUS_PLAYING
    else:
        return COLOR_TEXT_SECONDARY


def get_status_symbol(status: str) -> str:
    """
    Get symbol for a given status.

    Args:
        status: Status string (ready, pending, generating, error, playing)

    Returns:
        Status symbol
    """
    status_lower = status.lower()
    if status_lower == "ready":
        return SYMBOL_READY
    elif status_lower == "pending":
        return SYMBOL_PENDING
    elif status_lower in ["generating", "rendering"]:
        return SYMBOL_GENERATING
    elif status_lower == "error":
        return SYMBOL_ERROR
    elif status_lower == "playing":
        return SYMBOL_PLAYING
    else:
        return SYMBOL_PENDING


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to MM:SS format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
