# Hardcore Music TUI - Terminal User Interface

## Overview

A professional Terminal User Interface (TUI) for the Hardcore Music Assistant, built with Textual framework. Provides an interactive, conversational interface for generating hardcore/industrial electronic music using AI.

## Architecture

### Components

```
src/tui/
├── app.py                      # Main Textual application entry point
├── controllers/                # Business logic controllers
│   ├── session_controller.py  # Orchestrates generation/audio services
│   └── playback_controller.py # Manages audio playback
├── models/                     # Data models
│   └── session_state.py       # Pydantic models for session state
├── widgets/                    # UI components
│   ├── conversation_panel.py  # Chat-style conversation display
│   ├── input_field.py         # Text input for prompts
│   ├── track_list.py          # Table of generated tracks
│   ├── progress_display.py    # Real-time progress indicators
│   ├── details_panel.py       # Track metadata display
│   └── status_bar.py          # Keyboard shortcuts footer
├── screens/                    # Screen compositions
│   └── main_screen.py         # Primary screen layout
└── utils/                      # Utilities
    ├── theme.py               # Industrial dark theme constants
    └── audio_player.py        # Cross-platform audio playback
```

## Features

### Phase 1 (MVP - IMPLEMENTED)

✅ **Interactive Generation Loop**
- Chat-style conversation interface
- Text input for natural language prompts
- Real-time progress feedback during generation/rendering

✅ **Track Management**
- Table view of all generated tracks
- Track status indicators (pending, generating, ready, error)
- Track selection and navigation (j/k keys)

✅ **Playback Control**
- Play/pause selected tracks (space)
- Stop playback (s)
- Cross-platform audio support (Linux, macOS, Windows)

✅ **Progress Display**
- Real-time progress bar
- Stage indicators (parsing, AI generation, synthesis, export)
- Status messages

✅ **Track Details**
- Comprehensive metadata view
- Musical parameters (BPM, key, length, note count)
- Generation method (AI vs algorithmic)
- File paths and timestamps

✅ **Keyboard Shortcuts**
- q: Quit
- space: Play/Pause
- j/k: Navigate tracks
- s: Stop playback
- i: Focus input
- c: Clear session

### Industrial Dark Theme

- Almost-black backgrounds (#0a0a0a)
- Aggressive red accents (#ff4444)
- High contrast text
- ASCII/Unicode status symbols
- Professional terminal aesthetic

## Integration with Existing Services

The TUI wraps existing services **without modification**:

### GenerationService Integration
- `SessionController` wraps `GenerationService`
- Handles `text_to_midi()` calls
- Translates service operations to state updates

### AudioService Integration
- `SessionController` wraps `AudioService`
- Progress callbacks mapped to UI updates
- Renders MIDI clips to WAV files

### Playback Integration
- `PlaybackController` manages `AudioPlayer`
- Cross-platform audio playback
- State synchronization with UI

## Data Flow

```
User Input → InputField
    ↓
MainScreen → SessionController
    ↓
GenerationService.text_to_midi()
    ↓
AudioService.render_to_wav()
    ↓
SessionState Update
    ↓
UI Refresh (all widgets)
```

## Session State Management

All UI state lives in `SessionState` (Pydantic model):
- **Conversation**: List of messages (user, assistant, system)
- **Tracks**: List of `TrackInfo` objects
- **Progress**: Current operation progress (stage, %, message)
- **Playback**: Current playback state (playing, track ID, volume)
- **Selection**: Currently selected track index

Controllers update state → State change callback → UI refresh

## Running the TUI

### Prerequisites

```bash
# Install dependencies
pip install textual pydantic soundfile numpy
```

### Launch

```bash
# From project root
python3 -m src.tui.app

# Or directly
python3 src/tui/app.py
```

### Environment Setup

Ensure you have a `.env` file with API keys:

```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Usage Examples

### Generate a Track

1. Type a prompt in the input field:
   ```
   aggressive 180 bpm gabber kick with heavy distortion
   ```

2. Press Enter

3. Watch progress indicators:
   - Parsing prompt...
   - Generating MIDI with AI...
   - Rendering audio...
   - Track ready!

4. Track appears in the track list with ● (ready) indicator

### Play a Track

1. Navigate to track using `j`/`k` keys
2. Press `space` to play
3. Press `space` again to stop, or `s` to stop explicitly

### View Track Details

- Select any track (j/k navigation)
- Details panel automatically updates with:
  - Track metadata
  - Musical parameters
  - File paths
  - Original prompt

### Slash Commands

- `/help` - Show help message
- `/clear` - Clear session
- `/stats` - Show generation statistics
- `/quit` - Exit application

## Design Principles

### Code Quality Standards (from CLAUDE.md)

✅ **Quality Over Speed**: Carefully structured, no rush
✅ **No Reinventing Wheels**: Uses existing services
✅ **Use Existing Infrastructure**: AbstractSynthesizer, existing models
✅ **Modular Everything**: All components importable/reusable
✅ **Zero Magic Numbers**: All constants in theme.py
✅ **Professional Architecture**: Clean separation of concerns

### Controller Pattern

Controllers sit between UI and services:
- **SessionController**: Orchestrates generation workflow
- **PlaybackController**: Manages playback state

No direct service calls from widgets - all via controllers.

### Widget Design

Each widget is:
- Self-contained with its own CSS
- Receives `SessionState` as dependency
- Updates via explicit `update_*()` methods
- Emits custom messages for inter-widget communication

### Error Handling

- Services return None on failure (not exceptions)
- Errors captured in `TrackInfo.error_message`
- Error status shown in UI with red indicators
- Error messages added to conversation

## Testing

### Integration Test

```bash
python3 test_tui_integration.py
```

Tests:
- Module imports
- Service integration
- Session state creation
- Controller initialization
- Audio player availability

### Manual Testing Checklist

- [ ] Launch TUI successfully
- [ ] Enter prompt and generate track
- [ ] Watch progress indicators
- [ ] View generated track in list
- [ ] Select track with j/k
- [ ] Play track with space
- [ ] Stop playback with s
- [ ] View track details
- [ ] Generate multiple tracks
- [ ] Navigate between tracks
- [ ] Clear session with c
- [ ] Use slash commands
- [ ] Quit with q

## Future Enhancements (Phase 2+)

### Planned Features
- [ ] Waveform visualization
- [ ] Volume control (+ / -)
- [ ] Export tracks (MIDI + WAV)
- [ ] Session save/load
- [ ] Track editing (transpose, quantize)
- [ ] Undo/redo
- [ ] Multiple selection
- [ ] Playlist mode
- [ ] Keyboard chord generation

### Technical Improvements
- [ ] Async progress updates (streaming)
- [ ] Cancellable generation
- [ ] Background task queue
- [ ] Caching of generated clips
- [ ] Hot reload of configuration
- [ ] Theme customization

## Known Limitations

1. **Single track generation**: Can't generate multiple tracks simultaneously
2. **No track editing**: Generated tracks are immutable
3. **No persistence**: Session lost on quit
4. **Simple playback**: No seeking, no visualizations
5. **Text-only**: No graphical waveforms

## File Organization

### Models
- `session_state.py`: 200 lines - Complete session state (tracks, conversation, progress, playback)

### Controllers
- `session_controller.py`: 250 lines - Generation workflow orchestration
- `playback_controller.py`: 200 lines - Playback state management

### Widgets
- `conversation_panel.py`: 150 lines - Chat display
- `track_list.py`: 200 lines - Track table
- `details_panel.py`: 150 lines - Metadata display
- `progress_display.py`: 100 lines - Progress indicators
- `input_field.py`: 80 lines - Text input
- `status_bar.py`: 100 lines - Footer with shortcuts

### Screens
- `main_screen.py`: 250 lines - Layout composition + event handling

### Utils
- `theme.py`: 400 lines - Complete theme constants
- `audio_player.py`: 300 lines - Cross-platform audio playback

### App
- `app.py`: 300 lines - Main application + entry point

**Total**: ~2,500 lines of clean, documented code

## Dependencies

### Required
- `textual>=0.47.0` - TUI framework
- `pydantic>=2.0.0` - Data validation
- `rich>=13.0.0` - Text formatting (via Textual)

### Existing (from main project)
- `numpy` - Audio processing
- `soundfile` - WAV file I/O

### System (for audio playback)
- Linux: `paplay` (PulseAudio) or `aplay` (ALSA)
- macOS: `afplay` (built-in)
- Windows: PowerShell (built-in)

## Troubleshooting

### No audio playback
- Check if audio player is available: `/stats` command shows player status
- Linux: Install PulseAudio (`sudo apt install pulseaudio`)
- Verify audio files exist in `output/` directory

### Import errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: Requires Python 3.11+

### Generation fails
- Check API keys in `.env` file
- Review logs in `tui.log`
- Fallback to algorithmic generation if AI fails

### UI rendering issues
- Ensure terminal supports Unicode
- Try different terminal emulator
- Reduce terminal size if widgets overlap

## Contributing

When modifying the TUI:

1. **Follow existing patterns**: Controllers wrap services, widgets update via state
2. **Use theme constants**: All colors/dimensions in `theme.py`
3. **Type everything**: Full type hints on all functions
4. **Document thoroughly**: Docstrings on all classes/methods
5. **Test integration**: Run `test_tui_integration.py`
6. **No magic numbers**: Extract to constants
7. **No direct service calls**: Go through controllers

## Credits

- **Framework**: [Textual](https://textual.textualize.io/) by Textualize
- **Architecture**: Following @architect specifications
- **Design**: @ux-expert industrial theme
- **Requirements**: @po user stories

Built with BMAD methodology - quality over speed, proper architecture, clean code.
