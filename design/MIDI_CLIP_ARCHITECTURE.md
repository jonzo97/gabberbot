# MIDI Clip Architecture Design

## Overview

The MIDI Clip Architecture replaces our previous scattered tool approach with a unified system based on industry-standard MIDI representation. This provides maximum reusability, DAW compatibility, and future-proofing while maintaining the power of TidalCycles and SuperCollider.

## Core Principles

### 1. Standard MIDI Foundation
- All musical content represented as MIDI clips internally
- Full compatibility with DAWs and hardware
- Industry-standard timing, velocity, and pitch representation

### 2. Multiple Export Formats
- **TidalCycles patterns** for rhythmic complexity and live coding
- **Standard MIDI files** for DAW integration and hardware
- **OSC messages** for direct SuperCollider control
- **Pattern strings** for simple pattern representation

### 3. Reusable Pattern Generators
- All generators output standard MIDIClip or TriggerClip objects
- Can be combined, transposed, quantized, and manipulated
- Built-in converters handle backend-specific formatting

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Conversation Engine                   │
├─────────────────────────────────────────────────────────────┤
│  Unified Clip Tools:                                       │
│  • create_midi_clip(type="acid_bassline")                  │
│  • create_trigger_clip(pattern="x ~ x ~ x ~ x ~")          │
│  • apply_generator(generator="riff", scale="Am")            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Pattern Generators                          │
├─────────────────────────────────────────────────────────────┤
│  • AcidBasslineGenerator    • ChordProgressionGenerator     │
│  • RiffGenerator           • ArpeggioGenerator             │
│  • TunedKickGenerator      • PadChordGenerator             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Core MIDI Clip Classes                         │
├─────────────────────────────────────────────────────────────┤
│  MIDIClip:                  TriggerClip:                    │
│  • Notes with pitch         • Triggers without pitch        │
│  • Velocity & timing        • Velocity & timing            │
│  • Chord progressions       • Drum patterns               │
│  • Melodic content          • Percussion                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
┌─────────▼─┐  ┌──────▼──┐  ┌─────▼──────┐
│TidalCycles│  │MIDI File│  │OSC Messages│
│ Patterns  │  │ Export  │  │(Direct SC) │
└───────────┘  └─────────┘  └────────────┘
      │                           │
┌─────▼──────┐               ┌────▼─────────┐
│ SuperDirt  │               │ SuperCollider│
│ (via SC)   │               │  (Direct)    │
└────────────┘               └──────────────┘
```

## Core Classes

### MIDIClip
**Purpose**: Melodic/harmonic content with pitch information
**Use Cases**: 
- Synth riffs and melodies
- Acid basslines with pitch bends
- Chord progressions and pads
- Arpeggios and melodic sequences
- Tuned kick sequences (pitched kicks)

**Key Features**:
```python
class MIDIClip:
    notes: List[MIDINote]           # Pitch, velocity, timing, duration
    length_bars: float              # Clip length in musical bars
    time_signature: TimeSignature   # 4/4, 7/8, etc.
    bpm: float                      # Tempo
    key_signature: str              # "C", "Am", "F#m", etc.
    
    # Core methods
    def to_tidal_pattern() -> str   # Convert to TidalCycles
    def to_midi_file() -> MidiFile  # Export standard MIDI
    def transpose(semitones: int)   # Transpose all notes
    def quantize(grid: float)       # Snap to grid
```

### TriggerClip
**Purpose**: Rhythmic content without pitch information  
**Use Cases**:
- Kick drum patterns
- Hi-hat rhythms
- Percussion sequences
- Sample triggers
- Rhythmic effects

**Key Features**:
```python
class TriggerClip:
    triggers: List[Trigger]         # Sample ID, velocity, timing
    length_bars: float              # Clip length in musical bars
    time_signature: TimeSignature   # Timing framework
    bpm: float                      # Tempo
    
    # Core methods
    def to_tidal_pattern(sample_map) -> str  # Convert to TidalCycles
    def to_pattern_string() -> str           # Simple "x ~ x ~" format
```

## Pattern Generators

All pattern generators follow the same interface and output standard clip objects:

### AcidBasslineGenerator
```python
generator = AcidBasslineGenerator(
    scale="A_minor",
    accent_pattern="x ~ ~ x ~ ~ x ~",
    glide_probability=0.7,
    octave_range=(2, 4)
)
clip = generator.generate(length_bars=4, bpm=180)
```

### RiffGenerator  
```python
generator = RiffGenerator(
    scale="E_minor_pentatonic", 
    rhythm_complexity=0.8,
    note_density=0.6
)
clip = generator.generate(length_bars=2, bpm=200)
```

### ChordProgressionGenerator
```python
generator = ChordProgressionGenerator(
    progression=["Am", "F", "C", "G"],
    voicing_style="close",
    rhythm_pattern="whole_notes"
)
clip = generator.generate(length_bars=4, bpm=140)
```

### TunedKickGenerator
```python
generator = TunedKickGenerator(
    root_note="C1",
    pattern="x ~ x ~ x ~ x ~",
    pitch_sequence=[0, 0, 3, 0]  # Semitone offsets
)
clip = generator.generate(length_bars=1, bpm=180)
```

## Backend Integration

### TidalCycles Integration
- MIDIClips convert to TidalCycles pattern strings
- Preserves TidalCycles' rhythmic complexity advantages
- Uses existing TidalCycles → SuperCollider pipeline
- Maintains real-time pattern modification capabilities

### Direct SuperCollider Integration  
- MIDIClips can generate OSC messages directly
- Bypasses TidalCycles when not needed
- Useful for precise synthesis control
- Lower latency for simple patterns

### MIDI File Export
- Standard MIDI file generation using mido library
- Full DAW compatibility (Ableton, Logic, Cubase, etc.)
- Hardware synthesizer compatibility
- Session export and backup capabilities

## AI Tool Integration

### Unified Clip Tools
Instead of scattered specific tools, we have unified tools that work with any generator:

```python
# OLD APPROACH (scattered tools)
create_riff_tool()
create_acid_tool()  
create_arpeggio_tool()
create_tuned_kick_tool()

# NEW APPROACH (unified clip tools)
create_midi_clip(type="acid_bassline", scale="A_minor", length=2)
create_midi_clip(type="chord_progression", chords=["Am", "F", "C", "G"])
create_trigger_clip(pattern="x ~ x ~ x ~ x ~", sample="kick")
apply_generator(generator="riff", parameters={...})
```

### Benefits for AI Conversation
- **Consistent Interface**: All tools work the same way
- **Composability**: Clips can be combined and modified
- **Export Options**: AI can offer multiple output formats
- **Future Extensions**: New generators plug into existing system

## File Organization

```
cli_shared/
├── models/
│   ├── midi_clips.py              # Core MIDIClip and TriggerClip classes
│   └── hardcore_models.py         # Existing models (updated)
├── generators/                    # Pattern generators
│   ├── __init__.py
│   ├── acid_bassline.py
│   ├── riff_generator.py
│   ├── chord_progression.py
│   ├── arpeggio_generator.py
│   └── tuned_kick.py
├── converters/                    # Format converters
│   ├── __init__.py
│   ├── midi_to_tidal.py
│   ├── midi_to_osc.py
│   └── pattern_to_midi.py
└── ai/
    └── clip_tools.py              # Updated AI tools using clips
```

## Migration Strategy

### Phase 1: Foundation (COMPLETED)
- ✅ Core MIDIClip and TriggerClip classes created
- ✅ Basic MIDI file export functionality
- ✅ TidalCycles pattern conversion (basic implementation)

### Phase 2: Pattern Generators (CURRENT)
- 🔄 Build reusable pattern generators
- 🔄 Test generator output quality
- 🔄 Integrate with existing synthesizer backends

### Phase 3: AI Integration (NEXT)
- ⏳ Update AI conversation engine to use clips
- ⏳ Replace scattered tools with unified clip tools
- ⏳ Test AI tool workflow with TUI

### Phase 4: Backend Integration (FINAL)
- ⏳ Enhance TidalCycles integration
- ⏳ Add direct SuperCollider OSC messaging
- ⏳ Performance optimization and testing

## Benefits of This Architecture

### 1. Maximum Reusability
- All generators output standard MIDI clips
- Clips work with any backend (Tidal, SC, MIDI hardware)
- Can combine, transpose, and modify clips easily

### 2. Industry Compatibility  
- Standard MIDI representation
- DAW export/import capabilities
- Hardware synthesizer compatibility

### 3. Future-Proof Design
- Easy to add new generators
- New backends can be added without changing generators
- Standard MIDI ensures long-term compatibility

### 4. Clean AI Tool Architecture
- Unified tool interface instead of scattered specific tools
- Composable functionality
- Consistent user experience

### 5. Best of All Worlds
- Keep TidalCycles' rhythmic complexity
- Maintain SuperCollider's synthesis power  
- Add standard MIDI compatibility
- Enable hardware integration

This architecture provides a solid foundation for building sophisticated music generation tools while maintaining compatibility with existing systems and industry standards.