# 🔥 Hardcore Music Production System

Professional-grade hardcore/gabber/industrial techno production system with **MIDI Clip architecture** and **multiple synthesis engines**.

## ✨ MIDI Clip Architecture (v3.0)

**MIDI-Based System**: Industry-standard MIDI clips with multiple export formats
- **Standard MIDI Foundation**: Full DAW compatibility and hardware integration
- **Multiple Backends**: TidalCycles patterns, SuperCollider OSC, MIDI files
- **Reusable Generators**: Acid basslines, riffs, chord progressions, tuned kicks
- **AI-Friendly Tools**: Unified clip-based interface instead of scattered tools

## 🚀 Quick Start

### MIDI Clip System
```python
from cli_shared.models.midi_clips import MIDIClip, TriggerClip, MIDINote
from cli_shared.generators import AcidBasslineGenerator, TunedKickGenerator

# Create acid bassline clip
bassline = AcidBasslineGenerator(scale="A_minor", accent_pattern="x ~ ~ x")
clip = bassline.generate(length_bars=4, bpm=180)

# Export to multiple formats
tidal_pattern = clip.to_tidal_pattern()     # → TidalCycles
midi_file = clip.to_midi_file()             # → Standard MIDI file  
osc_messages = clip.to_osc_messages()       # → SuperCollider OSC

# Create kick pattern with tuned pitches
kick_gen = TunedKickGenerator(root_note="C1", pattern="x ~ x ~ x ~ x ~")
kick_clip = kick_gen.generate(length_bars=1, bpm=200)
```

### Test the System
```bash
# Test MIDI clip system
python -c "from cli_shared.models.midi_clips import MIDIClip; print('MIDI clips ready!')"

# Test enhanced AI agent with clip tools
python test_enhanced_ai_tools.py

# Launch TUI interface with MIDI clip support
python cli_tui/hardcore_tui.py

# Test pattern generators
python examples/test_pattern_generators.py
```

## 🎛️ Multi-Engine Architecture

| Engine Type | Location | Purpose | Dependencies |
|-------------|----------|---------|--------------|
| **[FRAMEWORK-AGNOSTIC]** | `cli_shared/` | Interfaces & Models | Pure Python |
| **[STRUDEL-BASED]** | `cli_strudel/` | JavaScript Live Coding | Node.js, Strudel.js |
| **[SUPERCOLLIDER-BASED]** | `cli_sc/` | Professional Synthesis | SuperCollider Server |
| **[CUSTOM-PYTHON]** | `audio/` | Direct NumPy Synthesis | NumPy, SciPy |
| **[TIDALCYCLES-BASED]** | `cli_tidal/` (planned) | Advanced Patterns | Haskell, TidalCycles |

## 📚 Documentation

- **[docs/README.md](docs/README.md)** - Complete documentation index
- **[docs/architecture/](docs/architecture/)** - System architecture and engine integration
- **[docs/user_guide/](docs/user_guide/)** - How to use the Track system
- **[docs/development/](docs/development/)** - Coding standards and testing
- **[CLAUDE.md](CLAUDE.md)** - AI assistant context and project guidelines

## 🏗️ Project Structure

```
├── audio/              # [CUSTOM-PYTHON] Track system & synthesis
├── cli_shared/         # [FRAMEWORK-AGNOSTIC] Interfaces & models  
├── cli_strudel/        # [STRUDEL-BASED] JavaScript integration
├── cli_sc/             # [SUPERCOLLIDER-BASED] SC server integration
├── cli_tui/            # Terminal user interface
├── web_interface/      # Web server interfaces
├── docs/               # Comprehensive documentation
├── examples/           # Working code examples
├── tests/              # Unit, integration & performance tests
└── archive/            # Legacy code (engines/ moved here)
```

## 🎯 Current Capabilities

- ✅ **Professional Track Architecture**: DAW-style composition system
- ✅ **Multi-Engine Support**: Strudel, SuperCollider, Custom Python synthesis
- ✅ **Authentic Hardcore Synthesis**: TR-909 kicks, Rotterdam doorlussen, industrial rumble
- ✅ **Modular Effects**: Distortion, compression, reverb, filtering - all extractable and reusable
- ✅ **Real-time Performance**: <15ms total latency for 4-track sessions
- ✅ **Professional Constants**: NO magic numbers, all parameters documented and user-validated
- ✅ **Engine Interchangeability**: Swap synthesis backends without code changes
- ✅ **Clean Architecture**: Composition over inheritance, fully testable components

## 💻 Requirements

### Core Dependencies
- **Python 3.11+** - Main runtime
- **NumPy, SciPy** - Audio processing mathematics
- **Pedalboard** - Professional effects processing

### Optional Engine Dependencies
- **Node.js + Strudel.js** - For [STRUDEL-BASED] engines
- **SuperCollider Server** - For [SUPERCOLLIDER-BASED] engines
- **TidalCycles + Haskell** - For planned [TIDALCYCLES-BASED] engines

### System Requirements  
- **PulseAudio** (Linux/WSL) or **CoreAudio** (macOS) for audio playback
- **4GB+ RAM** recommended for multi-engine sessions
- **Multi-core CPU** recommended for real-time synthesis

## 🚀 Installation

```bash
# Clone repository
git clone <repository-url>
cd hardcore-music-production

# Install core dependencies
pip install numpy scipy pedalboard pydub mido

# Optional: Install engine dependencies
# For Strudel integration
npm install @strudel/core @strudel/web

# For SuperCollider integration  
# Download and install SuperCollider from supercollider.github.io

# Test installation
python examples/track_architecture_demo.py
```

## 🎵 What You Can Make

- **Rotterdam Gabber**: Authentic doorlussen distortion, 170 BPM kicks
- **Berlin Industrial**: Rumbling sub-bass, warehouse atmosphere
- **UK Hardcore**: Complex breakbeats, screeching leads  
- **Frenchcore**: Extreme distortion, 200+ BPM aggression
- **Multi-Engine Sessions**: Combine different synthesis approaches seamlessly

Built for making **hard, aggressive electronic music** that destroys sound systems.

---

## 🔄 Migration from Legacy Code

**Old engines/ folder** → **Moved to archive/**
- ❌ `ProfessionalHardcoreEngine` → ✅ `Track` + `TrackCollection`  
- ❌ `FrankensteinEngine` → ✅ `audio.synthesis.oscillators`
- ❌ Magic numbers → ✅ `audio.parameters.synthesis_constants`
- ❌ Spaghetti inheritance → ✅ Clean composition patterns

See `docs/user_guide/track_system_guide.md` for migration examples.