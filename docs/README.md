# 📚 Documentation Index

## Architecture Documentation

- **[System Overview](architecture/system_overview.md)** - Complete system architecture with mermaid diagrams
- **[Track Architecture](architecture/track_architecture.md)** - DAW-style Track system replacing legacy engines/
- **[Engine Integration](architecture/engine_integration.md)** - How different synthesis engines integrate

## Development Documentation  

- **[Coding Standards](development/coding_standards.md)** - MANDATORY standards for all development
- **[Testing Strategy](development/testing_strategy.md)** - Unit, integration, and performance testing

## User Guides

- **[Track System Guide](user_guide/track_system_guide.md)** - How to use the new Track system
- **Quick Start Examples**: See `examples/track_architecture_demo.py`

## Research Documentation

- **[Hardcore Synthesis Research](research/hardcore_synthesis_research.md)** - Gabber/industrial synthesis techniques
- **[Alternative Framework Research](research/alternative_audio_framework_research.md)** - Framework analysis and recommendations

## Project Navigation

- **Root README.md** - Project overview and quick start
- **CLAUDE.md** - AI assistant context and project instructions  
- **design/** - Living architecture documentation (being reorganized)
- **examples/** - Working code examples and demos

## Quick Reference

### Engine Types
- `[FRAMEWORK-AGNOSTIC]` - cli_shared/ pure Python abstractions
- `[STRUDEL-BASED]` - cli_strudel/ JavaScript live coding
- `[SUPERCOLLIDER-BASED]` - cli_sc/ professional synthesis server  
- `[CUSTOM-PYTHON]` - audio/ NumPy/SciPy direct synthesis
- `[TIDALCYCLES-BASED]` - Future cli_tidal/ advanced patterns

### Key Concepts
- **Track Architecture**: Control Source → Audio Source → FX Chain → Mixer
- **Engine Backends**: Pluggable synthesis engines via AbstractSynthesizer
- **Composition over Inheritance**: NO more spaghetti engines/ code
- **Constants System**: NO more magic numbers, all in audio.parameters

### Development Workflow
1. Check existing infrastructure (cli_shared/, cli_strudel/, cli_sc/)
2. Use existing interfaces (AbstractSynthesizer, HardcorePattern, etc.)
3. Add proper engine tagging to all modules
4. Follow Track architecture composition pattern
5. Write tests and documentation

---

**Architecture Migration Status**: ✅ **COMPLETE**
- ❌ **Old**: Spaghetti engines/ inheritance with magic numbers  
- ✅ **New**: Professional DAW-style Track system with pluggable engines