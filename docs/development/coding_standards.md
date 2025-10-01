# 🔧 Coding Standards

## MANDATORY CODE QUALITY STANDARDS

### Non-Negotiable Principles

1. **Quality Over Speed** - NO rushing to implement features with poor architecture
2. **Use Existing Infrastructure** - We have professional interfaces in cli_shared/
3. **NO Reinventing Wheels** - Check cli_shared/, cli_strudel/, cli_sc/ before writing new code
4. **Composition Over Inheritance** - Use Track system composition pattern
5. **NO Magic Numbers** - All constants must be in audio.parameters.synthesis_constants
6. **Engine Tagging Required** - All modules must be tagged with engine type

### Engine Type Documentation

All modules MUST include engine identification:

```python
"""
[Module Description]

Engine Type: [STRUDEL-BASED] | [SUPERCOLLIDER-BASED] | [CUSTOM-PYTHON] | [FRAMEWORK-AGNOSTIC]
Dependencies: [Specific dependencies]
Abstraction Level: [LOW-LEVEL] | [MID-LEVEL] | [HIGH-LEVEL]
Integration: [Integration description]
"""
```

### Engine Categories

- **[FRAMEWORK-AGNOSTIC]**: cli_shared/ - Pure Python abstractions
- **[STRUDEL-BASED]**: cli_strudel/ - JavaScript/Strudel integration  
- **[SUPERCOLLIDER-BASED]**: cli_sc/ - SuperCollider server integration
- **[CUSTOM-PYTHON]**: audio/ - NumPy/SciPy synthesis
- **[TIDALCYCLES-BASED]**: Future cli_tidal/ - Haskell TidalCycles integration

### Interface Requirements

All new components MUST use existing interfaces:

```python
# MANDATORY - Use AbstractSynthesizer interface
from cli_shared.interfaces.synthesizer import AbstractSynthesizer

# PREFERRED - Use existing models
from cli_shared.models.hardcore_models import HardcorePattern, SynthParams

# REQUIRED - Use extraction constants, not magic numbers
from audio.parameters.synthesis_constants import HardcoreConstants
```

### Track Architecture Pattern

Use composition, NOT inheritance:

```python
# ✅ CORRECT - Composition
class Track:
    def __init__(self):
        self.control_source = None  # Pluggable component
        self.audio_source = None    # Pluggable component
        self.effects_chain = None   # Pluggable component

# ❌ WRONG - Spaghetti inheritance
class MyTrack(ProfessionalEngine):  # NO MORE OF THIS
    pass
```

### File Organization Standards

- **Root Level**: Only README.md, CLAUDE.md, and framework research
- **Module Level**: Proper __init__.py with engine tagging
- **Import Paths**: Always use absolute paths from project root
- **Archive Policy**: Move legacy/prototype code to archive/ immediately

### Testing Requirements

- All new Track components must have unit tests
- Integration tests required for engine backends
- Performance benchmarks for synthesis functions
- Audio quality validation for effects

### Performance Standards

- Track rendering: <5ms per track per step
- Effect processing: <3ms per effect
- Memory usage: <50MB per session
- Engine startup: Document and optimize startup times

### Documentation Standards

- All public functions must have docstrings
- Architecture diagrams must be kept current
- Engine integration points must be documented
- Usage examples required for all new components

These standards are MANDATORY and will be enforced in all code reviews and development work.