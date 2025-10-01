# 🎵 TidalCycles Integration Architecture

## Overview

TidalCycles will serve as the advanced pattern generation engine for our hardcore music production system, working alongside SuperCollider for synthesis. **Importantly**: Users will NOT write TidalCycles code directly - the AI chatbot translates natural language commands into Tidal patterns.

## Integration Architecture

```
User: "Make a gabber kick pattern with some swing"
         ↓
┌─────────────────────────┐
│   AI Chatbot + NLP      │
│  (Natural Language)     │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Pattern Generator      │
│  (Creates Tidal Code)   │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│    TidalCycles          │
│  (Pattern Evaluation)   │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│    SuperCollider        │
│  (Audio Synthesis)      │
└─────────────────────────┘
```

## Key Design Principles

### 1. User Never Sees Tidal Code
- Natural language is the ONLY interface
- AI chatbot handles all Tidal pattern generation
- Users say "make it faster" not `fast 2 $ s "bd"`

### 2. Pattern Generation Pipeline

```python
class TidalPatternGenerator:
    """
    Engine Type: [TIDALCYCLES-BASED]
    Dependencies: TidalCycles, Haskell runtime, SuperDirt
    Abstraction Level: [HIGH-LEVEL]
    """
    
    def natural_language_to_pattern(self, user_input: str) -> str:
        """
        Converts user's natural language to Tidal pattern
        
        Examples:
        "gabber kick every beat" → 's "bd:5" # struct "x ~ x ~"'
        "make it swing" → 'swing 0.1 $ [previous pattern]'
        "add reverb" → 'room 0.8 # size 0.9 $ [previous pattern]'
        """
        intent = self.nlp_engine.classify(user_input)
        params = self.extract_parameters(user_input)
        return self.generate_tidal_code(intent, params)
```

## Implementation Phases

### Phase 1: Basic Integration (Month 2 of roadmap)

#### Week 5: TidalCycles Setup
```bash
# Install TidalCycles
cabal update
cabal install tidal

# Install SuperDirt in SuperCollider
Quarks.checkForUpdates({Quarks.install("SuperDirt", "v1.7.3"); thisProcess.recompile()})

# Python-Tidal bridge
pip install pytidal  # or custom implementation
```

#### Week 6: Pattern Primitives
Map basic user intents to Tidal patterns:

```python
PATTERN_MAPPINGS = {
    "kick_pattern": {
        "gabber": 's "bd:5" # gain 1.2 # shape 0.9',
        "industrial": 's "bd:2" # room 0.9 # size 0.8',
        "hardcore": 's "bd:8" # speed 1.5 # crush 4'
    },
    "rhythm_modifiers": {
        "faster": lambda p: f"fast 2 $ {p}",
        "slower": lambda p: f"slow 2 $ {p}",
        "swing": lambda p: f"swing 0.1 $ {p}",
        "shuffle": lambda p: f"shuffle 4 $ {p}"
    }
}
```

### Phase 2: Advanced Features (Month 3-4)

#### Complex Pattern Generation
Support for multi-layered patterns and polyrhythms:

```python
def generate_complex_pattern(self, description: str) -> str:
    """
    User: "Layer a 4/4 kick with a 3/4 hi-hat pattern"
    
    Returns:
    stack [
        s "bd*4",
        s "hh*3"
    ]
    """
    layers = self.parse_layers(description)
    return self.build_stack(layers)
```

#### Style-Specific Templates
Pre-built pattern templates for different hardcore subgenres:

```python
STYLE_TEMPLATES = {
    "rotterdam_terror": {
        "base_bpm": 190,
        "kick": 's "bd:5" # shape 0.95 # gain 1.3',
        "pattern": "x ~ x ~ x ~ x ~",
        "effects": ["distortion", "compression"]
    },
    "frenchcore": {
        "base_bpm": 200,
        "kick": 's "bd:8" # crush 8 # gain 1.5',
        "pattern": "x x x x",
        "effects": ["overdrive", "bitcrusher"]
    }
}
```

## Integration with Track System

### Connecting to Existing Architecture

```python
from audio import Track
from cli_tidal.generator import TidalPatternGenerator

class TidalAudioSource(AudioSource):
    """
    Engine Type: [TIDALCYCLES-BASED]
    Integrates TidalCycles patterns with Track system
    """
    
    def __init__(self):
        self.tidal = TidalPatternGenerator()
        self.current_pattern = None
        
    def set_pattern_from_chat(self, user_input: str):
        """Called by AI chatbot when user requests pattern changes"""
        tidal_code = self.tidal.natural_language_to_pattern(user_input)
        self.current_pattern = tidal_code
        self.evaluate_pattern()
        
    def generate_audio(self, params, sample_rate):
        # TidalCycles → SuperCollider → Audio
        return self.render_current_pattern(params)
```

## Natural Language Mappings

### Intent Classification

```python
TIDAL_INTENTS = {
    # Rhythm patterns
    "four_on_floor": 's "bd*4"',
    "breakbeat": 's "[bd*2, ~ cp]"',
    "syncopated": 'off 0.125 $ s "bd*4"',
    
    # Effects
    "warehouse_reverb": 'room 0.9 # size 0.95',
    "gabber_distortion": 'shape 0.95 # crush 4',
    "sidechain": 'gain (slow 4 $ scale 0.5 1 sine)',
    
    # Transformations
    "build_up": 'slow 16 $ striate 128',
    "breakdown": 'degradeBy 0.5',
    "drop": 'gain 1.5 # speed 1'
}
```

### Parameter Extraction

```python
def extract_tidal_params(user_input: str) -> dict:
    """
    Extract Tidal-relevant parameters from natural language
    
    "Make it 20% faster with heavy reverb"
    → {"speed": 1.2, "room": 0.8, "size": 0.9}
    """
    params = {}
    
    # Speed/tempo modifications
    if "faster" in user_input:
        params["speed"] = extract_percentage(user_input, default=1.2)
    
    # Effects
    if "reverb" in user_input:
        intensity = extract_intensity(user_input)  # light/medium/heavy
        params.update(reverb_settings[intensity])
    
    return params
```

## Performance Considerations

### Latency Management
- Target: <20ms from pattern change to audio output
- Use OSC for real-time communication
- Pre-compile common patterns

### Pattern Caching
```python
class PatternCache:
    """Cache frequently used patterns for instant recall"""
    
    def __init__(self):
        self.cache = {}
        self.precompile_common_patterns()
        
    def get_pattern(self, description: str) -> Optional[str]:
        # Check cache first, generate if not found
        if description in self.cache:
            return self.cache[description]
        return None
```

## Error Handling

### Graceful Degradation
If TidalCycles fails, fall back to simpler pattern generation:

```python
def generate_pattern_safe(self, user_input: str) -> str:
    try:
        return self.tidal_generator.create_pattern(user_input)
    except TidalError:
        # Fall back to simple Python pattern
        return self.simple_pattern_fallback(user_input)
```

## Testing Strategy

### Unit Tests
```python
def test_natural_language_to_tidal():
    generator = TidalPatternGenerator()
    
    # Test basic patterns
    assert generator.convert("kick every beat") == 's "bd*4"'
    assert generator.convert("add swing") == 'swing 0.1 $ s "bd*4"'
    
    # Test complex patterns
    pattern = generator.convert("gabber kick with reverb")
    assert 'bd:5' in pattern
    assert 'room' in pattern
```

### Integration Tests
- Test full pipeline: Chat → Tidal → SuperCollider → Audio
- Verify latency requirements (<20ms)
- Test pattern transitions and morphing

## Migration Path

### From Strudel to TidalCycles

1. **Maintain Compatibility**: Keep Strudel as fallback during transition
2. **Feature Parity**: Ensure all Strudel patterns can be recreated in Tidal
3. **Gradual Migration**: Move one pattern type at a time
4. **User Transparency**: Users shouldn't notice the backend change

```python
class PatternBackendSelector:
    """Dynamically choose between Strudel and TidalCycles"""
    
    def __init__(self):
        self.use_tidal = feature_flag("USE_TIDALCYCLES")
        
    def generate(self, user_input: str) -> str:
        if self.use_tidal and self.tidal_available():
            return self.tidal_generator.create(user_input)
        else:
            return self.strudel_generator.create(user_input)
```

## Future Enhancements

### Advanced AI Integration
- Train models on TidalCycles pattern corpus
- Learn user's preferred pattern styles
- Generate novel patterns using ML

### Live Coding Features
- Record chat sessions as reproducible performances
- Export conversation history as Tidal script (for advanced users)
- Real-time collaboration through shared chat sessions

## Summary

TidalCycles integration provides:
- **Superior pattern generation** compared to current Strudel implementation
- **Professional quality** through SuperCollider backend
- **User-friendly** through natural language interface
- **Extensible** for future AI enhancements

The key success factor: Users get professional pattern generation without ever seeing or writing TidalCycles code.