#!/usr/bin/env python3
"""
Professional Audio System for Hardcore Music Production

Engine Type: [CUSTOM-PYTHON]
Dependencies: NumPy, SciPy, pure Python mathematics
Abstraction Level: [LOW-LEVEL]
Integration: Implements audio processing using cli_shared interfaces

Clean, modular architecture following DAW principles.
Extracted from legacy engines/ spaghetti code and made properly reusable.
All synthesis and effects processing performed using pure Python/NumPy/SciPy.

Architecture: Control Source → Audio Source → FX Chain → Mixer
Uses existing cli_shared interfaces - NO reinventing wheels.
"""

# Version info
__version__ = "2.0.0"
__description__ = "Professional Audio System - Refactored from Legacy Engines"

# Import core track architecture
from .core import (
    Track, TrackCollection, EffectsChain,
    PatternControlSource, MidiControlSource,
    KickAudioSource, SynthesizerAudioSource
)

# Import modular effects and synthesis
from .effects import (
    rotterdam_doorlussen, gabber_distortion, warehouse_reverb,
    apply_compression, apply_hardcore_limiter
)

from .synthesis import (
    synthesize_simple_kick, gabber_oscillator_bank, 
    industrial_oscillator_bank
)

from .parameters.synthesis_constants import (
    HardcoreConstants, SynthesisParams, 
    HARDCORE_STYLE_PRESETS, PedalboardPresets
)