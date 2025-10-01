#!/usr/bin/env python3
"""
Shared Components for Hardcore Music Production

Engine Type: [FRAMEWORK-AGNOSTIC]
Dependencies: Pure Python, NumPy, standard libraries only
Abstraction Level: [HIGH-LEVEL]
Integration: Provides interfaces and models used by all engine backends

Common interfaces, models, and utilities used by all synthesis backends
(Strudel, SuperCollider, Custom Python). This module contains NO engine-specific
code - only pure Python abstractions that enable interchangeable backends.
"""

from .interfaces.synthesizer import (
    AbstractSynthesizer,
    AbstractPatternSequencer,
    AbstractAudioAnalyzer,
    AbstractEffectsProcessor,
    BackendType,
    SynthesizerState,
    create_synthesizer,
    compare_backends
)

from .models.hardcore_models import (
    SynthType,
    SynthParams,
    PatternStep,
    HardcoreTrack,
    HardcorePattern,
    AudioAnalysisResult,
    SessionState,
    create_gabber_kick_pattern,
    create_industrial_pattern,
    create_acid_pattern
)

from .utils.audio_utils import (
    db_to_linear,
    linear_to_db,
    calculate_rms,
    calculate_peak,
    calculate_lufs,
    analyze_frequency_bands,
    detect_kick_drum,
    apply_hardcore_processing,
    generate_kick_envelope,
    save_audio_wav,
    load_audio_wav,
    calculate_spectral_features,
    estimate_bpm,
    create_hardcore_test_signals
)

from .utils.midi_utils import (
    HardcoreMIDIController,
    MIDIControllerType,
    MIDIMessage,
    LaunchpadMK1,
    MIDIFighter3D,
    scan_midi_devices
)

from .utils.config import (
    ConfigManager,
    GabberbotConfig,
    AudioConfig,
    MIDIConfig,
    UIConfig,
    HardcoreConfig,
    KeybindConfig,
    ThemeType
)

__version__ = "1.0.0"
__author__ = "Gabberbot Team"
__description__ = "Shared components for hardcore music production"

# Export main classes for easy import
__all__ = [
    # Interfaces
    "AbstractSynthesizer",
    "AbstractPatternSequencer", 
    "AbstractAudioAnalyzer",
    "AbstractEffectsProcessor",
    "BackendType",
    "SynthesizerState",
    "create_synthesizer",
    "compare_backends",
    
    # Models
    "SynthType",
    "SynthParams",
    "PatternStep",
    "HardcoreTrack",
    "HardcorePattern", 
    "AudioAnalysisResult",
    "SessionState",
    "create_gabber_kick_pattern",
    "create_industrial_pattern",
    "create_acid_pattern",
    
    # Audio utilities
    "db_to_linear",
    "linear_to_db",
    "calculate_rms",
    "calculate_peak",
    "calculate_lufs",
    "analyze_frequency_bands", 
    "detect_kick_drum",
    "apply_hardcore_processing",
    "generate_kick_envelope",
    "save_audio_wav",
    "load_audio_wav",
    "calculate_spectral_features",
    "estimate_bpm",
    "create_hardcore_test_signals",
    
    # MIDI utilities
    "HardcoreMIDIController",
    "MIDIControllerType",
    "MIDIMessage",
    "LaunchpadMK1",
    "MIDIFighter3D",
    "scan_midi_devices",
    
    # Configuration
    "ConfigManager",
    "GabberbotConfig",
    "AudioConfig",
    "MIDIConfig", 
    "UIConfig",
    "HardcoreConfig",
    "KeybindConfig",
    "ThemeType"
]