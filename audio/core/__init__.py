#!/usr/bin/env python3
"""
Audio Core Module - Professional Track Architecture

Clean, DAW-style track architecture that uses existing interfaces
and extracted audio modules. NO spaghetti inheritance.

Architecture: Control Source → Audio Source → FX Chain → Mixer
"""

# Version info
__version__ = "2.0.0"
__description__ = "Professional Track Architecture - Built on Existing Interfaces"

# Import main track components
from .track import (
    # Core track classes
    Track,
    TrackCollection,
    EffectsChain,
    EffectSettings,
    
    # Control sources
    ControlSource,
    PatternControlSource,
    MidiControlSource,
    
    # Audio sources
    AudioSource,
    KickAudioSource,
    SynthesizerAudioSource
)