#!/usr/bin/env python3
"""
Audio Synthesis Module - Extracted from Legacy Engines

Clean synthesis functions extracted from engines/ spaghetti code.
All functions are modular, reusable, and follow professional audio standards.
"""

# Version info
__version__ = "2.0.0" 
__description__ = "Professional Synthesis Module - Refactored from Legacy Engines"

# Import main synthesis functions
from .oscillators import (
    synthesize_oscillator_bank,
    create_brutalist_envelope, 
    synthesize_simple_kick,
    gabber_oscillator_bank,
    industrial_oscillator_bank,
    terrorcore_oscillator_bank
)