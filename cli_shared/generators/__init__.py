#!/usr/bin/env python3
"""
Pattern Generators for Hardcore Music Production

Reusable generators that create standard MIDIClip and TriggerClip objects.
All generators follow consistent interface and output industry-standard MIDI.
"""

from .acid_bassline import AcidBasslineGenerator
from .tuned_kick import TunedKickGenerator

# TODO: Add these generators
# from .riff_generator import RiffGenerator  
# from .chord_progression import ChordProgressionGenerator
# from .arpeggio_generator import ArpeggioGenerator

__all__ = [
    'AcidBasslineGenerator',
    'TunedKickGenerator',
    # 'RiffGenerator', 
    # 'ChordProgressionGenerator',
    # 'ArpeggioGenerator',
]