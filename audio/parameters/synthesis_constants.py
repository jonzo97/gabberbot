#!/usr/bin/env python3
"""
Synthesis Constants - Extracted from Legacy Engines

All synthesis parameters consolidated from engines/ spaghetti code.
NO MORE MAGIC NUMBERS - everything documented and parameterized.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class HardcoreStyle(Enum):
    """Hardcore music styles with specific synthesis characteristics"""
    ROTTERDAM_GABBER = "rotterdam_gabber"
    BERLIN_INDUSTRIAL = "berlin_industrial" 
    UK_HARDCORE = "uk_hardcore"
    FRENCHCORE = "frenchcore"
    SPEEDCORE = "speedcore"
    TERRORCORE = "terrorcore"


@dataclass
class SynthesisParams:
    """Complete synthesis parameters for hardcore styles"""
    # Core synthesis
    frequency: float = 60.0          # Base frequency (Hz)
    duration_ms: int = 400           # Note duration
    bpm: int = 180                   # Target BPM
    
    # Character parameters
    brutality: float = 0.8           # Overall aggression (0-1)
    crunch_factor: float = 0.7       # Digital crispiness (0-1)
    analog_warmth: float = 0.5       # Analog character (0-1)
    
    # Rotterdam doorlussen technique
    doorlussen_stages: int = 3       # Serial distortion stages
    mixer_overdrive: float = 2.5     # Analog mixer modeling
    
    # Industrial elements
    rumble_tail: float = 0.8         # Sub-bass rumble amount (0-1)
    metallic_ring: float = 0.3       # Metallic resonance (0-1)
    
    # Digital processing
    bit_crushing: float = 0.2        # Digital degradation (0-1)
    spectral_tilt: float = 0.4       # Frequency balance (0-1)
    
    # Compression characteristics
    serial_compression: bool = True   # Multiple compressor stages
    parallel_compression: float = 0.6 # NY compression amount (0-1)


class HardcoreConstants:
    """Hardcore synthesis constants extracted from working engines"""
    
    # Sample rates
    SAMPLE_RATE_44K = 44100
    SAMPLE_RATE_48K = 48000
    
    # Kick synthesis (from final_brutal_hardcore.py)
    KICK_ATTACK_MS = 0.5
    KICK_DECAY_MS = 50
    KICK_SUB_FREQS = [41.2, 82.4, 123.6]  # E1, E2, E2+fifth Hz - USER VALIDATED
    KICK_EQ_BOOST_FREQ = 60              # Hz
    KICK_EQ_BOOST_DB = 4                 # dB gain
    
    # Arp/Lead synthesis (from midi_based_hardcore.py) 
    DETUNE_CENTS = [-19, -10, -5, 0, 5, 10, 19, 29]  # Reduced by 20% per user feedback
    FILTER_CUTOFF_HZ = 2000
    FILTER_RESONANCE = 0.7
    ENVELOPE_ATTACK = 0.001
    ENVELOPE_DECAY = 0.05
    ENVELOPE_SUSTAIN = 0.6
    ENVELOPE_RELEASE = 0.1
    
    # Distortion (user-validated)
    HARDCORE_DISTORTION_DB = 15          # Reduced from 18 per user feedback
    GABBER_DISTORTION_DB = 18           # Rotterdam style
    INDUSTRIAL_DISTORTION_DB = 12       # Berlin style
    
    # Filtering
    HIGHPASS_KICK_SPACE_HZ = 120        # Clean low-end for kick space
    LOWPASS_HARSHNESS_HZ = 8000         # Control harsh frequencies
    
    # Compression
    HARDCORE_COMPRESSION_RATIO = 8       # 8:1 ratio
    HARDCORE_THRESHOLD_DB = -10         # Threshold
    INDUSTRIAL_COMPRESSION_RATIO = 4    # Gentler for industrial
    
    # Limiting
    MASTER_LIMITER_THRESHOLD = -0.5     # Final limiting
    TRACK_LIMITER_THRESHOLD = -1.0      # Per-track limiting
    
    # Bitcrush (user-validated)
    BITCRUSH_DEPTH = 12                 # Reduced for cleaner sound


# Style-specific presets extracted from frankenstein_engine.py
HARDCORE_STYLE_PRESETS: Dict[HardcoreStyle, SynthesisParams] = {
    HardcoreStyle.ROTTERDAM_GABBER: SynthesisParams(
        frequency=55, duration_ms=600, brutality=0.7, crunch_factor=0.4,
        analog_warmth=0.8, doorlussen_stages=2, mixer_overdrive=2.0,
        rumble_tail=0.9, metallic_ring=0.6, bit_crushing=0.0,
        spectral_tilt=0.7, parallel_compression=0.8
    ),
    
    HardcoreStyle.BERLIN_INDUSTRIAL: SynthesisParams(
        frequency=45, duration_ms=800, brutality=0.6, crunch_factor=0.3,
        analog_warmth=0.9, doorlussen_stages=1, mixer_overdrive=1.5,
        rumble_tail=1.0, metallic_ring=0.4, bit_crushing=0.0,
        spectral_tilt=0.3, parallel_compression=0.9
    ),
    
    HardcoreStyle.UK_HARDCORE: SynthesisParams(
        frequency=70, duration_ms=300, brutality=0.85, crunch_factor=0.9,
        analog_warmth=0.4, doorlussen_stages=3, mixer_overdrive=2.8,
        rumble_tail=0.4, bit_crushing=0.3, serial_compression=True
    ),
    
    HardcoreStyle.FRENCHCORE: SynthesisParams(
        frequency=75, duration_ms=350, brutality=0.95, crunch_factor=0.95,
        analog_warmth=0.3, doorlussen_stages=5, mixer_overdrive=4.0,
        rumble_tail=0.5, bit_crushing=0.4, serial_compression=True,
        parallel_compression=0.7
    ),
    
    HardcoreStyle.SPEEDCORE: SynthesisParams(
        frequency=80, duration_ms=200, brutality=1.0, crunch_factor=1.0,
        analog_warmth=0.2, doorlussen_stages=6, mixer_overdrive=5.0,
        rumble_tail=0.2, bit_crushing=0.6, serial_compression=True
    ),
    
    HardcoreStyle.TERRORCORE: SynthesisParams(
        frequency=90, duration_ms=150, brutality=1.0, crunch_factor=1.0,
        analog_warmth=0.1, doorlussen_stages=8, mixer_overdrive=6.0,
        rumble_tail=0.1, bit_crushing=0.8, serial_compression=True
    )
}


class PedalboardPresets:
    """Pedalboard preset constants extracted from professional_hardcore_engine.py"""
    
    # ANGERFIST (Rotterdam Gabber)
    ANGERFIST_HIGHPASS_HZ = 120
    ANGERFIST_DISTORTION_DB = 18
    ANGERFIST_COMPRESSION_THRESHOLD = -10
    ANGERFIST_COMPRESSION_RATIO = 8
    ANGERFIST_FILTER_CUTOFF = 2000
    ANGERFIST_FILTER_RESONANCE = 0.7
    ANGERFIST_LIMITER_THRESHOLD = -0.5
    
    # HOOVER (Classic Alpha Juno)
    HOOVER_FILTER_CUTOFF = 800
    HOOVER_FILTER_RESONANCE = 0.8
    HOOVER_DISTORTION_DB = 15
    HOOVER_CHORUS_RATE = 0.5
    HOOVER_CHORUS_DEPTH = 0.3
    HOOVER_CHORUS_DELAY = 7  # ms
    HOOVER_HIGHPASS_HZ = 100
    HOOVER_COMPRESSION_THRESHOLD = -12
    HOOVER_COMPRESSION_RATIO = 6
    
    # WAREHOUSE (Industrial Atmosphere)  
    WAREHOUSE_REVERB_ROOM_SIZE = 0.85
    WAREHOUSE_REVERB_WET_LEVEL = 0.3
    WAREHOUSE_DELAY_TIME = 0.125  # seconds (8th note)
    WAREHOUSE_DELAY_FEEDBACK = 0.4
    WAREHOUSE_DELAY_MIX = 0.2
    WAREHOUSE_LOWPASS_HZ = 8000
    WAREHOUSE_COMPRESSION_THRESHOLD = -15
    WAREHOUSE_COMPRESSION_RATIO = 4


class MIDIConstants:
    """MIDI constants extracted from midi_based_hardcore.py"""
    
    # Standard MIDI
    MIDI_A4_NOTE = 69
    MIDI_A4_FREQ = 440.0
    
    # Velocity ranges for hardcore
    HARDCORE_VELOCITY_MIN = 80
    HARDCORE_VELOCITY_MAX = 127
    GABBER_VELOCITY_MIN = 100
    GABBER_VELOCITY_MAX = 127
    
    # Common hardcore scales (MIDI note numbers)
    E_MINOR_SCALE = [64, 66, 67, 69, 71, 72, 74, 76]  # E minor
    A_MINOR_SCALE = [57, 59, 60, 62, 64, 65, 67, 69]  # A minor (common in hardcore)


class TimingConstants:
    """Timing constants for hardcore music"""
    
    # BPM ranges
    HARDCORE_BPM_MIN = 150
    HARDCORE_BPM_MAX = 200
    GABBER_BPM_MIN = 150  
    GABBER_BPM_MAX = 180
    FRENCHCORE_BPM_MIN = 180
    FRENCHCORE_BPM_MAX = 220
    SPEEDCORE_BPM_MIN = 200
    SPEEDCORE_BPM_MAX = 300
    
    # Pattern lengths (measures)
    STANDARD_PATTERN_LENGTH = 4  # 4/4 measures
    EXTENDED_PATTERN_LENGTH = 8  # 8 measure patterns
    BREAKDOWN_PATTERN_LENGTH = 2  # Short breakdown patterns