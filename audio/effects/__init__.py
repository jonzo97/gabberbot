#!/usr/bin/env python3
"""
Audio Effects Module - Extracted from Legacy Engines

Clean effects functions extracted from engines/ spaghetti code.
All functions are modular, reusable, and follow professional audio standards.
"""

# Version info
__version__ = "2.0.0"
__description__ = "Professional Audio Effects - Refactored from Legacy Engines"

# Import main effects functions
from .distortion import (
    rotterdam_doorlussen,
    analog_warmth,
    alpha_juno_distortion,
    industrial_distortion,
    gabber_distortion,
    soft_compression_overdrive,
    apply_rotterdam_gabber_distortion,
    apply_industrial_hardcore_distortion,
    apply_classic_hoover_distortion,
    apply_extreme_gabber_distortion
)

from .spatial import (
    warehouse_reverb,
    industrial_reverb,
    create_echo_delay,
    apply_warehouse_atmosphere,
    apply_industrial_atmosphere,
    apply_eighth_note_delay
)

from .dynamics import (
    apply_compression,
    apply_serial_compression,
    apply_parallel_compression,
    apply_hardcore_compression_chain,
    apply_hardcore_limiter,
    apply_gabber_compression,
    apply_industrial_compression,
    apply_terrorcore_compression
)

from .filters import (
    analog_lowpass_filter,
    apply_industrial_rumble,
    highpass_filter,
    lowpass_filter,
    bandpass_filter,
    apply_kick_space_highpass,
    apply_harshness_lowpass,
    apply_industrial_rumble_preset
)