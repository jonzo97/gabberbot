#!/usr/bin/env python3
"""
Spatial Audio Effects - Extracted from Legacy Engines

Modular, reusable spatial audio processing functions for hardcore music production.
All functions are standalone and can be imported/used independently.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional

# Constants extracted from magic numbers
class SpatialConstants:
    """Spatial effects constants - no more magic numbers!"""
    
    # Warehouse reverb parameters
    WAREHOUSE_EARLY_DELAYS_MS = [23, 37, 61, 89, 127]  # Prime numbers for less periodicity
    WAREHOUSE_LATE_DELAYS_MS = [180, 240, 320, 410, 520, 640]  # Late reflections
    WAREHOUSE_DAMPING_FREQ = 3000  # High frequency damping cutoff
    WAREHOUSE_DEFAULT_ROOM_SIZE = 0.8
    WAREHOUSE_DEFAULT_WET_LEVEL = 0.3
    WAREHOUSE_DEFAULT_DAMPING = 0.4
    
    # Industrial reverb parameters
    INDUSTRIAL_REVERB_DECAY = 0.7
    INDUSTRIAL_WET_LEVEL = 0.5
    INDUSTRIAL_PREDELAY_MS = 12
    
    # Standard sample rates
    SAMPLE_RATE_44K = 44100
    SAMPLE_RATE_48K = 48000


def warehouse_reverb(audio: np.ndarray, 
                    sample_rate: int = SpatialConstants.SAMPLE_RATE_44K,
                    room_size: float = SpatialConstants.WAREHOUSE_DEFAULT_ROOM_SIZE,
                    wet_level: float = SpatialConstants.WAREHOUSE_DEFAULT_WET_LEVEL,
                    damping: float = SpatialConstants.WAREHOUSE_DEFAULT_DAMPING) -> np.ndarray:
    """
    Apply warehouse-style reverb for industrial hardcore atmosphere.
    
    Extracted from final_brutal_hardcore.py and made modular.
    Creates characteristic warehouse/rave venue reverb using multiple delay lines.
    
    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate
        room_size: Size of warehouse space (0.0-1.0)
        wet_level: Reverb wet/dry mix (0.0-1.0)
        damping: High frequency damping amount (0.0-1.0)
    
    Returns:
        Audio with warehouse reverb applied
    """
    if len(audio) == 0:
        return audio
        
    reverb = audio.copy()
    
    # Early reflections (warehouse walls)
    early_delays = [int(delay * room_size) for delay in SpatialConstants.WAREHOUSE_EARLY_DELAYS_MS]
    
    for delay_ms in early_delays:
        delay_samples = int(delay_ms * sample_rate / 1000)
        if delay_samples < len(audio):
            delayed = np.roll(audio, delay_samples) * (0.4 / len(early_delays))
            reverb += delayed
            
    # Late reverb (warehouse space)
    late_delays = [int(delay * room_size) for delay in SpatialConstants.WAREHOUSE_LATE_DELAYS_MS]
    
    for i, delay_ms in enumerate(late_delays):
        delay_samples = int(delay_ms * sample_rate / 1000)
        if delay_samples < len(audio):
            # Apply damping (high freq roll-off)
            delayed = np.roll(audio, delay_samples)
            
            # High frequency damping for realism
            damping_freq = SpatialConstants.WAREHOUSE_DAMPING_FREQ * (1.0 - damping)
            if damping_freq > 0:
                b, a = signal.butter(2, damping_freq / (sample_rate / 2), btype='low')
                delayed = signal.filtfilt(b, a, delayed)
            
            # Decay over time (each reflection quieter)
            decay_factor = 0.6 ** (i + 1) * room_size
            delayed *= decay_factor
            
            reverb += delayed
            
    # Mix wet/dry
    return audio * (1.0 - wet_level) + reverb * wet_level


def industrial_reverb(audio: np.ndarray,
                     sample_rate: int = SpatialConstants.SAMPLE_RATE_44K,
                     decay: float = SpatialConstants.INDUSTRIAL_REVERB_DECAY,
                     wet_level: float = SpatialConstants.INDUSTRIAL_WET_LEVEL,
                     predelay_ms: float = SpatialConstants.INDUSTRIAL_PREDELAY_MS) -> np.ndarray:
    """
    Apply industrial-style reverb for dark hardcore atmosphere.
    
    Creates characteristic industrial reverb with metallic characteristics.
    
    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate
        decay: Reverb decay time (0.0-1.0)
        wet_level: Reverb wet/dry mix (0.0-1.0)
        predelay_ms: Pre-delay in milliseconds
        
    Returns:
        Audio with industrial reverb applied
    """
    if len(audio) == 0:
        return audio
    
    # Pre-delay
    predelay_samples = int(predelay_ms * sample_rate / 1000)
    if predelay_samples > 0:
        delayed_audio = np.concatenate([np.zeros(predelay_samples), audio])[:len(audio)]
    else:
        delayed_audio = audio.copy()
    
    # Industrial reverb characteristics - shorter, darker
    reverb = delayed_audio.copy()
    
    # Industrial delay pattern (metallic, harsh)
    delays_ms = [45, 73, 112, 167, 251]  # Irregular spacing for metallic character
    
    for i, delay_ms in enumerate(delays_ms):
        delay_samples = int(delay_ms * sample_rate / 1000)
        if delay_samples < len(audio):
            delayed = np.roll(delayed_audio, delay_samples)
            
            # Apply metallic filtering (boost mids, cut highs)
            b, a = signal.butter(2, [800, 2500], btype='band', fs=sample_rate)
            delayed = signal.filtfilt(b, a, delayed)
            
            # Exponential decay
            decay_factor = decay ** (i + 1)
            delayed *= decay_factor
            
            reverb += delayed
    
    # Mix wet/dry
    return audio * (1.0 - wet_level) + reverb * wet_level


def create_echo_delay(audio: np.ndarray,
                     sample_rate: int = SpatialConstants.SAMPLE_RATE_44K,
                     delay_ms: float = 125.0,  # 8th note at 120 BPM
                     feedback: float = 0.4,
                     wet_level: float = 0.2) -> np.ndarray:
    """
    Create echo delay effect for hardcore music.
    
    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate
        delay_ms: Delay time in milliseconds
        feedback: Feedback amount (0.0-0.95)
        wet_level: Wet/dry mix (0.0-1.0)
        
    Returns:
        Audio with echo delay applied
    """
    if len(audio) == 0:
        return audio
        
    delay_samples = int(delay_ms * sample_rate / 1000)
    if delay_samples >= len(audio) or delay_samples <= 0:
        return audio
    
    output = audio.copy()
    delayed = np.concatenate([np.zeros(delay_samples), audio])[:len(audio)]
    
    # Add feedback (delayed signal fed back into delay line)
    if feedback > 0.0:
        feedback_signal = delayed * feedback
        output += feedback_signal
    
    # Mix wet/dry
    return audio * (1.0 - wet_level) + (output + delayed) * wet_level


# Preset functions for easy use
def apply_warehouse_atmosphere(audio: np.ndarray, sample_rate: int = SpatialConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply warehouse atmosphere preset - ready to use for hardcore tracks"""
    return warehouse_reverb(audio, sample_rate, room_size=0.85, wet_level=0.3, damping=0.4)


def apply_industrial_atmosphere(audio: np.ndarray, sample_rate: int = SpatialConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply industrial atmosphere preset - ready to use for dark hardcore"""
    return industrial_reverb(audio, sample_rate, decay=0.6, wet_level=0.4, predelay_ms=8)


def apply_eighth_note_delay(audio: np.ndarray, bpm: float, sample_rate: int = SpatialConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply 8th note delay based on tempo - ready to use for any BPM"""
    delay_ms = (60.0 / bpm) * 1000 / 2  # 8th note delay
    return create_echo_delay(audio, sample_rate, delay_ms, feedback=0.35, wet_level=0.2)