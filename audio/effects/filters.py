#!/usr/bin/env python3
"""
Audio Filters - Extracted from Legacy Engines

Digital and analog-modeled filtering functions for hardcore music production.
All functions extracted from frankenstein_engine.py and made properly modular.
"""

import numpy as np
from scipy import signal
from typing import Optional
from ..parameters.synthesis_constants import SynthesisParams, HardcoreConstants


def analog_lowpass_filter(audio: np.ndarray,
                         cutoff_hz: float,
                         sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply analog-modeled lowpass filter.
    
    Extracted from frankenstein_engine.py _analog_filter().
    One-pole lowpass with analog-style response characteristics.
    
    Args:
        audio: Input audio signal
        cutoff_hz: Cutoff frequency in Hz
        sample_rate: Audio sample rate
        
    Returns:
        Filtered audio signal
    """
    if len(audio) == 0:
        return audio
        
    nyquist = sample_rate / 2
    if cutoff_hz >= nyquist * 0.95:
        return audio
        
    # One-pole lowpass with analog-style response
    alpha = np.exp(-2 * np.pi * cutoff_hz / sample_rate)
    
    filtered = np.zeros_like(audio)
    filtered[0] = audio[0] * (1 - alpha)
    
    for i in range(1, len(audio)):
        filtered[i] = audio[i] * (1 - alpha) + filtered[i-1] * alpha
        
    return filtered


def apply_industrial_rumble(audio: np.ndarray, 
                           params: SynthesisParams,
                           sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Add Berlin-style industrial sub rumble.
    
    Extracted from frankenstein_engine.py apply_industrial_rumble().
    Creates characteristic industrial sub-bass rumble with metallic resonance.
    
    Args:
        audio: Input audio signal
        params: Synthesis parameters including rumble_tail and metallic_ring
        sample_rate: Audio sample rate
        
    Returns:
        Audio with industrial rumble added
    """
    if len(audio) == 0 or params.rumble_tail <= 0:
        return audio
        
    samples = len(audio)
    t = np.arange(samples) / sample_rate
    
    # Generate sub-bass rumble
    rumble_freq = params.frequency * 0.4  # Sub frequency
    rumble_envelope = np.exp(-2 * t) * params.rumble_tail
    
    # Multiple rumble components
    rumble = np.zeros_like(t)
    rumble += np.sin(2 * np.pi * rumble_freq * t) * rumble_envelope
    rumble += np.sin(2 * np.pi * rumble_freq * 0.75 * t) * rumble_envelope * 0.6
    
    # Add metallic resonance if specified
    if params.metallic_ring > 0:
        metallic_freq = params.frequency * 1.5
        metallic_envelope = np.exp(-8 * t) * params.metallic_ring * 0.3
        resonance = np.sin(2 * np.pi * metallic_freq * t) * metallic_envelope
        rumble += resonance
        
    # Filter rumble to sub-bass range
    rumble = analog_lowpass_filter(rumble, 120, sample_rate)
    
    return audio + rumble * 0.7


def highpass_filter(audio: np.ndarray,
                   cutoff_hz: float,
                   sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K,
                   order: int = 2) -> np.ndarray:
    """
    Apply highpass filter for kick space cleaning.
    
    Standard Butterworth highpass filter for removing low frequencies
    and creating space for kick drums.
    
    Args:
        audio: Input audio signal
        cutoff_hz: Cutoff frequency in Hz
        sample_rate: Audio sample rate
        order: Filter order (steepness)
        
    Returns:
        Highpass filtered audio
    """
    if len(audio) == 0:
        return audio
        
    # Butterworth highpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    if normalized_cutoff >= 1.0:
        return np.zeros_like(audio)
    if normalized_cutoff <= 0.0:
        return audio
        
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio)


def lowpass_filter(audio: np.ndarray,
                  cutoff_hz: float, 
                  sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K,
                  order: int = 2) -> np.ndarray:
    """
    Apply lowpass filter for harshness control.
    
    Standard Butterworth lowpass filter for removing harsh high frequencies.
    
    Args:
        audio: Input audio signal
        cutoff_hz: Cutoff frequency in Hz
        sample_rate: Audio sample rate
        order: Filter order (steepness)
        
    Returns:
        Lowpass filtered audio
    """
    if len(audio) == 0:
        return audio
        
    # Butterworth lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    if normalized_cutoff >= 1.0:
        return audio
    if normalized_cutoff <= 0.0:
        return np.zeros_like(audio)
        
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)


def bandpass_filter(audio: np.ndarray,
                   low_cutoff_hz: float,
                   high_cutoff_hz: float,
                   sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K,
                   order: int = 2) -> np.ndarray:
    """
    Apply bandpass filter for frequency isolation.
    
    Standard Butterworth bandpass filter for isolating frequency ranges.
    
    Args:
        audio: Input audio signal
        low_cutoff_hz: Low cutoff frequency in Hz
        high_cutoff_hz: High cutoff frequency in Hz
        sample_rate: Audio sample rate
        order: Filter order (steepness)
        
    Returns:
        Bandpass filtered audio
    """
    if len(audio) == 0:
        return audio
        
    # Butterworth bandpass filter
    nyquist = sample_rate / 2
    low_norm = low_cutoff_hz / nyquist
    high_norm = high_cutoff_hz / nyquist
    
    if low_norm >= 1.0 or high_norm <= 0.0 or low_norm >= high_norm:
        return np.zeros_like(audio)
    if low_norm <= 0.0 and high_norm >= 1.0:
        return audio
        
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    return signal.filtfilt(b, a, audio)


# Preset functions for hardcore music
def apply_kick_space_highpass(audio: np.ndarray, 
                             sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply highpass filter to create kick space (120Hz)"""
    return highpass_filter(audio, HardcoreConstants.HIGHPASS_KICK_SPACE_HZ, sample_rate)


def apply_harshness_lowpass(audio: np.ndarray,
                           sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply lowpass filter to control harsh frequencies (8kHz)"""
    return lowpass_filter(audio, HardcoreConstants.LOWPASS_HARSHNESS_HZ, sample_rate)


def apply_industrial_rumble_preset(audio: np.ndarray, 
                                  frequency: float = 45.0,
                                  sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply Berlin industrial rumble preset"""
    params = SynthesisParams(
        frequency=frequency,
        rumble_tail=1.0,
        metallic_ring=0.4
    )
    return apply_industrial_rumble(audio, params, sample_rate)