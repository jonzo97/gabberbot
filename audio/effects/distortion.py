#!/usr/bin/env python3
"""
Distortion Effects - Extracted from Legacy Engines

Modular distortion and saturation functions for hardcore music production.
Includes authentic Rotterdam doorlussen technique and other hardcore distortion methods.
"""

import numpy as np
from scipy import signal
from typing import Optional
from ..parameters.synthesis_constants import HardcoreConstants, PedalboardPresets


def rotterdam_doorlussen(audio: np.ndarray,
                        stages: int = 3,
                        drive_per_stage: float = 2.5,
                        sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply Rotterdam "doorlussen" (looping through) distortion technique.
    
    This is the authentic gabber distortion method: serial mixer overdrive stages
    with EQ shaping between each stage. Extracted from frankenstein_engine.py.
    
    Args:
        audio: Input audio signal
        stages: Number of serial distortion stages (2-8 typical)
        drive_per_stage: Overdrive amount per stage
        sample_rate: Audio sample rate
        
    Returns:
        Audio with doorlussen distortion applied
    """
    if len(audio) == 0:
        return audio
        
    processed = audio.copy()
    
    for stage in range(stages):
        # Analog mixer overdrive modeling
        # Each stage: gain -> clipping -> EQ -> next stage
        
        # Drive the signal
        driven = processed * drive_per_stage
        
        # Analog-style soft clipping (tanh saturation)
        clipped = np.tanh(driven)
        
        # EQ shaping between stages (crucial for doorlussen character)
        if stage < stages - 1:  # Don't EQ the final stage
            # High-pass filter to add "crack" (typical mixer channel EQ)
            if stage % 2 == 0:  # Alternate between HP and peak boost
                b, a = signal.butter(2, 200 / (sample_rate / 2), btype='high')
                clipped = signal.filtfilt(b, a, clipped)
            else:
                # Peak boost around 2-3kHz for aggressive character
                b, a = signal.butter(2, [1800, 3200], btype='band', fs=sample_rate)
                peak_boost = signal.filtfilt(b, a, clipped) * 1.5
                clipped = clipped + peak_boost
        
        processed = clipped
        
        # Reduce gain for next stage (prevent runaway)
        processed *= 0.8
    
    # Final level adjustment
    processed *= 0.5
    return processed


def analog_warmth(audio: np.ndarray,
                 saturation: float = 0.7,
                 harmonic_enhancement: float = 0.3) -> np.ndarray:
    """
    Apply analog warmth and harmonic enhancement.
    
    Adds even-order harmonics characteristic of analog equipment.
    Extracted from frankenstein_engine synthesis algorithms.
    
    Args:
        audio: Input audio signal
        saturation: Amount of analog saturation (0.0-1.0)
        harmonic_enhancement: Even harmonic enhancement (0.0-1.0)
        
    Returns:
        Audio with analog warmth applied
    """
    if len(audio) == 0:
        return audio
        
    # Gentle saturation curve for analog warmth
    warmth_curve = lambda x: x * (1.0 - saturation) + np.tanh(x * 2) * saturation
    warmed = warmth_curve(audio)
    
    # Even harmonic enhancement (characteristic of tube/analog saturation)
    if harmonic_enhancement > 0:
        # Generate 2nd harmonic
        second_harmonic = np.tanh(audio * 0.5) * 0.1 * harmonic_enhancement
        warmed += second_harmonic
    
    return warmed


def alpha_juno_distortion(audio: np.ndarray,
                         drive_db: float = PedalboardPresets.HOOVER_DISTORTION_DB) -> np.ndarray:
    """
    Apply Alpha Juno style distortion for hoover sounds.
    
    Characteristic distortion of the Roland Alpha Juno synthesizer,
    essential for authentic hoover sound recreation.
    
    Args:
        audio: Input audio signal
        drive_db: Drive amount in dB
        
    Returns:
        Audio with Alpha Juno style distortion
    """
    if len(audio) == 0:
        return audio
        
    # Convert dB to linear gain
    drive_linear = 10 ** (drive_db / 20)
    
    # Alpha Juno characteristic: soft saturation with slight asymmetry
    driven = audio * drive_linear
    
    # Asymmetric saturation (characteristic of analog circuits)
    positive = np.where(driven >= 0, np.tanh(driven * 1.2), 0)
    negative = np.where(driven < 0, np.tanh(driven * 0.8), 0) 
    
    saturated = positive + negative
    
    # Slight high-frequency emphasis (Alpha Juno filter characteristics)
    # This gives the "edge" to hoover sounds
    return saturated * 0.7  # Level adjustment


def industrial_distortion(audio: np.ndarray,
                         metallic_factor: float = 0.6,
                         sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply industrial-style distortion for Berlin hardcore.
    
    Creates harsh, metallic distortion characteristic of industrial hardcore.
    Less aggressive than gabber distortion but more textured.
    
    Args:
        audio: Input audio signal
        metallic_factor: Amount of metallic character (0.0-1.0)
        sample_rate: Audio sample rate
        
    Returns:
        Audio with industrial distortion applied
    """
    if len(audio) == 0:
        return audio
        
    # Industrial distortion: combination of bit reduction and frequency shaping
    
    # Slight bit reduction for digital harshness
    if metallic_factor > 0.3:
        bit_depth = max(8, 16 - int(metallic_factor * 8))
        scale = 2 ** (bit_depth - 1)
        crushed = np.round(audio * scale) / scale
    else:
        crushed = audio
    
    # Metallic frequency emphasis (mid-high frequencies)
    b, a = signal.butter(2, [800, 4000], btype='band', fs=sample_rate)
    metallic = signal.filtfilt(b, a, crushed)
    
    # Mix original with metallic enhancement
    return crushed + metallic * metallic_factor * 0.3


def gabber_distortion(audio: np.ndarray,
                     intensity: float = 0.8) -> np.ndarray:
    """
    Apply aggressive gabber distortion.
    
    Maximum aggression distortion for gabber/hardcore. 
    Combines multiple distortion techniques for maximum impact.
    
    Args:
        audio: Input audio signal
        intensity: Distortion intensity (0.0-1.0)
        
    Returns:
        Audio with gabber distortion applied
    """
    if len(audio) == 0:
        return audio
        
    # Gabber distortion: extreme but musical
    drive = 3.0 + intensity * 5.0  # Up to 8x drive
    
    # Multi-stage distortion
    stage1 = np.tanh(audio * drive * 0.5)              # Soft saturation
    stage2 = np.clip(stage1 * 2.0, -0.8, 0.8)         # Hard clipping
    stage3 = np.tanh(stage2 * 1.5)                     # Final saturation
    
    # Mix stages for complex harmonic content
    result = stage1 * 0.3 + stage2 * 0.4 + stage3 * 0.3
    
    return result * 0.6  # Level adjustment


def soft_compression_overdrive(audio: np.ndarray,
                              threshold: float = 0.5,
                              ratio: float = 4.0,
                              makeup_gain: float = 1.2) -> np.ndarray:
    """
    Apply soft compression with overdrive character.
    
    Combination compressor/overdrive similar to analog channel strips.
    Adds punch while controlling dynamics.
    
    Args:
        audio: Input audio signal
        threshold: Compression threshold (0.0-1.0)
        ratio: Compression ratio (1.0-20.0)
        makeup_gain: Makeup gain multiplier
        
    Returns:
        Audio with soft compression overdrive applied
    """
    if len(audio) == 0:
        return audio
        
    # Soft knee compression
    abs_audio = np.abs(audio)
    
    # Compression curve
    compressed_level = np.where(
        abs_audio <= threshold,
        abs_audio,
        threshold + (abs_audio - threshold) / ratio
    )
    
    # Apply compression while preserving sign
    compressed = np.sign(audio) * compressed_level
    
    # Subtle saturation from compression circuit
    saturated = np.tanh(compressed * 1.2) * makeup_gain
    
    return saturated


# Preset functions for easy use
def apply_rotterdam_gabber_distortion(audio: np.ndarray, sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply authentic Rotterdam gabber distortion preset"""
    return rotterdam_doorlussen(audio, stages=3, drive_per_stage=2.5, sample_rate=sample_rate)


def apply_industrial_hardcore_distortion(audio: np.ndarray, sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply industrial hardcore distortion preset"""
    return industrial_distortion(audio, metallic_factor=0.6, sample_rate=sample_rate)


def apply_classic_hoover_distortion(audio: np.ndarray) -> np.ndarray:
    """Apply classic Alpha Juno hoover distortion preset"""
    return alpha_juno_distortion(audio, drive_db=15)


def apply_extreme_gabber_distortion(audio: np.ndarray) -> np.ndarray:
    """Apply extreme gabber distortion preset"""
    return gabber_distortion(audio, intensity=0.9)