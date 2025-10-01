#!/usr/bin/env python3
"""
Dynamics Processing - Extracted from Legacy Engines

Compression, limiting, and dynamics processing functions for hardcore music production.
All functions extracted from frankenstein_engine.py and made properly modular.
"""

import numpy as np
from typing import Optional
from ..parameters.synthesis_constants import SynthesisParams, HardcoreConstants


def apply_compression(audio: np.ndarray,
                     ratio: float = 4.0,
                     threshold_db: float = -12.0,
                     attack_ms: float = 5.0,
                     release_ms: float = 100.0,
                     sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply compression to audio signal.
    
    Extracted from frankenstein_engine.py _compress() function.
    Simple but effective compressor with attack/release envelope following.
    
    Args:
        audio: Input audio signal
        ratio: Compression ratio (1.0-20.0)
        threshold_db: Threshold in dB
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        sample_rate: Audio sample rate
        
    Returns:
        Compressed audio signal
    """
    if len(audio) == 0:
        return audio
        
    # Convert threshold to linear
    threshold_lin = 10 ** (threshold_db / 20)
    
    # Envelope follower
    envelope = np.abs(audio)
    
    # Attack/release smoothing coefficients
    attack_coeff = np.exp(-1 / (attack_ms * sample_rate / 1000))
    release_coeff = np.exp(-1 / (release_ms * sample_rate / 1000))
    
    # Smooth envelope following
    smoothed = np.zeros_like(envelope)
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i-1]:
            # Attack
            smoothed[i] = envelope[i] * (1 - attack_coeff) + smoothed[i-1] * attack_coeff
        else:
            # Release
            smoothed[i] = envelope[i] * (1 - release_coeff) + smoothed[i-1] * release_coeff
    
    # Compression curve
    gain_reduction = np.ones_like(smoothed)
    over_threshold = smoothed > threshold_lin
    
    if np.any(over_threshold):
        # Calculate gain reduction for samples over threshold
        gain_reduction[over_threshold] = (
            threshold_lin + (smoothed[over_threshold] - threshold_lin) / ratio
        ) / smoothed[over_threshold]
    
    return audio * gain_reduction


def apply_serial_compression(audio: np.ndarray, 
                           params: SynthesisParams,
                           sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply hardcore serial compression stages.
    
    Extracted from frankenstein_engine.py apply_compression_madness().
    Three-stage compression chain for hardcore music dynamics.
    
    Args:
        audio: Input audio signal
        params: Synthesis parameters
        sample_rate: Audio sample rate
        
    Returns:
        Audio with serial compression applied
    """
    if len(audio) == 0:
        return audio
        
    processed = audio.copy()
    
    # Stage 1: Fast attack compressor (transient control)
    processed = apply_compression(processed, ratio=4.0, threshold_db=-18, 
                                attack_ms=0.1, release_ms=50, sample_rate=sample_rate)
    
    # Stage 2: Slower compressor (body control)  
    processed = apply_compression(processed, ratio=8.0, threshold_db=-12,
                                attack_ms=5, release_ms=100, sample_rate=sample_rate)
    
    # Stage 3: Limiter (peak control)
    processed = apply_compression(processed, ratio=20.0, threshold_db=-6,
                                attack_ms=0.05, release_ms=20, sample_rate=sample_rate)
    
    return processed


def apply_parallel_compression(audio: np.ndarray,
                              parallel_amount: float = 0.6,
                              sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply parallel (NY style) compression.
    
    Extracted from frankenstein_engine.py apply_compression_madness().
    Parallel compression for punch and body while preserving transients.
    
    Args:
        audio: Input audio signal
        parallel_amount: Amount of parallel compression (0.0-1.0)
        sample_rate: Audio sample rate
        
    Returns:
        Audio with parallel compression applied
    """
    if len(audio) == 0 or parallel_amount <= 0:
        return audio
        
    # Heavy compression on parallel path
    parallel = apply_compression(audio, ratio=10.0, threshold_db=-20,
                               attack_ms=0.1, release_ms=30, sample_rate=sample_rate)
    
    # Mix with original
    return audio * (1 - parallel_amount) + parallel * parallel_amount


def apply_hardcore_compression_chain(audio: np.ndarray, 
                                   params: SynthesisParams,
                                   sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply complete hardcore compression processing chain.
    
    Combines serial and parallel compression based on synthesis parameters.
    
    Args:
        audio: Input audio signal
        params: Synthesis parameters
        sample_rate: Audio sample rate
        
    Returns:
        Audio with complete compression chain applied
    """
    if len(audio) == 0:
        return audio
        
    processed = audio.copy()
    
    # Serial compression stages if enabled
    if params.serial_compression:
        processed = apply_serial_compression(processed, params, sample_rate)
    
    # Parallel compression if specified
    if hasattr(params, 'parallel_compression') and params.parallel_compression > 0:
        processed = apply_parallel_compression(processed, params.parallel_compression, sample_rate)
    
    return processed


def apply_hardcore_limiter(audio: np.ndarray,
                          threshold_db: float = HardcoreConstants.MASTER_LIMITER_THRESHOLD,
                          sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Apply hardcore-style brick wall limiter.
    
    Final limiting stage for hardcore music production.
    
    Args:
        audio: Input audio signal
        threshold_db: Limiting threshold in dB
        sample_rate: Audio sample rate
        
    Returns:
        Limited audio signal
    """
    if len(audio) == 0:
        return audio
        
    # Convert threshold to linear
    threshold_lin = 10 ** (threshold_db / 20)
    
    # Brick wall limiting with slight lookahead
    lookahead_samples = int(0.001 * sample_rate)  # 1ms lookahead
    
    if lookahead_samples >= len(audio):
        lookahead_samples = 0
    
    if lookahead_samples > 0:
        # Simple lookahead limiting
        peak_envelope = np.maximum.accumulate(np.abs(audio))
        # Prevent division by zero
        peak_envelope = np.maximum(peak_envelope, 1e-8)
        gain_reduction = np.minimum(1.0, threshold_lin / peak_envelope)
        
        # Apply lookahead delay
        limited = np.roll(audio, lookahead_samples) * gain_reduction
        limited[:lookahead_samples] = 0  # Clear delay artifacts
    else:
        # Simple hard limiting
        limited = np.clip(audio, -threshold_lin, threshold_lin)
    
    return limited


# Preset functions for easy use
def apply_gabber_compression(audio: np.ndarray, sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply gabber-style compression preset"""
    # Heavy compression with parallel processing
    serial = apply_compression(audio, ratio=8.0, threshold_db=-10, 
                             attack_ms=0.1, release_ms=50, sample_rate=sample_rate)
    return apply_parallel_compression(serial, parallel_amount=0.8, sample_rate=sample_rate)


def apply_industrial_compression(audio: np.ndarray, sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply industrial-style compression preset"""
    # Gentler compression for industrial character
    return apply_compression(audio, ratio=4.0, threshold_db=-15,
                           attack_ms=5.0, release_ms=100, sample_rate=sample_rate)


def apply_terrorcore_compression(audio: np.ndarray, sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Apply extreme terrorcore compression preset"""
    # Extreme compression for maximum aggression
    params = SynthesisParams(serial_compression=True, parallel_compression=0.7)
    return apply_hardcore_compression_chain(audio, params, sample_rate)