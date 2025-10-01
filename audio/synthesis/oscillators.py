#!/usr/bin/env python3
"""
Oscillator Synthesis - Extracted from Legacy Engines

Core synthesis functions for hardcore kick generation and oscillator banks.
Extracted from frankenstein_engine.py and made properly modular.
"""

import numpy as np
from typing import Optional
from ..parameters.synthesis_constants import SynthesisParams, HardcoreConstants


def synthesize_oscillator_bank(params: SynthesisParams, 
                              sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Generate raw oscillator with multiple waveforms for hardcore kick synthesis.
    
    Extracted from frankenstein_engine.py - TR-909 style multi-oscillator bank
    with frequency sweeping and harmonic content based on brutality settings.
    
    Args:
        params: Synthesis parameters including frequency, duration, brutality
        sample_rate: Audio sample rate
        
    Returns:
        Raw oscillator audio signal
    """
    if params.duration_ms <= 0:
        return np.array([])
        
    samples = int(params.duration_ms * sample_rate / 1000)
    t = np.linspace(0, params.duration_ms/1000, samples)
    
    # Create pitch envelope (909-style frequency sweep)
    start_freq = params.frequency
    end_freq = params.frequency * 0.4  # ~60% down
    pitch_envelope = start_freq * np.exp(-5 * t) + end_freq
    
    # Phase accumulator for accurate FM synthesis
    phase = np.zeros_like(t)
    for i in range(1, len(t)):
        phase[i] = phase[i-1] + 2 * np.pi * pitch_envelope[i-1] / sample_rate
        
    # Multi-oscillator bank
    osc_bank = np.zeros_like(t)
    
    # 1. Fundamental sine (TR-909 style)
    osc_bank += np.sin(phase) * 0.8
    
    # 2. Sub-harmonic for body (analog modeling)
    osc_bank += np.sin(phase * 0.5) * 0.3
    
    # 3. Even harmonics for punch
    osc_bank += np.sin(phase * 2) * 0.4
    osc_bank += np.sin(phase * 4) * 0.2
    
    # 4. Odd harmonics for aggression (if brutal)
    if params.brutality > 0.5:
        osc_bank += np.sin(phase * 3) * 0.3 * params.brutality
        osc_bank += np.sin(phase * 5) * 0.15 * params.brutality
    
    # 5. High frequency component for crunch
    if params.crunch_factor > 0:
        noise_mod = np.random.normal(0, 0.1, samples) * params.crunch_factor
        osc_bank += np.sin(phase * 8 + noise_mod) * 0.1 * params.crunch_factor
        
    return osc_bank


def create_brutalist_envelope(params: SynthesisParams,
                             sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Create envelope with different attack characteristics per hardcore style.
    
    Extracted from frankenstein_engine.py - style-dependent envelope shaping
    for different hardcore subgenres.
    
    Args:
        params: Synthesis parameters including duration and style characteristics
        sample_rate: Audio sample rate
        
    Returns:
        Envelope signal for amplitude modulation
    """
    if params.duration_ms <= 0:
        return np.array([])
        
    samples = int(params.duration_ms * sample_rate / 1000)
    t = np.linspace(0, params.duration_ms/1000, samples)
    
    # Base exponential decay (TR-909 style)
    base_envelope = np.exp(-8 * t)
    
    # Style-dependent modifications
    envelope = base_envelope.copy()
    
    # Brutality affects decay curve
    if params.brutality > 0.7:
        # More aggressive decay for brutal styles
        envelope *= np.exp(-2 * t * params.brutality)
    
    # Analog warmth affects envelope shape
    if params.analog_warmth > 0.5:
        # Softer attack for analog character
        attack_samples = int(0.001 * sample_rate)  # 1ms attack
        if attack_samples < len(envelope):
            attack_curve = np.linspace(0, 1, attack_samples)
            envelope[:attack_samples] *= attack_curve
    
    # Crunch factor adds envelope modulation
    if params.crunch_factor > 0.3:
        # Add slight envelope wobble for digital character
        wobble = 1.0 + 0.05 * params.crunch_factor * np.sin(2 * np.pi * 50 * t)
        envelope *= wobble
    
    return envelope


def synthesize_simple_kick(frequency: float = 60.0,
                          duration_ms: int = 400,
                          sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """
    Simple kick synthesizer for basic usage.
    
    Wrapper around oscillator_bank with sensible defaults for quick kick generation.
    
    Args:
        frequency: Base frequency in Hz
        duration_ms: Duration in milliseconds
        sample_rate: Audio sample rate
        
    Returns:
        Synthesized kick drum audio
    """
    params = SynthesisParams(
        frequency=frequency,
        duration_ms=duration_ms,
        brutality=0.7,
        crunch_factor=0.5,
        analog_warmth=0.6
    )
    
    oscillator = synthesize_oscillator_bank(params, sample_rate)
    envelope = create_brutalist_envelope(params, sample_rate)
    
    return oscillator * envelope


# Preset functions for easy use
def gabber_oscillator_bank(frequency: float = 55.0,
                          duration_ms: int = 600,
                          sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Generate Rotterdam gabber style oscillator bank"""
    params = SynthesisParams(
        frequency=frequency,
        duration_ms=duration_ms,
        brutality=0.7,
        crunch_factor=0.4,
        analog_warmth=0.8
    )
    
    oscillator = synthesize_oscillator_bank(params, sample_rate)
    envelope = create_brutalist_envelope(params, sample_rate)
    
    return oscillator * envelope


def industrial_oscillator_bank(frequency: float = 45.0,
                              duration_ms: int = 800,
                              sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Generate Berlin industrial style oscillator bank"""
    params = SynthesisParams(
        frequency=frequency,
        duration_ms=duration_ms,
        brutality=0.6,
        crunch_factor=0.3,
        analog_warmth=0.9
    )
    
    oscillator = synthesize_oscillator_bank(params, sample_rate)
    envelope = create_brutalist_envelope(params, sample_rate)
    
    return oscillator * envelope


def terrorcore_oscillator_bank(frequency: float = 90.0,
                              duration_ms: int = 150,
                              sample_rate: int = HardcoreConstants.SAMPLE_RATE_44K) -> np.ndarray:
    """Generate extreme terrorcore style oscillator bank"""
    params = SynthesisParams(
        frequency=frequency,
        duration_ms=duration_ms,
        brutality=1.0,
        crunch_factor=1.0,
        analog_warmth=0.1
    )
    
    oscillator = synthesize_oscillator_bank(params, sample_rate)
    envelope = create_brutalist_envelope(params, sample_rate)
    
    return oscillator * envelope