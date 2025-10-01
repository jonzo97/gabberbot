#!/usr/bin/env python3
"""
Shared Audio Utilities for Hardcore Music Production

Common audio processing functions used by both backends.
"""

import numpy as np
import wave
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import time

def db_to_linear(db: float) -> float:
    """Convert decibels to linear scale"""
    return 10.0 ** (db / 20.0)

def linear_to_db(linear: float) -> float:
    """Convert linear scale to decibels"""
    return 20.0 * np.log10(max(linear, 1e-10))

def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS (Root Mean Square) of audio signal"""
    return np.sqrt(np.mean(audio ** 2))

def calculate_peak(audio: np.ndarray) -> float:
    """Calculate peak amplitude of audio signal"""
    return np.max(np.abs(audio))

def calculate_lufs(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Calculate LUFS (Loudness Units relative to Full Scale)
    Simplified implementation for hardcore music
    """
    # Pre-filter (high-pass at ~38Hz)
    if len(audio) > 1000:
        # Simple high-pass filter approximation
        filtered = audio - np.convolve(audio, np.ones(100)/100, mode='same')
        
        # K-weighting approximation (boosts ~1-5kHz)
        # For hardcore music, we emphasize the kick range
        rms = calculate_rms(filtered)
        lufs = -0.691 + 10 * np.log10(rms ** 2 + 1e-10)
        return lufs
    
    return -23.0  # Default quiet level

def analyze_frequency_bands(audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, float]:
    """
    Analyze energy in hardcore-relevant frequency bands
    
    Returns:
        Dictionary with band energies
    """
    if len(audio) < 512:
        return {}
    
    # Perform FFT
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
    magnitudes = np.abs(fft)
    
    # Define hardcore-relevant frequency bands
    bands = {
        "sub_bass": (20, 60),      # Kick fundamentals
        "bass": (60, 200),         # Kick body
        "low_mid": (200, 500),     # Kick attack
        "mid": (500, 2000),        # Synth fundamentals
        "high_mid": (2000, 6000),  # Presence
        "high": (6000, 12000),     # Brilliance
        "ultra_high": (12000, 22000) # Air/crunch
    }
    
    band_energies = {}
    nyquist = sample_rate / 2
    
    for band_name, (low_freq, high_freq) in bands.items():
        # Find frequency bins for this band
        low_bin = int(low_freq * len(freqs) / sample_rate)
        high_bin = int(min(high_freq, nyquist) * len(freqs) / sample_rate)
        
        if high_bin > low_bin:
            band_energy = np.sum(magnitudes[low_bin:high_bin] ** 2)
            total_energy = np.sum(magnitudes[1:len(magnitudes)//2] ** 2)
            band_energies[band_name] = band_energy / (total_energy + 1e-10)
        else:
            band_energies[band_name] = 0.0
    
    return band_energies

def detect_kick_drum(audio: np.ndarray, sample_rate: int = 44100, 
                    threshold: float = 0.3) -> Dict[str, Any]:
    """
    Detect kick drum characteristics in audio
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        threshold: Detection threshold
        
    Returns:
        Dictionary with kick analysis
    """
    analysis = {
        "detected": False,
        "fundamental_freq": 0.0,
        "punch_factor": 0.0,
        "weight_factor": 0.0,
        "confidence": 0.0
    }
    
    if len(audio) < 1024:
        return analysis
    
    # Analyze frequency content
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
    magnitudes = np.abs(fft)
    
    # Look for kick fundamental in 40-100Hz range
    kick_range_mask = (freqs >= 40) & (freqs <= 100)
    if np.any(kick_range_mask):
        kick_magnitudes = magnitudes[kick_range_mask]
        kick_freqs = freqs[kick_range_mask]
        
        if len(kick_magnitudes) > 0:
            fundamental_idx = np.argmax(kick_magnitudes)
            fundamental_freq = kick_freqs[fundamental_idx]
            fundamental_magnitude = kick_magnitudes[fundamental_idx]
            
            # Check if this is strong enough to be a kick
            total_energy = np.sum(magnitudes[1:len(magnitudes)//2] ** 2)
            kick_energy = fundamental_magnitude ** 2
            
            if kick_energy / (total_energy + 1e-10) > threshold:
                analysis["detected"] = True
                analysis["fundamental_freq"] = float(fundamental_freq)
                
                # Calculate punch factor (attack vs body)
                band_energies = analyze_frequency_bands(audio, sample_rate)
                punch_factor = band_energies.get("low_mid", 0) / (band_energies.get("bass", 0) + 1e-10)
                analysis["punch_factor"] = float(punch_factor)
                
                # Calculate weight factor (low end dominance)
                weight_factor = (band_energies.get("sub_bass", 0) + band_energies.get("bass", 0))
                analysis["weight_factor"] = float(weight_factor)
                
                # Confidence based on energy and frequency stability
                analysis["confidence"] = min(1.0, kick_energy / (total_energy + 1e-10) * 10)
    
    return analysis

def apply_hardcore_processing(audio: np.ndarray, 
                            crunch: float = 0.5, 
                            drive: float = 1.0,
                            doorlussen: float = 0.0) -> np.ndarray:
    """
    Apply hardcore-style audio processing
    
    Args:
        audio: Input audio
        crunch: Crunch factor (0-1)
        drive: Overdrive amount (1-10)
        doorlussen: Serial distortion intensity (0-1)
        
    Returns:
        Processed audio
    """
    processed = audio.copy()
    
    # Apply overdrive
    if drive > 1.0:
        processed = processed * drive
        processed = np.tanh(processed)
        processed = processed / np.max(np.abs(processed) + 1e-10) * 0.9
    
    # Apply crunch (bit crushing and saturation)
    if crunch > 0.0:
        # Bit crushing
        if crunch > 0.3:
            bits = max(4, int(16 - crunch * 8))
            processed = np.round(processed * (2**bits - 1)) / (2**bits - 1)
        
        # Saturation
        processed = processed * (1 + crunch * 2)
        processed = np.tanh(processed * 0.8)
    
    # Apply doorlussen (serial distortion chain)
    if doorlussen > 0.0:
        # First distortion stage
        processed = processed * (1 + doorlussen)
        processed = np.clip(processed, -0.9, 0.9)
        
        # EQ simulation (boost mids)
        if len(processed) > 100:
            # Simple high-pass filter effect
            processed = processed - np.convolve(processed, np.ones(20)/20, mode='same')
        
        # Second distortion stage
        processed = processed * (1 + doorlussen * 0.5)
        processed = np.tanh(processed)
    
    # Final limiting
    peak = np.max(np.abs(processed))
    if peak > 0.95:
        processed = processed / peak * 0.95
    
    return processed

def generate_kick_envelope(duration: float, sample_rate: int = 44100,
                          attack: float = 0.001, decay: float = 0.1,
                          sustain: float = 0.3, release: float = 0.2) -> np.ndarray:
    """Generate ADSR envelope for kick drums"""
    samples = int(duration * sample_rate)
    envelope = np.zeros(samples)
    
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    
    # Ensure we don't exceed total samples
    attack_samples = min(attack_samples, samples // 4)
    decay_samples = min(decay_samples, samples // 4)
    release_samples = min(release_samples, samples // 2)
    
    sustain_samples = samples - attack_samples - decay_samples - release_samples
    sustain_samples = max(0, sustain_samples)
    
    idx = 0
    
    # Attack phase
    if attack_samples > 0:
        envelope[idx:idx + attack_samples] = np.linspace(0, 1, attack_samples)
        idx += attack_samples
    
    # Decay phase
    if decay_samples > 0:
        envelope[idx:idx + decay_samples] = np.linspace(1, sustain, decay_samples)
        idx += decay_samples
    
    # Sustain phase
    if sustain_samples > 0:
        envelope[idx:idx + sustain_samples] = sustain
        idx += sustain_samples
    
    # Release phase
    if release_samples > 0 and idx < samples:
        envelope[idx:idx + release_samples] = np.linspace(sustain, 0, release_samples)
    
    return envelope

def save_audio_wav(audio: np.ndarray, filename: str, sample_rate: int = 44100):
    """Save audio to WAV file"""
    # Ensure audio is in the correct format
    if audio.dtype != np.int16:
        # Convert to 16-bit integers
        audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    else:
        audio_int = audio
    
    # Handle stereo/mono
    if len(audio_int.shape) == 1:
        channels = 1
    else:
        channels = audio_int.shape[1]
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())

def load_audio_wav(filename: str) -> Tuple[np.ndarray, int]:
    """Load audio from WAV file"""
    with wave.open(filename, 'r') as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(-1)
        audio_int = np.frombuffer(frames, dtype=np.int16)
        
        # Convert to float
        audio = audio_int.astype(np.float32) / 32768.0
        
        # Handle stereo
        if wav_file.getnchannels() == 2:
            audio = audio.reshape(-1, 2)
    
    return audio, sample_rate

def calculate_spectral_features(audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, float]:
    """Calculate spectral features for audio analysis"""
    if len(audio) < 512:
        return {}
    
    # FFT analysis
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
    magnitudes = np.abs(fft[:len(fft)//2])
    freqs = freqs[:len(freqs)//2]
    
    features = {}
    
    # Spectral centroid (brightness)
    if np.sum(magnitudes) > 0:
        features["spectral_centroid"] = np.sum(freqs * magnitudes) / np.sum(magnitudes)
    else:
        features["spectral_centroid"] = 0.0
    
    # Spectral rolloff (85% energy point)
    cumulative_energy = np.cumsum(magnitudes ** 2)
    total_energy = cumulative_energy[-1]
    if total_energy > 0:
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        if len(rolloff_idx) > 0:
            features["spectral_rolloff"] = freqs[rolloff_idx[0]]
        else:
            features["spectral_rolloff"] = freqs[-1]
    else:
        features["spectral_rolloff"] = 0.0
    
    # Spectral spread (width)
    if "spectral_centroid" in features and np.sum(magnitudes) > 0:
        centroid = features["spectral_centroid"]
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / np.sum(magnitudes))
        features["spectral_spread"] = spread
    else:
        features["spectral_spread"] = 0.0
    
    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
    features["zero_crossing_rate"] = zero_crossings / (2 * len(audio))
    
    return features

def estimate_bpm(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Estimate BPM from audio using onset detection
    Simplified implementation for hardcore music (150-250 BPM range)
    """
    if len(audio) < sample_rate:  # Need at least 1 second
        return 170.0  # Default hardcore BPM
    
    # Simple onset detection using spectral flux
    hop_size = 512
    flux = []
    
    for i in range(0, len(audio) - hop_size, hop_size):
        frame = audio[i:i + hop_size]
        spectrum = np.abs(np.fft.fft(frame))
        
        if len(flux) > 0:
            # Calculate spectral flux (difference between consecutive frames)
            diff = spectrum - prev_spectrum
            flux_val = np.sum(np.maximum(0, diff))
            flux.append(flux_val)
        
        prev_spectrum = spectrum
    
    if len(flux) < 10:
        return 170.0
    
    # Find peaks in flux (potential beat locations)
    flux = np.array(flux)
    threshold = np.mean(flux) + np.std(flux)
    peaks = []
    
    for i in range(1, len(flux) - 1):
        if flux[i] > flux[i-1] and flux[i] > flux[i+1] and flux[i] > threshold:
            peaks.append(i)
    
    if len(peaks) < 2:
        return 170.0
    
    # Calculate intervals between peaks
    intervals = np.diff(peaks) * hop_size / sample_rate  # Convert to seconds
    
    # Filter for reasonable beat intervals (hardcore range: 150-250 BPM)
    min_interval = 60.0 / 250.0  # Fastest BPM
    max_interval = 60.0 / 150.0  # Slowest BPM
    
    valid_intervals = intervals[(intervals >= min_interval) & (intervals <= max_interval)]
    
    if len(valid_intervals) > 0:
        avg_interval = np.median(valid_intervals)
        bpm = 60.0 / avg_interval
        return float(np.clip(bpm, 150, 250))
    
    return 170.0

def create_hardcore_test_signals(sample_rate: int = 44100) -> Dict[str, np.ndarray]:
    """Create test signals for hardcore music analysis"""
    duration = 2.0
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    signals = {}
    
    # Gabber kick with frequency sweep
    kick_freq = 60 * np.exp(-t * 8)
    kick_phase = 2 * np.pi * np.cumsum(kick_freq) / sample_rate
    kick_env = np.exp(-t * 12)
    signals["gabber_kick"] = kick_env * np.sin(kick_phase) * 0.8
    
    # Industrial rumble kick
    rumble_freq = 45
    rumble = np.sin(2 * np.pi * rumble_freq * t) * 0.6
    rumble += np.sin(2 * np.pi * rumble_freq * 0.5 * t) * 0.4  # Sub component
    rumble_env = np.exp(-t * 6)
    signals["industrial_kick"] = rumble * rumble_env * 0.9
    
    # Acid bassline
    bass_freq = 110
    sawtooth = 2 * (bass_freq * t - np.floor(bass_freq * t + 0.5))
    # Add filter sweep
    cutoff_env = 1200 * (1 + np.sin(2 * np.pi * 0.5 * t))
    # Simple lowpass approximation
    bass = sawtooth * 0.6
    signals["acid_bass"] = bass
    
    # Hardcore stab
    stab_freqs = [220, 220 * 1.2, 220 * 1.5, 220 * 1.78]  # Minor chord with dissonance
    stab = sum(np.sin(2 * np.pi * freq * t) for freq in stab_freqs) / len(stab_freqs)
    stab_env = np.exp(-t * 20)  # Very fast decay
    signals["hardcore_stab"] = stab * stab_env * 0.7
    
    # White noise (for testing)
    signals["white_noise"] = np.random.normal(0, 0.1, samples)
    
    # Apply hardcore processing to some signals
    signals["crunchy_kick"] = apply_hardcore_processing(
        signals["gabber_kick"], crunch=0.8, drive=2.5, doorlussen=0.6
    )
    
    return signals