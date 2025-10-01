#!/usr/bin/env python3
"""
Local Audio Analyzer - Numpy-Only Implementation
Self-contained audio analysis without librosa or external dependencies
Provides core functionality for hardcore music production
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..models.hardcore_models import AudioAnalysisResult

class AnalysisMode(Enum):
    REALTIME = "realtime"
    BATCH = "batch"
    KICK_FOCUS = "kick_focus"

@dataclass
class SpectralFeatures:
    centroid: float
    rolloff: float
    flux: float
    bandwidth: float
    contrast: float

@dataclass
class KickAnalysis:
    attack_time: float  # Time to peak in ms
    sustain_level: float  # Sustain level 0-1
    decay_rate: float  # Decay slope
    fundamental_freq: float  # Main frequency Hz
    harmonic_ratio: float  # Harmonic content ratio
    punch_factor: float  # Transient strength 0-1
    rumble_factor: float  # Low frequency content 0-1

class LocalAudioAnalyzer:
    """Self-contained audio analyzer using only numpy"""
    
    def __init__(self, sample_rate: int = 44100, frame_size: int = 1024):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = frame_size // 2
        
        # Analysis state
        self.is_analyzing = False
        self.analysis_history: List[AudioAnalysisResult] = []
        self.spectral_history: List[np.ndarray] = []
        
        # Frequency bins for analysis
        self.freq_bins = np.fft.fftfreq(frame_size, 1/sample_rate)
        self.freq_bins_pos = self.freq_bins[:frame_size//2]
        
        # Initialize analysis windows and filters
        self._initialize_analysis_tools()
    
    def _initialize_analysis_tools(self):
        """Initialize windows and filters for analysis"""
        # Hann window for spectral analysis
        self.window = np.hanning(self.frame_size)
        
        # Frequency band definitions for electronic music
        self.freq_bands = {
            'sub_bass': (20, 60),      # Sub bass / kick fundamentals
            'bass': (60, 250),         # Bass range
            'low_mid': (250, 500),     # Low mids
            'mid': (500, 2000),        # Mids
            'high_mid': (2000, 4000),  # High mids
            'high': (4000, 8000),      # Highs
            'air': (8000, 20000)       # Air band
        }
        
        # Create mel-like filter bank for spectral analysis
        self.mel_filters = self._create_mel_filterbank(n_filters=13)
        
        # Kick detection templates (simple energy patterns)
        self.kick_templates = self._create_kick_templates()
    
    def _create_mel_filterbank(self, n_filters: int) -> np.ndarray:
        """Create mel-scale filter bank"""
        # Simplified mel scale implementation
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel-spaced frequency points
        mel_min = hz_to_mel(80)  # Start from 80Hz (good for electronic music)
        mel_max = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to bin indices
        bin_points = np.floor((self.frame_size + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create triangular filters
        filters = np.zeros((n_filters, self.frame_size // 2))
        for i in range(n_filters):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            if right < len(filters[i]):
                # Rising slope
                filters[i, left:center] = np.linspace(0, 1, center - left)
                # Falling slope
                filters[i, center:right] = np.linspace(1, 0, right - center)
        
        return filters
    
    def _create_kick_templates(self) -> Dict[str, np.ndarray]:
        """Create kick drum detection templates"""
        templates = {}
        
        # Gabber kick template (sharp attack, quick decay)
        t = np.linspace(0, 0.5, int(0.5 * self.sample_rate))
        gabber_env = np.exp(-t * 15) * (1 - np.exp(-t * 50))
        templates['gabber'] = gabber_env
        
        # Industrial kick template (longer decay, rumble)
        industrial_env = np.exp(-t * 8) * (1 - np.exp(-t * 20))
        templates['industrial'] = industrial_env
        
        # Hardcore kick template (very sharp attack)
        hardcore_env = np.exp(-t * 25) * (1 - np.exp(-t * 100))
        templates['hardcore'] = hardcore_env
        
        return templates
    
    async def analyze_audio(self, audio: np.ndarray, mode: AnalysisMode = AnalysisMode.BATCH) -> AudioAnalysisResult:
        """Analyze audio data and return comprehensive results"""
        
        if len(audio) == 0:
            return self._empty_analysis_result()
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize audio
        audio = self._normalize_audio(audio)
        
        # Compute basic metrics
        peak_level = np.max(np.abs(audio))
        rms_level = np.sqrt(np.mean(audio**2))
        
        # Compute spectral features
        spectrum = self._compute_spectrum(audio)
        spectral_features = self._analyze_spectrum(spectrum)
        
        # Frequency band analysis
        band_energies = self._analyze_frequency_bands(spectrum)
        
        # Create frequency spectrum for visualization (8 bands)
        freq_spectrum = self._create_frequency_spectrum_bands(band_energies)
        
        # Zero crossing rate
        zero_crossings = self._compute_zero_crossing_rate(audio)
        
        return AudioAnalysisResult(
            peak=float(peak_level),
            rms=float(rms_level),
            spectral_centroid=float(spectral_features.centroid)
        )
    
    async def analyze_kick_dna(self, audio: np.ndarray) -> KickAnalysis:
        """Analyze kick drum characteristics in detail"""
        
        if len(audio) == 0:
            return self._empty_kick_analysis()
        
        # Ensure mono and normalize
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = self._normalize_audio(audio)
        
        # Find attack time (time to peak)
        peak_idx = np.argmax(np.abs(audio))
        attack_time = (peak_idx / self.sample_rate) * 1000  # Convert to ms
        
        # Analyze envelope
        envelope = self._compute_envelope(audio)
        
        # Find sustain level and decay rate
        sustain_level = self._find_sustain_level(envelope)
        decay_rate = self._compute_decay_rate(envelope, peak_idx)
        
        # Spectral analysis for frequency content
        spectrum = self._compute_spectrum(audio)
        fundamental_freq = self._find_fundamental_frequency(spectrum)
        harmonic_ratio = self._compute_harmonic_ratio(spectrum, fundamental_freq)
        
        # Punch factor (transient strength)
        punch_factor = self._compute_punch_factor(audio)
        
        # Rumble factor (low frequency content)
        rumble_factor = self._compute_rumble_factor(spectrum)
        
        return KickAnalysis(
            attack_time=float(attack_time),
            sustain_level=float(sustain_level),
            decay_rate=float(decay_rate),
            fundamental_freq=float(fundamental_freq),
            harmonic_ratio=float(harmonic_ratio),
            punch_factor=float(punch_factor),
            rumble_factor=float(rumble_factor)
        )
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _compute_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """Compute frequency spectrum using FFT"""
        # Pad or truncate to frame size
        if len(audio) > self.frame_size:
            audio = audio[:self.frame_size]
        elif len(audio) < self.frame_size:
            audio = np.pad(audio, (0, self.frame_size - len(audio)))
        
        # Apply window and compute FFT
        windowed = audio * self.window
        fft = np.fft.fft(windowed)
        magnitude = np.abs(fft[:self.frame_size//2])
        
        return magnitude
    
    def _analyze_spectrum(self, spectrum: np.ndarray) -> SpectralFeatures:
        """Analyze spectral features"""
        # Spectral centroid
        freqs = self.freq_bins_pos
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)
        
        # Spectral rolloff (95% of energy)
        cumsum = np.cumsum(spectrum)
        total_energy = cumsum[-1]
        rolloff_idx = np.where(cumsum >= 0.95 * total_energy)[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Spectral flux (change between frames)
        if len(self.spectral_history) > 0:
            prev_spectrum = self.spectral_history[-1]
            flux = np.sum((spectrum - prev_spectrum)**2)
        else:
            flux = 0.0
        
        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * spectrum) / (np.sum(spectrum) + 1e-8))
        
        # Simple spectral contrast
        contrast = np.std(spectrum) / (np.mean(spectrum) + 1e-8)
        
        # Store for next flux calculation
        self.spectral_history.append(spectrum.copy())
        if len(self.spectral_history) > 10:  # Keep only recent history
            self.spectral_history.pop(0)
        
        return SpectralFeatures(
            centroid=centroid,
            rolloff=rolloff,
            flux=flux,
            bandwidth=bandwidth,
            contrast=contrast
        )
    
    def _analyze_frequency_bands(self, spectrum: np.ndarray) -> Dict[str, float]:
        """Analyze energy in different frequency bands"""
        band_energies = {}
        
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find bin indices for this frequency range
            low_bin = int(low_freq * self.frame_size / self.sample_rate)
            high_bin = int(high_freq * self.frame_size / self.sample_rate)
            
            # Ensure bins are within spectrum range
            low_bin = max(0, min(low_bin, len(spectrum) - 1))
            high_bin = max(low_bin + 1, min(high_bin, len(spectrum)))
            
            # Compute energy in this band
            band_energy = np.sum(spectrum[low_bin:high_bin]**2)
            band_energies[band_name] = band_energy
        
        # Normalize by total energy
        total_energy = sum(band_energies.values()) + 1e-8
        for band_name in band_energies:
            band_energies[band_name] /= total_energy
        
        return band_energies
    
    def _create_frequency_spectrum_bands(self, band_energies: Dict[str, float]) -> List[float]:
        """Create 8-band frequency spectrum for visualization"""
        # Map our 7 bands to 8 visualization bands
        vis_bands = [
            band_energies['sub_bass'],
            band_energies['bass'],
            band_energies['low_mid'],
            band_energies['mid'] * 0.6,  # Split mid band
            band_energies['mid'] * 0.4,
            band_energies['high_mid'],
            band_energies['high'],
            band_energies['air']
        ]
        
        return vis_bands
    
    def _compute_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Compute zero crossing rate"""
        if len(audio) < 2:
            return 0.0
        
        # Find zero crossings
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        
        # Normalize by length
        zcr = zero_crossings / len(audio)
        
        return zcr
    
    def _compute_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Compute amplitude envelope"""
        # Simple envelope using absolute value and smoothing
        envelope = np.abs(audio)
        
        # Simple moving average for smoothing
        window_size = max(1, len(envelope) // 100)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            envelope = np.convolve(envelope, kernel, mode='same')
        
        return envelope
    
    def _find_sustain_level(self, envelope: np.ndarray) -> float:
        """Find sustain level of the envelope"""
        # Find peak
        peak_idx = np.argmax(envelope)
        
        # Look for sustain region after peak
        if peak_idx < len(envelope) - 10:
            sustain_region = envelope[peak_idx + 5:peak_idx + 15]
            sustain_level = np.mean(sustain_region) / envelope[peak_idx]
        else:
            sustain_level = 0.1  # Default for very short sounds
        
        return min(1.0, sustain_level)
    
    def _compute_decay_rate(self, envelope: np.ndarray, peak_idx: int) -> float:
        """Compute decay rate after peak"""
        if peak_idx >= len(envelope) - 2:
            return 1.0  # Very fast decay for short sounds
        
        # Analyze decay slope
        decay_region = envelope[peak_idx:min(peak_idx + 100, len(envelope))]
        
        if len(decay_region) < 2:
            return 1.0
        
        # Fit exponential decay
        t = np.arange(len(decay_region))
        log_envelope = np.log(decay_region + 1e-8)
        
        # Simple linear fit to log envelope
        if len(log_envelope) > 1:
            decay_slope = (log_envelope[-1] - log_envelope[0]) / len(log_envelope)
            decay_rate = abs(decay_slope)
        else:
            decay_rate = 1.0
        
        return min(10.0, decay_rate)  # Cap at reasonable value
    
    def _find_fundamental_frequency(self, spectrum: np.ndarray) -> float:
        """Find fundamental frequency from spectrum"""
        # Look for peak in low frequency range (typical for kicks)
        low_freq_bins = int(200 * self.frame_size / self.sample_rate)  # Up to 200Hz
        low_spectrum = spectrum[:low_freq_bins]
        
        if len(low_spectrum) == 0:
            return 60.0  # Default kick frequency
        
        # Find peak frequency
        peak_bin = np.argmax(low_spectrum)
        fundamental_freq = peak_bin * self.sample_rate / self.frame_size
        
        return max(30.0, min(200.0, fundamental_freq))  # Reasonable range for kicks
    
    def _compute_harmonic_ratio(self, spectrum: np.ndarray, fundamental_freq: float) -> float:
        """Compute ratio of harmonic to inharmonic content"""
        # Simple version: compare energy at harmonic frequencies vs total
        harmonics = [2, 3, 4, 5]  # First few harmonics
        harmonic_energy = 0.0
        
        for h in harmonics:
            harmonic_freq = fundamental_freq * h
            harmonic_bin = int(harmonic_freq * self.frame_size / self.sample_rate)
            
            if harmonic_bin < len(spectrum):
                # Sum energy around harmonic (±2 bins)
                start_bin = max(0, harmonic_bin - 2)
                end_bin = min(len(spectrum), harmonic_bin + 3)
                harmonic_energy += np.sum(spectrum[start_bin:end_bin])
        
        total_energy = np.sum(spectrum) + 1e-8
        harmonic_ratio = harmonic_energy / total_energy
        
        return min(1.0, harmonic_ratio)
    
    def _compute_punch_factor(self, audio: np.ndarray) -> float:
        """Compute punch factor (transient strength)"""
        # Look at initial transient vs sustained portion
        transient_length = min(len(audio) // 10, int(0.01 * self.sample_rate))  # 10ms max
        
        if transient_length < 2:
            return 0.5  # Default for very short sounds
        
        transient_energy = np.sum(audio[:transient_length]**2)
        total_energy = np.sum(audio**2) + 1e-8
        
        punch_factor = transient_energy / (total_energy / len(audio) * transient_length)
        
        return min(1.0, punch_factor)
    
    def _compute_rumble_factor(self, spectrum: np.ndarray) -> float:
        """Compute rumble factor (low frequency content)"""
        # Energy in sub-bass and bass regions
        sub_bass_bin = int(60 * self.frame_size / self.sample_rate)
        rumble_energy = np.sum(spectrum[:sub_bass_bin])
        
        total_energy = np.sum(spectrum) + 1e-8
        rumble_factor = rumble_energy / total_energy
        
        return min(1.0, rumble_factor)
    
    def _empty_analysis_result(self) -> AudioAnalysisResult:
        """Return empty analysis result for error cases"""
        return AudioAnalysisResult(
            peak=0.0,
            rms=0.0,
            spectral_centroid=0.0
        )
    
    def _empty_kick_analysis(self) -> KickAnalysis:
        """Return empty kick analysis for error cases"""
        return KickAnalysis(
            attack_time=0.0,
            sustain_level=0.0,
            decay_rate=1.0,
            fundamental_freq=60.0,
            harmonic_ratio=0.5,
            punch_factor=0.5,
            rumble_factor=0.5
        )
    
    async def analyze_pattern_groove(self, audio_segments: List[np.ndarray]) -> Dict[str, float]:
        """Analyze groove and timing characteristics across multiple segments"""
        if not audio_segments:
            return {"groove_consistency": 0.0, "timing_accuracy": 0.0}
        
        # Analyze each segment
        segment_features = []
        for segment in audio_segments:
            analysis = await self.analyze_audio(segment)
            segment_features.append({
                'peak': analysis.peak,
                'rms': analysis.rms,
                'centroid': analysis.spectral_centroid
            })
        
        # Compute groove metrics
        if len(segment_features) > 1:
            # Consistency across segments
            peak_var = np.var([f['peak'] for f in segment_features])
            rms_var = np.var([f['rms'] for f in segment_features])
            groove_consistency = 1.0 / (1.0 + peak_var + rms_var)
            
            # Timing accuracy (placeholder - would need onset detection)
            timing_accuracy = 0.8  # Assume good timing for now
        else:
            groove_consistency = 1.0
            timing_accuracy = 1.0
        
        return {
            "groove_consistency": float(groove_consistency),
            "timing_accuracy": float(timing_accuracy),
            "segment_count": len(segment_features)
        }
    
    def get_kick_template(self, kick_type: str) -> Optional[np.ndarray]:
        """Get kick template for matching"""
        return self.kick_templates.get(kick_type)
    
    def compare_with_template(self, audio: np.ndarray, template_name: str) -> float:
        """Compare audio with kick template"""
        template = self.get_kick_template(template_name)
        if template is None:
            return 0.0
        
        # Simple correlation-based matching
        if len(audio) == 0:
            return 0.0
        
        # Normalize both signals
        audio_norm = self._normalize_audio(audio)
        template_norm = self._normalize_audio(template)
        
        # Resize to match (simple approach)
        min_len = min(len(audio_norm), len(template_norm))
        if min_len == 0:
            return 0.0
        
        audio_resized = audio_norm[:min_len]
        template_resized = template_norm[:min_len]
        
        # Compute correlation
        correlation = np.corrcoef(audio_resized, template_resized)[0, 1]
        
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0


# Factory function
def create_local_audio_analyzer(sample_rate: int = 44100, frame_size: int = 1024) -> LocalAudioAnalyzer:
    """Create local audio analyzer instance"""
    return LocalAudioAnalyzer(sample_rate, frame_size)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_audio_analyzer():
        print("🎵 Testing Local Audio Analyzer 🎵")
        print("=" * 50)
        
        analyzer = create_local_audio_analyzer()
        
        # Generate test audio (kick-like sound)
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create gabber-style kick
        frequency = 55  # Hz
        envelope = np.exp(-t * 10)  # Exponential decay
        test_audio = np.sin(2 * np.pi * frequency * t) * envelope
        
        print("1️⃣ Basic Audio Analysis:")
        analysis = await analyzer.analyze_audio(test_audio)
        print(f"   Peak Level: {analysis.peak:.3f}")
        print(f"   RMS Level: {analysis.rms:.3f}")
        print(f"   Spectral Centroid: {analysis.spectral_centroid:.1f} Hz")
        
        print("\n2️⃣ Kick DNA Analysis:")
        kick_analysis = await analyzer.analyze_kick_dna(test_audio)
        print(f"   Attack Time: {kick_analysis.attack_time:.1f} ms")
        print(f"   Sustain Level: {kick_analysis.sustain_level:.3f}")
        print(f"   Decay Rate: {kick_analysis.decay_rate:.3f}")
        print(f"   Fundamental: {kick_analysis.fundamental_freq:.1f} Hz")
        print(f"   Harmonic Ratio: {kick_analysis.harmonic_ratio:.3f}")
        print(f"   Punch Factor: {kick_analysis.punch_factor:.3f}")
        print(f"   Rumble Factor: {kick_analysis.rumble_factor:.3f}")
        
        print("\n3️⃣ Template Matching:")
        for template_name in ['gabber', 'industrial', 'hardcore']:
            similarity = analyzer.compare_with_template(test_audio, template_name)
            print(f"   {template_name.title()}: {similarity:.3f}")
        
        print("\n✨ Local Audio Analyzer fully operational!")
    
    asyncio.run(test_audio_analyzer())