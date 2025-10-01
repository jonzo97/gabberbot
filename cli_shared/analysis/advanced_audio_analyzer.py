#!/usr/bin/env python3
"""
Advanced Multi-Modal Audio Analysis Engine for Hardcore Music

This is the brain of Gabberbot - analyzes hardcore music at a molecular level:
- Real-time kick drum DNA profiling and classification
- Psychoacoustic modeling (brightness, roughness, crunch factor)
- Pattern evolution tracking with ML-powered insights
- Cross-reference with artist database for style detection
- Multi-dimensional analysis for live performance optimization
"""

import numpy as np
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json

# Audio analysis libraries
try:
    import librosa
    import scipy.signal as signal
    from scipy.fft import fft, fftfreq
    ADVANCED_ANALYSIS = True
except ImportError:
    ADVANCED_ANALYSIS = False
    print("⚠️ Advanced analysis libraries not available. Install: pip install librosa scipy")

# ML libraries for pattern analysis
try:
    import sklearn.cluster as cluster
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not available. Install: pip install scikit-learn")

class KickDNAType(Enum):
    """Kick drum DNA classifications"""
    GABBER_CLASSIC = "gabber_classic"          # TR-909 analog kick with heavy distortion
    GABBER_MODERN = "gabber_modern"            # Digital gabber with pristine punch
    RAWSTYLE_POWER = "rawstyle_power"          # Rawstyle with reverse bass
    INDUSTRIAL_RUMBLE = "industrial_rumble"     # 3-layer industrial system
    PVC_KICK = "pvc_kick"                      # Typical PVC hardcore kick
    ZAAG_LEAD = "zaag_lead"                    # Dutch zaag-style lead kick
    EARTHQUAKE = "earthquake"                   # Massive sub-bass emphasis
    PIEP_KICK = "piep_kick"                    # High-pitched uptempo kick
    MILLENNIUM = "millennium"                   # Y2K millennium hardcore
    UNKNOWN = "unknown"                        # Unclassified kick

class PsychoacousticProfile(Enum):
    """Psychoacoustic profiling categories"""
    BRIGHTNESS = "brightness"        # Spectral centroid analysis
    ROUGHNESS = "roughness"         # Sensory dissonance
    WARMTH = "warmth"               # Low-mid frequency content
    PRESENCE = "presence"           # Mid-high frequency content  
    CRUNCH_FACTOR = "crunch_factor" # Digital distortion artifacts
    AGGRESSION = "aggression"       # Overall intensity and attack
    WAREHOUSE_FACTOR = "warehouse"  # Industrial/warehouse characteristics

@dataclass
class KickDNAProfile:
    """Complete DNA profile of a kick drum"""
    kick_type: KickDNAType
    confidence: float = 0.0              # Classification confidence (0-1)
    
    # Frequency characteristics
    fundamental_freq: float = 0.0        # Primary frequency component
    sub_bass_energy: float = 0.0         # 20-60Hz energy
    body_energy: float = 0.0             # 60-200Hz energy  
    attack_energy: float = 0.0           # 200-500Hz energy
    click_energy: float = 0.0            # 2-6kHz energy
    
    # Temporal characteristics
    attack_time: float = 0.0             # Time to peak (seconds)
    decay_time: float = 0.0              # Time to sustain level
    sustain_level: float = 0.0           # Sustain amplitude ratio
    punch_factor: float = 0.0            # Attack/body energy ratio
    
    # Processing characteristics
    distortion_level: float = 0.0        # Amount of harmonic distortion
    compression_ratio: float = 0.0       # Dynamic range compression
    eq_curve: List[Tuple[float, float]] = field(default_factory=list)  # Frequency response
    
    # Style identifiers
    doorlussen_factor: float = 0.0       # Serial distortion intensity
    rumble_factor: float = 0.0           # Industrial low-end rumble
    crunch_signature: List[float] = field(default_factory=list)  # Crunch fingerprint
    
    # Artist/era classification
    era_classification: str = "modern"    # 90s, 2000s, 2010s, modern
    regional_style: str = "dutch"        # Dutch, German, Belgian, etc.
    artist_similarity: Dict[str, float] = field(default_factory=dict)  # Artist match scores

@dataclass  
class PsychoacousticAnalysis:
    """Psychoacoustic analysis results"""
    brightness: float = 0.0              # Spectral brightness (0-1)
    roughness: float = 0.0               # Perceptual roughness (0-1)
    warmth: float = 0.0                  # Low-frequency warmth (0-1)
    presence: float = 0.0                # Mid-high presence (0-1)
    crunch_factor: float = 0.0           # Digital crunch artifacts (0-1)
    aggression: float = 0.0              # Overall aggression level (0-1)
    warehouse_factor: float = 0.0        # Industrial/warehouse character (0-1)
    
    # Advanced psychoacoustic metrics
    spectral_centroid: float = 0.0       # Center of mass of spectrum
    spectral_rolloff: float = 0.0        # 85% energy rolloff point
    spectral_flux: float = 0.0           # Rate of spectral change
    zero_crossing_rate: float = 0.0      # Measure of noisiness
    mfcc_features: List[float] = field(default_factory=list)  # Mel-frequency cepstral coefficients
    
    # Perceptual descriptors
    descriptors: Dict[str, float] = field(default_factory=dict)

@dataclass
class PatternEvolutionMetrics:
    """Metrics for tracking pattern evolution"""
    pattern_id: str
    generation: int = 0                   # Evolution generation number
    parent_patterns: List[str] = field(default_factory=list)  # Parent pattern IDs
    
    # Complexity metrics
    rhythmic_complexity: float = 0.0     # Measure of rhythmic intricacy
    harmonic_complexity: float = 0.0     # Harmonic content complexity
    structural_complexity: float = 0.0   # Pattern structure complexity
    
    # Fitness scores
    hardcore_authenticity: float = 0.0   # How "hardcore" the pattern sounds
    danceability: float = 0.0            # Estimated dancefloor effectiveness
    innovation: float = 0.0              # Novelty compared to existing patterns
    technical_quality: float = 0.0       # Audio quality and production value
    
    # Evolution tracking
    mutations_applied: List[str] = field(default_factory=list)  # Applied mutations
    crossover_points: List[int] = field(default_factory=list)   # Crossover locations
    selection_pressure: float = 0.0      # Selection pressure applied
    
    # Performance metrics
    play_count: int = 0                  # Times pattern was played
    user_rating: float = 0.0             # User feedback rating
    crowd_response: float = 0.0          # Estimated audience response

class AdvancedAudioAnalyzer:
    """
    Advanced multi-modal audio analysis engine for hardcore music
    
    Features:
    - Real-time kick drum DNA profiling
    - Psychoacoustic modeling and analysis
    - Pattern evolution tracking with ML
    - Artist style classification
    - Live performance optimization
    """
    
    def __init__(self, sample_rate: int = 44100, frame_size: int = 2048):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = frame_size // 4
        
        # Analysis state
        self.is_analyzing = False
        self.analysis_thread = None
        
        # Pattern evolution database
        self.pattern_database = {}
        self.evolution_history = []
        
        # Artist signature database
        self.artist_signatures = self._load_artist_signatures()
        
        # Kick DNA classifier (if ML available)
        self.kick_classifier = None
        if ML_AVAILABLE:
            self._initialize_kick_classifier()
        
        # Real-time analysis buffers
        self.audio_buffer = np.zeros(frame_size * 10)  # 10-frame rolling buffer
        self.analysis_history = []
        self.max_history = 1000  # Keep last 1000 analyses
        
        # Performance optimization
        self.analysis_cache = {}
        self.cache_timeout = 30.0  # 30 second cache
        
        self.logger = logging.getLogger(__name__)
        
    def _load_artist_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Load artist signature database"""
        # In a real implementation, this would load from a comprehensive database
        return {
            "angerfist": {
                "kick_dna": KickDNAType.GABBER_MODERN,
                "psychoacoustic": {
                    "brightness": 0.7,
                    "aggression": 0.9,
                    "crunch_factor": 0.8,
                    "warehouse_factor": 0.6
                },
                "signature_frequencies": [60, 200, 3000, 8000],
                "typical_bpm_range": (180, 200)
            },
            
            "mad_dog": {
                "kick_dna": KickDNAType.GABBER_CLASSIC,
                "psychoacoustic": {
                    "brightness": 0.6,
                    "aggression": 0.95,
                    "crunch_factor": 0.9,
                    "warehouse_factor": 0.8
                },
                "signature_frequencies": [55, 180, 2500, 6000],
                "typical_bpm_range": (175, 195)
            },
            
            "the_prophet": {
                "kick_dna": KickDNAType.RAWSTYLE_POWER,
                "psychoacoustic": {
                    "brightness": 0.8,
                    "aggression": 0.85,
                    "crunch_factor": 0.7,
                    "warehouse_factor": 0.5
                },
                "signature_frequencies": [65, 220, 4000, 12000],
                "typical_bpm_range": (145, 165)
            },
            
            "surgeon": {
                "kick_dna": KickDNAType.INDUSTRIAL_RUMBLE,
                "psychoacoustic": {
                    "brightness": 0.4,
                    "aggression": 0.8,
                    "crunch_factor": 0.6,
                    "warehouse_factor": 0.95
                },
                "signature_frequencies": [45, 150, 1500, 5000],
                "typical_bpm_range": (130, 150)
            },
            
            "partyraiser": {
                "kick_dna": KickDNAType.MILLENNIUM,
                "psychoacoustic": {
                    "brightness": 0.75,
                    "aggression": 0.9,
                    "crunch_factor": 0.85,
                    "warehouse_factor": 0.7
                },
                "signature_frequencies": [62, 190, 3500, 9000],
                "typical_bpm_range": (170, 190)
            }
        }
    
    def _initialize_kick_classifier(self):
        """Initialize ML classifier for kick drum DNA"""
        if not ML_AVAILABLE:
            return
        
        # In a real implementation, this would load a pre-trained model
        # For now, create a basic clustering-based classifier
        self.kick_classifier = {
            "method": "clustering",
            "model": cluster.KMeans(n_clusters=len(KickDNAType), random_state=42),
            "scaler": None,
            "trained": False
        }
    
    async def analyze_kick_dna(self, audio: np.ndarray) -> KickDNAProfile:
        """
        Perform comprehensive kick drum DNA analysis
        
        Args:
            audio: Audio signal containing kick drum
            
        Returns:
            Complete kick DNA profile
        """
        if not ADVANCED_ANALYSIS:
            return self._basic_kick_analysis(audio)
        
        profile = KickDNAProfile(KickDNAType.UNKNOWN)
        
        try:
            # Spectral analysis
            stft = librosa.stft(audio, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            frequencies = librosa.fft_frequencies(sr=self.sample_rate)
            
            # Find fundamental frequency
            low_freq_mask = (frequencies >= 30) & (frequencies <= 120)
            if np.any(low_freq_mask):
                low_freq_spectrum = np.mean(magnitude[low_freq_mask], axis=1)
                fundamental_idx = np.argmax(low_freq_spectrum)
                profile.fundamental_freq = frequencies[low_freq_mask][fundamental_idx]
            
            # Energy distribution analysis
            profile.sub_bass_energy = self._calculate_band_energy(magnitude, frequencies, 20, 60)
            profile.body_energy = self._calculate_band_energy(magnitude, frequencies, 60, 200)
            profile.attack_energy = self._calculate_band_energy(magnitude, frequencies, 200, 500)
            profile.click_energy = self._calculate_band_energy(magnitude, frequencies, 2000, 6000)
            
            # Temporal analysis
            envelope = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            if len(envelope) > 0:
                peak_idx = np.argmax(envelope)
                profile.attack_time = peak_idx * self.hop_length / self.sample_rate
                
                # Find decay characteristics
                decay_start = peak_idx
                decay_envelope = envelope[decay_start:]
                if len(decay_envelope) > 10:
                    # Find 63% decay point (1/e)
                    target_level = envelope[peak_idx] * 0.37
                    decay_idx = np.where(decay_envelope <= target_level)[0]
                    if len(decay_idx) > 0:
                        profile.decay_time = decay_idx[0] * self.hop_length / self.sample_rate
            
            # Punch factor calculation
            profile.punch_factor = profile.attack_energy / (profile.body_energy + 1e-10)
            
            # Distortion analysis
            profile.distortion_level = self._analyze_harmonic_distortion(audio)
            
            # Processing signature analysis
            profile.doorlussen_factor = self._analyze_doorlussen_signature(magnitude, frequencies)
            profile.rumble_factor = self._analyze_rumble_signature(magnitude, frequencies)
            
            # Classify kick type using ML or heuristics
            profile.kick_type, profile.confidence = self._classify_kick_type(profile)
            
            # Artist similarity analysis
            profile.artist_similarity = self._calculate_artist_similarity(profile)
            
            # Era and regional classification
            profile.era_classification = self._classify_era(profile)
            profile.regional_style = self._classify_regional_style(profile)
            
        except Exception as e:
            self.logger.error(f"Error in kick DNA analysis: {e}")
        
        return profile
    
    def _calculate_band_energy(self, magnitude: np.ndarray, frequencies: np.ndarray, 
                              low_freq: float, high_freq: float) -> float:
        """Calculate energy in frequency band"""
        band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        if np.any(band_mask):
            return float(np.sum(magnitude[band_mask] ** 2))
        return 0.0
    
    def _analyze_harmonic_distortion(self, audio: np.ndarray) -> float:
        """Analyze harmonic distortion content"""
        if not ADVANCED_ANALYSIS:
            return 0.0
        
        # Calculate THD using FFT
        fft_data = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1/self.sample_rate)
        
        # Find fundamental
        positive_freqs = freqs[freqs > 0]
        positive_fft = fft_data[freqs > 0]
        
        if len(positive_fft) == 0:
            return 0.0
        
        # Find peak (fundamental)
        fundamental_idx = np.argmax(positive_fft[:len(positive_fft)//4])  # Look in lower frequencies
        fundamental_freq = positive_freqs[fundamental_idx]
        fundamental_power = positive_fft[fundamental_idx] ** 2
        
        # Sum harmonic powers
        harmonic_power = 0.0
        for harmonic in range(2, 11):  # 2nd through 10th harmonic
            harmonic_freq = fundamental_freq * harmonic
            # Find closest frequency bin
            freq_diff = np.abs(positive_freqs - harmonic_freq)
            if np.min(freq_diff) < fundamental_freq * 0.1:  # 10% tolerance
                closest_idx = np.argmin(freq_diff)
                harmonic_power += positive_fft[closest_idx] ** 2
        
        # Calculate THD
        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power)
            return float(min(1.0, thd))  # Clamp to 0-1
        
        return 0.0
    
    def _analyze_doorlussen_signature(self, magnitude: np.ndarray, frequencies: np.ndarray) -> float:
        """Analyze doorlussen (serial distortion) signature"""
        # Doorlussen creates characteristic spectral patterns
        # Look for multiple distortion peaks and spectral irregularities
        
        # Calculate spectral irregularity
        spectral_mean = np.mean(magnitude, axis=1)
        if len(spectral_mean) < 10:
            return 0.0
        
        # Look for spectral peaks that indicate serial processing
        smoothed = signal.savgol_filter(spectral_mean, 5, 3)
        irregularity = np.mean(np.abs(spectral_mean - smoothed))
        max_magnitude = np.max(spectral_mean)
        
        if max_magnitude > 0:
            doorlussen_factor = irregularity / max_magnitude
            return float(min(1.0, doorlussen_factor * 3))  # Scale and clamp
        
        return 0.0
    
    def _analyze_rumble_signature(self, magnitude: np.ndarray, frequencies: np.ndarray) -> float:
        """Analyze industrial rumble signature"""
        # Industrial kicks have characteristic sub-bass rumble
        rumble_mask = (frequencies >= 20) & (frequencies <= 80)
        mid_mask = (frequencies >= 200) & (frequencies <= 800)
        
        if np.any(rumble_mask) and np.any(mid_mask):
            rumble_energy = np.sum(magnitude[rumble_mask] ** 2)
            mid_energy = np.sum(magnitude[mid_mask] ** 2)
            
            if mid_energy > 0:
                rumble_ratio = rumble_energy / mid_energy
                return float(min(1.0, rumble_ratio))
        
        return 0.0
    
    def _classify_kick_type(self, profile: KickDNAProfile) -> Tuple[KickDNAType, float]:
        """Classify kick drum type based on analysis"""
        # Heuristic classification based on characteristics
        scores = {}
        
        # Gabber Classic characteristics
        gabber_classic_score = 0.0
        if 55 <= profile.fundamental_freq <= 70:
            gabber_classic_score += 0.3
        if profile.distortion_level > 0.6:
            gabber_classic_score += 0.2
        if profile.doorlussen_factor > 0.5:
            gabber_classic_score += 0.3
        if profile.punch_factor > 0.8:
            gabber_classic_score += 0.2
        scores[KickDNAType.GABBER_CLASSIC] = gabber_classic_score
        
        # Gabber Modern characteristics  
        gabber_modern_score = 0.0
        if 58 <= profile.fundamental_freq <= 68:
            gabber_modern_score += 0.3
        if 0.4 <= profile.distortion_level <= 0.7:
            gabber_modern_score += 0.2
        if profile.click_energy > 0.1:
            gabber_modern_score += 0.3
        if profile.punch_factor > 0.6:
            gabber_modern_score += 0.2
        scores[KickDNAType.GABBER_MODERN] = gabber_modern_score
        
        # Industrial Rumble characteristics
        industrial_score = 0.0
        if 40 <= profile.fundamental_freq <= 55:
            industrial_score += 0.3
        if profile.rumble_factor > 0.6:
            industrial_score += 0.4
        if profile.distortion_level < 0.5:
            industrial_score += 0.2
        if profile.attack_time > 0.005:
            industrial_score += 0.1
        scores[KickDNAType.INDUSTRIAL_RUMBLE] = industrial_score
        
        # Rawstyle Power characteristics
        rawstyle_score = 0.0
        if 62 <= profile.fundamental_freq <= 75:
            rawstyle_score += 0.3
        if profile.click_energy > 0.2:
            rawstyle_score += 0.3
        if profile.punch_factor > 1.0:
            rawstyle_score += 0.4
        scores[KickDNAType.RAWSTYLE_POWER] = rawstyle_score
        
        # PVC Kick characteristics
        pvc_score = 0.0
        if 60 <= profile.fundamental_freq <= 75:
            pvc_score += 0.2
        if 0.3 <= profile.distortion_level <= 0.6:
            pvc_score += 0.3
        if 0.5 <= profile.punch_factor <= 0.9:
            pvc_score += 0.3
        if profile.doorlussen_factor > 0.3:
            pvc_score += 0.2
        scores[KickDNAType.PVC_KICK] = pvc_score
        
        # Find best match
        if scores:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[best_type]
            return best_type, confidence
        
        return KickDNAType.UNKNOWN, 0.0
    
    def _calculate_artist_similarity(self, profile: KickDNAProfile) -> Dict[str, float]:
        """Calculate similarity to known artists"""
        similarities = {}
        
        for artist_name, artist_data in self.artist_signatures.items():
            similarity = 0.0
            
            # Kick DNA type match
            if profile.kick_type == artist_data["kick_dna"]:
                similarity += 0.4
            
            # Frequency signature match
            freq_match = 0.0
            for sig_freq in artist_data["signature_frequencies"]:
                # Check if our profile has energy near this frequency
                freq_diff = abs(profile.fundamental_freq - sig_freq)
                if freq_diff < 20:  # Within 20Hz
                    freq_match += 1.0 / len(artist_data["signature_frequencies"])
            similarity += freq_match * 0.3
            
            # Psychoacoustic match (would need psychoacoustic analysis of profile)
            # Simplified matching for now
            similarity += 0.3  # Placeholder
            
            similarities[artist_name] = min(1.0, similarity)
        
        return similarities
    
    def _classify_era(self, profile: KickDNAProfile) -> str:
        """Classify the era of the kick drum style"""
        # Heuristic era classification
        if profile.distortion_level > 0.8 and profile.doorlussen_factor > 0.7:
            return "90s"
        elif profile.click_energy > 0.2 and profile.punch_factor > 1.0:
            return "2000s"  
        elif profile.distortion_level > 0.6 and profile.attack_energy > 0.3:
            return "2010s"
        else:
            return "modern"
    
    def _classify_regional_style(self, profile: KickDNAProfile) -> str:
        """Classify regional hardcore style"""
        # Simplified regional classification
        if profile.doorlussen_factor > 0.6:
            return "dutch"  # Netherlands - heavy doorlussen
        elif profile.rumble_factor > 0.7:
            return "german"  # Germany - industrial influence
        elif profile.distortion_level > 0.8:
            return "belgian"  # Belgium - extreme processing
        else:
            return "international"
    
    def _basic_kick_analysis(self, audio: np.ndarray) -> KickDNAProfile:
        """Basic kick analysis without advanced libraries"""
        profile = KickDNAProfile(KickDNAType.GABBER_CLASSIC)
        
        # Simple frequency analysis using FFT
        fft_data = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1/self.sample_rate)
        positive_freqs = freqs[freqs > 0]
        positive_fft = fft_data[freqs > 0]
        
        if len(positive_fft) > 0:
            # Find peak frequency in bass range
            bass_mask = (positive_freqs >= 30) & (positive_freqs <= 120)
            if np.any(bass_mask):
                bass_spectrum = positive_fft[bass_mask]
                bass_freqs = positive_freqs[bass_mask]
                peak_idx = np.argmax(bass_spectrum)
                profile.fundamental_freq = bass_freqs[peak_idx]
        
        # Basic energy calculations
        total_energy = np.sum(positive_fft ** 2)
        if total_energy > 0:
            # Sub-bass energy (20-60Hz)
            sub_mask = (positive_freqs >= 20) & (positive_freqs <= 60)
            if np.any(sub_mask):
                profile.sub_bass_energy = np.sum(positive_fft[sub_mask] ** 2) / total_energy
        
        profile.confidence = 0.5  # Lower confidence for basic analysis
        return profile
    
    async def analyze_psychoacoustic(self, audio: np.ndarray) -> PsychoacousticAnalysis:
        """
        Perform comprehensive psychoacoustic analysis
        
        Args:
            audio: Audio signal to analyze
            
        Returns:
            Psychoacoustic analysis results
        """
        analysis = PsychoacousticAnalysis()
        
        try:
            if ADVANCED_ANALYSIS:
                # Advanced analysis with librosa
                
                # Spectral centroid (brightness)
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
                analysis.spectral_centroid = float(np.mean(spectral_centroids))
                analysis.brightness = min(1.0, analysis.spectral_centroid / (self.sample_rate / 4))
                
                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
                analysis.spectral_rolloff = float(np.mean(rolloff))
                
                # Zero crossing rate (roughness indicator)
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                analysis.zero_crossing_rate = float(np.mean(zcr))
                analysis.roughness = min(1.0, analysis.zero_crossing_rate * 10)
                
                # MFCC features
                mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
                analysis.mfcc_features = np.mean(mfccs, axis=1).tolist()
                
                # Spectral flux (rate of change)
                stft = librosa.stft(audio)
                magnitude = np.abs(stft)
                spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
                analysis.spectral_flux = float(np.mean(spectral_flux))
                
                # Calculate perceptual descriptors
                analysis.warmth = self._calculate_warmth(magnitude, librosa.fft_frequencies(sr=self.sample_rate))
                analysis.presence = self._calculate_presence(magnitude, librosa.fft_frequencies(sr=self.sample_rate))
                analysis.crunch_factor = self._calculate_crunch_factor(audio)
                analysis.aggression = self._calculate_aggression(analysis)
                analysis.warehouse_factor = self._calculate_warehouse_factor(analysis)
                
            else:
                # Basic analysis without librosa
                analysis = self._basic_psychoacoustic_analysis(audio)
                
        except Exception as e:
            self.logger.error(f"Error in psychoacoustic analysis: {e}")
        
        return analysis
    
    def _calculate_warmth(self, magnitude: np.ndarray, frequencies: np.ndarray) -> float:
        """Calculate perceptual warmth (low-mid emphasis)"""
        warm_mask = (frequencies >= 200) & (frequencies <= 1000)
        total_energy = np.sum(magnitude ** 2)
        
        if np.any(warm_mask) and total_energy > 0:
            warm_energy = np.sum(magnitude[warm_mask] ** 2)
            return float(min(1.0, (warm_energy / total_energy) * 3))
        return 0.0
    
    def _calculate_presence(self, magnitude: np.ndarray, frequencies: np.ndarray) -> float:
        """Calculate perceptual presence (mid-high emphasis)"""
        presence_mask = (frequencies >= 2000) & (frequencies <= 8000)
        total_energy = np.sum(magnitude ** 2)
        
        if np.any(presence_mask) and total_energy > 0:
            presence_energy = np.sum(magnitude[presence_mask] ** 2)
            return float(min(1.0, (presence_energy / total_energy) * 2))
        return 0.0
    
    def _calculate_crunch_factor(self, audio: np.ndarray) -> float:
        """Calculate digital crunch factor"""
        # Look for high-frequency noise and aliasing artifacts
        fft_data = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1/self.sample_rate)
        
        # High-frequency energy (above 10kHz)
        hf_mask = freqs >= 10000
        total_energy = np.sum(fft_data ** 2)
        
        if np.any(hf_mask) and total_energy > 0:
            hf_energy = np.sum(fft_data[hf_mask] ** 2)
            hf_ratio = hf_energy / total_energy
            
            # Also check for clipping indicators
            clipping_factor = np.sum(np.abs(audio) > 0.95) / len(audio)
            
            crunch = (hf_ratio * 2) + (clipping_factor * 3)
            return float(min(1.0, crunch))
        
        return 0.0
    
    def _calculate_aggression(self, analysis: PsychoacousticAnalysis) -> float:
        """Calculate overall aggression level"""
        # Combine multiple factors for aggression
        aggression = (
            analysis.brightness * 0.3 +
            analysis.roughness * 0.3 +
            analysis.crunch_factor * 0.4
        )
        return min(1.0, aggression)
    
    def _calculate_warehouse_factor(self, analysis: PsychoacousticAnalysis) -> float:
        """Calculate warehouse/industrial character"""
        # Industrial/warehouse sound characteristics
        warehouse = (
            (1.0 - analysis.brightness) * 0.4 +  # Darker sound
            analysis.warmth * 0.3 +              # Warm low-mids
            (analysis.spectral_flux / 1000) * 0.3  # Spectral movement
        )
        return min(1.0, warehouse)
    
    def _basic_psychoacoustic_analysis(self, audio: np.ndarray) -> PsychoacousticAnalysis:
        """Basic psychoacoustic analysis without advanced libraries"""
        analysis = PsychoacousticAnalysis()
        
        # Simple FFT-based analysis
        fft_data = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1/self.sample_rate)
        positive_freqs = freqs[freqs > 0]
        positive_fft = fft_data[freqs > 0]
        
        if len(positive_fft) > 0:
            total_energy = np.sum(positive_fft ** 2)
            
            # Simple spectral centroid
            if total_energy > 0:
                analysis.spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
                analysis.brightness = min(1.0, analysis.spectral_centroid / (self.sample_rate / 4))
            
            # Simple zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
            analysis.zero_crossing_rate = zero_crossings / (2 * len(audio))
            analysis.roughness = min(1.0, analysis.zero_crossing_rate * 10)
            
            # Basic crunch factor
            analysis.crunch_factor = self._calculate_crunch_factor(audio)
            
            # Calculate other metrics from available data
            analysis.warmth = self._calculate_warmth(positive_fft.reshape(-1, 1), positive_freqs)
            analysis.presence = self._calculate_presence(positive_fft.reshape(-1, 1), positive_freqs)
            analysis.aggression = self._calculate_aggression(analysis)
            analysis.warehouse_factor = self._calculate_warehouse_factor(analysis)
        
        return analysis

# Test functions
async def test_advanced_analyzer():
    """Test the advanced audio analyzer"""
    print("🔬 Testing Advanced Audio Analyzer")
    print("=" * 50)
    
    analyzer = AdvancedAudioAnalyzer()
    
    # Generate test signals
    sample_rate = 44100
    duration = 1.0
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Create gabber kick test signal
    kick_freq = 60 * np.exp(-t * 8)  # Frequency sweep
    kick_phase = 2 * np.pi * np.cumsum(kick_freq) / sample_rate
    kick_env = np.exp(-t * 12)  # Amplitude envelope
    gabber_kick = kick_env * np.sin(kick_phase) * 0.8
    
    # Apply some hardcore processing
    gabber_kick = gabber_kick * 2.5  # Drive
    gabber_kick = np.tanh(gabber_kick)  # Saturation
    gabber_kick = np.clip(gabber_kick, -0.9, 0.9)  # Clipping
    
    print("🥁 Analyzing Gabber Kick DNA...")
    kick_dna = await analyzer.analyze_kick_dna(gabber_kick)
    
    print(f"   Kick Type: {kick_dna.kick_type.value}")
    print(f"   Confidence: {kick_dna.confidence:.2f}")
    print(f"   Fundamental: {kick_dna.fundamental_freq:.1f} Hz")
    print(f"   Punch Factor: {kick_dna.punch_factor:.2f}")
    print(f"   Distortion Level: {kick_dna.distortion_level:.2f}")
    print(f"   Doorlussen Factor: {kick_dna.doorlussen_factor:.2f}")
    print(f"   Era: {kick_dna.era_classification}")
    print(f"   Style: {kick_dna.regional_style}")
    
    if kick_dna.artist_similarity:
        best_match = max(kick_dna.artist_similarity.items(), key=lambda x: x[1])
        print(f"   Most Similar Artist: {best_match[0]} ({best_match[1]:.2f})")
    
    print("\n🧠 Performing Psychoacoustic Analysis...")
    psycho = await analyzer.analyze_psychoacoustic(gabber_kick)
    
    print(f"   Brightness: {psycho.brightness:.2f}")
    print(f"   Roughness: {psycho.roughness:.2f}")
    print(f"   Warmth: {psycho.warmth:.2f}")
    print(f"   Presence: {psycho.presence:.2f}")
    print(f"   Crunch Factor: {psycho.crunch_factor:.2f}")
    print(f"   Aggression: {psycho.aggression:.2f}")
    print(f"   Warehouse Factor: {psycho.warehouse_factor:.2f}")
    print(f"   Spectral Centroid: {psycho.spectral_centroid:.1f} Hz")
    
    print("\n✅ Advanced Audio Analyzer test completed!")

if __name__ == "__main__":
    asyncio.run(test_advanced_analyzer())