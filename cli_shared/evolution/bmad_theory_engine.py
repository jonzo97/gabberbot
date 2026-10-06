#!/usr/bin/env python3
"""
BMAD Mathematical Music Theory Engine
@theory-engine (Cipher) - Advanced Music Theory Analysis for Hardcore Evolution

Mathematical frameworks for analyzing hardcore music patterns:
- Harmonic progression analysis using interval matrices
- Rhythmic complexity scoring with fractal dimensions
- Energy curve optimization for warehouse sound systems
- Genre authenticity scoring based on hardcore tradition
- Pattern uniqueness measurement using musical DNA
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..models.hardcore_models import HardcorePattern, HardcoreTrack, PatternStep, SynthType, SynthParams
from ..models.midi_clips import MIDIClip, MIDINote


class HarmonicFunction(Enum):
    """Harmonic functions in hardcore music"""
    TONIC = "tonic"               # Stability, resolution
    SUBDOMINANT = "subdominant"   # Departure, preparation
    DOMINANT = "dominant"         # Tension, movement to tonic
    CHROMATIC = "chromatic"       # Color, passing tones
    DECEPTIVE = "deceptive"       # Unexpected resolution
    INDUSTRIAL = "industrial"     # Atonal, aggressive


class RhythmicComplexity(Enum):
    """Levels of rhythmic complexity"""
    MINIMAL = 1     # Simple 4/4 patterns
    MODERATE = 2    # Syncopation, some offbeats
    COMPLEX = 3     # Polyrhythms, irregular groupings
    CHAOTIC = 4     # Extreme complexity, nearly random
    FRENZIED = 5    # Maximum hardcore complexity


@dataclass
class HarmonicAnalysis:
    """Results of harmonic analysis"""
    intervals: List[int] = field(default_factory=list)  # Interval sequence
    chord_functions: List[HarmonicFunction] = field(default_factory=list)
    tension_curve: List[float] = field(default_factory=list)  # 0-1 tension over time
    key_stability: float = 0.0  # How well pattern stays in key
    chromaticism: float = 0.0   # Amount of chromatic movement
    harmonic_rhythm: float = 0.0  # Rate of harmonic change
    dissonance_density: float = 0.0  # Amount of dissonance
    resolution_quality: float = 0.0  # Quality of resolutions


@dataclass
class RhythmicAnalysis:
    """Results of rhythmic analysis"""
    complexity_level: RhythmicComplexity = RhythmicComplexity.MINIMAL
    syncopation_density: float = 0.0  # Amount of syncopation
    polyrhythmic_layers: int = 0  # Number of simultaneous rhythms
    fractal_dimension: float = 0.0  # Fractal complexity measure
    groove_quality: float = 0.0  # How well it grooves
    kick_pattern_strength: float = 0.0  # Strength of kick pattern
    offbeat_density: float = 0.0  # Amount of offbeat activity
    rhythmic_variance: float = 0.0  # Rhythmic unpredictability


@dataclass
class EnergyAnalysis:
    """Energy curve analysis for warehouse systems"""
    peak_energy: float = 0.0  # Maximum energy level
    average_energy: float = 0.0  # Average energy level
    energy_variance: float = 0.0  # Energy variation
    build_quality: float = 0.0  # Quality of energy builds
    drop_impact: float = 0.0  # Impact of energy drops
    sustained_intensity: float = 0.0  # Ability to maintain energy
    warehouse_compatibility: float = 0.0  # Suitability for large venues


@dataclass
class GenreAnalysis:
    """Genre authenticity analysis"""
    gabber_authenticity: float = 0.0
    frenchcore_elements: float = 0.0
    industrial_characteristics: float = 0.0
    rawstyle_features: float = 0.0
    speedcore_intensity: float = 0.0
    uptempo_hardcore: float = 0.0
    mainstream_hardcore: float = 0.0
    underground_rating: float = 0.0


@dataclass
class PatternDNA:
    """Musical DNA of a pattern for evolution"""
    harmonic_signature: List[float] = field(default_factory=list)
    rhythmic_signature: List[float] = field(default_factory=list)
    energy_signature: List[float] = field(default_factory=list)
    genre_signature: List[float] = field(default_factory=list)
    complexity_vector: List[float] = field(default_factory=list)
    unique_features: Set[str] = field(default_factory=set)
    
    def similarity_to(self, other_dna: 'PatternDNA') -> float:
        """Calculate similarity to another pattern DNA (0-1)"""
        similarities = []
        
        # Harmonic similarity
        if self.harmonic_signature and other_dna.harmonic_signature:
            harmonic_sim = np.corrcoef(self.harmonic_signature, other_dna.harmonic_signature)[0, 1]
            similarities.append(max(0, harmonic_sim))
        
        # Rhythmic similarity  
        if self.rhythmic_signature and other_dna.rhythmic_signature:
            rhythmic_sim = np.corrcoef(self.rhythmic_signature, other_dna.rhythmic_signature)[0, 1]
            similarities.append(max(0, rhythmic_sim))
        
        # Energy similarity
        if self.energy_signature and other_dna.energy_signature:
            energy_sim = np.corrcoef(self.energy_signature, other_dna.energy_signature)[0, 1]
            similarities.append(max(0, energy_sim))
        
        # Genre similarity
        if self.genre_signature and other_dna.genre_signature:
            genre_sim = np.corrcoef(self.genre_signature, other_dna.genre_signature)[0, 1]
            similarities.append(max(0, genre_sim))
        
        # Feature overlap
        if self.unique_features and other_dna.unique_features:
            common_features = len(self.unique_features.intersection(other_dna.unique_features))
            total_features = len(self.unique_features.union(other_dna.unique_features))
            feature_sim = common_features / total_features if total_features > 0 else 0
            similarities.append(feature_sim)
        
        return np.mean(similarities) if similarities else 0.0


class BMADTheoryEngine:
    """
    Advanced mathematical music theory engine for hardcore pattern analysis
    
    Provides deep analysis of patterns for intelligent evolution:
    - Harmonic progression analysis with tension curves
    - Rhythmic complexity measurement using fractal dimensions
    - Energy optimization for warehouse sound systems
    - Genre authenticity scoring
    - Pattern DNA extraction for evolution
    """
    
    def __init__(self):
        # Music theory constants
        self.CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # Chromatic circle
        self.CONSONANT_INTERVALS = {0, 3, 4, 7, 8, 9}  # Perfect unison, major/minor 3rd, perfect 4th/5th, major 6th
        self.DISSONANT_INTERVALS = {1, 2, 5, 6, 10, 11}  # Minor/major 2nd, tritone, minor 7th, major 7th
        
        # Hardcore-specific theory
        self.GABBER_FREQUENCIES = [45, 50, 55, 60, 65, 70]  # Typical gabber kick frequencies
        self.HARDCORE_SCALES = {
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "chromatic": list(range(12))
        }
        
        # Energy curve templates for different hardcore styles
        self.ENERGY_TEMPLATES = {
            "gabber": [0.9, 0.9, 0.9, 0.9, 0.95, 0.95, 0.95, 0.95],  # Sustained high energy
            "frenchcore": [0.7, 0.8, 0.6, 0.9, 0.5, 1.0, 0.4, 0.95],  # Build and drop
            "industrial": [0.6, 0.7, 0.8, 0.6, 0.7, 0.9, 0.8, 0.7],  # Grinding progression
            "speedcore": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # Maximum intensity
        }
    
    def analyze_harmonic_content(self, pattern: HardcorePattern) -> HarmonicAnalysis:
        """Analyze harmonic content and progression"""
        analysis = HarmonicAnalysis()
        
        # Extract pitch sequences from all tracks
        pitch_sequences = self._extract_pitch_sequences(pattern)
        if not pitch_sequences:
            return analysis
        
        # Analyze intervals between notes
        all_intervals = []
        for track_name, pitches in pitch_sequences.items():
            if len(pitches) > 1:
                intervals = [abs(pitches[i+1] - pitches[i]) % 12 for i in range(len(pitches)-1)]
                all_intervals.extend(intervals)
        
        analysis.intervals = all_intervals
        
        # Calculate harmonic metrics
        if all_intervals:
            analysis.chromaticism = sum(1 for i in all_intervals if i in [1, 11]) / len(all_intervals)
            
            # Key stability (how often we use scale tones)
            scale_tones = set(self.HARDCORE_SCALES["minor"])
            scale_hits = sum(1 for i in all_intervals if i in scale_tones)
            analysis.key_stability = scale_hits / len(all_intervals)
            
            # Dissonance density
            dissonant_hits = sum(1 for i in all_intervals if i in self.DISSONANT_INTERVALS)
            analysis.dissonance_density = dissonant_hits / len(all_intervals)
        
        # Analyze harmonic rhythm (rate of chord/note changes)
        analysis.harmonic_rhythm = self._calculate_harmonic_rhythm(pattern)
        
        # Generate tension curve
        analysis.tension_curve = self._calculate_tension_curve(pattern, pitch_sequences)
        
        return analysis
    
    def analyze_rhythmic_complexity(self, pattern: HardcorePattern) -> RhythmicAnalysis:
        """Analyze rhythmic complexity and patterns"""
        analysis = RhythmicAnalysis()
        
        # Get step density for each track
        track_densities = {}
        track_patterns = {}
        
        for track_name, track in pattern.tracks.items():
            active_steps = [i for i, step in enumerate(track.steps) if step is not None]
            density = len(active_steps) / len(track.steps)
            track_densities[track_name] = density
            track_patterns[track_name] = active_steps
        
        if not track_densities:
            return analysis
        
        # Calculate syncopation density
        analysis.syncopation_density = self._calculate_syncopation(pattern)
        
        # Count polyrhythmic layers
        analysis.polyrhythmic_layers = len([d for d in track_densities.values() if d > 0.1])
        
        # Calculate fractal dimension of rhythm
        analysis.fractal_dimension = self._calculate_fractal_dimension(pattern)
        
        # Analyze kick pattern strength
        analysis.kick_pattern_strength = self._analyze_kick_pattern(pattern)
        
        # Calculate offbeat density
        analysis.offbeat_density = self._calculate_offbeat_density(pattern)
        
        # Determine complexity level
        complexity_score = (
            analysis.syncopation_density * 0.3 +
            analysis.polyrhythmic_layers * 0.2 +
            analysis.fractal_dimension * 0.3 +
            analysis.offbeat_density * 0.2
        )
        
        if complexity_score < 0.2:
            analysis.complexity_level = RhythmicComplexity.MINIMAL
        elif complexity_score < 0.4:
            analysis.complexity_level = RhythmicComplexity.MODERATE
        elif complexity_score < 0.6:
            analysis.complexity_level = RhythmicComplexity.COMPLEX
        elif complexity_score < 0.8:
            analysis.complexity_level = RhythmicComplexity.CHAOTIC
        else:
            analysis.complexity_level = RhythmicComplexity.FRENZIED
        
        # Calculate groove quality (balance between complexity and predictability)
        analysis.groove_quality = self._calculate_groove_quality(pattern, analysis)
        
        return analysis
    
    def analyze_energy_curve(self, pattern: HardcorePattern) -> EnergyAnalysis:
        """Analyze energy curve for warehouse optimization"""
        analysis = EnergyAnalysis()
        
        # Calculate energy for each step
        energy_curve = []
        for step in range(pattern.steps):
            step_energy = 0.0
            step_events = pattern.get_step_events(step)
            
            for track_name, pattern_step in step_events:
                # Energy from amplitude and velocity
                energy = pattern_step.params.amp * pattern_step.velocity
                
                # Boost energy for low frequencies (kicks)
                if pattern_step.params.freq < 100:
                    energy *= 1.5
                
                # Boost energy for crunch and drive
                energy *= (1 + pattern_step.params.crunch * 0.5)
                energy *= (1 + pattern_step.params.drive * 0.1)
                
                step_energy += energy
            
            energy_curve.append(step_energy)
        
        if energy_curve:
            analysis.peak_energy = max(energy_curve)
            analysis.average_energy = np.mean(energy_curve)
            analysis.energy_variance = np.var(energy_curve)
            
            # Analyze builds and drops
            analysis.build_quality = self._analyze_energy_builds(energy_curve)
            analysis.drop_impact = self._analyze_energy_drops(energy_curve)
            
            # Calculate sustained intensity
            high_energy_steps = sum(1 for e in energy_curve if e > analysis.average_energy * 1.2)
            analysis.sustained_intensity = high_energy_steps / len(energy_curve)
            
            # Warehouse compatibility (prefers sustained high energy with good builds)
            analysis.warehouse_compatibility = (
                analysis.sustained_intensity * 0.4 +
                analysis.build_quality * 0.3 +
                min(1.0, analysis.peak_energy / 3.0) * 0.3
            )
        
        return analysis
    
    def analyze_genre_authenticity(self, pattern: HardcorePattern) -> GenreAnalysis:
        """Analyze hardcore genre authenticity"""
        analysis = GenreAnalysis()
        
        # BPM-based genre indicators
        bpm = pattern.bpm
        
        # Gabber authenticity (180-200 BPM, 4/4 kicks, high crunch)
        if 170 <= bpm <= 200:
            analysis.gabber_authenticity += 0.3
        kick_strength = self._analyze_kick_pattern(pattern)
        analysis.gabber_authenticity += kick_strength * 0.4
        
        # Check for gabber-style parameters
        total_crunch = 0.0
        param_count = 0
        low_freq_count = 0
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    total_crunch += step.params.crunch
                    param_count += 1
                    if step.params.freq < 100:
                        low_freq_count += 1
        
        if param_count > 0:
            avg_crunch = total_crunch / param_count
            analysis.gabber_authenticity += min(0.3, avg_crunch * 0.4)
        
        # Frenchcore elements (200+ BPM, melodic elements, complex patterns)
        if bpm >= 200:
            analysis.frenchcore_elements += 0.4
        
        # Check for melodic content
        pitch_variety = len(self._extract_unique_pitches(pattern))
        if pitch_variety > 5:
            analysis.frenchcore_elements += 0.3
        
        # Industrial characteristics (lower BPM, noise elements, grinding rhythms)
        if 130 <= bpm <= 160:
            analysis.industrial_characteristics += 0.3
        
        # Check for industrial synth types
        industrial_synths = {SynthType.INDUSTRIAL_KICK, SynthType.INDUSTRIAL_NOISE}
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None and step.synth_type in industrial_synths:
                    analysis.industrial_characteristics += 0.1
        
        # Speedcore intensity (extremely high BPM)
        if bpm >= 250:
            analysis.speedcore_intensity = min(1.0, (bpm - 250) / 100)
        
        # Underground rating (complex rhythms, unconventional structures)
        rhythmic_analysis = self.analyze_rhythmic_complexity(pattern)
        if rhythmic_analysis.complexity_level.value >= 3:
            analysis.underground_rating += 0.4
        
        if rhythmic_analysis.polyrhythmic_layers > 2:
            analysis.underground_rating += 0.3
        
        return analysis
    
    def extract_pattern_dna(self, pattern: HardcorePattern) -> PatternDNA:
        """Extract musical DNA for evolution algorithms"""
        dna = PatternDNA()
        
        # Harmonic signature
        harmonic_analysis = self.analyze_harmonic_content(pattern)
        dna.harmonic_signature = [
            harmonic_analysis.key_stability,
            harmonic_analysis.chromaticism,
            harmonic_analysis.harmonic_rhythm,
            harmonic_analysis.dissonance_density,
            len(harmonic_analysis.intervals) / 20.0  # Normalized interval count
        ]
        
        # Rhythmic signature
        rhythmic_analysis = self.analyze_rhythmic_complexity(pattern)
        dna.rhythmic_signature = [
            rhythmic_analysis.syncopation_density,
            rhythmic_analysis.polyrhythmic_layers / 5.0,  # Normalized
            rhythmic_analysis.fractal_dimension,
            rhythmic_analysis.groove_quality,
            rhythmic_analysis.kick_pattern_strength,
            rhythmic_analysis.offbeat_density
        ]
        
        # Energy signature
        energy_analysis = self.analyze_energy_curve(pattern)
        dna.energy_signature = [
            energy_analysis.peak_energy / 10.0,  # Normalized
            energy_analysis.average_energy / 5.0,
            energy_analysis.energy_variance,
            energy_analysis.build_quality,
            energy_analysis.drop_impact,
            energy_analysis.sustained_intensity,
            energy_analysis.warehouse_compatibility
        ]
        
        # Genre signature
        genre_analysis = self.analyze_genre_authenticity(pattern)
        dna.genre_signature = [
            genre_analysis.gabber_authenticity,
            genre_analysis.frenchcore_elements,
            genre_analysis.industrial_characteristics,
            genre_analysis.speedcore_intensity,
            genre_analysis.underground_rating
        ]
        
        # Complexity vector
        dna.complexity_vector = [
            len(pattern.tracks) / 8.0,  # Track count normalized
            pattern.bpm / 300.0,  # BPM normalized
            sum(len(track.steps) for track in pattern.tracks.values()) / 100.0,  # Total steps
            rhythmic_analysis.complexity_level.value / 5.0
        ]
        
        # Unique features
        dna.unique_features = self._extract_unique_features(pattern)
        
        return dna
    
    def calculate_evolution_fitness(self, pattern: HardcorePattern, 
                                  target_style: str = "gabber") -> Dict[str, float]:
        """Calculate fitness scores for evolution"""
        fitness = {}
        
        # Analyze all aspects
        harmonic = self.analyze_harmonic_content(pattern)
        rhythmic = self.analyze_rhythmic_complexity(pattern)
        energy = self.analyze_energy_curve(pattern)
        genre = self.analyze_genre_authenticity(pattern)
        
        # Hardcore authenticity (weighted by target style)
        if target_style == "gabber":
            fitness["authenticity"] = genre.gabber_authenticity
        elif target_style == "frenchcore":
            fitness["authenticity"] = genre.frenchcore_elements
        elif target_style == "industrial":
            fitness["authenticity"] = genre.industrial_characteristics
        else:
            fitness["authenticity"] = genre.gabber_authenticity  # Default
        
        # Danceability (balance of energy and groove)
        fitness["danceability"] = (
            energy.warehouse_compatibility * 0.6 +
            rhythmic.groove_quality * 0.4
        )
        
        # Rhythmic complexity
        fitness["rhythmic_complexity"] = rhythmic.complexity_level.value / 5.0
        
        # Harmonic richness
        fitness["harmonic_richness"] = (
            (1 - harmonic.key_stability) * 0.3 +  # Some chromaticism good
            harmonic.harmonic_rhythm * 0.4 +
            harmonic.dissonance_density * 0.3
        )
        
        # Energy level
        fitness["energy_level"] = min(1.0, energy.average_energy / 3.0)
        
        # Technical quality (parameter sanity)
        fitness["technical_quality"] = self._calculate_technical_quality(pattern)
        
        return fitness
    
    # Helper methods
    def _extract_pitch_sequences(self, pattern: HardcorePattern) -> Dict[str, List[int]]:
        """Extract pitch sequences from pattern tracks"""
        sequences = {}
        for track_name, track in pattern.tracks.items():
            pitches = []
            for step in track.steps:
                if step is not None:
                    # Convert frequency to MIDI note approximately
                    midi_note = int(69 + 12 * math.log2(step.params.freq / 440))
                    pitches.append(max(0, min(127, midi_note)))
            if pitches:
                sequences[track_name] = pitches
        return sequences
    
    def _calculate_harmonic_rhythm(self, pattern: HardcorePattern) -> float:
        """Calculate rate of harmonic change"""
        total_changes = 0
        total_steps = 0
        
        for track in pattern.tracks.values():
            prev_freq = None
            for step in track.steps:
                if step is not None:
                    if prev_freq is not None and abs(step.params.freq - prev_freq) > 10:
                        total_changes += 1
                    prev_freq = step.params.freq
                    total_steps += 1
        
        return total_changes / max(1, total_steps)
    
    def _calculate_tension_curve(self, pattern: HardcorePattern, 
                               pitch_sequences: Dict[str, List[int]]) -> List[float]:
        """Calculate tension curve over pattern"""
        tension_curve = []
        
        for step in range(pattern.steps):
            step_tension = 0.0
            step_events = pattern.get_step_events(step)
            
            for track_name, pattern_step in step_events:
                # Base tension from dissonance
                midi_note = int(69 + 12 * math.log2(pattern_step.params.freq / 440))
                interval = midi_note % 12
                if interval in self.DISSONANT_INTERVALS:
                    step_tension += 0.7
                elif interval in self.CONSONANT_INTERVALS:
                    step_tension += 0.3
                
                # Add tension from distortion
                step_tension += pattern_step.params.crunch * 0.5
                step_tension += pattern_step.params.drive * 0.1
            
            tension_curve.append(min(1.0, step_tension))
        
        return tension_curve
    
    def _calculate_syncopation(self, pattern: HardcorePattern) -> float:
        """Calculate amount of syncopation"""
        syncopated_hits = 0
        total_hits = 0
        
        # Strong beats in 4/4: 0, 4, 8, 12 (quarter notes)
        # Weak beats: 2, 6, 10, 14 (offbeats)
        # Syncopated: 1, 3, 5, 7, 9, 11, 13, 15 (16th note offbeats)
        
        strong_beats = {0, 4, 8, 12}
        weak_beats = {2, 6, 10, 14}
        syncopated_beats = {1, 3, 5, 7, 9, 11, 13, 15}
        
        for track in pattern.tracks.values():
            for i, step in enumerate(track.steps):
                if step is not None:
                    total_hits += 1
                    if i in syncopated_beats:
                        syncopated_hits += 1
        
        return syncopated_hits / max(1, total_hits)
    
    def _calculate_fractal_dimension(self, pattern: HardcorePattern) -> float:
        """Calculate fractal dimension of rhythm pattern"""
        # Create binary rhythm representation
        rhythm_matrix = []
        for track in pattern.tracks.values():
            track_rhythm = [1 if step is not None else 0 for step in track.steps]
            rhythm_matrix.append(track_rhythm)
        
        if not rhythm_matrix:
            return 0.0
        
        # Calculate box-counting dimension (simplified)
        combined_rhythm = [max(steps) for steps in zip(*rhythm_matrix)]
        
        # Count pattern changes at different scales
        scales = [1, 2, 4, 8]
        complexities = []
        
        for scale in scales:
            patterns = []
            for i in range(0, len(combined_rhythm), scale):
                pattern_chunk = combined_rhythm[i:i+scale]
                if len(pattern_chunk) == scale:
                    patterns.append(tuple(pattern_chunk))
            
            unique_patterns = len(set(patterns))
            total_patterns = len(patterns)
            complexity = unique_patterns / max(1, total_patterns)
            complexities.append(complexity)
        
        # Estimate fractal dimension
        if len(complexities) > 1:
            dimension = np.mean(complexities)
        else:
            dimension = complexities[0] if complexities else 0.0
        
        return min(1.0, dimension)
    
    def _analyze_kick_pattern(self, pattern: HardcorePattern) -> float:
        """Analyze strength of kick pattern"""
        kick_tracks = [track for name, track in pattern.tracks.items() 
                      if "kick" in name.lower()]
        
        if not kick_tracks:
            return 0.0
        
        kick_track = kick_tracks[0]  # Use first kick track
        
        # Check for strong beat emphasis
        strong_beats = {0, 4, 8, 12}  # Quarter note positions
        strong_beat_hits = sum(1 for i in strong_beats 
                              if i < len(kick_track.steps) and kick_track.steps[i] is not None)
        
        strength = strong_beat_hits / len(strong_beats)
        
        # Bonus for consistent kick pattern
        active_steps = [i for i, step in enumerate(kick_track.steps) if step is not None]
        if len(active_steps) >= 4:
            # Check for regular pattern
            intervals = [active_steps[i+1] - active_steps[i] for i in range(len(active_steps)-1)]
            if intervals and len(set(intervals)) <= 2:  # Regular pattern
                strength += 0.3
        
        return min(1.0, strength)
    
    def _calculate_offbeat_density(self, pattern: HardcorePattern) -> float:
        """Calculate density of offbeat activity"""
        offbeat_positions = {1, 3, 5, 7, 9, 11, 13, 15}  # 16th note offbeats
        
        offbeat_hits = 0
        total_offbeats = len(offbeat_positions)
        
        for track in pattern.tracks.values():
            for i in offbeat_positions:
                if i < len(track.steps) and track.steps[i] is not None:
                    offbeat_hits += 1
        
        return offbeat_hits / (total_offbeats * len(pattern.tracks)) if pattern.tracks else 0.0
    
    def _calculate_groove_quality(self, pattern: HardcorePattern, 
                                rhythmic_analysis: RhythmicAnalysis) -> float:
        """Calculate groove quality (balance of predictability and interest)"""
        # Good groove = moderate complexity + strong kick + some syncopation
        groove_score = 0.0
        
        # Moderate complexity is better for groove
        complexity_factor = rhythmic_analysis.complexity_level.value
        if complexity_factor == 2:  # MODERATE
            groove_score += 0.4
        elif complexity_factor in [1, 3]:  # MINIMAL or COMPLEX
            groove_score += 0.3
        else:
            groove_score += 0.1
        
        # Strong kick pattern helps groove
        groove_score += rhythmic_analysis.kick_pattern_strength * 0.3
        
        # Some syncopation adds interest
        optimal_syncopation = 0.3  # Sweet spot
        syncopation_quality = 1 - abs(rhythmic_analysis.syncopation_density - optimal_syncopation)
        groove_score += syncopation_quality * 0.3
        
        return min(1.0, groove_score)
    
    def _analyze_energy_builds(self, energy_curve: List[float]) -> float:
        """Analyze quality of energy builds"""
        if len(energy_curve) < 4:
            return 0.0
        
        builds = 0
        total_segments = 0
        
        # Look for ascending energy segments
        for i in range(len(energy_curve) - 3):
            segment = energy_curve[i:i+4]
            if all(segment[j] <= segment[j+1] for j in range(3)):
                builds += 1
            total_segments += 1
        
        return builds / max(1, total_segments)
    
    def _analyze_energy_drops(self, energy_curve: List[float]) -> float:
        """Analyze impact of energy drops"""
        if len(energy_curve) < 2:
            return 0.0
        
        max_drop = 0.0
        for i in range(len(energy_curve) - 1):
            drop = energy_curve[i] - energy_curve[i+1]
            if drop > max_drop:
                max_drop = drop
        
        # Normalize by maximum possible drop
        max_possible = max(energy_curve) if energy_curve else 1.0
        return min(1.0, max_drop / max_possible)
    
    def _extract_unique_pitches(self, pattern: HardcorePattern) -> Set[int]:
        """Extract set of unique pitches used"""
        pitches = set()
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    midi_note = int(69 + 12 * math.log2(step.params.freq / 440))
                    pitches.add(midi_note % 12)  # Pitch class
        return pitches
    
    def _extract_unique_features(self, pattern: HardcorePattern) -> Set[str]:
        """Extract unique features for pattern DNA"""
        features = set()
        
        # BPM category
        if pattern.bpm < 150:
            features.add("slow_tempo")
        elif pattern.bpm < 180:
            features.add("medium_tempo")
        elif pattern.bpm < 220:
            features.add("fast_tempo")
        else:
            features.add("extreme_tempo")
        
        # Track types
        for track_name in pattern.tracks.keys():
            features.add(f"has_{track_name}")
        
        # Synth types used
        synth_types = set()
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    synth_types.add(step.synth_type.value)
        
        for synth_type in synth_types:
            features.add(f"uses_{synth_type}")
        
        # Parameter characteristics
        total_crunch = sum(step.params.crunch for track in pattern.tracks.values() 
                          for step in track.steps if step is not None)
        step_count = sum(1 for track in pattern.tracks.values() 
                        for step in track.steps if step is not None)
        
        if step_count > 0:
            avg_crunch = total_crunch / step_count
            if avg_crunch > 0.8:
                features.add("high_crunch")
            elif avg_crunch > 0.5:
                features.add("medium_crunch")
            else:
                features.add("low_crunch")
        
        return features
    
    def _calculate_technical_quality(self, pattern: HardcorePattern) -> float:
        """Calculate technical quality score"""
        quality_score = 0.0
        param_count = 0
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    params = step.params
                    
                    # Check parameter ranges
                    if 20 <= params.freq <= 8000:
                        quality_score += 0.2
                    if 0.0 <= params.amp <= 1.0:
                        quality_score += 0.2
                    if 0.001 <= params.attack <= 2.0:
                        quality_score += 0.2
                    if params.cutoff > params.freq:
                        quality_score += 0.2
                    if 0.0 <= params.resonance <= 1.0:
                        quality_score += 0.2
                    
                    param_count += 1
        
        return quality_score / max(1, param_count)


# Test function
def test_theory_engine():
    """Test the mathematical theory engine"""
    print("🧮 Testing BMAD Mathematical Theory Engine")
    print("=" * 50)
    
    engine = BMADTheoryEngine()
    
    # Import pattern creation functions
    from ..models.hardcore_models import create_gabber_kick_pattern, create_acid_pattern
    
    # Test patterns
    test_patterns = [
        create_gabber_kick_pattern("Test Gabber", 180),
        create_acid_pattern("Test Acid", 175)
    ]
    
    for pattern in test_patterns:
        print(f"\n🎵 Analyzing: {pattern.name}")
        
        # Harmonic analysis
        harmonic = engine.analyze_harmonic_content(pattern)
        print(f"   Harmonic: Key stability={harmonic.key_stability:.3f}, "
              f"Chromaticism={harmonic.chromaticism:.3f}")
        
        # Rhythmic analysis
        rhythmic = engine.analyze_rhythmic_complexity(pattern)
        print(f"   Rhythmic: Complexity={rhythmic.complexity_level.name}, "
              f"Syncopation={rhythmic.syncopation_density:.3f}")
        
        # Energy analysis
        energy = engine.analyze_energy_curve(pattern)
        print(f"   Energy: Peak={energy.peak_energy:.3f}, "
              f"Warehouse compatibility={energy.warehouse_compatibility:.3f}")
        
        # Genre analysis
        genre = engine.analyze_genre_authenticity(pattern)
        print(f"   Genre: Gabber={genre.gabber_authenticity:.3f}, "
              f"Industrial={genre.industrial_characteristics:.3f}")
        
        # Extract DNA
        dna = engine.extract_pattern_dna(pattern)
        print(f"   DNA: Harmonic signature length={len(dna.harmonic_signature)}, "
              f"Features={len(dna.unique_features)}")
        
        # Calculate fitness
        fitness = engine.calculate_evolution_fitness(pattern)
        print(f"   Fitness: {', '.join(f'{k}={v:.3f}' for k, v in fitness.items())}")
    
    print("\n✅ Theory engine test completed!")


if __name__ == "__main__":
    test_theory_engine()