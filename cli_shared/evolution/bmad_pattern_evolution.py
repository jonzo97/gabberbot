#!/usr/bin/env python3
"""
BMAD Advanced Pattern Evolution Engine
@theory-engine (Cipher) - Intelligent Evolution for Hardcore Music

Advanced pattern evolution system that takes hardcore generation to the next level:
- Mathematical music theory constraints for intelligent mutations
- BPM progression ladders (180→200→220) with automatic adaptation
- Genre fusion algorithms (gabber + frenchcore + industrial)
- Multi-generational pattern families with mathematical progression tracking
- Energy curve optimization for warehouse sound systems
- Pattern uniqueness scoring to avoid repetition
"""

import numpy as np
import random
import asyncio
import time
import copy
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

from ..models.hardcore_models import (
    HardcorePattern, HardcoreTrack, PatternStep, SynthType, SynthParams,
    create_gabber_kick_pattern, create_acid_pattern, create_industrial_pattern
)
from .bmad_theory_engine import BMADTheoryEngine, PatternDNA, HarmonicAnalysis, RhythmicAnalysis


class EvolutionStrategy(Enum):
    """Evolution strategy types"""
    MUSICAL_PROGRESSION = "musical_progression"    # Follow music theory rules
    GENRE_FUSION = "genre_fusion"                  # Blend hardcore genres
    BPM_LADDER = "bpm_ladder"                      # Progressive BPM increases
    ENERGY_OPTIMIZATION = "energy_optimization"    # Optimize for warehouse systems
    CHAOS_MODE = "chaos_mode"                      # Maximum randomness
    INTELLIGENCE_HYBRID = "intelligence_hybrid"    # AI-guided evolution


class MutationConstraint(Enum):
    """Musical constraints for mutations"""
    STAY_IN_KEY = "stay_in_key"                   # Maintain key signature
    PRESERVE_GROOVE = "preserve_groove"           # Keep rhythmic groove
    MAINTAIN_ENERGY = "maintain_energy"           # Keep energy level
    GENRE_AUTHENTIC = "genre_authentic"           # Stay true to genre
    HARMONIC_PROGRESSION = "harmonic_progression" # Follow chord progressions
    RHYTHMIC_COHERENCE = "rhythmic_coherence"     # Maintain rhythmic logic


@dataclass
class EvolutionGeneration:
    """Single generation in pattern evolution"""
    generation_number: int
    patterns: List['EvolutionPattern']
    best_fitness: float
    average_fitness: float
    diversity_score: float
    bpm_range: Tuple[float, float]
    dominant_genre: str
    created_at: float = field(default_factory=time.time)


@dataclass
class EvolutionPattern:
    """Pattern with evolution metadata"""
    pattern: HardcorePattern
    pattern_dna: PatternDNA
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    total_fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    age: int = 0
    play_count: int = 0
    user_ratings: List[float] = field(default_factory=list)
    energy_classification: str = "unknown"
    genre_tags: Set[str] = field(default_factory=set)
    
    def calculate_total_fitness(self, weights: Dict[str, float]) -> float:
        """Calculate weighted total fitness"""
        total = 0.0
        for metric, weight in weights.items():
            score = self.fitness_scores.get(metric, 0.0)
            total += score * weight
        
        # Age bonus for survivors
        age_bonus = min(0.1, self.age * 0.01)
        total += age_bonus
        
        # User rating bonus
        if self.user_ratings:
            user_bonus = np.mean(self.user_ratings) * 0.1
            total += user_bonus
        
        self.total_fitness = total
        return total


@dataclass
class BMADEvolutionConfig:
    """Advanced evolution configuration"""
    population_size: int = 60
    elite_size: int = 15
    mutation_rate: float = 0.4
    crossover_rate: float = 0.8
    generations: int = 50
    
    # Evolution strategy
    strategy: EvolutionStrategy = EvolutionStrategy.MUSICAL_PROGRESSION
    constraints: List[MutationConstraint] = field(default_factory=lambda: [
        MutationConstraint.STAY_IN_KEY,
        MutationConstraint.PRESERVE_GROOVE
    ])
    
    # BPM progression settings
    bpm_start: float = 180.0
    bpm_end: float = 220.0
    bpm_progression_rate: float = 0.1  # BPM increase per generation
    
    # Genre fusion settings
    target_genres: List[str] = field(default_factory=lambda: ["gabber", "frenchcore"])
    fusion_rate: float = 0.3  # Rate of genre mixing
    
    # Fitness weights
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "authenticity": 0.25,
        "danceability": 0.20,
        "energy_level": 0.15,
        "rhythmic_complexity": 0.15,
        "harmonic_richness": 0.10,
        "technical_quality": 0.10,
        "novelty": 0.05
    })
    
    # Diversity preservation
    diversity_pressure: float = 0.3  # Pressure to maintain population diversity
    novelty_threshold: float = 0.7   # Minimum novelty for new patterns
    
    # Warehouse optimization
    optimize_for_warehouse: bool = True
    warehouse_energy_target: float = 0.8


class BMADPatternEvolution:
    """
    Advanced pattern evolution engine using mathematical music theory
    
    Features:
    - Intelligent mutations guided by music theory
    - BPM progression ladders with automatic adaptation
    - Genre fusion algorithms
    - Multi-generational pattern families
    - Energy curve optimization
    - Pattern uniqueness tracking
    """
    
    def __init__(self, config: BMADEvolutionConfig = None):
        self.config = config or BMADEvolutionConfig()
        self.theory_engine = BMADTheoryEngine()
        
        # Evolution state
        self.population: List[EvolutionPattern] = []
        self.generation = 0
        self.evolution_history: List[EvolutionGeneration] = []
        
        # Pattern database for novelty tracking
        self.pattern_database: Dict[str, PatternDNA] = {}
        self.pattern_families: Dict[str, List[str]] = {}  # Family trees
        
        # Genre templates for fusion
        self.genre_templates = {
            "gabber": {"bpm_range": (170, 200), "crunch_range": (0.7, 1.0), "kick_pattern": "4/4"},
            "frenchcore": {"bpm_range": (200, 250), "crunch_range": (0.8, 1.0), "kick_pattern": "complex"},
            "industrial": {"bpm_range": (130, 160), "crunch_range": (0.5, 0.8), "kick_pattern": "irregular"},
            "speedcore": {"bpm_range": (250, 400), "crunch_range": (0.9, 1.0), "kick_pattern": "blast"},
            "rawstyle": {"bpm_range": (150, 160), "crunch_range": (0.6, 0.9), "kick_pattern": "pitched"}
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_population(self, seed_patterns: List[HardcorePattern] = None) -> List[EvolutionPattern]:
        """Initialize population with intelligent seed patterns"""
        self.population = []
        
        # Create diverse seed patterns if none provided
        if not seed_patterns:
            seed_patterns = self._create_diverse_seeds()
        
        # Convert to evolution patterns
        for pattern in seed_patterns[:self.config.population_size]:
            evo_pattern = await self._create_evolution_pattern(pattern, generation=0)
            self.population.append(evo_pattern)
            
            # Add to pattern database
            self.pattern_database[pattern.name] = evo_pattern.pattern_dna
        
        # Fill remaining population with intelligent variations
        while len(self.population) < self.config.population_size:
            base_pattern = random.choice(seed_patterns)
            variation = await self._create_intelligent_variation(base_pattern)
            variation.pattern.name = f"gen0_var_{len(self.population)}"
            
            evo_pattern = await self._create_evolution_pattern(variation.pattern, generation=0)
            self.population.append(evo_pattern)
            
            self.pattern_database[variation.pattern.name] = evo_pattern.pattern_dna
        
        # Initial fitness evaluation
        await self._evaluate_population()
        
        # Record initial generation
        self._record_generation()
        
        return self.population
    
    async def evolve_generation(self) -> List[EvolutionPattern]:
        """Evolve one generation with intelligent algorithms"""
        self.logger.info(f"🧬 Evolving generation {self.generation}")
        
        # Update BPM progression if using BPM ladder strategy
        current_bpm_target = self._calculate_current_bpm_target()
        
        # Select parents using intelligent selection
        parents = self._intelligent_parent_selection()
        
        # Create next generation
        next_generation = []
        
        # Preserve elite patterns (but allow them to age)
        elite = sorted(self.population, key=lambda p: p.total_fitness, reverse=True)[:self.config.elite_size]
        for pattern in elite:
            elite_copy = copy.deepcopy(pattern)
            elite_copy.age += 1
            next_generation.append(elite_copy)
        
        # Generate offspring through intelligent crossover and mutation
        while len(next_generation) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Intelligent crossover
                parent1, parent2 = self._select_compatible_parents(parents)
                child1, child2 = await self._intelligent_crossover(parent1, parent2)
                
                # Intelligent mutation
                if random.random() < self.config.mutation_rate:
                    child1 = await self._intelligent_mutation(child1, current_bpm_target)
                
                if random.random() < self.config.mutation_rate:
                    child2 = await self._intelligent_mutation(child2, current_bpm_target)
                
                # Update generation info
                child1.generation = self.generation + 1
                child2.generation = self.generation + 1
                
                next_generation.extend([child1, child2])
            
            else:
                # Mutation-only reproduction
                parent = random.choice(parents)
                child = copy.deepcopy(parent)
                child = await self._intelligent_mutation(child, current_bpm_target)
                child.generation = self.generation + 1
                child.parent_ids = [parent.pattern.name]
                
                next_generation.append(child)
        
        # Limit to population size
        next_generation = next_generation[:self.config.population_size]
        
        # Apply diversity pressure
        next_generation = self._apply_diversity_pressure(next_generation)
        
        # Update population
        self.population = next_generation
        self.generation += 1
        
        # Evaluate new generation
        await self._evaluate_population()
        
        # Record generation history
        self._record_generation()
        
        # Update pattern database
        for pattern in self.population:
            self.pattern_database[pattern.pattern.name] = pattern.pattern_dna
        
        return self.population
    
    async def _create_evolution_pattern(self, pattern: HardcorePattern, generation: int) -> EvolutionPattern:
        """Create evolution pattern from hardcore pattern"""
        # Extract pattern DNA
        pattern_dna = self.theory_engine.extract_pattern_dna(pattern)
        
        # Calculate fitness scores
        fitness_scores = self.theory_engine.calculate_evolution_fitness(
            pattern, target_style=self.config.target_genres[0] if self.config.target_genres else "gabber"
        )
        
        # Classify energy and genre
        energy_analysis = self.theory_engine.analyze_energy_curve(pattern)
        genre_analysis = self.theory_engine.analyze_genre_authenticity(pattern)
        
        energy_class = "high" if energy_analysis.average_energy > 2.0 else "medium" if energy_analysis.average_energy > 1.0 else "low"
        genre_tags = set()
        
        if genre_analysis.gabber_authenticity > 0.6:
            genre_tags.add("gabber")
        if genre_analysis.frenchcore_elements > 0.6:
            genre_tags.add("frenchcore")
        if genre_analysis.industrial_characteristics > 0.6:
            genre_tags.add("industrial")
        
        evo_pattern = EvolutionPattern(
            pattern=pattern,
            pattern_dna=pattern_dna,
            fitness_scores=fitness_scores,
            generation=generation,
            energy_classification=energy_class,
            genre_tags=genre_tags
        )
        
        # Calculate total fitness
        evo_pattern.calculate_total_fitness(self.config.fitness_weights)
        
        return evo_pattern
    
    def _create_diverse_seeds(self) -> List[HardcorePattern]:
        """Create diverse seed patterns covering different styles"""
        seeds = []
        
        # Classic patterns
        seeds.append(create_gabber_kick_pattern("Seed_Gabber_180", 180))
        seeds.append(create_acid_pattern("Seed_Acid_175", 175))
        seeds.append(create_industrial_pattern("Seed_Industrial_140", 140))
        
        # Variations with different BPMs
        seeds.append(create_gabber_kick_pattern("Seed_Gabber_200", 200))
        seeds.append(create_gabber_kick_pattern("Seed_Gabber_160", 160))
        
        # Create frenchcore-style pattern
        frenchcore = create_gabber_kick_pattern("Seed_Frenchcore_220", 220)
        # Add complexity to make it more frenchcore-like
        frenchcore.add_track("lead")
        lead_params = SynthParams(freq=400, crunch=0.9, drive=3.0, cutoff=8000)
        lead_step = PatternStep(SynthType.SCREECH_LEAD, lead_params, velocity=0.8)
        for step in [2, 6, 10, 14]:  # Offbeat lead
            frenchcore.set_step("lead", step, lead_step)
        seeds.append(frenchcore)
        
        return seeds
    
    async def _create_intelligent_variation(self, base_pattern: HardcorePattern) -> EvolutionPattern:
        """Create intelligent variation of base pattern"""
        variation = copy.deepcopy(base_pattern)
        variation.name = f"{base_pattern.name}_variation"
        
        # Apply small intelligent mutations
        mutation_types = ["adjust_bpm", "modify_crunch", "add_syncopation", "adjust_frequencies"]
        mutation = random.choice(mutation_types)
        
        if mutation == "adjust_bpm":
            variation.bpm *= random.uniform(0.95, 1.05)
        elif mutation == "modify_crunch":
            for track in variation.tracks.values():
                for step in track.steps:
                    if step is not None:
                        step.params.crunch = max(0, min(1, step.params.crunch + random.uniform(-0.1, 0.1)))
        elif mutation == "add_syncopation":
            self._add_syncopation_to_pattern(variation)
        elif mutation == "adjust_frequencies":
            self._adjust_frequencies_musically(variation)
        
        return await self._create_evolution_pattern(variation, generation=0)
    
    def _calculate_current_bpm_target(self) -> float:
        """Calculate current BPM target based on generation and strategy"""
        if self.config.strategy == EvolutionStrategy.BPM_LADDER:
            progress = self.generation / self.config.generations
            return self.config.bpm_start + (self.config.bpm_end - self.config.bpm_start) * progress
        return self.config.bpm_start
    
    def _intelligent_parent_selection(self) -> List[EvolutionPattern]:
        """Select parents using intelligent criteria"""
        parents = []
        
        # Always include top performers
        sorted_pop = sorted(self.population, key=lambda p: p.total_fitness, reverse=True)
        parents.extend(sorted_pop[:self.config.elite_size])
        
        # Add diverse patterns to maintain genetic diversity
        remaining = [p for p in sorted_pop[self.config.elite_size:]]
        
        # Select for diversity
        for _ in range(self.config.population_size - len(parents)):
            if remaining:
                # Prefer patterns that are different from already selected
                best_candidate = None
                best_diversity = -1
                
                for candidate in remaining[:20]:  # Check top 20 remaining
                    diversity = self._calculate_pattern_diversity(candidate, parents)
                    if diversity > best_diversity:
                        best_diversity = diversity
                        best_candidate = candidate
                
                if best_candidate:
                    parents.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    parents.append(random.choice(remaining))
        
        return parents
    
    def _select_compatible_parents(self, parents: List[EvolutionPattern]) -> Tuple[EvolutionPattern, EvolutionPattern]:
        """Select compatible parents for crossover"""
        # Prefer parents from similar BPM ranges or complementary genres
        attempts = 0
        while attempts < 10:
            parent1, parent2 = random.sample(parents, 2)
            
            # Check BPM compatibility
            bpm_diff = abs(parent1.pattern.bpm - parent2.pattern.bpm)
            if bpm_diff < 30:  # Similar BPM
                return parent1, parent2
            
            # Check genre compatibility
            if parent1.genre_tags.intersection(parent2.genre_tags):
                return parent1, parent2
            
            attempts += 1
        
        # Fall back to random selection
        return random.sample(parents, 2)
    
    async def _intelligent_crossover(self, parent1: EvolutionPattern, parent2: EvolutionPattern) -> Tuple[EvolutionPattern, EvolutionPattern]:
        """Perform intelligent crossover between parents"""
        p1_pattern = copy.deepcopy(parent1.pattern)
        p2_pattern = copy.deepcopy(parent2.pattern)
        
        # Create child patterns
        child1_pattern = copy.deepcopy(p1_pattern)
        child2_pattern = copy.deepcopy(p2_pattern)
        
        child1_pattern.name = f"gen{self.generation+1}_cross_{int(time.time())}_1"
        child2_pattern.name = f"gen{self.generation+1}_cross_{int(time.time())}_2"
        
        # Intelligent crossover strategies
        crossover_type = random.choice(["harmonic_crossover", "rhythmic_crossover", "energy_crossover", "track_wise"])
        
        if crossover_type == "harmonic_crossover":
            self._harmonic_crossover(child1_pattern, child2_pattern, p1_pattern, p2_pattern)
        elif crossover_type == "rhythmic_crossover":
            self._rhythmic_crossover(child1_pattern, child2_pattern, p1_pattern, p2_pattern)
        elif crossover_type == "energy_crossover":
            self._energy_crossover(child1_pattern, child2_pattern, p1_pattern, p2_pattern)
        else:
            self._track_wise_crossover(child1_pattern, child2_pattern, p1_pattern, p2_pattern)
        
        # Create evolution patterns
        child1 = await self._create_evolution_pattern(child1_pattern, self.generation + 1)
        child2 = await self._create_evolution_pattern(child2_pattern, self.generation + 1)
        
        child1.parent_ids = [parent1.pattern.name, parent2.pattern.name]
        child2.parent_ids = [parent1.pattern.name, parent2.pattern.name]
        
        return child1, child2
    
    async def _intelligent_mutation(self, pattern: EvolutionPattern, bpm_target: float) -> EvolutionPattern:
        """Apply intelligent mutations based on constraints and strategy"""
        mutated_pattern = copy.deepcopy(pattern.pattern)
        mutated_pattern.name = f"gen{self.generation+1}_mut_{int(time.time())}"
        
        mutations_applied = []
        
        # Strategy-based mutations
        if self.config.strategy == EvolutionStrategy.BPM_LADDER:
            if abs(mutated_pattern.bpm - bpm_target) > 5:
                # Gradually move BPM toward target
                if mutated_pattern.bpm < bpm_target:
                    mutated_pattern.bpm += random.uniform(2, 8)
                else:
                    mutated_pattern.bpm -= random.uniform(2, 8)
                mutations_applied.append("bpm_progression")
        
        elif self.config.strategy == EvolutionStrategy.GENRE_FUSION:
            self._apply_genre_fusion_mutation(mutated_pattern)
            mutations_applied.append("genre_fusion")
        
        elif self.config.strategy == EvolutionStrategy.ENERGY_OPTIMIZATION:
            self._apply_energy_optimization_mutation(mutated_pattern)
            mutations_applied.append("energy_optimization")
        
        # Constraint-respecting mutations
        for constraint in self.config.constraints:
            if constraint == MutationConstraint.STAY_IN_KEY:
                self._apply_key_respecting_mutation(mutated_pattern)
                mutations_applied.append("key_respecting")
            elif constraint == MutationConstraint.PRESERVE_GROOVE:
                self._apply_groove_preserving_mutation(mutated_pattern)
                mutations_applied.append("groove_preserving")
        
        # General intelligent mutations
        mutation_types = [
            "harmonic_progression", "rhythmic_complexity", 
            "energy_boost", "parameter_evolution"
        ]
        
        for _ in range(random.randint(1, 3)):
            mutation = random.choice(mutation_types)
            if mutation == "harmonic_progression":
                self._apply_harmonic_progression_mutation(mutated_pattern)
            elif mutation == "rhythmic_complexity":
                self._apply_rhythmic_complexity_mutation(mutated_pattern)
            elif mutation == "energy_boost":
                self._apply_energy_boost_mutation(mutated_pattern)
            elif mutation == "parameter_evolution":
                self._apply_parameter_evolution_mutation(mutated_pattern)
            
            mutations_applied.append(mutation)
        
        # Create new evolution pattern
        new_evo_pattern = await self._create_evolution_pattern(mutated_pattern, self.generation + 1)
        new_evo_pattern.parent_ids = [pattern.pattern.name]
        new_evo_pattern.mutation_history = pattern.mutation_history + mutations_applied
        
        return new_evo_pattern
    
    # Crossover methods
    def _harmonic_crossover(self, child1: HardcorePattern, child2: HardcorePattern, 
                           parent1: HardcorePattern, parent2: HardcorePattern):
        """Crossover based on harmonic content"""
        # Exchange harmonic tracks (bass, lead, etc.)
        harmonic_tracks = ["bass", "acid_bass", "lead", "stab"]
        
        for track_name in harmonic_tracks:
            if track_name in parent1.tracks and track_name in parent2.tracks:
                if random.random() < 0.5:
                    # Swap harmonic tracks
                    child1.tracks[track_name] = copy.deepcopy(parent2.tracks[track_name])
                    child2.tracks[track_name] = copy.deepcopy(parent1.tracks[track_name])
    
    def _rhythmic_crossover(self, child1: HardcorePattern, child2: HardcorePattern,
                           parent1: HardcorePattern, parent2: HardcorePattern):
        """Crossover based on rhythmic patterns"""
        # Exchange rhythmic elements
        rhythmic_tracks = ["kick", "snare", "hihat", "perc"]
        
        for track_name in rhythmic_tracks:
            if track_name in parent1.tracks and track_name in parent2.tracks:
                if random.random() < 0.5:
                    child1.tracks[track_name] = copy.deepcopy(parent2.tracks[track_name])
                    child2.tracks[track_name] = copy.deepcopy(parent1.tracks[track_name])
    
    def _energy_crossover(self, child1: HardcorePattern, child2: HardcorePattern,
                         parent1: HardcorePattern, parent2: HardcorePattern):
        """Crossover based on energy characteristics"""
        # Mix high and low energy elements
        energy1 = self.theory_engine.analyze_energy_curve(parent1)
        energy2 = self.theory_engine.analyze_energy_curve(parent2)
        
        if energy1.average_energy > energy2.average_energy:
            high_energy_parent = parent1
            low_energy_parent = parent2
        else:
            high_energy_parent = parent2
            low_energy_parent = parent1
        
        # Child1 gets high energy elements, Child2 gets low energy elements
        for track_name, track in high_energy_parent.tracks.items():
            if track_name in child1.tracks:
                # Copy high energy steps
                for i, step in enumerate(track.steps):
                    if step is not None and step.params.amp > 0.7:
                        child1.set_step(track_name, i, copy.deepcopy(step))
    
    def _track_wise_crossover(self, child1: HardcorePattern, child2: HardcorePattern,
                             parent1: HardcorePattern, parent2: HardcorePattern):
        """Simple track-wise crossover"""
        all_tracks = set(parent1.tracks.keys()) | set(parent2.tracks.keys())
        
        for track_name in all_tracks:
            if random.random() < 0.5 and track_name in parent2.tracks:
                child1.tracks[track_name] = copy.deepcopy(parent2.tracks[track_name])
            if random.random() < 0.5 and track_name in parent1.tracks:
                child2.tracks[track_name] = copy.deepcopy(parent1.tracks[track_name])
    
    # Mutation methods
    def _apply_genre_fusion_mutation(self, pattern: HardcorePattern):
        """Apply genre fusion mutations"""
        target_genres = self.config.target_genres
        if len(target_genres) < 2:
            return
        
        genre1, genre2 = random.sample(target_genres, 2)
        template1 = self.genre_templates.get(genre1, {})
        template2 = self.genre_templates.get(genre2, {})
        
        # Blend BPM ranges
        if "bpm_range" in template1 and "bpm_range" in template2:
            bpm1_range = template1["bpm_range"]
            bpm2_range = template2["bpm_range"]
            fusion_bpm = (bpm1_range[0] + bpm2_range[0]) / 2
            pattern.bpm = fusion_bpm + random.uniform(-10, 10)
        
        # Blend crunch characteristics
        if "crunch_range" in template1 and "crunch_range" in template2:
            crunch1_range = template1["crunch_range"]
            crunch2_range = template2["crunch_range"]
            fusion_crunch = (crunch1_range[0] + crunch2_range[0]) / 2
            
            for track in pattern.tracks.values():
                for step in track.steps:
                    if step is not None:
                        step.params.crunch = fusion_crunch + random.uniform(-0.1, 0.1)
    
    def _apply_energy_optimization_mutation(self, pattern: HardcorePattern):
        """Optimize pattern for warehouse energy"""
        target_energy = self.config.warehouse_energy_target
        
        # Boost energy parameters
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    # Increase amplitude for low-frequency elements
                    if step.params.freq < 100:
                        step.params.amp = min(1.0, step.params.amp * 1.2)
                    
                    # Optimize crunch for energy
                    step.params.crunch = min(1.0, step.params.crunch + 0.1)
    
    def _apply_key_respecting_mutation(self, pattern: HardcorePattern):
        """Apply mutations that respect musical key"""
        # Define scale intervals for common hardcore keys
        minor_scale = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None and step.params.freq > 100:  # Melodic elements
                    # Convert frequency to MIDI note
                    midi_note = int(69 + 12 * math.log2(step.params.freq / 440))
                    pitch_class = midi_note % 12
                    
                    # If not in scale, move to nearest scale tone
                    if pitch_class not in minor_scale:
                        distances = [abs(pitch_class - scale_tone) for scale_tone in minor_scale]
                        nearest_scale_tone = minor_scale[np.argmin(distances)]
                        
                        # Adjust frequency
                        semitone_diff = nearest_scale_tone - pitch_class
                        step.params.freq *= 2 ** (semitone_diff / 12)
    
    def _apply_groove_preserving_mutation(self, pattern: HardcorePattern):
        """Apply mutations that preserve rhythmic groove"""
        # Identify kick pattern and preserve it
        kick_tracks = [name for name in pattern.tracks.keys() if "kick" in name.lower()]
        
        for kick_track_name in kick_tracks:
            track = pattern.tracks[kick_track_name]
            # Preserve strong beat kicks (0, 4, 8, 12)
            strong_beats = [0, 4, 8, 12]
            for beat in strong_beats:
                if beat < len(track.steps) and track.steps[beat] is None:
                    # Add kick on strong beat if missing
                    kick_params = SynthParams(
                        freq=60, amp=0.9, crunch=0.8, drive=2.0,
                        attack=0.001, decay=0.08, sustain=0.3, release=0.2
                    )
                    kick_step = PatternStep(SynthType.GABBER_KICK, kick_params)
                    track.steps[beat] = kick_step
    
    def _apply_harmonic_progression_mutation(self, pattern: HardcorePattern):
        """Apply harmonic progression mutations"""
        harmonic_tracks = ["bass", "acid_bass", "lead"]
        
        for track_name in harmonic_tracks:
            if track_name in pattern.tracks:
                track = pattern.tracks[track_name]
                
                # Create simple harmonic progression
                progression_freqs = [110, 130, 146, 110]  # i - III - iv - i in A minor
                
                for i, step in enumerate(track.steps):
                    if step is not None:
                        prog_index = (i // 4) % len(progression_freqs)
                        target_freq = progression_freqs[prog_index]
                        # Gradually move toward target frequency
                        step.params.freq = step.params.freq * 0.8 + target_freq * 0.2
    
    def _apply_rhythmic_complexity_mutation(self, pattern: HardcorePattern):
        """Add rhythmic complexity"""
        # Add syncopation or polyrhythmic elements
        if random.random() < 0.5:
            self._add_syncopation_to_pattern(pattern)
        else:
            self._add_polyrhythm_to_pattern(pattern)
    
    def _apply_energy_boost_mutation(self, pattern: HardcorePattern):
        """Boost energy characteristics"""
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    # Boost amplitude and drive
                    step.params.amp = min(1.0, step.params.amp * 1.1)
                    step.params.drive = min(10.0, step.params.drive * 1.1)
                    step.velocity = min(1.0, step.velocity * 1.05)
    
    def _apply_parameter_evolution_mutation(self, pattern: HardcorePattern):
        """Evolve synthesis parameters"""
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    # Randomly evolve parameters
                    step.params.cutoff *= random.uniform(0.9, 1.1)
                    step.params.resonance = max(0, min(1, step.params.resonance + random.uniform(-0.1, 0.1)))
                    step.params.crunch = max(0, min(1, step.params.crunch + random.uniform(-0.05, 0.05)))
    
    # Helper methods
    def _add_syncopation_to_pattern(self, pattern: HardcorePattern):
        """Add syncopated elements to pattern"""
        syncopated_positions = [1, 3, 5, 7, 9, 11, 13, 15]  # 16th note offbeats
        
        for track_name, track in pattern.tracks.items():
            if "kick" not in track_name.lower():  # Don't syncopate kick
                for pos in syncopated_positions:
                    if pos < len(track.steps) and track.steps[pos] is None and random.random() < 0.3:
                        # Add syncopated hit
                        if track.steps:
                            # Copy from nearby step
                            source_step = None
                            for i in range(max(0, pos-2), min(len(track.steps), pos+3)):
                                if track.steps[i] is not None:
                                    source_step = track.steps[i]
                                    break
                            
                            if source_step:
                                synco_step = copy.deepcopy(source_step)
                                synco_step.velocity *= 0.7  # Quieter syncopation
                                track.steps[pos] = synco_step
    
    def _add_polyrhythm_to_pattern(self, pattern: HardcorePattern):
        """Add polyrhythmic elements"""
        # Add a track with different rhythmic cycle
        if "polyrhythm" not in pattern.tracks:
            pattern.add_track("polyrhythm")
            
            # Create 3-against-4 polyrhythm
            poly_positions = [0, 5, 10]  # Positions for 3-beat cycle in 16-step pattern
            
            poly_params = SynthParams(
                freq=800, amp=0.5, crunch=0.6, drive=2.0,
                attack=0.001, decay=0.05, sustain=0.1, release=0.1
            )
            poly_step = PatternStep(SynthType.INDUSTRIAL_NOISE, poly_params, velocity=0.6)
            
            for pos in poly_positions:
                pattern.set_step("polyrhythm", pos, poly_step)
    
    def _adjust_frequencies_musically(self, pattern: HardcorePattern):
        """Adjust frequencies following musical intervals"""
        perfect_fifth = 2 ** (7/12)  # Frequency ratio for perfect fifth
        octave = 2.0
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None and step.params.freq > 100:  # Melodic content
                    if random.random() < 0.3:
                        # Move by perfect fifth
                        step.params.freq *= perfect_fifth
                    elif random.random() < 0.2:
                        # Move by octave
                        step.params.freq *= octave if random.random() < 0.5 else 0.5
    
    def _calculate_pattern_diversity(self, pattern: EvolutionPattern, population: List[EvolutionPattern]) -> float:
        """Calculate how diverse a pattern is compared to population"""
        if not population:
            return 1.0
        
        similarities = []
        for other in population[-20:]:  # Compare to last 20 patterns
            similarity = pattern.pattern_dna.similarity_to(other.pattern_dna)
            similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def _apply_diversity_pressure(self, population: List[EvolutionPattern]) -> List[EvolutionPattern]:
        """Apply pressure to maintain population diversity"""
        if self.config.diversity_pressure <= 0:
            return population
        
        # Calculate diversity scores
        for pattern in population:
            pattern.diversity_score = self._calculate_pattern_diversity(pattern, population)
        
        # Sort by combination of fitness and diversity
        def combined_score(pattern):
            return (pattern.total_fitness * (1 - self.config.diversity_pressure) + 
                   pattern.diversity_score * self.config.diversity_pressure)
        
        population.sort(key=combined_score, reverse=True)
        return population
    
    def _record_generation(self):
        """Record current generation in evolution history"""
        if not self.population:
            return
        
        best_fitness = max(p.total_fitness for p in self.population)
        avg_fitness = np.mean([p.total_fitness for p in self.population])
        diversity = np.mean([self._calculate_pattern_diversity(p, self.population) for p in self.population])
        
        bpm_values = [p.pattern.bpm for p in self.population]
        bpm_range = (min(bpm_values), max(bpm_values))
        
        # Determine dominant genre
        all_genres = set()
        for pattern in self.population:
            all_genres.update(pattern.genre_tags)
        
        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = sum(1 for p in self.population if genre in p.genre_tags)
        
        dominant_genre = max(genre_counts.items(), key=lambda x: x[1])[0] if genre_counts else "unknown"
        
        generation = EvolutionGeneration(
            generation_number=self.generation,
            patterns=copy.deepcopy(self.population),
            best_fitness=best_fitness,
            average_fitness=avg_fitness,
            diversity_score=diversity,
            bpm_range=bpm_range,
            dominant_genre=dominant_genre
        )
        
        self.evolution_history.append(generation)
    
    async def _evaluate_population(self):
        """Evaluate fitness for entire population"""
        for pattern in self.population:
            pattern.calculate_total_fitness(self.config.fitness_weights)
    
    def get_best_patterns(self, n: int = 10) -> List[EvolutionPattern]:
        """Get top N patterns from current population"""
        return sorted(self.population, key=lambda p: p.total_fitness, reverse=True)[:n]
    
    def get_pattern_families(self) -> Dict[str, List[str]]:
        """Get pattern family trees"""
        families = {}
        
        for pattern in self.population:
            if pattern.parent_ids:
                parent_id = pattern.parent_ids[0]  # Use first parent as family root
                if parent_id not in families:
                    families[parent_id] = []
                families[parent_id].append(pattern.pattern.name)
        
        return families
    
    def save_evolution_state(self, filepath: str):
        """Save complete evolution state"""
        state = {
            "generation": self.generation,
            "config": {
                "population_size": self.config.population_size,
                "strategy": self.config.strategy.value,
                "bpm_start": self.config.bpm_start,
                "bpm_end": self.config.bpm_end,
                "target_genres": self.config.target_genres
            },
            "population": [
                {
                    "pattern": pattern.pattern.to_dict(),
                    "fitness_scores": pattern.fitness_scores,
                    "total_fitness": pattern.total_fitness,
                    "generation": pattern.generation,
                    "parent_ids": pattern.parent_ids,
                    "mutation_history": pattern.mutation_history,
                    "energy_classification": pattern.energy_classification,
                    "genre_tags": list(pattern.genre_tags)
                }
                for pattern in self.population
            ],
            "evolution_history": [
                {
                    "generation_number": gen.generation_number,
                    "best_fitness": gen.best_fitness,
                    "average_fitness": gen.average_fitness,
                    "diversity_score": gen.diversity_score,
                    "bpm_range": gen.bmp_range,
                    "dominant_genre": gen.dominant_genre,
                    "created_at": gen.created_at
                }
                for gen in self.evolution_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)


# Test function
async def test_bmad_evolution():
    """Test the BMAD pattern evolution engine"""
    print("🧬 Testing BMAD Advanced Pattern Evolution")
    print("=" * 60)
    
    # Create evolution config
    config = BMADEvolutionConfig(
        population_size=20,
        generations=5,
        strategy=EvolutionStrategy.MUSICAL_PROGRESSION,
        bpm_start=180,
        bpm_end=200,
        target_genres=["gabber", "frenchcore"]
    )
    
    engine = BMADPatternEvolution(config)
    
    print(f"🌱 Initializing population...")
    population = await engine.initialize_population()
    
    print(f"   Population size: {len(population)}")
    print(f"   Strategy: {config.strategy.value}")
    print(f"   BPM progression: {config.bmp_start} → {config.bmp_end}")
    
    # Evolve for several generations
    print(f"\n🔄 Evolving for {config.generations} generations...")
    
    for gen in range(config.generations):
        population = await engine.evolve_generation()
        
        best_pattern = max(population, key=lambda p: p.total_fitness)
        avg_fitness = np.mean([p.total_fitness for p in population])
        bpm_range = (min(p.pattern.bpm for p in population), max(p.pattern.bpm for p in population))
        
        print(f"   Gen {engine.generation}: Best={best_pattern.total_fitness:.3f}, "
              f"Avg={avg_fitness:.3f}, BPM={bpm_range[0]:.0f}-{bpm_range[1]:.0f}")
    
    # Show evolution results
    print(f"\n🏆 Evolution Results:")
    best_patterns = engine.get_best_patterns(5)
    
    for i, pattern in enumerate(best_patterns):
        print(f"   {i+1}. {pattern.pattern.name}")
        print(f"      BPM: {pattern.pattern.bpm:.1f}, Fitness: {pattern.total_fitness:.3f}")
        print(f"      Generation: {pattern.generation}, Energy: {pattern.energy_classification}")
        print(f"      Genres: {', '.join(pattern.genre_tags) if pattern.genre_tags else 'None'}")
        print(f"      Mutations: {', '.join(pattern.mutation_history[-3:])}")  # Last 3 mutations
    
    # Show pattern families
    print(f"\n🌳 Pattern Families:")
    families = engine.get_pattern_families()
    for parent, children in families.items():
        if children:
            print(f"   {parent} → {', '.join(children[:3])}{'...' if len(children) > 3 else ''}")
    
    print(f"\n🧬 Evolution completed successfully!")
    print(f"   Final generation: {engine.generation}")
    print(f"   Total patterns created: {len(engine.pattern_database)}")
    
    return engine


if __name__ == "__main__":
    asyncio.run(test_bmad_evolution())