#!/usr/bin/env python3
"""
Pattern Evolution Engine with Genetic Algorithms for Hardcore Music

This engine evolves hardcore patterns using genetic algorithms:
- Crossover breeding between patterns
- Mutation operators for rhythm, harmony, and structure
- Fitness scoring based on hardcore authenticity and danceability
- Population diversity management
- Interactive evolution with human feedback
- Pattern DNA tracking and genealogy
"""

import numpy as np
import random
import asyncio
import time
import copy
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

# Import shared models
import sys
sys.path.append('/home/onathan_rgill/music_code_cli')

from cli_shared import (
    HardcorePattern, HardcoreTrack, PatternStep, SynthType, SynthParams
)

class MutationType(Enum):
    """Types of mutations that can be applied to patterns"""
    ADD_STEP = "add_step"                    # Add a step to a track
    REMOVE_STEP = "remove_step"              # Remove a step from a track
    MODIFY_VELOCITY = "modify_velocity"       # Change step velocity
    MODIFY_PARAMS = "modify_params"          # Modify synthesis parameters
    SHIFT_PATTERN = "shift_pattern"          # Shift pattern timing
    ADD_TRACK = "add_track"                  # Add new track
    REMOVE_TRACK = "remove_track"            # Remove existing track
    CLONE_STEP = "clone_step"                # Duplicate step elsewhere
    RANDOMIZE_STEP = "randomize_step"        # Completely randomize step
    EVOLVE_CRUNCH = "evolve_crunch"          # Evolve crunch parameters
    EVOLVE_BPM = "evolve_bpm"                # Mutate tempo
    EVOLVE_SWING = "evolve_swing"            # Mutate swing/groove

class CrossoverType(Enum):
    """Types of crossover operations for breeding patterns"""
    SINGLE_POINT = "single_point"            # Single crossover point
    TWO_POINT = "two_point"                  # Two crossover points
    UNIFORM = "uniform"                      # Uniform crossover
    TRACK_WISE = "track_wise"                # Crossover entire tracks
    PARAMETER_WISE = "parameter_wise"        # Crossover synthesis parameters
    RHYTHM_CROSSOVER = "rhythm_crossover"    # Crossover rhythm patterns
    HARMONIC_CROSSOVER = "harmonic_crossover" # Crossover harmonic content

class FitnessMetric(Enum):
    """Fitness metrics for evaluating patterns"""
    HARDCORE_AUTHENTICITY = "hardcore_authenticity"  # How "hardcore" it sounds
    DANCEABILITY = "danceability"                    # Dance floor effectiveness
    RHYTHMIC_COMPLEXITY = "rhythmic_complexity"      # Rhythm intricacy
    HARMONIC_RICHNESS = "harmonic_richness"         # Harmonic content
    ENERGY_LEVEL = "energy_level"                   # Overall energy
    NOVELTY = "novelty"                             # Uniqueness
    TECHNICAL_QUALITY = "technical_quality"          # Audio/production quality
    USER_RATING = "user_rating"                     # Human feedback

@dataclass
class EvolutionConfig:
    """Configuration for genetic algorithm evolution"""
    population_size: int = 50                # Number of patterns in population
    elite_size: int = 10                     # Number of elite patterns to preserve
    mutation_rate: float = 0.3               # Probability of mutation
    crossover_rate: float = 0.7              # Probability of crossover
    generations: int = 100                   # Maximum generations to evolve
    
    # Fitness weights
    fitness_weights: Dict[FitnessMetric, float] = field(default_factory=lambda: {
        FitnessMetric.HARDCORE_AUTHENTICITY: 0.25,
        FitnessMetric.DANCEABILITY: 0.20,
        FitnessMetric.ENERGY_LEVEL: 0.15,
        FitnessMetric.RHYTHMIC_COMPLEXITY: 0.15,
        FitnessMetric.HARMONIC_RICHNESS: 0.10,
        FitnessMetric.NOVELTY: 0.10,
        FitnessMetric.TECHNICAL_QUALITY: 0.05
    })
    
    # Mutation parameters
    mutation_strength: float = 0.5           # Strength of mutations
    preserve_structure: bool = True          # Preserve basic pattern structure
    allow_tempo_mutation: bool = True        # Allow BPM changes
    
    # Selection parameters
    tournament_size: int = 5                 # Tournament selection size
    selection_pressure: float = 1.5         # Selection pressure multiplier

@dataclass
class PatternGenome:
    """Genome representation of a hardcore pattern"""
    pattern: HardcorePattern
    fitness_scores: Dict[FitnessMetric, float] = field(default_factory=dict)
    total_fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations_applied: List[MutationType] = field(default_factory=list)
    age: int = 0                             # How many generations this genome survived
    play_count: int = 0                      # How many times it was played
    user_ratings: List[float] = field(default_factory=list)
    
    def calculate_total_fitness(self, config: EvolutionConfig) -> float:
        """Calculate total fitness based on weighted metrics"""
        total = 0.0
        for metric, weight in config.fitness_weights.items():
            score = self.fitness_scores.get(metric, 0.0)
            total += score * weight
        
        # Bonus for user ratings
        if self.user_ratings:
            user_bonus = np.mean(self.user_ratings) * 0.1
            total += user_bonus
        
        # Age bonus (older patterns that survive are likely good)
        age_bonus = min(0.1, self.age * 0.01)
        total += age_bonus
        
        self.total_fitness = total
        return total

class PatternEvolutionEngine:
    """
    Genetic algorithm engine for evolving hardcore patterns
    
    Features:
    - Multi-objective fitness evaluation
    - Diverse mutation operators
    - Advanced crossover strategies
    - Population diversity management
    - Interactive evolution with human feedback
    - Pattern genealogy tracking
    """
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.population: List[PatternGenome] = []
        self.generation = 0
        self.evolution_history = []
        
        # Pattern database for novelty calculation
        self.pattern_database = {}
        
        # User feedback system
        self.user_feedback_callback: Optional[Callable[[HardcorePattern], float]] = None
        
        # Fitness evaluators
        self.fitness_evaluators = {
            FitnessMetric.HARDCORE_AUTHENTICITY: self._evaluate_hardcore_authenticity,
            FitnessMetric.DANCEABILITY: self._evaluate_danceability,
            FitnessMetric.RHYTHMIC_COMPLEXITY: self._evaluate_rhythmic_complexity,
            FitnessMetric.HARMONIC_RICHNESS: self._evaluate_harmonic_richness,
            FitnessMetric.ENERGY_LEVEL: self._evaluate_energy_level,
            FitnessMetric.NOVELTY: self._evaluate_novelty,
            FitnessMetric.TECHNICAL_QUALITY: self._evaluate_technical_quality
        }
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_population(self, seed_patterns: List[HardcorePattern] = None) -> List[PatternGenome]:
        """Initialize population with seed patterns or random generation"""
        self.population = []
        
        if seed_patterns:
            # Use provided seed patterns
            for pattern in seed_patterns[:self.config.population_size]:
                genome = PatternGenome(
                    pattern=pattern,
                    generation=0
                )
                self.population.append(genome)
        
        # Fill remaining population with variations of seed patterns or random
        while len(self.population) < self.config.population_size:
            if seed_patterns:
                # Create variation of random seed pattern
                base_pattern = random.choice(seed_patterns)
                mutated_pattern = self._mutate_pattern(copy.deepcopy(base_pattern))
                mutated_pattern.name = f"gen0_var_{len(self.population)}"
            else:
                # Generate completely random pattern
                mutated_pattern = self._generate_random_pattern(f"gen0_random_{len(self.population)}")
            
            genome = PatternGenome(
                pattern=mutated_pattern,
                generation=0
            )
            self.population.append(genome)
        
        # Evaluate initial population
        asyncio.create_task(self._evaluate_population())
        
        return self.population
    
    def _generate_random_pattern(self, name: str) -> HardcorePattern:
        """Generate a random hardcore pattern"""
        bpm = random.uniform(160, 220)  # Hardcore BPM range
        steps = random.choice([16, 32])  # Pattern length
        
        pattern = HardcorePattern(name=name, bpm=bpm, steps=steps)
        
        # Add random tracks
        track_types = [
            ("kick", SynthType.GABBER_KICK),
            ("bass", SynthType.ACID_BASS),
            ("lead", random.choice([SynthType.HOOVER_SYNTH, SynthType.SCREECH_LEAD])),
            ("stab", SynthType.HARDCORE_STAB)
        ]
        
        for track_name, synth_type in track_types:
            if random.random() < 0.8:  # 80% chance to include each track
                pattern.add_track(track_name)
                
                # Add random steps
                for step in range(steps):
                    if random.random() < self._get_step_probability(track_name, step):
                        params = self._generate_random_synth_params(synth_type)
                        step_obj = PatternStep(
                            synth_type=synth_type,
                            params=params,
                            velocity=random.uniform(0.6, 1.0),
                            probability=random.uniform(0.8, 1.0)
                        )
                        pattern.set_step(track_name, step, step_obj)
        
        return pattern
    
    def _get_step_probability(self, track_name: str, step: int) -> float:
        """Get probability of a step being active based on track and position"""
        if track_name == "kick":
            # Kick drums usually on strong beats
            if step % 4 == 0:  # On beats 1, 5, 9, 13
                return 0.8
            elif step % 2 == 0:  # On beats 3, 7, 11, 15
                return 0.3
            else:
                return 0.1
        
        elif track_name == "bass":
            # Bass often offbeat or syncopated
            if step % 4 == 2:  # Off-beats
                return 0.6
            else:
                return 0.2
        
        elif track_name == "lead":
            # Lead patterns more varied
            return random.uniform(0.1, 0.4)
        
        elif track_name == "stab":
            # Stabs on accents
            if step % 8 == 4:  # Beat 5
                return 0.7
            else:
                return 0.1
        
        return 0.3  # Default
    
    def _generate_random_synth_params(self, synth_type: SynthType) -> SynthParams:
        """Generate random synthesis parameters for a synth type"""
        base_params = {
            SynthType.GABBER_KICK: {
                "freq": random.uniform(55, 75),
                "crunch": random.uniform(0.6, 0.9),
                "drive": random.uniform(2.0, 4.0),
                "doorlussen": random.uniform(0.5, 0.8)
            },
            SynthType.ACID_BASS: {
                "freq": random.uniform(80, 150),
                "cutoff": random.uniform(800, 2000),
                "resonance": random.uniform(0.6, 0.9),
                "crunch": random.uniform(0.2, 0.5)
            },
            SynthType.HOOVER_SYNTH: {
                "freq": random.uniform(200, 400),
                "mod_index": random.uniform(3, 8),
                "crunch": random.uniform(0.7, 0.9),
                "cutoff": random.uniform(3000, 8000)
            },
            SynthType.SCREECH_LEAD: {
                "freq": random.uniform(600, 1200),
                "crunch": random.uniform(0.8, 1.0),
                "drive": random.uniform(3.0, 6.0),
                "cutoff": random.uniform(8000, 15000)
            },
            SynthType.HARDCORE_STAB: {
                "freq": random.uniform(180, 300),
                "crunch": random.uniform(0.9, 1.0),
                "drive": random.uniform(4.0, 8.0),
                "attack": random.uniform(0.001, 0.005)
            }
        }
        
        params = SynthParams()
        synth_defaults = base_params.get(synth_type, {})
        
        for param, value in synth_defaults.items():
            setattr(params, param, value)
        
        # Add some random variation to other parameters
        params.amp = random.uniform(0.7, 1.0)
        params.attack = random.uniform(0.001, 0.02)
        params.decay = random.uniform(0.05, 0.3)
        params.sustain = random.uniform(0.3, 0.8)
        params.release = random.uniform(0.1, 0.5)
        
        return params
    
    async def evolve_generation(self) -> List[PatternGenome]:
        """Evolve one generation of the population"""
        self.logger.info(f"Evolving generation {self.generation}")
        
        # Evaluate current population
        await self._evaluate_population()
        
        # Select parents for reproduction
        parents = self._select_parents()
        
        # Create next generation
        next_generation = []
        
        # Preserve elite patterns
        elite = sorted(self.population, key=lambda g: g.total_fitness, reverse=True)[:self.config.elite_size]
        for genome in elite:
            elite_copy = copy.deepcopy(genome)
            elite_copy.age += 1
            next_generation.append(elite_copy)
        
        # Generate offspring through crossover and mutation
        while len(next_generation) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutate offspring
                if random.random() < self.config.mutation_rate:
                    child1.pattern = self._mutate_pattern(child1.pattern)
                    child1.mutations_applied.extend(self._last_mutations)
                
                if random.random() < self.config.mutation_rate:
                    child2.pattern = self._mutate_pattern(child2.pattern)
                    child2.mutations_applied.extend(self._last_mutations)
                
                child1.generation = self.generation + 1
                child2.generation = self.generation + 1
                
                next_generation.extend([child1, child2])
            
            else:
                # Mutation only
                parent = random.choice(parents)
                child = copy.deepcopy(parent)
                child.pattern = self._mutate_pattern(child.pattern)
                child.mutations_applied.extend(self._last_mutations)
                child.generation = self.generation + 1
                child.parent_ids = [parent.pattern.name]
                
                next_generation.append(child)
        
        # Limit to population size
        next_generation = next_generation[:self.config.population_size]
        
        # Update population
        self.population = next_generation
        self.generation += 1
        
        # Record evolution history
        self.evolution_history.append({
            "generation": self.generation,
            "best_fitness": max(g.total_fitness for g in self.population),
            "avg_fitness": np.mean([g.total_fitness for g in self.population]),
            "diversity": self._calculate_population_diversity()
        })
        
        return self.population
    
    async def _evaluate_population(self):
        """Evaluate fitness for entire population"""
        for genome in self.population:
            await self._evaluate_genome(genome)
    
    async def _evaluate_genome(self, genome: PatternGenome):
        """Evaluate fitness for a single genome"""
        pattern = genome.pattern
        
        # Evaluate each fitness metric
        for metric, evaluator in self.fitness_evaluators.items():
            try:
                score = await evaluator(pattern)
                genome.fitness_scores[metric] = score
            except Exception as e:
                self.logger.error(f"Error evaluating {metric.value}: {e}")
                genome.fitness_scores[metric] = 0.0
        
        # Calculate total fitness
        genome.calculate_total_fitness(self.config)
    
    async def _evaluate_hardcore_authenticity(self, pattern: HardcorePattern) -> float:
        """Evaluate how authentic the pattern sounds as hardcore"""
        score = 0.0
        
        # BPM check (hardcore is typically 150-220 BPM)
        if 150 <= pattern.bpm <= 220:
            score += 0.3
        elif 170 <= pattern.bpm <= 200:
            score += 0.5  # Bonus for core hardcore range
        
        # Kick drum presence and pattern
        if "kick" in pattern.tracks:
            kick_track = pattern.tracks["kick"]
            kick_steps = sum(1 for step in kick_track.steps if step is not None)
            
            # Hardcore should have prominent kick
            if kick_steps >= 4:
                score += 0.3
            
            # Check for 4/4 pattern (steps 0, 4, 8, 12)
            strong_beats = [0, 4, 8, 12]
            strong_beat_hits = sum(1 for beat in strong_beats if beat < len(kick_track.steps) and kick_track.steps[beat] is not None)
            score += (strong_beat_hits / 4) * 0.2
        
        # Check for aggressive synthesis parameters
        total_crunch = 0.0
        param_count = 0
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    total_crunch += step.params.crunch
                    param_count += 1
        
        if param_count > 0:
            avg_crunch = total_crunch / param_count
            score += min(0.2, avg_crunch * 0.3)  # Bonus for high crunch
        
        return min(1.0, score)
    
    async def _evaluate_danceability(self, pattern: HardcorePattern) -> float:
        """Evaluate dancefloor effectiveness"""
        score = 0.0
        
        # Tempo in dance range
        if 140 <= pattern.bpm <= 200:
            score += 0.3
        
        # Strong kick pattern
        if "kick" in pattern.tracks:
            kick_track = pattern.tracks["kick"]
            kick_density = sum(1 for step in kick_track.steps if step is not None) / len(kick_track.steps)
            
            # Optimal kick density for dancing
            if 0.2 <= kick_density <= 0.5:
                score += 0.4
        
        # Rhythmic variation (not too repetitive, not too chaotic)
        rhythmic_variation = self._calculate_rhythmic_variation(pattern)
        if 0.3 <= rhythmic_variation <= 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    async def _evaluate_rhythmic_complexity(self, pattern: HardcorePattern) -> float:
        """Evaluate rhythmic complexity and intricacy"""
        if not pattern.tracks:
            return 0.0
        
        complexity_scores = []
        
        for track in pattern.tracks.values():
            track_complexity = 0.0
            
            # Step density
            active_steps = sum(1 for step in track.steps if step is not None)
            density = active_steps / len(track.steps)
            track_complexity += min(0.5, density * 2)  # Reward moderate density
            
            # Velocity variation
            velocities = [step.velocity for step in track.steps if step is not None]
            if velocities:
                velocity_std = np.std(velocities)
                track_complexity += min(0.3, velocity_std * 3)
            
            # Probability variation (for polyrhythms)
            probabilities = [step.probability for step in track.steps if step is not None]
            if probabilities:
                prob_variation = np.std(probabilities)
                track_complexity += min(0.2, prob_variation * 5)
            
            complexity_scores.append(track_complexity)
        
        return min(1.0, np.mean(complexity_scores))
    
    async def _evaluate_harmonic_richness(self, pattern: HardcorePattern) -> float:
        """Evaluate harmonic content richness"""
        if not pattern.tracks:
            return 0.0
        
        # Count unique frequencies used
        frequencies = set()
        synth_types = set()
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    frequencies.add(step.params.freq)
                    synth_types.add(step.synth_type)
        
        # Reward frequency diversity
        freq_diversity = min(1.0, len(frequencies) / 10)
        
        # Reward synth type diversity
        synth_diversity = min(1.0, len(synth_types) / 5)
        
        return (freq_diversity * 0.6) + (synth_diversity * 0.4)
    
    async def _evaluate_energy_level(self, pattern: HardcorePattern) -> float:
        """Evaluate overall energy level"""
        total_energy = 0.0
        step_count = 0
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    # Energy from amplitude and crunch
                    energy = step.params.amp * step.velocity
                    energy += step.params.crunch * 0.5
                    energy += step.params.drive * 0.1
                    
                    total_energy += energy
                    step_count += 1
        
        if step_count > 0:
            avg_energy = total_energy / step_count
            return min(1.0, avg_energy / 3)  # Normalize
        
        return 0.0
    
    async def _evaluate_novelty(self, pattern: HardcorePattern) -> float:
        """Evaluate novelty compared to existing patterns"""
        if not self.pattern_database:
            return 0.5  # Neutral novelty if no database
        
        # Simple novelty based on pattern fingerprint
        fingerprint = self._calculate_pattern_fingerprint(pattern)
        
        max_similarity = 0.0
        for existing_fingerprint in self.pattern_database.values():
            similarity = self._calculate_fingerprint_similarity(fingerprint, existing_fingerprint)
            max_similarity = max(max_similarity, similarity)
        
        # Novelty is inverse of similarity
        novelty = 1.0 - max_similarity
        return novelty
    
    async def _evaluate_technical_quality(self, pattern: HardcorePattern) -> float:
        """Evaluate technical/production quality"""
        score = 0.0
        
        # Check for reasonable parameter ranges
        param_quality = 0.0
        param_count = 0
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    params = step.params
                    
                    # Check parameter sanity
                    if 20 <= params.freq <= 8000:  # Reasonable frequency range
                        param_quality += 0.2
                    if 0.0 <= params.amp <= 1.0:  # Valid amplitude
                        param_quality += 0.2
                    if 0.001 <= params.attack <= 2.0:  # Reasonable envelope
                        param_quality += 0.2
                    if params.cutoff > params.freq:  # Filter above fundamental
                        param_quality += 0.2
                    if 0.0 <= params.resonance <= 1.0:  # Valid resonance
                        param_quality += 0.2
                    
                    param_count += 1
        
        if param_count > 0:
            score = param_quality / param_count
        
        return min(1.0, score)
    
    def _select_parents(self) -> List[PatternGenome]:
        """Select parents for reproduction using tournament selection"""
        parents = []
        population_with_fitness = [(g, g.total_fitness) for g in self.population]
        
        for _ in range(self.config.population_size):
            # Tournament selection
            tournament = random.sample(population_with_fitness, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x[1] * self.config.selection_pressure)[0]
            parents.append(winner)
        
        return parents
    
    def _crossover(self, parent1: PatternGenome, parent2: PatternGenome) -> Tuple[PatternGenome, PatternGenome]:
        """Perform crossover between two patterns"""
        crossover_type = random.choice(list(CrossoverType))
        
        if crossover_type == CrossoverType.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif crossover_type == CrossoverType.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif crossover_type == CrossoverType.TRACK_WISE:
            return self._track_wise_crossover(parent1, parent2)
        else:
            return self._single_point_crossover(parent1, parent2)  # Default
    
    def _single_point_crossover(self, parent1: PatternGenome, parent2: PatternGenome) -> Tuple[PatternGenome, PatternGenome]:
        """Single point crossover"""
        p1_pattern = parent1.pattern
        p2_pattern = parent2.pattern
        
        # Create child patterns
        child1_pattern = copy.deepcopy(p1_pattern)
        child2_pattern = copy.deepcopy(p2_pattern)
        
        child1_pattern.name = f"gen{self.generation+1}_cross_{int(time.time())}_1"
        child2_pattern.name = f"gen{self.generation+1}_cross_{int(time.time())}_2"
        
        # Find common tracks
        common_tracks = set(p1_pattern.tracks.keys()) & set(p2_pattern.tracks.keys())
        
        if common_tracks:
            # Choose crossover point
            max_steps = max(len(p1_pattern.tracks[list(common_tracks)[0]].steps), 
                           len(p2_pattern.tracks[list(common_tracks)[0]].steps))
            crossover_point = random.randint(1, max_steps - 1)
            
            # Perform crossover for each common track
            for track_name in common_tracks:
                t1_steps = p1_pattern.tracks[track_name].steps
                t2_steps = p2_pattern.tracks[track_name].steps
                
                # Crossover steps
                child1_steps = t1_steps[:crossover_point] + t2_steps[crossover_point:]
                child2_steps = t2_steps[:crossover_point] + t1_steps[crossover_point:]
                
                child1_pattern.tracks[track_name].steps = child1_steps
                child2_pattern.tracks[track_name].steps = child2_steps
        
        # Create genomes
        child1 = PatternGenome(
            pattern=child1_pattern,
            parent_ids=[p1_pattern.name, p2_pattern.name]
        )
        child2 = PatternGenome(
            pattern=child2_pattern,
            parent_ids=[p1_pattern.name, p2_pattern.name]
        )
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: PatternGenome, parent2: PatternGenome) -> Tuple[PatternGenome, PatternGenome]:
        """Two point crossover"""
        # Similar to single point but with two crossover points
        p1_pattern = parent1.pattern
        p2_pattern = parent2.pattern
        
        child1_pattern = copy.deepcopy(p1_pattern)
        child2_pattern = copy.deepcopy(p2_pattern)
        
        child1_pattern.name = f"gen{self.generation+1}_2cross_{int(time.time())}_1"
        child2_pattern.name = f"gen{self.generation+1}_2cross_{int(time.time())}_2"
        
        common_tracks = set(p1_pattern.tracks.keys()) & set(p2_pattern.tracks.keys())
        
        if common_tracks:
            max_steps = max(len(p1_pattern.tracks[list(common_tracks)[0]].steps), 
                           len(p2_pattern.tracks[list(common_tracks)[0]].steps))
            
            point1 = random.randint(1, max_steps // 2)
            point2 = random.randint(max_steps // 2 + 1, max_steps - 1)
            
            for track_name in common_tracks:
                t1_steps = p1_pattern.tracks[track_name].steps
                t2_steps = p2_pattern.tracks[track_name].steps
                
                # Two-point crossover
                child1_steps = t1_steps[:point1] + t2_steps[point1:point2] + t1_steps[point2:]
                child2_steps = t2_steps[:point1] + t1_steps[point1:point2] + t2_steps[point2:]
                
                child1_pattern.tracks[track_name].steps = child1_steps
                child2_pattern.tracks[track_name].steps = child2_steps
        
        child1 = PatternGenome(pattern=child1_pattern, parent_ids=[p1_pattern.name, p2_pattern.name])
        child2 = PatternGenome(pattern=child2_pattern, parent_ids=[p1_pattern.name, p2_pattern.name])
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: PatternGenome, parent2: PatternGenome) -> Tuple[PatternGenome, PatternGenome]:
        """Uniform crossover - each step has 50% chance of coming from either parent"""
        p1_pattern = parent1.pattern
        p2_pattern = parent2.pattern
        
        child1_pattern = copy.deepcopy(p1_pattern)
        child2_pattern = copy.deepcopy(p2_pattern)
        
        child1_pattern.name = f"gen{self.generation+1}_uniform_{int(time.time())}_1"
        child2_pattern.name = f"gen{self.generation+1}_uniform_{int(time.time())}_2"
        
        common_tracks = set(p1_pattern.tracks.keys()) & set(p2_pattern.tracks.keys())
        
        for track_name in common_tracks:
            t1_steps = p1_pattern.tracks[track_name].steps
            t2_steps = p2_pattern.tracks[track_name].steps
            
            child1_steps = []
            child2_steps = []
            
            max_len = max(len(t1_steps), len(t2_steps))
            
            for i in range(max_len):
                if random.random() < 0.5:
                    # Choose from parent 1
                    c1_step = t1_steps[i] if i < len(t1_steps) else None
                    c2_step = t2_steps[i] if i < len(t2_steps) else None
                else:
                    # Choose from parent 2
                    c1_step = t2_steps[i] if i < len(t2_steps) else None
                    c2_step = t1_steps[i] if i < len(t1_steps) else None
                
                child1_steps.append(c1_step)
                child2_steps.append(c2_step)
            
            child1_pattern.tracks[track_name].steps = child1_steps
            child2_pattern.tracks[track_name].steps = child2_steps
        
        child1 = PatternGenome(pattern=child1_pattern, parent_ids=[p1_pattern.name, p2_pattern.name])
        child2 = PatternGenome(pattern=child2_pattern, parent_ids=[p1_pattern.name, p2_pattern.name])
        
        return child1, child2
    
    def _track_wise_crossover(self, parent1: PatternGenome, parent2: PatternGenome) -> Tuple[PatternGenome, PatternGenome]:
        """Crossover entire tracks between parents"""
        p1_pattern = parent1.pattern
        p2_pattern = parent2.pattern
        
        child1_pattern = copy.deepcopy(p1_pattern)
        child2_pattern = copy.deepcopy(p2_pattern)
        
        child1_pattern.name = f"gen{self.generation+1}_trackwise_{int(time.time())}_1"
        child2_pattern.name = f"gen{self.generation+1}_trackwise_{int(time.time())}_2"
        
        # Swap some tracks
        all_tracks = set(p1_pattern.tracks.keys()) | set(p2_pattern.tracks.keys())
        
        for track_name in all_tracks:
            if random.random() < 0.5:  # 50% chance to swap
                if track_name in p1_pattern.tracks and track_name in p2_pattern.tracks:
                    # Swap tracks
                    child1_pattern.tracks[track_name] = copy.deepcopy(p2_pattern.tracks[track_name])
                    child2_pattern.tracks[track_name] = copy.deepcopy(p1_pattern.tracks[track_name])
        
        child1 = PatternGenome(pattern=child1_pattern, parent_ids=[p1_pattern.name, p2_pattern.name])
        child2 = PatternGenome(pattern=child2_pattern, parent_ids=[p1_pattern.name, p2_pattern.name])
        
        return child1, child2
    
    def _mutate_pattern(self, pattern: HardcorePattern) -> HardcorePattern:
        """Apply random mutations to a pattern"""
        self._last_mutations = []
        mutated_pattern = copy.deepcopy(pattern)
        mutated_pattern.name = f"gen{self.generation+1}_mut_{int(time.time())}"
        
        # Apply multiple mutations
        num_mutations = random.randint(1, 3)
        
        for _ in range(num_mutations):
            mutation_type = random.choice(list(MutationType))
            
            try:
                if mutation_type == MutationType.ADD_STEP:
                    self._mutate_add_step(mutated_pattern)
                elif mutation_type == MutationType.REMOVE_STEP:
                    self._mutate_remove_step(mutated_pattern)
                elif mutation_type == MutationType.MODIFY_VELOCITY:
                    self._mutate_modify_velocity(mutated_pattern)
                elif mutation_type == MutationType.MODIFY_PARAMS:
                    self._mutate_modify_params(mutated_pattern)
                elif mutation_type == MutationType.SHIFT_PATTERN:
                    self._mutate_shift_pattern(mutated_pattern)
                elif mutation_type == MutationType.EVOLVE_CRUNCH:
                    self._mutate_evolve_crunch(mutated_pattern)
                elif mutation_type == MutationType.EVOLVE_BPM:
                    self._mutate_evolve_bpm(mutated_pattern)
                elif mutation_type == MutationType.CLONE_STEP:
                    self._mutate_clone_step(mutated_pattern)
                
                self._last_mutations.append(mutation_type)
                
            except Exception as e:
                self.logger.warning(f"Mutation {mutation_type.value} failed: {e}")
        
        return mutated_pattern
    
    def _mutate_add_step(self, pattern: HardcorePattern):
        """Add a step to a random track"""
        if not pattern.tracks:
            return
        
        track_name = random.choice(list(pattern.tracks.keys()))
        track = pattern.tracks[track_name]
        
        # Find empty step
        empty_steps = [i for i, step in enumerate(track.steps) if step is None]
        if empty_steps:
            step_idx = random.choice(empty_steps)
            
            # Create new step based on existing steps in track
            existing_steps = [s for s in track.steps if s is not None]
            if existing_steps:
                base_step = copy.deepcopy(random.choice(existing_steps))
                # Add some variation
                base_step.velocity *= random.uniform(0.8, 1.2)
                base_step.params.freq *= random.uniform(0.9, 1.1)
            else:
                # Create completely new step
                synth_type = SynthType.GABBER_KICK if "kick" in track_name.lower() else SynthType.ACID_BASS
                base_step = PatternStep(
                    synth_type=synth_type,
                    params=self._generate_random_synth_params(synth_type)
                )
            
            track.steps[step_idx] = base_step
    
    def _mutate_remove_step(self, pattern: HardcorePattern):
        """Remove a step from a random track"""
        if not pattern.tracks:
            return
        
        track_name = random.choice(list(pattern.tracks.keys()))
        track = pattern.tracks[track_name]
        
        # Find active steps
        active_steps = [i for i, step in enumerate(track.steps) if step is not None]
        if active_steps:
            step_idx = random.choice(active_steps)
            track.steps[step_idx] = None
    
    def _mutate_modify_velocity(self, pattern: HardcorePattern):
        """Modify velocity of random steps"""
        all_steps = []
        for track in pattern.tracks.values():
            for i, step in enumerate(track.steps):
                if step is not None:
                    all_steps.append(step)
        
        if all_steps:
            step = random.choice(all_steps)
            step.velocity *= random.uniform(0.7, 1.3)
            step.velocity = max(0.1, min(1.0, step.velocity))  # Clamp
    
    def _mutate_modify_params(self, pattern: HardcorePattern):
        """Modify synthesis parameters of random steps"""
        all_steps = []
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    all_steps.append(step)
        
        if all_steps:
            step = random.choice(all_steps)
            params = step.params
            
            # Choose random parameter to mutate
            param_mutations = {
                'freq': lambda x: x * random.uniform(0.9, 1.1),
                'amp': lambda x: max(0.1, min(1.0, x * random.uniform(0.8, 1.2))),
                'crunch': lambda x: max(0.0, min(1.0, x + random.uniform(-0.2, 0.2))),
                'drive': lambda x: max(1.0, min(10.0, x * random.uniform(0.8, 1.3))),
                'cutoff': lambda x: x * random.uniform(0.8, 1.5),
                'resonance': lambda x: max(0.0, min(1.0, x + random.uniform(-0.2, 0.2)))
            }
            
            param_name = random.choice(list(param_mutations.keys()))
            if hasattr(params, param_name):
                current_value = getattr(params, param_name)
                new_value = param_mutations[param_name](current_value)
                setattr(params, param_name, new_value)
    
    def _mutate_shift_pattern(self, pattern: HardcorePattern):
        """Shift pattern timing"""
        if not pattern.tracks:
            return
        
        track_name = random.choice(list(pattern.tracks.keys()))
        track = pattern.tracks[track_name]
        
        # Rotate steps
        shift = random.randint(1, len(track.steps) - 1)
        track.steps = track.steps[shift:] + track.steps[:shift]
    
    def _mutate_evolve_crunch(self, pattern: HardcorePattern):
        """Evolve crunch parameters across the pattern"""
        crunch_delta = random.uniform(-0.3, 0.3)
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.crunch = max(0.0, min(1.0, step.params.crunch + crunch_delta))
    
    def _mutate_evolve_bpm(self, pattern: HardcorePattern):
        """Mutate BPM within hardcore ranges"""
        bpm_delta = random.uniform(-10, 10)
        new_bpm = pattern.bpm + bpm_delta
        pattern.bpm = max(120, min(250, new_bpm))  # Keep in reasonable range
    
    def _mutate_clone_step(self, pattern: HardcorePattern):
        """Clone a step to another position"""
        if not pattern.tracks:
            return
        
        track_name = random.choice(list(pattern.tracks.keys()))
        track = pattern.tracks[track_name]
        
        # Find source step
        active_steps = [(i, step) for i, step in enumerate(track.steps) if step is not None]
        if active_steps:
            source_idx, source_step = random.choice(active_steps)
            
            # Find target position
            empty_steps = [i for i, step in enumerate(track.steps) if step is None]
            if empty_steps:
                target_idx = random.choice(empty_steps)
                track.steps[target_idx] = copy.deepcopy(source_step)
    
    def _calculate_pattern_fingerprint(self, pattern: HardcorePattern) -> List[float]:
        """Calculate a fingerprint for pattern similarity comparison"""
        fingerprint = []
        
        # BPM component
        fingerprint.append(pattern.bpm / 250.0)  # Normalize
        
        # Track density components
        for track_name in ["kick", "bass", "lead", "stab"]:
            if track_name in pattern.tracks:
                track = pattern.tracks[track_name]
                density = sum(1 for step in track.steps if step is not None) / len(track.steps)
                fingerprint.append(density)
            else:
                fingerprint.append(0.0)
        
        # Parameter averages
        avg_crunch = 0.0
        avg_freq = 0.0
        param_count = 0
        
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    avg_crunch += step.params.crunch
                    avg_freq += step.params.freq
                    param_count += 1
        
        if param_count > 0:
            fingerprint.append(avg_crunch / param_count)
            fingerprint.append(avg_freq / (param_count * 1000))  # Normalize
        else:
            fingerprint.extend([0.0, 0.0])
        
        return fingerprint
    
    def _calculate_fingerprint_similarity(self, fp1: List[float], fp2: List[float]) -> float:
        """Calculate similarity between two fingerprints"""
        if len(fp1) != len(fp2):
            return 0.0
        
        # Euclidean distance
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(fp1, fp2)))
        
        # Convert to similarity (0-1)
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def _calculate_rhythmic_variation(self, pattern: HardcorePattern) -> float:
        """Calculate rhythmic variation in pattern"""
        if not pattern.tracks:
            return 0.0
        
        # Calculate step density variation across tracks
        densities = []
        for track in pattern.tracks.values():
            density = sum(1 for step in track.steps if step is not None) / len(track.steps)
            densities.append(density)
        
        if len(densities) > 1:
            return float(np.std(densities))
        
        return 0.0
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of current population"""
        if len(self.population) < 2:
            return 0.0
        
        fingerprints = [self._calculate_pattern_fingerprint(g.pattern) for g in self.population]
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(fingerprints[i], fingerprints[j])))
                total_distance += distance
                comparisons += 1
        
        if comparisons > 0:
            return total_distance / comparisons
        
        return 0.0
    
    def add_user_feedback(self, pattern_name: str, rating: float):
        """Add user feedback for a pattern"""
        for genome in self.population:
            if genome.pattern.name == pattern_name:
                genome.user_ratings.append(rating)
                genome.play_count += 1
                break
    
    def get_best_patterns(self, n: int = 10) -> List[PatternGenome]:
        """Get top N patterns from current population"""
        return sorted(self.population, key=lambda g: g.total_fitness, reverse=True)[:n]
    
    def save_population(self, filename: str):
        """Save current population to file"""
        data = {
            "generation": self.generation,
            "config": {
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate
            },
            "population": [
                {
                    "pattern": genome.pattern.to_dict(),
                    "fitness_scores": {k.value: v for k, v in genome.fitness_scores.items()},
                    "total_fitness": genome.total_fitness,
                    "generation": genome.generation,
                    "parent_ids": genome.parent_ids,
                    "mutations_applied": [m.value for m in genome.mutations_applied],
                    "age": genome.age,
                    "play_count": genome.play_count,
                    "user_ratings": genome.user_ratings
                }
                for genome in self.population
            ],
            "evolution_history": self.evolution_history
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

# Test function
async def test_evolution_engine():
    """Test the pattern evolution engine"""
    print("🧬 Testing Pattern Evolution Engine")
    print("=" * 50)
    
    # Create evolution config
    config = EvolutionConfig(
        population_size=20,
        generations=5,
        mutation_rate=0.4,
        crossover_rate=0.8
    )
    
    engine = PatternEvolutionEngine(config)
    
    # Create seed patterns
    from cli_shared import create_gabber_kick_pattern, create_industrial_pattern
    
    seed_patterns = [
        create_gabber_kick_pattern("Seed Gabber", 180),
        create_industrial_pattern("Seed Industrial", 140)
    ]
    
    print(f"🌱 Initializing population with {len(seed_patterns)} seed patterns...")
    population = engine.initialize_population(seed_patterns)
    
    print(f"   Population size: {len(population)}")
    
    # Evolve for several generations
    print(f"\n🔄 Evolving for {config.generations} generations...")
    
    for gen in range(config.generations):
        population = await engine.evolve_generation()
        
        best_genome = max(population, key=lambda g: g.total_fitness)
        avg_fitness = np.mean([g.total_fitness for g in population])
        
        print(f"   Gen {engine.generation}: Best={best_genome.total_fitness:.3f}, "
              f"Avg={avg_fitness:.3f}, Diversity={engine._calculate_population_diversity():.3f}")
    
    # Show best patterns
    print(f"\n🏆 Top 5 evolved patterns:")
    best_patterns = engine.get_best_patterns(5)
    
    for i, genome in enumerate(best_patterns):
        pattern = genome.pattern
        print(f"   {i+1}. {pattern.name}")
        print(f"      BPM: {pattern.bpm:.1f}, Fitness: {genome.total_fitness:.3f}")
        print(f"      Tracks: {list(pattern.tracks.keys())}")
        print(f"      Generation: {genome.generation}, Age: {genome.age}")
        
        if genome.mutations_applied:
            mutations = [m.value for m in genome.mutations_applied[-3:]]  # Last 3 mutations
            print(f"      Recent mutations: {', '.join(mutations)}")
        
        # Show fitness breakdown
        fitness_breakdown = []
        for metric, score in genome.fitness_scores.items():
            if score > 0:
                fitness_breakdown.append(f"{metric.value}={score:.2f}")
        if fitness_breakdown:
            print(f"      Fitness: {', '.join(fitness_breakdown[:3])}...")  # Show top 3
    
    print(f"\n🧬 Evolution completed! Final population diversity: {engine._calculate_population_diversity():.3f}")
    print("✅ Pattern Evolution Engine test completed!")

if __name__ == "__main__":
    asyncio.run(test_evolution_engine())