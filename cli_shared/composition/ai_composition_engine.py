#!/usr/bin/env python3
"""
AI-Powered Composition Engine for Hardcore Music Production
Advanced composition features using multiple AI models and creative algorithms
"""

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import uuid
import numpy as np
from collections import defaultdict, deque
import math

from ..interfaces.synthesizer import AbstractSynthesizer
from ..models.hardcore_models import HardcorePattern, SynthParams, SynthType
from ..ai.conversation_engine import ConversationEngine, ConversationType
from ..ai.hardcore_knowledge_base import HardcoreKnowledgeBase, hardcore_kb, ArtistStyle
from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer
from ..evolution.pattern_evolution_engine import PatternEvolutionEngine, FitnessScore


logger = logging.getLogger(__name__)


class CompositionStyle(Enum):
    MINIMALIST = "minimalist"                # Sparse, surgical placement
    MAXIMALIST = "maximalist"               # Dense, wall of sound
    PROGRESSIVE = "progressive"             # Building complexity over time
    BREAKDOWN = "breakdown"                 # Strategic element removal
    CALL_RESPONSE = "call_response"         # Interactive elements
    POLYRHYTHMIC = "polyrhythmic"           # Complex rhythmic interactions
    ATMOSPHERIC = "atmospheric"             # Ambient, textural focus
    AGGRESSIVE = "aggressive"               # Maximum intensity throughout


class CompositionStructure(Enum):
    CLASSIC_TECHNO = "classic_techno"       # Intro-Verse-Break-Drop-Outro
    HARDCORE_ANTHEM = "hardcore_anthem"     # Build-Drop-Build-Massive Drop
    INDUSTRIAL_JOURNEY = "industrial_journey" # Atmospheric-Dark-Intense-Resolution
    GABBER_ASSAULT = "gabber_assault"       # Immediate intensity-Breakdown-Final assault
    UNDERGROUND_LOOP = "underground_loop"   # Hypnotic, minimal changes
    PEAK_TIME_DESTROYER = "peak_time_destroyer" # Maximum dancefloor impact


class CompositionElement(Enum):
    KICK = "kick"
    BASS = "bass" 
    LEAD = "lead"
    PERCUSSION = "percussion"
    ATMOSPHERE = "atmosphere"
    EFFECTS = "effects"
    VOCALS = "vocals"
    NOISE = "noise"


@dataclass
class CompositionSection:
    """A section within a composition"""
    name: str
    start_bar: int
    end_bar: int
    bpm: int
    key: str
    elements: Dict[CompositionElement, Dict[str, Any]]
    energy_level: float  # 0.0 to 1.0
    tension_curve: List[float]  # Tension over time within section
    transition_in: Optional[str] = None
    transition_out: Optional[str] = None
    
    def get_duration_bars(self) -> int:
        return self.end_bar - self.start_bar
    
    def get_duration_seconds(self, bpm: int) -> float:
        bars = self.get_duration_bars()
        return (bars * 4 * 60) / bpm  # 4 beats per bar


@dataclass
class CompositionBlueprint:
    """Complete composition blueprint"""
    title: str
    total_bars: int
    base_bpm: int
    base_key: str
    style: CompositionStyle
    structure: CompositionStructure
    genre: str
    target_energy_curve: List[float]  # Overall energy progression
    sections: List[CompositionSection]
    global_effects: Dict[str, Any]
    arrangement_notes: List[str]
    estimated_duration: float
    
    def get_section_at_bar(self, bar: int) -> Optional[CompositionSection]:
        """Get section that contains the specified bar"""
        for section in self.sections:
            if section.start_bar <= bar < section.end_bar:
                return section
        return None


@dataclass
class GeneratedComposition:
    """A complete generated composition with patterns and arrangement"""
    blueprint: CompositionBlueprint
    patterns: Dict[str, HardcorePattern]  # Pattern ID -> Pattern
    arrangement: List[Dict[str, Any]]     # Time-ordered arrangement events
    audio_data: Optional[np.ndarray] = None
    analysis_results: Optional[Dict[str, Any]] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0


class AbstractCompositionStrategy(ABC):
    """Abstract strategy for composition generation"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def generate_blueprint(self, 
                               requirements: Dict[str, Any], 
                               knowledge_base: HardcoreKnowledgeBase) -> CompositionBlueprint:
        """Generate composition blueprint based on requirements"""
        pass
    
    @abstractmethod
    async def realize_blueprint(self,
                              blueprint: CompositionBlueprint,
                              synthesizer: AbstractSynthesizer,
                              pattern_generator: Callable) -> GeneratedComposition:
        """Convert blueprint into actual patterns and arrangement"""
        pass


class HardcoreAnthemStrategy(AbstractCompositionStrategy):
    """Strategy for hardcore anthem compositions"""
    
    def __init__(self):
        super().__init__(
            "Hardcore Anthem",
            "Classic hardcore structure with massive drops and crowd-pleasing arrangements"
        )
    
    async def generate_blueprint(self, 
                               requirements: Dict[str, Any], 
                               knowledge_base: HardcoreKnowledgeBase) -> CompositionBlueprint:
        
        bpm = requirements.get("bpm", 180)
        key = requirements.get("key", "Am")
        duration_minutes = requirements.get("duration", 6.0)
        
        # Calculate total bars (assuming 4/4 time)
        bars_per_minute = bpm / 4  # 4 beats per bar, so bars = bpm/4 per minute
        total_bars = int(duration_minutes * bars_per_minute)
        
        # Hardcore anthem structure: Intro(16) -> Build(32) -> Drop(32) -> Break(16) -> Final Drop(32) -> Outro(16)
        sections = [
            CompositionSection(
                name="Intro",
                start_bar=0,
                end_bar=16,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "filtered", "volume": 0.6},
                    CompositionElement.ATMOSPHERE: {"type": "industrial_pad", "volume": 0.4}
                },
                energy_level=0.3,
                tension_curve=[0.2, 0.3, 0.4, 0.5],
                transition_out="filter_sweep"
            ),
            CompositionSection(
                name="Build",
                start_bar=16,
                end_bar=48,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "building", "volume": 0.8},
                    CompositionElement.BASS: {"pattern": "acid_rise", "volume": 0.7},
                    CompositionElement.LEAD: {"pattern": "hoover_stab", "volume": 0.6},
                    CompositionElement.PERCUSSION: {"pattern": "building_hats", "volume": 0.5}
                },
                energy_level=0.7,
                tension_curve=np.linspace(0.5, 0.9, 8).tolist(),
                transition_out="slam"
            ),
            CompositionSection(
                name="Drop",
                start_bar=48,
                end_bar=80,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "gabber_full", "volume": 1.0},
                    CompositionElement.BASS: {"pattern": "acid_full", "volume": 0.9},
                    CompositionElement.LEAD: {"pattern": "hoover_full", "volume": 0.8},
                    CompositionElement.PERCUSSION: {"pattern": "full_kit", "volume": 0.7},
                    CompositionElement.EFFECTS: {"type": "reverb_tail", "volume": 0.3}
                },
                energy_level=1.0,
                tension_curve=[1.0] * 8,
                transition_out="breakdown"
            ),
            CompositionSection(
                name="Break",
                start_bar=80,
                end_bar=96,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "filtered", "volume": 0.4},
                    CompositionElement.ATMOSPHERE: {"type": "vinyl_crackle", "volume": 0.6},
                    CompositionElement.VOCALS: {"type": "hardcore_vocal", "volume": 0.5}
                },
                energy_level=0.4,
                tension_curve=[0.4, 0.5, 0.6, 0.8],
                transition_out="slam"
            ),
            CompositionSection(
                name="Final Drop",
                start_bar=96,
                end_bar=128,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "gabber_ultimate", "volume": 1.2},
                    CompositionElement.BASS: {"pattern": "acid_ultimate", "volume": 1.0},
                    CompositionElement.LEAD: {"pattern": "hoover_ultimate", "volume": 0.9},
                    CompositionElement.PERCUSSION: {"pattern": "ultimate_kit", "volume": 0.8},
                    CompositionElement.NOISE: {"type": "white_noise_sweep", "volume": 0.4},
                    CompositionElement.EFFECTS: {"type": "max_reverb", "volume": 0.5}
                },
                energy_level=1.2,  # Over the top!
                tension_curve=[1.2] * 8,
                transition_out="fade"
            ),
            CompositionSection(
                name="Outro",
                start_bar=128,
                end_bar=144,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "fading", "volume": 0.3},
                    CompositionElement.ATMOSPHERE: {"type": "aftermath", "volume": 0.2}
                },
                energy_level=0.2,
                tension_curve=[0.5, 0.4, 0.3, 0.1],
                transition_out="fade_out"
            )
        ]
        
        # Calculate overall energy curve
        energy_curve = []
        for section in sections:
            section_length = section.get_duration_bars()
            section_energy = [section.energy_level] * section_length
            energy_curve.extend(section_energy)
        
        return CompositionBlueprint(
            title=f"Hardcore Anthem {random.randint(1000, 9999)}",
            total_bars=144,
            base_bpm=bpm,
            base_key=key,
            style=CompositionStyle.PROGRESSIVE,
            structure=CompositionStructure.HARDCORE_ANTHEM,
            genre="gabber",
            target_energy_curve=energy_curve,
            sections=sections,
            global_effects={"reverb": 0.3, "compression": 0.8, "limiting": 0.9},
            arrangement_notes=[
                "Classic hardcore anthem structure",
                "Build tension through filtering and layering",
                "Slam transitions for maximum impact",
                "Final drop goes beyond normal limits"
            ],
            estimated_duration=total_bars * 4 * 60 / bpm  # bars * beats/bar * seconds/minute / bpm
        )
    
    async def realize_blueprint(self,
                              blueprint: CompositionBlueprint,
                              synthesizer: AbstractSynthesizer,
                              pattern_generator: Callable) -> GeneratedComposition:
        
        patterns = {}
        arrangement = []
        
        for section in blueprint.sections:
            logger.info(f"Realizing section: {section.name}")
            
            # Generate patterns for each element in this section
            for element, config in section.elements.items():
                pattern_name = f"{section.name}_{element.value}_{section.start_bar}"
                
                # Create pattern based on element type and config
                pattern_prompt = self._create_pattern_prompt(element, config, section, blueprint)
                
                try:
                    # Generate pattern using the pattern generator
                    pattern = await pattern_generator(
                        prompt=pattern_prompt,
                        bpm=section.bpm,
                        genre=blueprint.genre
                    )
                    
                    if pattern:
                        pattern.name = pattern_name
                        patterns[pattern_name] = pattern
                        
                        # Add to arrangement
                        arrangement.append({
                            "type": "pattern_start",
                            "bar": section.start_bar,
                            "pattern_id": pattern_name,
                            "element": element.value,
                            "volume": config.get("volume", 1.0),
                            "section": section.name
                        })
                        
                        arrangement.append({
                            "type": "pattern_end", 
                            "bar": section.end_bar,
                            "pattern_id": pattern_name,
                            "element": element.value
                        })
                    
                except Exception as e:
                    logger.error(f"Failed to generate pattern for {element.value}: {e}")
            
            # Add section transitions
            if section.transition_out:
                arrangement.append({
                    "type": "transition",
                    "bar": section.end_bar - 2,  # Start transition 2 bars before section end
                    "transition_type": section.transition_out,
                    "duration_bars": 2,
                    "section": section.name
                })
        
        # Sort arrangement by bar
        arrangement.sort(key=lambda x: x["bar"])
        
        return GeneratedComposition(
            blueprint=blueprint,
            patterns=patterns,
            arrangement=arrangement,
            generation_metadata={
                "strategy": self.name,
                "patterns_generated": len(patterns),
                "total_arrangement_events": len(arrangement),
                "generation_time": time.time()
            },
            confidence_score=0.85
        )
    
    def _create_pattern_prompt(self, 
                             element: CompositionElement, 
                             config: Dict[str, Any],
                             section: CompositionSection,
                             blueprint: CompositionBlueprint) -> str:
        """Create natural language prompt for pattern generation"""
        
        base_prompt = f"Create a {blueprint.genre} {element.value} pattern for {section.name} section"
        
        # Add BPM and key info
        base_prompt += f" at {section.bpm} BPM in {section.key}"
        
        # Add energy level context
        if section.energy_level > 0.8:
            base_prompt += " with maximum intensity and aggression"
        elif section.energy_level > 0.6:
            base_prompt += " with high energy and driving force"
        elif section.energy_level > 0.4:
            base_prompt += " with moderate energy and building tension"
        else:
            base_prompt += " with low energy and atmospheric mood"
        
        # Element-specific additions
        if element == CompositionElement.KICK:
            pattern_type = config.get("pattern", "standard")
            if pattern_type == "filtered":
                base_prompt += ", heavily filtered and muffled"
            elif pattern_type == "building":
                base_prompt += ", gradually increasing in power and presence"
            elif pattern_type == "gabber_full":
                base_prompt += ", full gabber style with maximum crunch and distortion"
            elif pattern_type == "gabber_ultimate":
                base_prompt += ", ultimate gabber kick that destroys sound systems"
        
        elif element == CompositionElement.BASS:
            pattern_type = config.get("pattern", "standard")
            if pattern_type == "acid_rise":
                base_prompt += ", acidic bassline building in intensity"
            elif pattern_type == "acid_full":
                base_prompt += ", full acid bassline with squelchy resonance"
            elif pattern_type == "acid_ultimate":
                base_prompt += ", ultimate acid bassline with maximum resonance and drive"
        
        elif element == CompositionElement.LEAD:
            pattern_type = config.get("pattern", "standard")
            if pattern_type == "hoover_stab":
                base_prompt += ", classic hoover stab with aggressive edge"
            elif pattern_type == "hoover_full":
                base_prompt += ", full hoover lead with maximum aggression"
            elif pattern_type == "hoover_ultimate":
                base_prompt += ", ultimate hoover lead that cuts through everything"
        
        elif element == CompositionElement.ATMOSPHERE:
            atmo_type = config.get("type", "standard")
            if atmo_type == "industrial_pad":
                base_prompt += ", dark industrial atmospheric pad"
            elif atmo_type == "vinyl_crackle":
                base_prompt += ", vintage vinyl crackle and atmospheric noise"
            elif atmo_type == "aftermath":
                base_prompt += ", post-apocalyptic aftermath atmosphere"
        
        return base_prompt


class IndustrialJourneyStrategy(AbstractCompositionStrategy):
    """Strategy for industrial journey compositions"""
    
    def __init__(self):
        super().__init__(
            "Industrial Journey",
            "Atmospheric journey through dark industrial soundscapes"
        )
    
    async def generate_blueprint(self, 
                               requirements: Dict[str, Any], 
                               knowledge_base: HardcoreKnowledgeBase) -> CompositionBlueprint:
        
        bpm = requirements.get("bpm", 140)
        key = requirements.get("key", "Dm")  # Minor key for darkness
        duration_minutes = requirements.get("duration", 8.0)
        
        bars_per_minute = bpm / 4
        total_bars = int(duration_minutes * bars_per_minute)
        
        # Industrial journey: Ambient(32) -> Dark(48) -> Intense(64) -> Climax(32) -> Resolution(32)
        sections = [
            CompositionSection(
                name="Ambient Intro",
                start_bar=0,
                end_bar=32,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.ATMOSPHERE: {"type": "drone_pad", "volume": 0.5},
                    CompositionElement.NOISE: {"type": "industrial_ambience", "volume": 0.3},
                    CompositionElement.EFFECTS: {"type": "long_reverb", "volume": 0.4}
                },
                energy_level=0.2,
                tension_curve=np.linspace(0.1, 0.3, 8).tolist(),
                transition_out="fade_crossfade"
            ),
            CompositionSection(
                name="Dark Development",
                start_bar=32,
                end_bar=80,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "industrial_rumble", "volume": 0.6},
                    CompositionElement.PERCUSSION: {"pattern": "metallic_hits", "volume": 0.5},
                    CompositionElement.ATMOSPHERE: {"type": "dark_drone", "volume": 0.6},
                    CompositionElement.BASS: {"pattern": "industrial_bass", "volume": 0.4},
                    CompositionElement.NOISE: {"type": "machinery", "volume": 0.3}
                },
                energy_level=0.5,
                tension_curve=np.linspace(0.3, 0.7, 12).tolist(),
                transition_out="tension_build"
            ),
            CompositionSection(
                name="Intense Build",
                start_bar=80,
                end_bar=144,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "industrial_drive", "volume": 0.8},
                    CompositionElement.BASS: {"pattern": "driving_bass", "volume": 0.7},
                    CompositionElement.LEAD: {"pattern": "industrial_lead", "volume": 0.6},
                    CompositionElement.PERCUSSION: {"pattern": "complex_industrial", "volume": 0.7},
                    CompositionElement.ATMOSPHERE: {"type": "tension_pad", "volume": 0.5},
                    CompositionElement.NOISE: {"type": "distorted_machinery", "volume": 0.4}
                },
                energy_level=0.8,
                tension_curve=np.linspace(0.7, 0.95, 16).tolist(),
                transition_out="dramatic_pause"
            ),
            CompositionSection(
                name="Climax",
                start_bar=144,
                end_bar=176,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.KICK: {"pattern": "industrial_maximum", "volume": 1.0},
                    CompositionElement.BASS: {"pattern": "maximum_bass", "volume": 0.9},
                    CompositionElement.LEAD: {"pattern": "screaming_lead", "volume": 0.8},
                    CompositionElement.PERCUSSION: {"pattern": "chaotic_industrial", "volume": 0.8},
                    CompositionElement.ATMOSPHERE: {"type": "chaos_pad", "volume": 0.7},
                    CompositionElement.NOISE: {"type": "white_noise_storm", "volume": 0.6},
                    CompositionElement.EFFECTS: {"type": "maximum_reverb", "volume": 0.5}
                },
                energy_level=1.0,
                tension_curve=[1.0] * 8,
                transition_out="gradual_decay"
            ),
            CompositionSection(
                name="Resolution",
                start_bar=176,
                end_bar=208,
                bpm=bpm,
                key=key,
                elements={
                    CompositionElement.ATMOSPHERE: {"type": "aftermath_drone", "volume": 0.4},
                    CompositionElement.NOISE: {"type": "distant_machinery", "volume": 0.2},
                    CompositionElement.EFFECTS: {"type": "long_decay", "volume": 0.3}
                },
                energy_level=0.1,
                tension_curve=np.linspace(0.5, 0.05, 8).tolist(),
                transition_out="fade_to_silence"
            )
        ]
        
        # Calculate energy curve
        energy_curve = []
        for section in sections:
            section_length = section.get_duration_bars()
            # Use tension curve if available, otherwise flat energy level
            if section.tension_curve and len(section.tension_curve) > 1:
                section_energy = np.interp(
                    np.linspace(0, len(section.tension_curve)-1, section_length),
                    range(len(section.tension_curve)),
                    section.tension_curve
                ).tolist()
            else:
                section_energy = [section.energy_level] * section_length
            energy_curve.extend(section_energy)
        
        return CompositionBlueprint(
            title=f"Industrial Journey {random.randint(1000, 9999)}",
            total_bars=208,
            base_bpm=bpm,
            base_key=key,
            style=CompositionStyle.ATMOSPHERIC,
            structure=CompositionStructure.INDUSTRIAL_JOURNEY,
            genre="industrial",
            target_energy_curve=energy_curve,
            sections=sections,
            global_effects={"reverb": 0.7, "delay": 0.4, "compression": 0.6},
            arrangement_notes=[
                "Long-form atmospheric journey",
                "Gradual tension building over 8 minutes",
                "Focus on texture and atmosphere",
                "Industrial soundscape with organic development"
            ],
            estimated_duration=total_bars * 4 * 60 / bpm
        )
    
    async def realize_blueprint(self,
                              blueprint: CompositionBlueprint,
                              synthesizer: AbstractSynthesizer,
                              pattern_generator: Callable) -> GeneratedComposition:
        # Similar implementation to HardcoreAnthemStrategy but with industrial-specific prompts
        # Implementation would be very similar, just different prompt generation
        patterns = {}
        arrangement = []
        
        # Implement industrial-specific pattern generation...
        # (Implementation omitted for brevity, would be similar to HardcoreAnthemStrategy)
        
        return GeneratedComposition(
            blueprint=blueprint,
            patterns=patterns,
            arrangement=arrangement,
            generation_metadata={
                "strategy": self.name,
                "patterns_generated": len(patterns),
                "generation_time": time.time()
            },
            confidence_score=0.80
        )


class AICompositionEngine:
    """Main AI composition engine"""
    
    def __init__(self,
                 synthesizer: AbstractSynthesizer,
                 conversation_engine: ConversationEngine,
                 audio_analyzer: AdvancedAudioAnalyzer,
                 evolution_engine: PatternEvolutionEngine):
        
        self.synthesizer = synthesizer
        self.conversation_engine = conversation_engine
        self.audio_analyzer = audio_analyzer
        self.evolution_engine = evolution_engine
        self.knowledge_base = hardcore_kb
        
        # Composition strategies
        self.strategies: Dict[CompositionStructure, AbstractCompositionStrategy] = {
            CompositionStructure.HARDCORE_ANTHEM: HardcoreAnthemStrategy(),
            CompositionStructure.INDUSTRIAL_JOURNEY: IndustrialJourneyStrategy(),
        }
        
        # Generated compositions storage
        self.compositions: Dict[str, GeneratedComposition] = {}
        
        # Composition statistics
        self.stats = {
            "compositions_created": 0,
            "total_patterns_generated": 0,
            "average_composition_time": 0.0,
            "most_popular_structure": CompositionStructure.HARDCORE_ANTHEM.value,
            "success_rate": 100.0
        }
    
    async def create_composition(self,
                               description: str,
                               requirements: Optional[Dict[str, Any]] = None,
                               structure: Optional[CompositionStructure] = None) -> GeneratedComposition:
        """Create a complete composition from natural language description"""
        
        start_time = time.time()
        
        try:
            # Parse requirements from description if not provided
            if not requirements:
                requirements = await self._parse_composition_requirements(description)
            
            # Determine structure if not specified
            if not structure:
                structure = self._determine_composition_structure(description, requirements)
            
            # Get appropriate strategy
            strategy = self.strategies.get(structure)
            if not strategy:
                raise ValueError(f"No strategy available for structure: {structure}")
            
            logger.info(f"Creating composition with {strategy.name} strategy")
            
            # Generate blueprint
            blueprint = await strategy.generate_blueprint(requirements, self.knowledge_base)
            
            # Realize blueprint into actual composition
            composition = await strategy.realize_blueprint(
                blueprint=blueprint,
                synthesizer=self.synthesizer,
                pattern_generator=self._generate_pattern_from_prompt
            )
            
            # Post-process composition
            composition = await self._post_process_composition(composition)
            
            # Store composition
            composition_id = str(uuid.uuid4())
            self.compositions[composition_id] = composition
            
            # Update statistics
            execution_time = time.time() - start_time
            self._update_stats(execution_time, composition)
            
            logger.info(f"Composition created successfully in {execution_time:.2f}s")
            
            return composition
            
        except Exception as e:
            logger.error(f"Composition creation failed: {e}")
            raise
    
    async def _parse_composition_requirements(self, description: str) -> Dict[str, Any]:
        """Parse natural language description into structured requirements"""
        
        requirements = {
            "bpm": 150,  # Default
            "key": "Am",
            "duration": 6.0,
            "genre": "gabber",
            "energy_level": 0.8,
            "artist_style": None,
            "mood": "aggressive"
        }
        
        description_lower = description.lower()
        
        # Extract BPM
        import re
        bpm_match = re.search(r'(\d{2,3})\s*bpm', description_lower)
        if bpm_match:
            requirements["bpm"] = int(bpm_match.group(1))
        
        # Extract duration
        duration_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)', description_lower)
        if duration_match:
            requirements["duration"] = float(duration_match.group(1))
        
        # Extract key
        key_match = re.search(r'\b([A-G]#?m?)\s*(?:key|major|minor)', description_lower)
        if key_match:
            requirements["key"] = key_match.group(1)
        
        # Extract genre
        genres = ["gabber", "hardcore", "industrial", "techno", "speedcore", "doomcore"]
        for genre in genres:
            if genre in description_lower:
                requirements["genre"] = genre
                break
        
        # Extract artist style
        artists = ["angerfist", "surgeon", "perc", "thunderdome", "ancient methods"]
        for artist in artists:
            if artist in description_lower:
                requirements["artist_style"] = artist
                break
        
        # Extract mood/energy descriptors
        if any(word in description_lower for word in ["brutal", "violent", "extreme", "maximum"]):
            requirements["energy_level"] = 1.0
        elif any(word in description_lower for word in ["hard", "aggressive", "intense"]):
            requirements["energy_level"] = 0.8
        elif any(word in description_lower for word in ["atmospheric", "dark", "ambient"]):
            requirements["energy_level"] = 0.4
        elif any(word in description_lower for word in ["minimal", "subtle", "quiet"]):
            requirements["energy_level"] = 0.2
        
        return requirements
    
    def _determine_composition_structure(self, 
                                       description: str, 
                                       requirements: Dict[str, Any]) -> CompositionStructure:
        """Determine the best composition structure based on description and requirements"""
        
        description_lower = description.lower()
        
        # Structure keywords
        if any(word in description_lower for word in ["anthem", "peak", "dancefloor", "crowd", "festival"]):
            return CompositionStructure.HARDCORE_ANTHEM
        elif any(word in description_lower for word in ["journey", "atmospheric", "story", "progression"]):
            return CompositionStructure.INDUSTRIAL_JOURNEY
        elif any(word in description_lower for word in ["assault", "brutal", "maximum", "destroy"]):
            return CompositionStructure.GABBER_ASSAULT
        elif any(word in description_lower for word in ["loop", "minimal", "hypnotic", "underground"]):
            return CompositionStructure.UNDERGROUND_LOOP
        
        # Based on genre
        genre = requirements.get("genre", "gabber")
        if genre == "gabber":
            return CompositionStructure.HARDCORE_ANTHEM
        elif genre == "industrial":
            return CompositionStructure.INDUSTRIAL_JOURNEY
        elif genre == "techno":
            return CompositionStructure.CLASSIC_TECHNO
        
        # Default
        return CompositionStructure.HARDCORE_ANTHEM
    
    async def _generate_pattern_from_prompt(self,
                                          prompt: str,
                                          bpm: int,
                                          genre: str) -> Optional[HardcorePattern]:
        """Generate a pattern from natural language prompt using conversation engine"""
        
        try:
            # Create a temporary session for pattern generation
            session_id = f"composition_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Enhance prompt with context
            enhanced_prompt = f"{prompt}. Generate Strudel code at {bpm} BPM in {genre} style."
            
            # Get AI response
            response = await self.conversation_engine.chat(session_id, enhanced_prompt)
            
            if response.code and response.confidence > 0.5:
                # Create pattern from generated code
                pattern = HardcorePattern(
                    name=f"ai_pattern_{int(time.time())}",
                    bpm=bpm,
                    pattern_data=response.code,
                    synth_type=self._determine_synth_type_from_prompt(prompt),
                    genre=genre
                )
                
                return pattern
            else:
                logger.warning(f"Failed to generate pattern from prompt: {prompt}")
                return None
                
        except Exception as e:
            logger.error(f"Pattern generation failed for prompt '{prompt}': {e}")
            return None
    
    def _determine_synth_type_from_prompt(self, prompt: str) -> SynthType:
        """Determine synth type from prompt content"""
        prompt_lower = prompt.lower()
        
        if "kick" in prompt_lower:
            if "gabber" in prompt_lower:
                return SynthType.GABBER_KICK
            elif "industrial" in prompt_lower:
                return SynthType.INDUSTRIAL_KICK
            else:
                return SynthType.GABBER_KICK
        elif "bass" in prompt_lower:
            return SynthType.ACID_BASS
        elif "lead" in prompt_lower:
            return SynthType.HOOVER_STAB
        else:
            return SynthType.GABBER_KICK  # Default
    
    async def _post_process_composition(self, composition: GeneratedComposition) -> GeneratedComposition:
        """Post-process composition with analysis and optimization"""
        
        try:
            # Generate audio if possible
            if composition.patterns:
                # For demo, we'll just analyze one pattern
                first_pattern = next(iter(composition.patterns.values()))
                audio_data = await self.synthesizer.play_pattern(first_pattern)
                
                if audio_data is not None:
                    composition.audio_data = audio_data
                    
                    # Analyze composition
                    analysis = await self.audio_analyzer.analyze_pattern_dna(audio_data)
                    composition.analysis_results = analysis.to_dict()
                    
                    # Update confidence based on analysis
                    if analysis.overall_quality > 0.7:
                        composition.confidence_score = min(1.0, composition.confidence_score + 0.1)
            
            # Add generation metadata
            composition.generation_metadata.update({
                "post_processed": True,
                "has_audio": composition.audio_data is not None,
                "has_analysis": composition.analysis_results is not None,
                "pattern_count": len(composition.patterns),
                "arrangement_events": len(composition.arrangement)
            })
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
        
        return composition
    
    def _update_stats(self, execution_time: float, composition: GeneratedComposition):
        """Update engine statistics"""
        self.stats["compositions_created"] += 1
        self.stats["total_patterns_generated"] += len(composition.patterns)
        
        # Update average composition time
        total_time = self.stats["average_composition_time"] * (self.stats["compositions_created"] - 1)
        self.stats["average_composition_time"] = (total_time + execution_time) / self.stats["compositions_created"]
        
        # Update success rate based on confidence
        if composition.confidence_score > 0.7:
            # Successful composition
            pass
        else:
            # Update success rate calculation would go here
            pass
    
    async def analyze_composition(self, composition_id: str) -> Dict[str, Any]:
        """Analyze a generated composition"""
        
        if composition_id not in self.compositions:
            raise ValueError(f"Composition {composition_id} not found")
        
        composition = self.compositions[composition_id]
        blueprint = composition.blueprint
        
        analysis = {
            "composition_id": composition_id,
            "blueprint_analysis": {
                "title": blueprint.title,
                "total_duration": blueprint.estimated_duration,
                "section_count": len(blueprint.sections),
                "energy_analysis": {
                    "min_energy": min(blueprint.target_energy_curve),
                    "max_energy": max(blueprint.target_energy_curve),
                    "avg_energy": sum(blueprint.target_energy_curve) / len(blueprint.target_energy_curve),
                    "energy_variance": np.var(blueprint.target_energy_curve)
                },
                "structure": blueprint.structure.value,
                "style": blueprint.style.value
            },
            "pattern_analysis": {
                "total_patterns": len(composition.patterns),
                "pattern_distribution": self._analyze_pattern_distribution(composition.patterns),
                "code_complexity": self._analyze_code_complexity(composition.patterns)
            },
            "arrangement_analysis": {
                "total_events": len(composition.arrangement),
                "event_types": self._analyze_arrangement_events(composition.arrangement),
                "timing_analysis": self._analyze_arrangement_timing(composition.arrangement)
            },
            "quality_metrics": {
                "confidence_score": composition.confidence_score,
                "generation_success": len(composition.patterns) > 0,
                "has_audio": composition.audio_data is not None,
                "has_analysis": composition.analysis_results is not None
            }
        }
        
        return analysis
    
    def _analyze_pattern_distribution(self, patterns: Dict[str, HardcorePattern]) -> Dict[str, int]:
        """Analyze distribution of pattern types"""
        distribution = defaultdict(int)
        
        for pattern in patterns.values():
            synth_type = pattern.synth_type.value
            distribution[synth_type] += 1
        
        return dict(distribution)
    
    def _analyze_code_complexity(self, patterns: Dict[str, HardcorePattern]) -> Dict[str, Any]:
        """Analyze complexity of generated code"""
        if not patterns:
            return {"average_length": 0, "total_lines": 0}
        
        total_length = sum(len(p.pattern_data) for p in patterns.values() if p.pattern_data)
        total_patterns = len([p for p in patterns.values() if p.pattern_data])
        
        return {
            "average_length": total_length / max(1, total_patterns),
            "total_lines": sum(p.pattern_data.count('\n') + 1 for p in patterns.values() if p.pattern_data),
            "patterns_with_code": total_patterns
        }
    
    def _analyze_arrangement_events(self, arrangement: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze types of arrangement events"""
        event_types = defaultdict(int)
        
        for event in arrangement:
            event_type = event.get("type", "unknown")
            event_types[event_type] += 1
        
        return dict(event_types)
    
    def _analyze_arrangement_timing(self, arrangement: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing of arrangement events"""
        if not arrangement:
            return {"span": 0, "density": 0}
        
        bars = [event.get("bar", 0) for event in arrangement]
        
        return {
            "first_event": min(bars),
            "last_event": max(bars),
            "span": max(bars) - min(bars),
            "density": len(arrangement) / max(1, max(bars) - min(bars))
        }
    
    def get_composition_library(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all compositions in library"""
        
        library = {}
        
        for comp_id, composition in self.compositions.items():
            library[comp_id] = {
                "title": composition.blueprint.title,
                "duration": composition.blueprint.estimated_duration,
                "bpm": composition.blueprint.base_bpm,
                "key": composition.blueprint.base_key,
                "genre": composition.blueprint.genre,
                "structure": composition.blueprint.structure.value,
                "style": composition.blueprint.style.value,
                "confidence": composition.confidence_score,
                "pattern_count": len(composition.patterns),
                "created_at": composition.generation_metadata.get("generation_time"),
                "has_audio": composition.audio_data is not None
            }
        
        return library
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.stats,
            "available_structures": [s.value for s in CompositionStructure],
            "available_strategies": len(self.strategies),
            "compositions_in_library": len(self.compositions)
        }


# Factory function
def create_ai_composition_engine(
    synthesizer: AbstractSynthesizer,
    conversation_engine: ConversationEngine,
    audio_analyzer: AdvancedAudioAnalyzer,
    evolution_engine: PatternEvolutionEngine
) -> AICompositionEngine:
    """Create AI composition engine with all dependencies"""
    
    return AICompositionEngine(
        synthesizer=synthesizer,
        conversation_engine=conversation_engine,
        audio_analyzer=audio_analyzer,
        evolution_engine=evolution_engine
    )


if __name__ == "__main__":
    # Demo the AI composition engine
    async def demo_ai_composition():
        from ..interfaces.synthesizer import MockSynthesizer
        from ..ai.conversation_engine import create_conversation_engine
        from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer
        from ..evolution.pattern_evolution_engine import PatternEvolutionEngine
        
        print("🤖 AI COMPOSITION ENGINE DEMO 🤖")
        print("=" * 50)
        
        # Create dependencies
        synth = MockSynthesizer()
        conv_engine = create_conversation_engine(synth)
        audio_analyzer = AdvancedAudioAnalyzer()
        evolution_engine = PatternEvolutionEngine()
        
        # Create composition engine
        comp_engine = create_ai_composition_engine(
            synthesizer=synth,
            conversation_engine=conv_engine,
            audio_analyzer=audio_analyzer,
            evolution_engine=evolution_engine
        )
        
        print("🎵 Creating hardcore anthem composition...")
        
        # Create test composition
        composition = await comp_engine.create_composition(
            description="Create a brutal hardcore anthem at 180 BPM with maximum energy for peak time dancefloor destruction",
            requirements={
                "bpm": 180,
                "duration": 6.0,
                "genre": "gabber",
                "energy_level": 1.0
            }
        )
        
        print(f"✅ Composition created: {composition.blueprint.title}")
        print(f"📊 Structure: {composition.blueprint.structure.value}")
        print(f"🎼 Sections: {len(composition.blueprint.sections)}")
        print(f"🎵 Patterns: {len(composition.patterns)}")
        print(f"⚡ Confidence: {composition.confidence_score:.2f}")
        print(f"⏱️ Estimated Duration: {composition.blueprint.estimated_duration:.1f}s")
        
        # Analyze composition
        comp_id = list(comp_engine.compositions.keys())[0]
        analysis = await comp_engine.analyze_composition(comp_id)
        
        print(f"\n📈 COMPOSITION ANALYSIS:")
        print(f"Energy Range: {analysis['blueprint_analysis']['energy_analysis']['min_energy']:.2f} - {analysis['blueprint_analysis']['energy_analysis']['max_energy']:.2f}")
        print(f"Pattern Types: {analysis['pattern_analysis']['pattern_distribution']}")
        print(f"Arrangement Events: {analysis['arrangement_analysis']['total_events']}")
        
        # Show library
        library = comp_engine.get_composition_library()
        print(f"\n📚 COMPOSITION LIBRARY ({len(library)} compositions)")
        for comp_id, info in library.items():
            print(f"• {info['title'][:30]}: {info['bpm']} BPM, {info['duration']:.1f}s")
        
        # Show engine stats
        stats = comp_engine.get_engine_stats()
        print(f"\n🔧 ENGINE STATS:")
        print(f"Compositions Created: {stats['compositions_created']}")
        print(f"Patterns Generated: {stats['total_patterns_generated']}")
        print(f"Average Creation Time: {stats['average_composition_time']:.2f}s")
        
        print("\n🎯 AI COMPOSITION DEMO COMPLETED")
    
    # Run demo
    asyncio.run(demo_ai_composition())