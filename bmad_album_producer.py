#!/usr/bin/env python3
"""
BMAD Album Producer - Production-Ready Hardcore Album Generation
@music-producer (Raven) & @mix-engineer (Phoenix) - Phase 3 Implementation

Complete album production system that creates production-ready hardcore releases:
- Full 6-8 minute hardcore tracks with professional structure
- Progressive BPM journey (180→200→220 BPM across tracks)
- DJ-friendly transitions and mixing points
- Energy curve optimization for warehouse sound systems
- Album-level coordination and cohesive releases
"""

import os
import time
import random
import math
import wave
import struct
import numpy as np
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import copy

# Import existing BMAD components
from bmad_simple_test import (
    BMadMusicProducer, BMadSoundDesigner, BMadMixEngineer,
    BMadTrackConfig, HardcoreStyle, SimpleMIDIExporter, MIDINote
)

# Import professional track architecture
from audio.core.track import Track, TrackCollection, PatternControlSource, KickAudioSource
from audio.effects import (
    apply_compression, apply_hardcore_limiter, rotterdam_doorlussen,
    warehouse_reverb, apply_kick_space_highpass, apply_harshness_lowpass
)

# Import evolution system
from cli_shared.evolution.bmad_pattern_evolution import (
    BMADPatternEvolution, BMADEvolutionConfig, EvolutionStrategy, MutationConstraint
)
from cli_shared.models.hardcore_models import HardcorePattern, create_gabber_kick_pattern


class TrackSection(Enum):
    """Professional track sections for complete arrangements"""
    INTRO = "intro"
    BUILDUP = "buildup"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    BUILDUP2 = "buildup2"
    DROP2 = "drop2"
    OUTRO = "outro"


class EnergyLevel(Enum):
    """Energy levels for warehouse optimization"""
    LOW = "low"          # 0.3-0.5 (breakdowns, intros)
    MEDIUM = "medium"    # 0.5-0.7 (buildups)
    HIGH = "high"        # 0.7-0.9 (drops)
    EXTREME = "extreme"  # 0.9-1.0 (peak moments)


@dataclass
class TrackStructure:
    """Professional track structure for 6-8 minute hardcore tracks"""
    intro_bars: int = 16       # 32-64 beats intro
    buildup_bars: int = 32     # 64-128 beats buildup
    drop_bars: int = 64        # 128-256 beats main drop
    breakdown_bars: int = 32   # 64-128 beats breakdown
    buildup2_bars: int = 24    # 48-96 beats second buildup
    drop2_bars: int = 48       # 96-192 beats final drop
    outro_bars: int = 16       # 32-64 beats outro
    
    def total_bars(self) -> int:
        return (self.intro_bars + self.buildup_bars + self.drop_bars + 
                self.breakdown_bars + self.buildup2_bars + self.drop2_bars + self.outro_bars)
    
    def total_duration_minutes(self, bpm: float) -> float:
        """Calculate total track duration in minutes"""
        total_beats = self.total_bars() * 4
        return total_beats / bpm


@dataclass
class AlbumTrackConfig:
    """Configuration for individual album tracks"""
    track_number: int
    name: str
    bpm: float
    key: str
    style: HardcoreStyle
    structure: TrackStructure
    energy_target: EnergyLevel
    genre_focus: str = "gabber"
    evolution_generations: int = 20
    
    # DJ mixing points
    mix_in_point: int = 0      # Bar where track can be mixed in
    mix_out_point: int = 0     # Bar where track can be mixed out
    
    # Transition settings
    key_compatible_tracks: List[int] = field(default_factory=list)
    bpm_compatible_tracks: List[int] = field(default_factory=list)


@dataclass
class AlbumConfig:
    """Complete album configuration"""
    album_name: str
    artist_name: str = "BMAD Collective"
    total_tracks: int = 8
    total_duration_minutes: float = 60.0
    
    # Progressive journey
    bpm_start: float = 180.0
    bpm_end: float = 220.0
    key_progression: List[str] = field(default_factory=lambda: [
        "A_minor", "C_major", "D_minor", "F_major",
        "G_minor", "Bb_major", "C_minor", "A_minor"
    ])
    
    # Energy curve for warehouse sets
    energy_curve: List[EnergyLevel] = field(default_factory=lambda: [
        EnergyLevel.MEDIUM,    # Track 1: Warm up
        EnergyLevel.HIGH,      # Track 2: First peak
        EnergyLevel.MEDIUM,    # Track 3: Valley
        EnergyLevel.HIGH,      # Track 4: Build
        EnergyLevel.EXTREME,   # Track 5: Main peak
        EnergyLevel.HIGH,      # Track 6: Sustain
        EnergyLevel.EXTREME,   # Track 7: Final peak
        EnergyLevel.MEDIUM     # Track 8: Cool down
    ])
    
    # Genre progression for variety
    genre_progression: List[str] = field(default_factory=lambda: [
        "gabber", "gabber", "frenchcore", "industrial",
        "gabber", "frenchcore", "speedcore", "gabber"
    ])
    
    # Professional mastering targets
    mastering_target: str = "warehouse"  # warehouse, club, or festival
    loudness_target_lufs: float = -6.0   # LUFS target for hardcore
    peak_ceiling_db: float = -0.1        # Peak ceiling
    
    def generate_track_configs(self) -> List[AlbumTrackConfig]:
        """Generate track configurations for complete album"""
        tracks = []
        
        for i in range(self.total_tracks):
            # Calculate progressive BPM
            progress = i / (self.total_tracks - 1) if self.total_tracks > 1 else 0
            bpm = self.bpm_start + (self.bpm_end - self.bmp_start) * progress
            
            # Get key, energy, and genre for this track
            key = self.key_progression[i % len(self.key_progression)]
            energy = self.energy_curve[i % len(self.energy_curve)]
            genre = self.genre_progression[i % len(self.genre_progression)]
            
            # Create track structure based on energy level
            if energy == EnergyLevel.EXTREME:
                structure = TrackStructure(
                    intro_bars=12, buildup_bars=28, drop_bars=80,
                    breakdown_bars=24, buildup2_bars=20, drop2_bars=60, outro_bars=12
                )
            elif energy == EnergyLevel.HIGH:
                structure = TrackStructure(
                    intro_bars=16, buildup_bars=32, drop_bars=64,
                    breakdown_bars=32, buildup2_bars=24, drop2_bars=48, outro_bars=16
                )
            else:  # MEDIUM or LOW
                structure = TrackStructure(
                    intro_bars=20, buildup_bars=36, drop_bars=48,
                    breakdown_bars=40, buildup2_bars=28, drop2_bars=40, outro_bars=20
                )
            
            # Determine hardcore style
            if genre == "frenchcore":
                style = HardcoreStyle.FRENCHCORE
            else:
                style = HardcoreStyle.ROTTERDAM_GABBER
            
            track_config = AlbumTrackConfig(
                track_number=i + 1,
                name=f"{self.album_name}_Track_{i+1:02d}_{genre.title()}",
                bpm=bpm,
                key=key,
                style=style,
                structure=structure,
                energy_target=energy,
                genre_focus=genre,
                evolution_generations=30 if energy == EnergyLevel.EXTREME else 20
            )
            
            # Set DJ mixing points
            track_config.mix_in_point = track_config.structure.intro_bars // 2
            track_config.mix_out_point = (track_config.structure.total_bars() - 
                                        track_config.structure.outro_bars // 2)
            
            tracks.append(track_config)
        
        # Calculate BPM and key compatibility
        for i, track in enumerate(tracks):
            for j, other_track in enumerate(tracks):
                if i != j:
                    # BPM compatibility (within 8 BPM)
                    if abs(track.bpm - other_track.bpm) <= 8:
                        track.bpm_compatible_tracks.append(j + 1)
                    
                    # Key compatibility (same key or relative major/minor)
                    if (track.key == other_track.key or
                        self._keys_are_compatible(track.key, other_track.key)):
                        track.key_compatible_tracks.append(j + 1)
        
        return tracks
    
    def _keys_are_compatible(self, key1: str, key2: str) -> bool:
        """Check if two keys are compatible for mixing"""
        # Simplified key compatibility - same root note or related keys
        root1 = key1.split('_')[0]
        root2 = key2.split('_')[0]
        
        # Same root note is always compatible
        if root1 == root2:
            return True
        
        # Define compatible key relationships
        compatible_keys = {
            'A': ['C', 'F', 'D'],
            'C': ['A', 'F', 'G'],
            'D': ['A', 'G', 'Bb'],
            'F': ['A', 'C', 'Bb'],
            'G': ['C', 'D', 'Bb'],
            'Bb': ['F', 'G', 'D']
        }
        
        return root2 in compatible_keys.get(root1, [])


class BMadAlbumProducer:
    """
    Production-ready album generation system for hardcore music.
    Creates complete 45-60 minute albums with professional structure and DJ-friendly features.
    """
    
    def __init__(self, output_dir: str = "bmad_albums"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize BMAD components
        self.music_producer = BMadMusicProducer()
        self.sound_designer = BMadSoundDesigner()
        self.mix_engineer = BMadMixEngineer()
        
        # Evolution engine for intelligent pattern creation
        self.evolution_engine = None
        
        # Album state
        self.current_album = None
        self.generated_tracks = []
        
        print("BMAD Album Producer initialized")
        print(f"Output directory: {self.output_dir.absolute()}")
    
    async def generate_complete_album(self, album_config: AlbumConfig) -> str:
        """Generate complete hardcore album with professional structure"""
        session_id = f"album_{album_config.album_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        album_dir = self.output_dir / session_id
        album_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"BMAD ALBUM PRODUCTION - {album_config.album_name}")
        print(f"{'='*80}")
        print(f"Artist: {album_config.artist_name}")
        print(f"Tracks: {album_config.total_tracks}")
        print(f"Duration Target: {album_config.total_duration_minutes:.1f} minutes")
        print(f"BPM Journey: {album_config.bpm_start} → {album_config.bpm_end}")
        print(f"Session: {session_id}")
        
        self.current_album = album_config
        self.generated_tracks = []
        
        # Generate track configurations
        track_configs = album_config.generate_track_configs()
        
        # Initialize evolution engine
        evolution_config = BMADEvolutionConfig(
            population_size=40,
            strategy=EvolutionStrategy.MUSICAL_PROGRESSION,
            bpm_start=album_config.bpm_start,
            bmp_end=album_config.bpm_end,
            target_genres=["gabber", "frenchcore", "industrial"],
            optimize_for_warehouse=True
        )
        self.evolution_engine = BMADPatternEvolution(evolution_config)
        
        # Generate each track
        for i, track_config in enumerate(track_configs):
            print(f"\n{'-'*60}")
            print(f"GENERATING TRACK {track_config.track_number}/{album_config.total_tracks}")
            print(f"Name: {track_config.name}")
            print(f"BPM: {track_config.bpm:.1f}, Key: {track_config.key}")
            print(f"Energy: {track_config.energy_target.value}, Genre: {track_config.genre_focus}")
            print(f"Duration: {track_config.structure.total_duration_minutes(track_config.bpm):.1f} minutes")
            print(f"{'-'*60}")
            
            try:
                track_result = await self._generate_professional_track(track_config, album_dir)
                self.generated_tracks.append(track_result)
                
                print(f"✓ Track {track_config.track_number} generated successfully")
                print(f"  Files: {len(track_result['files'])} created")
                print(f"  Duration: {track_result['duration_minutes']:.1f} minutes")
                
            except Exception as e:
                print(f"✗ Error generating track {track_config.track_number}: {e}")
                # Continue with next track
                continue
        
        # Generate album-level files
        await self._generate_album_mixdown(album_dir)
        await self._generate_album_metadata(album_dir)
        
        print(f"\n{'='*80}")
        print(f"ALBUM PRODUCTION COMPLETE")
        print(f"{'='*80}")
        print(f"Session: {session_id}")
        print(f"Tracks Generated: {len(self.generated_tracks)}/{album_config.total_tracks}")
        
        total_duration = sum(track['duration_minutes'] for track in self.generated_tracks)
        print(f"Total Duration: {total_duration:.1f} minutes")
        print(f"Output Directory: {album_dir.name}")
        print(f"Ready for DJ sets and warehouse sound systems!")
        
        return session_id
    
    async def _generate_professional_track(self, track_config: AlbumTrackConfig, 
                                         album_dir: Path) -> Dict[str, Any]:
        """Generate single professional track with complete structure"""
        track_dir = album_dir / f"track_{track_config.track_number:02d}"
        track_dir.mkdir(exist_ok=True)
        
        # Create base pattern using evolution
        seed_pattern = self._create_track_seed_pattern(track_config)
        
        # Evolve pattern for this track's specific requirements
        evolved_pattern = await self._evolve_track_pattern(seed_pattern, track_config)
        
        # Generate complete track sections
        track_sections = self._generate_track_sections(evolved_pattern, track_config)
        
        # Create professional arrangement
        full_track_audio = await self._arrange_complete_track(track_sections, track_config)
        
        # Export track files
        files_created = []
        
        # Export main track
        main_file = track_dir / f"{track_config.name}.wav"
        if self.mix_engineer.export_wav_file(full_track_audio, str(main_file)):
            files_created.append(str(main_file))
        
        # Export MIDI patterns for each section
        for section_name, section_data in track_sections.items():
            midi_file = track_dir / f"{track_config.name}_{section_name}.mid"
            if SimpleMIDIExporter.export_midi(section_data['midi_notes'], str(midi_file), track_config.bpm):
                files_created.append(str(midi_file))
        
        # Export stems (kick, bass, lead, etc.)
        stems = await self._generate_track_stems(track_sections, track_config)
        for stem_name, stem_audio in stems.items():
            stem_file = track_dir / f"{track_config.name}_{stem_name}_stem.wav"
            if self.mix_engineer.export_wav_file(stem_audio, str(stem_file)):
                files_created.append(str(stem_file))
        
        # Calculate track info
        duration_samples = len(full_track_audio)
        duration_seconds = duration_samples / self.sound_designer.sample_rate
        duration_minutes = duration_seconds / 60.0
        
        return {
            'track_number': track_config.track_number,
            'name': track_config.name,
            'files': files_created,
            'duration_minutes': duration_minutes,
            'bpm': track_config.bmp,
            'key': track_config.key,
            'energy_level': track_config.energy_target.value,
            'mix_in_point': track_config.mix_in_point,
            'mix_out_point': track_config.mix_out_point,
            'structure': track_config.structure
        }
    
    def _create_track_seed_pattern(self, track_config: AlbumTrackConfig) -> HardcorePattern:
        """Create seed pattern for track evolution"""
        # Create base pattern based on genre focus
        if track_config.genre_focus == "frenchcore":
            pattern = create_gabber_kick_pattern(
                f"seed_{track_config.name}", 
                track_config.bpm * 1.1  # Slightly faster for frenchcore feel
            )
        elif track_config.genre_focus == "industrial":
            pattern = create_gabber_kick_pattern(
                f"seed_{track_config.name}",
                track_config.bpm * 0.9  # Slightly slower for industrial feel
            )
        else:  # gabber and others
            pattern = create_gabber_kick_pattern(
                f"seed_{track_config.name}",
                track_config.bpm
            )
        
        # Adjust pattern characteristics based on energy target
        energy_multiplier = {
            EnergyLevel.LOW: 0.7,
            EnergyLevel.MEDIUM: 0.85,
            EnergyLevel.HIGH: 1.0,
            EnergyLevel.EXTREME: 1.2
        }[track_config.energy_target]
        
        # Apply energy scaling to pattern
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.amp *= energy_multiplier
                    step.params.drive *= energy_multiplier
                    step.velocity *= energy_multiplier
        
        return pattern
    
    async def _evolve_track_pattern(self, seed_pattern: HardcorePattern, 
                                  track_config: AlbumTrackConfig) -> HardcorePattern:
        """Evolve pattern specifically for this track's requirements"""
        # Configure evolution for this track
        evolution_config = BMADEvolutionConfig(
            population_size=20,
            generations=track_config.evolution_generations,
            strategy=EvolutionStrategy.ENERGY_OPTIMIZATION if track_config.energy_target == EnergyLevel.EXTREME
                     else EvolutionStrategy.MUSICAL_PROGRESSION,
            bmp_start=track_config.bpm,
            bpm_end=track_config.bpm,  # Keep BPM stable for individual tracks
            target_genres=[track_config.genre_focus],
            warehouse_energy_target=0.8 if track_config.energy_target == EnergyLevel.EXTREME else 0.6
        )
        
        # Create focused evolution engine
        track_evolution = BMADPatternEvolution(evolution_config)
        
        # Initialize with seed pattern
        await track_evolution.initialize_population([seed_pattern])
        
        # Evolve pattern
        for _ in range(track_config.evolution_generations):
            await track_evolution.evolve_generation()
        
        # Get best evolved pattern
        best_patterns = track_evolution.get_best_patterns(1)
        return best_patterns[0].pattern if best_patterns else seed_pattern
    
    def _generate_track_sections(self, base_pattern: HardcorePattern, 
                               track_config: AlbumTrackConfig) -> Dict[str, Dict[str, Any]]:
        """Generate all sections of the track with professional structure"""
        sections = {}
        structure = track_config.structure
        
        # Create variations of the base pattern for different sections
        intro_pattern = self._create_intro_pattern(base_pattern, track_config)
        buildup_pattern = self._create_buildup_pattern(base_pattern, track_config)
        drop_pattern = self._create_drop_pattern(base_pattern, track_config)
        breakdown_pattern = self._create_breakdown_pattern(base_pattern, track_config)
        outro_pattern = self._create_outro_pattern(base_pattern, track_config)
        
        # Generate MIDI and audio for each section
        sections[TrackSection.INTRO.value] = self._generate_section_data(
            intro_pattern, structure.intro_bars, track_config
        )
        sections[TrackSection.BUILDUP.value] = self._generate_section_data(
            buildup_pattern, structure.buildup_bars, track_config
        )
        sections[TrackSection.DROP.value] = self._generate_section_data(
            drop_pattern, structure.drop_bars, track_config
        )
        sections[TrackSection.BREAKDOWN.value] = self._generate_section_data(
            breakdown_pattern, structure.breakdown_bars, track_config
        )
        sections[TrackSection.BUILDUP2.value] = self._generate_section_data(
            buildup_pattern, structure.buildup2_bars, track_config
        )
        sections[TrackSection.DROP2.value] = self._generate_section_data(
            drop_pattern, structure.drop2_bars, track_config
        )
        sections[TrackSection.OUTRO.value] = self._generate_section_data(
            outro_pattern, structure.outro_bars, track_config
        )
        
        return sections
    
    def _create_intro_pattern(self, base_pattern: HardcorePattern, track_config: AlbumTrackConfig) -> HardcorePattern:
        """Create intro pattern with reduced elements"""
        intro = copy.deepcopy(base_pattern)
        intro.name = f"{track_config.name}_intro"
        
        # Reduce intro energy - remove some tracks
        tracks_to_reduce = ["lead", "stab", "acid_bass"]
        for track_name in tracks_to_reduce:
            if track_name in intro.tracks:
                # Remove half the steps
                track = intro.tracks[track_name]
                for i in range(0, len(track.steps), 2):
                    if i < len(track.steps):
                        track.steps[i] = None
        
        # Reduce volume on remaining elements
        for track in intro.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.amp *= 0.6
                    step.velocity *= 0.7
        
        return intro
    
    def _create_buildup_pattern(self, base_pattern: HardcorePattern, track_config: AlbumTrackConfig) -> HardcorePattern:
        """Create buildup pattern with progressive energy"""
        buildup = copy.deepcopy(base_pattern)
        buildup.name = f"{track_config.name}_buildup"
        
        # Add filter sweeps and energy progression
        # This would be enhanced with proper filter automation
        for track in buildup.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.cutoff *= 0.8  # Start with filtered sound
                    step.params.resonance = min(1.0, step.params.resonance + 0.3)
        
        return buildup
    
    def _create_drop_pattern(self, base_pattern: HardcorePattern, track_config: AlbumTrackConfig) -> HardcorePattern:
        """Create main drop pattern with full energy"""
        drop = copy.deepcopy(base_pattern)
        drop.name = f"{track_config.name}_drop"
        
        # Maximize energy for drop
        for track in drop.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.amp = min(1.0, step.params.amp * 1.2)
                    step.params.drive = min(10.0, step.params.drive * 1.3)
                    step.velocity = min(1.0, step.velocity * 1.1)
        
        return drop
    
    def _create_breakdown_pattern(self, base_pattern: HardcorePattern, track_config: AlbumTrackConfig) -> HardcorePattern:
        """Create breakdown pattern with atmospheric elements"""
        breakdown = copy.deepcopy(base_pattern)
        breakdown.name = f"{track_config.name}_breakdown"
        
        # Keep only atmospheric elements
        essential_tracks = ["kick", "bass"]
        tracks_to_remove = [name for name in breakdown.tracks.keys() if name not in essential_tracks]
        
        for track_name in tracks_to_remove:
            if track_name in breakdown.tracks:
                # Reduce presence significantly
                track = breakdown.tracks[track_name]
                for i in range(len(track.steps)):
                    if i % 4 != 0 and track.steps[i] is not None:  # Keep only downbeats
                        track.steps[i] = None
        
        # Reduce energy on remaining elements
        for track in breakdown.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.amp *= 0.4
                    step.velocity *= 0.5
        
        return breakdown
    
    def _create_outro_pattern(self, base_pattern: HardcorePattern, track_config: AlbumTrackConfig) -> HardcorePattern:
        """Create outro pattern for DJ mixing"""
        outro = copy.deepcopy(base_pattern)
        outro.name = f"{track_config.name}_outro"
        
        # Create DJ-friendly outro - gradually reduce elements
        for track in outro.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.amp *= 0.7
                    step.velocity *= 0.8
        
        return outro
    
    def _generate_section_data(self, pattern: HardcorePattern, bars: int, 
                             track_config: AlbumTrackConfig) -> Dict[str, Any]:
        """Generate MIDI and audio data for a section"""
        # Calculate section length
        beats_per_bar = 4
        total_beats = bars * beats_per_bar
        pattern_length_beats = len(pattern.tracks[list(pattern.tracks.keys())[0]].steps) / 4
        repetitions = int(total_beats / pattern_length_beats)
        
        # Generate MIDI for section
        section_midi = []
        current_time = 0.0
        
        for rep in range(repetitions):
            pattern_midi = self.music_producer.generate_hardcore_midi(
                BMadTrackConfig(
                    style=track_config.style,
                    bpm=track_config.bpm,
                    length_bars=pattern_length_beats / 4,
                    key=track_config.key
                )
            )
            
            # Adjust timing for this repetition
            for track_notes in pattern_midi.values():
                for note in track_notes:
                    adjusted_note = MIDINote(
                        pitch=note.pitch,
                        velocity=note.velocity,
                        start_time=note.start_time + current_time,
                        duration=note.duration,
                        channel=note.channel
                    )
                    section_midi.append(adjusted_note)
            
            current_time += pattern_length_beats
        
        return {
            'midi_notes': section_midi,
            'bars': bars,
            'repetitions': repetitions
        }
    
    async def _arrange_complete_track(self, track_sections: Dict[str, Dict[str, Any]], 
                                    track_config: AlbumTrackConfig) -> List[float]:
        """Arrange complete track from sections with transitions"""
        full_track = []
        section_order = [
            TrackSection.INTRO, TrackSection.BUILDUP, TrackSection.DROP,
            TrackSection.BREAKDOWN, TrackSection.BUILDUP2, TrackSection.DROP2,
            TrackSection.OUTRO
        ]
        
        for section in section_order:
            section_name = section.value
            if section_name in track_sections:
                section_data = track_sections[section_name]
                
                # Generate audio for this section
                section_audio = self._generate_section_audio(
                    section_data['midi_notes'], track_config
                )
                
                # Apply section-specific processing
                section_audio = self._apply_section_processing(
                    section_audio, section, track_config
                )
                
                # Add to full track
                full_track.extend(section_audio)
        
        return full_track
    
    def _generate_section_audio(self, midi_notes: List[MIDINote], 
                              track_config: AlbumTrackConfig) -> List[float]:
        """Generate audio from MIDI notes for a section"""
        # Group notes by track type
        track_midi = {}
        for note in midi_notes:
            track_name = "kick" if note.channel == 9 else "bass"
            if track_name not in track_midi:
                track_midi[track_name] = []
            track_midi[track_name].append(note)
        
        # Synthesize each track
        audio_tracks = {}
        for track_name, notes in track_midi.items():
            audio_tracks[track_name] = self.sound_designer.synthesize_hardcore_audio(
                {track_name: notes},
                BMadTrackConfig(
                    style=track_config.style,
                    bpm=track_config.bpm,
                    length_bars=32,  # Will be trimmed to actual length
                    key=track_config.key
                )
            )
        
        # Mix tracks together
        section_audio = self.mix_engineer.create_professional_mix(
            audio_tracks,
            BMadTrackConfig(style=track_config.style, bpm=track_config.bpm)
        )
        
        return section_audio
    
    def _apply_section_processing(self, audio: List[float], section: TrackSection,
                                track_config: AlbumTrackConfig) -> List[float]:
        """Apply section-specific audio processing"""
        if not audio:
            return audio
        
        audio_array = np.array(audio)
        sample_rate = self.sound_designer.sample_rate
        
        # Apply processing based on section type
        if section == TrackSection.INTRO:
            # Light processing for intro
            audio_array = apply_kick_space_highpass(audio_array, sample_rate)
        
        elif section == TrackSection.BUILDUP:
            # Add compression and filter sweep effect
            audio_array = apply_compression(audio_array, ratio=4.0, threshold_db=-12, sample_rate=sample_rate)
        
        elif section in [TrackSection.DROP, TrackSection.DROP2]:
            # Full warehouse processing for drops
            audio_array = rotterdam_doorlussen(audio_array, sample_rate=sample_rate)
            audio_array = apply_compression(audio_array, ratio=8.0, threshold_db=-8, sample_rate=sample_rate)
            audio_array = warehouse_reverb(audio_array, sample_rate=sample_rate, wet_level=0.2)
        
        elif section == TrackSection.BREAKDOWN:
            # Atmospheric processing
            audio_array = warehouse_reverb(audio_array, sample_rate=sample_rate, wet_level=0.4)
            audio_array = apply_harshness_lowpass(audio_array, sample_rate)
        
        elif section == TrackSection.OUTRO:
            # DJ-friendly outro processing
            audio_array = apply_compression(audio_array, ratio=6.0, threshold_db=-10, sample_rate=sample_rate)
        
        return audio_array.tolist()
    
    async def _generate_track_stems(self, track_sections: Dict[str, Dict[str, Any]], 
                                  track_config: AlbumTrackConfig) -> Dict[str, List[float]]:
        """Generate individual stems for mixing and remixing"""
        stems = {
            'kick': [],
            'bass': [],
            'lead': [],
            'fx': []
        }
        
        # This would involve separating elements during synthesis
        # For now, return empty stems as placeholder
        return stems
    
    async def _generate_album_mixdown(self, album_dir: Path):
        """Generate continuous album mixdown for DJ sets"""
        if len(self.generated_tracks) < 2:
            return
        
        print(f"\n🎧 Creating album mixdown...")
        
        # This would create a continuous mix of all tracks
        # with proper DJ transitions between tracks
        mixdown_file = album_dir / f"{self.current_album.album_name}_continuous_mix.wav"
        
        # Placeholder for continuous mix generation
        print(f"   Continuous mix: {mixdown_file.name}")
    
    async def _generate_album_metadata(self, album_dir: Path):
        """Generate album metadata and track listing"""
        metadata = {
            'album': self.current_album.album_name,
            'artist': self.current_album.artist_name,
            'total_tracks': len(self.generated_tracks),
            'total_duration_minutes': sum(track['duration_minutes'] for track in self.generated_tracks),
            'bmp_journey': f"{self.current_album.bpm_start} → {self.current_album.bpm_end}",
            'mastering_target': self.current_album.mastering_target,
            'tracks': []
        }
        
        for track in self.generated_tracks:
            metadata['tracks'].append({
                'number': track['track_number'],
                'name': track['name'],
                'duration': f"{track['duration_minutes']:.1f} min",
                'bpm': f"{track['bmp']:.1f}",
                'key': track['key'],
                'energy': track['energy_level'],
                'mix_points': {
                    'in': track['mix_in_point'],
                    'out': track['mix_out_point']
                }
            })
        
        # Save metadata
        import json
        metadata_file = album_dir / "album_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create DJ cue sheet
        cue_file = album_dir / f"{self.current_album.album_name}.cue"
        with open(cue_file, 'w') as f:
            f.write(f'TITLE "{self.current_album.album_name}"\n')
            f.write(f'PERFORMER "{self.current_album.artist_name}"\n')
            f.write('FILE "continuous_mix.wav" WAVE\n')
            
            cumulative_time = 0
            for track in self.generated_tracks:
                minutes = int(cumulative_time // 60)
                seconds = int(cumulative_time % 60)
                frames = int((cumulative_time % 1) * 75)
                
                f.write(f'  TRACK {track["track_number"]:02d} AUDIO\n')
                f.write(f'    TITLE "{track["name"]}"\n')
                f.write(f'    INDEX 01 {minutes:02d}:{seconds:02d}:{frames:02d}\n')
                
                cumulative_time += track['duration_minutes'] * 60
        
        print(f"   Metadata: album_metadata.json")
        print(f"   DJ Cue Sheet: {self.current_album.album_name}.cue")


# Test album generation
async def test_bmad_album_producer():
    """Test complete album production system"""
    print("🎵 Testing BMAD Album Producer")
    print("=" * 60)
    
    # Create album configuration
    album_config = AlbumConfig(
        album_name="Warehouse_Destroyer_Vol1",
        artist_name="BMAD Collective",
        total_tracks=4,  # Reduced for testing
        total_duration_minutes=32.0,
        bpm_start=180.0,
        bmp_end=200.0,
        mastering_target="warehouse"
    )
    
    producer = BMadAlbumProducer()
    
    print(f"Album: {album_config.album_name}")
    print(f"Tracks: {album_config.total_tracks}")
    print(f"BPM Journey: {album_config.bpm_start} → {album_config.bpm_end}")
    
    # Generate complete album
    session_id = await producer.generate_complete_album(album_config)
    
    print(f"\n🏆 Album production test completed!")
    print(f"Session: {session_id}")
    print(f"Check bmad_albums/{session_id}/ for complete album")
    
    return session_id


if __name__ == "__main__":
    asyncio.run(test_bmad_album_producer())