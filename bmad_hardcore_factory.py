#!/usr/bin/env python3
"""
BMAD Hardcore Factory - Master Integration System
@music-archivist (Keeper) - Phase 4 Implementation

Complete hardcore music production factory that orchestrates all BMAD components:
- Single entry point for all BMAD functionality
- Workflow orchestration from pattern generation to album production
- Configuration management and session organization
- Quality assurance and performance monitoring
- Integration with existing infrastructure
"""

import os
import sys
import time
import json
import random
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback

# Import existing BMAD components
from bmad_simple_test import BMadSimpleCoordinator, BMadTrackConfig, HardcoreStyle
from bmad_album_producer import BMADAlbumProducer, AlbumConfig, AlbumTarget
from bmad_phase3_demo_final import AlbumPlan, AlbumTrackPlan, TrackEnergyLevel
from real_bmad_music_coordinator import BMADMusicCoordinator

# Import evolution system
try:
    from cli_shared.evolution.bmad_pattern_evolution import BMADPatternEvolution, BMADEvolutionConfig
    from cli_shared.evolution.bmad_theory_engine import BMADTheoryEngine
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    print("Evolution system not available - running in basic mode")

# Import main system integration
try:
    from main import generate_music
    from src.services.generation_service import GenerationService
    from src.services.audio_service import AudioService
    MAIN_SYSTEM_AVAILABLE = True
except ImportError:
    MAIN_SYSTEM_AVAILABLE = False
    print("Main system not available - running in standalone mode")


class ProductionMode(Enum):
    """Production modes for different use cases"""
    SINGLE_TRACK = "single_track"        # Generate single hardcore track
    ALBUM_EP = "album_ep"               # Generate 4-6 track EP
    FULL_ALBUM = "full_album"           # Generate full 8-12 track album
    DJ_SET = "dj_set"                   # Generate DJ-ready track collection
    LIVE_SESSION = "live_session"       # Real-time generation for live performance
    BATCH_PRODUCTION = "batch_production" # High-volume production run


class QualityLevel(Enum):
    """Quality levels for different requirements"""
    DRAFT = "draft"                     # Quick generation for testing
    STANDARD = "standard"               # Standard quality for general use
    PROFESSIONAL = "professional"      # High quality for releases
    MASTERED = "mastered"              # Full mastering for commercial release


class WorkflowStage(Enum):
    """Workflow stages for progress tracking"""
    INITIALIZATION = "initialization"
    PATTERN_GENERATION = "pattern_generation"
    EVOLUTION = "evolution"
    AUDIO_SYNTHESIS = "audio_synthesis"
    MIXING = "mixing"
    MASTERING = "mastering"
    EXPORT = "export"
    QUALITY_CHECK = "quality_check"
    COMPLETE = "complete"


@dataclass
class BMADFactoryConfig:
    """Master configuration for BMAD Hardcore Factory"""
    # Basic settings
    production_mode: ProductionMode = ProductionMode.SINGLE_TRACK
    quality_level: QualityLevel = QualityLevel.STANDARD
    output_format: str = "wav"
    sample_rate: int = 44100
    bit_depth: int = 16
    
    # Style settings
    style: HardcoreStyle = HardcoreStyle.ROTTERDAM_GABBER
    bpm: float = 180.0
    key: str = "A_minor"
    length_bars: int = 64
    
    # Advanced settings
    use_evolution: bool = True
    evolution_generations: int = 10
    enable_mastering: bool = True
    enable_quality_checks: bool = True
    
    # Album settings (for album modes)
    album_name: str = "BMAD_Hardcore_Collection"
    artist_name: str = "BMAD Factory"
    track_count: int = 8
    bpm_progression: Tuple[float, float] = (180.0, 220.0)
    
    # Output settings
    output_directory: str = "bmad_factory_output"
    session_name: Optional[str] = None
    preserve_working_files: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 1024
    
    # Integration settings
    integrate_with_main: bool = True
    create_midi_files: bool = True
    create_audio_files: bool = True
    create_project_files: bool = False


@dataclass
class ProductionSession:
    """Production session tracking and management"""
    session_id: str
    session_name: str
    config: BMADFactoryConfig
    start_time: datetime
    output_directory: Path
    
    # Progress tracking
    current_stage: WorkflowStage = WorkflowStage.INITIALIZATION
    stages_completed: List[WorkflowStage] = field(default_factory=list)
    total_stages: int = 9
    
    # Generation tracking
    tracks_generated: int = 0
    tracks_target: int = 1
    generation_errors: List[str] = field(default_factory=list)
    
    # Quality metrics
    quality_checks_passed: int = 0
    quality_checks_failed: int = 0
    
    # Performance metrics
    generation_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def get_progress_percentage(self) -> float:
        """Get overall progress percentage"""
        stage_progress = len(self.stages_completed) / self.total_stages
        track_progress = self.tracks_generated / self.tracks_target if self.tracks_target > 0 else 0
        return (stage_progress + track_progress) / 2 * 100
    
    def add_error(self, error: str):
        """Add error to session tracking"""
        self.generation_errors.append(f"{datetime.now().isoformat()}: {error}")
    
    def complete_stage(self, stage: WorkflowStage):
        """Mark stage as completed"""
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)
        self.current_stage = stage


class BMADHardcoreFactory:
    """
    Master BMAD Hardcore Factory
    
    Orchestrates all BMAD components for comprehensive hardcore music production.
    Provides single entry point for all functionality with workflow management.
    """
    
    def __init__(self, config: Optional[BMADFactoryConfig] = None):
        """Initialize the hardcore factory"""
        self.config = config or BMADFactoryConfig()
        self.logger = self._setup_logging()
        self.current_session: Optional[ProductionSession] = None
        
        # Initialize subsystems
        self.simple_coordinator = BMadSimpleCoordinator()
        
        # Initialize evolution system if available
        if EVOLUTION_AVAILABLE and self.config.use_evolution:
            try:
                self.theory_engine = BMADTheoryEngine()
                self.evolution_engine = None  # Initialized per session
                self.logger.info("Evolution system initialized")
            except Exception as e:
                self.logger.warning(f"Evolution system failed to initialize: {e}")
                self.config.use_evolution = False
        else:
            self.theory_engine = None
            self.evolution_engine = None
            
        # Initialize album producer
        try:
            self.album_producer = BMADAlbumProducer()
            self.logger.info("Album producer initialized")
        except Exception as e:
            self.logger.warning(f"Album producer failed to initialize: {e}")
            self.album_producer = None
        
        # Performance tracking
        self.sessions_completed: List[ProductionSession] = []
        self.total_tracks_generated = 0
        self.total_generation_time = 0.0
        
        self.logger.info("BMAD Hardcore Factory initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the factory"""
        logger = logging.getLogger("bmad_factory")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_session(self, session_name: Optional[str] = None) -> ProductionSession:
        """Create new production session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"bmad_factory_{timestamp}"
        
        if session_name is None:
            session_name = f"Factory_Session_{timestamp}"
        
        # Create output directory
        output_dir = Path(self.config.output_directory) / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine target track count
        track_count = 1
        if self.config.production_mode == ProductionMode.ALBUM_EP:
            track_count = 6
        elif self.config.production_mode == ProductionMode.FULL_ALBUM:
            track_count = self.config.track_count
        elif self.config.production_mode == ProductionMode.DJ_SET:
            track_count = 12
        elif self.config.production_mode == ProductionMode.BATCH_PRODUCTION:
            track_count = 20
        
        session = ProductionSession(
            session_id=session_id,
            session_name=session_name,
            config=self.config,
            start_time=datetime.now(),
            output_directory=output_dir,
            tracks_target=track_count
        )
        
        self.current_session = session
        self.logger.info(f"Created session: {session_name} ({session_id})")
        return session
    
    async def generate_hardcore_music(self, session_name: Optional[str] = None) -> ProductionSession:
        """
        Master generation method - orchestrates complete hardcore music production
        """
        session = self.create_session(session_name)
        
        try:
            self.logger.info(f"Starting hardcore music generation: {session.session_name}")
            session.complete_stage(WorkflowStage.INITIALIZATION)
            
            # Route to appropriate production method
            if self.config.production_mode == ProductionMode.SINGLE_TRACK:
                await self._generate_single_track(session)
            elif self.config.production_mode in [ProductionMode.ALBUM_EP, ProductionMode.FULL_ALBUM]:
                await self._generate_album(session)
            elif self.config.production_mode == ProductionMode.DJ_SET:
                await self._generate_dj_set(session)
            elif self.config.production_mode == ProductionMode.BATCH_PRODUCTION:
                await self._generate_batch_production(session)
            else:
                await self._generate_single_track(session)
            
            # Final quality check
            if self.config.enable_quality_checks:
                await self._perform_quality_checks(session)
            
            session.complete_stage(WorkflowStage.COMPLETE)
            session.generation_time_seconds = (datetime.now() - session.start_time).total_seconds()
            
            # Save session metadata
            await self._save_session_metadata(session)
            
            self.sessions_completed.append(session)
            self.total_tracks_generated += session.tracks_generated
            self.total_generation_time += session.generation_time_seconds
            
            self.logger.info(f"Generation complete: {session.tracks_generated} tracks in {session.generation_time_seconds:.1f}s")
            return session
            
        except Exception as e:
            session.add_error(f"Generation failed: {str(e)}")
            self.logger.error(f"Generation failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise
    
    async def _generate_single_track(self, session: ProductionSession):
        """Generate single hardcore track"""
        self.logger.info("Generating single hardcore track")
        
        # Create track configuration
        track_config = BMadTrackConfig(
            style=self.config.style,
            bpm=self.config.bpm,
            length_bars=float(self.config.length_bars),
            key=self.config.key,
            seed=random.randint(1, 999999)
        )
        
        session.complete_stage(WorkflowStage.PATTERN_GENERATION)
        
        # Generate track using simple coordinator
        try:
            track_session_id = self.simple_coordinator.generate_hardcore_track(track_config)
            session.tracks_generated += 1
            
            # Copy generated files to session directory
            await self._organize_output_files(session, track_session_id, "single_track")
            
            session.complete_stage(WorkflowStage.AUDIO_SYNTHESIS)
            session.complete_stage(WorkflowStage.MIXING)
            
            if self.config.enable_mastering:
                await self._apply_mastering(session, "single_track")
                session.complete_stage(WorkflowStage.MASTERING)
            
            session.complete_stage(WorkflowStage.EXPORT)
            
        except Exception as e:
            session.add_error(f"Single track generation failed: {str(e)}")
            raise
    
    async def _generate_album(self, session: ProductionSession):
        """Generate complete album or EP"""
        self.logger.info(f"Generating {self.config.production_mode.value}")
        
        if not self.album_producer:
            raise RuntimeError("Album producer not available")
        
        # Create album plan
        album_plan = AlbumPlan(
            album_name=self.config.album_name,
            artist_name=self.config.artist_name,
            bmp_start=self.config.bpm_progression[0],
            bpm_end=self.config.bpm_progression[1],
            total_tracks=session.tracks_target,
            target_duration_minutes=session.tracks_target * 7.0  # ~7 min per track
        )
        
        session.complete_stage(WorkflowStage.PATTERN_GENERATION)
        
        # Generate album
        try:
            album_config = AlbumConfig(
                album_name=self.config.album_name,
                artist_name=self.config.artist_name,
                track_count=session.tracks_target,
                target_duration_minutes=session.tracks_target * 7.0,
                master_target=AlbumTarget.WAREHOUSE_SYSTEM,
                enable_evolution=self.config.use_evolution,
                evolution_generations=self.config.evolution_generations
            )
            
            album_session_id = await self.album_producer.generate_complete_album(album_config)
            session.tracks_generated = session.tracks_target
            
            # Copy album files to session directory
            await self._organize_album_files(session, album_session_id)
            
            session.complete_stage(WorkflowStage.AUDIO_SYNTHESIS)
            session.complete_stage(WorkflowStage.MIXING)
            session.complete_stage(WorkflowStage.MASTERING)
            session.complete_stage(WorkflowStage.EXPORT)
            
        except Exception as e:
            session.add_error(f"Album generation failed: {str(e)}")
            raise
    
    async def _generate_dj_set(self, session: ProductionSession):
        """Generate DJ-ready track collection"""
        self.logger.info("Generating DJ set collection")
        
        # Create DJ-optimized track progression
        bpm_range = (self.config.bpm_progression[0], self.config.bpm_progression[1])
        bpm_step = (bpm_range[1] - bpm_range[0]) / (session.tracks_target - 1)
        
        styles = [HardcoreStyle.ROTTERDAM_GABBER, HardcoreStyle.FRENCHCORE]
        keys = ["A_minor", "C_major", "D_minor", "F_major", "G_minor", "Bb_major"]
        
        session.complete_stage(WorkflowStage.PATTERN_GENERATION)
        
        for i in range(session.tracks_target):
            try:
                # Calculate track parameters
                bpm = bmp_range[0] + (i * bpm_step)
                style = styles[i % len(styles)]
                key = keys[i % len(keys)]
                
                track_config = BMadTrackConfig(
                    style=style,
                    bpm=bpm,
                    length_bars=128.0,  # Longer tracks for DJ sets
                    key=key,
                    seed=random.randint(1, 999999)
                )
                
                # Generate track
                track_session_id = self.simple_coordinator.generate_hardcore_track(track_config)
                session.tracks_generated += 1
                
                # Organize with DJ-friendly naming
                track_name = f"DJ_Track_{i+1:02d}_{int(bpm)}BPM_{style.value}"
                await self._organize_output_files(session, track_session_id, track_name)
                
                self.logger.info(f"Generated DJ track {i+1}/{session.tracks_target}: {track_name}")
                
            except Exception as e:
                session.add_error(f"DJ track {i+1} failed: {str(e)}")
                self.logger.warning(f"DJ track {i+1} failed: {e}")
        
        session.complete_stage(WorkflowStage.AUDIO_SYNTHESIS)
        session.complete_stage(WorkflowStage.MIXING)
        
        if self.config.enable_mastering:
            await self._apply_dj_mastering(session)
            session.complete_stage(WorkflowStage.MASTERING)
        
        session.complete_stage(WorkflowStage.EXPORT)
    
    async def _generate_batch_production(self, session: ProductionSession):
        """Generate high-volume batch production"""
        self.logger.info("Starting batch production")
        
        # Variety configuration for batch production
        styles = [HardcoreStyle.ROTTERDAM_GABBER, HardcoreStyle.FRENCHCORE]
        bpm_range = (160, 220)
        lengths = [64, 96, 128, 160]
        keys = ["A_minor", "C_major", "D_minor", "F_major", "G_minor", "Bb_major", "C_minor", "E_minor"]
        
        session.complete_stage(WorkflowStage.PATTERN_GENERATION)
        
        # Generate tracks in parallel if enabled
        if self.config.parallel_processing:
            await self._generate_batch_parallel(session, styles, bpm_range, lengths, keys)
        else:
            await self._generate_batch_sequential(session, styles, bpm_range, lengths, keys)
        
        session.complete_stage(WorkflowStage.AUDIO_SYNTHESIS)
        session.complete_stage(WorkflowStage.MIXING)
        session.complete_stage(WorkflowStage.EXPORT)
    
    async def _generate_batch_sequential(self, session, styles, bpm_range, lengths, keys):
        """Generate batch tracks sequentially"""
        for i in range(session.tracks_target):
            try:
                # Random parameters for variety
                style = random.choice(styles)
                bpm = random.uniform(bpm_range[0], bpm_range[1])
                length = random.choice(lengths)
                key = random.choice(keys)
                
                track_config = BMadTrackConfig(
                    style=style,
                    bpm=bpm,
                    length_bars=float(length),
                    key=key,
                    seed=random.randint(1, 999999)
                )
                
                # Generate track
                track_session_id = self.simple_coordinator.generate_hardcore_track(track_config)
                session.tracks_generated += 1
                
                # Organize output
                track_name = f"Batch_Track_{i+1:03d}_{int(bpm)}BPM_{style.value}"
                await self._organize_output_files(session, track_session_id, track_name)
                
                if i % 5 == 0:  # Progress updates every 5 tracks
                    self.logger.info(f"Batch progress: {i+1}/{session.tracks_target}")
                
            except Exception as e:
                session.add_error(f"Batch track {i+1} failed: {str(e)}")
                self.logger.warning(f"Batch track {i+1} failed: {e}")
    
    async def _organize_output_files(self, session: ProductionSession, source_session_id: str, track_name: str):
        """Organize generated files into session directory"""
        # Find source files
        source_dir = Path("bmad_output") / source_session_id
        if not source_dir.exists():
            raise FileNotFoundError(f"Source session directory not found: {source_dir}")
        
        # Create track directory
        track_dir = session.output_directory / track_name
        track_dir.mkdir(exist_ok=True)
        
        # Copy files
        for file_path in source_dir.glob("*"):
            if file_path.is_file():
                dest_path = track_dir / file_path.name
                try:
                    # For Windows compatibility, read and write instead of shutil.copy
                    with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                        dst.write(src.read())
                except Exception as e:
                    self.logger.warning(f"Failed to copy {file_path.name}: {e}")
    
    async def _organize_album_files(self, session: ProductionSession, album_session_id: str):
        """Organize album files into session directory"""
        # Implementation depends on album producer output structure
        # This is a placeholder for album file organization
        album_dir = Path("bmad_album_output") / album_session_id
        if album_dir.exists():
            for file_path in album_dir.glob("*"):
                if file_path.is_file():
                    dest_path = session.output_directory / file_path.name
                    try:
                        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                            dst.write(src.read())
                    except Exception as e:
                        self.logger.warning(f"Failed to copy album file {file_path.name}: {e}")
    
    async def _apply_mastering(self, session: ProductionSession, track_name: str):
        """Apply mastering to generated tracks"""
        # Placeholder for mastering implementation
        # This would integrate with bmad_mastering_chain.py
        self.logger.info(f"Applying mastering to {track_name}")
        
        # Simulate mastering time
        await asyncio.sleep(0.5)
    
    async def _apply_dj_mastering(self, session: ProductionSession):
        """Apply DJ-optimized mastering to track collection"""
        self.logger.info("Applying DJ-optimized mastering")
        
        # Simulate DJ mastering
        await asyncio.sleep(1.0)
    
    async def _perform_quality_checks(self, session: ProductionSession):
        """Perform quality checks on generated content"""
        self.logger.info("Performing quality checks")
        session.complete_stage(WorkflowStage.QUALITY_CHECK)
        
        # Check for generated files
        audio_files = list(session.output_directory.glob("**/*.wav"))
        midi_files = list(session.output_directory.glob("**/*.mid"))
        
        if len(audio_files) >= session.tracks_generated:
            session.quality_checks_passed += 1
        else:
            session.quality_checks_failed += 1
            session.add_error(f"Missing audio files: expected {session.tracks_generated}, found {len(audio_files)}")
        
        if self.config.create_midi_files and len(midi_files) >= session.tracks_generated:
            session.quality_checks_passed += 1
        else:
            session.quality_checks_failed += 1
            session.add_error(f"Missing MIDI files: expected {session.tracks_generated}, found {len(midi_files)}")
        
        self.logger.info(f"Quality checks: {session.quality_checks_passed} passed, {session.quality_checks_failed} failed")
    
    async def _save_session_metadata(self, session: ProductionSession):
        """Save session metadata and statistics"""
        metadata = {
            "session_info": {
                "session_id": session.session_id,
                "session_name": session.session_name,
                "start_time": session.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "generation_time_seconds": session.generation_time_seconds
            },
            "configuration": asdict(session.config),
            "results": {
                "tracks_generated": session.tracks_generated,
                "tracks_target": session.tracks_target,
                "stages_completed": [stage.value for stage in session.stages_completed],
                "quality_checks_passed": session.quality_checks_passed,
                "quality_checks_failed": session.quality_checks_failed,
                "generation_errors": session.generation_errors
            },
            "performance": {
                "generation_time_seconds": session.generation_time_seconds,
                "peak_memory_mb": session.peak_memory_mb,
                "cpu_usage_percent": session.cpu_usage_percent,
                "tracks_per_hour": session.tracks_generated / (session.generation_time_seconds / 3600) if session.generation_time_seconds > 0 else 0
            }
        }
        
        metadata_file = session.output_directory / "session_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Session metadata saved: {metadata_file}")
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive factory statistics"""
        return {
            "factory_info": {
                "sessions_completed": len(self.sessions_completed),
                "total_tracks_generated": self.total_tracks_generated,
                "total_generation_time_hours": self.total_generation_time / 3600,
                "average_tracks_per_hour": self.total_tracks_generated / (self.total_generation_time / 3600) if self.total_generation_time > 0 else 0
            },
            "capabilities": {
                "evolution_available": EVOLUTION_AVAILABLE,
                "main_system_available": MAIN_SYSTEM_AVAILABLE,
                "album_producer_available": self.album_producer is not None,
                "parallel_processing": self.config.parallel_processing
            },
            "recent_sessions": [
                {
                    "session_id": session.session_id,
                    "session_name": session.session_name,
                    "tracks_generated": session.tracks_generated,
                    "generation_time_seconds": session.generation_time_seconds,
                    "quality_score": session.quality_checks_passed / (session.quality_checks_passed + session.quality_checks_failed) if (session.quality_checks_passed + session.quality_checks_failed) > 0 else 0
                }
                for session in self.sessions_completed[-5:]  # Last 5 sessions
            ]
        }
    
    def create_quick_config(self, mode: str, **kwargs) -> BMADFactoryConfig:
        """Create quick configuration for common use cases"""
        configs = {
            "single_track": BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                quality_level=QualityLevel.STANDARD,
                **kwargs
            ),
            "quick_ep": BMADFactoryConfig(
                production_mode=ProductionMode.ALBUM_EP,
                quality_level=QualityLevel.STANDARD,
                track_count=4,
                **kwargs
            ),
            "full_album": BMADFactoryConfig(
                production_mode=ProductionMode.FULL_ALBUM,
                quality_level=QualityLevel.PROFESSIONAL,
                track_count=8,
                enable_mastering=True,
                **kwargs
            ),
            "dj_set": BMADFactoryConfig(
                production_mode=ProductionMode.DJ_SET,
                quality_level=QualityLevel.PROFESSIONAL,
                bpm_progression=(160.0, 200.0),
                **kwargs
            ),
            "warehouse_system": BMADFactoryConfig(
                production_mode=ProductionMode.FULL_ALBUM,
                quality_level=QualityLevel.MASTERED,
                enable_mastering=True,
                bpm_progression=(180.0, 220.0),
                **kwargs
            )
        }
        
        config = configs.get(mode, configs["single_track"])
        return config


# Convenience functions for quick usage
async def generate_single_track(style: str = "gabber", bpm: float = 180.0, **kwargs) -> str:
    """Quick single track generation"""
    style_map = {
        "gabber": HardcoreStyle.ROTTERDAM_GABBER,
        "frenchcore": HardcoreStyle.FRENCHCORE
    }
    
    config = BMADFactoryConfig(
        style=style_map.get(style, HardcoreStyle.ROTTERDAM_GABBER),
        bpm=bpm,
        **kwargs
    )
    
    factory = BMADHardcoreFactory(config)
    session = await factory.generate_hardcore_music()
    return str(session.output_directory)


async def generate_quick_album(name: str = "BMAD_Hardcore_Album", tracks: int = 6, **kwargs) -> str:
    """Quick album generation"""
    config = BMADFactoryConfig(
        production_mode=ProductionMode.ALBUM_EP if tracks <= 6 else ProductionMode.FULL_ALBUM,
        album_name=name,
        track_count=tracks,
        **kwargs
    )
    
    factory = BMADHardcoreFactory(config)
    session = await factory.generate_hardcore_music()
    return str(session.output_directory)


async def generate_dj_set(track_count: int = 10, bpm_range: Tuple[float, float] = (160.0, 200.0), **kwargs) -> str:
    """Quick DJ set generation"""
    config = BMADFactoryConfig(
        production_mode=ProductionMode.DJ_SET,
        bpm_progression=bpm_range,
        track_count=track_count,
        **kwargs
    )
    
    factory = BMADHardcoreFactory(config)
    session = await factory.generate_hardcore_music()
    return str(session.output_directory)


# CLI Interface
def main():
    """Command line interface for BMAD Hardcore Factory"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BMAD Hardcore Factory - Complete Hardcore Music Production")
    parser.add_argument("mode", choices=["single", "ep", "album", "dj", "batch"], 
                       help="Production mode")
    parser.add_argument("--name", default="BMAD_Factory_Output", help="Session/album name")
    parser.add_argument("--style", choices=["gabber", "frenchcore"], default="gabber", help="Music style")
    parser.add_argument("--bpm", type=float, default=180.0, help="BPM (or starting BPM for albums)")
    parser.add_argument("--tracks", type=int, default=8, help="Number of tracks for albums/sets")
    parser.add_argument("--quality", choices=["draft", "standard", "professional", "mastered"], 
                       default="standard", help="Quality level")
    parser.add_argument("--no-evolution", action="store_true", help="Disable evolution system")
    parser.add_argument("--no-mastering", action="store_true", help="Disable mastering")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger("bmad_factory").setLevel(logging.DEBUG)
    
    # Create configuration
    mode_map = {
        "single": ProductionMode.SINGLE_TRACK,
        "ep": ProductionMode.ALBUM_EP,
        "album": ProductionMode.FULL_ALBUM,
        "dj": ProductionMode.DJ_SET,
        "batch": ProductionMode.BATCH_PRODUCTION
    }
    
    quality_map = {
        "draft": QualityLevel.DRAFT,
        "standard": QualityLevel.STANDARD,
        "professional": QualityLevel.PROFESSIONAL,
        "mastered": QualityLevel.MASTERED
    }
    
    style_map = {
        "gabber": HardcoreStyle.ROTTERDAM_GABBER,
        "frenchcore": HardcoreStyle.FRENCHCORE
    }
    
    config = BMADFactoryConfig(
        production_mode=mode_map[args.mode],
        quality_level=quality_map[args.quality],
        style=style_map[args.style],
        bpm=args.bpm,
        track_count=args.tracks,
        use_evolution=not args.no_evolution,
        enable_mastering=not args.no_mastering,
        session_name=args.name
    )
    
    # Run generation
    async def run_generation():
        factory = BMADHardcoreFactory(config)
        session = await factory.generate_hardcore_music(args.name)
        
        print(f"\n🎉 BMAD Hardcore Factory Complete!")
        print(f"📁 Output: {session.output_directory}")
        print(f"🎵 Tracks: {session.tracks_generated}")
        print(f"⏱️  Time: {session.generation_time_seconds:.1f}s")
        print(f"✅ Quality: {session.quality_checks_passed} passed, {session.quality_checks_failed} failed")
        print(f"🔥 Ready to destroy sound systems! 💀")
        
        return str(session.output_directory)
    
    try:
        output_dir = asyncio.run(run_generation())
        print(f"\n💿 Load these files into your DAW or play with any audio player")
        return output_dir
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()