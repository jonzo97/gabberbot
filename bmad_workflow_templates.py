#!/usr/bin/env python3
"""
BMAD Production Workflow Templates - Ready-to-Use Examples
@music-archivist (Keeper) - Phase 4 Workflow Templates

Production-ready workflow templates for common hardcore music scenarios:
- Professional release workflows (singles, EPs, albums)
- DJ performance workflows (sets, tools, collections)
- Label production workflows (catalog building, batch production)
- Live performance workflows (real-time generation, adaptation)
- Educational workflows (learning, comparison, analysis)
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

# Import BMAD components
from bmad_hardcore_factory import (
    BMADHardcoreFactory, BMADFactoryConfig, ProductionMode, QualityLevel,
    generate_single_track, generate_quick_album, generate_dj_set
)
from bmad_simple_test import HardcoreStyle
from bmad_qa_suite import BMADQualityAssurance, validate_output_directory


class WorkflowTemplate(Enum):
    """Available workflow templates"""
    SINGLE_RELEASE = "single_release"
    EP_RELEASE = "ep_release"
    ALBUM_RELEASE = "album_release"
    DJ_PERFORMANCE_SET = "dj_performance_set"
    DJ_TOOLS_COLLECTION = "dj_tools_collection"
    LABEL_CATALOG_BUILD = "label_catalog_build"
    LIVE_PERFORMANCE = "live_performance"
    WAREHOUSE_SHOWCASE = "warehouse_showcase"
    EDUCATIONAL_COMPARISON = "educational_comparison"
    STYLE_EXPLORATION = "style_exploration"
    BPM_PROGRESSION_STUDY = "bpm_progression_study"
    REMIX_GENERATION = "remix_generation"


@dataclass
class WorkflowConfig:
    """Configuration for workflow templates"""
    template: WorkflowTemplate
    name: str
    description: str
    
    # Basic parameters
    style: HardcoreStyle = HardcoreStyle.ROTTERDAM_GABBER
    bpm_range: Tuple[float, float] = (180.0, 200.0)
    track_count: int = 1
    quality_level: QualityLevel = QualityLevel.PROFESSIONAL
    
    # Advanced parameters
    use_evolution: bool = True
    enable_mastering: bool = True
    enable_qa: bool = True
    
    # Output parameters
    output_prefix: str = ""
    create_documentation: bool = True
    create_metadata: bool = True
    
    # Performance parameters
    parallel_processing: bool = True
    max_generation_time_minutes: int = 30


@dataclass
class WorkflowResult:
    """Result from workflow execution"""
    template: WorkflowTemplate
    workflow_name: str
    execution_time_seconds: float
    
    # Generated content
    sessions: List[Any] = field(default_factory=list)
    output_directories: List[str] = field(default_factory=list)
    total_tracks_generated: int = 0
    
    # Quality metrics
    qa_results: Optional[Dict[str, Any]] = None
    quality_score: float = 0.0
    
    # Metadata
    configuration: Dict[str, Any] = field(default_factory=dict)
    documentation: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BMADWorkflowTemplates:
    """
    BMAD Production Workflow Templates
    
    Provides ready-to-use workflow templates for common hardcore music
    production scenarios, from single releases to complete label catalogs.
    """
    
    def __init__(self, output_directory: str = "bmad_workflow_output"):
        """Initialize workflow templates system"""
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        self.qa = BMADQualityAssurance()
        
        # Workflow tracking
        self.executed_workflows: List[WorkflowResult] = []
        
        self.logger.info("BMAD Workflow Templates initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for workflow templates"""
        logger = logging.getLogger("bmad_workflows")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def execute_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """Execute workflow template"""
        self.logger.info(f"Executing workflow: {config.name} ({config.template.value})")
        
        start_time = time.time()
        result = WorkflowResult(
            template=config.template,
            workflow_name=config.name,
            execution_time_seconds=0.0,
            configuration=asdict(config)
        )
        
        try:
            # Route to appropriate workflow implementation
            if config.template == WorkflowTemplate.SINGLE_RELEASE:
                await self._workflow_single_release(config, result)
            elif config.template == WorkflowTemplate.EP_RELEASE:
                await self._workflow_ep_release(config, result)
            elif config.template == WorkflowTemplate.ALBUM_RELEASE:
                await self._workflow_album_release(config, result)
            elif config.template == WorkflowTemplate.DJ_PERFORMANCE_SET:
                await self._workflow_dj_performance_set(config, result)
            elif config.template == WorkflowTemplate.DJ_TOOLS_COLLECTION:
                await self._workflow_dj_tools_collection(config, result)
            elif config.template == WorkflowTemplate.LABEL_CATALOG_BUILD:
                await self._workflow_label_catalog_build(config, result)
            elif config.template == WorkflowTemplate.LIVE_PERFORMANCE:
                await self._workflow_live_performance(config, result)
            elif config.template == WorkflowTemplate.WAREHOUSE_SHOWCASE:
                await self._workflow_warehouse_showcase(config, result)
            elif config.template == WorkflowTemplate.EDUCATIONAL_COMPARISON:
                await self._workflow_educational_comparison(config, result)
            elif config.template == WorkflowTemplate.STYLE_EXPLORATION:
                await self._workflow_style_exploration(config, result)
            elif config.template == WorkflowTemplate.BPM_PROGRESSION_STUDY:
                await self._workflow_bpm_progression_study(config, result)
            elif config.template == WorkflowTemplate.REMIX_GENERATION:
                await self._workflow_remix_generation(config, result)
            else:
                raise ValueError(f"Unknown workflow template: {config.template}")
            
            # Post-processing
            await self._post_process_workflow(config, result)
            
            result.execution_time_seconds = time.time() - start_time
            self.executed_workflows.append(result)
            
            self.logger.info(f"Workflow complete: {config.name} ({result.total_tracks_generated} tracks, {result.execution_time_seconds:.1f}s)")
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.execution_time_seconds = time.time() - start_time
            
            self.logger.error(f"Workflow failed: {config.name} - {e}")
            return result
    
    # Workflow implementations
    async def _workflow_single_release(self, config: WorkflowConfig, result: WorkflowResult):
        """Professional single track release workflow"""
        factory_config = BMADFactoryConfig(
            production_mode=ProductionMode.SINGLE_TRACK,
            style=config.style,
            bpm=config.bpm_range[0],
            quality_level=config.quality_level,
            use_evolution=config.use_evolution,
            enable_mastering=config.enable_mastering,
            album_name=f"{config.name}_Single",
            session_name=config.name
        )
        
        factory = BMADHardcoreFactory(factory_config)
        session = await factory.generate_hardcore_music(config.name)
        
        result.sessions.append(session)
        result.output_directories.append(str(session.output_directory))
        result.total_tracks_generated = session.tracks_generated
        
        # Create professional single package
        await self._create_single_package(session, config, result)
    
    async def _workflow_ep_release(self, config: WorkflowConfig, result: WorkflowResult):
        """Professional EP release workflow"""
        ep_track_count = config.track_count if config.track_count > 1 else 5
        
        factory_config = BMADFactoryConfig(
            production_mode=ProductionMode.ALBUM_EP,
            track_count=ep_track_count,
            style=config.style,
            bmp_progression=config.bpm_range,
            quality_level=config.quality_level,
            use_evolution=config.use_evolution,
            enable_mastering=config.enable_mastering,
            album_name=f"{config.name}_EP",
            session_name=config.name
        )
        
        factory = BMADHardcoreFactory(factory_config)
        session = await factory.generate_hardcore_music(config.name)
        
        result.sessions.append(session)
        result.output_directories.append(str(session.output_directory))
        result.total_tracks_generated = session.tracks_generated
        
        # Create professional EP package
        await self._create_ep_package(session, config, result)
    
    async def _workflow_album_release(self, config: WorkflowConfig, result: WorkflowResult):
        """Professional album release workflow"""
        album_track_count = config.track_count if config.track_count > 6 else 8
        
        factory_config = BMADFactoryConfig(
            production_mode=ProductionMode.FULL_ALBUM,
            track_count=album_track_count,
            style=config.style,
            bpm_progression=config.bmp_range,
            quality_level=QualityLevel.MASTERED,  # Force mastered for albums
            use_evolution=config.use_evolution,
            enable_mastering=True,
            album_name=f"{config.name}_Album",
            artist_name="BMAD Artists",
            session_name=config.name
        )
        
        factory = BMADHardcoreFactory(factory_config)
        session = await factory.generate_hardcore_music(config.name)
        
        result.sessions.append(session)
        result.output_directories.append(str(session.output_directory))
        result.total_tracks_generated = session.tracks_generated
        
        # Create professional album package
        await self._create_album_package(session, config, result)
    
    async def _workflow_dj_performance_set(self, config: WorkflowConfig, result: WorkflowResult):
        """DJ performance set workflow"""
        dj_track_count = config.track_count if config.track_count > 8 else 12
        
        factory_config = BMADFactoryConfig(
            production_mode=ProductionMode.DJ_SET,
            track_count=dj_track_count,
            bpm_progression=config.bpm_range,
            quality_level=config.quality_level,
            use_evolution=config.use_evolution,
            enable_mastering=True,  # DJ-optimized mastering
            session_name=f"{config.name}_DJ_Set"
        )
        
        factory = BMADHardcoreFactory(factory_config)
        session = await factory.generate_hardcore_music(config.name)
        
        result.sessions.append(session)
        result.output_directories.append(str(session.output_directory))
        result.total_tracks_generated = session.tracks_generated
        
        # Create DJ-ready package
        await self._create_dj_package(session, config, result)
    
    async def _workflow_dj_tools_collection(self, config: WorkflowConfig, result: WorkflowResult):
        """DJ tools and utility tracks workflow"""
        # Generate various DJ tools
        tool_configs = [
            ("Intro_Tool", 140.0, 32),
            ("Breakdown_Tool", 160.0, 64),
            ("Buildup_Tool", 180.0, 48),
            ("Peak_Tool", 200.0, 96),
            ("Outro_Tool", 180.0, 32)
        ]
        
        for tool_name, bpm, length in tool_configs:
            factory_config = BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                style=config.style,
                bpm=bpm,
                length_bars=length,
                quality_level=config.quality_level,
                session_name=f"{config.name}_{tool_name}"
            )
            
            factory = BMADHardcoreFactory(factory_config)
            session = await factory.generate_hardcore_music(f"{config.name}_{tool_name}")
            
            result.sessions.append(session)
            result.output_directories.append(str(session.output_directory))
            result.total_tracks_generated += session.tracks_generated
        
        # Create DJ tools package
        await self._create_dj_tools_package(result.sessions, config, result)
    
    async def _workflow_label_catalog_build(self, config: WorkflowConfig, result: WorkflowResult):
        """Label catalog building workflow"""
        catalog_size = config.track_count if config.track_count > 10 else 20
        
        # Generate diverse catalog content
        styles = [HardcoreStyle.ROTTERDAM_GABBER, HardcoreStyle.FRENCHCORE]
        bpm_ranges = [(160, 180), (180, 200), (200, 220)]
        qualities = [QualityLevel.STANDARD, QualityLevel.PROFESSIONAL, QualityLevel.MASTERED]
        
        for i in range(catalog_size):
            style = styles[i % len(styles)]
            bpm_range = bpm_ranges[i % len(bmp_ranges)]
            quality = qualities[i % len(qualities)]
            
            factory_config = BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                style=style,
                bpm=random.uniform(bpm_range[0], bmp_range[1]),
                quality_level=quality,
                use_evolution=config.use_evolution,
                session_name=f"{config.name}_Catalog_{i+1:03d}"
            )
            
            factory = BMADHardcoreFactory(factory_config)
            session = await factory.generate_hardcore_music(f"{config.name}_Catalog_{i+1:03d}")
            
            result.sessions.append(session)
            result.output_directories.append(str(session.output_directory))
            result.total_tracks_generated += session.tracks_generated
            
            # Progress logging
            if (i + 1) % 5 == 0:
                self.logger.info(f"Catalog progress: {i+1}/{catalog_size}")
        
        # Create label catalog package
        await self._create_label_package(result.sessions, config, result)
    
    async def _workflow_live_performance(self, config: WorkflowConfig, result: WorkflowResult):
        """Live performance workflow with real-time adaptation"""
        # Pre-generate base tracks for live performance
        base_tracks = 8
        
        for i in range(base_tracks):
            # Progressive BPM for live energy building
            bpm = config.bpm_range[0] + (config.bpm_range[1] - config.bpm_range[0]) * (i / (base_tracks - 1))
            
            factory_config = BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                style=config.style,
                bpm=bpm,
                length_bars=128,  # Longer tracks for live performance
                quality_level=config.quality_level,
                use_evolution=True,
                evolution_generations=5,  # Fast evolution for live
                session_name=f"{config.name}_Live_{i+1:02d}"
            )
            
            factory = BMADHardcoreFactory(factory_config)
            session = await factory.generate_hardcore_music(f"{config.name}_Live_{i+1:02d}")
            
            result.sessions.append(session)
            result.output_directories.append(str(session.output_directory))
            result.total_tracks_generated += session.tracks_generated
        
        # Create live performance package
        await self._create_live_package(result.sessions, config, result)
    
    async def _workflow_warehouse_showcase(self, config: WorkflowConfig, result: WorkflowResult):
        """Warehouse sound system showcase workflow"""
        # Generate tracks optimized for warehouse sound systems
        showcase_configs = [
            ("Bass_Test", 140.0, 64, "bass_heavy"),
            ("Kick_Power", 180.0, 96, "kick_focused"),
            ("Mid_Range_Clarity", 200.0, 64, "mid_focused"),
            ("High_Energy_Peak", 220.0, 128, "high_energy"),
            ("Full_Spectrum", 200.0, 160, "full_spectrum")
        ]
        
        for track_name, bpm, length, focus in showcase_configs:
            factory_config = BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                style=config.style,
                bpm=bpm,
                length_bars=length,
                quality_level=QualityLevel.MASTERED,
                enable_mastering=True,
                session_name=f"{config.name}_{track_name}"
            )
            
            factory = BMADHardcoreFactory(factory_config)
            session = await factory.generate_hardcore_music(f"{config.name}_{track_name}")
            
            result.sessions.append(session)
            result.output_directories.append(str(session.output_directory))
            result.total_tracks_generated += session.tracks_generated
        
        # Create warehouse showcase package
        await self._create_warehouse_package(result.sessions, config, result)
    
    async def _workflow_educational_comparison(self, config: WorkflowConfig, result: WorkflowResult):
        """Educational comparison workflow"""
        # Generate tracks for educational comparison
        comparison_configs = [
            ("Basic_Gabber", HardcoreStyle.ROTTERDAM_GABBER, 180.0, False),
            ("Evolved_Gabber", HardcoreStyle.ROTTERDAM_GABBER, 180.0, True),
            ("Basic_Frenchcore", HardcoreStyle.FRENCHCORE, 200.0, False),
            ("Evolved_Frenchcore", HardcoreStyle.FRENCHCORE, 200.0, True)
        ]
        
        for track_name, style, bmp, use_evolution in comparison_configs:
            factory_config = BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                style=style,
                bpm=bpm,
                length_bars=64,
                quality_level=config.quality_level,
                use_evolution=use_evolution,
                evolution_generations=15 if use_evolution else 0,
                session_name=f"{config.name}_{track_name}"
            )
            
            factory = BMADHardcoreFactory(factory_config)
            session = await factory.generate_hardcore_music(f"{config.name}_{track_name}")
            
            result.sessions.append(session)
            result.output_directories.append(str(session.output_directory))
            result.total_tracks_generated += session.tracks_generated
        
        # Create educational package
        await self._create_educational_package(result.sessions, config, result)
    
    async def _workflow_style_exploration(self, config: WorkflowConfig, result: WorkflowResult):
        """Style exploration and comparison workflow"""
        styles = [HardcoreStyle.ROTTERDAM_GABBER, HardcoreStyle.FRENCHCORE]
        
        for style in styles:
            # Generate multiple variations of each style
            for variation in range(3):
                bpm = config.bmp_range[0] + (config.bpm_range[1] - config.bpm_range[0]) * (variation / 2)
                
                factory_config = BMADFactoryConfig(
                    production_mode=ProductionMode.SINGLE_TRACK,
                    style=style,
                    bpm=bmp,
                    length_bars=64,
                    quality_level=config.quality_level,
                    use_evolution=config.use_evolution,
                    session_name=f"{config.name}_{style.value}_Var_{variation+1}"
                )
                
                factory = BMADHardcoreFactory(factory_config)
                session = await factory.generate_hardcore_music(f"{config.name}_{style.value}_Var_{variation+1}")
                
                result.sessions.append(session)
                result.output_directories.append(str(session.output_directory))
                result.total_tracks_generated += session.tracks_generated
        
        # Create style exploration package
        await self._create_style_package(result.sessions, config, result)
    
    async def _workflow_bmp_progression_study(self, config: WorkflowConfig, result: WorkflowResult):
        """BPM progression study workflow"""
        bmp_steps = 10
        bpm_step = (config.bpm_range[1] - config.bpm_range[0]) / (bpm_steps - 1)
        
        for i in range(bpm_steps):
            bpm = config.bpm_range[0] + (i * bpm_step)
            
            factory_config = BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                style=config.style,
                bpm=bpm,
                length_bars=48,
                quality_level=config.quality_level,
                use_evolution=config.use_evolution,
                session_name=f"{config.name}_BPM_{int(bpm):03d}"
            )
            
            factory = BMADHardcoreFactory(factory_config)
            session = await factory.generate_hardcore_music(f"{config.name}_BPM_{int(bpm):03d}")
            
            result.sessions.append(session)
            result.output_directories.append(str(session.output_directory))
            result.total_tracks_generated += session.tracks_generated
        
        # Create BPM study package
        await self._create_bmp_study_package(result.sessions, config, result)
    
    async def _workflow_remix_generation(self, config: WorkflowConfig, result: WorkflowResult):
        """Remix generation workflow"""
        # Generate original track
        original_config = BMADFactoryConfig(
            production_mode=ProductionMode.SINGLE_TRACK,
            style=config.style,
            bmp=config.bpm_range[0],
            quality_level=config.quality_level,
            use_evolution=False,
            session_name=f"{config.name}_Original"
        )
        
        factory = BMADHardcoreFactory(original_config)
        original_session = await factory.generate_hardcore_music(f"{config.name}_Original")
        
        result.sessions.append(original_session)
        result.output_directories.append(str(original_session.output_directory))
        result.total_tracks_generated += original_session.tracks_generated
        
        # Generate remixes with different parameters
        remix_configs = [
            ("Speed_Remix", config.bpm_range[1], True),
            ("Evolution_Remix", config.bpm_range[0], True),
            ("Style_Remix", config.bpm_range[0] + 20, True)
        ]
        
        for remix_name, bpm, use_evolution in remix_configs:
            remix_config = BMADFactoryConfig(
                production_mode=ProductionMode.SINGLE_TRACK,
                style=config.style,
                bpm=bpm,
                quality_level=config.quality_level,
                use_evolution=use_evolution,
                evolution_generations=20 if use_evolution else 0,
                session_name=f"{config.name}_{remix_name}"
            )
            
            factory = BMADHardcoreFactory(remix_config)
            remix_session = await factory.generate_hardcore_music(f"{config.name}_{remix_name}")
            
            result.sessions.append(remix_session)
            result.output_directories.append(str(remix_session.output_directory))
            result.total_tracks_generated += remix_session.tracks_generated
        
        # Create remix package
        await self._create_remix_package(result.sessions, config, result)
    
    # Package creation methods
    async def _create_single_package(self, session, config: WorkflowConfig, result: WorkflowResult):
        """Create professional single release package"""
        package_dir = self.output_directory / f"{config.name}_Single_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy main files
        await self._copy_session_files(session, package_dir)
        
        # Create single-specific documentation
        single_info = {
            "release_type": "Single",
            "title": config.name,
            "style": config.style.value,
            "bpm": config.bmp_range[0],
            "duration_estimate": "3-5 minutes",
            "quality": config.quality_level.value,
            "mastered": config.enable_mastering,
            "evolved": config.use_evolution
        }
        
        with open(package_dir / "single_info.json", 'w') as f:
            json.dump(single_info, f, indent=2)
        
        result.documentation["single_package"] = str(package_dir)
    
    async def _create_ep_package(self, session, config: WorkflowConfig, result: WorkflowResult):
        """Create professional EP release package"""
        package_dir = self.output_directory / f"{config.name}_EP_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy main files
        await self._copy_session_files(session, package_dir)
        
        # Create EP-specific documentation
        ep_info = {
            "release_type": "EP",
            "title": f"{config.name} EP",
            "track_count": result.total_tracks_generated,
            "style": config.style.value,
            "bpm_range": config.bpm_range,
            "total_duration_estimate": f"{result.total_tracks_generated * 6}-{result.total_tracks_generated * 8} minutes",
            "quality": config.quality_level.value,
            "mastered": config.enable_mastering,
            "evolved": config.use_evolution
        }
        
        with open(package_dir / "ep_info.json", 'w') as f:
            json.dump(ep_info, f, indent=2)
        
        result.documentation["ep_package"] = str(package_dir)
    
    async def _create_album_package(self, session, config: WorkflowConfig, result: WorkflowResult):
        """Create professional album release package"""
        package_dir = self.output_directory / f"{config.name}_Album_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy main files
        await self._copy_session_files(session, package_dir)
        
        # Create album-specific documentation
        album_info = {
            "release_type": "Album",
            "title": f"{config.name} Album",
            "artist": "BMAD Artists",
            "track_count": result.total_tracks_generated,
            "style": config.style.value,
            "bpm_progression": config.bpm_range,
            "total_duration_estimate": f"{result.total_tracks_generated * 7}-{result.total_tracks_generated * 9} minutes",
            "quality": "Mastered",
            "release_ready": True,
            "evolved": config.use_evolution
        }
        
        with open(package_dir / "album_info.json", 'w') as f:
            json.dump(album_info, f, indent=2)
        
        # Create tracklist
        tracklist = []
        for i in range(result.total_tracks_generated):
            bpm = config.bpm_range[0] + (config.bpm_range[1] - config.bpm_range[0]) * (i / max(1, result.total_tracks_generated - 1))
            tracklist.append({
                "track_number": i + 1,
                "title": f"{config.name} Track {i+1:02d}",
                "bpm": int(bpm),
                "style": config.style.value,
                "duration_estimate": "7-9 minutes"
            })
        
        with open(package_dir / "tracklist.json", 'w') as f:
            json.dump(tracklist, f, indent=2)
        
        result.documentation["album_package"] = str(package_dir)
    
    async def _create_dj_package(self, session, config: WorkflowConfig, result: WorkflowResult):
        """Create DJ-ready package"""
        package_dir = self.output_directory / f"{config.name}_DJ_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy main files
        await self._copy_session_files(session, package_dir)
        
        # Create DJ-specific documentation
        dj_info = {
            "package_type": "DJ Performance Set",
            "set_name": config.name,
            "track_count": result.total_tracks_generated,
            "bpm_progression": config.bpm_range,
            "recommended_order": "Sequential BPM progression",
            "mixing_notes": "Harmonic key progression for seamless mixing",
            "energy_curve": "Progressive build from warm-up to peak",
            "total_set_time": f"{result.total_tracks_generated * 8}-{result.total_tracks_generated * 10} minutes"
        }
        
        with open(package_dir / "dj_info.json", 'w') as f:
            json.dump(dj_info, f, indent=2)
        
        result.documentation["dj_package"] = str(package_dir)
    
    async def _create_dj_tools_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create DJ tools package"""
        package_dir = self.output_directory / f"{config.name}_DJ_Tools_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy files from all sessions
        for session in sessions:
            await self._copy_session_files(session, package_dir / session.session_name)
        
        # Create tools documentation
        tools_info = {
            "package_type": "DJ Tools Collection",
            "collection_name": config.name,
            "tool_count": len(sessions),
            "tools": [
                {"name": session.session_name, "purpose": "Performance utility"}
                for session in sessions
            ]
        }
        
        with open(package_dir / "tools_info.json", 'w') as f:
            json.dump(tools_info, f, indent=2)
        
        result.documentation["dj_tools_package"] = str(package_dir)
    
    async def _create_label_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create label catalog package"""
        package_dir = self.output_directory / f"{config.name}_Label_Catalog"
        package_dir.mkdir(exist_ok=True)
        
        # Organize by quality/style
        for session in sessions:
            style_dir = package_dir / session.config.style.value
            style_dir.mkdir(exist_ok=True)
            await self._copy_session_files(session, style_dir / session.session_name)
        
        # Create catalog documentation
        catalog_info = {
            "catalog_type": "Label Catalog",
            "label_name": config.name,
            "track_count": result.total_tracks_generated,
            "styles_included": list(set(session.config.style.value for session in sessions)),
            "quality_levels": list(set(session.config.quality_level.value for session in sessions)),
            "catalog_ready": True
        }
        
        with open(package_dir / "catalog_info.json", 'w') as f:
            json.dump(catalog_info, f, indent=2)
        
        result.documentation["label_package"] = str(package_dir)
    
    async def _create_live_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create live performance package"""
        package_dir = self.output_directory / f"{config.name}_Live_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy files from all sessions
        for i, session in enumerate(sessions):
            await self._copy_session_files(session, package_dir / f"Track_{i+1:02d}")
        
        # Create live performance documentation
        live_info = {
            "package_type": "Live Performance Set",
            "set_name": config.name,
            "track_count": result.total_tracks_generated,
            "performance_notes": "Pre-generated tracks for live performance with progressive BPM",
            "energy_progression": "Builds from opening to peak energy",
            "recommended_use": "Warehouse events, festival performances"
        }
        
        with open(package_dir / "live_info.json", 'w') as f:
            json.dump(live_info, f, indent=2)
        
        result.documentation["live_package"] = str(package_dir)
    
    async def _create_warehouse_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create warehouse showcase package"""
        package_dir = self.output_directory / f"{config.name}_Warehouse_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy files from all sessions
        for session in sessions:
            await self._copy_session_files(session, package_dir / session.session_name)
        
        # Create warehouse documentation
        warehouse_info = {
            "package_type": "Warehouse Sound System Showcase",
            "showcase_name": config.name,
            "track_count": result.total_tracks_generated,
            "optimization": "Optimized for large sound systems",
            "frequency_testing": "Full spectrum frequency response testing",
            "recommended_use": "Sound system testing, warehouse events"
        }
        
        with open(package_dir / "warehouse_info.json", 'w') as f:
            json.dump(warehouse_info, f, indent=2)
        
        result.documentation["warehouse_package"] = str(package_dir)
    
    async def _create_educational_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create educational package"""
        package_dir = self.output_directory / f"{config.name}_Educational_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Organize by comparison type
        for session in sessions:
            await self._copy_session_files(session, package_dir / session.session_name)
        
        # Create educational documentation
        edu_info = {
            "package_type": "Educational Comparison",
            "study_name": config.name,
            "comparison_count": result.total_tracks_generated,
            "learning_objectives": [
                "Compare basic vs evolved patterns",
                "Understand style differences",
                "Analyze evolution effects"
            ],
            "recommended_use": "Music production education, analysis"
        }
        
        with open(package_dir / "educational_info.json", 'w') as f:
            json.dump(edu_info, f, indent=2)
        
        result.documentation["educational_package"] = str(package_dir)
    
    async def _create_style_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create style exploration package"""
        package_dir = self.output_directory / f"{config.name}_Style_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Organize by style
        for session in sessions:
            style_name = session.session_name.split('_')[-2]  # Extract style from name
            style_dir = package_dir / style_name
            style_dir.mkdir(exist_ok=True)
            await self._copy_session_files(session, style_dir / session.session_name)
        
        # Create style documentation
        style_info = {
            "package_type": "Style Exploration",
            "exploration_name": config.name,
            "variation_count": result.total_tracks_generated,
            "styles_explored": list(set(s.value for s in [HardcoreStyle.ROTTERDAM_GABBER, HardcoreStyle.FRENCHCORE])),
            "analysis_focus": "Style characteristics and variations"
        }
        
        with open(package_dir / "style_info.json", 'w') as f:
            json.dump(style_info, f, indent=2)
        
        result.documentation["style_package"] = str(package_dir)
    
    async def _create_bmp_study_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create BPM study package"""
        package_dir = self.output_directory / f"{config.name}_BPM_Study_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy files from all sessions
        for session in sessions:
            await self._copy_session_files(session, package_dir / session.session_name)
        
        # Create BPM study documentation
        bpm_info = {
            "package_type": "BPM Progression Study",
            "study_name": config.name,
            "bpm_range": config.bpm_range,
            "step_count": result.total_tracks_generated,
            "analysis_focus": "BPM progression effects on energy and character",
            "recommended_use": "Tempo analysis, DJ set planning"
        }
        
        with open(package_dir / "bpm_study_info.json", 'w') as f:
            json.dump(bmp_info, f, indent=2)
        
        result.documentation["bpm_study_package"] = str(package_dir)
    
    async def _create_remix_package(self, sessions: List, config: WorkflowConfig, result: WorkflowResult):
        """Create remix package"""
        package_dir = self.output_directory / f"{config.name}_Remix_Package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy files from all sessions
        for session in sessions:
            await self._copy_session_files(session, package_dir / session.session_name)
        
        # Create remix documentation
        remix_info = {
            "package_type": "Remix Collection",
            "collection_name": config.name,
            "remix_count": result.total_tracks_generated,
            "original_included": True,
            "remix_types": ["Speed Remix", "Evolution Remix", "Style Remix"],
            "recommended_use": "Remix analysis, variation study"
        }
        
        with open(package_dir / "remix_info.json", 'w') as f:
            json.dump(remix_info, f, indent=2)
        
        result.documentation["remix_package"] = str(package_dir)
    
    # Helper methods
    async def _copy_session_files(self, session, destination_dir: Path):
        """Copy session files to destination"""
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        # Find source files
        source_dir = session.output_directory
        if source_dir.exists():
            for file_path in source_dir.glob("*"):
                if file_path.is_file():
                    dest_path = destination_dir / file_path.name
                    try:
                        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                            dst.write(src.read())
                    except Exception as e:
                        self.logger.warning(f"Failed to copy {file_path.name}: {e}")
    
    async def _post_process_workflow(self, config: WorkflowConfig, result: WorkflowResult):
        """Post-process workflow results"""
        # Run QA if enabled
        if config.enable_qa and result.output_directories:
            try:
                qa_results = []
                for output_dir in result.output_directories:
                    qa_result = await validate_output_directory(output_dir)
                    qa_results.append(qa_result)
                
                result.qa_results = qa_results
                
                # Calculate overall quality score
                if qa_results:
                    quality_scores = [qa["audio_quality_score"] for qa in qa_results if "audio_quality_score" in qa]
                    if quality_scores:
                        result.quality_score = sum(quality_scores) / len(quality_scores)
                
            except Exception as e:
                result.warnings.append(f"QA failed: {str(e)}")
        
        # Create workflow summary
        if config.create_documentation:
            await self._create_workflow_summary(config, result)
    
    async def _create_workflow_summary(self, config: WorkflowConfig, result: WorkflowResult):
        """Create workflow summary documentation"""
        summary_dir = self.output_directory / f"{config.name}_Summary"
        summary_dir.mkdir(exist_ok=True)
        
        # Create comprehensive summary
        summary = {
            "workflow_info": {
                "template": config.template.value,
                "name": config.name,
                "description": config.description,
                "execution_time": result.execution_time_seconds,
                "success": result.success
            },
            "configuration": asdict(config),
            "results": {
                "total_tracks": result.total_tracks_generated,
                "output_directories": result.output_directories,
                "quality_score": result.quality_score,
                "errors": result.errors,
                "warnings": result.warnings
            },
            "qa_results": result.qa_results,
            "documentation": result.documentation
        }
        
        with open(summary_dir / "workflow_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create README
        readme_content = self._generate_workflow_readme(config, result)
        with open(summary_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        result.documentation["workflow_summary"] = str(summary_dir)
    
    def _generate_workflow_readme(self, config: WorkflowConfig, result: WorkflowResult) -> str:
        """Generate README for workflow"""
        return f"""# {config.name} - {config.template.value.replace('_', ' ').title()}

## Workflow Summary

**Template**: {config.template.value}
**Description**: {config.description}
**Execution Time**: {result.execution_time_seconds:.1f} seconds
**Success**: {'✅ Yes' if result.success else '❌ No'}

## Results

- **Total Tracks Generated**: {result.total_tracks_generated}
- **Quality Score**: {result.quality_score:.2f}
- **Output Directories**: {len(result.output_directories)}

## Configuration

- **Style**: {config.style.value}
- **BPM Range**: {config.bpm_range[0]} - {config.bpm_range[1]}
- **Quality Level**: {config.quality_level.value}
- **Evolution Enabled**: {'Yes' if config.use_evolution else 'No'}
- **Mastering Enabled**: {'Yes' if config.enable_mastering else 'No'}

## Output Structure

{chr(10).join(f"- {Path(d).name}" for d in result.output_directories)}

## Usage

The generated tracks are ready for:
- DJ performance and mixing
- Music production and remixing
- Educational analysis and comparison
- Professional release and distribution

## Files Generated

Each output directory contains:
- MIDI files (.mid) for DAW import
- Audio files (.wav) for playback
- Session metadata (JSON)

## Quality Assurance

{'✅ QA checks passed' if result.qa_results and all(qa.get('overall_pass', False) for qa in result.qa_results) else '⚠️ Some QA checks failed' if result.qa_results else 'No QA performed'}

---

Generated by BMAD Workflow Templates - Phase 4 Production System
"""
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow system statistics"""
        return {
            "workflows_executed": len(self.executed_workflows),
            "total_tracks_generated": sum(w.total_tracks_generated for w in self.executed_workflows),
            "total_execution_time": sum(w.execution_time_seconds for w in self.executed_workflows),
            "success_rate": sum(1 for w in self.executed_workflows if w.success) / max(1, len(self.executed_workflows)) * 100,
            "average_quality_score": sum(w.quality_score for w in self.executed_workflows) / max(1, len(self.executed_workflows)),
            "templates_used": list(set(w.template.value for w in self.executed_workflows))
        }


# Convenience functions for common workflows
async def generate_professional_single(name: str, style: str = "gabber", bmp: float = 180.0) -> WorkflowResult:
    """Generate professional single release"""
    style_map = {"gabber": HardcoreStyle.ROTTERDAM_GABBER, "frenchcore": HardcoreStyle.FRENCHCORE}
    
    config = WorkflowConfig(
        template=WorkflowTemplate.SINGLE_RELEASE,
        name=name,
        description=f"Professional {style} single release",
        style=style_map.get(style, HardcoreStyle.ROTTERDAM_GABBER),
        bmp_range=(bpm, bpm),
        quality_level=QualityLevel.MASTERED
    )
    
    templates = BMADWorkflowTemplates()
    return await templates.execute_workflow(config)


async def generate_dj_performance_set(name: str, track_count: int = 12, bpm_range: Tuple[float, float] = (160.0, 200.0)) -> WorkflowResult:
    """Generate DJ performance set"""
    config = WorkflowConfig(
        template=WorkflowTemplate.DJ_PERFORMANCE_SET,
        name=name,
        description="Complete DJ performance set with BPM progression",
        track_count=track_count,
        bpm_range=bpm_range,
        quality_level=QualityLevel.PROFESSIONAL
    )
    
    templates = BMADWorkflowTemplates()
    return await templates.execute_workflow(config)


async def generate_educational_comparison(name: str) -> WorkflowResult:
    """Generate educational comparison set"""
    config = WorkflowConfig(
        template=WorkflowTemplate.EDUCATIONAL_COMPARISON,
        name=name,
        description="Educational comparison of basic vs evolved patterns",
        quality_level=QualityLevel.STANDARD
    )
    
    templates = BMADWorkflowTemplates()
    return await templates.execute_workflow(config)


# CLI Interface
def main():
    """Command line interface for workflow templates"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BMAD Workflow Templates")
    parser.add_argument("template", choices=[t.value for t in WorkflowTemplate], help="Workflow template")
    parser.add_argument("name", help="Workflow name")
    parser.add_argument("--description", default="BMAD workflow execution", help="Workflow description")
    parser.add_argument("--style", choices=["gabber", "frenchcore"], default="gabber", help="Hardcore style")
    parser.add_argument("--bpm-min", type=float, default=180.0, help="Minimum BPM")
    parser.add_argument("--bpm-max", type=float, default=200.0, help="Maximum BPM")
    parser.add_argument("--tracks", type=int, default=1, help="Number of tracks")
    parser.add_argument("--quality", choices=["draft", "standard", "professional", "mastered"], default="professional", help="Quality level")
    parser.add_argument("--no-evolution", action="store_true", help="Disable evolution")
    parser.add_argument("--no-mastering", action="store_true", help="Disable mastering")
    parser.add_argument("--no-qa", action="store_true", help="Disable quality assurance")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger("bmad_workflows").setLevel(logging.DEBUG)
    
    # Create configuration
    style_map = {"gabber": HardcoreStyle.ROTTERDAM_GABBER, "frenchcore": HardcoreStyle.FRENCHCORE}
    quality_map = {"draft": QualityLevel.DRAFT, "standard": QualityLevel.STANDARD, "professional": QualityLevel.PROFESSIONAL, "mastered": QualityLevel.MASTERED}
    
    config = WorkflowConfig(
        template=WorkflowTemplate(args.template),
        name=args.name,
        description=args.description,
        style=style_map[args.style],
        bpm_range=(args.bpm_min, args.bpm_max),
        track_count=args.tracks,
        quality_level=quality_map[args.quality],
        use_evolution=not args.no_evolution,
        enable_mastering=not args.no_mastering,
        enable_qa=not args.no_qa
    )
    
    # Execute workflow
    async def run_workflow():
        templates = BMADWorkflowTemplates()
        result = await templates.execute_workflow(config)
        
        print(f"\n🎭 BMAD Workflow Complete!")
        print(f"📝 Template: {args.template}")
        print(f"🎵 Tracks: {result.total_tracks_generated}")
        print(f"⏱️  Time: {result.execution_time_seconds:.1f}s")
        print(f"✅ Success: {'Yes' if result.success else 'No'}")
        print(f"🎯 Quality: {result.quality_score:.2f}")
        
        if result.output_directories:
            print(f"📁 Output:")
            for directory in result.output_directories:
                print(f"   {directory}")
        
        if result.documentation:
            print(f"📚 Documentation:")
            for doc_type, doc_path in result.documentation.items():
                print(f"   {doc_type}: {doc_path}")
        
        if result.errors:
            print(f"❌ Errors:")
            for error in result.errors:
                print(f"   {error}")
        
        if result.warnings:
            print(f"⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   {warning}")
        
        return result
    
    try:
        result = asyncio.run(run_workflow())
        print(f"\n🔥 Ready to destroy sound systems! 💀")
        return result
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()