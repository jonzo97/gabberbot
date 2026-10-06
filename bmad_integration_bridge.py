#!/usr/bin/env python3
"""
BMAD Integration Bridge - Main System and TUI Integration
@music-archivist (Keeper) - Phase 4 Integration Bridge

Complete integration bridge between BMAD Factory and existing systems:
- Integration with main.py CLI system
- TUI interface integration
- Unified command routing
- Cross-system workflow management
- Legacy compatibility layer
- Enhanced feature exposure
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Import BMAD Factory system
from bmad_hardcore_factory import (
    BMADHardcoreFactory, BMADFactoryConfig, ProductionMode, QualityLevel, WorkflowStage,
    generate_single_track, generate_quick_album, generate_dj_set
)
from bmad_simple_test import HardcoreStyle
from bmad_workflow_templates import BMADWorkflowTemplates, WorkflowTemplate, WorkflowConfig
from bmad_qa_suite import BMADQualityAssurance, quick_system_test, validate_output_directory
from bmad_performance_monitor import BMADPerformanceMonitor, monitor_generation, get_global_monitor

# Import existing system components
try:
    from main import generate_music as main_generate_music
    from src.services.generation_service import GenerationService
    from src.services.audio_service import AudioService
    from src.models.config import Settings
    from src.utils.env import load_settings
    MAIN_SYSTEM_AVAILABLE = True
except ImportError:
    MAIN_SYSTEM_AVAILABLE = False
    print("Main system not available - BMAD Factory will run in standalone mode")

try:
    from src.tui.app import TUIApp
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False
    print("TUI system not available")


class IntegrationMode(Enum):
    """Integration modes for different system combinations"""
    BMAD_ONLY = "bmad_only"
    MAIN_ONLY = "main_only"
    HYBRID = "hybrid"
    TUI_ENHANCED = "tui_enhanced"
    FULL_INTEGRATION = "full_integration"


class CommandSource(Enum):
    """Source of command execution"""
    CLI = "cli"
    TUI = "tui"
    API = "api"
    INTERNAL = "internal"


@dataclass
class IntegrationConfig:
    """Configuration for integration bridge"""
    integration_mode: IntegrationMode = IntegrationMode.FULL_INTEGRATION
    enable_main_system: bool = True
    enable_tui: bool = True
    enable_performance_monitoring: bool = True
    enable_quality_assurance: bool = True
    enable_workflow_templates: bool = True
    
    # Cross-system routing
    route_hardcore_to_bmad: bool = True
    route_other_to_main: bool = True
    enable_hybrid_workflows: bool = True
    
    # Output management
    unified_output_directory: bool = True
    output_directory: str = "unified_output"
    preserve_individual_outputs: bool = True
    
    # Performance settings
    auto_optimize: bool = True
    monitoring_interval: float = 1.0
    
    # Quality settings
    auto_qa: bool = True
    qa_threshold: float = 0.8


@dataclass
class UnifiedCommand:
    """Unified command structure for cross-system execution"""
    command_type: str
    prompt: str
    source: CommandSource
    parameters: Dict[str, Any]
    target_system: str = "auto"
    session_id: Optional[str] = None
    
    def is_hardcore_command(self) -> bool:
        """Determine if this is a hardcore music command"""
        hardcore_keywords = [
            "hardcore", "gabber", "frenchcore", "kickdrum", "acid bassline",
            "warehouse", "rotterdam", "speedcore", "industrial hardcore",
            "bmad", "factory", "evolution"
        ]
        
        prompt_lower = self.prompt.lower()
        return any(keyword in prompt_lower for keyword in hardcore_keywords)
    
    def extract_bmad_config(self) -> BMADFactoryConfig:
        """Extract BMAD configuration from command"""
        config = BMADFactoryConfig()
        
        # Extract style
        if "frenchcore" in self.prompt.lower():
            config.style = HardcoreStyle.FRENCHCORE
        elif "gabber" in self.prompt.lower() or "rotterdam" in self.prompt.lower():
            config.style = HardcoreStyle.ROTTERDAM_GABBER
        
        # Extract BPM
        import re
        bpm_match = re.search(r'(\d+)\s*bpm', self.prompt.lower())
        if bpm_match:
            config.bpm = float(bpm_match.group(1))
        
        # Extract production mode
        if "album" in self.prompt.lower():
            config.production_mode = ProductionMode.FULL_ALBUM
        elif "ep" in self.prompt.lower():
            config.production_mode = ProductionMode.ALBUM_EP
        elif "dj set" in self.prompt.lower() or "dj" in self.prompt.lower():
            config.production_mode = ProductionMode.DJ_SET
        else:
            config.production_mode = ProductionMode.SINGLE_TRACK
        
        # Extract quality level
        if "master" in self.prompt.lower() or "professional" in self.prompt.lower():
            config.quality_level = QualityLevel.MASTERED
        elif "draft" in self.prompt.lower() or "quick" in self.prompt.lower():
            config.quality_level = QualityLevel.DRAFT
        else:
            config.quality_level = QualityLevel.STANDARD
        
        # Apply parameters
        for key, value in self.parameters.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


class BMADIntegrationBridge:
    """
    BMAD Integration Bridge
    
    Provides seamless integration between BMAD Factory and existing systems,
    enabling unified workflows and cross-system functionality.
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """Initialize integration bridge"""
        self.config = config or IntegrationConfig()
        self.logger = self._setup_logging()
        
        # Initialize systems
        self.bmad_factory = BMADHardcoreFactory()
        self.workflow_templates = BMADWorkflowTemplates() if self.config.enable_workflow_templates else None
        self.qa_system = BMADQualityAssurance() if self.config.enable_quality_assurance else None
        self.performance_monitor = get_global_monitor() if self.config.enable_performance_monitoring else None
        
        # Main system integration
        self.main_system_available = MAIN_SYSTEM_AVAILABLE and self.config.enable_main_system
        if self.main_system_available:
            try:
                self.settings = load_settings()
                self.generation_service = GenerationService(self.settings)
                self.audio_service = AudioService(self.settings)
                self.logger.info("Main system integration enabled")
            except Exception as e:
                self.logger.warning(f"Main system integration failed: {e}")
                self.main_system_available = False
        
        # TUI integration
        self.tui_available = TUI_AVAILABLE and self.config.enable_tui
        
        # Output management
        self.output_directory = Path(self.config.output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Command history
        self.command_history: List[UnifiedCommand] = []
        self.execution_results: List[Dict[str, Any]] = []
        
        self.logger.info(f"BMAD Integration Bridge initialized - Mode: {self.config.integration_mode.value}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up integration bridge logging"""
        logger = logging.getLogger("bmad_integration")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def execute_unified_command(self, command: UnifiedCommand) -> Dict[str, Any]:
        """Execute unified command across systems"""
        self.logger.info(f"Executing command: {command.command_type} from {command.source.value}")
        
        # Add to history
        self.command_history.append(command)
        
        # Generate session ID
        if not command.session_id:
            command.session_id = f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Route command to appropriate system
        try:
            if command.target_system == "bmad" or (command.target_system == "auto" and command.is_hardcore_command()):
                result = await self._execute_bmad_command(command)
            elif command.target_system == "main" or (command.target_system == "auto" and self.main_system_available):
                result = await self._execute_main_command(command)
            elif command.target_system == "hybrid" and self.config.enable_hybrid_workflows:
                result = await self._execute_hybrid_command(command)
            else:
                # Fallback to BMAD
                result = await self._execute_bmad_command(command)
            
            # Post-processing
            await self._post_process_result(command, result)
            
            self.execution_results.append(result)
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "command": asdict(command),
                "system": "integration_bridge",
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.error(f"Command execution failed: {e}")
            self.execution_results.append(error_result)
            return error_result
    
    async def _execute_bmad_command(self, command: UnifiedCommand) -> Dict[str, Any]:
        """Execute command via BMAD Factory"""
        config = command.extract_bmad_config()
        config.session_name = command.session_id
        
        # Start performance monitoring if enabled
        if self.performance_monitor:
            with monitor_generation(command.session_id, config) as session:
                factory = BMADHardcoreFactory(config)
                bmad_session = await factory.generate_hardcore_music(command.session_id)
        else:
            factory = BMADHardcoreFactory(config)
            bmad_session = await factory.generate_hardcore_music(command.session_id)
        
        # Prepare result
        result = {
            "success": True,
            "system": "bmad_factory",
            "session_id": bmad_session.session_id,
            "output_directory": str(bmad_session.output_directory),
            "tracks_generated": bmad_session.tracks_generated,
            "generation_time": bmad_session.generation_time_seconds,
            "quality_score": bmad_session.quality_checks_passed / max(1, bmad_session.quality_checks_passed + bmad_session.quality_checks_failed),
            "configuration": asdict(config),
            "command": asdict(command),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    async def _execute_main_command(self, command: UnifiedCommand) -> Dict[str, Any]:
        """Execute command via main system"""
        if not self.main_system_available:
            raise RuntimeError("Main system not available")
        
        # Convert to main system format
        output_path = self.output_directory / f"{command.session_id}.wav"
        
        try:
            generated_file = await main_generate_music(
                prompt=command.prompt,
                output_path=output_path,
                settings=self.settings
            )
            
            result = {
                "success": True,
                "system": "main_system",
                "session_id": command.session_id,
                "output_file": str(generated_file),
                "output_directory": str(output_path.parent),
                "tracks_generated": 1,
                "command": asdict(command),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            result = {
                "success": False,
                "system": "main_system",
                "error": str(e),
                "command": asdict(command),
                "timestamp": datetime.now().isoformat()
            }
        
        return result
    
    async def _execute_hybrid_command(self, command: UnifiedCommand) -> Dict[str, Any]:
        """Execute hybrid command using both systems"""
        self.logger.info("Executing hybrid workflow")
        
        # Step 1: Generate base content with BMAD
        bmad_result = await self._execute_bmad_command(command)
        
        if not bmad_result["success"]:
            return bmad_result
        
        # Step 2: Enhance with main system if available
        if self.main_system_available:
            try:
                # Create enhancement prompt
                enhancement_prompt = f"enhance and master the hardcore track with additional effects and professional polish"
                
                # Find generated audio file
                output_dir = Path(bmad_result["output_directory"])
                audio_files = list(output_dir.glob("**/*.wav"))
                
                if audio_files:
                    # Enhance the first audio file
                    enhanced_file = await main_generate_music(
                        prompt=enhancement_prompt,
                        output_path=output_dir / f"{command.session_id}_enhanced.wav",
                        settings=self.settings
                    )
                    
                    bmad_result["enhanced_file"] = str(enhanced_file)
                    bmad_result["system"] = "hybrid_bmad_main"
                    bmad_result["enhancement"] = "main_system_polish"
                
            except Exception as e:
                self.logger.warning(f"Hybrid enhancement failed: {e}")
                bmad_result["enhancement_error"] = str(e)
        
        return bmad_result
    
    async def _post_process_result(self, command: UnifiedCommand, result: Dict[str, Any]):
        """Post-process command result"""
        if not result["success"]:
            return
        
        # Quality assurance
        if self.config.auto_qa and self.qa_system and "output_directory" in result:
            try:
                qa_result = await validate_output_directory(result["output_directory"])
                result["qa_results"] = qa_result
                result["qa_passed"] = qa_result.get("overall_pass", False)
                
                if result["qa_passed"]:
                    self.logger.info(f"QA passed for session {command.session_id}")
                else:
                    self.logger.warning(f"QA issues detected for session {command.session_id}")
                
            except Exception as e:
                self.logger.warning(f"QA failed for session {command.session_id}: {e}")
        
        # Unified output management
        if self.config.unified_output_directory:
            await self._manage_unified_output(command, result)
    
    async def _manage_unified_output(self, command: UnifiedCommand, result: Dict[str, Any]):
        """Manage unified output directory structure"""
        if "output_directory" not in result:
            return
        
        try:
            source_dir = Path(result["output_directory"])
            unified_dir = self.output_directory / command.session_id
            unified_dir.mkdir(exist_ok=True)
            
            # Copy files to unified directory
            if source_dir.exists():
                for file_path in source_dir.glob("**/*"):
                    if file_path.is_file():
                        dest_path = unified_dir / file_path.name
                        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                            dst.write(src.read())
            
            # Create session metadata
            session_metadata = {
                "session_id": command.session_id,
                "command": asdict(command),
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "unified_directory": str(unified_dir),
                "original_directory": str(source_dir)
            }
            
            metadata_file = unified_dir / "session_metadata.json"
            with open(metadata_file, 'w') as f:
                import json
                json.dump(session_metadata, f, indent=2, default=str)
            
            result["unified_directory"] = str(unified_dir)
            
        except Exception as e:
            self.logger.warning(f"Unified output management failed: {e}")
    
    # High-level workflow methods
    async def generate_hardcore_music(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """High-level hardcore music generation"""
        command = UnifiedCommand(
            command_type="generate_hardcore",
            prompt=prompt,
            source=CommandSource.API,
            parameters=kwargs,
            target_system="bmad"
        )
        
        return await self.execute_unified_command(command)
    
    async def generate_music_adaptive(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Adaptive music generation (auto-routing)"""
        command = UnifiedCommand(
            command_type="generate_adaptive",
            prompt=prompt,
            source=CommandSource.API,
            parameters=kwargs,
            target_system="auto"
        )
        
        return await self.execute_unified_command(command)
    
    async def execute_workflow_template(self, template: str, name: str, **kwargs) -> Dict[str, Any]:
        """Execute workflow template"""
        if not self.workflow_templates:
            raise RuntimeError("Workflow templates not available")
        
        # Map template string to enum
        template_map = {t.value: t for t in WorkflowTemplate}
        if template not in template_map:
            raise ValueError(f"Unknown template: {template}")
        
        # Create workflow config
        workflow_config = WorkflowConfig(
            template=template_map[template],
            name=name,
            description=f"Template execution: {template}",
            **kwargs
        )
        
        # Execute workflow
        workflow_result = await self.workflow_templates.execute_workflow(workflow_config)
        
        return {
            "success": workflow_result.success,
            "system": "workflow_template",
            "template": template,
            "workflow_name": name,
            "tracks_generated": workflow_result.total_tracks_generated,
            "execution_time": workflow_result.execution_time_seconds,
            "output_directories": workflow_result.output_directories,
            "quality_score": workflow_result.quality_score,
            "documentation": workflow_result.documentation,
            "timestamp": datetime.now().isoformat()
        }
    
    # TUI Integration methods
    def get_tui_commands(self) -> List[Dict[str, Any]]:
        """Get available commands for TUI integration"""
        commands = [
            {
                "name": "bmad_single",
                "description": "Generate single hardcore track with BMAD Factory",
                "category": "BMAD Generation",
                "parameters": ["style", "bpm", "quality"]
            },
            {
                "name": "bmad_album",
                "description": "Generate full hardcore album with BMAD Factory",
                "category": "BMAD Generation", 
                "parameters": ["track_count", "bpm_range", "quality"]
            },
            {
                "name": "bmad_dj_set",
                "description": "Generate DJ performance set",
                "category": "BMAD Generation",
                "parameters": ["track_count", "bpm_progression"]
            },
            {
                "name": "workflow_template",
                "description": "Execute production workflow template",
                "category": "Workflow Templates",
                "parameters": ["template", "name", "configuration"]
            },
            {
                "name": "hybrid_generate",
                "description": "Hybrid generation using BMAD + Main system",
                "category": "Hybrid Workflows",
                "parameters": ["prompt", "enhancement_level"]
            },
            {
                "name": "system_status",
                "description": "Get integrated system status",
                "category": "System Management",
                "parameters": []
            }
        ]
        
        if self.main_system_available:
            commands.extend([
                {
                    "name": "main_generate",
                    "description": "Generate music with main system",
                    "category": "Main System",
                    "parameters": ["prompt", "output_format"]
                }
            ])
        
        return commands
    
    async def execute_tui_command(self, command_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command from TUI"""
        command_map = {
            "bmad_single": self._tui_bmad_single,
            "bmad_album": self._tui_bmad_album,
            "bmad_dj_set": self._tui_bmad_dj_set,
            "workflow_template": self._tui_workflow_template,
            "hybrid_generate": self._tui_hybrid_generate,
            "main_generate": self._tui_main_generate,
            "system_status": self._tui_system_status
        }
        
        if command_name not in command_map:
            return {"success": False, "error": f"Unknown TUI command: {command_name}"}
        
        try:
            return await command_map[command_name](parameters)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _tui_bmad_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TUI command: BMAD single track"""
        style = params.get("style", "gabber")
        bpm = params.get("bpm", 180.0)
        quality = params.get("quality", "standard")
        
        prompt = f"Generate {style} hardcore track at {bpm} BPM with {quality} quality"
        
        return await self.generate_hardcore_music(prompt, 
                                                style=style, 
                                                bpm=bpm, 
                                                quality_level=quality)
    
    async def _tui_bmad_album(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TUI command: BMAD album"""
        track_count = params.get("track_count", 8)
        bmp_range = params.get("bpm_range", [180, 220])
        quality = params.get("quality", "professional")
        
        prompt = f"Generate hardcore album with {track_count} tracks from {bmp_range[0]} to {bmp_range[1]} BPM"
        
        return await self.generate_hardcore_music(prompt,
                                                track_count=track_count,
                                                bpm_progression=tuple(bpm_range),
                                                quality_level=quality,
                                                production_mode="full_album")
    
    async def _tui_bmad_dj_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TUI command: BMAD DJ set"""
        track_count = params.get("track_count", 12)
        bpm_progression = params.get("bmp_progression", [160, 200])
        
        prompt = f"Generate DJ set with {track_count} tracks from {bpm_progression[0]} to {bpm_progression[1]} BPM"
        
        return await self.generate_hardcore_music(prompt,
                                                track_count=track_count,
                                                bpm_progression=tuple(bpm_progression),
                                                production_mode="dj_set")
    
    async def _tui_workflow_template(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TUI command: Workflow template"""
        template = params.get("template", "single_release")
        name = params.get("name", f"TUI_Template_{datetime.now().strftime('%H%M%S')}")
        config = params.get("configuration", {})
        
        return await self.execute_workflow_template(template, name, **config)
    
    async def _tui_hybrid_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TUI command: Hybrid generation"""
        prompt = params.get("prompt", "Generate hardcore track")
        
        command = UnifiedCommand(
            command_type="hybrid_generate",
            prompt=prompt,
            source=CommandSource.TUI,
            parameters=params,
            target_system="hybrid"
        )
        
        return await self.execute_unified_command(command)
    
    async def _tui_main_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TUI command: Main system generation"""
        prompt = params.get("prompt", "Generate music")
        
        command = UnifiedCommand(
            command_type="main_generate",
            prompt=prompt,
            source=CommandSource.TUI,
            parameters=params,
            target_system="main"
        )
        
        return await self.execute_unified_command(command)
    
    async def _tui_system_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TUI command: System status"""
        status = {
            "integration_mode": self.config.integration_mode.value,
            "systems_available": {
                "bmad_factory": True,
                "main_system": self.main_system_available,
                "tui": self.tui_available,
                "workflow_templates": self.workflow_templates is not None,
                "quality_assurance": self.qa_system is not None,
                "performance_monitor": self.performance_monitor is not None
            },
            "command_history_count": len(self.command_history),
            "execution_results_count": len(self.execution_results),
            "output_directory": str(self.output_directory),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add performance status if available
        if self.performance_monitor:
            try:
                perf_status = self.performance_monitor.get_current_system_status()
                status["performance"] = perf_status
            except Exception as e:
                status["performance_error"] = str(e)
        
        return {"success": True, "status": status}
    
    # Statistics and monitoring
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration bridge statistics"""
        return {
            "integration_info": {
                "mode": self.config.integration_mode.value,
                "commands_executed": len(self.command_history),
                "successful_executions": sum(1 for r in self.execution_results if r.get("success", False)),
                "systems_used": list(set(r.get("system", "unknown") for r in self.execution_results))
            },
            "system_availability": {
                "bmad_factory": True,
                "main_system": self.main_system_available,
                "tui": self.tui_available,
                "workflow_templates": self.workflow_templates is not None,
                "quality_assurance": self.qa_system is not None,
                "performance_monitor": self.performance_monitor is not None
            },
            "recent_commands": [
                {
                    "command_type": cmd.command_type,
                    "source": cmd.source.value,
                    "target_system": cmd.target_system,
                    "is_hardcore": cmd.is_hardcore_command()
                }
                for cmd in self.command_history[-10:]  # Last 10 commands
            ],
            "performance_summary": self.performance_monitor.get_performance_summary(24) if self.performance_monitor else None
        }


# Global integration bridge instance
_global_bridge: Optional[BMADIntegrationBridge] = None

def get_integration_bridge(config: Optional[IntegrationConfig] = None) -> BMADIntegrationBridge:
    """Get or create global integration bridge"""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = BMADIntegrationBridge(config)
    return _global_bridge


# Enhanced main function with BMAD integration
async def enhanced_main_generate_music(prompt: str, **kwargs) -> str:
    """Enhanced main music generation with BMAD integration"""
    bridge = get_integration_bridge()
    
    # Create unified command
    command = UnifiedCommand(
        command_type="generate_music",
        prompt=prompt,
        source=CommandSource.CLI,
        parameters=kwargs,
        target_system="auto"
    )
    
    # Execute command
    result = await bridge.execute_unified_command(command)
    
    if result["success"]:
        return result.get("unified_directory", result.get("output_directory", ""))
    else:
        raise RuntimeError(result.get("error", "Generation failed"))


# TUI Integration class
class EnhancedTUICommands:
    """Enhanced TUI commands with BMAD integration"""
    
    def __init__(self):
        self.bridge = get_integration_bridge()
    
    async def get_available_commands(self) -> List[Dict[str, Any]]:
        """Get available TUI commands"""
        return self.bridge.get_tui_commands()
    
    async def execute_command(self, command_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TUI command"""
        return await self.bridge.execute_tui_command(command_name, parameters)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status for TUI"""
        return await self.bridge.execute_tui_command("system_status", {})


# CLI Interface
def main():
    """Command line interface for integration bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BMAD Integration Bridge")
    parser.add_argument("command", choices=["generate", "workflow", "status", "test"], help="Bridge command")
    parser.add_argument("prompt", nargs="?", help="Generation prompt")
    parser.add_argument("--system", choices=["auto", "bmad", "main", "hybrid"], default="auto", help="Target system")
    parser.add_argument("--template", help="Workflow template name")
    parser.add_argument("--name", help="Session/workflow name")
    parser.add_argument("--style", choices=["gabber", "frenchcore"], default="gabber", help="Hardcore style")
    parser.add_argument("--bpm", type=float, default=180.0, help="BPM")
    parser.add_argument("--quality", choices=["draft", "standard", "professional", "mastered"], default="standard", help="Quality level")
    parser.add_argument("--mode", choices=[m.value for m in IntegrationMode], default="full_integration", help="Integration mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger("bmad_integration").setLevel(logging.DEBUG)
    
    # Create integration config
    config = IntegrationConfig(
        integration_mode=IntegrationMode(args.mode),
        enable_main_system=MAIN_SYSTEM_AVAILABLE,
        enable_tui=TUI_AVAILABLE
    )
    
    bridge = BMADIntegrationBridge(config)
    
    async def run_command():
        if args.command == "generate":
            if not args.prompt:
                print("❌ Prompt required for generate command")
                return
            
            # Create command
            command = UnifiedCommand(
                command_type="generate",
                prompt=args.prompt,
                source=CommandSource.CLI,
                parameters={
                    "style": args.style,
                    "bpm": args.bpm,
                    "quality_level": args.quality
                },
                target_system=args.system
            )
            
            # Execute
            result = await bridge.execute_unified_command(command)
            
            if result["success"]:
                print(f"\n🎵 Generation Complete!")
                print(f"📁 Output: {result.get('unified_directory', result.get('output_directory'))}")
                print(f"🎛️  System: {result['system']}")
                print(f"🎵 Tracks: {result.get('tracks_generated', 1)}")
                print(f"⏱️  Time: {result.get('generation_time', 0):.1f}s")
                
                if "qa_results" in result:
                    qa_status = "✅ Passed" if result.get("qa_passed") else "⚠️ Issues"
                    print(f"🔍 QA: {qa_status}")
                
            else:
                print(f"❌ Generation failed: {result.get('error')}")
        
        elif args.command == "workflow":
            if not args.template or not args.name:
                print("❌ Template and name required for workflow command")
                return
            
            result = await bridge.execute_workflow_template(
                args.template,
                args.name,
                style=args.style,
                bpm_range=(args.bpm, args.bpm + 20),
                quality_level=args.quality
            )
            
            if result["success"]:
                print(f"\n🎭 Workflow Complete!")
                print(f"📝 Template: {args.template}")
                print(f"🎵 Tracks: {result['tracks_generated']}")
                print(f"⏱️  Time: {result['execution_time']:.1f}s")
                print(f"📁 Output: {result['output_directories']}")
            else:
                print(f"❌ Workflow failed: {result.get('error')}")
        
        elif args.command == "status":
            stats = bridge.get_integration_statistics()
            
            print(f"\n🔗 BMAD Integration Bridge Status")
            print(f"Mode: {stats['integration_info']['mode']}")
            print(f"Commands Executed: {stats['integration_info']['commands_executed']}")
            print(f"Success Rate: {stats['integration_info']['successful_executions']}/{stats['integration_info']['commands_executed']}")
            
            print(f"\n🖥️  System Availability:")
            for system, available in stats['system_availability'].items():
                status = "✅ Available" if available else "❌ Unavailable"
                print(f"  {system}: {status}")
            
            if stats['performance_summary']:
                perf = stats['performance_summary']
                print(f"\n📊 Performance (24h):")
                print(f"  Sessions: {perf['total_sessions']}")
                print(f"  Tracks: {perf['total_tracks_generated']}")
                print(f"  Avg Time: {perf['average_generation_time']:.1f}s")
                print(f"  Avg Score: {perf['average_performance_score']:.2f}")
        
        elif args.command == "test":
            print("🧪 Running integration bridge tests...")
            
            # Test system availability
            stats = bridge.get_integration_statistics()
            print(f"✅ Bridge initialized: {stats['integration_info']['mode']}")
            
            # Test BMAD generation
            test_result = await bridge.generate_hardcore_music("test gabber track at 180 bpm")
            if test_result["success"]:
                print("✅ BMAD generation test passed")
            else:
                print(f"❌ BMAD generation test failed: {test_result.get('error')}")
            
            # Test main system if available
            if bridge.main_system_available:
                try:
                    main_result = await bridge.generate_music_adaptive("test music generation")
                    if main_result["success"]:
                        print("✅ Main system integration test passed")
                    else:
                        print(f"❌ Main system test failed: {main_result.get('error')}")
                except Exception as e:
                    print(f"❌ Main system test error: {e}")
            
            print("\n🎉 Integration bridge tests complete!")
    
    try:
        asyncio.run(run_command())
    except Exception as e:
        print(f"❌ Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()