#!/usr/bin/env python3
"""
BMAD (Better Method for AI Development) Initialization System
Music Production Expansion Pack Initialization for Gabberbot

This script initializes the BMAD music production system and provides
the /init command functionality as referenced in CLAUDE.md.

Usage:
    python bmad_init.py
    python bmad_init.py --verify
    python bmad_init.py --status
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BMadInitializationStatus:
    """Track BMAD initialization status"""
    expansion_pack_installed: bool = False
    agents_configured: bool = False
    teams_configured: bool = False
    workflows_configured: bool = False
    integration_verified: bool = False
    initialization_time: Optional[datetime] = None
    version: str = "1.0.0"


class BMadInitializer:
    """BMAD Music Production Expansion Pack Initializer"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.bmad_root = self.project_root / "BMAD-AT-CLAUDE"
        self.expansion_pack_root = self.bmad_root / "expansion-packs" / "bmad-music-production"
        self.status = BMadInitializationStatus()
        
    def init_command(self) -> bool:
        """
        Execute the /init command as referenced in CLAUDE.md
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        print("BMAD Music Production Expansion Pack Initialization")
        print("=" * 60)
        print()
        
        try:
            # Check if BMAD structure exists
            if not self._verify_bmad_structure():
                print("[ERROR] BMAD structure not found or incomplete")
                return False
                
            # Initialize agents
            if not self._initialize_agents():
                print("❌ Agent initialization failed")
                return False
                
            # Initialize teams
            if not self._initialize_teams():
                print("❌ Team initialization failed")
                return False
                
            # Initialize workflows
            if not self._initialize_workflows():
                print("❌ Workflow initialization failed")
                return False
                
            # Verify integration
            if not self._verify_integration():
                print("❌ Integration verification failed")
                return False
                
            # Mark as initialized
            self._mark_initialized()
            
            print()
            print("✅ BMAD Music Production System Successfully Initialized!")
            print()
            self._display_initialization_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed with error: {e}")
            return False
    
    def _verify_bmad_structure(self) -> bool:
        """Verify BMAD directory structure exists"""
        print("🔍 Verifying BMAD structure...")
        
        required_dirs = [
            self.expansion_pack_root,
            self.expansion_pack_root / "agents",
            self.expansion_pack_root / "agent-teams", 
            self.expansion_pack_root / "workflows",
            self.expansion_pack_root / "tasks",
            self.expansion_pack_root / "templates",
            self.expansion_pack_root / "data"
        ]
        
        required_files = [
            self.expansion_pack_root / "config.yaml"
        ]
        
        # Check directories
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"   ❌ Missing directory: {dir_path}")
                return False
            print(f"   ✅ Found: {dir_path.name}/")
        
        # Check files
        for file_path in required_files:
            if not file_path.exists():
                print(f"   ❌ Missing file: {file_path}")
                return False
            print(f"   ✅ Found: {file_path.name}")
        
        self.status.expansion_pack_installed = True
        return True
    
    def _initialize_agents(self) -> bool:
        """Initialize and validate all 9 music production agents"""
        print("🤖 Initializing BMAD music production agents...")
        
        required_agents = [
            "music_orchestrator.yaml",
            "music_producer.yaml", 
            "sound_designer.yaml",
            "mix_engineer.yaml",
            "music_analyst.yaml",
            "theory_engine.yaml",
            "innovation_lab.yaml",
            "music_analyst_specialist.yaml",
            "music_archivist.yaml"
        ]
        
        agents_dir = self.expansion_pack_root / "agents"
        initialized_agents = []
        
        for agent_file in required_agents:
            agent_path = agents_dir / agent_file
            if not agent_path.exists():
                print(f"   ❌ Missing agent: {agent_file}")
                return False
                
            # Validate agent configuration
            try:
                with open(agent_path, 'r', encoding='utf-8') as f:
                    agent_config = yaml.safe_load(f)
                    
                if not self._validate_agent_config(agent_config, agent_file):
                    return False
                    
                initialized_agents.append(agent_config.get('name', agent_file))
                print(f"   ✅ Initialized: {agent_config.get('name', agent_file)} ({agent_config.get('agent_id', 'unknown')})")
                
            except Exception as e:
                print(f"   ❌ Failed to load {agent_file}: {e}")
                return False
        
        print(f"   🎯 Successfully initialized {len(initialized_agents)} agents")
        self.status.agents_configured = True
        return True
    
    def _validate_agent_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate agent configuration structure"""
        required_keys = ["agent_id", "name", "version", "personality", "specialties", "commands"]
        
        for key in required_keys:
            if key not in config:
                print(f"   ❌ {filename}: Missing required key '{key}'")
                return False
        
        return True
    
    def _initialize_teams(self) -> bool:
        """Initialize team coordination bundles"""
        print("👥 Initializing team coordination bundles...")
        
        required_teams = [
            "hardcore-music-team.yaml",
            "analysis-intelligence-team.yaml", 
            "innovation-research-team.yaml"
        ]
        
        teams_dir = self.expansion_pack_root / "agent-teams"
        initialized_teams = []
        
        for team_file in required_teams:
            team_path = teams_dir / team_file
            if not team_path.exists():
                print(f"   ❌ Missing team: {team_file}")
                return False
                
            try:
                with open(team_path, 'r', encoding='utf-8') as f:
                    team_config = yaml.safe_load(f)
                    
                initialized_teams.append(team_config.get('team_name', team_file))
                print(f"   ✅ Configured: {team_config.get('team_name', team_file)}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {team_file}: {e}")
                return False
        
        print(f"   🎯 Successfully configured {len(initialized_teams)} team bundles")
        self.status.teams_configured = True
        return True
    
    def _initialize_workflows(self) -> bool:
        """Initialize workflow definitions"""
        print("⚡ Initializing workflow definitions...")
        
        workflows_dir = self.expansion_pack_root / "workflows"
        workflow_files = list(workflows_dir.glob("*.yaml"))
        
        if not workflow_files:
            print("   ❌ No workflow files found")
            return False
        
        initialized_workflows = []
        
        for workflow_path in workflow_files:
            try:
                with open(workflow_path, 'r', encoding='utf-8') as f:
                    workflow_config = yaml.safe_load(f)
                    
                initialized_workflows.append(workflow_config.get('workflow_name', workflow_path.name))
                print(f"   ✅ Loaded: {workflow_config.get('workflow_name', workflow_path.name)}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {workflow_path.name}: {e}")
                return False
        
        print(f"   🎯 Successfully loaded {len(initialized_workflows)} workflows")
        self.status.workflows_configured = True
        return True
    
    def _verify_integration(self) -> bool:
        """Verify integration with existing infrastructure"""
        print("🔗 Verifying integration with existing infrastructure...")
        
        # Check for key infrastructure files
        integration_checks = [
            (self.project_root / "cli_shared" / "interfaces" / "synthesizer.py", "AbstractSynthesizer interface"),
            (self.project_root / "cli_shared" / "models" / "midi_clips.py", "MIDI clip models"),
            (self.project_root / "audio" / "synthesis", "Synthesis engines"),
            (self.project_root / "main.py", "Main application entry point")
        ]
        
        for path, description in integration_checks:
            if path.exists():
                print(f"   ✅ {description}: Found")
            else:
                print(f"   ⚠️  {description}: Not found (may affect integration)")
        
        # Verify BMAD config can be loaded
        try:
            config_path = self.expansion_pack_root / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                bmad_config = yaml.safe_load(f)
                
            print(f"   ✅ BMAD config loaded: v{bmad_config.get('version', 'unknown')}")
            print(f"   ✅ Agent count: {bmad_config.get('agents', {}).get('count', 'unknown')}")
            
        except Exception as e:
            print(f"   ❌ Failed to load BMAD config: {e}")
            return False
        
        self.status.integration_verified = True
        return True
    
    def _mark_initialized(self) -> None:
        """Mark system as initialized"""
        self.status.initialization_time = datetime.now()
        
        # Save initialization status
        status_file = self.bmad_root / ".bmad_status.json"
        status_data = {
            "expansion_pack_installed": self.status.expansion_pack_installed,
            "agents_configured": self.status.agents_configured,
            "teams_configured": self.status.teams_configured,
            "workflows_configured": self.status.workflows_configured,
            "integration_verified": self.status.integration_verified,
            "initialization_time": self.status.initialization_time.isoformat(),
            "version": self.status.version
        }
        
        try:
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"   ⚠️  Could not save status file: {e}")
    
    def _display_initialization_summary(self) -> None:
        """Display initialization summary"""
        print("📊 BMAD Initialization Summary")
        print("-" * 40)
        print(f"   🎵 Expansion Pack: ✅ Installed")
        print(f"   🤖 Agents: ✅ 9 agents configured")
        print(f"   👥 Teams: ✅ 3 team bundles ready")
        print(f"   ⚡ Workflows: ✅ Production workflows loaded")
        print(f"   🔗 Integration: ✅ Infrastructure verified")
        print()
        print("🚀 BMAD Music Production System Ready!")
        print()
        print("Available agent teams:")
        print("   • @music-orchestrator (Conductor) - Team coordination")
        print("   • @music-producer (Raven) - Creative composition")
        print("   • @sound-designer (Void) - Synthesis and processing")
        print("   • @mix-engineer (Phoenix) - Professional mixing")
        print("   • @music-analyst (Nexus) - Analysis and intelligence")
        print("   • @theory-engine (Cipher) - Music theory and math")
        print("   • @innovation-lab (Flux) - Experimental techniques")
        print("   • @music-analyst-specialist (Archive) - Research extraction")
        print("   • @music-archivist (Keeper) - Knowledge management")
        print()
        print("🎯 Use @music-orchestrator to coordinate any music production task!")
        print("🔥 Remember: Agents do the work, humans orchestrate!")
    
    def get_status(self) -> BMadInitializationStatus:
        """Get current initialization status"""
        status_file = self.bmad_root / ".bmad_status.json"
        
        if status_file.exists():
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                    
                self.status.expansion_pack_installed = status_data.get('expansion_pack_installed', False)
                self.status.agents_configured = status_data.get('agents_configured', False)
                self.status.teams_configured = status_data.get('teams_configured', False)
                self.status.workflows_configured = status_data.get('workflows_configured', False)
                self.status.integration_verified = status_data.get('integration_verified', False)
                self.status.version = status_data.get('version', '1.0.0')
                
                if status_data.get('initialization_time'):
                    self.status.initialization_time = datetime.fromisoformat(status_data['initialization_time'])
                    
            except Exception as e:
                print(f"Warning: Could not load status file: {e}")
        
        return self.status
    
    def verify_installation(self) -> bool:
        """Verify BMAD installation is complete and functional"""
        print("🔍 Verifying BMAD installation...")
        
        status = self.get_status()
        
        checks = [
            (status.expansion_pack_installed, "Expansion pack installed"),
            (status.agents_configured, "Agents configured"),
            (status.teams_configured, "Teams configured"),
            (status.workflows_configured, "Workflows configured"),
            (status.integration_verified, "Integration verified")
        ]
        
        all_passed = True
        for passed, description in checks:
            if passed:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}")
                all_passed = False
        
        if all_passed and status.initialization_time:
            print(f"   ✅ Initialized: {status.initialization_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("   🎵 BMAD system fully operational!")
        else:
            print("   ⚠️  BMAD system requires initialization")
            print("   💡 Run: python bmad_init.py")
        
        return all_passed


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BMAD Music Production Expansion Pack Initializer",
        epilog="Use this script to initialize the BMAD music production system as referenced in CLAUDE.md"
    )
    
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify BMAD installation without reinitializing"
    )
    
    parser.add_argument(
        "--status",
        action="store_true", 
        help="Show current BMAD status"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Specify project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize BMAD system
    initializer = BMadInitializer(args.project_root)
    
    if args.verify:
        success = initializer.verify_installation()
        sys.exit(0 if success else 1)
        
    elif args.status:
        status = initializer.get_status()
        print("📊 BMAD Status Report")
        print("=" * 30)
        print(f"Version: {status.version}")
        print(f"Expansion Pack: {'✅' if status.expansion_pack_installed else '❌'}")
        print(f"Agents: {'✅' if status.agents_configured else '❌'}")
        print(f"Teams: {'✅' if status.teams_configured else '❌'}")
        print(f"Workflows: {'✅' if status.workflows_configured else '❌'}")
        print(f"Integration: {'✅' if status.integration_verified else '❌'}")
        if status.initialization_time:
            print(f"Initialized: {status.initialization_time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0)
        
    else:
        # Run full initialization
        success = initializer.init_command()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()