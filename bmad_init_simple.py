#!/usr/bin/env python3
"""
BMAD (Better Method for AI Development) Initialization System
Music Production Expansion Pack Initialization for Gabberbot

Simple version without emoji characters to avoid encoding issues.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime


class BMadInitializer:
    """BMAD Music Production Expansion Pack Initializer"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.bmad_root = self.project_root / "BMAD-AT-CLAUDE"
        self.expansion_pack_root = self.bmad_root / "expansion-packs" / "bmad-music-production"
        
    def init_command(self):
        """Execute the /init command as referenced in CLAUDE.md"""
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
                print("[ERROR] Agent initialization failed")
                return False
                
            # Initialize teams
            if not self._initialize_teams():
                print("[ERROR] Team initialization failed")
                return False
                
            # Initialize workflows
            if not self._initialize_workflows():
                print("[ERROR] Workflow initialization failed")
                return False
                
            print()
            print("[SUCCESS] BMAD Music Production System Successfully Initialized!")
            print()
            self._display_initialization_summary()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Initialization failed with error: {e}")
            return False
    
    def _verify_bmad_structure(self):
        """Verify BMAD directory structure exists"""
        print("[STEP 1] Verifying BMAD structure...")
        
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
                print(f"   [MISSING] Directory: {dir_path}")
                return False
            print(f"   [FOUND] {dir_path.name}/")
        
        # Check files
        for file_path in required_files:
            if not file_path.exists():
                print(f"   [MISSING] File: {file_path}")
                return False
            print(f"   [FOUND] {file_path.name}")
        
        return True
    
    def _initialize_agents(self):
        """Initialize and validate all 9 music production agents"""
        print("[STEP 2] Initializing BMAD music production agents...")
        
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
                print(f"   [MISSING] Agent: {agent_file}")
                return False
                
            # Validate agent configuration
            try:
                with open(agent_path, 'r', encoding='utf-8') as f:
                    agent_config = yaml.safe_load(f)
                    
                if not self._validate_agent_config(agent_config, agent_file):
                    return False
                    
                initialized_agents.append(agent_config.get('name', agent_file))
                print(f"   [INIT] {agent_config.get('name', agent_file)} ({agent_config.get('agent_id', 'unknown')})")
                
            except Exception as e:
                print(f"   [ERROR] Failed to load {agent_file}: {e}")
                return False
        
        print(f"   [SUCCESS] Successfully initialized {len(initialized_agents)} agents")
        return True
    
    def _validate_agent_config(self, config, filename):
        """Validate agent configuration structure"""
        required_keys = ["agent_id", "name", "version", "personality", "specialties", "commands"]
        
        for key in required_keys:
            if key not in config:
                print(f"   [ERROR] {filename}: Missing required key '{key}'")
                return False
        
        return True
    
    def _initialize_teams(self):
        """Initialize team coordination bundles"""
        print("[STEP 3] Initializing team coordination bundles...")
        
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
                print(f"   [MISSING] Team: {team_file}")
                return False
                
            try:
                with open(team_path, 'r', encoding='utf-8') as f:
                    team_config = yaml.safe_load(f)
                    
                initialized_teams.append(team_config.get('team_name', team_file))
                print(f"   [CONFIGURED] {team_config.get('team_name', team_file)}")
                
            except Exception as e:
                print(f"   [ERROR] Failed to load {team_file}: {e}")
                return False
        
        print(f"   [SUCCESS] Successfully configured {len(initialized_teams)} team bundles")
        return True
    
    def _initialize_workflows(self):
        """Initialize workflow definitions"""
        print("[STEP 4] Initializing workflow definitions...")
        
        workflows_dir = self.expansion_pack_root / "workflows"
        workflow_files = list(workflows_dir.glob("*.yaml"))
        
        if not workflow_files:
            print("   [ERROR] No workflow files found")
            return False
        
        initialized_workflows = []
        
        for workflow_path in workflow_files:
            try:
                with open(workflow_path, 'r', encoding='utf-8') as f:
                    workflow_config = yaml.safe_load(f)
                    
                initialized_workflows.append(workflow_config.get('workflow_name', workflow_path.name))
                print(f"   [LOADED] {workflow_config.get('workflow_name', workflow_path.name)}")
                
            except Exception as e:
                print(f"   [ERROR] Failed to load {workflow_path.name}: {e}")
                return False
        
        print(f"   [SUCCESS] Successfully loaded {len(initialized_workflows)} workflows")
        return True
    
    def _display_initialization_summary(self):
        """Display initialization summary"""
        print("BMAD Initialization Summary")
        print("-" * 40)
        print("   Expansion Pack: INSTALLED")
        print("   Agents: 9 agents configured")
        print("   Teams: 3 team bundles ready")
        print("   Workflows: Production workflows loaded")
        print()
        print("BMAD Music Production System Ready!")
        print()
        print("Available agent teams:")
        print("   * @music-orchestrator (Conductor) - Team coordination")
        print("   * @music-producer (Raven) - Creative composition")
        print("   * @sound-designer (Void) - Synthesis and processing")
        print("   * @mix-engineer (Phoenix) - Professional mixing")
        print("   * @music-analyst (Nexus) - Analysis and intelligence")
        print("   * @theory-engine (Cipher) - Music theory and math")
        print("   * @innovation-lab (Flux) - Experimental techniques")
        print("   * @music-analyst-specialist (Archive) - Research extraction")
        print("   * @music-archivist (Keeper) - Knowledge management")
        print()
        print("Use @music-orchestrator to coordinate any music production task!")
        print("Remember: Agents do the work, humans orchestrate!")


def main():
    """Main CLI entry point"""
    initializer = BMadInitializer()
    success = initializer.init_command()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()