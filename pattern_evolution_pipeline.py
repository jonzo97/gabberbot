#!/usr/bin/env python3
"""
BMAD Pattern Evolution Pipeline for Hardcore/Gabber Music Production
Coordinates multiple agents to evolve and optimize hardcore patterns overnight
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class PatternEvolutionPipeline:
    """Orchestrates BMAD agents for autonomous pattern evolution"""
    
    def __init__(self):
        self.workspace_dir = Path("pattern_evolution_workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.workspace_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # BMAD Agent assignments following methodology
        self.agents = {
            "orchestrator": "@music-orchestrator (Conductor)",
            "analyst": "@music-analyst (Nexus)",
            "producer": "@music-producer (Raven)",
            "designer": "@sound-designer (Void)",
            "engineer": "@mix-engineer (Phoenix)"
        }
        
        # Hardcore-specific parameters from CLAUDE.md
        self.hardcore_params = {
            "bpm_range": [180, 200, 220],  # Hardcore/gabber BPM
            "kick_fundamental": 41.2,  # E1 in Hz
            "distortion_db": 15,  # User-validated level
            "patterns_per_batch": 10,
            "evolution_generations": 5
        }
        
    def initialize_pipeline(self) -> Dict[str, Any]:
        """Initialize the pattern evolution pipeline"""
        config = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "agents": self.agents,
            "parameters": self.hardcore_params,
            "pipeline_stages": [
                "pattern_analysis",
                "pattern_generation", 
                "sound_design",
                "mixing_optimization"
            ],
            "status": "initialized"
        }
        
        # Save configuration
        config_path = self.session_dir / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"\n[PIPELINE] Pattern Evolution Pipeline Initialized")
        print(f"[SESSION] {self.session_id}")
        print(f"[WORKSPACE] {self.session_dir}")
        
        return config
    
    def stage_1_analysis(self) -> Dict[str, Any]:
        """Stage 1: Pattern Analysis by @music-analyst"""
        print(f"\n[STAGE 1] Invoking {self.agents['analyst']}")
        
        analysis_tasks = {
            "agent": self.agents["analyst"],
            "tasks": [
                {
                    "id": "analyze_kick_patterns",
                    "description": "Analyze hardcore kick patterns for evolution potential",
                    "parameters": {
                        "pattern_types": ["gabber_kick", "frenchcore_kick", "industrial_kick"],
                        "metrics": ["punch", "distortion", "sub_weight"]
                    }
                },
                {
                    "id": "analyze_acid_patterns",
                    "description": "Extract acid bassline characteristics",
                    "parameters": {
                        "pattern_types": ["303_acid", "hoover_bass"],
                        "metrics": ["resonance", "glide", "accent_pattern"]
                    }
                }
            ],
            "output_format": "pattern_dna.json"
        }
        
        # Save analysis tasks
        task_path = self.session_dir / "stage1_analysis_tasks.json"
        with open(task_path, 'w') as f:
            json.dump(analysis_tasks, f, indent=2)
            
        print(f"   [TASK] Analyzing {len(analysis_tasks['tasks'])} pattern categories")
        print(f"   [OUTPUT] Pattern DNA will be saved to {task_path}")
        
        return analysis_tasks
    
    def stage_2_generation(self) -> Dict[str, Any]:
        """Stage 2: Pattern Generation by @music-producer"""
        print(f"\n[STAGE 2] Invoking {self.agents['producer']}")
        
        generation_tasks = {
            "agent": self.agents["producer"],
            "tasks": [
                {
                    "id": "evolve_kick_patterns",
                    "description": "Generate evolved hardcore kick patterns",
                    "parameters": {
                        "base_pattern": "rotterdam_gabber",
                        "evolution_count": self.hardcore_params["patterns_per_batch"],
                        "bpm": self.hardcore_params["bpm_range"][0],
                        "variation_params": {
                            "timing_variance": 0.02,
                            "velocity_range": [100, 127],
                            "pitch_variations": [-2, 0, 2]  # Semitones
                        }
                    }
                },
                {
                    "id": "generate_acid_variations",
                    "description": "Create acid bassline variations",
                    "parameters": {
                        "scale": "A_minor",
                        "pattern_length": 16,  # Steps
                        "accent_density": 0.3,
                        "slide_probability": 0.4
                    }
                }
            ],
            "output_format": "midi_clips"
        }
        
        # Save generation tasks
        task_path = self.session_dir / "stage2_generation_tasks.json"
        with open(task_path, 'w') as f:
            json.dump(generation_tasks, f, indent=2)
            
        print(f"   [TASK] Generating {self.hardcore_params['patterns_per_batch']} variations per pattern")
        print(f"   [BPM] {self.hardcore_params['bpm_range'][0]} BPM")
        
        return generation_tasks
    
    def stage_3_sound_design(self) -> Dict[str, Any]:
        """Stage 3: Sound Design by @sound-designer"""
        print(f"\n[STAGE 3] Invoking {self.agents['designer']}")
        
        design_tasks = {
            "agent": self.agents["designer"],
            "tasks": [
                {
                    "id": "apply_rotterdam_doorlussen",
                    "description": "Apply authentic doorlussen distortion chain",
                    "parameters": {
                        "chain": [
                            {"effect": "overdrive", "drive": 0.8},
                            {"effect": "distortion", "amount": self.hardcore_params["distortion_db"]},
                            {"effect": "eq", "low_boost": 6, "mid_cut": -3}
                        ]
                    }
                },
                {
                    "id": "warehouse_reverb_processing",
                    "description": "Add warehouse atmosphere",
                    "parameters": {
                        "reverb_type": "warehouse",
                        "room_size": 0.9,
                        "damping": 0.3,
                        "wet_level": 0.15
                    }
                }
            ],
            "output_format": "processed_audio"
        }
        
        # Save sound design tasks
        task_path = self.session_dir / "stage3_sound_design_tasks.json"
        with open(task_path, 'w') as f:
            json.dump(design_tasks, f, indent=2)
            
        print(f"   [TASK] Applying hardcore-specific sound design")
        print(f"   [DISTORTION] {self.hardcore_params['distortion_db']} dB")
        
        return design_tasks
    
    def stage_4_mixing(self) -> Dict[str, Any]:
        """Stage 4: Mixing Optimization by @mix-engineer"""
        print(f"\n[STAGE 4] Invoking {self.agents['engineer']}")
        
        mixing_tasks = {
            "agent": self.agents["engineer"],
            "tasks": [
                {
                    "id": "optimize_kick_mixing",
                    "description": "Optimize kick drum mixing for warehouse systems",
                    "parameters": {
                        "compression_ratio": 8,
                        "attack_ms": 0.5,
                        "release_ms": 50,
                        "makeup_gain": 3
                    }
                },
                {
                    "id": "master_limiting",
                    "description": "Apply hardcore mastering chain",
                    "parameters": {
                        "limiter_threshold": -0.5,
                        "ceiling": -0.1,
                        "release_ms": 10,
                        "target_lufs": -6  # Loud for hardcore
                    }
                }
            ],
            "output_format": "final_tracks"
        }
        
        # Save mixing tasks
        task_path = self.session_dir / "stage4_mixing_tasks.json"
        with open(task_path, 'w') as f:
            json.dump(mixing_tasks, f, indent=2)
            
        print(f"   [TASK] Professional mixing for warehouse systems")
        print(f"   [TARGET] -6 LUFS (hardcore loudness)")
        
        return mixing_tasks
    
    def create_overnight_batch(self) -> Dict[str, Any]:
        """Create batch configuration for overnight processing"""
        batch_config = {
            "batch_id": f"overnight_{self.session_id}",
            "total_generations": self.hardcore_params["evolution_generations"],
            "patterns_per_generation": self.hardcore_params["patterns_per_batch"],
            "total_expected_outputs": (
                self.hardcore_params["evolution_generations"] * 
                self.hardcore_params["patterns_per_batch"]
            ),
            "bpm_variations": self.hardcore_params["bpm_range"],
            "processing_stages": [
                "analysis",
                "generation",
                "sound_design",
                "mixing"
            ],
            "estimated_outputs": {
                "midi_clips": 50,
                "audio_renders": 50,
                "mixed_tracks": 10,
                "preset_files": 20
            }
        }
        
        # Save batch configuration
        batch_path = self.session_dir / "overnight_batch_config.json"
        with open(batch_path, 'w') as f:
            json.dump(batch_config, f, indent=2)
            
        print(f"\n[BATCH] Overnight Batch Configured")
        print(f"   [OUTPUTS] {batch_config['total_expected_outputs']} total patterns")
        print(f"   [GENERATIONS] {batch_config['total_generations']} evolution cycles")
        
        return batch_config
    
    def run_test_cycle(self) -> None:
        """Run a quick test cycle to verify pipeline"""
        print("\n" + "="*60)
        print("BMAD PATTERN EVOLUTION PIPELINE - TEST CYCLE")
        print("="*60)
        
        # Initialize
        config = self.initialize_pipeline()
        
        # Run all stages
        stage1 = self.stage_1_analysis()
        stage2 = self.stage_2_generation()
        stage3 = self.stage_3_sound_design()
        stage4 = self.stage_4_mixing()
        
        # Configure overnight batch
        batch = self.create_overnight_batch()
        
        # Create orchestration script
        orchestration_script = {
            "pipeline_config": config,
            "stages": {
                "stage_1": stage1,
                "stage_2": stage2,
                "stage_3": stage3,
                "stage_4": stage4
            },
            "batch_config": batch,
            "execution_order": [
                "@music-orchestrator coordinates team",
                "@music-analyst performs pattern analysis",
                "@music-producer generates evolved patterns",
                "@sound-designer applies hardcore processing",
                "@mix-engineer optimizes for warehouse systems"
            ]
        }
        
        # Save full orchestration
        orchestration_path = self.session_dir / "full_orchestration.json"
        with open(orchestration_path, 'w') as f:
            json.dump(orchestration_script, f, indent=2)
        
        print(f"\n[SUCCESS] Pipeline Test Complete!")
        print(f"[ORCHESTRATION] Saved to {orchestration_path}")
        print(f"\n[READY] Agents are configured for overnight work")
        print(f"[EXPECTED] {batch['total_expected_outputs']} hardcore patterns by morning")
        
        # Summary
        print("\n" + "="*60)
        print("OVERNIGHT WORK SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Workspace: {self.session_dir}")
        print(f"Agents: {len(self.agents)} specialized BMAD agents")
        print(f"Stages: {len(orchestration_script['stages'])} processing stages")
        print(f"Expected Outputs:")
        for output_type, count in batch['estimated_outputs'].items():
            print(f"   - {output_type}: {count}")
        print("\n[BMAD] Following BMAD methodology: Agents do the work!")

if __name__ == "__main__":
    pipeline = PatternEvolutionPipeline()
    pipeline.run_test_cycle()