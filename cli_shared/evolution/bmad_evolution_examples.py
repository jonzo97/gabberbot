#!/usr/bin/env python3
"""
BMAD Evolution Examples and Demonstrations
@theory-engine (Cipher) - Working demonstrations of advanced pattern evolution

Comprehensive examples showing the power of the BMAD evolution system:
- BPM progression ladders (180→200→220 BPM) 
- Genre fusion evolution (gabber + frenchcore + industrial)
- Mathematical progression tracking and family trees
- Energy curve optimization for warehouse sound systems
- Pattern uniqueness analysis and diversity maintenance
"""

import asyncio
import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..models.hardcore_models import (
    HardcorePattern, PatternStep, SynthType, SynthParams,
    create_gabber_kick_pattern, create_acid_pattern, create_industrial_pattern
)
from .bmad_pattern_evolution import (
    BMADPatternEvolution, BMADEvolutionConfig, EvolutionStrategy, MutationConstraint
)
from .bmad_theory_engine import BMADTheoryEngine


class BMADEvolutionDemonstrator:
    """
    Comprehensive demonstration of BMAD evolution capabilities
    
    Shows practical applications of the evolution system for @innovation-lab
    """
    
    def __init__(self):
        self.theory_engine = BMADTheoryEngine()
        self.results_dir = Path("evolution_results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def demo_bpm_progression_ladder(self) -> Dict[str, any]:
        """Demonstrate BPM progression from 180→220 BPM"""
        print("🎵 BPM Progression Ladder Demo (180→220 BPM)")
        print("=" * 50)
        
        config = BMADEvolutionConfig(
            population_size=30,
            generations=15,
            strategy=EvolutionStrategy.BPM_LADDER,
            bpm_start=180.0,
            bpm_end=220.0,
            bpm_progression_rate=0.1,
            constraints=[MutationConstraint.PRESERVE_GROOVE, MutationConstraint.MAINTAIN_ENERGY]
        )
        
        engine = BMADPatternEvolution(config)
        
        # Start with gabber patterns at 180 BPM
        seed_patterns = [
            create_gabber_kick_pattern("Gabber_Base_180", 180),
            create_acid_pattern("Acid_Base_180", 180),
        ]
        
        print(f"🌱 Starting with {len(seed_patterns)} seed patterns at 180 BPM")
        await engine.initialize_population(seed_patterns)
        
        bpm_progression = []
        fitness_progression = []
        
        for gen in range(config.generations):
            population = await engine.evolve_generation()
            
            # Track BPM progression
            current_bpms = [p.pattern.bpm for p in population]
            avg_bpm = np.mean(current_bpms)
            max_bpm = max(current_bpms)
            min_bpm = min(current_bpms)
            
            # Track fitness
            fitnesses = [p.total_fitness for p in population]
            avg_fitness = np.mean(fitnesses)
            best_fitness = max(fitnesses)
            
            bpm_progression.append({
                "generation": gen + 1,
                "avg_bpm": avg_bpm,
                "bpm_range": (min_bpm, max_bpm),
                "target_bpm": engine._calculate_current_bpm_target()
            })
            
            fitness_progression.append({
                "generation": gen + 1,
                "avg_fitness": avg_fitness,
                "best_fitness": best_fitness
            })
            
            print(f"   Gen {gen+1:2d}: BPM {avg_bpm:6.1f} (target {engine._calculate_current_bpm_target():6.1f}), "
                  f"Fitness {avg_fitness:.3f}, Best {best_fitness:.3f}")
        
        # Analyze final results
        final_patterns = engine.get_best_patterns(5)
        
        print(f"\n🎯 Final Results:")
        print(f"   Target BPM achieved: {final_patterns[0].pattern.bpm:.1f} / {config.bpm_end}")
        print(f"   BPM progression successful: {abs(final_patterns[0].pattern.bpm - config.bpm_end) < 10}")
        print(f"   Best final fitness: {final_patterns[0].total_fitness:.3f}")
        
        # Show pattern evolution
        print(f"\n🧬 Top Evolved Patterns:")
        for i, pattern in enumerate(final_patterns):
            print(f"   {i+1}. {pattern.pattern.name}")
            print(f"      BPM: {pattern.pattern.bpm:.1f}, Generation: {pattern.generation}")
            print(f"      Mutations: {', '.join(pattern.mutation_history[-3:])}")
        
        return {
            "strategy": "bpm_ladder",
            "bpm_progression": bmp_progression,
            "fitness_progression": fitness_progression,
            "final_patterns": [p.pattern.to_dict() for p in final_patterns],
            "success": abs(final_patterns[0].pattern.bpm - config.bmp_end) < 15
        }
    
    async def demo_genre_fusion_evolution(self) -> Dict[str, any]:
        """Demonstrate genre fusion: gabber + frenchcore + industrial"""
        print("\n🎭 Genre Fusion Evolution Demo")
        print("=" * 50)
        
        config = BMADEvolutionConfig(
            population_size=25,
            generations=12,
            strategy=EvolutionStrategy.GENRE_FUSION,
            target_genres=["gabber", "frenchcore", "industrial"],
            fusion_rate=0.4,
            constraints=[MutationConstraint.GENRE_AUTHENTIC]
        )
        
        engine = BMADPatternEvolution(config)
        
        # Create pure genre seeds
        seed_patterns = [
            create_gabber_kick_pattern("Pure_Gabber", 185),
            create_industrial_pattern("Pure_Industrial", 145),
            # Create frenchcore-style pattern
            self._create_frenchcore_pattern("Pure_Frenchcore", 210)
        ]
        
        print(f"🌱 Starting with pure genre patterns:")
        for pattern in seed_patterns:
            genre_analysis = self.theory_engine.analyze_genre_authenticity(pattern)
            print(f"   {pattern.name}: Gabber={genre_analysis.gabber_authenticity:.2f}, "
                  f"Industrial={genre_analysis.industrial_characteristics:.2f}, "
                  f"Frenchcore={genre_analysis.frenchcore_elements:.2f}")
        
        await engine.initialize_population(seed_patterns)
        
        fusion_progression = []
        
        for gen in range(config.generations):
            population = await engine.evolve_generation()
            
            # Analyze genre characteristics across population
            genre_scores = {"gabber": [], "industrial": [], "frenchcore": []}
            
            for pattern in population:
                genre_analysis = self.theory_engine.analyze_genre_authenticity(pattern.pattern)
                genre_scores["gabber"].append(genre_analysis.gabber_authenticity)
                genre_scores["industrial"].append(genre_analysis.industrial_characteristics)
                genre_scores["frenchcore"].append(genre_analysis.frenchcore_elements)
            
            # Calculate fusion metrics
            avg_scores = {genre: np.mean(scores) for genre, scores in genre_scores.items()}
            fusion_score = min(avg_scores.values())  # How well all genres are represented
            
            fusion_progression.append({
                "generation": gen + 1,
                "genre_averages": avg_scores,
                "fusion_score": fusion_score
            })
            
            print(f"   Gen {gen+1:2d}: Fusion score {fusion_score:.3f}, "
                  f"Gabber {avg_scores['gabber']:.2f}, "
                  f"Industrial {avg_scores['industrial']:.2f}, "
                  f"Frenchcore {avg_scores['frenchcore']:.2f}")
        
        # Analyze best fusion patterns
        final_patterns = engine.get_best_patterns(3)
        
        print(f"\n🎯 Best Fusion Patterns:")
        for i, pattern in enumerate(final_patterns):
            genre_analysis = self.theory_engine.analyze_genre_authenticity(pattern.pattern)
            print(f"   {i+1}. {pattern.pattern.name}")
            print(f"      BPM: {pattern.pattern.bpm:.1f}, Fitness: {pattern.total_fitness:.3f}")
            print(f"      Gabber: {genre_analysis.gabber_authenticity:.3f}")
            print(f"      Industrial: {genre_analysis.industrial_characteristics:.3f}")
            print(f"      Frenchcore: {genre_analysis.frenchcore_elements:.3f}")
            print(f"      Genre tags: {', '.join(pattern.genre_tags)}")
        
        return {
            "strategy": "genre_fusion",
            "fusion_progression": fusion_progression,
            "final_patterns": [p.pattern.to_dict() for p in final_patterns],
            "fusion_success": fusion_progression[-1]["fusion_score"] > 0.4
        }
    
    async def demo_energy_optimization(self) -> Dict[str, any]:
        """Demonstrate warehouse energy optimization"""
        print("\n⚡ Warehouse Energy Optimization Demo")
        print("=" * 50)
        
        config = BMADEvolutionConfig(
            population_size=20,
            generations=10,
            strategy=EvolutionStrategy.ENERGY_OPTIMIZATION,
            optimize_for_warehouse=True,
            warehouse_energy_target=0.85,
            constraints=[MutationConstraint.MAINTAIN_ENERGY]
        )
        
        engine = BMADPatternEvolution(config)
        
        # Start with mixed energy patterns
        seed_patterns = [
            create_gabber_kick_pattern("Low_Energy", 160),    # Lower energy
            create_acid_pattern("Medium_Energy", 180),        # Medium energy
            self._create_high_energy_pattern("High_Energy", 200)  # High energy
        ]
        
        print(f"🌱 Starting energy analysis:")
        for pattern in seed_patterns:
            energy_analysis = self.theory_engine.analyze_energy_curve(pattern)
            print(f"   {pattern.name}: Energy={energy_analysis.average_energy:.2f}, "
                  f"Warehouse={energy_analysis.warehouse_compatibility:.2f}")
        
        await engine.initialize_population(seed_patterns)
        
        energy_progression = []
        
        for gen in range(config.generations):
            population = await engine.evolve_generation()
            
            # Analyze energy characteristics
            energy_levels = []
            warehouse_scores = []
            
            for pattern in population:
                energy_analysis = self.theory_engine.analyze_energy_curve(pattern.pattern)
                energy_levels.append(energy_analysis.average_energy)
                warehouse_scores.append(energy_analysis.warehouse_compatibility)
            
            avg_energy = np.mean(energy_levels)
            avg_warehouse = np.mean(warehouse_scores)
            best_warehouse = max(warehouse_scores)
            
            energy_progression.append({
                "generation": gen + 1,
                "avg_energy": avg_energy,
                "avg_warehouse_score": avg_warehouse,
                "best_warehouse_score": best_warehouse
            })
            
            print(f"   Gen {gen+1:2d}: Energy {avg_energy:.2f}, "
                  f"Warehouse avg {avg_warehouse:.3f}, best {best_warehouse:.3f}")
        
        # Show optimized patterns
        final_patterns = engine.get_best_patterns(3)
        
        print(f"\n🎯 Energy-Optimized Patterns:")
        for i, pattern in enumerate(final_patterns):
            energy_analysis = self.theory_engine.analyze_energy_curve(pattern.pattern)
            print(f"   {i+1}. {pattern.pattern.name}")
            print(f"      Energy: {energy_analysis.average_energy:.2f}")
            print(f"      Warehouse compatibility: {energy_analysis.warehouse_compatibility:.3f}")
            print(f"      Sustained intensity: {energy_analysis.sustained_intensity:.3f}")
            print(f"      Peak energy: {energy_analysis.peak_energy:.2f}")
        
        return {
            "strategy": "energy_optimization",
            "energy_progression": energy_progression,
            "final_patterns": [p.pattern.to_dict() for p in final_patterns],
            "optimization_success": energy_progression[-1]["best_warehouse_score"] > 0.7
        }
    
    async def demo_pattern_family_trees(self) -> Dict[str, any]:
        """Demonstrate pattern genealogy and family trees"""
        print("\n🌳 Pattern Family Trees Demo")
        print("=" * 50)
        
        config = BMADEvolutionConfig(
            population_size=15,
            generations=8,
            strategy=EvolutionStrategy.MUSICAL_PROGRESSION,
            diversity_pressure=0.4  # Encourage diversity
        )
        
        engine = BMADPatternEvolution(config)
        
        # Start with a small number of founder patterns
        founders = [
            create_gabber_kick_pattern("Founder_Alpha", 180),
            create_acid_pattern("Founder_Beta", 175)
        ]
        
        await engine.initialize_population(founders)
        
        # Track genealogy through generations
        family_tree = {}
        generation_stats = []
        
        for gen in range(config.generations):
            population = await engine.evolve_generation()
            
            # Build family tree
            for pattern in population:
                if pattern.parent_ids:
                    parent = pattern.parent_ids[0]
                    if parent not in family_tree:
                        family_tree[parent] = []
                    family_tree[parent].append({
                        "name": pattern.pattern.name,
                        "generation": pattern.generation,
                        "fitness": pattern.total_fitness,
                        "mutations": pattern.mutation_history[-3:]  # Last 3 mutations
                    })
            
            # Calculate diversity
            diversity_scores = [engine._calculate_pattern_diversity(p, population) for p in population]
            avg_diversity = np.mean(diversity_scores)
            
            generation_stats.append({
                "generation": gen + 1,
                "population_size": len(population),
                "unique_patterns": len(set(p.pattern.name for p in population)),
                "avg_diversity": avg_diversity,
                "families": len(family_tree)
            })
            
            print(f"   Gen {gen+1:2d}: Families={len(family_tree)}, "
                  f"Diversity={avg_diversity:.3f}, Patterns={len(population)}")
        
        # Analyze family success
        print(f"\n🌳 Family Tree Analysis:")
        successful_families = {}
        
        for parent, children in family_tree.items():
            if len(children) >= 2:  # Families with multiple children
                avg_fitness = np.mean([child["fitness"] for child in children])
                successful_families[parent] = {
                    "children_count": len(children),
                    "avg_fitness": avg_fitness,
                    "generations_span": max(child["generation"] for child in children) - min(child["generation"] for child in children),
                    "children": children
                }
        
        for family_name, family_data in sorted(successful_families.items(), 
                                             key=lambda x: x[1]["avg_fitness"], reverse=True)[:3]:
            print(f"   Family '{family_name}':")
            print(f"      Children: {family_data['children_count']}")
            print(f"      Avg fitness: {family_data['avg_fitness']:.3f}")
            print(f"      Generations: {family_data['generations_span']}")
            for child in family_data["children"][:3]:  # Show first 3 children
                print(f"        → {child['name']} (Gen {child['generation']}, Fitness {child['fitness']:.3f})")
        
        return {
            "strategy": "family_trees",
            "family_tree": family_tree,
            "generation_stats": generation_stats,
            "successful_families": successful_families
        }
    
    async def demo_mathematical_progression(self) -> Dict[str, any]:
        """Demonstrate mathematical progression tracking"""
        print("\n🧮 Mathematical Progression Tracking Demo")
        print("=" * 50)
        
        config = BMADEvolutionConfig(
            population_size=20,
            generations=10,
            strategy=EvolutionStrategy.MUSICAL_PROGRESSION,
            constraints=[
                MutationConstraint.STAY_IN_KEY,
                MutationConstraint.HARMONIC_PROGRESSION,
                MutationConstraint.RHYTHMIC_COHERENCE
            ]
        )
        
        engine = BMADPatternEvolution(config)
        
        # Start with musically basic patterns
        basic_pattern = create_gabber_kick_pattern("Basic_Pattern", 180)
        await engine.initialize_population([basic_pattern])
        
        mathematical_progression = []
        
        for gen in range(config.generations):
            population = await engine.evolve_generation()
            
            # Analyze mathematical metrics
            harmonic_complexity = []
            rhythmic_complexity = []
            mathematical_fitness = []
            
            for pattern in population:
                # Harmonic analysis
                harmonic_analysis = self.theory_engine.analyze_harmonic_content(pattern.pattern)
                harmonic_score = (
                    harmonic_analysis.harmonic_rhythm * 0.4 +
                    harmonic_analysis.dissonance_density * 0.3 +
                    (1 - harmonic_analysis.key_stability) * 0.3  # Controlled chromaticism
                )
                harmonic_complexity.append(harmonic_score)
                
                # Rhythmic analysis
                rhythmic_analysis = self.theory_engine.analyze_rhythmic_complexity(pattern.pattern)
                rhythmic_score = (
                    rhythmic_analysis.syncopation_density * 0.3 +
                    rhythmic_analysis.fractal_dimension * 0.4 +
                    rhythmic_analysis.groove_quality * 0.3
                )
                rhythmic_complexity.append(rhythmic_score)
                
                # Combined mathematical fitness
                math_fitness = (harmonic_score + rhythmic_score) / 2
                mathematical_fitness.append(math_fitness)
            
            avg_harmonic = np.mean(harmonic_complexity)
            avg_rhythmic = np.mean(rhythmic_complexity)
            avg_math_fitness = np.mean(mathematical_fitness)
            best_math_fitness = max(mathematical_fitness)
            
            mathematical_progression.append({
                "generation": gen + 1,
                "harmonic_complexity": avg_harmonic,
                "rhythmic_complexity": avg_rhythmic,
                "mathematical_fitness": avg_math_fitness,
                "best_mathematical_fitness": best_math_fitness
            })
            
            print(f"   Gen {gen+1:2d}: Math fitness {avg_math_fitness:.3f} (best {best_math_fitness:.3f}), "
                  f"Harmonic {avg_harmonic:.3f}, Rhythmic {avg_rhythmic:.3f}")
        
        # Show mathematical evolution
        best_patterns = engine.get_best_patterns(3)
        
        print(f"\n🎯 Mathematically Evolved Patterns:")
        for i, pattern in enumerate(best_patterns):
            harmonic_analysis = self.theory_engine.analyze_harmonic_content(pattern.pattern)
            rhythmic_analysis = self.theory_engine.analyze_rhythmic_complexity(pattern.pattern)
            
            print(f"   {i+1}. {pattern.pattern.name}")
            print(f"      Harmonic rhythm: {harmonic_analysis.harmonic_rhythm:.3f}")
            print(f"      Key stability: {harmonic_analysis.key_stability:.3f}")
            print(f"      Rhythmic complexity: {rhythmic_analysis.complexity_level.name}")
            print(f"      Fractal dimension: {rhythmic_analysis.fractal_dimension:.3f}")
            print(f"      Groove quality: {rhythmic_analysis.groove_quality:.3f}")
        
        return {
            "strategy": "mathematical_progression",
            "mathematical_progression": mathematical_progression,
            "final_patterns": [p.pattern.to_dict() for p in best_patterns]
        }
    
    async def run_complete_demonstration(self) -> Dict[str, any]:
        """Run complete demonstration of all evolution capabilities"""
        print("🚀 BMAD Complete Evolution Demonstration")
        print("=" * 70)
        print("Showcasing advanced pattern evolution for @innovation-lab")
        print()
        
        all_results = {}
        
        # Run all demonstrations
        demos = [
            ("BPM Progression Ladder", self.demo_bpm_progression_ladder),
            ("Genre Fusion Evolution", self.demo_genre_fusion_evolution),
            ("Energy Optimization", self.demo_energy_optimization),
            ("Pattern Family Trees", self.demo_pattern_family_trees),
            ("Mathematical Progression", self.demo_mathematical_progression)
        ]
        
        for demo_name, demo_func in demos:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            try:
                result = await demo_func()
                all_results[demo_name.lower().replace(" ", "_")] = result
                print(f"✅ {demo_name} completed successfully!")
            except Exception as e:
                print(f"❌ {demo_name} failed: {e}")
                all_results[demo_name.lower().replace(" ", "_")] = {"error": str(e)}
        
        # Summary
        print(f"\n🎉 BMAD Evolution Demonstration Complete!")
        print("=" * 70)
        
        successful_demos = sum(1 for result in all_results.values() if "error" not in result)
        print(f"   Successful demonstrations: {successful_demos}/{len(demos)}")
        
        # Show key achievements
        achievements = []
        
        if "bpm_progression_ladder" in all_results and all_results["bmp_progression_ladder"].get("success"):
            achievements.append("✅ BPM progression 180→220 BPM achieved")
        
        if "genre_fusion_evolution" in all_results and all_results["genre_fusion_evolution"].get("fusion_success"):
            achievements.append("✅ Successful multi-genre fusion")
        
        if "energy_optimization" in all_results and all_results["energy_optimization"].get("optimization_success"):
            achievements.append("✅ Warehouse energy optimization successful")
        
        if "pattern_family_trees" in all_results:
            family_data = all_results["pattern_family_trees"]
            if family_data.get("successful_families"):
                achievements.append(f"✅ {len(family_data['successful_families'])} successful pattern families")
        
        if achievements:
            print("\n🏆 Key Achievements:")
            for achievement in achievements:
                print(f"   {achievement}")
        
        # Save complete results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"bmad_evolution_demo_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        print("\n🧬 Ready for @innovation-lab to push hardcore music into new territories!")
        
        return all_results
    
    # Helper methods for creating specific pattern types
    def _create_frenchcore_pattern(self, name: str, bpm: float) -> HardcorePattern:
        """Create a frenchcore-style pattern"""
        pattern = create_gabber_kick_pattern(name, bpm)
        
        # Add melodic lead for frenchcore character
        pattern.add_track("frenchcore_lead")
        lead_params = SynthParams(
            freq=600, amp=0.7, crunch=0.9, drive=4.0,
            cutoff=8000, resonance=0.7,
            attack=0.001, decay=0.1, sustain=0.4, release=0.2
        )
        
        # Create melodic sequence
        melody_steps = [0, 2, 4, 6, 8, 10, 12, 14]  # Every other step
        frequencies = [600, 800, 750, 900, 650, 850, 700, 950]  # Melodic pattern
        
        for i, step in enumerate(melody_steps):
            step_params = lead_params.copy()
            step_params.freq = frequencies[i]
            lead_step = PatternStep(SynthType.SCREECH_LEAD, step_params, velocity=0.8)
            pattern.set_step("frenchcore_lead", step, lead_step)
        
        return pattern
    
    def _create_high_energy_pattern(self, name: str, bpm: float) -> HardcorePattern:
        """Create a high-energy pattern"""
        pattern = create_gabber_kick_pattern(name, bpm)
        
        # Boost all energy parameters
        for track in pattern.tracks.values():
            for step in track.steps:
                if step is not None:
                    step.params.amp = min(1.0, step.params.amp * 1.3)
                    step.params.crunch = min(1.0, step.params.crunch + 0.2)
                    step.params.drive = min(10.0, step.params.drive * 1.5)
                    step.velocity = min(1.0, step.velocity * 1.2)
        
        # Add high-energy stabs
        pattern.add_track("energy_stabs")
        stab_params = SynthParams(
            freq=300, amp=0.9, crunch=1.0, drive=6.0,
            attack=0.001, decay=0.05, sustain=0.1, release=0.1
        )
        stab_step = PatternStep(SynthType.HARDCORE_STAB, stab_params, velocity=1.0)
        
        # Add stabs on offbeats for energy
        for step in [2, 6, 10, 14]:
            pattern.set_step("energy_stabs", step, stab_step)
        
        return pattern


# Standalone test and demo functions
async def run_quick_demo():
    """Run a quick demonstration of key features"""
    print("⚡ BMAD Evolution Quick Demo")
    print("=" * 40)
    
    demonstrator = BMADEvolutionDemonstrator()
    
    # Run BPM progression demo
    result = await demonstrator.demo_bpm_progression_ladder()
    
    print(f"\n🎯 Quick Demo Results:")
    print(f"   BPM progression successful: {result.get('success', False)}")
    print(f"   Generations evolved: {len(result.get('bpm_progression', []))}")
    print(f"   Final patterns created: {len(result.get('final_patterns', []))}")
    
    return result


async def run_full_demonstration():
    """Run the complete BMAD evolution demonstration"""
    demonstrator = BMADEvolutionDemonstrator()
    return await demonstrator.run_complete_demonstration()


# CLI interface
async def main():
    """Main CLI interface for BMAD evolution demonstrations"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        await run_quick_demo()
    else:
        await run_full_demonstration()


if __name__ == "__main__":
    asyncio.run(main())