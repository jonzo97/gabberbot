#!/usr/bin/env python3
"""
BMAD Music Coordinator Examples - Real Hardcore Music Generation

This file provides various examples of how to use the BMAD music coordinator
to generate authentic hardcore tracks. All examples generate real MIDI and WAV files.

Examples:
1. Quick single track generation
2. Style comparison generation  
3. BPM variation study
4. Collection generation
5. Overnight factory mode

All tracks are generated using the BMAD agent workflow:
- @music-analyst (Nexus): Pattern analysis
- @music-producer (Raven): MIDI generation
- @sound-designer (Void): Audio synthesis
- @mix-engineer (Phoenix): Professional mixing
"""

import random
from bmad_simple_test import BMadSimpleCoordinator, BMadTrackConfig, HardcoreStyle


def example_1_quick_gabber():
    """Example 1: Generate a quick Rotterdam gabber track"""
    print("=" * 60)
    print("EXAMPLE 1: Quick Rotterdam Gabber Track")
    print("=" * 60)
    
    coordinator = BMadSimpleCoordinator()
    
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=180.0,
        length_bars=16.0,
        key="A_minor",
        seed=12345  # Reproducible results
    )
    
    session_id = coordinator.generate_hardcore_track(config)
    print(f"\nGenerated session: {session_id}")
    return session_id


def example_2_style_comparison():
    """Example 2: Generate tracks in different hardcore styles"""
    print("=" * 60)
    print("EXAMPLE 2: Hardcore Style Comparison")
    print("=" * 60)
    
    coordinator = BMadSimpleCoordinator()
    sessions = []
    
    styles = [
        (HardcoreStyle.ROTTERDAM_GABBER, 180.0, "Classic Rotterdam sound"),
        (HardcoreStyle.FRENCHCORE, 200.0, "Aggressive French hardcore")
    ]
    
    for i, (style, bpm, description) in enumerate(styles):
        print(f"\nGenerating {style.name} ({description})...")
        
        config = BMadTrackConfig(
            style=style,
            bpm=bpm,
            length_bars=12.0,
            key="E_minor",
            seed=2000 + i
        )
        
        session_id = coordinator.generate_hardcore_track(config)
        sessions.append(session_id)
    
    print(f"\nStyle comparison complete!")
    print(f"Generated {len(sessions)} different hardcore styles")
    return sessions


def example_3_bpm_variations():
    """Example 3: Generate tracks at different BPMs"""
    print("=" * 60)
    print("EXAMPLE 3: BPM Variation Study")
    print("=" * 60)
    
    coordinator = BMadSimpleCoordinator()
    sessions = []
    
    bpm_variants = [160, 180, 200, 220]
    
    for i, bpm in enumerate(bpm_variants):
        print(f"\nGenerating gabber track at {bpm} BPM...")
        
        config = BMadTrackConfig(
            style=HardcoreStyle.ROTTERDAM_GABBER,
            bpm=float(bpm),
            length_bars=8.0,
            key="A_minor",
            seed=3000 + i
        )
        
        session_id = coordinator.generate_hardcore_track(config)
        sessions.append(session_id)
    
    print(f"\nBPM study complete!")
    print(f"Generated {len(sessions)} tracks at different tempos")
    return sessions


def example_4_track_collection():
    """Example 4: Generate a collection of varied hardcore tracks"""
    print("=" * 60)
    print("EXAMPLE 4: Hardcore Track Collection")
    print("=" * 60)
    
    coordinator = BMadSimpleCoordinator()
    sessions = []
    
    # Configuration variations
    track_configs = [
        {"style": HardcoreStyle.ROTTERDAM_GABBER, "bpm": 175, "bars": 16, "key": "A_minor"},
        {"style": HardcoreStyle.FRENCHCORE, "bpm": 195, "bars": 20, "key": "E_minor"},
        {"style": HardcoreStyle.ROTTERDAM_GABBER, "bpm": 185, "bars": 24, "key": "C_minor"},
        {"style": HardcoreStyle.FRENCHCORE, "bpm": 210, "bars": 12, "key": "A_minor"},
        {"style": HardcoreStyle.ROTTERDAM_GABBER, "bpm": 165, "bars": 32, "key": "E_minor"}
    ]
    
    for i, track_config in enumerate(track_configs):
        print(f"\nGenerating track {i+1}/{len(track_configs)}...")
        
        config = BMadTrackConfig(
            style=track_config["style"],
            bpm=float(track_config["bpm"]),
            length_bars=float(track_config["bars"]),
            key=track_config["key"],
            seed=4000 + i
        )
        
        try:
            session_id = coordinator.generate_hardcore_track(config)
            sessions.append(session_id)
        except Exception as e:
            print(f"Track {i+1} failed: {e}")
    
    print(f"\nCollection complete!")
    print(f"Generated {len(sessions)} hardcore tracks")
    return sessions


def example_5_random_generation():
    """Example 5: Generate random hardcore tracks"""
    print("=" * 60)
    print("EXAMPLE 5: Random Hardcore Generation")
    print("=" * 60)
    
    coordinator = BMadSimpleCoordinator()
    sessions = []
    
    num_tracks = 3
    styles = [HardcoreStyle.ROTTERDAM_GABBER, HardcoreStyle.FRENCHCORE]
    keys = ["A_minor", "E_minor", "C_minor"]
    
    for i in range(num_tracks):
        print(f"\nGenerating random track {i+1}/{num_tracks}...")
        
        # Random configuration
        style = random.choice(styles)
        bpm_range = (170, 190) if style == HardcoreStyle.ROTTERDAM_GABBER else (190, 220)
        
        config = BMadTrackConfig(
            style=style,
            bpm=random.uniform(*bpm_range),
            length_bars=random.choice([8, 12, 16, 20, 24]),
            key=random.choice(keys),
            seed=random.randint(5000, 9999)
        )
        
        try:
            session_id = coordinator.generate_hardcore_track(config)
            sessions.append(session_id)
        except Exception as e:
            print(f"Random track {i+1} failed: {e}")
    
    print(f"\nRandom generation complete!")
    print(f"Generated {len(sessions)} random hardcore tracks")
    return sessions


def example_6_mini_factory():
    """Example 6: Mini hardcore music factory"""
    print("=" * 60)
    print("EXAMPLE 6: Mini Hardcore Music Factory")
    print("=" * 60)
    
    coordinator = BMadSimpleCoordinator()
    sessions = []
    
    factory_configs = [
        # Classic gabber selection
        {"name": "Classic Gabber 1", "style": HardcoreStyle.ROTTERDAM_GABBER, "bpm": 175, "bars": 16},
        {"name": "Classic Gabber 2", "style": HardcoreStyle.ROTTERDAM_GABBER, "bpm": 180, "bars": 20},
        {"name": "Classic Gabber 3", "style": HardcoreStyle.ROTTERDAM_GABBER, "bpm": 185, "bars": 24},
        
        # Frenchcore selection
        {"name": "Frenchcore 1", "style": HardcoreStyle.FRENCHCORE, "bpm": 195, "bars": 16},
        {"name": "Frenchcore 2", "style": HardcoreStyle.FRENCHCORE, "bpm": 200, "bars": 12},
        {"name": "Frenchcore 3", "style": HardcoreStyle.FRENCHCORE, "bpm": 210, "bars": 18},
    ]
    
    for i, track_info in enumerate(factory_configs):
        print(f"\nFactory producing: {track_info['name']}")
        
        config = BMadTrackConfig(
            style=track_info["style"],
            bpm=float(track_info["bpm"]),
            length_bars=float(track_info["bars"]),
            key=random.choice(["A_minor", "E_minor"]),
            seed=6000 + i
        )
        
        try:
            session_id = coordinator.generate_hardcore_track(config)
            sessions.append(session_id)
            print(f"   Factory output: {session_id}")
        except Exception as e:
            print(f"   Factory error: {e}")
    
    print(f"\nMini factory run complete!")
    print(f"Factory produced {len(sessions)} hardcore tracks")
    return sessions


def run_all_examples():
    """Run all BMAD examples"""
    print("BMAD HARDCORE MUSIC COORDINATOR - ALL EXAMPLES")
    print("=" * 70)
    print("This will generate multiple hardcore tracks using different approaches")
    print("All tracks will be saved as MIDI and WAV files")
    print()
    
    all_sessions = []
    
    # Run examples
    try:
        sessions = example_1_quick_gabber()
        all_sessions.extend(sessions if isinstance(sessions, list) else [sessions])
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        sessions = example_2_style_comparison()
        all_sessions.extend(sessions)
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        sessions = example_3_bpm_variations()
        all_sessions.extend(sessions)
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print(f"Total tracks generated: {len(all_sessions)}")
    print(f"Check bmad_output/ directory for all your hardcore tracks!")
    print("Each session contains:")
    print("  - MIDI files (kick.mid, bassline.mid)")
    print("  - Individual audio tracks (kick_track.wav, bassline_track.wav)")
    print("  - Final mixed track (*_final.wav)")
    
    return all_sessions


def demo_bmad_workflow():
    """Demonstrate the BMAD agent workflow"""
    print("BMAD AGENT WORKFLOW DEMONSTRATION")
    print("=" * 50)
    print("This demonstrates how the BMAD agents work together:")
    print()
    print("1. @music-analyst (Nexus) - Analyzes hardcore patterns")
    print("2. @music-producer (Raven) - Generates MIDI patterns")
    print("3. @sound-designer (Void) - Synthesizes audio")
    print("4. @mix-engineer (Phoenix) - Creates professional mix")
    print()
    
    return example_1_quick_gabber()


if __name__ == "__main__":
    # Run a demonstration
    demo_bmad_workflow()