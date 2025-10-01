#!/usr/bin/env python3
"""
Track Architecture Demo - Clean Professional System

Demonstrates the new clean Track architecture that uses:
- Existing cli_shared interfaces (NO reinventing wheels)
- Extracted audio modules (NO spaghetti code) 
- DAW-style composition (NO arbitrary inheritance)

Architecture: Control Source → Audio Source → FX Chain → Mixer
"""

import numpy as np
import sys
import os

# Add project root to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from audio import (
    Track, TrackCollection,
    PatternControlSource, KickAudioSource,
    HardcoreConstants, SynthesisParams
)


def create_gabber_kick_track():
    """Create a Rotterdam gabber kick track"""
    
    # Create track using composition pattern
    kick_track = Track("Gabber Kick")
    
    # Add kick pattern using extracted synthesis
    kick_track.add_kick_pattern(
        pattern="x ~ x ~ x ~ x ~",  # 4/4 gabber pattern  
        frequency=55.0,            # Rotterdam frequency
        duration_ms=600            # Long gabber kick
    )
    
    # Add gabber effects chain using extracted modules
    kick_track.add_gabber_effects()
    
    # Mixer settings
    kick_track.volume = 0.8
    kick_track.pan = 0.0  # Center
    
    return kick_track


def create_industrial_session():
    """Create a complete industrial hardcore session"""
    
    # Create session
    session = TrackCollection("Industrial Hardcore")
    
    # Kick track
    kick_track = Track("Industrial Kick")
    kick_track.add_kick_pattern(
        pattern="x ~ ~ ~ x ~ x ~",  # Industrial pattern
        frequency=45.0,             # Lower for industrial 
        duration_ms=800             # Longer decay
    )
    
    # Add industrial effects
    kick_track.effects_chain.add_compression(ratio=4.0, threshold_db=-15)
    kick_track.effects_chain.add_reverb("warehouse", wet_level=0.4)
    
    session.add_track(kick_track)
    
    # Hi-hat track (would use synthesizer source in real usage)
    hihat_track = Track("Industrial Hihat")
    hihat_track.add_kick_pattern(
        pattern="~ x ~ x ~ x ~ x",  # Off-beat pattern
        frequency=8000.0,           # High frequency
        duration_ms=50              # Short hihat
    )
    hihat_track.volume = 0.3        # Quieter than kick
    
    session.add_track(hihat_track)
    
    return session


def render_demo_audio(session: TrackCollection, steps: int = 16, bpm: float = 150):
    """Render demo audio using the new Track architecture"""
    
    print(f"🎵 Rendering {session.name} - {steps} steps at {bpm} BPM")
    
    # Synthesis parameters  
    params = SynthesisParams(
        frequency=55.0,
        duration_ms=600,
        bpm=int(bpm),
        brutality=0.7,
        analog_warmth=0.8
    )
    
    # Render each step
    rendered_steps = []
    for step in range(steps):
        step_audio = session.render_step(step, bpm, params)
        rendered_steps.append(step_audio)
        
        print(f"Step {step:2d}: {len(step_audio):5d} samples")
    
    print(f"✅ Rendered {len(rendered_steps)} steps successfully")
    print(f"📊 Using clean Track architecture with extracted modules")
    print(f"🏗️  Architecture: Control Source → Audio Source → FX Chain → Mixer")
    
    return rendered_steps


def main():
    """Demo the new Track architecture"""
    
    print("=" * 60)
    print("🎛️  TRACK ARCHITECTURE DEMO - Clean Professional System")
    print("=" * 60)
    print()
    
    print("🧹 NO spaghetti inheritance")
    print("🔌 Uses existing cli_shared interfaces") 
    print("📦 Uses extracted audio modules")
    print("🎚️  DAW-style composition pattern")
    print()
    
    # Demo 1: Single gabber kick track
    print("Demo 1: Rotterdam Gabber Kick Track")
    print("-" * 40)
    
    gabber_track = create_gabber_kick_track()
    single_session = TrackCollection("Demo 1")
    single_session.add_track(gabber_track)
    
    render_demo_audio(single_session, steps=8, bpm=170)
    print()
    
    # Demo 2: Multi-track industrial session
    print("Demo 2: Multi-Track Industrial Session") 
    print("-" * 40)
    
    industrial_session = create_industrial_session()
    render_demo_audio(industrial_session, steps=16, bpm=150)
    print()
    
    print("=" * 60)
    print("✅ Track Architecture Demo Complete!")
    print("🏆 Clean, modular, professional architecture")
    print("🚫 NO engines/ spaghetti code")
    print("✨ Uses existing infrastructure properly")
    print("=" * 60)


if __name__ == "__main__":
    main()