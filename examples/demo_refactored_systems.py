#!/usr/bin/env python3
"""
Demonstration of Refactored Intelligent Music Systems

Shows integration between:
- Musical Context System for harmonic cohesion
- Modulation System for parameter automation  
- FM Synthesis Engine with AbstractSynthesizer interface
- Track Architecture with modulation support
- Genre Research integration
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the refactored systems
from cli_shared.musical_context import MusicalContext, Scale, ScaleType, Chord, ChordType
from audio.modulation.modulator import LFO, WaveformType, ModulationRouter
from audio.synthesis.fm_engine import FMSynthEngineExtended, FMAlgorithm
from audio.core.track import Track, TrackCollection
from cli_shared.models.hardcore_models import SynthParams
from cli_shared.models.midi_clips import note_name_to_midi


def demo_musical_context():
    """Demonstrate musical context system"""
    print("🎵 Musical Context System Demo")
    print("=" * 40)
    
    # Create A minor scale
    scale = Scale(root=note_name_to_midi("A3"), scale_type=ScaleType.NATURAL_MINOR)
    context = MusicalContext(key="Am", scale=scale, tempo=180.0, genre="hardcore")
    
    print(f"Key: {context.key}")
    print(f"Scale: {context.scale.scale_type.value}")
    print(f"Tempo: {context.tempo} BPM")
    print(f"Genre: {context.genre}")
    
    # Get scale notes
    scale_notes = scale.get_notes(octave_range=(3, 5))
    print(f"Scale notes (MIDI): {scale_notes[:8]}...")  # First 8 notes
    
    print()


def demo_modulation_system():
    """Demonstrate modulation system"""
    print("🌊 Modulation System Demo")
    print("=" * 40)
    
    # Create modulation router
    router = ModulationRouter()
    
    # Create LFO and envelope
    lfo = LFO(rate=1.0, waveform=WaveformType.SINE)
    print(f"LFO Rate: {lfo.rate} Hz")
    print(f"LFO Waveform: {lfo.waveform.value}")
    
    # Connect to virtual parameter
    connection = router.connect(lfo, "cutoff", amount=0.5)
    print(f"Connected LFO to 'cutoff' with amount {connection.amount}")
    
    # Update modulation at different times
    for time in [0.0, 0.25, 0.5, 0.75]:
        modulated = router.update(time, 44100)
        cutoff_mod = modulated.get("cutoff", 0.0)
        print(f"  Time {time}s: cutoff modulation = {cutoff_mod:.3f}")
    
    print()


def demo_fm_synthesis():
    """Demonstrate FM synthesis engine"""
    print("🎛️ FM Synthesis Demo")
    print("=" * 40)
    
    engine = FMSynthEngineExtended()
    print(f"Sample Rate: {engine.sample_rate} Hz")
    print(f"Backend Type: {engine.backend_type.value}")
    
    # Load hardcore preset
    engine.load_preset("classic_hoover")
    print(f"Loaded preset: classic_hoover")
    print(f"Algorithm: {engine.algorithm.name}")
    
    # Generate test note
    engine.note_on(220.0, 0.8)  # A3
    audio = engine.generate_samples(1000)
    engine.note_off()
    
    print(f"Generated {len(audio)} samples")
    print(f"Peak amplitude: {np.max(np.abs(audio)):.3f}")
    print(f"RMS level: {np.sqrt(np.mean(audio**2)):.3f}")
    
    print()


def demo_track_integration():
    """Demonstrate track system with modulation"""
    print("🎚️ Track Integration Demo")
    print("=" * 40)
    
    # Create track with kick pattern
    track = Track("Modulated Gabber Kick")
    track.add_kick_pattern("x ~ ~ ~ x ~ ~ ~", frequency=60.0)
    
    # Add LFO modulation to volume
    lfo = track.add_lfo_modulation("volume", rate=2.0, amplitude=0.3)
    print(f"Added LFO modulation to volume: {lfo.rate} Hz")
    
    # Render several steps to show modulation
    params = SynthParams()
    params.frequency = 60.0
    
    print("Rendering kick pattern with modulation:")
    for step in range(8):
        audio = track.render_step(step, 180.0, params)
        volume = track.volume
        pattern_char = "x" if len(audio) > 0 else "~"
        print(f"  Step {step}: '{pattern_char}' volume={volume:.3f} samples={len(audio)}")
    
    print()


def demo_complete_session():
    """Demonstrate complete session with multiple tracks"""
    print("🎼 Complete Session Demo")
    print("=" * 40)
    
    # Create musical context
    scale = Scale(root=note_name_to_midi("A2"), scale_type=ScaleType.NATURAL_MINOR)
    context = MusicalContext(key="Am", scale=scale, tempo=180.0, genre="hardcore")
    
    # Create track collection
    session = TrackCollection("Hardcore Demo Session")
    
    # Track 1: Kick drum
    kick_track = Track("Kick")
    kick_track.add_kick_pattern("x ~ ~ ~ x ~ ~ ~", frequency=60.0)
    kick_track.add_lfo_modulation("volume", rate=0.5, amplitude=0.1)
    session.add_track(kick_track)
    
    # Track 2: FM synth lead
    lead_track = Track("FM Lead")
    lead_track.add_kick_pattern("~ x ~ x ~ x ~ x", frequency=220.0)  # Using kick pattern for simplicity
    # In real usage, would use FM synthesizer source
    lead_track.add_lfo_modulation("volume", rate=1.0, amplitude=0.2)
    session.add_track(lead_track)
    
    print(f"Session: {session.name}")
    print(f"Tracks: {len(session.tracks)}")
    print(f"Musical Context: {context.key} {context.scale.scale_type.value} @ {context.tempo} BPM")
    
    # Render a few steps of the full mix
    params = SynthParams()
    params.frequency = 220.0
    
    print("Rendering multi-track session:")
    for step in range(4):
        mixed_audio = session.render_step(step, context.tempo, params)
        peak = np.max(np.abs(mixed_audio)) if len(mixed_audio) > 0 else 0.0
        print(f"  Step {step}: mixed samples={len(mixed_audio)} peak={peak:.3f}")
    
    print()


def demo_error_handling():
    """Demonstrate error handling and graceful degradation"""
    print("⚠️ Error Handling Demo")
    print("=" * 40)
    
    # Test track with no audio source
    empty_track = Track("Empty Track")
    params = SynthParams()
    
    audio = empty_track.render_step(0, 180.0, params)
    print(f"Empty track audio length: {len(audio)} (should be 0)")
    
    # Test modulation with invalid parameter
    try:
        empty_track.add_lfo_modulation("nonexistent_param", 1.0, 1.0)
        print("Invalid parameter modulation handled gracefully")
    except Exception as e:
        print(f"Error with invalid parameter: {e}")
    
    # Test FM engine with edge cases
    engine = FMSynthEngineExtended()
    try:
        # Zero velocity
        audio = engine.render_pattern_step(0.0, params)
        print(f"Zero velocity handled: {len(audio)} samples")
        
        # Negative velocity
        audio = engine.render_pattern_step(-0.5, params)
        print(f"Negative velocity handled: {len(audio)} samples")
    except Exception as e:
        print(f"FM engine error handling: {e}")
    
    print()


if __name__ == "__main__":
    print("🎹 Refactored Intelligent Music Systems Demo")
    print("=" * 50)
    print()
    
    try:
        demo_musical_context()
        demo_modulation_system()
        demo_fm_synthesis()
        demo_track_integration()
        demo_complete_session()
        demo_error_handling()
        
        print("✅ All refactored systems working correctly!")
        print()
        print("Next steps:")
        print("- Add intelligent pattern generation")
        print("- Integrate AI agent for creative assistance")
        print("- Add real-time audio analysis feedback")
        print("- Implement genre-specific style transfer")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()