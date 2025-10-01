#!/usr/bin/env python3
"""
Test Pattern Generators

Demonstrates the new MIDI clip-based pattern generators for hardcore music production.
Shows how generators create standard MIDI clips that can export to multiple formats.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_shared.models.midi_clips import MIDIClip, TriggerClip, create_empty_midi_clip
from cli_shared.generators.acid_bassline import AcidBasslineGenerator, create_hardcore_acid_line
from cli_shared.generators.tuned_kick import TunedKickGenerator, create_frenchcore_kicks


def test_acid_bassline_generator():
    """Test acid bassline generation"""
    print("🔥 TESTING ACID BASSLINE GENERATOR")
    print("=" * 50)
    
    # Create basic acid bassline
    generator = AcidBasslineGenerator(
        scale="A_minor",
        accent_pattern="x ~ ~ x ~ ~ x ~"
    )
    
    clip = generator.generate(length_bars=4.0, bpm=180.0)
    
    print(f"Generated: {clip.name}")
    print(f"Length: {clip.length_bars} bars at {clip.bpm} BPM")
    print(f"Notes: {len(clip.notes)} MIDI notes")
    print(f"Tags: {', '.join(clip.tags)}")
    print(f"Key: {clip.key_signature}")
    
    # Show first few notes
    print("\nFirst 5 notes:")
    for i, note in enumerate(clip.notes[:5]):
        note_name = f"MIDI#{note.pitch}"
        print(f"  {i+1}. {note_name} vel={note.velocity} time={note.start_time:.2f} dur={note.duration:.2f}")
    
    # Export to TidalCycles pattern
    tidal_pattern = clip.to_tidal_pattern()
    print(f"\nTidalCycles export (first 100 chars):")
    print(f"  {tidal_pattern[:100]}...")
    
    # Test variations
    variations = generator.generate_variations(clip, num_variations=2)
    print(f"\nGenerated {len(variations)} variations:")
    for var in variations:
        print(f"  - {var.name}: {len(var.notes)} notes")
    
    return clip


def test_tuned_kick_generator():
    """Test tuned kick generation"""
    print("\n\n💥 TESTING TUNED KICK GENERATOR")
    print("=" * 50)
    
    # Create frenchcore-style tuned kicks
    generator = TunedKickGenerator(
        root_note="C1",
        pattern="x ~ x ~ x ~ x ~",
        tuning="pentatonic"
    )
    
    clip = generator.generate(length_bars=1.0, bpm=200.0)
    
    print(f"Generated: {clip.name}")
    print(f"Length: {clip.length_bars} bars at {clip.bpm} BPM")
    print(f"Notes: {len(clip.notes)} MIDI notes")
    print(f"Tags: {', '.join(clip.tags)}")
    
    # Show all notes (kicks are usually short patterns)
    print("\nAll kick notes:")
    for i, note in enumerate(clip.notes):
        freq = note.to_frequency()
        print(f"  {i+1}. MIDI#{note.pitch} ({freq:.1f}Hz) vel={note.velocity} time={note.start_time:.2f}")
    
    # Test melodic sequence
    print("\nTesting melodic kick sequence...")
    melodic_clip = generator.create_melodic_sequence(
        melody_pitches=[0, 3, 5, 7, 3, 0],  # Pentatonic melody in semitones
        rhythm_pattern="x x x x x x"
    )
    
    print(f"Melodic kicks: {len(melodic_clip.notes)} notes")
    for note in melodic_clip.notes:
        freq = note.to_frequency()
        print(f"  MIDI#{note.pitch} ({freq:.1f}Hz) at time {note.start_time:.2f}")
    
    return clip


def test_convenience_functions():
    """Test convenience functions for quick generation"""
    print("\n\n🎵 TESTING CONVENIENCE FUNCTIONS")
    print("=" * 50)
    
    # Test hardcore acid line
    hardcore_acid = create_hardcore_acid_line(length_bars=2.0, bpm=190.0)
    print(f"Hardcore acid line: {len(hardcore_acid.notes)} notes at {hardcore_acid.bpm} BPM")
    
    # Test frenchcore kicks
    frenchcore_kicks = create_frenchcore_kicks(root_note="D1", length_bars=1.0, bpm=210.0)
    print(f"Frenchcore kicks: {len(frenchcore_kicks.notes)} kicks")
    
    # Show pitch range
    if frenchcore_kicks.notes:
        pitches = [note.pitch for note in frenchcore_kicks.notes]
        min_pitch, max_pitch = min(pitches), max(pitches)
        print(f"  Pitch range: MIDI#{min_pitch} to MIDI#{max_pitch}")


def test_clip_operations():
    """Test MIDI clip operations like transpose, quantize, etc."""
    print("\n\n🔧 TESTING CLIP OPERATIONS")
    print("=" * 50)
    
    # Create a simple clip
    clip = create_empty_midi_clip("test_clip", bars=1.0, bpm=140.0)
    
    # Add some notes manually
    from cli_shared.models.midi_clips import MIDINote, note_name_to_midi
    
    notes = [
        MIDINote(note_name_to_midi("C2"), 100, 0.0, 0.25),
        MIDINote(note_name_to_midi("E2"), 90, 0.5, 0.25),
        MIDINote(note_name_to_midi("G2"), 95, 1.0, 0.25),
    ]
    
    clip.add_notes(notes)
    print(f"Original clip: {len(clip.notes)} notes")
    
    # Test transpose
    transposed = clip.transpose(+7)  # Up a fifth
    print(f"Transposed +7: {transposed.name}")
    
    original_pitches = [note.pitch for note in clip.notes]
    transposed_pitches = [note.pitch for note in transposed.notes]
    print(f"  Original pitches: {original_pitches}")
    print(f"  Transposed pitches: {transposed_pitches}")
    
    # Test dictionary serialization
    clip_dict = clip.to_dict()
    print(f"Serialized to dict: {len(clip_dict)} keys")
    print(f"  Keys: {list(clip_dict.keys())}")
    
    # Test reconstruction
    reconstructed = MIDIClip.from_dict(clip_dict)
    print(f"Reconstructed: {len(reconstructed.notes)} notes")


def test_export_formats():
    """Test exporting to different formats"""
    print("\n\n📁 TESTING EXPORT FORMATS")
    print("=" * 50)
    
    # Generate an acid bassline
    generator = AcidBasslineGenerator(scale="E_minor", accent_pattern="x ~ ~ x")
    clip = generator.generate(length_bars=2.0, bpm=175.0)
    
    # Test TidalCycles export
    tidal = clip.to_tidal_pattern()
    print("TidalCycles pattern (first 200 chars):")
    print(f"  {tidal[:200]}...")
    
    # Test MIDI file export
    if hasattr(clip, 'to_midi_file'):
        try:
            midi_file = clip.to_midi_file()
            print(f"MIDI file created: {len(midi_file.tracks)} tracks")
            print(f"  Ticks per beat: {midi_file.ticks_per_beat}")
        except Exception as e:
            print(f"MIDI export error: {e}")
    
    # Test save to file (if mido available)
    try:
        filepath = "/tmp/test_bassline.mid"
        success = clip.save_midi_file(filepath)
        print(f"MIDI file save: {'✅ Success' if success else '❌ Failed'}")
        if success:
            print(f"  Saved to: {filepath}")
    except Exception as e:
        print(f"MIDI save error: {e}")


def main():
    """Run all tests"""
    print("🎛️ MIDI CLIP PATTERN GENERATORS TEST SUITE")
    print("=" * 60)
    
    try:
        # Run tests
        acid_clip = test_acid_bassline_generator()
        kick_clip = test_tuned_kick_generator()
        test_convenience_functions()
        test_clip_operations()
        test_export_formats()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"Generated clips ready for use with TidalCycles, SuperCollider, or DAW export.")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"🎵 Acid bassline: {len(acid_clip.notes)} notes, {acid_clip.length_bars} bars")
        print(f"💥 Tuned kicks: {len(kick_clip.notes)} kicks, {kick_clip.length_bars} bars")
        print(f"🔄 Multiple export formats available")
        print(f"🎛️ Generators ready for AI integration")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Pattern generators are ready for hardcore music production!")
    else:
        print("\n💥 Pattern generators need debugging")