#!/usr/bin/env python3
"""
Unit tests for MIDIClip class

Comprehensive testing of core MIDI clip functionality including:
- Note management
- Time calculations  
- Transposition
- Serialization/deserialization
- Export functionality
- Edge cases and error handling
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cli_shared.models.midi_clips import (
    MIDIClip, MIDINote, TimeSignature, 
    note_name_to_midi, midi_to_note_name,
    create_empty_midi_clip
)


class TestMIDINote(unittest.TestCase):
    """Test MIDINote class"""
    
    def test_note_creation(self):
        """Test basic note creation"""
        note = MIDINote(
            pitch=60,  # Middle C
            velocity=100,
            start_time=0.0,
            duration=1.0,
            channel=0
        )
        
        self.assertEqual(note.pitch, 60)
        self.assertEqual(note.velocity, 100)
        self.assertEqual(note.start_time, 0.0)
        self.assertEqual(note.duration, 1.0)
        self.assertEqual(note.channel, 0)
    
    def test_note_frequency_conversion(self):
        """Test MIDI note to frequency conversion"""
        # Middle C (C4) should be ~261.63 Hz
        middle_c = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        freq = middle_c.to_frequency()
        self.assertAlmostEqual(freq, 261.63, places=1)
        
        # A4 (440 Hz reference)
        a4 = MIDINote(pitch=69, velocity=100, start_time=0.0, duration=1.0)
        self.assertAlmostEqual(a4.to_frequency(), 440.0, places=1)
    
    def test_note_transposition(self):
        """Test note transposition"""
        note = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        
        # Transpose up an octave
        transposed = note.transpose(12)
        self.assertEqual(transposed.pitch, 72)
        self.assertEqual(transposed.velocity, 100)  # Other properties unchanged
        
        # Transpose down
        transposed_down = note.transpose(-5)
        self.assertEqual(transposed_down.pitch, 55)
        
        # Test bounds checking
        high_note = MIDINote(pitch=120, velocity=100, start_time=0.0, duration=1.0)
        transposed_high = high_note.transpose(10)
        self.assertEqual(transposed_high.pitch, 127)  # Should clamp to max
        
        low_note = MIDINote(pitch=5, velocity=100, start_time=0.0, duration=1.0)
        transposed_low = low_note.transpose(-10)
        self.assertEqual(transposed_low.pitch, 0)  # Should clamp to min
    
    def test_note_serialization(self):
        """Test note dictionary conversion"""
        note = MIDINote(pitch=64, velocity=127, start_time=1.5, duration=0.25, channel=9)
        
        # To dict
        note_dict = note.to_dict()
        expected_keys = {'pitch', 'velocity', 'start_time', 'duration', 'channel'}
        self.assertEqual(set(note_dict.keys()), expected_keys)
        self.assertEqual(note_dict['pitch'], 64)
        
        # From dict
        reconstructed = MIDINote.from_dict(note_dict)
        self.assertEqual(reconstructed.pitch, note.pitch)
        self.assertEqual(reconstructed.velocity, note.velocity)
        self.assertEqual(reconstructed.start_time, note.start_time)
        self.assertEqual(reconstructed.duration, note.duration)
        self.assertEqual(reconstructed.channel, note.channel)


class TestTimeSignature(unittest.TestCase):
    """Test TimeSignature class"""
    
    def test_standard_time_signatures(self):
        """Test common time signatures"""
        # 4/4 time
        ts_44 = TimeSignature(4, 4)
        self.assertEqual(ts_44.beats_per_bar(), 4.0)
        
        # 3/4 time
        ts_34 = TimeSignature(3, 4)
        self.assertEqual(ts_34.beats_per_bar(), 3.0)
        
        # 7/8 time
        ts_78 = TimeSignature(7, 8)
        self.assertEqual(ts_78.beats_per_bar(), 7.0)
    
    def test_time_signature_serialization(self):
        """Test time signature dict conversion"""
        ts = TimeSignature(5, 8)
        ts_dict = ts.to_dict()
        
        self.assertEqual(ts_dict['numerator'], 5)
        self.assertEqual(ts_dict['denominator'], 8)


class TestMIDIClip(unittest.TestCase):
    """Test MIDIClip class"""
    
    def setUp(self):
        """Set up test clip"""
        self.clip = MIDIClip(
            name="test_clip",
            length_bars=2.0,
            bpm=120.0
        )
    
    def test_clip_creation(self):
        """Test basic clip creation"""
        self.assertEqual(self.clip.name, "test_clip")
        self.assertEqual(self.clip.length_bars, 2.0)
        self.assertEqual(self.clip.bpm, 120.0)
        self.assertEqual(len(self.clip.notes), 0)
        self.assertEqual(self.clip.get_total_beats(), 8.0)  # 2 bars * 4 beats
    
    def test_note_management(self):
        """Test adding and managing notes"""
        # Add single note
        note1 = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.5)
        self.clip.add_note(note1)
        self.assertEqual(len(self.clip.notes), 1)
        
        # Add multiple notes
        notes = [
            MIDINote(pitch=64, velocity=90, start_time=1.0, duration=0.5),
            MIDINote(pitch=67, velocity=95, start_time=2.0, duration=0.5)
        ]
        self.clip.add_notes(notes)
        self.assertEqual(len(self.clip.notes), 3)
        
        # Clear notes
        self.clip.clear_notes()
        self.assertEqual(len(self.clip.notes), 0)
    
    def test_note_queries(self):
        """Test note query methods"""
        # Add test notes
        notes = [
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.5),
            MIDINote(pitch=64, velocity=90, start_time=0.0, duration=0.5),  # Same time
            MIDINote(pitch=67, velocity=95, start_time=1.0, duration=0.5),
            MIDINote(pitch=72, velocity=85, start_time=3.0, duration=0.5)
        ]
        self.clip.add_notes(notes)
        
        # Test get_notes_at_time
        notes_at_zero = self.clip.get_notes_at_time(0.0)
        self.assertEqual(len(notes_at_zero), 2)  # Two notes at time 0.0
        
        notes_at_one = self.clip.get_notes_at_time(1.0)
        self.assertEqual(len(notes_at_one), 1)
        
        # Test get_notes_in_range
        notes_in_range = self.clip.get_notes_in_range(0.0, 2.0)
        self.assertEqual(len(notes_in_range), 3)  # Notes at 0.0, 0.0, 1.0
        
        empty_range = self.clip.get_notes_in_range(5.0, 6.0)
        self.assertEqual(len(empty_range), 0)
    
    def test_transposition(self):
        """Test clip transposition"""
        # Add test notes
        notes = [
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.5),
            MIDINote(pitch=64, velocity=90, start_time=1.0, duration=0.5),
            MIDINote(pitch=67, velocity=95, start_time=2.0, duration=0.5)
        ]
        self.clip.add_notes(notes)
        
        # Transpose up a fifth (7 semitones)
        transposed_clip = self.clip.transpose(7)
        
        # Check clip properties
        self.assertEqual(transposed_clip.name, "test_clip_transposed_+7")
        self.assertEqual(transposed_clip.bpm, self.clip.bpm)
        self.assertEqual(len(transposed_clip.notes), 3)
        
        # Check note transposition
        expected_pitches = [67, 71, 74]  # Original + 7
        actual_pitches = [note.pitch for note in transposed_clip.notes]
        self.assertEqual(actual_pitches, expected_pitches)
        
        # Original should be unchanged
        original_pitches = [note.pitch for note in self.clip.notes]
        self.assertEqual(original_pitches, [60, 64, 67])
    
    def test_quantization(self):
        """Test note quantization"""
        # Add notes with off-grid timing
        notes = [
            MIDINote(pitch=60, velocity=100, start_time=0.08, duration=0.5),  # Closer to 0.0
            MIDINote(pitch=64, velocity=90, start_time=1.07, duration=0.5),   # Closer to 1.0
            MIDINote(pitch=67, velocity=95, start_time=2.0, duration=0.5)     # On grid
        ]
        self.clip.add_notes(notes)
        
        # Quantize to 16th notes (0.25 grid)
        self.clip.quantize(0.25)
        
        # Check quantized times
        expected_times = [0.0, 1.0, 2.0]  # Snapped to grid
        actual_times = [note.start_time for note in self.clip.notes]
        self.assertEqual(actual_times, expected_times)
    
    def test_tidal_pattern_export(self):
        """Test TidalCycles pattern generation"""
        # Add simple notes
        notes = [
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.25),
            MIDINote(pitch=64, velocity=90, start_time=0.5, duration=0.25),
            MIDINote(pitch=67, velocity=95, start_time=1.0, duration=0.25)
        ]
        self.clip.add_notes(notes)
        
        pattern = self.clip.to_tidal_pattern()
        
        # Should contain superpiano and note frequencies
        self.assertIn("superpiano", pattern)
        self.assertIn("note", pattern)
        self.assertIn("~", pattern)  # Should have rests
    
    def test_midi_file_export(self):
        """Test MIDI file generation"""
        # Skip if mido not available
        try:
            import mido
        except ImportError:
            self.skipTest("mido not available")
        
        # Add test notes
        notes = [
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0),
            MIDINote(pitch=64, velocity=90, start_time=1.0, duration=1.0),
            MIDINote(pitch=67, velocity=95, start_time=2.0, duration=1.0)
        ]
        self.clip.add_notes(notes)
        
        # Generate MIDI file
        midi_file = self.clip.to_midi_file()
        
        # Check basic structure
        self.assertEqual(len(midi_file.tracks), 1)
        self.assertGreater(len(midi_file.tracks[0]), 0)
        
        # Test save to file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            success = self.clip.save_midi_file(tmp.name)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(tmp.name))
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_serialization(self):
        """Test clip serialization and deserialization"""
        # Set up complex clip
        self.clip.tags = ["acid", "hardcore"]
        self.clip.genre = "gabber"
        notes = [
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.5),
            MIDINote(pitch=64, velocity=90, start_time=1.0, duration=0.25)
        ]
        self.clip.add_notes(notes)
        
        # Serialize
        clip_dict = self.clip.to_dict()
        
        # Check structure
        expected_keys = {
            'name', 'length_bars', 'time_signature', 'key_signature', 'bpm', 
            'notes', 'created_at', 'tags', 'genre'
        }
        self.assertEqual(set(clip_dict.keys()), expected_keys)
        self.assertEqual(len(clip_dict['notes']), 2)
        
        # Deserialize
        reconstructed = MIDIClip.from_dict(clip_dict)
        
        # Check reconstruction
        self.assertEqual(reconstructed.name, self.clip.name)
        self.assertEqual(reconstructed.bpm, self.clip.bpm)
        self.assertEqual(reconstructed.tags, self.clip.tags)
        self.assertEqual(reconstructed.genre, self.clip.genre)
        self.assertEqual(len(reconstructed.notes), 2)
        
        # Check note reconstruction
        for orig, recon in zip(self.clip.notes, reconstructed.notes):
            self.assertEqual(orig.pitch, recon.pitch)
            self.assertEqual(orig.velocity, recon.velocity)
            self.assertEqual(orig.start_time, recon.start_time)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Empty clip TidalCycles export
        empty_clip = MIDIClip()
        pattern = empty_clip.to_tidal_pattern()
        self.assertEqual(pattern, "silence")
        
        # Very long clip
        long_clip = MIDIClip(length_bars=1000.0)
        self.assertEqual(long_clip.get_total_beats(), 4000.0)
        
        # Invalid time signature doesn't break things
        try:
            clip_with_custom_ts = MIDIClip(time_signature=TimeSignature(0, 4))
            # Should not crash, but might have unexpected behavior
        except:
            pass  # OK if it fails gracefully


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_note_name_conversion(self):
        """Test note name to MIDI number conversion"""
        # Test basic notes
        self.assertEqual(note_name_to_midi("C4"), 60)  # Middle C
        self.assertEqual(note_name_to_midi("A4"), 69)  # A440
        self.assertEqual(note_name_to_midi("C1"), 24)  # Low C
        
        # Test sharps and flats
        self.assertEqual(note_name_to_midi("C#4"), 61)
        self.assertEqual(note_name_to_midi("Db4"), 61)  # Enharmonic equivalent
        
        # Test extreme ranges
        self.assertEqual(note_name_to_midi("C0"), 12)
        self.assertEqual(note_name_to_midi("G9"), 127)
        
        # Test invalid notes
        with self.assertRaises(ValueError):
            note_name_to_midi("H4")  # Invalid note
        
        with self.assertRaises(ValueError):
            note_name_to_midi("C")  # No octave
    
    def test_midi_to_note_name(self):
        """Test MIDI number to note name conversion"""
        self.assertEqual(midi_to_note_name(60), "C4")
        self.assertEqual(midi_to_note_name(69), "A4")
        self.assertEqual(midi_to_note_name(61), "C#4")
        
        # Test extreme ranges
        self.assertEqual(midi_to_note_name(0), "C-1")
        self.assertEqual(midi_to_note_name(127), "G9")
    
    def test_create_empty_midi_clip(self):
        """Test empty clip creation utility"""
        clip = create_empty_midi_clip("test", bars=8.0, bpm=140.0)
        
        self.assertEqual(clip.name, "test")
        self.assertEqual(clip.length_bars, 8.0)
        self.assertEqual(clip.bpm, 140.0)
        self.assertEqual(len(clip.notes), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main()