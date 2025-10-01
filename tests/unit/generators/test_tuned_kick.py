#!/usr/bin/env python3
"""
Unit tests for TunedKickGenerator

Comprehensive testing of tuned kick generation including:
- Tuning system validation
- Pattern interpretation
- Pitch sequence handling
- Musical quality validation
- Edge cases
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cli_shared.generators.tuned_kick import (
    TunedKickGenerator, KickTuning, TunedKickSettings,
    create_frenchcore_kicks, create_hardstyle_kicks, 
    create_industrial_kicks, create_octave_kicks,
    TUNING_INTERVALS
)
from cli_shared.models.midi_clips import MIDIClip, note_name_to_midi


class TestKickTuning(unittest.TestCase):
    """Test kick tuning system definitions"""
    
    def test_tuning_completeness(self):
        """Test that all tuning systems have interval data"""
        for tuning in KickTuning:
            self.assertIn(tuning, TUNING_INTERVALS, f"Missing intervals for {tuning}")
    
    def test_tuning_intervals(self):
        """Test tuning interval validity"""
        for tuning, intervals in TUNING_INTERVALS.items():
            # Should have at least one interval
            self.assertGreater(len(intervals), 0, f"Tuning {tuning} has no intervals")
            
            # Should start with 0 (root)
            self.assertEqual(intervals[0], 0, f"Tuning {tuning} should start with 0")
            
            # Should be sorted
            self.assertEqual(intervals, sorted(intervals), f"Tuning {tuning} intervals should be sorted")
            
            # Should be within octave
            for interval in intervals:
                self.assertGreaterEqual(interval, 0, f"Tuning {tuning} has negative interval")
                self.assertLess(interval, 12, f"Tuning {tuning} has interval >= 12")
    
    def test_specific_tunings(self):
        """Test specific tuning systems"""
        # Chromatic should have all 12 semitones
        self.assertEqual(len(TUNING_INTERVALS[KickTuning.CHROMATIC]), 12)
        self.assertEqual(TUNING_INTERVALS[KickTuning.CHROMATIC], list(range(12)))
        
        # Pentatonic should have 5 notes
        self.assertEqual(len(TUNING_INTERVALS[KickTuning.PENTATONIC]), 5)
        
        # Fifths should have root and fifth
        self.assertEqual(TUNING_INTERVALS[KickTuning.FIFTHS], [0, 7])
        
        # Octaves should have only root
        self.assertEqual(TUNING_INTERVALS[KickTuning.OCTAVES], [0])


class TestTunedKickSettings(unittest.TestCase):
    """Test TunedKickSettings configuration"""
    
    def test_default_settings(self):
        """Test default settings are reasonable"""
        settings = TunedKickSettings()
        
        # Check tuning
        self.assertIsInstance(settings.tuning, KickTuning)
        
        # Check octave range
        min_oct, max_oct = settings.octave_range
        self.assertLess(min_oct, max_oct)
        self.assertGreaterEqual(min_oct, 0)
        self.assertLessEqual(max_oct, 9)
        
        # Check probabilities
        self.assertGreaterEqual(settings.pitch_variation, 0.0)
        self.assertLessEqual(settings.pitch_variation, 1.0)
        
        # Check velocities
        self.assertGreaterEqual(settings.base_velocity, 1)
        self.assertLessEqual(settings.base_velocity, 127)
        
        self.assertGreaterEqual(settings.accent_velocity, 1)
        self.assertLessEqual(settings.accent_velocity, 127)
    
    def test_pattern_characters(self):
        """Test pattern character definitions"""
        settings = TunedKickSettings()
        
        # Should have different characters for different types
        chars = [
            settings.accent_char,
            settings.normal_char,
            settings.ghost_char,
            settings.rest_char
        ]
        
        # All should be single characters
        for char in chars:
            self.assertEqual(len(char), 1)
        
        # Should be unique
        self.assertEqual(len(set(chars)), len(chars))


class TestTunedKickGenerator(unittest.TestCase):
    """Test TunedKickGenerator class"""
    
    def setUp(self):
        """Set up test generator"""
        self.generator = TunedKickGenerator(
            root_note="C1",
            pattern="x ~ x ~ x ~ x ~",
            tuning="pentatonic"
        )
    
    def test_generator_creation(self):
        """Test generator initialization"""
        self.assertEqual(self.generator.root_note, note_name_to_midi("C1"))
        self.assertEqual(self.generator.pattern, "x~x~x~x~")  # Spaces removed
        self.assertEqual(self.generator.tuning, KickTuning.PENTATONIC)
        self.assertIsNone(self.generator.pitch_sequence)
        self.assertIsInstance(self.generator.settings, TunedKickSettings)
        self.assertGreater(len(self.generator.available_pitches), 0)
    
    def test_pitch_palette_building(self):
        """Test available pitch generation"""
        # Should have pitches across octave range
        min_octave, max_octave = self.generator.settings.octave_range
        
        # All pitches should be valid MIDI notes
        for pitch in self.generator.available_pitches:
            self.assertGreaterEqual(pitch, 0)
            self.assertLessEqual(pitch, 127)
        
        # Should be sorted
        self.assertEqual(self.generator.available_pitches, sorted(self.generator.available_pitches))
        
        # Should contain root note (or octave variations)
        root = self.generator.root_note
        root_class = root % 12
        pitch_classes = [p % 12 for p in self.generator.available_pitches]
        self.assertIn(root_class, pitch_classes)
    
    def test_pattern_character_parsing(self):
        """Test pattern character interpretation"""
        # Test basic pattern
        gen = TunedKickGenerator(pattern="x ~ X ~")
        
        self.assertEqual(gen._get_pattern_char_at_step(0), 'x')  # normal
        self.assertEqual(gen._get_pattern_char_at_step(1), '~')  # rest
        self.assertEqual(gen._get_pattern_char_at_step(2), 'X')  # accent
        self.assertEqual(gen._get_pattern_char_at_step(3), '~')  # rest
        
        # Test pattern wrapping
        self.assertEqual(gen._get_pattern_char_at_step(4), 'x')  # Wraps around
        
        # Test empty pattern
        gen_empty = TunedKickGenerator(pattern="")
        self.assertEqual(gen_empty._get_pattern_char_at_step(0), '~')  # Default rest
    
    def test_pitch_choice_logic(self):
        """Test pitch selection logic"""
        # Test with no previous pitch
        pitch1 = self.generator._choose_pitch_for_step(0, None)
        self.assertIn(pitch1, self.generator.available_pitches)
        
        # Test with previous pitch
        current = self.generator.available_pitches[2]
        pitch2 = self.generator._choose_pitch_for_step(1, current)
        self.assertIn(pitch2, self.generator.available_pitches)
        
        # Test with predetermined sequence
        gen_with_seq = TunedKickGenerator(
            root_note="C1",
            pattern="x x x x",
            pitch_sequence=[0, 3, 5, 0]  # Root, minor third, fifth, root
        )
        
        # Should follow the sequence
        root_midi = note_name_to_midi("C1")
        self.assertEqual(gen_with_seq._choose_pitch_for_step(0, None), root_midi)
        self.assertEqual(gen_with_seq._choose_pitch_for_step(1, None), root_midi + 3)
        self.assertEqual(gen_with_seq._choose_pitch_for_step(2, None), root_midi + 5)
        self.assertEqual(gen_with_seq._choose_pitch_for_step(3, None), root_midi)
        
        # Should wrap around
        self.assertEqual(gen_with_seq._choose_pitch_for_step(4, None), root_midi)
    
    def test_velocity_assignment(self):
        """Test velocity based on pattern character"""
        # Test accent
        accent_vel = self.generator._get_velocity_for_char('X', 0)
        self.assertGreaterEqual(accent_vel, 1)
        self.assertLessEqual(accent_vel, 127)
        self.assertGreaterEqual(accent_vel, self.generator.settings.base_velocity)
        
        # Test normal
        normal_vel = self.generator._get_velocity_for_char('x', 1)
        self.assertGreaterEqual(normal_vel, 1)
        self.assertLessEqual(normal_vel, 127)
        
        # Test ghost
        ghost_vel = self.generator._get_velocity_for_char('o', 2)
        self.assertGreaterEqual(ghost_vel, 1)
        self.assertLessEqual(ghost_vel, 127)
        self.assertLess(ghost_vel, self.generator.settings.base_velocity)
    
    def test_duration_assignment(self):
        """Test note duration based on pattern character"""
        # All durations should be positive and reasonable
        for char in ['X', 'x', 'o']:
            duration = self.generator._get_note_duration(char)
            self.assertGreater(duration, 0.0)
            self.assertLess(duration, 1.0)  # Should be fairly short
    
    def test_basic_generation(self):
        """Test basic kick pattern generation"""
        clip = self.generator.generate(length_bars=1.0, bpm=200.0)
        
        # Check clip properties
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 1.0)
        self.assertEqual(clip.bpm, 200.0)
        self.assertIn("tuned_kick", clip.tags)
        
        # Should have generated some notes
        self.assertGreater(len(clip.notes), 0)
        
        # All notes should be valid
        for note in clip.notes:
            self.assertGreaterEqual(note.pitch, 0)
            self.assertLessEqual(note.pitch, 127)
            self.assertGreaterEqual(note.velocity, 1)
            self.assertLessEqual(note.velocity, 127)
            self.assertGreaterEqual(note.start_time, 0.0)
            self.assertLess(note.start_time, 4.0)  # Within 1 bar
            self.assertEqual(note.channel, 9)  # Drum channel
            
            # Notes should be from available pitches
            self.assertIn(note.pitch, self.generator.available_pitches)
    
    def test_different_tunings(self):
        """Test generation with different tuning systems"""
        tunings_to_test = ["chromatic", "pentatonic", "minor", "fifths", "octaves"]
        
        for tuning_name in tunings_to_test:
            with self.subTest(tuning=tuning_name):
                gen = TunedKickGenerator(
                    root_note="C1",
                    pattern="x ~ x ~ x ~ x ~",
                    tuning=tuning_name
                )
                
                clip = gen.generate(length_bars=1.0, bpm=180.0)
                
                self.assertIsInstance(clip, MIDIClip)
                self.assertGreater(len(clip.notes), 0)
                
                # All notes should be from the correct tuning
                tuning_enum = KickTuning(tuning_name.upper())
                expected_intervals = TUNING_INTERVALS[tuning_enum]
                
                for note in clip.notes:
                    # Check if note matches expected intervals
                    relative_pitch = (note.pitch - gen.root_note) % 12
                    self.assertIn(relative_pitch, expected_intervals,
                                f"Note {note.pitch} not in {tuning_name} tuning")
    
    def test_different_root_notes(self):
        """Test generation with different root notes"""
        root_notes = ["C1", "D1", "E1", "F#1", "G1", "A1", "Bb1"]
        
        for root in root_notes:
            with self.subTest(root=root):
                gen = TunedKickGenerator(root_note=root, pattern="x ~ x ~")
                clip = gen.generate(length_bars=1.0, bpm=180.0)
                
                self.assertIsInstance(clip, MIDIClip)
                self.assertGreater(len(clip.notes), 0)
                
                # Check that root note is used correctly
                root_midi = note_name_to_midi(root)
                root_class = root_midi % 12
                
                # At least some notes should be in the same pitch class as root
                note_classes = [note.pitch % 12 for note in clip.notes]
                self.assertIn(root_class, note_classes)
    
    def test_pattern_interpretation(self):
        """Test different pattern styles"""
        patterns = [
            "x ~ x ~ x ~ x ~",  # Basic kick pattern
            "X ~ x ~ X ~ x ~",  # With accents
            "x x x x x x x x",  # Constant kicks
            "x ~ ~ ~ x ~ ~ ~",  # Sparse pattern
            "X o x o X o x o",  # With ghost notes
        ]
        
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                gen = TunedKickGenerator(pattern=pattern)
                clip = gen.generate(length_bars=1.0, bpm=180.0)
                
                self.assertIsInstance(clip, MIDIClip)
                
                # Count expected kicks (non-rest characters)
                expected_kicks = len([c for c in pattern.replace(" ", "") if c != '~'])
                
                # Should generate roughly the expected number (allowing for randomness)
                self.assertGreaterEqual(len(clip.notes), max(1, expected_kicks - 2))
                self.assertLessEqual(len(clip.notes), expected_kicks + 2)
    
    def test_melodic_sequence_creation(self):
        """Test melodic kick sequence generation"""
        melody = [0, 3, 5, 7, 3, 0]  # Simple melody in semitones
        rhythm = "x x x x x x"
        
        clip = self.generator.create_melodic_sequence(melody, rhythm)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertGreater(len(clip.notes), 0)
        self.assertIn("melodic_kicks", clip.name)
        
        # Should follow the melody
        root = self.generator.root_note
        expected_pitches = [root + semitones for semitones in melody]
        
        # Generate enough notes to cover the melody (with possible repeats)
        note_pitches = [note.pitch for note in clip.notes]
        
        # At least some notes should match the melody
        matches = sum(1 for pitch in note_pitches if pitch in expected_pitches)
        self.assertGreater(matches, 0)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very short pattern
        gen_short = TunedKickGenerator(pattern="x")
        short_clip = gen_short.generate(length_bars=0.25, bpm=180.0)
        self.assertIsInstance(short_clip, MIDIClip)
        
        # Very long pattern
        gen_long = TunedKickGenerator(pattern="x " * 32)
        long_clip = gen_long.generate(length_bars=4.0, bpm=180.0)
        self.assertIsInstance(long_clip, MIDIClip)
        
        # Pattern with only rests
        gen_rests = TunedKickGenerator(pattern="~ ~ ~ ~")
        rest_clip = gen_rests.generate(length_bars=1.0, bpm=180.0)
        self.assertIsInstance(rest_clip, MIDIClip)
        # Should have no notes or very few
        self.assertLessEqual(len(rest_clip.notes), 1)
        
        # Extreme BPMs
        slow_clip = self.generator.generate(length_bars=1.0, bpm=60.0)
        fast_clip = self.generator.generate(length_bars=1.0, bpm=300.0)
        self.assertIsInstance(slow_clip, MIDIClip)
        self.assertIsInstance(fast_clip, MIDIClip)
        
        # High octave root
        gen_high = TunedKickGenerator(root_note="C4")  # Much higher than typical
        high_clip = gen_high.generate(length_bars=1.0, bpm=180.0)
        self.assertIsInstance(high_clip, MIDIClip)
        
        # Low octave root
        gen_low = TunedKickGenerator(root_note="C0")
        low_clip = gen_low.generate(length_bars=1.0, bpm=180.0)
        self.assertIsInstance(low_clip, MIDIClip)
    
    def test_settings_variations(self):
        """Test different settings configurations"""
        # High pitch variation
        high_var_settings = TunedKickSettings(pitch_variation=0.9)
        gen_varied = TunedKickGenerator(settings=high_var_settings)
        varied_clip = gen_varied.generate(length_bars=2.0, bpm=180.0)
        
        # Low pitch variation (more repetitive)
        low_var_settings = TunedKickSettings(pitch_variation=0.1)
        gen_repetitive = TunedKickGenerator(settings=low_var_settings)
        repetitive_clip = gen_repetitive.generate(length_bars=2.0, bpm=180.0)
        
        # Both should be valid
        self.assertIsInstance(varied_clip, MIDIClip)
        self.assertIsInstance(repetitive_clip, MIDIClip)
        
        # Different octave ranges
        wide_range_settings = TunedKickSettings(octave_range=(0, 4))
        gen_wide = TunedKickGenerator(settings=wide_range_settings)
        wide_clip = gen_wide.generate(length_bars=1.0, bpm=180.0)
        
        narrow_range_settings = TunedKickSettings(octave_range=(1, 2))
        gen_narrow = TunedKickGenerator(settings=narrow_range_settings)
        narrow_clip = gen_narrow.generate(length_bars=1.0, bpm=180.0)
        
        # Wide range should potentially have more pitch variety
        if wide_clip.notes and narrow_clip.notes:
            wide_pitches = set(note.pitch for note in wide_clip.notes)
            narrow_pitches = set(note.pitch for note in narrow_clip.notes)
            
            # This is probabilistic, so we can't guarantee it, but usually true
            # self.assertGreaterEqual(len(wide_pitches), len(narrow_pitches))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for quick generation"""
    
    def test_frenchcore_kicks(self):
        """Test frenchcore kick generation"""
        clip = create_frenchcore_kicks(root_note="D1", length_bars=2.0, bpm=210.0)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 2.0)
        self.assertEqual(clip.bpm, 210.0)
        self.assertGreater(len(clip.notes), 0)
        
        # Should have complex pattern typical of frenchcore
        self.assertGreater(len(clip.notes), 8)  # Should be fairly dense
    
    def test_hardstyle_kicks(self):
        """Test hardstyle kick generation"""
        clip = create_hardstyle_kicks(root_note="C1", length_bars=1.0, bpm=150.0)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 1.0)
        self.assertEqual(clip.bpm, 150.0)
        self.assertGreater(len(clip.notes), 0)
        
        # Should follow a specific pitch pattern (root then fifth)
        if len(clip.notes) >= 2:
            root_midi = note_name_to_midi("C1")
            # Should contain root and fifth
            pitches = [note.pitch for note in clip.notes]
            pitch_classes = set((p - root_midi) % 12 for p in pitches)
            # Should contain root (0) and possibly fifth (7)
            self.assertIn(0, pitch_classes)  # Root
    
    def test_industrial_kicks(self):
        """Test industrial kick generation"""
        clip = create_industrial_kicks(root_note="C1", length_bars=4.0, bpm=140.0)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 4.0)
        self.assertEqual(clip.bmp, 140.0)
        self.assertGreater(len(clip.notes), 0)
        
        # Should use chromatic tuning (more experimental pitches)
        # Hard to test specifically, but should not crash
    
    def test_octave_kicks(self):
        """Test octave jumping kicks"""
        clip = create_octave_kicks(root_note="C1", length_bars=1.0, bpm=180.0)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 1.0)
        self.assertEqual(clip.bpm, 180.0)
        self.assertGreater(len(clip.notes), 0)
        
        # All kicks should be same pitch class (octaves only)
        if len(clip.notes) > 1:
            root_midi = note_name_to_midi("C1")
            root_class = root_midi % 12
            
            for note in clip.notes:
                self.assertEqual(note.pitch % 12, root_class,
                               "Octave kicks should all be same pitch class")
    
    def test_style_differences(self):
        """Test that different styles produce different results"""
        frenchcore = create_frenchcore_kicks()
        hardstyle = create_hardstyle_kicks()
        industrial = create_industrial_kicks()
        octave = create_octave_kicks()
        
        styles = [frenchcore, hardstyle, industrial, octave]
        
        # All should be valid
        for clip in styles:
            self.assertIsInstance(clip, MIDIClip)
            self.assertGreater(len(clip.notes), 0)
        
        # They should have some differences
        note_counts = [len(clip.notes) for clip in styles]
        pitch_counts = [len(set(note.pitch for note in clip.notes)) for clip in styles]
        
        # At least note counts or pitch variety should differ between styles
        all_same_notes = len(set(note_counts)) == 1
        all_same_pitches = len(set(pitch_counts)) == 1
        
        self.assertFalse(all_same_notes and all_same_pitches,
                        "All styles produced identical results")


if __name__ == '__main__':
    unittest.main()