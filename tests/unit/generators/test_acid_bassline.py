#!/usr/bin/env python3
"""
Unit tests for AcidBasslineGenerator

Comprehensive testing of acid bassline generation including:
- Scale handling
- Pattern generation
- Musical quality validation
- Parameter variations
- Edge cases
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cli_shared.generators.acid_bassline import (
    AcidBasslineGenerator, AcidScale, AcidSettings,
    create_classic_acid_line, create_hardcore_acid_line, create_minimal_acid_line,
    SCALE_INTERVALS, SCALE_ROOTS
)
from cli_shared.models.midi_clips import MIDIClip, note_name_to_midi


class TestAcidScale(unittest.TestCase):
    """Test acid scale definitions"""
    
    def test_scale_completeness(self):
        """Test that all scales have required data"""
        for scale in AcidScale:
            self.assertIn(scale, SCALE_INTERVALS, f"Missing intervals for {scale}")
            self.assertIn(scale, SCALE_ROOTS, f"Missing root for {scale}")
    
    def test_scale_intervals(self):
        """Test scale interval validity"""
        for scale, intervals in SCALE_INTERVALS.items():
            # Should start with 0 (root)
            self.assertEqual(intervals[0], 0, f"Scale {scale} should start with 0")
            
            # Should be sorted
            self.assertEqual(intervals, sorted(intervals), f"Scale {scale} intervals should be sorted")
            
            # Should be within octave
            for interval in intervals:
                self.assertGreaterEqual(interval, 0, f"Scale {scale} has negative interval")
                self.assertLess(interval, 12, f"Scale {scale} has interval >= 12")
    
    def test_scale_roots(self):
        """Test scale root note validity"""
        for scale, root in SCALE_ROOTS.items():
            # Should be valid MIDI note
            self.assertGreaterEqual(root, 0, f"Scale {scale} root too low")
            self.assertLessEqual(root, 127, f"Scale {scale} root too high")


class TestAcidSettings(unittest.TestCase):
    """Test AcidSettings configuration"""
    
    def test_default_settings(self):
        """Test default settings are reasonable"""
        settings = AcidSettings()
        
        # Check probability ranges
        self.assertGreaterEqual(settings.note_density, 0.0)
        self.assertLessEqual(settings.note_density, 1.0)
        
        self.assertGreaterEqual(settings.accent_probability, 0.0)
        self.assertLessEqual(settings.accent_probability, 1.0)
        
        # Check velocity ranges
        self.assertGreaterEqual(settings.normal_velocity, 1)
        self.assertLessEqual(settings.normal_velocity, 127)
        
        self.assertGreaterEqual(settings.accent_velocity, 1)
        self.assertLessEqual(settings.accent_velocity, 127)
        
        # Check octave range makes sense
        min_oct, max_oct = settings.octave_range
        self.assertLess(min_oct, max_oct)
        self.assertGreaterEqual(min_oct, 0)
        self.assertLessEqual(max_oct, 9)
    
    def test_custom_settings(self):
        """Test custom settings creation"""
        settings = AcidSettings(
            scale=AcidScale.E_MINOR,
            octave_range=(3, 5),
            note_density=0.9,
            accent_velocity=127
        )
        
        self.assertEqual(settings.scale, AcidScale.E_MINOR)
        self.assertEqual(settings.octave_range, (3, 5))
        self.assertEqual(settings.note_density, 0.9)
        self.assertEqual(settings.accent_velocity, 127)


class TestAcidBasslineGenerator(unittest.TestCase):
    """Test AcidBasslineGenerator class"""
    
    def setUp(self):
        """Set up test generator"""
        self.generator = AcidBasslineGenerator(
            scale="A_minor",
            accent_pattern="x ~ ~ x ~ ~ x ~"
        )
    
    def test_generator_creation(self):
        """Test generator initialization"""
        self.assertEqual(self.generator.scale, AcidScale.A_MINOR)
        self.assertEqual(self.generator.accent_pattern, "x~~x~~x~")  # Spaces removed
        self.assertIsInstance(self.generator.settings, AcidSettings)
        self.assertGreater(len(self.generator.scale_notes), 0)
    
    def test_scale_note_building(self):
        """Test scale note generation"""
        # Should have notes across octave range
        min_octave, max_octave = self.generator.settings.octave_range
        expected_octaves = max_octave - min_octave + 1
        scale_length = len(SCALE_INTERVALS[AcidScale.A_MINOR])
        
        # Should have roughly scale_length * octaves notes
        expected_min_notes = scale_length * expected_octaves
        self.assertGreaterEqual(len(self.generator.scale_notes), expected_min_notes - 2)
        
        # All notes should be in valid MIDI range
        for note in self.generator.scale_notes:
            self.assertGreaterEqual(note, 0)
            self.assertLessEqual(note, 127)
        
        # Should be sorted
        self.assertEqual(self.generator.scale_notes, sorted(self.generator.scale_notes))
    
    def test_accent_pattern_detection(self):
        """Test accent pattern parsing"""
        # Test simple pattern
        gen1 = AcidBasslineGenerator(accent_pattern="x ~ x ~")
        self.assertTrue(gen1._get_accent_at_step(0))   # x
        self.assertFalse(gen1._get_accent_at_step(1))  # ~
        self.assertTrue(gen1._get_accent_at_step(2))   # x
        self.assertFalse(gen1._get_accent_at_step(3))  # ~
        
        # Test pattern wrapping
        self.assertTrue(gen1._get_accent_at_step(4))   # Wraps to position 0
        
        # Test empty pattern
        gen2 = AcidBasslineGenerator(accent_pattern="")
        self.assertFalse(gen2._get_accent_at_step(0))
    
    def test_note_choice_logic(self):
        """Test note selection logic"""
        # Test with no current note
        note1 = self.generator._choose_next_note(None, 0)
        self.assertIn(note1, self.generator.scale_notes)
        
        # Test with current note
        current = self.generator.scale_notes[3]  # Pick a note in the middle
        note2 = self.generator._choose_next_note(current, 1)
        
        if note2 is not None:  # Might be None for rest
            self.assertIn(note2, self.generator.scale_notes)
        
        # Test edge case - note not in scale
        invalid_note = 50  # Probably not in A minor scale at low octaves
        note3 = self.generator._choose_next_note(invalid_note, 2)
        if note3 is not None:
            self.assertIn(note3, self.generator.scale_notes)
    
    def test_velocity_generation(self):
        """Test velocity assignment"""
        # Test accent velocity
        accent_vel = self.generator._get_note_velocity(0, True)
        self.assertGreaterEqual(accent_vel, 1)
        self.assertLessEqual(accent_vel, 127)
        # Should be higher than normal
        self.assertGreaterEqual(accent_vel, self.generator.settings.normal_velocity - 20)
        
        # Test normal velocity
        normal_vel = self.generator._get_note_velocity(1, False)
        self.assertGreaterEqual(normal_vel, 1)
        self.assertLessEqual(normal_vel, 127)
    
    def test_duration_generation(self):
        """Test note duration assignment"""
        # Test accent duration
        accent_dur = self.generator._get_note_duration(0, True)
        self.assertGreater(accent_dur, 0.0)
        self.assertLessEqual(accent_dur, 2.0)  # Reasonable upper bound
        
        # Test normal duration
        normal_dur = self.generator._get_note_duration(1, False)
        self.assertGreater(normal_dur, 0.0)
        self.assertLessEqual(normal_dur, 2.0)
    
    def test_basic_generation(self):
        """Test basic clip generation"""
        clip = self.generator.generate(length_bars=2.0, bpm=140.0)
        
        # Check clip properties
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 2.0)
        self.assertEqual(clip.bpm, 140.0)
        self.assertIn("acid", clip.tags)
        self.assertEqual(clip.key_signature, "A_minor")
        
        # Should have generated some notes
        self.assertGreater(len(clip.notes), 0)
        
        # All notes should be valid
        for note in clip.notes:
            self.assertGreaterEqual(note.pitch, 0)
            self.assertLessEqual(note.pitch, 127)
            self.assertGreaterEqual(note.velocity, 1)
            self.assertLessEqual(note.velocity, 127)
            self.assertGreaterEqual(note.start_time, 0.0)
            self.assertLess(note.start_time, 8.0)  # Within 2 bars
            self.assertGreater(note.duration, 0.0)
    
    def test_different_scales(self):
        """Test generation with different scales"""
        scales_to_test = ["A_minor", "E_minor", "A_harmonic_minor", "E_phrygian"]
        
        for scale_name in scales_to_test:
            with self.subTest(scale=scale_name):
                gen = AcidBasslineGenerator(scale=scale_name)
                clip = gen.generate(length_bars=1.0, bpm=180.0)
                
                self.assertIsInstance(clip, MIDIClip)
                self.assertGreater(len(clip.notes), 0)
                self.assertEqual(clip.key_signature, scale_name)
                
                # Notes should be from the correct scale
                scale_enum = AcidScale(scale_name.upper())
                root = SCALE_ROOTS[scale_enum]
                intervals = SCALE_INTERVALS[scale_enum]
                
                # Build expected pitches
                expected_pitches = set()
                for octave in gen.settings.octave_range:
                    octave_offset = (octave - 2) * 12
                    for interval in intervals:
                        pitch = root + octave_offset + interval
                        if 0 <= pitch <= 127:
                            expected_pitches.add(pitch)
                
                # All generated notes should be in scale
                for note in clip.notes:
                    self.assertIn(note.pitch, expected_pitches, 
                                f"Note {note.pitch} not in {scale_name} scale")
    
    def test_different_bpms(self):
        """Test generation at different BPMs"""
        bpms = [120, 140, 180, 200, 250]
        
        for bpm in bpms:
            with self.subTest(bpm=bpm):
                clip = self.generator.generate(length_bars=1.0, bpm=bpm)
                
                self.assertEqual(clip.bpm, bpm)
                self.assertGreater(len(clip.notes), 0)
                
                # All notes should be within time bounds
                max_time = 4.0  # 1 bar = 4 beats
                for note in clip.notes:
                    self.assertLess(note.start_time, max_time)
    
    def test_pattern_variations(self):
        """Test different accent patterns"""
        patterns = [
            "x ~ ~ ~",
            "x ~ x ~ x ~ x ~", 
            "x x ~ x ~ x ~ x",
            "X ~ x ~ X ~ x ~",  # Mixed case should work
        ]
        
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                gen = AcidBasslineGenerator(accent_pattern=pattern)
                clip = gen.generate(length_bars=2.0, bpm=160.0)
                
                self.assertIsInstance(clip, MIDIClip)
                self.assertGreater(len(clip.notes), 0)
    
    def test_variation_generation(self):
        """Test generating variations of a base clip"""
        base_clip = self.generator.generate(length_bars=4.0, bpm=175.0)
        variations = self.generator.generate_variations(base_clip, num_variations=3)
        
        self.assertEqual(len(variations), 3)
        
        for i, variation in enumerate(variations):
            self.assertIsInstance(variation, MIDIClip)
            self.assertEqual(variation.length_bars, base_clip.length_bars)
            self.assertEqual(variation.bpm, base_clip.bpm)
            self.assertIn(f"var_{i+1}", variation.name)
            
            # Should be different from base
            base_pitches = [note.pitch for note in base_clip.notes]
            var_pitches = [note.pitch for note in variation.notes]
            # Allow some similarity but expect some difference
            self.assertNotEqual(base_pitches, var_pitches)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very short clip
        short_clip = self.generator.generate(length_bars=0.25, bpm=180.0)
        self.assertIsInstance(short_clip, MIDIClip)
        
        # Very long clip
        long_clip = self.generator.generate(length_bars=16.0, bpm=180.0)
        self.assertIsInstance(long_clip, MIDIClip)
        self.assertGreater(len(long_clip.notes), len(short_clip.notes))
        
        # Extreme BPMs
        slow_clip = self.generator.generate(length_bars=1.0, bpm=60.0)
        fast_clip = self.generator.generate(length_bars=1.0, bpm=300.0)
        self.assertIsInstance(slow_clip, MIDIClip)
        self.assertIsInstance(fast_clip, MIDIClip)
        
        # Empty accent pattern
        gen_no_pattern = AcidBasslineGenerator(accent_pattern="")
        clip_no_pattern = gen_no_pattern.generate(length_bars=1.0, bpm=180.0)
        self.assertIsInstance(clip_no_pattern, MIDIClip)
    
    def test_settings_variations(self):
        """Test different settings configurations"""
        # High density
        high_density_settings = AcidSettings(note_density=0.95)
        gen_dense = AcidBasslineGenerator(settings=high_density_settings)
        dense_clip = gen_dense.generate(length_bars=2.0, bpm=180.0)
        
        # Low density
        low_density_settings = AcidSettings(note_density=0.3)
        gen_sparse = AcidBasslineGenerator(settings=low_density_settings)
        sparse_clip = gen_sparse.generate(length_bars=2.0, bpm=180.0)
        
        # Dense clip should have more notes
        self.assertGreater(len(dense_clip.notes), len(sparse_clip.notes))
        
        # High accent probability
        high_accent_settings = AcidSettings(accent_probability=0.8)
        gen_accents = AcidBasslineGenerator(settings=high_accent_settings)
        accent_clip = gen_accents.generate(length_bars=2.0, bpm=180.0)
        self.assertIsInstance(accent_clip, MIDIClip)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for quick generation"""
    
    def test_classic_acid_line(self):
        """Test classic acid house style generation"""
        clip = create_classic_acid_line(length_bars=4.0, bpm=140.0)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 4.0)
        self.assertEqual(clip.bpm, 140.0)
        self.assertGreater(len(clip.notes), 0)
        self.assertIn("acid", clip.tags)
    
    def test_hardcore_acid_line(self):
        """Test hardcore acid style generation"""
        clip = create_hardcore_acid_line(length_bars=2.0, bpm=190.0)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 2.0)
        self.assertEqual(clip.bpm, 190.0)
        self.assertGreater(len(clip.notes), 0)
        
        # Should be more aggressive (more notes, higher velocities)
        velocities = [note.velocity for note in clip.notes]
        avg_velocity = sum(velocities) / len(velocities)
        self.assertGreater(avg_velocity, 100)  # Should be quite loud
    
    def test_minimal_acid_line(self):
        """Test minimal acid style generation"""
        clip = create_minimal_acid_line(length_bars=8.0, bpm=130.0)
        
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 8.0)
        self.assertEqual(clip.bpm, 130.0)
        self.assertGreater(len(clip.notes), 0)
        
        # Should be sparser than other styles
        # (Hard to test directly, but should not crash)
    
    def test_style_differences(self):
        """Test that different styles produce different results"""
        classic = create_classic_acid_line()
        hardcore = create_hardcore_acid_line()
        minimal = create_minimal_acid_line()
        
        # All should be valid
        for clip in [classic, hardcore, minimal]:
            self.assertIsInstance(clip, MIDIClip)
            self.assertGreater(len(clip.notes), 0)
        
        # They should have different characteristics
        # (Specific differences are hard to test deterministically,
        # but they should at least be different lengths or note counts)
        classic_notes = len(classic.notes)
        hardcore_notes = len(hardcore.notes)
        minimal_notes = len(minimal.notes)
        
        # At least one should be different from the others
        all_same = (classic_notes == hardcore_notes == minimal_notes)
        self.assertFalse(all_same, "All styles produced identical note counts")


if __name__ == '__main__':
    unittest.main()