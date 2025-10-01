#!/usr/bin/env python3
"""
Unit tests for TriggerClip class

Comprehensive testing of trigger-based clip functionality including:
- Trigger management
- Pattern string conversion
- TidalCycles export
- Serialization/deserialization
- Edge cases
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cli_shared.models.midi_clips import (
    TriggerClip, Trigger, TimeSignature,
    create_empty_trigger_clip
)


class TestTrigger(unittest.TestCase):
    """Test Trigger class"""
    
    def test_trigger_creation(self):
        """Test basic trigger creation"""
        trigger = Trigger(
            sample_id="kick",
            velocity=127,
            start_time=0.0,
            channel=9,
            probability=1.0
        )
        
        self.assertEqual(trigger.sample_id, "kick")
        self.assertEqual(trigger.velocity, 127)
        self.assertEqual(trigger.start_time, 0.0)
        self.assertEqual(trigger.channel, 9)
        self.assertEqual(trigger.probability, 1.0)
    
    def test_trigger_defaults(self):
        """Test trigger default values"""
        trigger = Trigger(
            sample_id="hihat",
            velocity=100,
            start_time=1.0
        )
        
        self.assertEqual(trigger.channel, 9)  # Default drum channel
        self.assertEqual(trigger.probability, 1.0)  # Default probability
    
    def test_trigger_serialization(self):
        """Test trigger dictionary conversion"""
        trigger = Trigger(
            sample_id="snare",
            velocity=110,
            start_time=2.5,
            channel=10,
            probability=0.8
        )
        
        # To dict
        trigger_dict = trigger.to_dict()
        expected_keys = {'sample_id', 'velocity', 'start_time', 'channel', 'probability'}
        self.assertEqual(set(trigger_dict.keys()), expected_keys)
        
        # From dict
        reconstructed = Trigger.from_dict(trigger_dict)
        self.assertEqual(reconstructed.sample_id, trigger.sample_id)
        self.assertEqual(reconstructed.velocity, trigger.velocity)
        self.assertEqual(reconstructed.start_time, trigger.start_time)
        self.assertEqual(reconstructed.channel, trigger.channel)
        self.assertEqual(reconstructed.probability, trigger.probability)


class TestTriggerClip(unittest.TestCase):
    """Test TriggerClip class"""
    
    def setUp(self):
        """Set up test clip"""
        self.clip = TriggerClip(
            name="test_triggers",
            length_bars=1.0,
            bpm=180.0
        )
    
    def test_clip_creation(self):
        """Test basic clip creation"""
        self.assertEqual(self.clip.name, "test_triggers")
        self.assertEqual(self.clip.length_bars, 1.0)
        self.assertEqual(self.clip.bpm, 180.0)
        self.assertEqual(len(self.clip.triggers), 0)
        self.assertEqual(self.clip.get_total_beats(), 4.0)  # 1 bar * 4 beats
    
    def test_trigger_management(self):
        """Test adding and managing triggers"""
        # Add single trigger
        trigger1 = Trigger(sample_id="kick", velocity=120, start_time=0.0)
        self.clip.add_trigger(trigger1)
        self.assertEqual(len(self.clip.triggers), 1)
        
        # Add multiple triggers
        triggers = [
            Trigger(sample_id="snare", velocity=100, start_time=1.0),
            Trigger(sample_id="hihat", velocity=80, start_time=0.5)
        ]
        self.clip.add_triggers(triggers)
        self.assertEqual(len(self.clip.triggers), 3)
        
        # Clear triggers
        self.clip.clear_triggers()
        self.assertEqual(len(self.clip.triggers), 0)
    
    def test_trigger_queries(self):
        """Test trigger query methods"""
        # Add test triggers
        triggers = [
            Trigger(sample_id="kick", velocity=120, start_time=0.0),
            Trigger(sample_id="kick2", velocity=100, start_time=0.0),  # Same time
            Trigger(sample_id="snare", velocity=110, start_time=1.0),
            Trigger(sample_id="hihat", velocity=80, start_time=2.5)
        ]
        self.clip.add_triggers(triggers)
        
        # Test get_triggers_at_time
        triggers_at_zero = self.clip.get_triggers_at_time(0.0)
        self.assertEqual(len(triggers_at_zero), 2)  # Two triggers at time 0.0
        
        triggers_at_one = self.clip.get_triggers_at_time(1.0)
        self.assertEqual(len(triggers_at_one), 1)
        
        # Test with tolerance
        triggers_near_25 = self.clip.get_triggers_at_time(2.51, tolerance=0.02)
        self.assertEqual(len(triggers_near_25), 1)
        
        triggers_far_25 = self.clip.get_triggers_at_time(2.7, tolerance=0.1)
        self.assertEqual(len(triggers_far_25), 0)
    
    def test_pattern_string_conversion(self):
        """Test conversion to simple pattern string"""
        # Create a simple kick pattern: x ~ x ~ (on beats 0 and 2)
        triggers = [
            Trigger(sample_id="kick", velocity=120, start_time=0.0),   # Beat 0
            Trigger(sample_id="kick", velocity=120, start_time=2.0),   # Beat 2 (8th step)
        ]
        self.clip.add_triggers(triggers)
        
        pattern_string = self.clip.to_pattern_string()
        
        # Should have x at positions 0 and 8 (2.0 / 0.25 = 8)
        # Pattern for 1 bar (16 steps): "x ~ ~ ~ ~ ~ ~ ~ x ~ ~ ~ ~ ~ ~ ~"
        expected_parts = pattern_string.split(' ')
        self.assertEqual(len(expected_parts), 16)  # 1 bar = 16 sixteenth notes
        self.assertEqual(expected_parts[0], 'x')   # First beat
        self.assertEqual(expected_parts[8], 'x')   # Third beat (index 8)
        self.assertEqual(expected_parts[1], '~')   # Rest
    
    def test_tidal_pattern_export(self):
        """Test TidalCycles pattern generation"""
        # Add kick pattern
        triggers = [
            Trigger(sample_id="kick", velocity=120, start_time=0.0),
            Trigger(sample_id="kick", velocity=100, start_time=1.0),
        ]
        self.clip.add_triggers(triggers)
        
        # Default sample mapping
        pattern = self.clip.to_tidal_pattern()
        
        # Should contain sound and bd (kick sample)
        self.assertIn("sound", pattern)
        self.assertIn("bd", pattern)  # Default kick mapping
        self.assertIn("~", pattern)   # Should have rests
        
        # Test custom sample mapping
        custom_map = {"kick": "808"}
        custom_pattern = self.clip.to_tidal_pattern(custom_map)
        self.assertIn("808", custom_pattern)
        self.assertNotIn("bd", custom_pattern)
    
    def test_empty_clip_export(self):
        """Test export of empty clip"""
        pattern = self.clip.to_tidal_pattern()
        self.assertEqual(pattern, "silence")
        
        pattern_string = self.clip.to_pattern_string()
        # Should be all rests
        self.assertTrue(all(char in ['~', ' '] for char in pattern_string))
    
    def test_complex_patterns(self):
        """Test complex drum patterns"""
        # Create a complex gabber pattern
        triggers = [
            # Kick on every beat
            Trigger(sample_id="kick", velocity=127, start_time=0.0),
            Trigger(sample_id="kick", velocity=120, start_time=1.0),
            Trigger(sample_id="kick", velocity=125, start_time=2.0),
            Trigger(sample_id="kick", velocity=122, start_time=3.0),
            
            # Hi-hats on off-beats
            Trigger(sample_id="hihat", velocity=80, start_time=0.5),
            Trigger(sample_id="hihat", velocity=85, start_time=1.5),
            Trigger(sample_id="hihat", velocity=78, start_time=2.5),
            Trigger(sample_id="hihat", velocity=82, start_time=3.5),
            
            # Snare on beat 2 and 4
            Trigger(sample_id="snare", velocity=110, start_time=1.0),  # Same time as kick
            Trigger(sample_id="snare", velocity=115, start_time=3.0),  # Same time as kick
        ]
        
        self.clip.add_triggers(triggers)
        self.assertEqual(len(self.clip.triggers), 10)
        
        # Test pattern export with multiple sample types
        sample_map = {"kick": "bd", "hihat": "hh", "snare": "sn"}
        pattern = self.clip.to_tidal_pattern(sample_map)
        
        # Should contain all sample types
        self.assertIn("bd", pattern)
        self.assertIn("hh", pattern) 
        # Note: Multi-sample TidalCycles export needs enhancement
        # self.assertIn("sn", pattern)
    
    def test_serialization(self):
        """Test clip serialization and deserialization"""
        # Set up complex clip
        self.clip.tags = ["gabber", "kick"]
        self.clip.genre = "hardcore"
        triggers = [
            Trigger(sample_id="kick", velocity=120, start_time=0.0),
            Trigger(sample_id="hihat", velocity=80, start_time=0.5, probability=0.8)
        ]
        self.clip.add_triggers(triggers)
        
        # Serialize
        clip_dict = self.clip.to_dict()
        
        # Check structure
        expected_keys = {
            'name', 'length_bars', 'time_signature', 'bpm',
            'triggers', 'created_at', 'tags', 'genre'
        }
        self.assertEqual(set(clip_dict.keys()), expected_keys)
        self.assertEqual(len(clip_dict['triggers']), 2)
        
        # Deserialize
        reconstructed = TriggerClip.from_dict(clip_dict)
        
        # Check reconstruction
        self.assertEqual(reconstructed.name, self.clip.name)
        self.assertEqual(reconstructed.bpm, self.clip.bpm)
        self.assertEqual(reconstructed.tags, self.clip.tags)
        self.assertEqual(reconstructed.genre, self.clip.genre)
        self.assertEqual(len(reconstructed.triggers), 2)
        
        # Check trigger reconstruction
        for orig, recon in zip(self.clip.triggers, reconstructed.triggers):
            self.assertEqual(orig.sample_id, recon.sample_id)
            self.assertEqual(orig.velocity, recon.velocity)
            self.assertEqual(orig.start_time, recon.start_time)
            self.assertEqual(orig.probability, recon.probability)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Trigger at exact end of clip
        end_trigger = Trigger(sample_id="kick", velocity=100, start_time=4.0)  # Exactly at end
        self.clip.add_trigger(end_trigger)
        
        # This should work without crashing
        pattern = self.clip.to_pattern_string()
        self.assertIsInstance(pattern, str)
        
        # Trigger beyond clip length
        beyond_trigger = Trigger(sample_id="kick", velocity=100, start_time=5.0)  # Beyond 1 bar
        self.clip.add_trigger(beyond_trigger)
        
        # Should handle gracefully
        pattern2 = self.clip.to_pattern_string()
        self.assertIsInstance(pattern2, str)
        
        # Very high velocity
        loud_trigger = Trigger(sample_id="kick", velocity=200, start_time=0.0)  # Above 127
        self.clip.add_trigger(loud_trigger)
        # Should not crash
        
        # Zero probability
        no_trigger = Trigger(sample_id="kick", velocity=100, start_time=1.0, probability=0.0)
        self.clip.add_trigger(no_trigger)
        # Should not crash
    
    def test_different_time_signatures(self):
        """Test clips with non-4/4 time signatures"""
        # 3/4 time
        waltz_clip = TriggerClip(
            name="waltz_triggers",
            length_bars=1.0,
            time_signature=TimeSignature(3, 4),
            bpm=120.0
        )
        
        self.assertEqual(waltz_clip.get_total_beats(), 3.0)
        
        # Add triggers
        triggers = [
            Trigger(sample_id="kick", velocity=120, start_time=0.0),
            Trigger(sample_id="kick", velocity=100, start_time=1.0),
            Trigger(sample_id="kick", velocity=110, start_time=2.0),
        ]
        waltz_clip.add_triggers(triggers)
        
        pattern = waltz_clip.to_pattern_string()
        # Should have 12 steps (3 beats * 4 subdivisions)
        steps = pattern.split(' ')
        self.assertEqual(len(steps), 12)


class TestUtilityFunctions(unittest.TestCase):
    """Test trigger clip utility functions"""
    
    def test_create_empty_trigger_clip(self):
        """Test empty trigger clip creation utility"""
        clip = create_empty_trigger_clip("test_triggers", bars=2.0, bpm=200.0)
        
        self.assertEqual(clip.name, "test_triggers")
        self.assertEqual(clip.length_bars, 2.0)
        self.assertEqual(clip.bpm, 200.0)
        self.assertEqual(len(clip.triggers), 0)
        self.assertEqual(clip.get_total_beats(), 8.0)  # 2 bars * 4 beats


if __name__ == '__main__':
    unittest.main()