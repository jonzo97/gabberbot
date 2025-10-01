#!/usr/bin/env python3
"""
Core Integration Test for Refactored Systems

Simple test to validate basic integration is working
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cli_shared.musical_context import MusicalContext, Scale, ScaleType
    from audio.modulation.modulator import LFO, WaveformType
    from audio.synthesis.fm_engine import FMSynthEngineExtended
    from audio.core.track import Track
    from cli_shared.models.hardcore_models import SynthParams
    from cli_shared.models.midi_clips import note_name_to_midi
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestCoreRefactoredSystems(unittest.TestCase):
    """Simple integration tests for core refactored systems"""
    
    def test_note_name_to_midi_works(self):
        """Test basic MIDI note conversion works"""
        c4 = note_name_to_midi("C4")
        self.assertEqual(c4, 60)
        
        a4 = note_name_to_midi("A4") 
        self.assertEqual(a4, 69)
    
    def test_musical_context_basic(self):
        """Test basic musical context creation"""
        # Create scale object first
        scale = Scale(root=60, scale_type=ScaleType.NATURAL_MINOR)  # C minor
        
        # Create musical context with scale object
        context = MusicalContext(key="Cm", scale=scale)
        
        self.assertEqual(context.key, "Cm")
        self.assertEqual(context.scale.scale_type, ScaleType.NATURAL_MINOR)
    
    def test_lfo_basic(self):
        """Test basic LFO functionality"""
        lfo = LFO(rate=1.0, waveform=WaveformType.SINE)
        
        # Test it has the expected attributes
        self.assertEqual(lfo.rate, 1.0)
        self.assertEqual(lfo.waveform, WaveformType.SINE)
    
    def test_fm_engine_basic(self):
        """Test basic FM engine functionality"""
        engine = FMSynthEngineExtended()
        
        # Test it inherits from AbstractSynthesizer
        from cli_shared.interfaces.synthesizer import AbstractSynthesizer
        self.assertIsInstance(engine, AbstractSynthesizer)
        
        # Test basic note generation doesn't crash
        engine.note_on(220.0, 0.8)
        try:
            audio = engine.generate_samples(100)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(len(audio), 100)
        finally:
            engine.note_off()
    
    def test_track_basic_creation(self):
        """Test basic track creation"""
        track = Track("Test Track")
        
        self.assertEqual(track.name, "Test Track")
        self.assertTrue(track.enabled)
        self.assertFalse(track.muted)
        
        # Test modulation router exists
        self.assertIsNotNone(track.modulation_router)
    
    def test_simple_integration(self):
        """Test simple integration without errors"""
        # Create a track
        track = Track("Integration Test")
        
        # Add kick pattern
        track.add_kick_pattern("x ~ ~ ~")
        
        # Test rendering doesn't crash  
        params = SynthParams()
        params.frequency = 60.0
        
        try:
            audio = track.render_step(0, 180.0, params)
            # Should have audio on step 0
            self.assertGreater(len(audio), 0)
            
            audio = track.render_step(1, 180.0, params)  
            # Should be silent on step 1
            self.assertEqual(len(audio), 0)
        except Exception as e:
            self.fail(f"Basic track rendering failed: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)