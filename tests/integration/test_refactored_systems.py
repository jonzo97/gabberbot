#!/usr/bin/env python3
"""
Integration Tests for Refactored Intelligent Music Systems

Tests the integration between:
- Musical Context System
- Modulation System  
- FM Synthesis Engine
- Track Architecture
- AI Agent
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli_shared.musical_context import MusicalContext, ScaleType, ChordType
from audio.modulation.modulator import LFO, Envelope, ModulationRouter
from audio.synthesis.fm_engine import FMSynthEngineExtended, FMAlgorithm
from audio.core.track import Track, PatternControlSource, KickAudioSource
from cli_shared.models.hardcore_models import SynthParams
from cli_shared.ai.intelligent_music_agent_v2 import IntelligentMusicAgentV2


class TestRefactoredSystemsIntegration(unittest.TestCase):
    """Test integration between all refactored systems"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 44100
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_musical_context_note_integration(self):
        """Test musical context integrates with existing MIDI utilities"""
        context = MusicalContext(key="C", scale_type=ScaleType.NATURAL_MINOR)
        
        # Test note name conversion integration
        notes = context.get_scale_notes()
        self.assertIsInstance(notes, list)
        self.assertGreater(len(notes), 0)
        
        # Test chord generation
        chord = context.get_chord(ChordType.MINOR, root_note="C")
        self.assertIsInstance(chord.notes, list)
        self.assertEqual(len(chord.notes), 3)  # Triad
    
    def test_fm_engine_abstract_synthesizer(self):
        """Test FM engine properly implements AbstractSynthesizer"""
        engine = FMSynthEngineExtended()
        
        # Test AbstractSynthesizer methods are implemented
        self.assertTrue(hasattr(engine, 'start'))
        self.assertTrue(hasattr(engine, 'stop'))
        self.assertTrue(hasattr(engine, 'generate_audio'))
        self.assertTrue(hasattr(engine, 'render_pattern_step'))
        
        # Test async methods work
        import asyncio
        async def test_async():
            result = await engine.start()
            self.assertTrue(result)
            
            result = await engine.stop()
            self.assertTrue(result)
        
        asyncio.run(test_async())
        
        # Test render_pattern_step compatibility with Track system
        params = SynthParams()
        params.frequency = 220.0
        params.duration_ms = 100
        
        audio = engine.render_pattern_step(velocity=0.8, params=params)
        self.assertIsInstance(audio, np.ndarray)
    
    def test_modulation_system_integration(self):
        """Test modulation system integrates with track architecture"""
        # Create track with modulation
        track = Track("Modulated Kick")
        track.add_kick_pattern()
        
        # Add LFO modulation
        lfo = track.add_lfo_modulation("volume", frequency=1.0, amplitude=0.2)
        self.assertIsInstance(lfo, LFO)
        
        # Test modulation updates parameters
        track.update_modulation(0.0)  # At time 0
        initial_volume = track.volume
        
        track.update_modulation(0.25)  # Quarter second later
        modulated_volume = track.volume
        
        # Volume should be different due to LFO
        self.assertNotEqual(initial_volume, modulated_volume)
    
    def test_track_system_complete_integration(self):
        """Test complete track system with all new features"""
        track = Track("Complete Test Track")
        
        # Set up kick pattern
        track.add_kick_pattern("x ~ ~ ~ x ~ ~ ~")
        
        # Add modulated gabber effects
        track.add_modulated_gabber_effects()
        
        # Test rendering
        params = SynthParams()
        params.frequency = 60.0
        
        # Render several steps to test modulation over time
        audio_outputs = []
        for step in range(8):
            audio = track.render_step(step, bpm=180.0, params=params)
            audio_outputs.append(audio)
        
        # Should have audio on steps 0 and 4 (kick pattern)
        self.assertGreater(len(audio_outputs[0]), 0)  # Step 0: kick
        self.assertEqual(len(audio_outputs[1]), 0)    # Step 1: silence
        self.assertGreater(len(audio_outputs[4]), 0)  # Step 4: kick
    
    def test_fm_engine_presets_integration(self):
        """Test FM engine presets work with track system"""
        engine = FMSynthEngineExtended()
        
        # Load hardcore preset
        engine.load_preset("classic_hoover")
        
        # Test note generation
        engine.note_on(220.0, 0.8)
        audio = engine.render_samples(1024)
        engine.note_off()
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(len(audio), 1024)
        self.assertGreater(np.max(np.abs(audio)), 0.0)  # Should have signal
    
    def test_musical_context_chord_progressions(self):
        """Test musical context chord progression generation"""
        context = MusicalContext(key="A", scale_type=ScaleType.NATURAL_MINOR)
        
        # Get dark hardcore progression
        progression = context.get_chord_progression("dark_hardcore")
        self.assertIsInstance(progression, list)
        self.assertGreater(len(progression), 0)
        
        # Each chord should have notes
        for chord in progression:
            self.assertIsInstance(chord.notes, list)
            self.assertGreater(len(chord.notes), 0)
    
    def test_modulation_router_connections(self):
        """Test modulation router handles multiple connections"""
        router = ModulationRouter()
        
        # Add multiple modulators
        lfo1 = LFO(frequency=1.0, amplitude=0.5)
        lfo2 = LFO(frequency=2.0, amplitude=0.3)
        env = Envelope()
        
        router.add_connection(lfo1, "cutoff", 1.0)
        router.add_connection(lfo2, "resonance", 0.8)
        router.add_connection(env, "volume", 1.0)
        
        # Process modulation
        values = router.process_modulation(1.0)
        
        self.assertIn("cutoff", values)
        self.assertIn("resonance", values)
        self.assertIn("volume", values)
    
    def test_ai_agent_database_integration(self):
        """Test AI agent database functionality"""
        db_path = os.path.join(self.temp_dir, "test_agent.db")
        agent = IntelligentMusicAgentV2(
            model_preferences=["claude-3-sonnet"],
            db_path=db_path
        )
        
        # Test database initialization
        self.assertTrue(os.path.exists(db_path))
        
        # Test preference storage
        agent.learn_vocabulary_mapping("heavy kick", {"distortion": 3.0})
        
        # Test preference retrieval
        mapping = agent.get_vocabulary_mapping("heavy kick")
        self.assertIsNotNone(mapping)
        self.assertIn("distortion", mapping)
    
    def test_error_handling_graceful_degradation(self):
        """Test systems handle errors gracefully"""
        # Test track with no audio source
        track = Track("Empty Track")
        audio = track.render_step(0, 180.0, SynthParams())
        self.assertEqual(len(audio), 0)
        
        # Test FM engine with invalid parameters
        engine = FMSynthEngineExtended()
        try:
            audio = engine.render_pattern_step(-1.0, SynthParams())  # Invalid velocity
            self.assertEqual(len(audio), 0)
        except Exception as e:
            self.fail(f"FM engine should handle invalid velocity gracefully: {e}")
        
        # Test modulation with invalid parameters
        track = Track("Invalid Modulation")
        try:
            track.add_lfo_modulation("nonexistent_param", 1.0, 1.0)
            # Should not crash, just ignore invalid parameter
        except Exception as e:
            self.fail(f"Modulation should handle invalid parameters gracefully: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance is acceptable"""
        import time
        
        # Benchmark track rendering
        track = Track("Performance Test")
        track.add_kick_pattern()
        track.add_modulated_gabber_effects()
        
        params = SynthParams()
        params.frequency = 60.0
        
        start_time = time.time()
        for step in range(100):  # Render 100 steps
            audio = track.render_step(step, 180.0, params)
        end_time = time.time()
        
        total_time = end_time - start_time
        self.assertLess(total_time, 5.0, "Track rendering should be fast enough for real-time")
        
        # Benchmark FM synthesis
        engine = FMSynthEngineExtended()
        engine.load_preset("classic_hoover")
        
        start_time = time.time()
        engine.note_on(220.0, 0.8)
        audio = engine.render_samples(44100)  # 1 second
        engine.note_off()
        end_time = time.time()
        
        synthesis_time = end_time - start_time
        self.assertLess(synthesis_time, 2.0, "FM synthesis should be efficient")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)