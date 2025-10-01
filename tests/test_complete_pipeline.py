#!/usr/bin/env python3
"""
Comprehensive Test Suite for Complete Text-to-Audio Pipeline.

Tests the entire BMAD Story 3 implementation from natural language
prompt to final WAV file output.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.config import Settings
from src.services.generation_service import GenerationService
from src.services.audio_service import AudioService, AudioConfig
from main import generate_music


class TestCompletePipeline:
    """Test the complete text-to-audio pipeline."""
    
    
    @pytest.mark.asyncio
    async def test_gabber_kick_generation(self, test_settings, audio_config):
        """Test gabber kick pattern generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_gabber.wav"
            
            result = await generate_music(
                prompt="create 180 BPM gabber kick pattern",
                output_path=output_path,
                settings=test_settings
            )
            
            assert result.exists()
            assert result.stat().st_size > 1000  # Should be substantial file
    
    @pytest.mark.asyncio
    async def test_acid_bassline_generation(self, test_settings, audio_config):
        """Test acid bassline generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_acid.wav"
            
            result = await generate_music(
                prompt="make acid bassline at 160 BPM in A minor",
                output_path=output_path,
                settings=test_settings
            )
            
            assert result.exists()
            assert result.stat().st_size > 1000
    
    @pytest.mark.asyncio
    async def test_hardcore_riff_generation(self, test_settings, audio_config):
        """Test hardcore riff generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_riff.wav"
            
            result = await generate_music(
                prompt="generate hardcore riff with heavy distortion",
                output_path=output_path,
                settings=test_settings
            )
            
            assert result.exists()
            assert result.stat().st_size > 1000
    
    def test_audio_service_initialization(self, test_settings, audio_config):
        """Test AudioService initializes correctly."""
        service = AudioService(test_settings, audio_config)
        
        assert service.audio_config.sample_rate == 22050
        assert service.audio_config.bit_depth == 16
        assert service.hardcore_synth is not None
        assert service.kick_synth is not None
        assert service.acid_synth is not None
    
    def test_progress_callback(self, test_settings, audio_config):
        """Test progress callback functionality."""
        progress_calls = []
        
        def progress_callback(progress):
            progress_calls.append(progress)
        
        service = AudioService(
            test_settings, 
            audio_config, 
            progress_callback=progress_callback
        )
        
        # Test progress reporting
        service._report_progress("test", 0.5, "Testing")
        
        assert len(progress_calls) == 1
        assert progress_calls[0].stage == "test"
        assert progress_calls[0].progress == 0.5
        assert progress_calls[0].message == "Testing"


class TestSynthesisModules:
    """Test individual synthesis components."""
    
    def test_kick_synthesizer(self):
        """Test kick drum synthesis."""
        from src.audio.synthesis import KickSynthesizer
        
        kick_synth = KickSynthesizer(sample_rate=22050)
        
        # Generate gabber kick
        kick_audio = kick_synth.generate_kick(
            duration=0.15,
            velocity=127,
            pitch=36,
            style="gabber"
        )
        
        assert len(kick_audio) > 0
        assert kick_audio.dtype == 'float32'
        assert kick_audio.max() <= 1.0
        assert kick_audio.min() >= -1.0
    
    def test_hardcore_synthesizer(self):
        """Test hardcore lead/riff synthesis."""
        from src.audio.synthesis import HardcoreSynthesizer
        
        hardcore_synth = HardcoreSynthesizer(sample_rate=22050)
        
        # Generate hardcore note
        note_audio = hardcore_synth.generate_note(
            pitch=60,  # Middle C
            duration=0.5,
            velocity=120,
            waveform="sawtooth"
        )
        
        assert len(note_audio) > 0
        assert note_audio.dtype == 'float32'
    
    def test_acid_synthesizer(self):
        """Test acid bassline synthesis."""
        from src.audio.synthesis import AcidSynthesizer
        from src.models.core import MIDINote
        
        acid_synth = AcidSynthesizer(sample_rate=22050)
        
        # Create test notes
        notes = [
            MIDINote(pitch=40, velocity=100, start_time=0.0, duration=0.25),
            MIDINote(pitch=43, velocity=100, start_time=0.25, duration=0.25),
        ]
        
        # Generate acid sequence
        acid_audio = acid_synth.generate_sequence(
            notes=notes,
            bpm=160,
            total_duration=2.0
        )
        
        assert len(acid_audio) > 0
        assert acid_audio.dtype == 'float32'


class TestAudioExport:
    """Test audio export functionality."""
    
    def test_wav_export_16bit(self, test_settings):
        """Test 16-bit WAV export."""
        from src.models.core import MIDIClip, MIDINote
        
        # Create simple test clip
        clip = MIDIClip(name="Test Clip", length_bars=1.0, bpm=120)
        clip.add_note(MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0))
        
        audio_service = AudioService(test_settings)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_16bit.wav"
            
            result = audio_service.render_to_wav(clip, output_path)
            
            assert result.exists()
            assert result.suffix == '.wav'
    
    def test_wav_export_24bit(self, test_settings):
        """Test 24-bit WAV export."""
        from src.models.core import MIDIClip, MIDINote
        
        # Create test clip
        clip = MIDIClip(name="Test Clip", length_bars=1.0, bpm=120)
        clip.add_note(MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0))
        
        audio_config = AudioConfig(bit_depth=24)
        audio_service = AudioService(test_settings, audio_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_24bit.wav"
            
            result = audio_service.render_to_wav(clip, output_path, audio_config)
            
            assert result.exists()
            assert result.suffix == '.wav'


class TestRenderStatistics:
    """Test rendering statistics tracking."""
    
    def test_stats_tracking(self, test_settings):
        """Test that rendering stats are properly tracked."""
        service = AudioService(test_settings)
        
        initial_stats = service.get_stats()
        assert initial_stats['total_renders'] == 0
        assert initial_stats['successful_renders'] == 0
        
        # Stats should update after renders (tested in integration tests)
        assert initial_stats['avg_render_time'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])