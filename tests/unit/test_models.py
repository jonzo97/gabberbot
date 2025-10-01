"""
Unit tests for core data models.

Tests all Pydantic models for validation, serialization, and musical logic.
"""

import pytest
import json
from datetime import datetime
from typing import List

from src.models.core import MIDINote, MIDIClip, AutomationPoint, AutomationClip
from src.models.config import Settings, AIConfig, AudioConfig, HardcoreConfig


class TestMIDINote:
    """Test cases for MIDINote model."""
    
    def test_valid_note_creation(self):
        """Test creating a valid MIDI note."""
        note = MIDINote(
            pitch=60,  # Middle C
            velocity=100,
            start_time=0.0,
            duration=1.0
        )
        
        assert note.pitch == 60
        assert note.velocity == 100
        assert note.start_time == 0.0
        assert note.duration == 1.0
        assert note.channel == 0  # Default
    
    def test_note_with_channel(self):
        """Test creating note with specific channel."""
        note = MIDINote(
            pitch=64,
            velocity=80,
            start_time=1.0,
            duration=0.5,
            channel=9  # Drum channel
        )
        
        assert note.channel == 9
    
    def test_invalid_pitch_validation(self):
        """Test pitch validation (must be 0-127)."""
        # Test pitch too low
        with pytest.raises(ValueError):
            MIDINote(pitch=-1, velocity=100, start_time=0.0, duration=1.0)
        
        # Test pitch too high
        with pytest.raises(ValueError):
            MIDINote(pitch=128, velocity=100, start_time=0.0, duration=1.0)
    
    def test_invalid_velocity_validation(self):
        """Test velocity validation (must be 0-127)."""
        # Test velocity too low
        with pytest.raises(ValueError):
            MIDINote(pitch=60, velocity=-1, start_time=0.0, duration=1.0)
        
        # Test velocity too high
        with pytest.raises(ValueError):
            MIDINote(pitch=60, velocity=128, start_time=0.0, duration=1.0)
    
    def test_invalid_timing_validation(self):
        """Test timing validation."""
        # Test negative start time
        with pytest.raises(ValueError):
            MIDINote(pitch=60, velocity=100, start_time=-1.0, duration=1.0)
        
        # Test zero or negative duration
        with pytest.raises(ValueError):
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.0)
        
        with pytest.raises(ValueError):
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=-1.0)
    
    def test_invalid_channel_validation(self):
        """Test channel validation (must be 0-15)."""
        # Test channel too low
        with pytest.raises(ValueError):
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0, channel=-1)
        
        # Test channel too high
        with pytest.raises(ValueError):
            MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0, channel=16)
    
    def test_end_time_property(self):
        """Test end_time calculated property."""
        note = MIDINote(pitch=60, velocity=100, start_time=2.0, duration=1.5)
        assert note.end_time == 3.5
    
    def test_frequency_calculation(self):
        """Test frequency calculation from MIDI pitch."""
        # A4 = MIDI 69 = 440 Hz
        note_a4 = MIDINote(pitch=69, velocity=100, start_time=0.0, duration=1.0)
        assert abs(note_a4.frequency_hz - 440.0) < 0.01
        
        # Middle C = MIDI 60 ≈ 261.63 Hz
        note_c4 = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        assert abs(note_c4.frequency_hz - 261.63) < 0.01
    
    def test_string_representation(self):
        """Test human-readable string output."""
        note = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        str_repr = str(note)
        assert "C4" in str_repr
        assert "vel:100" in str_repr
        assert "t:0.0-1.0" in str_repr
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        note = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        
        # Test serialization
        json_data = note.json()
        assert isinstance(json_data, str)
        
        # Test deserialization
        note_restored = MIDINote.parse_raw(json_data)
        assert note_restored.pitch == note.pitch
        assert note_restored.velocity == note.velocity
        assert note_restored.start_time == note.start_time
        assert note_restored.duration == note.duration


class TestMIDIClip:
    """Test cases for MIDIClip model."""
    
    def test_empty_clip_creation(self):
        """Test creating an empty MIDI clip."""
        clip = MIDIClip(id="test_clip", name="Test Clip")
        
        assert clip.id == "test_clip"
        assert clip.name == "Test Clip"
        assert len(clip.notes) == 0
        assert clip.length_bars == 4.0  # Default
        assert clip.bpm == 120.0  # Default
        assert clip.time_signature == (4, 4)  # Default
        assert clip.key is None  # Default
        assert isinstance(clip.created_at, datetime)
    
    def test_clip_with_custom_params(self):
        """Test creating clip with custom parameters."""
        clip = MIDIClip(
            id="hardcore_clip",
            name="Gabber Pattern",
            length_bars=2.0,
            bpm=180.0,
            time_signature=(4, 4),
            key="Am"
        )
        
        assert clip.bpm == 180.0
        assert clip.length_bars == 2.0
        assert clip.key == "Am"
    
    def test_invalid_bpm_validation(self):
        """Test BPM validation."""
        # Test BPM too low
        with pytest.raises(ValueError):
            MIDIClip(id="test", name="test", bpm=59.0)
        
        # Test BPM too high
        with pytest.raises(ValueError):
            MIDIClip(id="test", name="test", bpm=301.0)
    
    def test_invalid_time_signature_validation(self):
        """Test time signature validation."""
        # Test invalid tuple length
        with pytest.raises(ValueError):
            MIDIClip(id="test", name="test", time_signature=(4,))
        
        # Test invalid values
        with pytest.raises(ValueError):
            MIDIClip(id="test", name="test", time_signature=(0, 4))
        
        # Test invalid denominator
        with pytest.raises(ValueError):
            MIDIClip(id="test", name="test", time_signature=(4, 3))
    
    def test_invalid_key_validation(self):
        """Test musical key validation."""
        # Test invalid key
        with pytest.raises(ValueError):
            MIDIClip(id="test", name="test", key="H")  # Invalid note
        
        with pytest.raises(ValueError):
            MIDIClip(id="test", name="test", key="Cmaj")  # Invalid format
    
    def test_valid_keys(self):
        """Test various valid musical keys."""
        valid_keys = ["C", "C#", "Db", "Am", "F#m", "Bbm"]
        
        for key in valid_keys:
            clip = MIDIClip(id="test", name="test", key=key)
            assert clip.key == key
    
    def test_add_remove_notes(self):
        """Test adding and removing notes from clip."""
        clip = MIDIClip(id="test", name="test")
        
        note1 = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        note2 = MIDINote(pitch=64, velocity=80, start_time=1.0, duration=1.0)
        
        # Test adding notes
        clip.add_note(note1)
        clip.add_note(note2)
        
        assert len(clip.notes) == 2
        assert note1 in clip.notes
        assert note2 in clip.notes
        
        # Test removing notes
        assert clip.remove_note(note1) is True
        assert len(clip.notes) == 1
        assert note1 not in clip.notes
        
        # Test removing non-existent note
        assert clip.remove_note(note1) is False
    
    def test_length_calculations(self):
        """Test various length calculation properties."""
        clip = MIDIClip(
            id="test", 
            name="test", 
            length_bars=2.0, 
            bpm=120.0, 
            time_signature=(4, 4)
        )
        
        assert clip.length_beats == 8.0  # 2 bars * 4 beats/bar
        assert clip.length_seconds == 4.0  # 8 beats / (120 BPM / 60 s/min)
    
    def test_note_query_methods(self):
        """Test methods for querying notes."""
        clip = MIDIClip(id="test", name="test")
        
        # Add test notes
        note1 = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        note2 = MIDINote(pitch=64, velocity=80, start_time=0.0, duration=0.5)  # Same start time
        note3 = MIDINote(pitch=67, velocity=90, start_time=2.0, duration=1.0)
        
        clip.add_note(note1)
        clip.add_note(note2)
        clip.add_note(note3)
        
        # Test getting notes at specific time
        notes_at_zero = clip.get_notes_at_time(0.0)
        assert len(notes_at_zero) == 2
        assert note1 in notes_at_zero
        assert note2 in notes_at_zero
        
        # Test getting notes in range
        notes_in_range = clip.get_notes_in_range(0.0, 1.5)
        assert len(notes_in_range) == 2  # note1 and note2 overlap this range
        assert note3 not in notes_in_range
    
    def test_quantize_method(self):
        """Test note quantization."""
        clip = MIDIClip(id="test", name="test")
        
        # Add notes with off-grid timing
        note1 = MIDINote(pitch=60, velocity=100, start_time=0.1, duration=1.0)
        note2 = MIDINote(pitch=64, velocity=80, start_time=1.3, duration=0.5)
        
        clip.add_note(note1)
        clip.add_note(note2)
        
        # Quantize to quarter note grid
        clip.quantize(grid=1.0)
        
        assert clip.notes[0].start_time == 0.0  # 0.1 -> 0.0
        assert clip.notes[1].start_time == 1.0  # 1.3 -> 1.0
    
    def test_transpose_method(self):
        """Test transposition of all notes."""
        clip = MIDIClip(id="test", name="test")
        
        # Add notes
        note1 = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)  # C4
        note2 = MIDINote(pitch=64, velocity=80, start_time=1.0, duration=1.0)   # E4
        
        clip.add_note(note1)
        clip.add_note(note2)
        
        # Transpose up by 2 semitones
        clip.transpose(2)
        
        assert clip.notes[0].pitch == 62  # C4 -> D4
        assert clip.notes[1].pitch == 66  # E4 -> F#4
    
    def test_transpose_clamping(self):
        """Test transposition clamping to MIDI range."""
        clip = MIDIClip(id="test", name="test")
        
        # Add note near boundaries
        note_low = MIDINote(pitch=1, velocity=100, start_time=0.0, duration=1.0)
        note_high = MIDINote(pitch=126, velocity=100, start_time=1.0, duration=1.0)
        
        clip.add_note(note_low)
        clip.add_note(note_high)
        
        # Transpose down (should clamp to 0)
        clip.transpose(-5)
        assert clip.notes[0].pitch == 0
        
        # Reset and transpose up (should clamp to 127)
        clip.notes[1].pitch = 126
        clip.transpose(5)
        assert clip.notes[1].pitch == 127
    
    def test_json_serialization(self):
        """Test clip JSON serialization and deserialization."""
        clip = MIDIClip(
            id="test_clip",
            name="Test Pattern",
            bpm=180.0,
            key="Am"
        )
        
        # Add a note
        note = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        clip.add_note(note)
        
        # Test serialization
        json_str = clip.to_json()
        assert isinstance(json_str, str)
        
        # Test deserialization
        clip_restored = MIDIClip.from_json(json_str)
        assert clip_restored.id == clip.id
        assert clip_restored.name == clip.name
        assert clip_restored.bpm == clip.bpm
        assert clip_restored.key == clip.key
        assert len(clip_restored.notes) == 1
        assert clip_restored.notes[0].pitch == 60


class TestAutomationModels:
    """Test cases for automation models."""
    
    def test_automation_point_creation(self):
        """Test creating automation points."""
        point = AutomationPoint(time=1.0, value=0.5)
        assert point.time == 1.0
        assert point.value == 0.5
    
    def test_automation_clip_creation(self):
        """Test creating automation clips."""
        clip = AutomationClip(
            id="filter_cutoff",
            parameter_name="filter.cutoff",
            min_value=0.0,
            max_value=1.0
        )
        
        assert clip.id == "filter_cutoff"
        assert clip.parameter_name == "filter.cutoff"
        assert len(clip.points) == 0
    
    def test_automation_point_ordering(self):
        """Test that automation points maintain chronological order."""
        clip = AutomationClip(id="test", parameter_name="volume")
        
        # Add points out of order
        point3 = AutomationPoint(time=3.0, value=0.8)
        point1 = AutomationPoint(time=1.0, value=0.2)
        point2 = AutomationPoint(time=2.0, value=0.5)
        
        clip.add_point(point3)
        clip.add_point(point1)
        clip.add_point(point2)
        
        # Verify they're in chronological order
        assert clip.points[0].time == 1.0
        assert clip.points[1].time == 2.0
        assert clip.points[2].time == 3.0
    
    def test_automation_value_interpolation(self):
        """Test automation value interpolation."""
        clip = AutomationClip(id="test", parameter_name="volume")
        
        # Add control points
        clip.add_point(AutomationPoint(time=0.0, value=0.0))
        clip.add_point(AutomationPoint(time=2.0, value=1.0))
        
        # Test interpolation at midpoint
        value_at_1 = clip.get_value_at_time(1.0)
        assert abs(value_at_1 - 0.5) < 0.001  # Should be halfway
        
        # Test values outside range
        assert clip.get_value_at_time(-1.0) == 0.0  # Before first point
        assert clip.get_value_at_time(3.0) == 1.0   # After last point
    
    def test_automation_value_validation(self):
        """Test automation value range validation."""
        clip = AutomationClip(
            id="test", 
            parameter_name="volume",
            min_value=0.0,
            max_value=1.0
        )
        
        # Test valid point
        valid_point = AutomationPoint(time=1.0, value=0.5)
        clip.add_point(valid_point)
        
        # Test invalid points during creation
        with pytest.raises(ValueError):
            AutomationClip(
                id="test2",
                parameter_name="volume",
                min_value=0.0,
                max_value=1.0,
                points=[AutomationPoint(time=1.0, value=1.5)]  # Above max
            )


class TestConfigModels:
    """Test cases for configuration models."""
    
    def test_ai_config_creation(self):
        """Test AI configuration creation."""
        config = AIConfig(
            default_provider="anthropic",
            max_tokens=500,
            temperature=0.8
        )
        
        assert config.default_provider == "anthropic"
        assert config.max_tokens == 500
        assert config.temperature == 0.8
    
    def test_ai_config_validation(self):
        """Test AI configuration validation."""
        # Test invalid provider
        with pytest.raises(ValueError):
            AIConfig(default_provider="invalid_provider")
        
        # Test invalid token count
        with pytest.raises(ValueError):
            AIConfig(max_tokens=0)
        
        with pytest.raises(ValueError):
            AIConfig(max_tokens=5000)
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            AIConfig(temperature=-1.0)
        
        with pytest.raises(ValueError):
            AIConfig(temperature=3.0)
    
    def test_audio_config_validation(self):
        """Test audio configuration validation."""
        # Test invalid sample rate
        with pytest.raises(ValueError):
            AudioConfig(sample_rate=32000)
        
        # Test invalid bit depth
        with pytest.raises(ValueError):
            AudioConfig(bit_depth=12)
    
    def test_hardcore_config_validation(self):
        """Test hardcore music configuration validation."""
        # Test invalid BPM range
        with pytest.raises(ValueError):
            HardcoreConfig(default_bpm_range=(200, 150))  # Min > Max
        
        with pytest.raises(ValueError):
            HardcoreConfig(default_bpm_range=(50, 150))  # Below minimum
        
        with pytest.raises(ValueError):
            HardcoreConfig(default_bpm_range=(150, 350))  # Above maximum
    
    def test_settings_creation(self):
        """Test main settings creation."""
        settings = Settings()
        
        assert settings.project_name == "Hardcore Music Prototyper"
        assert settings.version == "0.1.0"
        assert settings.environment == "development"
        assert isinstance(settings.ai, AIConfig)
        assert isinstance(settings.audio, AudioConfig)
        assert isinstance(settings.hardcore, HardcoreConfig)
    
    def test_settings_api_key_methods(self):
        """Test API key utility methods."""
        settings = Settings()
        
        # Test with no API keys
        assert not settings.has_api_key("openai")
        assert settings.get_api_key("openai") is None
        assert len(settings.get_available_providers()) == 0
    
    def test_settings_dict_conversion(self):
        """Test settings dictionary conversion."""
        settings = Settings()
        config_dict = settings.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "project_name" in config_dict
        assert "ai" in config_dict
        assert "audio" in config_dict


# Performance tests
class TestPerformance:
    """Test performance requirements."""
    
    def test_model_creation_performance(self):
        """Test that model creation is under 1ms requirement."""
        import time
        
        # Test MIDINote creation
        start_time = time.perf_counter()
        for _ in range(100):
            note = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=1.0)
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        assert avg_time_ms < 1.0  # Should be under 1ms per note
        
        # Test MIDIClip creation
        start_time = time.perf_counter()
        for _ in range(50):
            clip = MIDIClip(id=f"clip_{_}", name=f"Test Clip {_}")
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / 50) * 1000
        assert avg_time_ms < 1.0  # Should be under 1ms per clip


# Integration test examples
class TestModelIntegration:
    """Test model integration scenarios."""
    
    def test_complete_midi_workflow(self):
        """Test complete MIDI creation and manipulation workflow."""
        # Create clip
        clip = MIDIClip(
            id="test_pattern",
            name="Hardcore Pattern",
            bpm=180.0,
            key="Am",
            length_bars=1.0
        )
        
        # Add notes for a simple pattern
        notes = [
            MIDINote(pitch=57, velocity=127, start_time=0.0, duration=0.25),   # A2 kick
            MIDINote(pitch=57, velocity=100, start_time=0.5, duration=0.25),   # A2 kick
            MIDINote(pitch=57, velocity=127, start_time=1.0, duration=0.25),   # A2 kick
            MIDINote(pitch=57, velocity=100, start_time=1.5, duration=0.25),   # A2 kick
        ]
        
        for note in notes:
            clip.add_note(note)
        
        # Verify pattern
        assert len(clip.notes) == 4
        assert clip.note_count == 4
        
        # Test serialization
        json_data = clip.to_json()
        restored_clip = MIDIClip.from_json(json_data)
        
        assert restored_clip.bpm == 180.0
        assert restored_clip.key == "Am"
        assert len(restored_clip.notes) == 4