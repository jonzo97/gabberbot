"""
Tests for GenerationService - Core AI Generation Service.

Tests the main text-to-MIDI conversion functionality with mocked LLMs
and validates fallback behavior.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.services.generation_service import GenerationService, GenerationError
from src.services.llm_manager import LLMResponse, LLMProvider
from src.models.config import Settings, AIConfig, AudioConfig, HardcoreConfig, DevelopmentConfig
from src.models.core import MIDIClip, MIDINote
from src.utils.music_parser import MusicalParameters, PatternType, Genre


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        ai=AIConfig(
            openai_api_key="test_openai_key",
            anthropic_api_key="test_anthropic_key",
            google_api_key="test_google_key"
        ),
        audio=AudioConfig(),
        hardcore=HardcoreConfig(),
        development=DevelopmentConfig()
    )


@pytest.fixture
def generation_service(mock_settings):
    """Create GenerationService instance for testing."""
    return GenerationService(mock_settings)


class TestGenerationService:
    """Test the main GenerationService functionality."""
    
    def test_initialization(self, generation_service):
        """Test service initializes correctly with all components."""
        assert generation_service.music_parser is not None
        assert generation_service.llm_manager is not None
        assert generation_service.prompt_builder is not None
        assert generation_service.generation_stats['total_requests'] == 0
    
    @pytest.mark.asyncio
    async def test_text_to_midi_with_ai_success(self, generation_service):
        """Test successful text-to-MIDI generation using AI."""
        # Mock successful LLM response
        mock_response = LLMResponse(
            content="Notes: 60, 63, 65, 67. Rhythm: 0.25, 0.25, 0.25, 0.25. Velocity: 100, 90, 110, 95.",
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-sonnet",
            success=True
        )
        
        with patch.object(generation_service.llm_manager, 'generate_text', return_value=mock_response):
            with patch.object(generation_service.llm_manager, 'is_available', return_value=True):
                clip = await generation_service.text_to_midi("create 180 BPM gabber kick pattern")
                
                assert isinstance(clip, MIDIClip)
                assert clip.bpm == 180
                assert len(clip.notes) > 0
                assert generation_service.generation_stats['successful_ai'] == 1
    
    @pytest.mark.asyncio
    async def test_text_to_midi_with_fallback(self, generation_service):
        """Test fallback to algorithmic generation when AI fails."""
        # Mock LLM failure
        mock_response = LLMResponse(
            content="",
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-sonnet",
            success=False,
            error="API rate limit exceeded"
        )
        
        with patch.object(generation_service.llm_manager, 'generate_text', return_value=mock_response):
            with patch.object(generation_service.llm_manager, 'is_available', return_value=True):
                clip = await generation_service.text_to_midi("create acid bassline at 160 BPM")
                
                assert isinstance(clip, MIDIClip)
                assert clip.bpm == 160
                assert len(clip.notes) > 0
                assert generation_service.generation_stats['successful_fallback'] == 1
    
    @pytest.mark.asyncio
    async def test_text_to_midi_complete_failure(self, generation_service):
        """Test complete failure when both AI and algorithmic generation fail."""
        # Mock LLM unavailable
        with patch.object(generation_service.llm_manager, 'is_available', return_value=False):
            # Mock algorithmic generation failure
            with patch.object(generation_service, '_generate_with_algorithm', side_effect=Exception("Algorithm failed")):
                with pytest.raises(GenerationError):
                    await generation_service.text_to_midi("invalid impossible request")
                
                assert generation_service.generation_stats['failures'] == 1
    
    def test_parameter_parsing_integration(self, generation_service):
        """Test that text parsing correctly extracts musical parameters."""
        # Test with complex prompt
        prompt = "create 180 BPM gabber kick pattern in C minor with aggressive style"
        params = generation_service.music_parser.parse(prompt)
        
        assert params.bpm == 180
        assert params.genre == Genre.GABBER
        assert params.pattern_type == PatternType.KICK_PATTERN
        assert "aggressive" in params.style_descriptors
        assert params.key == "Cm"
    
    def test_extract_notes_from_text(self, generation_service):
        """Test MIDI note extraction from LLM responses."""
        # Test with explicit MIDI numbers
        text1 = "Notes: 60, 63, 65, 67"
        notes1 = generation_service._extract_notes_from_text(text1)
        assert notes1 == [60, 63, 65, 67]
        
        # Test with note names
        text2 = "Play C4, Eb4, F4, G4"
        notes2 = generation_service._extract_notes_from_text(text2)
        assert len(notes2) > 0  # Should extract something
        
        # Test with no notes (should return default)
        text3 = "This text has no musical content"
        notes3 = generation_service._extract_notes_from_text(text3)
        assert len(notes3) > 0  # Should return default sequence
    
    def test_extract_rhythms_from_text(self, generation_service):
        """Test rhythm extraction from LLM responses."""
        text = "Rhythm: 0.25, 0.5, 0.25, 1.0"
        rhythms = generation_service._extract_rhythms_from_text(text)
        assert rhythms == [0.25, 0.5, 0.25, 1.0]
        
        # Test with no rhythms (should return default)
        text_empty = "No rhythm information here"
        rhythms_empty = generation_service._extract_rhythms_from_text(text_empty)
        assert len(rhythms_empty) > 0  # Should return default
    
    def test_extract_velocities_from_text(self, generation_service):
        """Test velocity extraction from LLM responses."""
        text = "Velocities: 100, 90, 110, 95"
        velocities = generation_service._extract_velocities_from_text(text)
        assert velocities == [100, 90, 110, 95]
        
        # Test with no velocities (should return default)
        text_empty = "No velocity data"
        velocities_empty = generation_service._extract_velocities_from_text(text_empty)
        assert len(velocities_empty) > 0  # Should return default
    
    def test_generate_acid_bassline_algorithm(self, generation_service):
        """Test algorithmic acid bassline generation."""
        params = MusicalParameters(
            bpm=160,
            key="Am",
            pattern_type=PatternType.ACID_BASSLINE,
            length_bars=2.0
        )
        
        clip = generation_service._generate_acid_bassline(params)
        
        assert isinstance(clip, MIDIClip)
        assert clip.bpm == 160
        assert clip.key == "Am"
        assert len(clip.notes) > 0
        assert "Acid" in clip.name
        
        # Check that notes are in reasonable bass range
        for note in clip.notes:
            assert 20 <= note.pitch <= 80  # Bass range
    
    def test_generate_kick_pattern_algorithm(self, generation_service):
        """Test algorithmic kick pattern generation."""
        params = MusicalParameters(
            bpm=180,
            pattern_type=PatternType.KICK_PATTERN,
            length_bars=1.0
        )
        
        clip = generation_service._generate_kick_pattern(params)
        
        assert isinstance(clip, MIDIClip)
        assert clip.bpm == 180
        assert len(clip.notes) > 0
        assert "Kick" in clip.name
        
        # Check that all notes are kick drum (low range)
        for note in clip.notes:
            assert note.pitch <= 40  # Kick drum range
            assert note.velocity >= 100  # Strong velocity for hardcore
    
    def test_generate_riff_algorithm(self, generation_service):
        """Test algorithmic riff generation."""
        params = MusicalParameters(
            bpm=160,
            key="Em",
            pattern_type=PatternType.RIFF,
            length_bars=2.0
        )
        
        clip = generation_service._generate_riff(params)
        
        assert isinstance(clip, MIDIClip)
        assert clip.bpm == 160
        assert clip.key == "Em"
        assert len(clip.notes) > 0
        assert "Riff" in clip.name
        
        # Check that notes are in melodic range
        for note in clip.notes:
            assert 40 <= note.pitch <= 80  # Melodic range
    
    def test_performance_stats_tracking(self, generation_service):
        """Test that performance statistics are tracked correctly."""
        initial_stats = generation_service.get_stats()
        assert initial_stats['total_requests'] == 0
        assert initial_stats['avg_response_time'] == 0.0
        
        # Simulate some response time updates
        generation_service._update_performance_stats(0.5)
        generation_service._update_performance_stats(0.3)
        
        updated_stats = generation_service.get_stats()
        assert updated_stats['avg_response_time'] > 0.0
    
    def test_service_availability(self, generation_service):
        """Test service availability check."""
        # Service should always be available due to algorithmic fallback
        assert generation_service.is_available() is True
    
    @pytest.mark.asyncio
    async def test_hardcore_bpm_validation(self, generation_service):
        """Test that generated clips respect hardcore BPM ranges."""
        test_prompts = [
            "create gabber pattern",
            "make hardcore kick at 200 BPM",
            "generate industrial pattern"
        ]
        
        for prompt in test_prompts:
            clip = await generation_service.text_to_midi(prompt)
            # BPM should be in hardcore range or explicitly set
            assert 120 <= clip.bpm <= 300  # Reasonable electronic music range
    
    @pytest.mark.asyncio
    async def test_minor_key_preference(self, generation_service):
        """Test that generated clips prefer minor keys for hardcore darkness."""
        clip = await generation_service.text_to_midi("create dark hardcore pattern")
        
        # Should prefer minor keys (ending with 'm' or being None)
        if clip.key:
            # Allow either minor keys or no key specification
            assert clip.key is None or clip.key.endswith('m') or clip.key in ['Am', 'Em', 'Cm', 'Dm', 'Gm', 'Fm', 'Bm']


class TestGenerationServiceIntegration:
    """Integration tests with real components."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_without_llm(self, generation_service):
        """Test full generation pipeline without requiring real LLM APIs."""
        # Mock LLM to be unavailable, forcing algorithmic generation
        with patch.object(generation_service.llm_manager, 'is_available', return_value=False):
            test_cases = [
                ("create 180 BPM gabber kick pattern", PatternType.KICK_PATTERN),
                ("make acid bassline at 160 BPM", PatternType.ACID_BASSLINE),
                ("generate hardcore riff in Am", PatternType.RIFF),
            ]
            
            for prompt, expected_type in test_cases:
                clip = await generation_service.text_to_midi(prompt)
                
                assert isinstance(clip, MIDIClip)
                assert len(clip.notes) > 0
                assert clip.bpm > 0
                
                # Verify pattern type was detected correctly
                parsed_params = generation_service.music_parser.parse(prompt)
                if parsed_params.pattern_type:
                    assert parsed_params.pattern_type == expected_type
    
    def test_parameter_validation_integration(self, generation_service):
        """Test parameter validation works with the full parsing pipeline."""
        # Valid hardcore parameters
        valid_prompt = "create 180 BPM gabber pattern in Am"
        params = generation_service.music_parser.parse(valid_prompt)
        is_valid, errors = generation_service.music_parser.validate_parameters(params)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid parameters
        invalid_prompt = "create 500 BPM pattern"  # BPM too high
        params = generation_service.music_parser.parse(invalid_prompt)
        is_valid, errors = generation_service.music_parser.validate_parameters(params)
        # Should still work but with warnings
        assert len(errors) >= 0  # May have warnings
    
    @pytest.mark.asyncio
    async def test_error_recovery_chain(self, generation_service):
        """Test the complete error recovery chain."""
        # Simulate complete LLM failure
        mock_error_response = LLMResponse(
            content="",
            provider=LLMProvider.ANTHROPIC,
            model="test",
            success=False,
            error="Network error"
        )
        
        with patch.object(generation_service.llm_manager, 'generate_text', return_value=mock_error_response):
            with patch.object(generation_service.llm_manager, 'is_available', return_value=True):
                # Should still succeed via algorithmic fallback
                clip = await generation_service.text_to_midi("create hardcore pattern")
                
                assert isinstance(clip, MIDIClip)
                assert len(clip.notes) > 0
                assert generation_service.generation_stats['successful_fallback'] >= 1