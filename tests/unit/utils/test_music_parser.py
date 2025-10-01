"""
Tests for MusicParser - Natural Language Musical Parameter Extraction.

Tests extraction of BPM, keys, genres, and other musical parameters
from natural language prompts.
"""

import pytest

from src.utils.music_parser import (
    MusicParser, MusicalParameters, PatternType, Genre,
    parse_musical_text, HARDCORE_PARAMS
)


class TestMusicParser:
    """Test the MusicParser natural language processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MusicParser()
    
    def test_bpm_extraction_basic(self):
        """Test basic BPM extraction from text."""
        test_cases = [
            ("create 180 bpm pattern", 180),
            ("make something at 160 BPM", 160),
            ("generate 200 beats per minute", 200),
            ("tempo 175", 175),
            ("at 220", 220),
        ]
        
        for text, expected_bpm in test_cases:
            params = self.parser.parse(text)
            assert params.bpm == expected_bpm, f"Failed for: {text}"
    
    def test_bpm_validation(self):
        """Test BPM validation and range checking."""
        # Valid BPM
        params = self.parser.parse("create 180 bpm pattern")
        assert params.bpm == 180
        
        # Invalid BPM (too high)
        params = self.parser.parse("create 500 bpm pattern")
        assert params.bpm == 500  # Parser extracts it
        
        # Validation should catch the error
        is_valid, errors = self.parser.validate_parameters(params)
        assert not is_valid
        assert any("outside valid range" in error for error in errors)
    
    def test_key_extraction(self):
        """Test musical key extraction from text."""
        test_cases = [
            ("pattern in C minor", "Cm"),
            ("create bassline in A minor", "Am"),
            ("make riff in E major", "E"),  # Explicit major
            ("make riff in E", "Em"),       # Defaults to minor for hardcore
            ("key of D minor", "Dm"),
            ("in Fm", "Fm"),
        ]
        
        for text, expected_key in test_cases:
            params = self.parser.parse(text)
            assert params.key == expected_key, f"Failed for: {text}"
    
    def test_genre_extraction(self):
        """Test genre detection from text."""
        test_cases = [
            ("create gabber pattern", Genre.GABBER),
            ("make hardcore kick", Genre.HARDCORE),
            ("industrial warehouse sound", Genre.INDUSTRIAL),
            ("frenchcore style", Genre.FRENCHCORE),
            ("acid bassline", Genre.ACID),
            ("uptempo pattern", Genre.UPTEMPO),
        ]
        
        for text, expected_genre in test_cases:
            params = self.parser.parse(text)
            assert params.genre == expected_genre, f"Failed for: {text}"
    
    def test_pattern_type_extraction(self):
        """Test pattern type detection from text."""
        test_cases = [
            ("create acid bassline", PatternType.ACID_BASSLINE),
            ("make kick pattern", PatternType.KICK_PATTERN),
            ("generate riff", PatternType.RIFF),
            ("chord progression", PatternType.CHORD_PROGRESSION),
            ("arpeggio pattern", PatternType.ARPEGGIO),
            ("drum pattern", PatternType.DRUM_PATTERN),
            ("hihat sequence", PatternType.HIHAT_PATTERN),
        ]
        
        for text, expected_type in test_cases:
            params = self.parser.parse(text)
            assert params.pattern_type == expected_type, f"Failed for: {text}"
    
    def test_style_descriptors_extraction(self):
        """Test style descriptor extraction."""
        test_cases = [
            ("aggressive hardcore pattern", ["aggressive"]),
            ("dark and heavy industrial", ["dark"]),
            ("fast complex gabber", ["fast", "complex"]),
            ("simple minimal techno", ["simple"]),
            ("groovy and tight rhythm", ["groovy", "tight"]),
        ]
        
        for text, expected_styles in test_cases:
            params = self.parser.parse(text)
            for style in expected_styles:
                assert style in params.style_descriptors, f"Missing {style} in: {text}"
    
    def test_effects_extraction(self):
        """Test audio effects extraction."""
        test_cases = [
            ("add reverb and delay", ["reverb", "delay"]),
            ("with distortion", ["distortion"]),
            ("sidechain compression", ["sidechain", "compression"]),
            ("filter sweep", ["filter"]),
        ]
        
        for text, expected_effects in test_cases:
            params = self.parser.parse(text)
            for effect in expected_effects:
                assert effect in params.effects, f"Missing {effect} in: {text}"
    
    def test_mood_determination(self):
        """Test mood inference from style descriptors."""
        test_cases = [
            ("aggressive brutal pattern", "aggressive"),
            ("dark sinister bassline", "dark"),
            ("groovy funky rhythm", "groovy"),
            ("simple basic beat", "minimal"),
            ("energetic pattern", "energetic"),  # Default case
        ]
        
        for text, expected_mood in test_cases:
            params = self.parser.parse(text)
            assert params.mood == expected_mood, f"Failed for: {text}"
    
    def test_complexity_determination(self):
        """Test complexity inference from style descriptors."""
        test_cases = [
            ("complex intricate pattern", "complex"),
            ("simple basic beat", "simple"),
            ("detailed sophisticated riff", "complex"),
            ("minimal clean sound", "simple"),
            ("normal pattern", "medium"),  # Default case
        ]
        
        for text, expected_complexity in test_cases:
            params = self.parser.parse(text)
            assert params.complexity == expected_complexity, f"Failed for: {text}"
    
    def test_genre_defaults_application(self):
        """Test that genre-specific defaults are applied."""
        # Test gabber defaults
        params = self.parser.parse("create gabber pattern")
        assert params.genre == Genre.GABBER
        assert 150 <= params.bpm <= 200  # Gabber BPM range
        assert params.key == "Am"  # Default minor key
        assert params.pattern_type == PatternType.ACID_BASSLINE  # Gabber default
        
        # Test hardcore defaults
        params = self.parser.parse("make hardcore pattern")
        assert params.genre == Genre.HARDCORE
        assert 180 <= params.bpm <= 250  # Hardcore BPM range
        assert params.pattern_type == PatternType.KICK_PATTERN  # Hardcore default
        
        # Test acid defaults
        params = self.parser.parse("create acid pattern")
        assert params.genre == Genre.ACID
        assert params.pattern_type == PatternType.ACID_BASSLINE  # Acid default
    
    def test_complex_prompt_parsing(self):
        """Test parsing of complex, multi-parameter prompts."""
        prompt = "create 180 BPM aggressive gabber kick pattern in C minor with distortion and reverb"
        params = self.parser.parse(prompt)
        
        assert params.bpm == 180
        assert params.genre == Genre.GABBER
        assert params.pattern_type == PatternType.KICK_PATTERN
        assert params.key == "Cm"
        assert "aggressive" in params.style_descriptors
        assert "distortion" in params.effects
        assert "reverb" in params.effects
        assert params.mood == "aggressive"
    
    def test_edge_cases(self):
        """Test edge cases and malformed input."""
        # Empty string
        params = self.parser.parse("")
        assert isinstance(params, MusicalParameters)
        assert params.bpm is None
        
        # No musical content
        params = self.parser.parse("hello world this is not about music")
        assert isinstance(params, MusicalParameters)
        assert params.bpm is None
        assert params.genre is None
        
        # Invalid BPM values
        params = self.parser.parse("create 999999 bpm pattern")
        assert params.bpm == 999999  # Parser extracts but validation catches
        
        is_valid, errors = self.parser.validate_parameters(params)
        assert not is_valid
    
    def test_parameter_validation(self):
        """Test comprehensive parameter validation."""
        # Valid parameters
        params = MusicalParameters(
            bpm=180,
            key="Am",
            genre=Genre.GABBER
        )
        is_valid, errors = self.parser.validate_parameters(params)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid BPM
        params.bpm = 500
        is_valid, errors = self.parser.validate_parameters(params)
        assert not is_valid
        assert any("outside valid range" in error for error in errors)
        
        # Invalid key format
        params.bpm = 180  # Fix BPM
        params.key = "InvalidKey"
        is_valid, errors = self.parser.validate_parameters(params)
        assert not is_valid
        assert any("Invalid key format" in error for error in errors)
        
        # BPM too slow for hardcore
        params.key = "Am"  # Fix key
        params.bpm = 80
        is_valid, errors = self.parser.validate_parameters(params)
        assert not is_valid
        assert any("too slow for hardcore" in error for error in errors)
    
    def test_case_insensitive_parsing(self):
        """Test that parsing is case-insensitive."""
        test_cases = [
            "CREATE 180 BPM GABBER PATTERN",
            "create 180 bpm gabber pattern",
            "Create 180 BPM Gabber Pattern",
            "cReAtE 180 bPm GaBbEr PaTtErN",
        ]
        
        for text in test_cases:
            params = self.parser.parse(text)
            assert params.bpm == 180
            assert params.genre == Genre.GABBER
    
    def test_pattern_type_priority(self):
        """Test pattern type detection with multiple keywords."""
        # Should prioritize most specific pattern type
        prompt = "create acid bassline kick pattern with riff elements"
        params = self.parser.parse(prompt)
        
        # Should detect multiple patterns but choose highest scoring one
        assert params.pattern_type is not None
        # The parser should pick the most relevant one based on scoring
    
    def test_hardcore_knowledge_integration(self):
        """Test integration with hardcore music knowledge."""
        # Verify hardcore parameters are loaded correctly
        assert 'gabber' in HARDCORE_PARAMS['bpm_ranges']
        assert 'hardcore' in HARDCORE_PARAMS['bpm_ranges']
        assert 'industrial' in HARDCORE_PARAMS['bpm_ranges']
        
        # Test that defaults align with hardcore knowledge
        params = self.parser.parse("create gabber pattern")
        gabber_range = HARDCORE_PARAMS['bpm_ranges']['gabber']
        assert gabber_range[0] <= params.bpm <= gabber_range[1]


class TestConvenienceFunction:
    """Test the convenience parse_musical_text function."""
    
    def test_parse_musical_text_function(self):
        """Test the standalone parsing function."""
        params = parse_musical_text("create 180 BPM gabber pattern")
        
        assert isinstance(params, MusicalParameters)
        assert params.bpm == 180
        assert params.genre == Genre.GABBER
    
    def test_parse_musical_text_with_complex_input(self):
        """Test convenience function with complex input."""
        prompt = "make aggressive 200 BPM hardcore kick in Am with heavy distortion"
        params = parse_musical_text(prompt)
        
        assert params.bpm == 200
        assert params.genre == Genre.HARDCORE
        assert params.pattern_type == PatternType.KICK_PATTERN
        assert params.key == "Am"
        assert "aggressive" in params.style_descriptors
        assert "distortion" in params.effects


class TestHardcoreKnowledge:
    """Test hardcore music specific knowledge and constraints."""
    
    def test_hardcore_bpm_ranges(self):
        """Test that BPM ranges are appropriate for hardcore genres."""
        ranges = HARDCORE_PARAMS['bpm_ranges']
        
        # Verify all ranges are in electronic music territory
        for genre, (min_bpm, max_bpm) in ranges.items():
            assert 100 <= min_bpm <= 300, f"{genre} min BPM out of range"
            assert 100 <= max_bpm <= 300, f"{genre} max BPM out of range"
            assert min_bpm < max_bpm, f"{genre} BPM range invalid"
    
    def test_preferred_keys_are_minor(self):
        """Test that preferred keys emphasize minor keys for darkness."""
        preferred_keys = HARDCORE_PARAMS['preferred_keys']
        
        # Most keys should be minor (ending with 'm')
        minor_keys = [key for key in preferred_keys if key.endswith('m')]
        assert len(minor_keys) >= len(preferred_keys) * 0.8  # At least 80% minor
    
    def test_pattern_types_are_comprehensive(self):
        """Test that pattern types cover main hardcore elements."""
        pattern_types = HARDCORE_PARAMS['pattern_types']
        
        essential_patterns = ['acid_bassline', 'kick_pattern', 'riff']
        for pattern in essential_patterns:
            assert pattern in pattern_types, f"Missing essential pattern: {pattern}"
    
    def test_genre_coverage(self):
        """Test that main hardcore genres are covered."""
        genres = HARDCORE_PARAMS['genres']
        
        essential_genres = ['gabber', 'hardcore', 'industrial', 'acid']
        for genre in essential_genres:
            assert genre in genres, f"Missing essential genre: {genre}"