"""
Shared test fixtures for hardcore music prototyper tests.
"""

import pytest
from src.models.config import Settings
from src.services.audio_service import AudioConfig


@pytest.fixture
def test_settings():
    """Create test settings without requiring real API keys."""
    return Settings(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key", 
        google_api_key="test-google-key",
        
        # Test-specific overrides
        default_bpm=120,
        default_key="C",
        default_length_bars=2.0
    )


@pytest.fixture 
def audio_config():
    """Create test audio configuration."""
    return AudioConfig(
        sample_rate=22050,  # Lower for faster testing
        bit_depth=16,
        channels=1
    )