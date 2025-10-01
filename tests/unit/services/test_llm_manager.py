"""
Tests for LLMManager - Multi-LLM Client Management.

Tests LLM provider fallback, error handling, and response processing.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.services.llm_manager import LLMManager, LLMProvider, LLMResponse
from src.models.config import Settings, AIConfig, AudioConfig, HardcoreConfig, DevelopmentConfig


@pytest.fixture
def mock_settings():
    """Create mock settings with API keys."""
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
def mock_settings_no_keys():
    """Create mock settings without API keys."""
    return Settings(
        ai=AIConfig(),
        audio=AudioConfig(),
        hardcore=HardcoreConfig(),
        development=DevelopmentConfig()
    )


class TestLLMManager:
    """Test LLM manager functionality."""
    
    def test_initialization_with_keys(self, mock_settings):
        """Test LLM manager initializes with API keys."""
        with patch('src.services.llm_manager.ANTHROPIC_AVAILABLE', True):
            with patch('src.services.llm_manager.OPENAI_AVAILABLE', True):
                with patch('src.services.llm_manager.GOOGLE_AI_AVAILABLE', True):
                    # Mock the actual client initialization
                    with patch('src.services.llm_manager.anthropic.Anthropic') as mock_anthropic:
                        with patch('src.services.llm_manager.openai.OpenAI') as mock_openai:
                            with patch('src.services.llm_manager.genai.configure') as mock_genai_config:
                                with patch('src.services.llm_manager.genai.GenerativeModel') as mock_genai_model:
                                    manager = LLMManager(mock_settings)
                                    
                                    # Verify client initialization was attempted
                                    mock_anthropic.assert_called_once()
                                    mock_openai.assert_called_once()
                                    mock_genai_config.assert_called_once()
                                    mock_genai_model.assert_called_once()
    
    def test_initialization_without_keys(self, mock_settings_no_keys):
        """Test LLM manager handles missing API keys gracefully."""
        manager = LLMManager(mock_settings_no_keys)
        
        # Should not crash, but no clients should be available
        available = manager.get_available_providers()
        assert len(available) == 0
    
    def test_get_available_providers(self, mock_settings):
        """Test getting list of available providers."""
        manager = LLMManager(mock_settings)
        
        # Mock some clients as available
        manager.anthropic_client = Mock()
        manager.openai_client = Mock()
        manager.google_client = None
        
        available = manager.get_available_providers()
        assert LLMProvider.ANTHROPIC in available
        assert LLMProvider.OPENAI in available
        assert LLMProvider.GOOGLE not in available
    
    @pytest.mark.asyncio
    async def test_generate_text_success_anthropic(self, mock_settings):
        """Test successful text generation with Anthropic."""
        manager = LLMManager(mock_settings)
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated hardcore pattern")]
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response
        manager.anthropic_client = mock_client
        
        response = await manager.generate_text(
            prompt="Create hardcore pattern",
            system_prompt="You are a hardcore producer"
        )
        
        assert response.success is True
        assert response.provider == LLMProvider.ANTHROPIC
        assert "hardcore pattern" in response.content
        assert response.tokens_used == 50
    
    @pytest.mark.asyncio
    async def test_generate_text_success_openai(self, mock_settings):
        """Test successful text generation with OpenAI."""
        manager = LLMManager(mock_settings)
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated music"))]
        mock_response.usage.total_tokens = 75
        mock_client.chat.completions.create.return_value = mock_response
        manager.openai_client = mock_client
        manager.anthropic_client = None  # Force OpenAI usage
        
        response = await manager.generate_text(
            prompt="Create music pattern",
            max_tokens=200
        )
        
        assert response.success is True
        assert response.provider == LLMProvider.OPENAI
        assert "Generated music" in response.content
        assert response.tokens_used == 75
    
    @pytest.mark.asyncio
    async def test_generate_text_success_google(self, mock_settings):
        """Test successful text generation with Google Gemini."""
        manager = LLMManager(mock_settings)
        
        # Mock Google client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Gemini generated content"
        mock_client.generate_content.return_value = mock_response
        manager.google_client = mock_client
        manager.anthropic_client = None  # Force Google usage
        manager.openai_client = None
        
        response = await manager.generate_text(
            prompt="Create pattern",
            temperature=0.8
        )
        
        assert response.success is True
        assert response.provider == LLMProvider.GOOGLE
        assert "Gemini generated content" in response.content
    
    @pytest.mark.asyncio
    async def test_generate_text_fallback_chain(self, mock_settings):
        """Test fallback from primary to secondary providers."""
        manager = LLMManager(mock_settings)
        
        # Mock Anthropic failure
        mock_anthropic = Mock()
        mock_anthropic.messages.create.side_effect = Exception("API error")
        manager.anthropic_client = mock_anthropic
        
        # Mock OpenAI success
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Fallback success"))]
        mock_response.usage.total_tokens = 30
        mock_openai.chat.completions.create.return_value = mock_response
        manager.openai_client = mock_openai
        
        response = await manager.generate_text(prompt="Create pattern")
        
        assert response.success is True
        assert response.provider == LLMProvider.OPENAI
        assert "Fallback success" in response.content
    
    @pytest.mark.asyncio
    async def test_generate_text_all_providers_fail(self, mock_settings):
        """Test behavior when all providers fail."""
        manager = LLMManager(mock_settings)
        
        # Mock all providers to fail
        manager.anthropic_client = Mock()
        manager.anthropic_client.messages.create.side_effect = Exception("Anthropic error")
        
        manager.openai_client = Mock()
        manager.openai_client.chat.completions.create.side_effect = Exception("OpenAI error")
        
        manager.google_client = Mock()
        manager.google_client.generate_content.side_effect = Exception("Google error")
        
        response = await manager.generate_text(prompt="Create pattern")
        
        assert response.success is False
        assert "All LLM providers failed" in response.error
    
    @pytest.mark.asyncio
    async def test_preferred_provider_ordering(self, mock_settings):
        """Test that preferred provider is tried first."""
        manager = LLMManager(mock_settings)
        
        # Mock all providers available
        manager.anthropic_client = Mock()
        manager.openai_client = Mock()
        manager.google_client = Mock()
        
        # Mock OpenAI to succeed
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="OpenAI response"))]
        mock_response.usage.total_tokens = 40
        manager.openai_client.chat.completions.create.return_value = mock_response
        
        # Mock Anthropic to fail (it would normally be tried first)
        manager.anthropic_client.messages.create.side_effect = Exception("Anthropic error")
        
        response = await manager.generate_text(
            prompt="Create pattern",
            preferred_provider=LLMProvider.OPENAI
        )
        
        assert response.success is True
        assert response.provider == LLMProvider.OPENAI
    
    @pytest.mark.asyncio
    async def test_response_time_tracking(self, mock_settings):
        """Test that response times are tracked."""
        manager = LLMManager(mock_settings)
        
        # Mock successful response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.usage.output_tokens = 20
        mock_client.messages.create.return_value = mock_response
        manager.anthropic_client = mock_client
        
        response = await manager.generate_text(prompt="Test")
        
        assert response.success is True
        assert response.response_time is not None
        assert response.response_time > 0
    
    def test_is_available(self, mock_settings):
        """Test availability check."""
        manager = LLMManager(mock_settings)
        
        # No clients available
        assert manager.is_available() is False
        
        # Add one client
        manager.anthropic_client = Mock()
        assert manager.is_available() is True
    
    @pytest.mark.asyncio
    async def test_anthropic_specific_error_handling(self, mock_settings):
        """Test Anthropic-specific error handling."""
        manager = LLMManager(mock_settings)
        
        # Mock Anthropic client with specific error
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Rate limit exceeded")
        manager.anthropic_client = mock_client
        
        response = await manager._generate_anthropic(
            prompt="Test",
            system_prompt="System",
            max_tokens=100,
            temperature=0.7
        )
        
        assert response.success is False
        assert "Rate limit exceeded" in response.error
        assert response.provider == LLMProvider.ANTHROPIC
    
    @pytest.mark.asyncio
    async def test_openai_specific_error_handling(self, mock_settings):
        """Test OpenAI-specific error handling."""
        manager = LLMManager(mock_settings)
        
        # Mock OpenAI client with specific error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
        manager.openai_client = mock_client
        
        response = await manager._generate_openai(
            prompt="Test",
            system_prompt="System",
            max_tokens=100,
            temperature=0.7
        )
        
        assert response.success is False
        assert "Invalid API key" in response.error
        assert response.provider == LLMProvider.OPENAI
    
    @pytest.mark.asyncio
    async def test_google_specific_error_handling(self, mock_settings):
        """Test Google-specific error handling."""
        manager = LLMManager(mock_settings)
        
        # Mock Google client with specific error
        mock_client = Mock()
        mock_client.generate_content.side_effect = Exception("Quota exceeded")
        manager.google_client = mock_client
        
        response = await manager._generate_google(
            prompt="Test",
            system_prompt="System",
            max_tokens=100,
            temperature=0.7
        )
        
        assert response.success is False
        assert "Quota exceeded" in response.error
        assert response.provider == LLMProvider.GOOGLE
    
    @pytest.mark.asyncio
    async def test_parameter_passing(self, mock_settings):
        """Test that generation parameters are passed correctly."""
        manager = LLMManager(mock_settings)
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test")]
        mock_response.usage.output_tokens = 10
        mock_client.messages.create.return_value = mock_response
        manager.anthropic_client = mock_client
        
        await manager.generate_text(
            prompt="Test prompt",
            system_prompt="Test system",
            max_tokens=300,
            temperature=0.9
        )
        
        # Verify parameters were passed to client
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['max_tokens'] == 300
        assert call_args[1]['temperature'] == 0.9
        assert call_args[1]['system'] == "Test system"
        assert call_args[1]['messages'][0]['content'] == "Test prompt"


class TestLLMResponse:
    """Test LLMResponse data structure."""
    
    def test_response_creation(self):
        """Test creating LLMResponse objects."""
        response = LLMResponse(
            content="Test content",
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-sonnet",
            tokens_used=50,
            response_time=0.5
        )
        
        assert response.content == "Test content"
        assert response.provider == LLMProvider.ANTHROPIC
        assert response.model == "claude-3-5-sonnet"
        assert response.tokens_used == 50
        assert response.response_time == 0.5
        assert response.success is True  # Default
        assert response.error is None  # Default
    
    def test_error_response_creation(self):
        """Test creating error LLMResponse objects."""
        response = LLMResponse(
            content="",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            success=False,
            error="Network timeout"
        )
        
        assert response.content == ""
        assert response.success is False
        assert response.error == "Network timeout"