"""
Multi-LLM Client Manager with Fallback Support.

Provides a unified interface for multiple LLM providers with automatic
fallback handling and error recovery.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import time

from ..models.config import Settings

# Optional imports for LLM providers
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers in order of preference."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class LLMManager:
    """
    Unified LLM client manager with fallback support.
    
    Handles multiple LLM providers with automatic fallback when primary
    providers fail. Implements retry logic and error handling.
    """
    
    def __init__(self, settings: Settings):
        """Initialize LLM manager with configuration."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.anthropic_client = None
        self.openai_client = None
        self.google_client = None
        
        # Fallback order (can be configured)
        self.provider_order = [
            LLMProvider.ANTHROPIC,
            LLMProvider.OPENAI, 
            LLMProvider.GOOGLE
        ]
        
        self._init_clients()
    
    def _init_clients(self) -> None:
        """Initialize all available LLM clients."""
        # Anthropic Claude (Primary for hardcore music)
        if ANTHROPIC_AVAILABLE and self.settings.ai.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.settings.ai.anthropic_api_key
                )
                self.logger.info("Anthropic Claude client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # OpenAI GPT (Secondary)
        if OPENAI_AVAILABLE and self.settings.ai.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(
                    api_key=self.settings.ai.openai_api_key
                )
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Google Gemini (Tertiary)
        if GOOGLE_AI_AVAILABLE and self.settings.ai.google_api_key:
            try:
                genai.configure(api_key=self.settings.ai.google_api_key)
                self.google_client = genai.GenerativeModel('gemini-pro')
                self.logger.info("Google Gemini client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google client: {e}")
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of currently available providers."""
        available = []
        if self.anthropic_client:
            available.append(LLMProvider.ANTHROPIC)
        if self.openai_client:
            available.append(LLMProvider.OPENAI)
        if self.google_client:
            available.append(LLMProvider.GOOGLE)
        return available
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.7,
        preferred_provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """
        Generate text using the best available LLM provider.
        
        Args:
            prompt: User prompt for generation
            system_prompt: System/instruction prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-1.0)
            preferred_provider: Preferred provider to try first
            
        Returns:
            LLMResponse with generated content and metadata
        """
        # Determine provider order
        providers_to_try = self.provider_order.copy()
        if preferred_provider and preferred_provider in providers_to_try:
            # Move preferred provider to front
            providers_to_try.remove(preferred_provider)
            providers_to_try.insert(0, preferred_provider)
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                start_time = time.time()
                response = await self._generate_with_provider(
                    provider, prompt, system_prompt, max_tokens, temperature
                )
                response.response_time = time.time() - start_time
                
                if response.success:
                    self.logger.info(f"Successfully generated text with {provider.value}")
                    return response
                else:
                    last_error = response.error
                    
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Provider {provider.value} failed: {e}")
                continue
        
        # All providers failed
        return LLMResponse(
            content="",
            provider=LLMProvider.ANTHROPIC,  # Default
            model="fallback",
            success=False,
            error=f"All LLM providers failed. Last error: {last_error}"
        )
    
    async def _generate_with_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Generate text with a specific provider."""
        
        if provider == LLMProvider.ANTHROPIC and self.anthropic_client:
            return await self._generate_anthropic(
                prompt, system_prompt, max_tokens, temperature
            )
        elif provider == LLMProvider.OPENAI and self.openai_client:
            return await self._generate_openai(
                prompt, system_prompt, max_tokens, temperature
            )
        elif provider == LLMProvider.GOOGLE and self.google_client:
            return await self._generate_google(
                prompt, system_prompt, max_tokens, temperature
            )
        else:
            raise ValueError(f"Provider {provider.value} not available")
    
    async def _generate_anthropic(
        self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Generate text using Anthropic Claude."""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.anthropic_client.messages.create(**kwargs)
            
            return LLMResponse(
                content=response.content[0].text,
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-5-sonnet-20241022",
                tokens_used=response.usage.output_tokens,
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-5-sonnet-20241022",
                success=False,
                error=str(e)
            )
    
    async def _generate_openai(
        self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Generate text using OpenAI GPT."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                tokens_used=response.usage.total_tokens,
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                success=False,
                error=str(e)
            )
    
    async def _generate_google(
        self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Generate text using Google Gemini."""
        try:
            # Combine system prompt and user prompt for Gemini
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            response = self.google_client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return LLMResponse(
                content=response.text,
                provider=LLMProvider.GOOGLE,
                model="gemini-pro",
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                provider=LLMProvider.GOOGLE,
                model="gemini-pro",
                success=False,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return len(self.get_available_providers()) > 0