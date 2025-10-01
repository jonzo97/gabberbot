"""
AI Generation Services for Hardcore Music Production.

Consolidated services for natural language to MIDI generation.
"""

from .generation_service import GenerationService
from .llm_manager import LLMManager
from .prompt_builder import PromptBuilder
from .audio_service import AudioService, AudioConfig

__all__ = ['GenerationService', 'LLMManager', 'PromptBuilder', 'AudioService', 'AudioConfig']