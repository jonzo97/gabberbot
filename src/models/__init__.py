"""
Data models for the Hardcore Music Production System.

This package contains Pydantic models for all musical data structures
following the Architecture Specification patterns.
"""

from .core import MIDINote, MIDIClip, AutomationPoint, AutomationClip
from .config import Settings, AudioConfig, AIConfig

__all__ = [
    "MIDINote",
    "MIDIClip", 
    "AutomationPoint",
    "AutomationClip",
    "Settings",
    "AudioConfig", 
    "AIConfig",
]