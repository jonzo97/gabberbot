#!/usr/bin/env python3
"""
Simplified Production Engine
Works without external dependencies for immediate functionality
"""

import asyncio
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..models.hardcore_models import HardcorePattern, SynthType
from ..interfaces.synthesizer import AbstractSynthesizer
from ..ai.local_conversation_engine import create_local_conversation_engine

@dataclass
class ProductionRequest:
    user_input: str
    session_id: str
    bpm: Optional[int] = None
    genre: Optional[str] = None
    style: Optional[str] = None

@dataclass
class ProductionResponse:
    success: bool
    message: str
    pattern: Optional[HardcorePattern] = None
    audio_generated: bool = False
    error: Optional[str] = None

class SimplifiedNaturalLanguageProcessor:
    """Simplified NLP without external dependencies"""
    
    def __init__(self):
        self.patterns = {
            'bpm': r'(\d{2,3})\s*bpm',
            'genre': r'(gabber|industrial|hardcore|techno)',
            'style': r'(brutal|dark|hard|aggressive|fast|slow)'
        }
    
    async def _parse_composition_requirements(self, user_input: str) -> Dict[str, Any]:
        """Parse requirements from user input"""
        input_lower = user_input.lower()
        requirements = {
            'bpm': 180,  # Default
            'genre': 'gabber',  # Default
            'style': 'brutal',  # Default
        }
        
        # Extract BPM
        bpm_match = re.search(self.patterns['bpm'], input_lower)
        if bpm_match:
            requirements['bpm'] = int(bpm_match.group(1))
        
        # Extract genre
        genre_match = re.search(self.patterns['genre'], input_lower)
        if genre_match:
            requirements['genre'] = genre_match.group(1)
        
        # Extract style
        style_match = re.search(self.patterns['style'], input_lower)
        if style_match:
            requirements['style'] = style_match.group(1)
        
        return requirements

class ConversationalProductionEngine:
    """Simplified production engine using local conversation engine"""
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.synthesizer = synthesizer
        self.conversation_engine = create_local_conversation_engine(synthesizer)
        self.nlp = SimplifiedNaturalLanguageProcessor()
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    async def process_request(self, user_input: str, session_id: str) -> ProductionResponse:
        """Process a production request"""
        try:
            # Initialize session if needed
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'current_pattern': None,
                    'conversation_count': 0
                }
            
            session = self.sessions[session_id]
            session['conversation_count'] += 1
            
            # Process through conversation engine
            response = await self.conversation_engine.process_message(user_input, session_id)
            
            # Update session with any new pattern
            if response.pattern:
                session['current_pattern'] = response.pattern
            
            # Try to generate audio if we have a pattern
            audio_generated = False
            if response.pattern:
                try:
                    audio = await self.synthesizer.play_pattern(response.pattern)
                    audio_generated = audio is not None
                except Exception as e:
                    # Non-critical error
                    pass
            
            return ProductionResponse(
                success=response.success,
                message=response.response_text,
                pattern=response.pattern,
                audio_generated=audio_generated
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message="Production engine error occurred",
                error=str(e)
            )
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        session = self.sessions.get(session_id, {})
        return {
            'session_id': session_id,
            'conversation_count': session.get('conversation_count', 0),
            'has_pattern': session.get('current_pattern') is not None,
            'pattern_name': session.get('current_pattern', {}).get('name') if session.get('current_pattern') else None
        }

# Factory function
def create_conversational_production_engine(synthesizer: AbstractSynthesizer) -> ConversationalProductionEngine:
    """Create a conversational production engine"""
    return ConversationalProductionEngine(synthesizer)

# Example usage and testing
if __name__ == "__main__":
    async def test_production_engine():
        print("🎵 Testing Simplified Production Engine 🎵")
        print("=" * 50)
        
        from ..interfaces.synthesizer import MockSynthesizer
        
        synth = MockSynthesizer()
        await synth.start()
        
        engine = create_conversational_production_engine(synth)
        
        test_requests = [
            "Make a brutal gabber kick at 180 BPM",
            "Create industrial pattern at 140 BPM", 
            "Make it harder and more aggressive",
            "Save as warehouse_destroyer"
        ]
        
        session_id = "test_session"
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n{i}. User: {request}")
            response = await engine.process_request(request, session_id)
            print(f"   Success: {response.success}")
            print(f"   Message: {response.message}")
            if response.pattern:
                print(f"   Pattern: {response.pattern.name} ({response.pattern.genre} @ {response.pattern.bpm} BPM)")
            print(f"   Audio Generated: {response.audio_generated}")
        
        # Check session info
        session_info = engine.get_session_info(session_id)
        print(f"\n📊 Session Info: {session_info}")
        
        print("\n✨ Simplified Production Engine fully operational!")
    
    asyncio.run(test_production_engine())