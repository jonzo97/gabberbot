"""
AI Generation Service for Natural Language to MIDI Conversion.

Core service that takes natural language prompts and generates MIDIClip objects
using LLMs with fallback to algorithmic generation.
"""

import json
import re
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import asdict

from ..models.core import MIDIClip, MIDINote
from ..models.config import Settings
from ..utils.music_parser import MusicParser, MusicalParameters, PatternType, Genre
from .llm_manager import LLMManager, LLMResponse
from .prompt_builder import PromptBuilder, PromptTemplate


class GenerationError(Exception):
    """Raised when generation fails across all methods."""
    pass


class GenerationService:
    """
    AI-powered service for converting natural language to MIDI clips.
    
    Consolidates all AI generation logic with multi-LLM support and
    fallback to algorithmic generation when AI fails.
    """
    
    def __init__(self, settings: Settings):
        """Initialize generation service with configuration."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.music_parser = MusicParser()
        self.llm_manager = LLMManager(settings)
        self.prompt_builder = PromptBuilder()
        
        # Performance tracking
        self.generation_stats = {
            'total_requests': 0,
            'successful_ai': 0,
            'successful_fallback': 0,
            'failures': 0,
            'avg_response_time': 0.0
        }
    
    async def text_to_midi(
        self,
        prompt: str,
        user_id: Optional[str] = None,
        fallback_to_algorithmic: bool = True
    ) -> MIDIClip:
        """
        Convert natural language prompt to MIDIClip.
        
        This is the main entry point for AI music generation.
        
        Args:
            prompt: Natural language description of desired music
            user_id: Optional user ID for personalization
            fallback_to_algorithmic: Whether to fallback to algorithmic generation
            
        Returns:
            MIDIClip with generated musical content
            
        Raises:
            GenerationError: If all generation methods fail
        """
        import time
        start_time = time.time()
        self.generation_stats['total_requests'] += 1
        
        try:
            self.logger.info(f"Generating MIDI for prompt: '{prompt}'")
            
            # Parse musical parameters from text
            params = self.music_parser.parse(prompt)
            self.logger.debug(f"Parsed parameters: {asdict(params)}")
            
            # Validate parameters
            is_valid, errors = self.music_parser.validate_parameters(params)
            if not is_valid:
                self.logger.warning(f"Parameter validation issues: {errors}")
                # Continue with warnings, don't fail completely
            
            # Try AI generation first
            try:
                clip = await self._generate_with_ai(prompt, params)
                if clip:
                    self.generation_stats['successful_ai'] += 1
                    self.logger.info("Successfully generated with AI")
                    return clip
            except Exception as e:
                self.logger.warning(f"AI generation failed: {e}")
            
            # Fallback to algorithmic generation
            if fallback_to_algorithmic:
                try:
                    clip = self._generate_with_algorithm(prompt, params)
                    if clip:
                        self.generation_stats['successful_fallback'] += 1
                        self.logger.info("Successfully generated with algorithmic fallback")
                        return clip
                except Exception as e:
                    self.logger.error(f"Algorithmic generation failed: {e}")
            
            # All methods failed
            self.generation_stats['failures'] += 1
            raise GenerationError(f"Failed to generate MIDI for prompt: '{prompt}'")
            
        finally:
            # Update performance stats
            response_time = time.time() - start_time
            self._update_performance_stats(response_time)
    
    async def _generate_with_ai(self, prompt: str, params: MusicalParameters) -> Optional[MIDIClip]:
        """Generate MIDI using LLM AI services."""
        if not self.llm_manager.is_available():
            self.logger.warning("No LLM providers available")
            return None
        
        # Build appropriate prompt based on pattern type
        template = self.prompt_builder.get_template_for_pattern(params.pattern_type)
        prompts = self.prompt_builder.build_generation_prompt(params, prompt, template)
        
        # Generate with LLM
        response = await self.llm_manager.generate_text(
            prompt=prompts['user'],
            system_prompt=prompts['system'],
            max_tokens=500,
            temperature=0.7
        )
        
        if not response.success:
            self.logger.error(f"LLM generation failed: {response.error}")
            return None
        
        # Parse LLM response to create MIDI clip
        return self._parse_llm_response(response.content, params)
    
    def _generate_with_algorithm(self, prompt: str, params: MusicalParameters) -> Optional[MIDIClip]:
        """Generate MIDI using algorithmic methods as fallback."""
        self.logger.info("Using algorithmic generation as fallback")
        
        # Apply defaults for missing parameters
        if not params.bpm:
            params.bpm = 180  # Default hardcore BPM
        if not params.key:
            params.key = "Am"  # Default minor key
        if not params.pattern_type:
            params.pattern_type = PatternType.KICK_PATTERN  # Safe default
        
        # Generate based on pattern type
        if params.pattern_type == PatternType.ACID_BASSLINE:
            return self._generate_acid_bassline(params)
        elif params.pattern_type == PatternType.KICK_PATTERN:
            return self._generate_kick_pattern(params)
        elif params.pattern_type == PatternType.RIFF:
            return self._generate_riff(params)
        else:
            # Default to simple pattern
            return self._generate_simple_pattern(params)
    
    def _parse_llm_response(self, response_text: str, params: MusicalParameters) -> Optional[MIDIClip]:
        """Parse LLM response text to extract MIDI data and create clip."""
        try:
            # Look for MIDI note sequences in the response
            notes = self._extract_notes_from_text(response_text)
            rhythms = self._extract_rhythms_from_text(response_text)
            velocities = self._extract_velocities_from_text(response_text)
            
            if not notes:
                self.logger.warning("No MIDI notes found in LLM response")
                return None
            
            # Create MIDIClip
            clip = MIDIClip(
                id=f"ai_generated_{int(time.time() * 1000)}",
                name=f"AI Generated {params.pattern_type.value if params.pattern_type else 'Pattern'}",
                length_bars=params.length_bars,
                bpm=params.bpm or 180,
                key=params.key
            )
            
            # Add notes to clip
            current_time = 0.0
            for i, note_num in enumerate(notes):
                duration = rhythms[i] if i < len(rhythms) else 0.25
                velocity = velocities[i] if i < len(velocities) else 100
                
                # Ensure valid MIDI values
                note_num = max(0, min(127, note_num))
                velocity = max(1, min(127, velocity))
                
                midi_note = MIDINote(
                    pitch=note_num,
                    velocity=velocity,
                    start_time=current_time,
                    duration=duration
                )
                clip.add_note(midi_note)
                current_time += duration
            
            return clip
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return None
    
    def _extract_notes_from_text(self, text: str) -> List[int]:
        """Extract MIDI note numbers from text response."""
        # Look for patterns like "60, 63, 65" or "C4, Eb4, F4"
        notes = []
        
        # Pattern 1: Direct MIDI numbers
        midi_pattern = r'(?:notes?|midi).*?:?\s*([0-9, ]+)'
        matches = re.findall(midi_pattern, text, re.IGNORECASE)
        for match in matches:
            note_nums = re.findall(r'\d+', match)
            notes.extend([int(n) for n in note_nums if 0 <= int(n) <= 127])
        
        # Pattern 2: Note names (simplified)
        note_names = {
            'c': 60, 'd': 62, 'e': 64, 'f': 65, 'g': 67, 'a': 69, 'b': 71
        }
        note_pattern = r'\b([CDEFGAB][#b]?\d?)\b'
        note_matches = re.findall(note_pattern, text, re.IGNORECASE)
        for note_match in note_matches:
            base_note = note_match[0].lower()
            if base_note in note_names:
                notes.append(note_names[base_note])
        
        # If no notes found, generate a simple default sequence
        if not notes:
            # Generate a simple Am pattern
            notes = [57, 60, 63, 65]  # A3, C4, Eb4, F4
        
        return notes[:16]  # Limit to reasonable length
    
    def _extract_rhythms_from_text(self, text: str) -> List[float]:
        """Extract rhythm durations from text response."""
        # Look for duration patterns
        rhythms = []
        
        duration_pattern = r'(?:duration|rhythm|timing).*?:?\s*([0-9., ]+)'
        matches = re.findall(duration_pattern, text, re.IGNORECASE)
        for match in matches:
            # Improved regex to properly match decimal numbers (not trailing periods)
            durations = re.findall(r'\d+(?:\.\d+)?', match)
            for d in durations:
                try:
                    val = float(d)
                    if 0.0625 <= val <= 4.0:
                        rhythms.append(val)
                except ValueError:
                    continue
        
        # Default to quarter notes if no rhythm found
        if not rhythms:
            rhythms = [0.25] * 8  # 8 quarter notes
        
        return rhythms
    
    def _extract_velocities_from_text(self, text: str) -> List[int]:
        """Extract velocity values from text response."""
        velocities = []
        
        velocity_pattern = r'(?:velocity|velocities).*?:?\s*([0-9, ]+)'
        matches = re.findall(velocity_pattern, text, re.IGNORECASE)
        for match in matches:
            vel_nums = re.findall(r'\d+', match)
            velocities.extend([int(v) for v in vel_nums if 1 <= int(v) <= 127])
        
        # Default to strong velocities for hardcore
        if not velocities:
            velocities = [100] * 8  # Strong but not max
        
        return velocities
    
    def _generate_acid_bassline(self, params: MusicalParameters) -> MIDIClip:
        """Generate algorithmic acid bassline."""
        clip = MIDIClip(
            id=f"acid_bassline_{int(time.time() * 1000)}",
            name="Algorithmic Acid Bassline",
            length_bars=params.length_bars,
            bpm=params.bpm or 160,
            key=params.key or "Am"
        )
        
        # Simple acid pattern in Am
        # A2=45, C3=48, E3=52, A3=57
        pattern = [45, 48, 45, 52, 45, 48, 57, 45]  # Classic acid progression
        durations = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25]
        velocities = [100, 80, 90, 100, 85, 75, 110, 95]  # Accent pattern
        
        current_time = 0.0
        for i, (note, duration, velocity) in enumerate(zip(pattern, durations, velocities)):
            midi_note = MIDINote(
                pitch=note,
                velocity=velocity,
                start_time=current_time,
                duration=duration
            )
            clip.add_note(midi_note)
            current_time += duration
        
        return clip
    
    def _generate_kick_pattern(self, params: MusicalParameters) -> MIDIClip:
        """Generate algorithmic kick pattern."""
        clip = MIDIClip(
            id=f"kick_pattern_{int(time.time() * 1000)}",
            name="Algorithmic Kick Pattern",
            length_bars=params.length_bars,
            bpm=params.bpm or 180,
            key=params.key
        )
        
        # Classic 4-on-the-floor hardcore kick
        kick_note = 36  # C2 - standard kick
        pattern = [0.0, 1.0, 2.0, 3.0]  # Every beat
        
        for beat in pattern:
            midi_note = MIDINote(
                pitch=kick_note,
                velocity=127,  # Max velocity for hardcore power
                start_time=beat,
                duration=0.1  # Short, punchy
            )
            clip.add_note(midi_note)
        
        return clip
    
    def _generate_riff(self, params: MusicalParameters) -> MIDIClip:
        """Generate algorithmic riff."""
        clip = MIDIClip(
            id=f"riff_{int(time.time() * 1000)}",
            name="Algorithmic Riff",
            length_bars=params.length_bars,
            bpm=params.bpm or 160,
            key=params.key or "Am"
        )
        
        # Simple minor riff
        pattern = [57, 60, 63, 60, 57, 55, 57, 60]  # A-C-Eb-C-A-G-A-C
        durations = [0.5] * 8
        velocities = [90, 100, 110, 95, 85, 80, 90, 100]
        
        current_time = 0.0
        for note, duration, velocity in zip(pattern, durations, velocities):
            midi_note = MIDINote(
                pitch=note,
                velocity=velocity,
                start_time=current_time,
                duration=duration
            )
            clip.add_note(midi_note)
            current_time += duration
        
        return clip
    
    def _generate_simple_pattern(self, params: MusicalParameters) -> MIDIClip:
        """Generate simple fallback pattern."""
        clip = MIDIClip(
            id=f"simple_pattern_{int(time.time() * 1000)}",
            name="Simple Pattern",
            length_bars=params.length_bars,
            bpm=params.bpm or 160,
            key=params.key or "Am"
        )
        
        # Very simple pattern - just root note
        root_note = 57  # A3
        midi_note = MIDINote(
            pitch=root_note,
            velocity=100,
            start_time=0.0,
            duration=1.0
        )
        clip.add_note(midi_note)
        
        return clip
    
    def _update_performance_stats(self, response_time: float) -> None:
        """Update internal performance statistics."""
        total = self.generation_stats['total_requests']
        current_avg = self.generation_stats['avg_response_time']
        
        # Update running average (handle first request case)
        if total > 0:
            self.generation_stats['avg_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )
        else:
            # First request, set initial average
            self.generation_stats['avg_response_time'] = response_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.generation_stats.copy()
    
    def is_available(self) -> bool:
        """Check if generation service is available."""
        return True  # Always available due to algorithmic fallback