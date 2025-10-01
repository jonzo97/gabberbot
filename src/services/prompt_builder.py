"""
Prompt Engineering System for LLM-based Music Generation.

Creates specialized prompts for different musical contexts and LLM providers,
optimized for hardcore/industrial music generation.
"""

from typing import Optional, Dict, Any
from enum import Enum

from ..utils.music_parser import MusicalParameters, Genre, PatternType


class PromptTemplate(Enum):
    """Available prompt templates for different generation tasks."""
    PATTERN_GENERATOR = "pattern_generator"
    BASSLINE_GENERATOR = "bassline_generator"
    KICK_GENERATOR = "kick_generator"
    RIFF_GENERATOR = "riff_generator"
    MUSIC_ANALYZER = "music_analyzer"


class PromptBuilder:
    """
    Builds optimized prompts for music generation using LLMs.
    
    Creates context-aware prompts that incorporate hardcore music knowledge,
    user parameters, and genre-specific requirements.
    """
    
    def __init__(self):
        """Initialize prompt builder with hardcore music knowledge."""
        self.hardcore_knowledge = self._load_hardcore_knowledge()
        self.system_prompts = self._create_system_prompts()
    
    def build_generation_prompt(
        self,
        params: MusicalParameters,
        user_prompt: str,
        template: PromptTemplate = PromptTemplate.PATTERN_GENERATOR
    ) -> Dict[str, str]:
        """
        Build a complete prompt for music generation.
        
        Args:
            params: Extracted musical parameters
            user_prompt: Original user request
            template: Prompt template to use
            
        Returns:
            Dict with 'system' and 'user' prompts
        """
        # Get appropriate system prompt
        system_prompt = self.system_prompts.get(template.value, "")
        
        # Build user prompt based on template
        if template == PromptTemplate.PATTERN_GENERATOR:
            user_prompt_text = self._build_pattern_prompt(params, user_prompt)
        elif template == PromptTemplate.BASSLINE_GENERATOR:
            user_prompt_text = self._build_bassline_prompt(params, user_prompt)
        elif template == PromptTemplate.KICK_GENERATOR:
            user_prompt_text = self._build_kick_prompt(params, user_prompt)
        elif template == PromptTemplate.RIFF_GENERATOR:
            user_prompt_text = self._build_riff_prompt(params, user_prompt)
        else:
            user_prompt_text = self._build_generic_prompt(params, user_prompt)
        
        return {
            'system': system_prompt,
            'user': user_prompt_text
        }
    
    def _build_pattern_prompt(self, params: MusicalParameters, user_prompt: str) -> str:
        """Build prompt for general pattern generation."""
        prompt_parts = [
            f"Original request: {user_prompt}",
            "",
            "Generate a hardcore music pattern with these specifications:",
        ]
        
        # Add extracted parameters
        if params.bpm:
            prompt_parts.append(f"- BPM: {params.bpm}")
        if params.key:
            prompt_parts.append(f"- Key: {params.key}")
        if params.genre:
            prompt_parts.append(f"- Genre: {params.genre.value}")
            # Add genre-specific knowledge
            genre_info = self.hardcore_knowledge.get(params.genre.value, {})
            if genre_info:
                prompt_parts.append(f"- Genre characteristics: {genre_info.get('description', '')}")
        if params.pattern_type:
            prompt_parts.append(f"- Pattern type: {params.pattern_type.value}")
        if params.style_descriptors:
            prompt_parts.append(f"- Style: {', '.join(params.style_descriptors)}")
        if params.mood:
            prompt_parts.append(f"- Mood: {params.mood}")
        
        prompt_parts.extend([
            "",
            "Provide the output as:",
            "1. MIDI note sequence (note numbers 0-127)",
            "2. Rhythm pattern (note durations in beats)",
            "3. Velocity values (0-127, prefer 80-127 for hardcore)",
            "4. Synthesis suggestions",
            "",
            "Focus on authentic hardcore music characteristics:",
            "- Aggressive, industrial sound",
            "- Minor keys for darkness",
            "- High energy and intensity",
            "- Proper hardcore timing and groove"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_bassline_prompt(self, params: MusicalParameters, user_prompt: str) -> str:
        """Build prompt specifically for acid bassline generation."""
        prompt_parts = [
            f"Create an acid bassline for: {user_prompt}",
            "",
            "Specifications:",
        ]
        
        if params.bpm:
            prompt_parts.append(f"- BPM: {params.bpm}")
        if params.key:
            prompt_parts.append(f"- Key: {params.key}")
        
        prompt_parts.extend([
            f"- Length: {params.length_bars} bars",
            "",
            "Acid bassline characteristics:",
            "- TB-303 style patterns",
            "- Sliding between notes (glides)",
            "- Accented notes for groove",
            "- Octave jumps for excitement",
            "- Minor scales for hardcore darkness",
            "",
            "Output format:",
            "- Note sequence with slides marked",
            "- Accent pattern (which notes get emphasis)",
            "- Suggested filter cutoff automation",
            "- Resonance settings for that squelchy acid sound"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_kick_prompt(self, params: MusicalParameters, user_prompt: str) -> str:
        """Build prompt for kick drum pattern generation."""
        prompt_parts = [
            f"Create a hardcore kick pattern for: {user_prompt}",
            "",
            "Specifications:",
        ]
        
        if params.bpm:
            prompt_parts.append(f"- BPM: {params.bpm}")
        if params.genre:
            genre_name = params.genre.value
            prompt_parts.append(f"- Genre: {genre_name}")
            
            # Add genre-specific kick characteristics
            if genre_name == "gabber":
                prompt_parts.append("- Gabber style: Extreme distortion, 'doorlussen' technique")
            elif genre_name == "hardcore":
                prompt_parts.append("- Hardcore style: Heavy compression, punchy transients")
            elif genre_name == "industrial":
                prompt_parts.append("- Industrial style: Berlin rumble, metallic reverb")
        
        prompt_parts.extend([
            "",
            "Hardcore kick requirements:",
            "- CRUNCHY and AGGRESSIVE (non-negotiable)",
            "- Proper hardcore timing patterns",
            "- 4-on-the-floor or hardcore variations",
            "- High impact, warehouse-destroying power",
            "",
            "Output format:",
            "- Kick timing pattern (beats in 4/4 bars)",
            "- Velocity levels (prefer 100-127)",
            "- Synthesis parameters for crunch",
            "- Processing chain suggestions"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_riff_prompt(self, params: MusicalParameters, user_prompt: str) -> str:
        """Build prompt for melodic riff generation."""
        prompt_parts = [
            f"Create a hardcore riff for: {user_prompt}",
            "",
            "Musical parameters:",
        ]
        
        if params.bpm:
            prompt_parts.append(f"- BPM: {params.bpm}")
        if params.key:
            prompt_parts.append(f"- Key: {params.key}")
        if params.complexity:
            prompt_parts.append(f"- Complexity: {params.complexity}")
        
        prompt_parts.extend([
            "",
            "Riff characteristics for hardcore:",
            "- Industrial, aggressive melodies",
            "- Minor keys for dark atmosphere",
            "- Memorable but not overly complex",
            "- Suitable for harsh synthesis",
            "",
            "Output:",
            "- MIDI note sequence",
            "- Rhythm pattern with timing",
            "- Suggested octave range",
            "- Synth type recommendations (leads, pads, etc.)"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_generic_prompt(self, params: MusicalParameters, user_prompt: str) -> str:
        """Build generic prompt for unspecified generation tasks."""
        return f"""
Generate hardcore music content for: {user_prompt}

Parameters: BPM={params.bpm}, Key={params.key}, Genre={params.genre}, Type={params.pattern_type}

Create appropriate hardcore/industrial music following these principles:
- Aggressive, industrial aesthetic
- Minor keys for darkness
- High energy and intensity
- Authentic hardcore characteristics

Provide specific musical data (notes, timing, synthesis suggestions).
"""
    
    def _create_system_prompts(self) -> Dict[str, str]:
        """Create system prompts for different generation contexts."""
        return {
            'pattern_generator': """You are a hardcore music producer specializing in gabber, industrial, and hardcore techno. You create aggressive, dark electronic music with crunchy kickdrums and industrial aesthetics.

Your expertise includes:
- Authentic hardcore/gabber production (150-250 BPM)
- Industrial techno and warehouse sounds
- TB-303 acid basslines and Rotterdam gabber techniques
- Aggressive synthesis and heavy processing
- Minor keys and dark harmonic content

Always prioritize the hardcore aesthetic: CRUNCHY KICKDRUMS, aggressive sound design, industrial atmosphere, and warehouse-destroying power. Never suggest "softening" unless explicitly requested.

Respond with specific musical data including MIDI notes, timing, and synthesis parameters.""",

            'bassline_generator': """You are an acid bassline specialist focusing on TB-303 style patterns for hardcore and gabber music. You understand the Rotterdam acid scene and create squelchy, aggressive basslines.

Expertise:
- TB-303 programming and acid house traditions
- Sliding basslines with proper glide/portamento
- Accent patterns for groove and pump
- Filter cutoff and resonance automation
- Integration with hardcore kick patterns

Create basslines that are dark, aggressive, and perfectly suited for hardcore dancefloors. Focus on minor scales and industrial atmosphere.""",

            'kick_generator': """You are a hardcore kick drum specialist creating the most important element of hardcore music: CRUNCHY, AGGRESSIVE kicks that destroy sound systems.

Your knowledge covers:
- Gabber "doorlussen" distortion techniques
- Industrial Berlin rumble and sub-bass design
- Hardcore compression and transient shaping
- 4-on-the-floor and hardcore rhythm variations
- TR-909 analog kick synthesis

Remember: The kick is the soul of hardcore music. It must be AGGRESSIVE, CRUNCHY, and POWERFUL. No weak or toy-like sounds allowed.""",

            'riff_generator': """You are a hardcore melody and riff specialist creating dark, industrial musical phrases for electronic hardcore music.

Your focus:
- Minor key melodies for dark atmosphere
- Industrial and aggressive harmonic content
- Memorable but not overly complex structures
- Compatibility with harsh synthesis and distortion
- Hardcore music arrangement principles

Create riffs that complement the aggressive hardcore aesthetic while maintaining musical interest and dancefloor impact."""
        }
    
    def _load_hardcore_knowledge(self) -> Dict[str, Any]:
        """Load hardcore music genre knowledge."""
        return {
            'gabber': {
                'description': 'Extreme kick distortion, Rotterdam style, 150-200 BPM',
                'characteristics': ['doorlussen technique', 'extreme distortion', 'analog kicks'],
                'bpm_range': (150, 200),
                'key_signatures': ['Am', 'Em', 'Cm']
            },
            'hardcore': {
                'description': 'Heavy compression, hoover sounds, complex breakbeats, 180-250 BPM',
                'characteristics': ['heavy compression', 'hoover synths', 'complex patterns'],
                'bpm_range': (180, 250),
                'key_signatures': ['Am', 'Dm', 'Gm']
            },
            'industrial': {
                'description': 'Berlin rumble kicks, metallic reverb, minimal arrangements, 130-150 BPM',
                'characteristics': ['rumble bass', 'metallic textures', 'warehouse atmosphere'],
                'bpm_range': (130, 150),
                'key_signatures': ['Am', 'Em', 'Fm']
            },
            'frenchcore': {
                'description': 'Fast hardcore with French flair, 200-250 BPM',
                'characteristics': ['fast kicks', 'melodic elements', 'uplifting'],
                'bpm_range': (200, 250),
                'key_signatures': ['Am', 'Dm', 'Em']
            },
            'acid': {
                'description': 'TB-303 basslines, squelchy filters, sliding notes',
                'characteristics': ['303 basslines', 'filter sweeps', 'slides'],
                'bpm_range': (150, 180),
                'key_signatures': ['Am', 'Em', 'Cm']
            }
        }
    
    def get_template_for_pattern(self, pattern_type: Optional[PatternType]) -> PromptTemplate:
        """Get the most appropriate prompt template for a pattern type."""
        if not pattern_type:
            return PromptTemplate.PATTERN_GENERATOR
        
        mapping = {
            PatternType.ACID_BASSLINE: PromptTemplate.BASSLINE_GENERATOR,
            PatternType.KICK_PATTERN: PromptTemplate.KICK_GENERATOR,
            PatternType.RIFF: PromptTemplate.RIFF_GENERATOR,
            PatternType.ARPEGGIO: PromptTemplate.RIFF_GENERATOR,
            PatternType.CHORD_PROGRESSION: PromptTemplate.RIFF_GENERATOR,
            PatternType.DRUM_PATTERN: PromptTemplate.KICK_GENERATOR,
            PatternType.HIHAT_PATTERN: PromptTemplate.PATTERN_GENERATOR,
        }
        
        return mapping.get(pattern_type, PromptTemplate.PATTERN_GENERATOR)