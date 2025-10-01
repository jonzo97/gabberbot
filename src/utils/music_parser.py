"""
Natural Language Music Parameter Parser.

Extracts musical parameters (BPM, key, style, etc.) from natural language prompts
for hardcore, gabber, and industrial music generation.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Hardcore music knowledge from CLAUDE.md
HARDCORE_PARAMS = {
    'bpm_ranges': {
        'gabber': (150, 200),
        'hardcore': (180, 250), 
        'industrial': (130, 150),
        'frenchcore': (200, 250),
        'uptempo': (160, 200),
        'speedcore': (250, 300)
    },
    'preferred_keys': ['Am', 'Em', 'Cm', 'Dm', 'Gm', 'Fm', 'Bm'],  # Minor keys for darkness
    'pattern_types': ['acid_bassline', 'kick_pattern', 'riff', 'arpeggio', 'chord_progression'],
    'genres': ['gabber', 'hardcore', 'industrial', 'frenchcore', 'uptempo', 'speedcore', 'acid']
}


class PatternType(Enum):
    """Types of musical patterns that can be generated."""
    ACID_BASSLINE = "acid_bassline"
    KICK_PATTERN = "kick_pattern"
    RIFF = "riff"
    ARPEGGIO = "arpeggio"
    CHORD_PROGRESSION = "chord_progression"
    DRUM_PATTERN = "drum_pattern"
    HIHAT_PATTERN = "hihat_pattern"


class Genre(Enum):
    """Supported hardcore music genres."""
    GABBER = "gabber"
    HARDCORE = "hardcore"
    INDUSTRIAL = "industrial"
    FRENCHCORE = "frenchcore"
    UPTEMPO = "uptempo"
    SPEEDCORE = "speedcore"
    ACID = "acid"


@dataclass
class MusicalParameters:
    """Extracted musical parameters from natural language."""
    bpm: Optional[int] = None
    key: Optional[str] = None
    genre: Optional[Genre] = None
    pattern_type: Optional[PatternType] = None
    style_descriptors: List[str] = None
    length_bars: float = 4.0
    instruments: List[str] = None
    effects: List[str] = None
    mood: Optional[str] = None
    complexity: str = "medium"  # simple, medium, complex
    
    def __post_init__(self):
        if self.style_descriptors is None:
            self.style_descriptors = []
        if self.instruments is None:
            self.instruments = []
        if self.effects is None:
            self.effects = []


class MusicParser:
    """
    Natural language parser for musical parameters.
    
    Extracts BPM, keys, genres, pattern types, and other musical elements
    from text prompts with focus on hardcore/industrial music.
    """
    
    def __init__(self):
        """Initialize parser with hardcore music patterns."""
        # BPM extraction patterns
        self.bpm_patterns = [
            r'(\d{1,3})\s*bpm',
            r'(\d{1,3})\s*beats?\s*per\s*minute',
            r'at\s+(\d{1,3})',
            r'tempo\s+(\d{1,3})',
        ]
        
        # Key extraction patterns
        self.key_patterns = [
            r'\b([ABCDEFG][#b]?)\s*(?:major|minor|m|maj|min)',
            r'in\s+([ABCDEFG][#b]?)',
            r'key\s+of\s+([ABCDEFG][#b]?)',
        ]
        
        # Genre patterns
        self.genre_patterns = {
            'gabber': r'\b(?:gabber|rotterdam|doorlussen)\b',
            'hardcore': r'\b(?:hardcore|hc|hard\s*core)\b',
            'industrial': r'\b(?:industrial|berlin|warehouse|rumble)\b',
            'frenchcore': r'\b(?:frenchcore|french\s*core)\b',
            'uptempo': r'\b(?:uptempo|up\s*tempo)\b',
            'speedcore': r'\b(?:speedcore|speed\s*core)\b',
            'acid': r'\b(?:acid|303|tb)\b'
        }
        
        # Pattern type patterns
        self.pattern_patterns = {
            'acid_bassline': r'\b(?:acid|bassline|bass\s*line|303|tb)\b',
            'kick_pattern': r'\b(?:kick|bd|bass\s*drum|kick\s*drum)\b',
            'riff': r'\b(?:riff|melody|melodic|tune)\b',
            'arpeggio': r'\b(?:arpeggio|arp|arpeggiated)\b',
            'chord_progression': r'\b(?:chord|progression|harmony|harmonic)\b',
            'drum_pattern': r'\b(?:drum|percussion|beat|rhythm)\b',
            'hihat_pattern': r'\b(?:hihat|hi\s*hat|hh|cymbals?)\b'
        }
        
        # Style descriptors
        self.style_patterns = {
            'aggressive': r'\b(?:aggressive|hard|brutal|heavy|intense)\b',
            'dark': r'\b(?:dark|evil|sinister|menacing|ominous)\b',
            'distorted': r'\b(?:distorted|crunchy|overdriven|saturated)\b',
            'fast': r'\b(?:fast|quick|rapid|speedy)\b',
            'slow': r'\b(?:slow|relaxed|laid\s*back)\b',
            'complex': r'\b(?:complex|intricate|detailed|sophisticated)\b',
            'simple': r'\b(?:simple|basic|minimal|clean)\b',
            'groovy': r'\b(?:groovy|groove|swing|pocket)\b',
            'tight': r'\b(?:tight|precise|quantized)\b',
            'loose': r'\b(?:loose|sloppy|human|natural)\b'
        }
        
        # Effects patterns
        self.effects_patterns = {
            'reverb': r'\b(?:reverb|hall|room|space|echo)\b',
            'delay': r'\b(?:delay|echo)\b',
            'distortion': r'\b(?:distortion|overdrive|fuzz)\b',
            'filter': r'\b(?:filter|cutoff|resonance|sweep)\b',
            'compression': r'\b(?:compression|compress|punch)\b',
            'sidechain': r'\b(?:sidechain|pumping|ducking)\b'
        }
    
    def parse(self, text: str) -> MusicalParameters:
        """
        Parse natural language text to extract musical parameters.
        
        Args:
            text: Natural language description of desired music
            
        Returns:
            MusicalParameters object with extracted values
        """
        text_lower = text.lower()
        params = MusicalParameters()
        
        # Extract BPM
        params.bpm = self._extract_bpm(text_lower)
        
        # Extract key
        params.key = self._extract_key(text)  # Use original case for key detection
        
        # Extract genre
        params.genre = self._extract_genre(text_lower)
        
        # Extract pattern type
        params.pattern_type = self._extract_pattern_type(text_lower)
        
        # Extract style descriptors
        params.style_descriptors = self._extract_style_descriptors(text_lower)
        
        # Extract effects
        params.effects = self._extract_effects(text_lower)
        
        # Determine mood from style descriptors
        params.mood = self._determine_mood(params.style_descriptors)
        
        # Determine complexity
        params.complexity = self._determine_complexity(params.style_descriptors)
        
        # Apply genre-specific defaults
        self._apply_genre_defaults(params)
        
        return params
    
    def _extract_bpm(self, text: str) -> Optional[int]:
        """Extract BPM from text using regex patterns."""
        for pattern in self.bpm_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                bpm = int(match.group(1))
                # Return BPM even if outside normal range (validation happens separately)
                return bpm
        return None
    
    def _extract_key(self, text: str) -> Optional[str]:
        """Extract musical key from text."""
        for pattern in self.key_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                key = match.group(1).upper()
                # Normalize key format (prefer minor keys for hardcore)
                if 'minor' in text.lower():
                    return f"{key}m"
                elif 'major' in text.lower() or 'maj' in text.lower():
                    return key
                else:
                    # Default to minor for hardcore darkness unless major is explicit
                    return f"{key}m"
        return None
    
    def _extract_genre(self, text: str) -> Optional[Genre]:
        """Extract genre from text using pattern matching."""
        for genre_name, pattern in self.genre_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return Genre(genre_name)
        return None
    
    def _extract_pattern_type(self, text: str) -> Optional[PatternType]:
        """Extract pattern type from text."""
        # Score each pattern type based on keyword matches
        scores = {}
        for pattern_name, pattern in self.pattern_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                scores[pattern_name] = matches
        
        if scores:
            # Return the pattern type with highest score
            best_pattern = max(scores.items(), key=lambda x: x[1])[0]
            return PatternType(best_pattern)
        
        return None
    
    def _extract_style_descriptors(self, text: str) -> List[str]:
        """Extract style descriptors from text."""
        descriptors = []
        for style, pattern in self.style_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                descriptors.append(style)
        return descriptors
    
    def _extract_effects(self, text: str) -> List[str]:
        """Extract effects from text."""
        effects = []
        for effect, pattern in self.effects_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                effects.append(effect)
        return effects
    
    def _determine_mood(self, style_descriptors: List[str]) -> Optional[str]:
        """Determine overall mood from style descriptors."""
        if 'aggressive' in style_descriptors or 'brutal' in style_descriptors:
            return 'aggressive'
        elif 'dark' in style_descriptors:
            return 'dark'
        elif 'groovy' in style_descriptors:
            return 'groovy'
        elif 'simple' in style_descriptors:
            return 'minimal'
        return 'energetic'  # Default for hardcore music
    
    def _determine_complexity(self, style_descriptors: List[str]) -> str:
        """Determine complexity level from style descriptors."""
        if 'complex' in style_descriptors or 'intricate' in style_descriptors:
            return 'complex'
        elif 'simple' in style_descriptors or 'basic' in style_descriptors:
            return 'simple'
        return 'medium'
    
    def _apply_genre_defaults(self, params: MusicalParameters) -> None:
        """Apply genre-specific defaults for missing parameters."""
        if params.genre:
            genre_name = params.genre.value
            
            # Apply BPM defaults if not specified
            if not params.bpm and genre_name in HARDCORE_PARAMS['bpm_ranges']:
                bpm_range = HARDCORE_PARAMS['bpm_ranges'][genre_name]
                # Use middle of range as default
                params.bpm = int((bpm_range[0] + bpm_range[1]) / 2)
            
            # Apply key defaults (prefer minor keys for darkness)
            if not params.key:
                if genre_name in ['gabber', 'hardcore', 'industrial']:
                    params.key = 'Am'  # A minor is versatile for hardcore
                else:
                    params.key = 'Em'  # E minor for variety
            
            # Apply pattern type defaults based on genre
            if not params.pattern_type:
                if genre_name in ['acid', 'gabber']:
                    params.pattern_type = PatternType.ACID_BASSLINE
                elif genre_name in ['hardcore', 'frenchcore']:
                    params.pattern_type = PatternType.KICK_PATTERN
                else:
                    params.pattern_type = PatternType.RIFF
    
    def validate_parameters(self, params: MusicalParameters) -> Tuple[bool, List[str]]:
        """
        Validate extracted parameters for hardcore music constraints.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate BPM range
        if params.bpm and not (60 <= params.bpm <= 300):
            errors.append(f"BPM {params.bpm} outside valid range (60-300)")
        
        # Validate key format
        if params.key and not re.match(r'^[ABCDEFG][#b]?m?$', params.key):
            errors.append(f"Invalid key format: {params.key}")
        
        # Check if parameters are appropriate for hardcore music
        if params.bpm and params.bpm < 120:
            errors.append("BPM too slow for hardcore music (consider 150+ BPM)")
        
        return len(errors) == 0, errors


def parse_musical_text(text: str) -> MusicalParameters:
    """Convenience function to parse musical parameters from text."""
    parser = MusicParser()
    return parser.parse(text)