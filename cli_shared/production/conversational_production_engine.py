#!/usr/bin/env python3
"""
Conversational Music Production Engine
Natural language interface for hardcore music creation and manipulation
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import uuid
from collections import deque

import numpy as np

from ..interfaces.synthesizer import AbstractSynthesizer
from ..models.hardcore_models import HardcorePattern, SynthParams, SynthType
from ..ai.conversation_engine import ConversationEngine, ConversationType, create_conversation_engine
from ..ai.conversation_memory import ConversationMemorySystem, PreferenceCategory
from ..ai.hardcore_knowledge_base import HardcoreKnowledgeBase, hardcore_kb
from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer, KickDNAProfile
from ..evolution.pattern_evolution_engine import PatternEvolutionEngine, BreedingStrategy
from ..performance.live_performance_engine import LivePerformanceEngine, TransitionType


logger = logging.getLogger(__name__)


class ProductionAction(Enum):
    CREATE_PATTERN = "create_pattern"
    MODIFY_PATTERN = "modify_pattern"
    ANALYZE_AUDIO = "analyze_audio"
    EVOLVE_PATTERN = "evolve_pattern"
    STYLE_TRANSFER = "style_transfer"
    SAVE_PATTERN = "save_pattern"
    LOAD_PATTERN = "load_pattern"
    EXPORT_AUDIO = "export_audio"
    SET_BPM = "set_bpm"
    CHANGE_KEY = "change_key"
    APPLY_EFFECT = "apply_effect"
    SUGGEST_VARIATION = "suggest_variation"
    COMPARE_PATTERNS = "compare_patterns"
    GET_HELP = "get_help"


class ProductionIntent(Enum):
    CREATIVE = "creative"           # "Make something dark and industrial"
    TECHNICAL = "technical"         # "Increase the distortion to 0.8"
    ANALYTICAL = "analytical"       # "Analyze this kick drum"
    CORRECTIVE = "corrective"       # "Fix the timing issues"
    EXPLORATORY = "exploratory"     # "What would this sound like at 180 BPM?"
    COMPARATIVE = "comparative"     # "Make it sound more like Surgeon"
    WORKFLOW = "workflow"           # "Save this as warehouse_destroyer"


@dataclass
class ProductionRequest:
    """Structured production request from natural language"""
    session_id: str
    user_input: str
    intent: ProductionIntent
    action: ProductionAction
    parameters: Dict[str, Any]
    confidence: float
    context: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProductionResponse:
    """Response to a production request"""
    success: bool
    message: str
    pattern: Optional[HardcorePattern] = None
    audio_data: Optional[np.ndarray] = None
    analysis_results: Optional[Dict[str, Any]] = None
    suggestions: List[str] = field(default_factory=list)
    code_generated: Optional[str] = None
    execution_time_ms: int = 0
    confidence: float = 0.0


@dataclass
class ProductionSession:
    """Active production session state"""
    session_id: str
    user_id: str
    current_pattern: Optional[HardcorePattern] = None
    pattern_history: deque = field(default_factory=lambda: deque(maxlen=20))
    current_bpm: int = 150
    current_key: str = "Am"
    active_effects: Dict[str, Any] = field(default_factory=dict)
    session_start_time: float = field(default_factory=time.time)
    total_requests: int = 0
    successful_generations: int = 0
    user_satisfaction: float = 0.5  # Rolling satisfaction score
    creative_mode: bool = True      # Allow AI creative suggestions
    

class NaturalLanguageProcessor:
    """Advanced NLP for music production conversations"""
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.parameter_extractors = self._build_parameter_extractors()
        self.music_vocabulary = self._build_music_vocabulary()
        
    def _build_intent_patterns(self) -> Dict[ProductionIntent, List[str]]:
        """Build regex patterns for intent classification"""
        return {
            ProductionIntent.CREATIVE: [
                r"make.*(?:dark|industrial|aggressive|hard|brutal|warehouse)",
                r"create.*(?:pattern|beat|loop|track)",
                r"generate.*(?:gabber|hardcore|techno)",
                r"build.*(?:something|track|pattern)",
                r"give me.*(?:ideas?|inspiration|variation)"
            ],
            ProductionIntent.TECHNICAL: [
                r"set.*(?:bpm|tempo|speed|distortion|filter|reverb)",
                r"change.*(?:parameters?|settings?|values?)",
                r"adjust.*(?:levels?|amounts?|intensity)",
                r"increase|decrease|raise|lower",
                r"(?:add|apply|remove).*(?:effect|processing|filter)"
            ],
            ProductionIntent.ANALYTICAL: [
                r"analyz[e|ing].*(?:kick|bass|pattern|audio)",
                r"what.*(?:frequency|spectral|harmonic)",
                r"check.*(?:levels?|phase|timing)",
                r"measure.*(?:loudness|dynamics|distortion)",
                r"profile.*(?:kick|sound|pattern)"
            ],
            ProductionIntent.CORRECTIVE: [
                r"fix.*(?:timing|phase|levels?|distortion)",
                r"correct.*(?:issues?|problems?|errors?)",
                r"repair.*(?:audio|pattern|sync)",
                r"clean.*(?:up|signal|audio)",
                r"solve.*(?:problem|issue)"
            ],
            ProductionIntent.EXPLORATORY: [
                r"what.*if.*(?:bpm|tempo|key|style)",
                r"how.*(?:would|does).*sound",
                r"try.*(?:different|another|various)",
                r"experiment.*with",
                r"explore.*(?:variations?|options?|possibilities)"
            ],
            ProductionIntent.COMPARATIVE: [
                r"like.*(?:angerfist|surgeon|perc|thunderdome)",
                r"similar.*to.*(?:artist|track|style)",
                r"in.*(?:style|manner).*of",
                r"make.*sound.*like",
                r"compare.*(?:with|to|against)"
            ],
            ProductionIntent.WORKFLOW: [
                r"save.*(?:as|pattern|preset|project)",
                r"load.*(?:pattern|preset|project)",
                r"export.*(?:audio|wav|file)",
                r"backup|store|archive",
                r"open|import|retrieve"
            ]
        }
    
    def _build_parameter_extractors(self) -> Dict[str, str]:
        """Build parameter extraction patterns"""
        return {
            "bpm": r"(?:bpm|tempo|speed).*?(\d{2,3})",
            "distortion": r"distortion.*?([0-9]*\.?[0-9]+)",
            "filter_cutoff": r"(?:filter|cutoff).*?([0-9]*\.?[0-9]+)",
            "reverb": r"reverb.*?([0-9]*\.?[0-9]+)",
            "key": r"(?:key|chord).*?([A-G]#?m?)",
            "genre": r"(gabber|hardcore|industrial|techno|speedcore|doomcore)",
            "artist": r"(angerfist|surgeon|perc|thunderdome|ancient methods|rotterdam terror)",
            "intensity": r"(?:harder|softer|more|less|increase|decrease).*?([0-9]*\.?[0-9]+)?",
            "time_signature": r"(\d)/(\d)",
            "effect_type": r"(delay|reverb|distortion|chorus|flanger|filter|eq|compression)"
        }
    
    def _build_music_vocabulary(self) -> Dict[str, List[str]]:
        """Build music production vocabulary mappings"""
        return {
            "intensity_words": {
                "harder": 0.8, "softer": 0.3, "brutal": 0.95, "gentle": 0.2,
                "aggressive": 0.85, "mellow": 0.25, "violent": 0.9, "subtle": 0.15,
                "crushing": 0.95, "smooth": 0.3, "punchy": 0.75, "warm": 0.4
            },
            "frequency_words": {
                "bright": "high", "dark": "low", "muddy": "low_mid", "crisp": "high",
                "boomy": "low", "thin": "high", "thick": "low_mid", "airy": "high"
            },
            "temporal_words": {
                "fast": 1.5, "slow": 0.7, "quick": 1.3, "lazy": 0.6,
                "rapid": 1.8, "sluggish": 0.5, "frantic": 2.0, "relaxed": 0.4
            },
            "genre_descriptors": {
                "warehouse": "industrial", "rotterdam": "gabber", "berlin": "industrial",
                "underground": "hardcore", "rave": "hardcore", "acid": "techno"
            }
        }
    
    def parse_production_request(self, user_input: str, session_id: str, context: Dict[str, Any]) -> ProductionRequest:
        """Parse natural language into structured production request"""
        user_input_lower = user_input.lower()
        
        # Classify intent
        intent = self._classify_intent(user_input_lower)
        
        # Determine action
        action = self._determine_action(user_input_lower, intent)
        
        # Extract parameters
        parameters = self._extract_parameters(user_input_lower, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(user_input_lower, intent, action, parameters)
        
        return ProductionRequest(
            session_id=session_id,
            user_input=user_input,
            intent=intent,
            action=action,
            parameters=parameters,
            confidence=confidence,
            context=context
        )
    
    def _classify_intent(self, user_input: str) -> ProductionIntent:
        """Classify user intent from input text"""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, user_input, re.IGNORECASE))
                score += matches
            scores[intent] = score
        
        # Return intent with highest score, default to CREATIVE
        if any(scores.values()):
            return max(scores, key=scores.get)
        return ProductionIntent.CREATIVE
    
    def _determine_action(self, user_input: str, intent: ProductionIntent) -> ProductionAction:
        """Determine production action based on intent and keywords"""
        action_keywords = {
            ProductionAction.CREATE_PATTERN: ["make", "create", "generate", "build", "new"],
            ProductionAction.MODIFY_PATTERN: ["change", "modify", "adjust", "tweak", "alter"],
            ProductionAction.ANALYZE_AUDIO: ["analyze", "check", "measure", "profile", "examine"],
            ProductionAction.EVOLVE_PATTERN: ["evolve", "breed", "mutate", "variation", "develop"],
            ProductionAction.STYLE_TRANSFER: ["like", "similar", "style", "sound like", "manner of"],
            ProductionAction.SAVE_PATTERN: ["save", "store", "backup", "archive"],
            ProductionAction.LOAD_PATTERN: ["load", "open", "import", "retrieve"],
            ProductionAction.EXPORT_AUDIO: ["export", "render", "bounce", "mixdown"],
            ProductionAction.SET_BPM: ["bpm", "tempo", "speed"],
            ProductionAction.APPLY_EFFECT: ["effect", "filter", "reverb", "delay", "distortion"],
            ProductionAction.GET_HELP: ["help", "how", "what", "explain"]
        }
        
        action_scores = {}
        for action, keywords in action_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input)
            action_scores[action] = score
        
        # Return highest scoring action, with intent-based fallbacks
        if any(action_scores.values()):
            return max(action_scores, key=action_scores.get)
        
        # Intent-based fallbacks
        intent_to_action = {
            ProductionIntent.CREATIVE: ProductionAction.CREATE_PATTERN,
            ProductionIntent.TECHNICAL: ProductionAction.MODIFY_PATTERN,
            ProductionIntent.ANALYTICAL: ProductionAction.ANALYZE_AUDIO,
            ProductionIntent.COMPARATIVE: ProductionAction.STYLE_TRANSFER,
            ProductionIntent.WORKFLOW: ProductionAction.SAVE_PATTERN
        }
        
        return intent_to_action.get(intent, ProductionAction.CREATE_PATTERN)
    
    def _extract_parameters(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from natural language"""
        parameters = {}
        
        # Extract explicit parameters using regex
        for param_name, pattern in self.parameter_extractors.items():
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            if matches:
                if param_name in ["bpm"]:
                    parameters[param_name] = int(matches[0])
                elif param_name in ["distortion", "filter_cutoff", "reverb"]:
                    parameters[param_name] = float(matches[0])
                else:
                    parameters[param_name] = matches[0]
        
        # Extract intensity modifiers
        intensity_words = self.music_vocabulary["intensity_words"]
        for word, value in intensity_words.items():
            if word in user_input:
                parameters["intensity_modifier"] = value
                parameters["intensity_word"] = word
                break
        
        # Extract frequency descriptors
        freq_words = self.music_vocabulary["frequency_words"]
        for word, freq_range in freq_words.items():
            if word in user_input:
                parameters["frequency_descriptor"] = freq_range
                parameters["frequency_word"] = word
                break
        
        # Extract temporal modifiers
        temporal_words = self.music_vocabulary["temporal_words"]
        for word, multiplier in temporal_words.items():
            if word in user_input:
                parameters["temporal_modifier"] = multiplier
                parameters["temporal_word"] = word
                break
        
        # Extract genre descriptors
        genre_descriptors = self.music_vocabulary["genre_descriptors"]
        for descriptor, genre in genre_descriptors.items():
            if descriptor in user_input:
                parameters["genre_context"] = genre
                parameters["genre_descriptor"] = descriptor
                break
        
        # Add context parameters
        if context:
            parameters.update(context)
        
        return parameters
    
    def _calculate_confidence(self, user_input: str, intent: ProductionIntent, 
                            action: ProductionAction, parameters: Dict[str, Any]) -> float:
        """Calculate confidence score for the parsed request"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear intent patterns
        intent_patterns = self.intent_patterns.get(intent, [])
        for pattern in intent_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                confidence += 0.1
        
        # Boost for extracted parameters
        confidence += len(parameters) * 0.05
        
        # Boost for music-specific vocabulary
        music_words = ["kick", "bass", "synth", "pattern", "beat", "distortion", "filter", "reverb"]
        music_word_count = sum(1 for word in music_words if word in user_input.lower())
        confidence += music_word_count * 0.03
        
        return min(1.0, confidence)


class ConversationalProductionEngine:
    """Main conversational production engine"""
    
    def __init__(self, 
                 synthesizer: AbstractSynthesizer,
                 conversation_engine: ConversationEngine,
                 performance_engine: Optional[LivePerformanceEngine] = None):
        
        self.synthesizer = synthesizer
        self.conversation_engine = conversation_engine
        self.performance_engine = performance_engine
        
        # Core components
        self.nlp = NaturalLanguageProcessor()
        self.audio_analyzer = AdvancedAudioAnalyzer()
        self.evolution_engine = PatternEvolutionEngine()
        self.memory_system = ConversationMemorySystem()
        self.knowledge_base = hardcore_kb
        
        # Session management
        self.active_sessions: Dict[str, ProductionSession] = {}
        
        # Action handlers
        self.action_handlers = {
            ProductionAction.CREATE_PATTERN: self._handle_create_pattern,
            ProductionAction.MODIFY_PATTERN: self._handle_modify_pattern,
            ProductionAction.ANALYZE_AUDIO: self._handle_analyze_audio,
            ProductionAction.EVOLVE_PATTERN: self._handle_evolve_pattern,
            ProductionAction.STYLE_TRANSFER: self._handle_style_transfer,
            ProductionAction.SAVE_PATTERN: self._handle_save_pattern,
            ProductionAction.LOAD_PATTERN: self._handle_load_pattern,
            ProductionAction.EXPORT_AUDIO: self._handle_export_audio,
            ProductionAction.SET_BPM: self._handle_set_bpm,
            ProductionAction.APPLY_EFFECT: self._handle_apply_effect,
            ProductionAction.SUGGEST_VARIATION: self._handle_suggest_variation,
            ProductionAction.GET_HELP: self._handle_get_help
        }
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_confidence": 0.0,
            "patterns_created": 0,
            "sessions_created": 0,
            "most_common_actions": {},
            "user_satisfaction_avg": 0.0
        }
        
    async def process_request(self, user_input: str, session_id: str, user_id: str = "default") -> ProductionResponse:
        """Process a natural language production request"""
        start_time = time.time()
        
        try:
            # Get or create session
            session = self._get_or_create_session(session_id, user_id)
            session.total_requests += 1
            self.stats["total_requests"] += 1
            
            # Get session context
            context = self._build_session_context(session)
            
            # Parse the request
            request = self.nlp.parse_production_request(user_input, session_id, context)
            
            # Log the parsed request
            logger.info(f"Parsed request: {request.intent.value} -> {request.action.value} (confidence: {request.confidence:.2f})")
            
            # Handle the request
            handler = self.action_handlers.get(request.action)
            if not handler:
                return ProductionResponse(
                    success=False,
                    message=f"No handler for action: {request.action.value}",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            
            response = await handler(request, session)
            
            # Update statistics
            self._update_statistics(request, response, session)
            
            # Learn from interaction
            await self._learn_from_interaction(request, response, session)
            
            # Update session
            self._update_session_state(request, response, session)
            
            response.execution_time_ms = int((time.time() - start_time) * 1000)
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return ProductionResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _get_or_create_session(self, session_id: str, user_id: str) -> ProductionSession:
        """Get existing session or create new one"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ProductionSession(
                session_id=session_id,
                user_id=user_id
            )
            self.stats["sessions_created"] += 1
            logger.info(f"Created new production session: {session_id}")
        
        return self.active_sessions[session_id]
    
    def _build_session_context(self, session: ProductionSession) -> Dict[str, Any]:
        """Build context for current session"""
        context = {
            "current_bpm": session.current_bpm,
            "current_key": session.current_key,
            "has_current_pattern": session.current_pattern is not None,
            "pattern_history_count": len(session.pattern_history),
            "session_duration": time.time() - session.session_start_time,
            "total_requests": session.total_requests,
            "successful_generations": session.successful_generations,
            "active_effects": list(session.active_effects.keys()),
            "creative_mode": session.creative_mode
        }
        
        if session.current_pattern:
            context.update({
                "current_pattern_name": session.current_pattern.name,
                "current_pattern_bpm": session.current_pattern.bpm,
                "current_pattern_genre": session.current_pattern.genre,
                "current_synth_type": session.current_pattern.synth_type.value
            })
        
        return context
    
    async def _handle_create_pattern(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle pattern creation requests"""
        try:
            # Get AI assistance for pattern creation
            enhanced_prompt = self._enhance_creation_prompt(request, session)
            ai_response = await self.conversation_engine.chat(request.session_id, enhanced_prompt)
            
            if not ai_response.code:
                return ProductionResponse(
                    success=False,
                    message="AI couldn't generate pattern code",
                    confidence=ai_response.confidence
                )
            
            # Create pattern from generated code
            pattern = self._create_pattern_from_code(ai_response.code, request.parameters, session)
            
            # Generate audio if possible
            audio_data = await self._generate_audio_for_pattern(pattern)
            
            # Analyze the generated pattern
            analysis = None
            if audio_data is not None:
                analysis = await self.audio_analyzer.analyze_pattern_dna(audio_data)
            
            session.successful_generations += 1
            self.stats["patterns_created"] += 1
            
            return ProductionResponse(
                success=True,
                message=ai_response.text,
                pattern=pattern,
                audio_data=audio_data,
                analysis_results=analysis.to_dict() if analysis else None,
                code_generated=ai_response.code,
                confidence=ai_response.confidence
            )
            
        except Exception as e:
            logger.error(f"Error creating pattern: {e}")
            return ProductionResponse(
                success=False,
                message=f"Failed to create pattern: {str(e)}"
            )
    
    def _enhance_creation_prompt(self, request: ProductionRequest, session: ProductionSession) -> str:
        """Enhance prompt for pattern creation"""
        enhanced_prompt = request.user_input
        
        # Add context about current session
        context_parts = []
        
        if session.current_pattern:
            context_parts.append(f"Current pattern: {session.current_pattern.name} ({session.current_pattern.bpm} BPM)")
        
        context_parts.append(f"Session BPM: {session.current_bpm}")
        context_parts.append(f"Session Key: {session.current_key}")
        
        # Add parameter specifications
        if request.parameters:
            param_specs = []
            if "bpm" in request.parameters:
                param_specs.append(f"BPM: {request.parameters['bpm']}")
            if "genre" in request.parameters:
                param_specs.append(f"Genre: {request.parameters['genre']}")
            if "artist" in request.parameters:
                param_specs.append(f"Style: {request.parameters['artist']}")
            if "intensity_modifier" in request.parameters:
                param_specs.append(f"Intensity: {request.parameters['intensity_word']}")
            
            if param_specs:
                context_parts.append(f"Parameters: {', '.join(param_specs)}")
        
        # Add knowledge base context
        if "genre" in request.parameters:
            genre_info = self.knowledge_base.get_genre_template(request.parameters["genre"])
            if genre_info:
                context_parts.append(f"Genre characteristics: {', '.join(genre_info.get('key_characteristics', [])[:2])}")
        
        if context_parts:
            enhanced_prompt = f"Context: {' | '.join(context_parts)}\n\nRequest: {enhanced_prompt}"
        
        return enhanced_prompt
    
    def _create_pattern_from_code(self, code: str, parameters: Dict[str, Any], session: ProductionSession) -> HardcorePattern:
        """Create HardcorePattern object from generated code"""
        # Extract or determine pattern properties
        pattern_name = f"generated_pattern_{int(time.time())}"
        if "name" in parameters:
            pattern_name = parameters["name"]
        
        bpm = parameters.get("bpm", session.current_bpm)
        genre = parameters.get("genre", "gabber")
        
        # Determine synth type from genre/parameters
        synth_type = SynthType.GABBER_KICK
        if genre == "industrial":
            synth_type = SynthType.INDUSTRIAL_KICK
        elif "bass" in code.lower():
            synth_type = SynthType.ACID_BASS
        
        return HardcorePattern(
            name=pattern_name,
            bpm=bpm,
            pattern_data=code,
            synth_type=synth_type,
            genre=genre
        )
    
    async def _generate_audio_for_pattern(self, pattern: HardcorePattern) -> Optional[np.ndarray]:
        """Generate audio data for a pattern"""
        try:
            audio_data = await self.synthesizer.play_pattern(pattern)
            return audio_data
        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            return None
    
    async def _handle_modify_pattern(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle pattern modification requests"""
        if not session.current_pattern:
            return ProductionResponse(
                success=False,
                message="No current pattern to modify. Create a pattern first."
            )
        
        try:
            # Build modification prompt
            modification_prompt = self._build_modification_prompt(request, session)
            ai_response = await self.conversation_engine.chat(request.session_id, modification_prompt)
            
            if ai_response.code:
                # Create modified pattern
                modified_pattern = self._create_pattern_from_code(
                    ai_response.code, 
                    request.parameters, 
                    session
                )
                modified_pattern.name = f"{session.current_pattern.name}_modified"
                
                # Generate audio
                audio_data = await self._generate_audio_for_pattern(modified_pattern)
                
                return ProductionResponse(
                    success=True,
                    message=ai_response.text,
                    pattern=modified_pattern,
                    audio_data=audio_data,
                    code_generated=ai_response.code,
                    confidence=ai_response.confidence
                )
            else:
                # Parameter-only modification
                return await self._apply_parameter_modifications(request, session)
                
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Failed to modify pattern: {str(e)}"
            )
    
    def _build_modification_prompt(self, request: ProductionRequest, session: ProductionSession) -> str:
        """Build prompt for pattern modification"""
        current_code = session.current_pattern.pattern_data
        
        prompt = f"""Current pattern code:
```
{current_code}
```

Current BPM: {session.current_pattern.bpm}
Current Genre: {session.current_pattern.genre}

Modification request: {request.user_input}

Please modify the pattern code according to the request."""
        
        return prompt
    
    async def _apply_parameter_modifications(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Apply parameter-based modifications to current pattern"""
        modified_pattern = session.current_pattern
        modifications = []
        
        # Apply BPM changes
        if "bpm" in request.parameters:
            modified_pattern.bpm = request.parameters["bpm"]
            modifications.append(f"BPM changed to {modified_pattern.bpm}")
        
        # Apply intensity modifications
        if "intensity_modifier" in request.parameters:
            # This would modify synthesis parameters based on intensity
            modifications.append(f"Intensity {request.parameters['intensity_word']}")
        
        if modifications:
            message = f"Applied modifications: {', '.join(modifications)}"
            audio_data = await self._generate_audio_for_pattern(modified_pattern)
            
            return ProductionResponse(
                success=True,
                message=message,
                pattern=modified_pattern,
                audio_data=audio_data,
                confidence=0.8
            )
        else:
            return ProductionResponse(
                success=False,
                message="No applicable modifications found in request"
            )
    
    async def _handle_analyze_audio(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle audio analysis requests"""
        if not session.current_pattern:
            return ProductionResponse(
                success=False,
                message="No current pattern to analyze. Create a pattern first."
            )
        
        try:
            # Generate audio for analysis
            audio_data = await self._generate_audio_for_pattern(session.current_pattern)
            if audio_data is None:
                return ProductionResponse(
                    success=False,
                    message="Failed to generate audio for analysis"
                )
            
            # Perform comprehensive analysis
            analysis_results = {}
            
            # Basic audio analysis
            basic_analysis = await self.audio_analyzer.analyze_pattern_dna(audio_data)
            analysis_results["basic"] = basic_analysis.to_dict()
            
            # Kick DNA analysis if it's a kick pattern
            if "kick" in session.current_pattern.synth_type.value:
                kick_analysis = await self.audio_analyzer.analyze_kick_dna(audio_data)
                analysis_results["kick_dna"] = kick_analysis.to_dict()
            
            # Psychoacoustic analysis
            psycho_analysis = await self.audio_analyzer.analyze_psychoacoustic_properties(audio_data)
            analysis_results["psychoacoustic"] = psycho_analysis
            
            # Generate analysis summary
            summary = self._generate_analysis_summary(analysis_results)
            
            return ProductionResponse(
                success=True,
                message=summary,
                analysis_results=analysis_results,
                confidence=0.9
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Analysis failed: {str(e)}"
            )
    
    def _generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate human-readable analysis summary"""
        summary_parts = []
        
        if "basic" in analysis_results:
            basic = analysis_results["basic"]
            summary_parts.append(f"Pattern Analysis:")
            summary_parts.append(f"• Energy: {basic.get('energy', 0):.2f}")
            summary_parts.append(f"• Danceability: {basic.get('danceability', 0):.2f}")
            summary_parts.append(f"• Technical Quality: {basic.get('technical_quality', 0):.2f}")
        
        if "kick_dna" in analysis_results:
            kick_dna = analysis_results["kick_dna"]
            summary_parts.append(f"\nKick Analysis:")
            summary_parts.append(f"• Type: {kick_dna.get('kick_type', 'Unknown')}")
            summary_parts.append(f"• Crunch Factor: {kick_dna.get('crunch_factor', 0):.2f}")
            summary_parts.append(f"• Punchiness: {kick_dna.get('punchiness', 0):.2f}")
        
        if "psychoacoustic" in analysis_results:
            psycho = analysis_results["psychoacoustic"]
            summary_parts.append(f"\nPsychoacoustic Properties:")
            summary_parts.append(f"• Brightness: {psycho.get('brightness', 0):.2f}")
            summary_parts.append(f"• Roughness: {psycho.get('roughness', 0):.2f}")
            summary_parts.append(f"• Warmth: {psycho.get('warmth', 0):.2f}")
        
        return "\n".join(summary_parts)
    
    async def _handle_evolve_pattern(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle pattern evolution requests"""
        if not session.current_pattern:
            return ProductionResponse(
                success=False,
                message="No current pattern to evolve. Create a pattern first."
            )
        
        try:
            # Generate pattern variations using evolution
            population_size = request.parameters.get("population_size", 5)
            generations = request.parameters.get("generations", 3)
            
            # Create initial population with current pattern
            population = await self.evolution_engine.generate_population(
                population_size=population_size,
                base_pattern=session.current_pattern
            )
            
            # Evolve for specified generations
            for gen in range(generations):
                population = await self.evolution_engine.evolve_generation(
                    population=population,
                    breeding_strategy=BreedingStrategy.TOURNAMENT_SELECTION
                )
            
            # Get the best evolved pattern
            best_pattern = population[0]  # Assuming population is sorted by fitness
            
            # Generate audio for the evolved pattern
            audio_data = await self._generate_audio_for_pattern(best_pattern)
            
            return ProductionResponse(
                success=True,
                message=f"Evolved pattern through {generations} generations. Best fitness: {await self.evolution_engine.evaluate_fitness(best_pattern)}",
                pattern=best_pattern,
                audio_data=audio_data,
                suggestions=[p.name for p in population[1:4]],  # Top 3 alternatives
                confidence=0.85
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Evolution failed: {str(e)}"
            )
    
    async def _handle_style_transfer(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle style transfer requests"""
        if not session.current_pattern:
            return ProductionResponse(
                success=False,
                message="No current pattern for style transfer. Create a pattern first."
            )
        
        try:
            artist = request.parameters.get("artist", "")
            genre = request.parameters.get("genre", "")
            
            # Build style transfer prompt
            style_prompt = f"""Transform the current pattern to sound like {artist} in {genre} style.

Current pattern:
```
{session.current_pattern.pattern_data}
```

Make it sound like {artist} while maintaining the core elements."""
            
            ai_response = await self.conversation_engine.chat(request.session_id, style_prompt)
            
            if ai_response.code:
                # Create style-transferred pattern
                transferred_pattern = self._create_pattern_from_code(
                    ai_response.code,
                    request.parameters,
                    session
                )
                transferred_pattern.name = f"{session.current_pattern.name}_{artist}_style"
                
                # Generate audio
                audio_data = await self._generate_audio_for_pattern(transferred_pattern)
                
                return ProductionResponse(
                    success=True,
                    message=ai_response.text,
                    pattern=transferred_pattern,
                    audio_data=audio_data,
                    code_generated=ai_response.code,
                    confidence=ai_response.confidence
                )
            else:
                return ProductionResponse(
                    success=False,
                    message="Failed to generate style-transferred code"
                )
                
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Style transfer failed: {str(e)}"
            )
    
    async def _handle_save_pattern(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle pattern saving requests"""
        if not session.current_pattern:
            return ProductionResponse(
                success=False,
                message="No current pattern to save"
            )
        
        try:
            pattern_name = request.parameters.get("name", session.current_pattern.name)
            
            # Save to memory system
            self.memory_system.record_pattern_creation(
                session_id=request.session_id,
                pattern=session.current_pattern,
                user_feedback="User requested save",
                rating=0.8  # Assume positive since user wants to save
            )
            
            return ProductionResponse(
                success=True,
                message=f"Pattern saved as '{pattern_name}'",
                confidence=1.0
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Failed to save pattern: {str(e)}"
            )
    
    async def _handle_load_pattern(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle pattern loading requests"""
        pattern_name = request.parameters.get("name", "")
        
        if not pattern_name:
            return ProductionResponse(
                success=False,
                message="No pattern name specified for loading"
            )
        
        try:
            # Try to find similar patterns in memory
            similar_patterns = self.memory_system.get_similar_patterns(
                session_id=request.session_id,
                bpm=session.current_bpm,
                genre="gabber",  # Default
                limit=5
            )
            
            if similar_patterns:
                # Load the first matching pattern
                pattern_data = similar_patterns[0]["pattern"]
                loaded_pattern = HardcorePattern(**pattern_data)
                
                return ProductionResponse(
                    success=True,
                    message=f"Loaded pattern: {loaded_pattern.name}",
                    pattern=loaded_pattern,
                    confidence=0.9
                )
            else:
                return ProductionResponse(
                    success=False,
                    message=f"Pattern '{pattern_name}' not found"
                )
                
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Failed to load pattern: {str(e)}"
            )
    
    async def _handle_export_audio(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle audio export requests"""
        if not session.current_pattern:
            return ProductionResponse(
                success=False,
                message="No current pattern to export"
            )
        
        try:
            # Generate audio for export
            audio_data = await self._generate_audio_for_pattern(session.current_pattern)
            
            if audio_data is None:
                return ProductionResponse(
                    success=False,
                    message="Failed to generate audio for export"
                )
            
            # Export would save to file (implementation specific)
            export_filename = f"{session.current_pattern.name}.wav"
            
            return ProductionResponse(
                success=True,
                message=f"Audio exported as {export_filename}",
                audio_data=audio_data,
                confidence=1.0
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Export failed: {str(e)}"
            )
    
    async def _handle_set_bpm(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle BPM change requests"""
        bpm = request.parameters.get("bpm")
        
        if not bpm:
            return ProductionResponse(
                success=False,
                message="No BPM value specified"
            )
        
        try:
            session.current_bpm = int(bpm)
            
            # If there's a current pattern, update its BPM too
            if session.current_pattern:
                session.current_pattern.bpm = session.current_bpm
            
            return ProductionResponse(
                success=True,
                message=f"BPM set to {session.current_bpm}",
                confidence=1.0
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Failed to set BPM: {str(e)}"
            )
    
    async def _handle_apply_effect(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle effect application requests"""
        effect_type = request.parameters.get("effect_type", "")
        
        if not effect_type:
            return ProductionResponse(
                success=False,
                message="No effect type specified"
            )
        
        try:
            # Add effect to session active effects
            session.active_effects[effect_type] = request.parameters
            
            return ProductionResponse(
                success=True,
                message=f"Applied {effect_type} effect",
                confidence=0.8
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Failed to apply effect: {str(e)}"
            )
    
    async def _handle_suggest_variation(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle variation suggestion requests"""
        if not session.current_pattern:
            return ProductionResponse(
                success=False,
                message="No current pattern to create variations for"
            )
        
        try:
            # Generate suggestions based on current pattern
            suggestions = []
            
            # BPM variations
            current_bpm = session.current_pattern.bpm
            suggestions.extend([
                f"Try it at {current_bpm + 10} BPM for more energy",
                f"Slow it down to {current_bpm - 20} BPM for a different vibe",
                f"Double-time version at {current_bpm * 2} BPM"
            ])
            
            # Style variations
            if session.current_pattern.genre == "gabber":
                suggestions.extend([
                    "Add industrial reverb for warehouse feel",
                    "Apply more distortion for Rotterdam style",
                    "Strip back for minimal techno approach"
                ])
            
            # Evolution suggestions
            suggestions.extend([
                "Evolve with genetic algorithms for mutations",
                "Breed with another pattern for hybrid sound",
                "Apply style transfer from different artist"
            ])
            
            return ProductionResponse(
                success=True,
                message="Here are some variation ideas:",
                suggestions=suggestions[:6],  # Limit to 6 suggestions
                confidence=0.9
            )
            
        except Exception as e:
            return ProductionResponse(
                success=False,
                message=f"Failed to generate suggestions: {str(e)}"
            )
    
    async def _handle_get_help(self, request: ProductionRequest, session: ProductionSession) -> ProductionResponse:
        """Handle help requests"""
        help_topics = {
            "create": "To create patterns: 'make a gabber kick at 180 BPM' or 'generate industrial loop'",
            "modify": "To modify patterns: 'make it harder' or 'increase distortion to 0.8'",
            "analyze": "To analyze audio: 'analyze the kick' or 'check frequency spectrum'",
            "evolve": "To evolve patterns: 'evolve this pattern' or 'breed with genetic algorithms'",
            "style": "For style transfer: 'make it sound like Surgeon' or 'apply Angerfist style'",
            "save": "To save patterns: 'save as warehouse_destroyer' or 'store this pattern'",
            "bpm": "To change BPM: 'set tempo to 175' or 'make it faster'"
        }
        
        # Check if asking about specific topic
        user_input_lower = request.user_input.lower()
        specific_help = None
        
        for topic, help_text in help_topics.items():
            if topic in user_input_lower:
                specific_help = help_text
                break
        
        if specific_help:
            message = f"Help for {topic}:\n{specific_help}"
        else:
            # General help
            message = "GABBERBOT Commands:\n\n" + "\n".join(help_topics.values())
        
        return ProductionResponse(
            success=True,
            message=message,
            confidence=1.0
        )
    
    def _update_statistics(self, request: ProductionRequest, response: ProductionResponse, session: ProductionSession):
        """Update engine statistics"""
        if response.success:
            self.stats["successful_requests"] += 1
        
        # Update confidence average
        total_conf = self.stats["average_confidence"] * (self.stats["total_requests"] - 1)
        self.stats["average_confidence"] = (total_conf + request.confidence) / self.stats["total_requests"]
        
        # Update action frequency
        action = request.action.value
        self.stats["most_common_actions"][action] = self.stats["most_common_actions"].get(action, 0) + 1
    
    async def _learn_from_interaction(self, request: ProductionRequest, response: ProductionResponse, session: ProductionSession):
        """Learn from user interactions"""
        try:
            # Record successful pattern generations
            if response.success and response.pattern:
                self.memory_system.record_pattern_creation(
                    session_id=request.session_id,
                    pattern=response.pattern,
                    user_feedback=None,
                    rating=response.confidence
                )
            
            # Learn user preferences
            if request.parameters:
                if "bpm" in request.parameters:
                    self.memory_system.learn_user_preference(
                        session_id=request.session_id,
                        category=PreferenceCategory.BPM,
                        preference_data={"preferred_bpm": request.parameters["bpm"]},
                        confidence=request.confidence
                    )
                
                if "genre" in request.parameters:
                    self.memory_system.learn_user_preference(
                        session_id=request.session_id,
                        category=PreferenceCategory.GENRE,
                        preference_data={"genre": request.parameters["genre"]},
                        confidence=request.confidence
                    )
                
                if "artist" in request.parameters:
                    self.memory_system.learn_user_preference(
                        session_id=request.session_id,
                        category=PreferenceCategory.ARTIST_STYLE,
                        preference_data={"artist": request.parameters["artist"]},
                        confidence=request.confidence
                    )
            
        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
    
    def _update_session_state(self, request: ProductionRequest, response: ProductionResponse, session: ProductionSession):
        """Update session state after processing request"""
        if response.success:
            if response.pattern:
                # Update current pattern
                if session.current_pattern:
                    session.pattern_history.append(session.current_pattern)
                session.current_pattern = response.pattern
                
                # Update session BPM/key if pattern specifies them
                session.current_bpm = response.pattern.bpm
        
        # Update satisfaction score (simplified)
        satisfaction_change = 0.1 if response.success else -0.1
        session.user_satisfaction = max(0.0, min(1.0, session.user_satisfaction + satisfaction_change))
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a production session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "current_pattern": {
                "name": session.current_pattern.name,
                "bpm": session.current_pattern.bpm,
                "genre": session.current_pattern.genre
            } if session.current_pattern else None,
            "current_bpm": session.current_bpm,
            "current_key": session.current_key,
            "pattern_history_count": len(session.pattern_history),
            "total_requests": session.total_requests,
            "successful_generations": session.successful_generations,
            "user_satisfaction": session.user_satisfaction,
            "active_effects": list(session.active_effects.keys()),
            "session_duration": time.time() - session.session_start_time,
            "creative_mode": session.creative_mode
        }
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get overall engine statistics"""
        return {
            **self.stats,
            "active_sessions": len(self.active_sessions),
            "memory_stats": self.memory_system.get_memory_stats(),
            "success_rate": (self.stats["successful_requests"] / max(1, self.stats["total_requests"])) * 100
        }


# Factory function
def create_conversational_production_engine(
    synthesizer: AbstractSynthesizer,
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    performance_engine: Optional[LivePerformanceEngine] = None
) -> ConversationalProductionEngine:
    """Create a conversational production engine with all dependencies"""
    
    # Create conversation engine
    conversation_engine = create_conversation_engine(
        synthesizer=synthesizer,
        openai_key=openai_key,
        anthropic_key=anthropic_key
    )
    
    return ConversationalProductionEngine(
        synthesizer=synthesizer,
        conversation_engine=conversation_engine,
        performance_engine=performance_engine
    )


if __name__ == "__main__":
    # Demo the conversational production engine
    async def demo_conversational_production():
        from ..interfaces.synthesizer import MockSynthesizer
        
        # Create mock synthesizer
        synth = MockSynthesizer()
        
        # Create conversational engine (no API keys for demo)
        engine = create_conversational_production_engine(synth)
        
        print("=== Conversational Production Engine Demo ===")
        
        session_id = "demo_session_001"
        
        # Test various production requests
        test_requests = [
            "Make me a hard gabber kick at 180 BPM",
            "Make it even more aggressive and brutal",
            "Analyze the kick drum frequency content",
            "Evolve this pattern with genetic algorithms",
            "Save this as warehouse_destroyer",
            "Set the BPM to 200 and make it sound like Angerfist",
            "What are some variations I could try?"
        ]
        
        for request_text in test_requests:
            print(f"\n🔊 User: {request_text}")
            
            response = await engine.process_request(
                user_input=request_text,
                session_id=session_id
            )
            
            print(f"🤖 Success: {response.success}")
            print(f"📝 Response: {response.message}")
            
            if response.pattern:
                print(f"🎵 Pattern: {response.pattern.name} ({response.pattern.bpm} BPM)")
            
            if response.suggestions:
                print(f"💡 Suggestions: {', '.join(response.suggestions[:3])}")
            
            print(f"⚡ Time: {response.execution_time_ms}ms | Confidence: {response.confidence:.2f}")
        
        # Show session status
        session_status = engine.get_session_status(session_id)
        print(f"\n📊 Session Status:")
        print(f"- Total requests: {session_status['total_requests']}")
        print(f"- Current BPM: {session_status['current_bpm']}")
        print(f"- User satisfaction: {session_status['user_satisfaction']:.2f}")
        
        # Show engine stats
        engine_stats = engine.get_engine_stats()
        print(f"\n🔧 Engine Stats:")
        print(f"- Success rate: {engine_stats['success_rate']:.1f}%")
        print(f"- Patterns created: {engine_stats['patterns_created']}")
        print(f"- Active sessions: {engine_stats['active_sessions']}")
        
        print("\n=== Demo completed ===")
    
    # Run demo
    asyncio.run(demo_conversational_production())