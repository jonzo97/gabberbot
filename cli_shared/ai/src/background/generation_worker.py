#!/usr/bin/env python3
"""
Enhanced Intelligent Music Agent V2 - With Persistent Learning
True AI-powered agent with cross-session learning and iterative improvement
"""

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import hashlib
import numpy as np

# AI SDK imports
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

# Redis for medium-term cache
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..models.hardcore_models import HardcorePattern, SynthType
from ..models.midi_clips import MIDIClip, TriggerClip
from ..interfaces.synthesizer import AbstractSynthesizer
from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer, KickDNAProfile, PsychoacousticAnalysis


@dataclass
class UserPreferences:
    """Learned user preferences"""
    user_id: str
    preferred_genres: Dict[str, float] = field(default_factory=dict)  # genre -> preference score
    preferred_bpm_range: Tuple[int, int] = (150, 200)
    preferred_artists: Dict[str, float] = field(default_factory=dict)  # artist -> preference score
    synthesis_preferences: Dict[str, float] = field(default_factory=dict)
    vocabulary_mappings: Dict[str, str] = field(default_factory=dict)  # user_term -> technical_term
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    interaction_style: str = "conversational"  # conversational, technical, brief
    technical_level: int = 3  # 1-5 scale
    total_patterns_created: int = 0
    successful_patterns: int = 0
    last_updated: float = 0.0


@dataclass
class PatternQualityMetrics:
    """Quality metrics for generated patterns"""
    pattern_id: str
    kick_dna_score: float = 0.0
    psychoacoustic_score: float = 0.0
    danceability: float = 0.0
    energy_level: float = 0.0
    technical_quality: float = 0.0
    user_rating: Optional[float] = None
    improvements_applied: List[str] = field(default_factory=list)
    generation_method: str = "ai_generated"
    timestamp: float = 0.0


@dataclass
class LearningInsight:
    """An insight learned by the agent"""
    insight_id: str
    user_id: str
    insight_type: str  # preference, pattern, correction, success
    content: Dict[str, Any]
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: float = 0.0
    last_used: float = 0.0


class IntelligentMusicAgentV2:
    """
    Enhanced AI-powered music agent with persistent learning
    
    Features:
    - Cross-session learning and memory
    - Real-time audio analysis feedback
    - Iterative improvement based on quality metrics
    - User preference learning and adaptation
    - Multi-model AI orchestration with fallback
    - Vocabulary mapping and personalization
    """
    
    def __init__(self, 
                 synthesizer: AbstractSynthesizer,
                 memory_dir: str = "~/.gabberbot/memory",
                 user_id: str = "default"):
        
        self.synthesizer = synthesizer
        self.user_id = user_id
        self.session_id = f"session_{int(time.time())}"
        
        # Memory storage paths
        self.memory_dir = Path(os.path.expanduser(memory_dir))
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.memory_dir / "agent_memory.db"
        
        # Initialize database
        self._init_database()
        
        # Load user preferences
        self.user_preferences = self._load_user_preferences()
        
        # Initialize AI clients
        self.anthropic_client = None
        self.openai_client = None
        self.google_client = None
        self._init_ai_clients()
        
        # Initialize audio analyzer
        self.audio_analyzer = AdvancedAudioAnalyzer()
        
        # Redis cache for medium-term memory
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
            except:
                self.redis_client = None
        
        # Session state
        self.conversation_history: List[Dict[str, Any]] = []
        self.generated_patterns: Dict[str, PatternQualityMetrics] = {}
        self.learning_insights: List[LearningInsight] = []
        
        # System prompts optimized for learning
        self.system_prompts = {
            "music_analyzer": self._create_analyzer_prompt(),
            "creative_generator": self._create_generator_prompt(),
            "improvement_advisor": self._create_improvement_prompt(),
            "preference_learner": self._create_preference_prompt()
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            # User preferences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Pattern history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    pattern_data JSON NOT NULL,
                    quality_metrics JSON,
                    user_rating REAL,
                    play_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_preferences(user_id)
                )
            """)
            
            # Learning insights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    content JSON NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_preferences(user_id)
                )
            """)
            
            # Conversation memory table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    messages JSON NOT NULL,
                    context JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_preferences(user_id)
                )
            """)
            
            # Vocabulary mappings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vocabulary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_term TEXT NOT NULL,
                    technical_term TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_preferences(user_id),
                    UNIQUE(user_id, user_term)
                )
            """)
            
            conn.commit()
    
    def _init_ai_clients(self):
        """Initialize AI clients with API keys"""
        # Anthropic Claude (Primary)
        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key and api_key != "your_claude_key_here":
                try:
                    self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                    self.logger.info("Claude AI initialized for enhanced agent")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Claude: {e}")
        
        # OpenAI GPT (Secondary)
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    openai.api_key = api_key
                    self.openai_client = openai
                    self.logger.info("OpenAI GPT initialized for enhanced agent")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Google Gemini (Tertiary)
        if GOOGLE_AI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.google_client = genai.GenerativeModel('gemini-pro')
                    self.logger.info("Gemini AI initialized for enhanced agent")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini: {e}")
    
    def _load_user_preferences(self) -> UserPreferences:
        """Load user preferences from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT preferences FROM user_preferences WHERE user_id = ?",
                    (self.user_id,)
                )
                row = cursor.fetchone()
                if row:
                    prefs_data = json.loads(row[0])
                    return UserPreferences(user_id=self.user_id, **prefs_data)
        except Exception as e:
            self.logger.error(f"Error loading preferences: {e}")
        
        # Return default preferences
        return UserPreferences(user_id=self.user_id)
    
    def _save_user_preferences(self):
        """Save user preferences to database"""
        try:
            prefs_dict = asdict(self.user_preferences)
            prefs_dict.pop('user_id')  # Remove user_id from JSON
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (self.user_id, json.dumps(prefs_dict)))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving preferences: {e}")
    
    def _create_analyzer_prompt(self) -> str:
        """Create analysis prompt with user context"""
        base_prompt = """You are an expert music producer analyzing hardcore/gabber/industrial tracks.

User Preferences:
{user_preferences}

Analyze the current musical context and provide:
1. Quality assessment (kick punch, mix clarity, energy)
2. Specific technical issues
3. Concrete improvement suggestions
4. Comparison to user's preferred style

Focus on: {focus_areas}

Respond with specific, actionable feedback in JSON format."""
        
        # Customize based on user preferences
        focus_areas = []
        if self.user_preferences.preferred_genres:
            top_genre = max(self.user_preferences.preferred_genres.items(), key=lambda x: x[1])
            focus_areas.append(f"{top_genre[0]} authenticity")
        
        if self.user_preferences.quality_thresholds:
            for metric, threshold in self.user_preferences.quality_thresholds.items():
                if threshold > 0.7:
                    focus_areas.append(metric)
        
        return base_prompt.format(
            user_preferences=json.dumps({
                "genres": list(self.user_preferences.preferred_genres.keys()),
                "bpm_range": self.user_preferences.preferred_bpm_range,
                "technical_level": self.user_preferences.technical_level
            }),
            focus_areas=", ".join(focus_areas) if focus_areas else "overall quality"
        )
    
    def _create_generator_prompt(self) -> str:
        """Create generation prompt with learned preferences"""
        return f"""You are a creative hardcore music producer who knows the user's style deeply.

User's Musical Identity:
- Favorite genres: {', '.join(self.user_preferences.preferred_genres.keys())}
- Preferred BPM: {self.user_preferences.preferred_bpm_range[0]}-{self.user_preferences.preferred_bpm_range[1]}
- Technical level: {self.user_preferences.technical_level}/5
- Successful patterns: {self.user_preferences.successful_patterns}

Generate creative ideas that:
1. Match the user's established style
2. Push boundaries while staying authentic
3. Use production techniques they appreciate
4. Build on their previous successes

Be specific about synthesis parameters, pattern structures, and sound design."""
    
    def _create_improvement_prompt(self) -> str:
        """Create improvement prompt based on analysis"""
        return """You are a mastering engineer improving hardcore tracks.

Given these quality metrics:
{quality_metrics}

And these identified issues:
{issues}

Provide specific improvements:
1. Parameter adjustments (exact values)
2. Processing chain modifications
3. Mix balance corrections
4. Energy and dynamics optimization

Format as actionable steps with specific values."""
    
    def _create_preference_prompt(self) -> str:
        """Create preference learning prompt"""
        return """You are analyzing user behavior to learn their preferences.

Based on these interactions:
{interactions}

Identify:
1. Consistent preferences (genres, sounds, techniques)
2. Vocabulary patterns (how they describe things)
3. Quality standards (what they consider good)
4. Workflow preferences (how they like to work)

Extract learnable insights that can improve future interactions."""
    
    async def analyze_and_improve(self, audio_data: np.ndarray, pattern: HardcorePattern) -> Tuple[Dict[str, Any], List[str]]:
        """
        Analyze audio and suggest improvements using AI + audio analysis
        
        Returns: (analysis_results, improvement_suggestions)
        """
        # Perform audio analysis
        kick_dna = await self.audio_analyzer.analyze_kick_dna(audio_data)
        psycho_analysis = await self.audio_analyzer.analyze_psychoacoustic(audio_data)
        
        # Create quality metrics
        metrics = PatternQualityMetrics(
            pattern_id=pattern.name,
            kick_dna_score=kick_dna.confidence,
            psychoacoustic_score=psycho_analysis.aggression,
            danceability=psycho_analysis.warehouse_factor,
            energy_level=(psycho_analysis.brightness + psycho_analysis.crunch_factor) / 2,
            technical_quality=1.0 - psycho_analysis.roughness,
            timestamp=time.time()
        )
        
        # Store metrics
        self.generated_patterns[pattern.name] = metrics
        
        # Use AI to interpret analysis and suggest improvements
        analysis_context = {
            "kick_type": kick_dna.kick_type.value,
            "kick_confidence": kick_dna.confidence,
            "fundamental_freq": kick_dna.fundamental_freq,
            "punch_factor": kick_dna.punch_factor,
            "distortion_level": kick_dna.distortion_level,
            "brightness": psycho_analysis.brightness,
            "roughness": psycho_analysis.roughness,
            "crunch_factor": psycho_analysis.crunch_factor,
            "aggression": psycho_analysis.aggression,
            "warehouse_factor": psycho_analysis.warehouse_factor
        }
        
        # Get AI recommendations
        improvements = await self._get_ai_improvements(analysis_context, pattern)
        
        # Learn from this analysis
        await self._learn_from_analysis(analysis_context, pattern)
        
        return analysis_context, improvements
    
    async def _get_ai_improvements(self, analysis: Dict[str, Any], pattern: HardcorePattern) -> List[str]:
        """Get improvement suggestions from AI"""
        if not any([self.anthropic_client, self.openai_client, self.google_client]):
            return ["Add more distortion to the kick", "Increase the low-end punch"]
        
        prompt = self.system_prompts["improvement_advisor"].format(
            quality_metrics=json.dumps(analysis, indent=2),
            issues=self._identify_issues(analysis)
        )
        
        try:
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    system=prompt,
                    messages=[{
                        "role": "user",
                        "content": f"Improve this {pattern.genre} pattern at {pattern.bpm} BPM"
                    }]
                )
                
                # Parse improvements from response
                improvements_text = response.content[0].text
                improvements = [line.strip() for line in improvements_text.split('\n') if line.strip()]
                return improvements[:5]  # Top 5 improvements
        except Exception as e:
            self.logger.error(f"Error getting AI improvements: {e}")
        
        # Fallback improvements based on analysis
        return self._generate_fallback_improvements(analysis)
    
    def _identify_issues(self, analysis: Dict[str, Any]) -> str:
        """Identify issues from analysis metrics"""
        issues = []
        
        if analysis.get("kick_confidence", 0) < 0.5:
            issues.append("Kick drum lacks definition")
        
        if analysis.get("punch_factor", 0) < 0.6:
            issues.append("Insufficient kick punch")
        
        if analysis.get("brightness", 0) < 0.3:
            issues.append("Mix too dark/muddy")
        elif analysis.get("brightness", 0) > 0.8:
            issues.append("Mix too bright/harsh")
        
        if analysis.get("roughness", 0) > 0.7:
            issues.append("Excessive roughness/distortion artifacts")
        
        if analysis.get("warehouse_factor", 0) < 0.4:
            issues.append("Lacks industrial/warehouse character")
        
        return ", ".join(issues) if issues else "Minor refinements needed"
    
    def _generate_fallback_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvements without AI"""
        improvements = []
        
        if analysis.get("punch_factor", 0) < 0.6:
            improvements.append("Increase kick attack: boost 200-500Hz by +3dB")
        
        if analysis.get("distortion_level", 0) < 0.5:
            improvements.append("Add saturation/distortion to increase aggression")
        
        if analysis.get("brightness", 0) < 0.4:
            improvements.append("Add presence: boost 3-5kHz range")
        
        if analysis.get("warehouse_factor", 0) < 0.5:
            improvements.append("Add reverb with 0.3 wet mix for warehouse atmosphere")
        
        return improvements
    
    async def _learn_from_analysis(self, analysis: Dict[str, Any], pattern: HardcorePattern):
        """Learn from analysis results to improve future generations"""
        # Update quality thresholds based on successful patterns
        if analysis.get("aggression", 0) > 0.7:
            self.user_preferences.quality_thresholds["aggression"] = max(
                self.user_preferences.quality_thresholds.get("aggression", 0),
                analysis["aggression"] * 0.9  # Set threshold slightly below success
            )
        
        # Learn genre characteristics
        if pattern.genre in self.user_preferences.preferred_genres:
            # This genre produced good results, increase preference
            self.user_preferences.preferred_genres[pattern.genre] += 0.1
        
        # Create learning insight
        insight = LearningInsight(
            insight_id=f"analysis_{int(time.time())}_{pattern.name}",
            user_id=self.user_id,
            insight_type="pattern_success",
            content={
                "pattern_type": pattern.genre,
                "bpm": pattern.bpm,
                "quality_metrics": analysis,
                "successful_elements": self._extract_successful_elements(analysis)
            },
            confidence=0.7,
            created_at=time.time()
        )
        
        self.learning_insights.append(insight)
        await self._save_insight(insight)
    
    def _extract_successful_elements(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract successful elements from analysis"""
        successful = []
        
        if analysis.get("kick_confidence", 0) > 0.7:
            successful.append(f"kick_{analysis.get('kick_type', 'unknown')}")
        
        if analysis.get("aggression", 0) > 0.7:
            successful.append("high_aggression")
        
        if analysis.get("warehouse_factor", 0) > 0.6:
            successful.append("warehouse_atmosphere")
        
        return successful
    
    async def generate_creative_riff(self, context: Dict[str, Any]) -> MIDIClip:
        """Generate creative riff using AI with learned preferences"""
        # Build prompt with user preferences
        prompt = f"""Generate a creative {context.get('style', 'hardcore')} riff.

User preferences:
- Genres: {', '.join(self.user_preferences.preferred_genres.keys())}
- BPM range: {self.user_preferences.preferred_bpm_range}
- Previous successes: {self.user_preferences.successful_patterns}

Context: {json.dumps(context)}

Create a unique but fitting melodic pattern. Include:
1. Note sequence (MIDI notes)
2. Rhythm pattern
3. Suggested sound/synth
4. Variation ideas

Respond with specific MIDI note numbers and timings."""
        
        try:
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=400,
                    system=self.system_prompts["creative_generator"],
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Parse response and create MIDI clip
                riff_data = self._parse_riff_response(response.content[0].text)
                return self._create_midi_clip_from_data(riff_data, context)
        except Exception as e:
            self.logger.error(f"Error generating creative riff: {e}")
        
        # Fallback to algorithmic generation
        return self._generate_algorithmic_riff(context)
    
    def _parse_riff_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response to extract riff data"""
        # This would parse the AI's response to extract MIDI data
        # For now, return example data
        return {
            "notes": [60, 63, 65, 67, 65, 63, 60, 58],  # C, Eb, F, G, F, Eb, C, Bb
            "rhythm": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            "velocities": [100, 80, 90, 100, 90, 80, 100, 80]
        }
    
    def _create_midi_clip_from_data(self, riff_data: Dict[str, Any], context: Dict[str, Any]) -> MIDIClip:
        """Create MIDI clip from parsed data"""
        from ..models.midi_clips import MIDIClip, MIDINote
        
        clip = MIDIClip(
            name=f"ai_riff_{int(time.time())}",
            length_bars=context.get("length_bars", 2.0),
            bpm=context.get("bpm", 180.0)
        )
        
        current_time = 0.0
        for note, duration, velocity in zip(
            riff_data["notes"],
            riff_data["rhythm"],
            riff_data["velocities"]
        ):
            clip.add_note(MIDINote(
                pitch=note,
                velocity=velocity,
                start_time=current_time,
                duration=duration
            ))
            current_time += duration
        
        return clip
    
    def _generate_algorithmic_riff(self, context: Dict[str, Any]) -> MIDIClip:
        """Fallback algorithmic riff generation"""
        from ..models.midi_clips import MIDIClip, MIDINote
        
        clip = MIDIClip(
            name=f"algo_riff_{int(time.time())}",
            length_bars=2.0,
            bpm=context.get("bpm", 180.0)
        )
        
        # Simple algorithmic pattern
        scale = [60, 62, 63, 65, 67, 68, 70, 72]  # C minor scale
        pattern_length = 8
        
        for i in range(pattern_length):
            note = np.random.choice(scale)
            velocity = np.random.randint(70, 110)
            clip.add_note(MIDINote(
                pitch=note,
                velocity=velocity,
                start_time=i * 0.25,
                duration=0.25
            ))
        
        return clip
    
    async def learn_vocabulary_mapping(self, user_term: str, technical_term: str):
        """Learn how user describes technical concepts"""
        self.user_preferences.vocabulary_mappings[user_term.lower()] = technical_term
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO vocabulary (user_id, user_term, technical_term, usage_count)
                    VALUES (?, ?, ?, 
                        COALESCE((SELECT usage_count + 1 FROM vocabulary 
                                 WHERE user_id = ? AND user_term = ?), 1))
                """, (self.user_id, user_term.lower(), technical_term, self.user_id, user_term.lower()))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving vocabulary mapping: {e}")
    
    def translate_user_input(self, user_input: str) -> str:
        """Translate user vocabulary to technical terms"""
        translated = user_input
        for user_term, technical_term in self.user_preferences.vocabulary_mappings.items():
            if user_term in translated.lower():
                translated = translated.replace(user_term, technical_term)
        return translated
    
    async def save_pattern_with_metrics(self, pattern: HardcorePattern, metrics: PatternQualityMetrics):
        """Save pattern with quality metrics to database"""
        try:
            pattern_data = {
                "name": pattern.name,
                "bpm": pattern.bpm,
                "genre": pattern.genre,
                "pattern_data": pattern.pattern_data,
                "synth_type": pattern.synth_type.value if hasattr(pattern.synth_type, 'value') else str(pattern.synth_type)
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO patterns (pattern_id, user_id, session_id, pattern_data, quality_metrics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    pattern.name,
                    self.user_id,
                    self.session_id,
                    json.dumps(pattern_data),
                    json.dumps(asdict(metrics)),
                    datetime.now()
                ))
                conn.commit()
                
            # Update successful patterns count if quality is good
            if metrics.psychoacoustic_score > 0.7:
                self.user_preferences.successful_patterns += 1
                self.user_preferences.total_patterns_created += 1
                await self._save_user_preferences()
                
        except Exception as e:
            self.logger.error(f"Error saving pattern: {e}")
    
    async def _save_insight(self, insight: LearningInsight):
        """Save learning insight to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO insights 
                    (insight_id, user_id, insight_type, content, confidence, usage_count, success_rate, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.insight_id,
                    self.user_id,
                    insight.insight_type,
                    json.dumps(insight.content),
                    insight.confidence,
                    insight.usage_count,
                    insight.success_rate,
                    datetime.now()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving insight: {e}")
    
    async def get_personalized_suggestion(self) -> str:
        """Generate personalized suggestion based on learned preferences"""
        # Analyze recent patterns
        recent_patterns = self._get_recent_patterns(limit=5)
        
        # Use AI to generate suggestion
        if self.anthropic_client:
            prompt = f"""Based on user's history:
- Preferred genres: {self.user_preferences.preferred_genres}
- Recent patterns: {len(recent_patterns)} created
- Success rate: {self.user_preferences.successful_patterns / max(1, self.user_preferences.total_patterns_created):.1%}
- Technical level: {self.user_preferences.technical_level}/5

Suggest the next creative step. Be specific and build on their strengths."""
            
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    system=self.system_prompts["creative_generator"],
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                self.logger.error(f"Error getting personalized suggestion: {e}")
        
        # Fallback suggestion
        if self.user_preferences.preferred_genres:
            top_genre = max(self.user_preferences.preferred_genres.items(), key=lambda x: x[1])[0]
            return f"Try creating a new {top_genre} pattern with more aggressive kicks"
        return "Experiment with adding acid basslines to your kicks"
    
    def _get_recent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent patterns from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern_data, quality_metrics, user_rating
                    FROM patterns
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (self.user_id, limit))
                
                patterns = []
                for row in cursor:
                    patterns.append({
                        "data": json.loads(row[0]),
                        "metrics": json.loads(row[1]) if row[1] else None,
                        "rating": row[2]
                    })
                return patterns
        except Exception as e:
            self.logger.error(f"Error loading recent patterns: {e}")
            return []
    
    async def update_pattern_rating(self, pattern_id: str, rating: float):
        """Update user rating for a pattern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE patterns
                    SET user_rating = ?, play_count = play_count + 1
                    WHERE pattern_id = ? AND user_id = ?
                """, (rating, pattern_id, self.user_id))
                conn.commit()
                
            # Learn from rating
            if rating >= 4.0:  # Good rating
                # Increase preference for this pattern's characteristics
                if pattern_id in self.generated_patterns:
                    metrics = self.generated_patterns[pattern_id]
                    # Update quality thresholds based on successful pattern
                    for key in ["psychoacoustic_score", "danceability", "energy_level"]:
                        value = getattr(metrics, key, 0)
                        if value > 0:
                            self.user_preferences.quality_thresholds[key] = value * 0.9
                
        except Exception as e:
            self.logger.error(f"Error updating pattern rating: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what the agent has learned"""
        return {
            "user_id": self.user_id,
            "patterns_created": self.user_preferences.total_patterns_created,
            "success_rate": self.user_preferences.successful_patterns / max(1, self.user_preferences.total_patterns_created),
            "preferred_genres": dict(sorted(
                self.user_preferences.preferred_genres.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]),
            "vocabulary_learned": len(self.user_preferences.vocabulary_mappings),
            "quality_standards": self.user_preferences.quality_thresholds,
            "insights_gathered": len(self.learning_insights)
        }


# Factory function
def create_intelligent_agent_v2(synthesizer: AbstractSynthesizer, 
                                user_id: str = "default") -> IntelligentMusicAgentV2:
    """Create enhanced intelligent music agent with persistent learning"""
    return IntelligentMusicAgentV2(synthesizer, user_id=user_id)


# Test
if __name__ == "__main__":
    import asyncio
    from ..interfaces.synthesizer import MockSynthesizer
    
    async def test_agent_v2():
        print("🧠 Testing Enhanced Intelligent Agent V2")
        print("=" * 50)
        
        synth = MockSynthesizer()
        await synth.start()
        
        agent = create_intelligent_agent_v2(synth, user_id="test_user")
        
        # Test learning
        print("\n1. Learning vocabulary...")
        await agent.learn_vocabulary_mapping("make it harder", "increase distortion")
        await agent.learn_vocabulary_mapping("more punch", "boost attack transient")
        
        # Test translation
        translated = agent.translate_user_input("make it harder with more punch")
        print(f"   Translated: {translated}")
        
        # Test pattern generation and analysis
        print("\n2. Generating and analyzing pattern...")
        from ..models.hardcore_models import HardcorePattern, SynthType
        
        pattern = HardcorePattern(
            name="test_pattern",
            bpm=180,
            pattern_data="x ~ x ~ x ~ x ~",
            synth_type=SynthType.GABBER_KICK,
            genre="gabber"
        )
        
        # Simulate audio data
        audio = np.random.randn(44100) * 0.5
        
        analysis, improvements = await agent.analyze_and_improve(audio, pattern)
        print(f"   Analysis: {analysis}")
        print(f"   Improvements: {improvements[:3]}")
        
        # Test personalized suggestion
        print("\n3. Getting personalized suggestion...")
        suggestion = await agent.get_personalized_suggestion()
        print(f"   Suggestion: {suggestion}")
        
        # Show learning summary
        print("\n4. Learning Summary:")
        summary = agent.get_learning_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("\n✨ Enhanced Agent V2 test complete!")
    
    asyncio.run(test_agent_v2())