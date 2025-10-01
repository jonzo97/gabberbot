#!/usr/bin/env python3
"""
Conversation Memory System for Music Production Sessions
Maintains context, preferences, and learning across conversations
"""

import json
import logging
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import sqlite3
from enum import Enum

from ..models.hardcore_models import HardcorePattern, SynthParams


logger = logging.getLogger(__name__)


class MemoryType(Enum):
    USER_PREFERENCE = "user_preference"
    PATTERN_HISTORY = "pattern_history"
    CONVERSATION_CONTEXT = "conversation_context"
    LEARNING_INSIGHT = "learning_insight"
    ERROR_PATTERN = "error_pattern"


class PreferenceCategory(Enum):
    GENRE = "genre"
    BPM = "bpm"
    SOUND_DESIGN = "sound_design"
    ARRANGEMENT = "arrangement"
    ARTIST_STYLE = "artist_style"
    SYNTHESIS = "synthesis"


@dataclass
class ConversationMemory:
    """Single memory entry for conversations"""
    id: str
    session_id: str
    memory_type: MemoryType
    timestamp: float
    content: Dict[str, Any]
    importance: float = 0.5  # 0.0 - 1.0
    decay_factor: float = 0.95  # How quickly memory fades
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass
class UserProfile:
    """User profile with learned preferences"""
    user_id: str
    preferred_bpm_range: List[int] = None
    favorite_genres: List[str] = None
    preferred_artists: List[str] = None
    synthesis_preferences: Dict[str, float] = None
    arrangement_style: str = "standard"
    technical_level: int = 3  # 1-5 scale
    created_patterns: int = 0
    total_sessions: int = 0
    last_active: float = 0.0
    
    def __post_init__(self):
        if self.preferred_bpm_range is None:
            self.preferred_bpm_range = [150, 180]
        if self.favorite_genres is None:
            self.favorite_genres = ["gabber"]
        if self.preferred_artists is None:
            self.preferred_artists = []
        if self.synthesis_preferences is None:
            self.synthesis_preferences = {
                "distortion": 0.8,
                "filter_cutoff": 0.6,
                "reverb": 0.3,
                "compression": 0.7
            }


class ConversationMemorySystem:
    """Advanced memory system for music production conversations"""
    
    def __init__(self, memory_dir: str = "/tmp/gabberbot_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.db_path = self.memory_dir / "conversation_memory.db"
        self.user_profiles_path = self.memory_dir / "user_profiles.json"
        
        self.memories: Dict[str, ConversationMemory] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Memory configuration
        self.max_memories_per_session = 100
        self.memory_decay_interval = 3600  # 1 hour
        self.importance_threshold = 0.1
        
        self._initialize_database()
        self._load_user_profiles()
        self._load_memories()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent memory storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    decay_factor REAL DEFAULT 0.95,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0.0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    bpm INTEGER NOT NULL,
                    genre TEXT NOT NULL,
                    rating REAL DEFAULT 0.0,
                    created_timestamp REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id);
            """)
            conn.commit()
    
    def _load_user_profiles(self):
        """Load user profiles from JSON"""
        if self.user_profiles_path.exists():
            try:
                with open(self.user_profiles_path, 'r') as f:
                    profiles_data = json.load(f)
                    
                for user_id, profile_data in profiles_data.items():
                    self.user_profiles[user_id] = UserProfile(user_id=user_id, **profile_data)
                    
                logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            except Exception as e:
                logger.error(f"Failed to load user profiles: {e}")
    
    def _save_user_profiles(self):
        """Save user profiles to JSON"""
        try:
            profiles_data = {
                user_id: asdict(profile) 
                for user_id, profile in self.user_profiles.items()
            }
            
            with open(self.user_profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save user profiles: {e}")
    
    def _load_memories(self):
        """Load recent memories from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load memories from last 24 hours
                cutoff_time = time.time() - 86400
                
                cursor = conn.execute("""
                    SELECT id, session_id, memory_type, timestamp, content, 
                           importance, decay_factor, access_count, last_accessed
                    FROM memories 
                    WHERE timestamp > ? AND importance > ?
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT 1000
                """, (cutoff_time, self.importance_threshold))
                
                for row in cursor.fetchall():
                    memory = ConversationMemory(
                        id=row[0],
                        session_id=row[1],
                        memory_type=MemoryType(row[2]),
                        timestamp=row[3],
                        content=json.loads(row[4]),
                        importance=row[5],
                        decay_factor=row[6],
                        access_count=row[7],
                        last_accessed=row[8]
                    )
                    self.memories[memory.id] = memory
                
                logger.info(f"Loaded {len(self.memories)} memories from database")
                
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def _save_memory_to_db(self, memory: ConversationMemory):
        """Save single memory to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, session_id, memory_type, timestamp, content, importance, 
                     decay_factor, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id, memory.session_id, memory.memory_type.value,
                    memory.timestamp, json.dumps(memory.content), memory.importance,
                    memory.decay_factor, memory.access_count, memory.last_accessed
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save memory to database: {e}")
    
    def add_memory(self, 
                   session_id: str,
                   memory_type: MemoryType,
                   content: Dict[str, Any],
                   importance: float = 0.5) -> str:
        """Add a new memory entry"""
        
        # Generate unique memory ID
        memory_id = hashlib.md5(
            f"{session_id}_{memory_type.value}_{time.time()}".encode()
        ).hexdigest()
        
        memory = ConversationMemory(
            id=memory_id,
            session_id=session_id,
            memory_type=memory_type,
            timestamp=time.time(),
            content=content,
            importance=importance
        )
        
        # Store in memory and database
        self.memories[memory_id] = memory
        self._save_memory_to_db(memory)
        
        # Clean up old memories if needed
        self._cleanup_session_memories(session_id)
        
        logger.debug(f"Added memory: {memory_type.value} for session {session_id}")
        return memory_id
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get relevant context for a session"""
        session_memories = [
            memory for memory in self.memories.values()
            if memory.session_id == session_id
        ]
        
        # Sort by importance and recency
        session_memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        context = {
            "recent_patterns": [],
            "user_preferences": {},
            "conversation_themes": [],
            "technical_insights": [],
            "error_patterns": []
        }
        
        for memory in session_memories[:20]:  # Top 20 most relevant
            memory.access_count += 1
            memory.last_accessed = time.time()
            
            if memory.memory_type == MemoryType.PATTERN_HISTORY:
                context["recent_patterns"].append(memory.content)
            elif memory.memory_type == MemoryType.USER_PREFERENCE:
                context["user_preferences"].update(memory.content)
            elif memory.memory_type == MemoryType.CONVERSATION_CONTEXT:
                context["conversation_themes"].append(memory.content.get("theme", ""))
            elif memory.memory_type == MemoryType.LEARNING_INSIGHT:
                context["technical_insights"].append(memory.content)
            elif memory.memory_type == MemoryType.ERROR_PATTERN:
                context["error_patterns"].append(memory.content)
        
        return context
    
    def learn_user_preference(self, 
                            session_id: str,
                            category: PreferenceCategory,
                            preference_data: Dict[str, Any],
                            confidence: float = 0.8):
        """Learn and store user preferences"""
        
        content = {
            "category": category.value,
            "preference": preference_data,
            "confidence": confidence,
            "learned_from": "conversation_analysis"
        }
        
        # Add to memory with high importance
        memory_id = self.add_memory(
            session_id=session_id,
            memory_type=MemoryType.USER_PREFERENCE,
            content=content,
            importance=0.8
        )
        
        # Update user profile if we have enough confidence
        if confidence > 0.7:
            self._update_user_profile(session_id, category, preference_data)
        
        return memory_id
    
    def _update_user_profile(self, 
                           session_id: str,
                           category: PreferenceCategory,
                           preference_data: Dict[str, Any]):
        """Update user profile with learned preferences"""
        
        # Use session_id as user_id for now (could be more sophisticated)
        user_id = f"user_{session_id.split('_')[0]}"
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        profile.last_active = time.time()
        
        if category == PreferenceCategory.BPM:
            if "preferred_bpm" in preference_data:
                bpm = preference_data["preferred_bpm"]
                profile.preferred_bpm_range = [max(100, bpm - 10), min(250, bpm + 10)]
        
        elif category == PreferenceCategory.GENRE:
            if "genre" in preference_data:
                genre = preference_data["genre"]
                if genre not in profile.favorite_genres:
                    profile.favorite_genres.insert(0, genre)
                    profile.favorite_genres = profile.favorite_genres[:5]  # Keep top 5
        
        elif category == PreferenceCategory.ARTIST_STYLE:
            if "artist" in preference_data:
                artist = preference_data["artist"]
                if artist not in profile.preferred_artists:
                    profile.preferred_artists.insert(0, artist)
                    profile.preferred_artists = profile.preferred_artists[:10]
        
        elif category == PreferenceCategory.SYNTHESIS:
            if "synth_params" in preference_data:
                params = preference_data["synth_params"]
                # Update synthesis preferences with weighted average
                for param, value in params.items():
                    if param in profile.synthesis_preferences:
                        # Weighted average: 70% existing, 30% new
                        profile.synthesis_preferences[param] = (
                            0.7 * profile.synthesis_preferences[param] + 0.3 * value
                        )
                    else:
                        profile.synthesis_preferences[param] = value
        
        self._save_user_profiles()
        logger.info(f"Updated user profile for {user_id}: {category.value}")
    
    def record_pattern_creation(self, 
                              session_id: str,
                              pattern: HardcorePattern,
                              user_feedback: Optional[str] = None,
                              rating: float = 0.0):
        """Record pattern creation for learning"""
        
        content = {
            "pattern_name": pattern.name,
            "bpm": pattern.bpm,
            "genre": pattern.genre,
            "synth_type": pattern.synth_type.value,
            "user_feedback": user_feedback,
            "rating": rating,
            "pattern_complexity": len(pattern.pattern_data) if pattern.pattern_data else 0
        }
        
        # Add to memory
        memory_id = self.add_memory(
            session_id=session_id,
            memory_type=MemoryType.PATTERN_HISTORY,
            content=content,
            importance=min(0.9, 0.5 + rating * 0.4)  # Higher rated patterns are more important
        )
        
        # Save pattern to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO patterns (id, session_id, pattern_data, bpm, genre, rating, created_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.name, session_id, json.dumps(asdict(pattern)), 
                    pattern.bpm, pattern.genre, rating, time.time()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
        
        # Learn preferences from pattern
        if rating > 0.6:  # Good rating, learn from it
            self.learn_user_preference(
                session_id=session_id,
                category=PreferenceCategory.BPM,
                preference_data={"preferred_bpm": pattern.bpm},
                confidence=min(0.9, rating)
            )
            
            self.learn_user_preference(
                session_id=session_id,
                category=PreferenceCategory.GENRE,
                preference_data={"genre": pattern.genre},
                confidence=min(0.9, rating)
            )
        
        return memory_id
    
    def get_similar_patterns(self, 
                           session_id: str,
                           bpm: int,
                           genre: str,
                           limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar patterns from memory"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern_data, rating, created_timestamp
                    FROM patterns 
                    WHERE abs(bpm - ?) <= 20 
                    AND (genre = ? OR session_id = ?)
                    AND rating > 0.5
                    ORDER BY rating DESC, created_timestamp DESC
                    LIMIT ?
                """, (bpm, genre, session_id, limit))
                
                patterns = []
                for row in cursor.fetchall():
                    pattern_data = json.loads(row[0])
                    patterns.append({
                        "pattern": pattern_data,
                        "rating": row[1],
                        "created": row[2]
                    })
                
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to get similar patterns: {e}")
            return []
    
    def record_error_pattern(self, 
                           session_id: str,
                           error_type: str,
                           error_context: Dict[str, Any],
                           resolution: Optional[str] = None):
        """Record error patterns for learning and prevention"""
        
        content = {
            "error_type": error_type,
            "context": error_context,
            "resolution": resolution,
            "frequency": 1
        }
        
        # Check if similar error exists
        existing_errors = [
            m for m in self.memories.values()
            if (m.memory_type == MemoryType.ERROR_PATTERN and 
                m.content.get("error_type") == error_type)
        ]
        
        if existing_errors:
            # Update frequency
            latest_error = max(existing_errors, key=lambda m: m.timestamp)
            latest_error.content["frequency"] += 1
            latest_error.importance = min(0.95, latest_error.importance + 0.1)
            latest_error.timestamp = time.time()
            self._save_memory_to_db(latest_error)
        else:
            # New error pattern
            self.add_memory(
                session_id=session_id,
                memory_type=MemoryType.ERROR_PATTERN,
                content=content,
                importance=0.7
            )
    
    def get_user_profile(self, session_id: str) -> UserProfile:
        """Get or create user profile for session"""
        user_id = f"user_{session_id.split('_')[0]}"
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        return self.user_profiles[user_id]
    
    def _cleanup_session_memories(self, session_id: str):
        """Clean up old memories for a session"""
        session_memories = [
            m for m in self.memories.values()
            if m.session_id == session_id
        ]
        
        if len(session_memories) > self.max_memories_per_session:
            # Sort by importance and keep the most important ones
            session_memories.sort(key=lambda m: m.importance, reverse=True)
            
            to_remove = session_memories[self.max_memories_per_session:]
            for memory in to_remove:
                del self.memories[memory.id]
    
    def decay_memories(self):
        """Apply decay to memory importance over time"""
        current_time = time.time()
        
        for memory in self.memories.values():
            age_hours = (current_time - memory.timestamp) / 3600
            
            # Apply decay based on age and access patterns
            decay_amount = memory.decay_factor ** age_hours
            if memory.access_count > 0:
                # Recently accessed memories decay slower
                decay_amount = decay_amount ** (1 + memory.access_count * 0.1)
            
            memory.importance *= decay_amount
            
            # Remove memories below threshold
            if memory.importance < self.importance_threshold:
                del self.memories[memory.id]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        total_memories = len(self.memories)
        
        memory_by_type = {}
        for memory in self.memories.values():
            memory_type = memory.memory_type.value
            memory_by_type[memory_type] = memory_by_type.get(memory_type, 0) + 1
        
        avg_importance = (
            sum(m.importance for m in self.memories.values()) / total_memories
            if total_memories > 0 else 0
        )
        
        return {
            "total_memories": total_memories,
            "memory_by_type": memory_by_type,
            "average_importance": avg_importance,
            "total_user_profiles": len(self.user_profiles),
            "database_path": str(self.db_path),
            "memory_threshold": self.importance_threshold
        }


# Global memory system instance
memory_system = ConversationMemorySystem()


if __name__ == "__main__":
    # Demo the memory system
    import asyncio
    from ..models.hardcore_models import HardcorePattern, SynthType
    
    memory = ConversationMemorySystem(memory_dir="/tmp/demo_memory")
    
    print("=== Conversation Memory System Demo ===")
    
    # Create test session
    session_id = "demo_session_001"
    
    # Add some memories
    memory.add_memory(
        session_id=session_id,
        memory_type=MemoryType.CONVERSATION_CONTEXT,
        content={"theme": "gabber_production", "bpm": 180},
        importance=0.8
    )
    
    # Learn user preference
    memory.learn_user_preference(
        session_id=session_id,
        category=PreferenceCategory.BPM,
        preference_data={"preferred_bpm": 180},
        confidence=0.9
    )
    
    # Record pattern creation
    test_pattern = HardcorePattern(
        name="test_gabber_001",
        bpm=180,
        pattern_data='s("bd:5").struct("x ~ x ~").shape(0.9)',
        synth_type=SynthType.GABBER_KICK,
        genre="gabber"
    )
    
    memory.record_pattern_creation(
        session_id=session_id,
        pattern=test_pattern,
        user_feedback="Love the crunch!",
        rating=0.9
    )
    
    # Get session context
    context = memory.get_session_context(session_id)
    print(f"Session context: {context}")
    
    # Get user profile
    profile = memory.get_user_profile(session_id)
    print(f"User profile BPM range: {profile.preferred_bpm_range}")
    
    # Get memory stats
    stats = memory.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    print("\n=== Demo completed ===")