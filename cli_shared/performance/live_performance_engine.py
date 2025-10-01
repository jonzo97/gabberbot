#!/usr/bin/env python3
"""
Live Performance Engine for Hardcore Music Production
Real-time pattern switching, crossfading, and performance features
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import queue
import json
from collections import deque

import numpy as np

from ..interfaces.synthesizer import AbstractSynthesizer
from ..models.hardcore_models import HardcorePattern, SynthParams, SynthType
from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer


logger = logging.getLogger(__name__)


class PerformanceMode(Enum):
    LIVE_SET = "live_set"           # Pre-arranged set with timed transitions
    JAM_SESSION = "jam_session"     # Free-form jamming with manual control
    DJ_MIX = "dj_mix"              # Crossfader-style mixing between patterns
    BATTLE_MODE = "battle_mode"     # Competitive pattern battles
    IMPROVISATION = "improvisation" # AI-assisted live improvisation


class TransitionType(Enum):
    HARD_CUT = "hard_cut"           # Instant pattern switch
    CROSSFADE = "crossfade"         # Gradual volume crossfade
    FILTER_SWEEP = "filter_sweep"   # Filter-based transition
    BREAKDOWN = "breakdown"         # Strip elements then build up
    SLAM = "slam"                   # Dramatic silence then drop
    MORPH = "morph"                # Gradual parameter morphing


class PerformanceState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    TRANSITIONING = "transitioning"
    PAUSED = "paused"
    RECORDING = "recording"


@dataclass
class PatternSlot:
    """A single pattern slot in the performance engine"""
    slot_id: str
    pattern: Optional[HardcorePattern] = None
    is_active: bool = False
    volume: float = 1.0
    filter_cutoff: float = 1.0
    is_muted: bool = False
    is_soloed: bool = False
    last_triggered: float = 0.0
    play_count: int = 0
    
    def activate(self):
        self.is_active = True
        self.last_triggered = time.time()
        self.play_count += 1


@dataclass
class TransitionParams:
    """Parameters for pattern transitions"""
    transition_type: TransitionType
    duration_beats: float = 4.0
    curve_type: str = "linear"  # linear, exponential, logarithmic
    sync_to_bar: bool = True
    pre_transition_fx: Optional[Dict[str, Any]] = None
    post_transition_fx: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceEvent:
    """Events for performance automation"""
    timestamp: float
    event_type: str
    target_slot: str
    parameters: Dict[str, Any]
    is_executed: bool = False


@dataclass
class LivePerformanceState:
    """Complete state of the live performance"""
    mode: PerformanceMode
    state: PerformanceState
    current_bpm: float
    master_volume: float
    crossfader_position: float  # -1.0 to 1.0
    pattern_slots: Dict[str, PatternSlot]
    active_transitions: List[Dict[str, Any]]
    performance_start_time: float
    total_patterns_played: int
    audience_energy: float  # 0.0 to 1.0 (mock audience response)


class PerformanceAutomation:
    """Automation system for live performances"""
    
    def __init__(self):
        self.events: List[PerformanceEvent] = []
        self.is_recording = False
        self.playback_start_time = 0.0
        
    def record_event(self, event_type: str, target_slot: str, parameters: Dict[str, Any]):
        """Record a performance event"""
        if not self.is_recording:
            return
            
        event = PerformanceEvent(
            timestamp=time.time() - self.playback_start_time,
            event_type=event_type,
            target_slot=target_slot,
            parameters=parameters
        )
        self.events.append(event)
        logger.debug(f"Recorded event: {event_type} on {target_slot}")
    
    def start_recording(self):
        """Start recording performance automation"""
        self.is_recording = True
        self.playback_start_time = time.time()
        self.events.clear()
        logger.info("Started performance automation recording")
    
    def stop_recording(self):
        """Stop recording performance automation"""
        self.is_recording = False
        logger.info(f"Stopped recording, captured {len(self.events)} events")
    
    def get_events_at_time(self, current_time: float) -> List[PerformanceEvent]:
        """Get events that should be executed at current time"""
        tolerance = 0.05  # 50ms tolerance
        
        return [
            event for event in self.events
            if (not event.is_executed and 
                abs(event.timestamp - current_time) <= tolerance)
        ]
    
    def save_automation(self, filename: str):
        """Save automation to file"""
        try:
            automation_data = {
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "target_slot": event.target_slot,
                        "parameters": event.parameters
                    }
                    for event in self.events
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(automation_data, f, indent=2)
                
            logger.info(f"Saved automation to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save automation: {e}")
    
    def load_automation(self, filename: str):
        """Load automation from file"""
        try:
            with open(filename, 'r') as f:
                automation_data = json.load(f)
            
            self.events = [
                PerformanceEvent(
                    timestamp=event_data["timestamp"],
                    event_type=event_data["event_type"],
                    target_slot=event_data["target_slot"],
                    parameters=event_data["parameters"]
                )
                for event_data in automation_data["events"]
            ]
            
            logger.info(f"Loaded {len(self.events)} automation events from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load automation: {e}")


class PatternTransitioner:
    """Handles smooth transitions between patterns"""
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.synthesizer = synthesizer
        self.active_transitions = []
        
    async def transition(self, 
                        from_slot: PatternSlot,
                        to_slot: PatternSlot,
                        params: TransitionParams) -> bool:
        """Execute a pattern transition"""
        
        try:
            logger.info(f"Starting transition: {params.transition_type.value}")
            
            if params.transition_type == TransitionType.HARD_CUT:
                await self._hard_cut_transition(from_slot, to_slot)
            
            elif params.transition_type == TransitionType.CROSSFADE:
                await self._crossfade_transition(from_slot, to_slot, params)
            
            elif params.transition_type == TransitionType.FILTER_SWEEP:
                await self._filter_sweep_transition(from_slot, to_slot, params)
            
            elif params.transition_type == TransitionType.BREAKDOWN:
                await self._breakdown_transition(from_slot, to_slot, params)
            
            elif params.transition_type == TransitionType.SLAM:
                await self._slam_transition(from_slot, to_slot, params)
            
            elif params.transition_type == TransitionType.MORPH:
                await self._morph_transition(from_slot, to_slot, params)
            
            return True
            
        except Exception as e:
            logger.error(f"Transition failed: {e}")
            return False
    
    async def _hard_cut_transition(self, from_slot: PatternSlot, to_slot: PatternSlot):
        """Instant pattern switch"""
        from_slot.is_active = False
        to_slot.activate()
        
        if to_slot.pattern:
            await self.synthesizer.play_pattern(to_slot.pattern)
    
    async def _crossfade_transition(self, from_slot: PatternSlot, to_slot: PatternSlot, params: TransitionParams):
        """Gradual volume crossfade"""
        duration_seconds = params.duration_beats * (60.0 / 150.0)  # Assume 150 BPM
        steps = int(duration_seconds * 20)  # 20 updates per second
        
        to_slot.activate()
        if to_slot.pattern:
            await self.synthesizer.play_pattern(to_slot.pattern)
        
        for i in range(steps):
            progress = i / (steps - 1)
            
            # Apply crossfade curve
            if params.curve_type == "exponential":
                progress = progress ** 2
            elif params.curve_type == "logarithmic":
                progress = np.sqrt(progress)
            
            from_slot.volume = 1.0 - progress
            to_slot.volume = progress
            
            # Update synthesizer volumes (implementation specific)
            await asyncio.sleep(duration_seconds / steps)
        
        from_slot.is_active = False
        from_slot.volume = 1.0
        to_slot.volume = 1.0
    
    async def _filter_sweep_transition(self, from_slot: PatternSlot, to_slot: PatternSlot, params: TransitionParams):
        """Filter-based transition"""
        duration_seconds = params.duration_beats * (60.0 / 150.0)
        
        # Start new pattern
        to_slot.activate()
        if to_slot.pattern:
            await self.synthesizer.play_pattern(to_slot.pattern)
        
        # Filter sweep on outgoing pattern
        steps = 20
        for i in range(steps):
            progress = i / (steps - 1)
            from_slot.filter_cutoff = 1.0 - progress
            to_slot.filter_cutoff = progress
            
            await asyncio.sleep(duration_seconds / steps)
        
        from_slot.is_active = False
        from_slot.filter_cutoff = 1.0
        to_slot.filter_cutoff = 1.0
    
    async def _breakdown_transition(self, from_slot: PatternSlot, to_slot: PatternSlot, params: TransitionParams):
        """Breakdown and buildup transition"""
        duration_seconds = params.duration_beats * (60.0 / 150.0)
        half_duration = duration_seconds / 2
        
        # Breakdown phase - strip elements
        from_slot.filter_cutoff = 0.3
        from_slot.volume = 0.5
        await asyncio.sleep(half_duration)
        
        # Buildup phase - introduce new pattern
        from_slot.is_active = False
        to_slot.activate()
        
        if to_slot.pattern:
            to_slot.filter_cutoff = 0.3
            to_slot.volume = 0.5
            await self.synthesizer.play_pattern(to_slot.pattern)
            
            # Gradual buildup
            steps = 10
            for i in range(steps):
                progress = i / (steps - 1)
                to_slot.filter_cutoff = 0.3 + (0.7 * progress)
                to_slot.volume = 0.5 + (0.5 * progress)
                await asyncio.sleep(half_duration / steps)
        
        # Reset parameters
        from_slot.filter_cutoff = 1.0
        from_slot.volume = 1.0
        to_slot.filter_cutoff = 1.0
        to_slot.volume = 1.0
    
    async def _slam_transition(self, from_slot: PatternSlot, to_slot: PatternSlot, params: TransitionParams):
        """Dramatic silence then drop"""
        silence_duration = 0.5  # 500ms of silence
        
        # Silence
        from_slot.volume = 0.0
        await asyncio.sleep(silence_duration)
        
        # SLAM!
        from_slot.is_active = False
        to_slot.activate()
        to_slot.volume = 1.2  # Slightly louder for impact
        
        if to_slot.pattern:
            await self.synthesizer.play_pattern(to_slot.pattern)
        
        # Return to normal volume
        await asyncio.sleep(0.1)
        to_slot.volume = 1.0
        from_slot.volume = 1.0
    
    async def _morph_transition(self, from_slot: PatternSlot, to_slot: PatternSlot, params: TransitionParams):
        """Gradual parameter morphing"""
        # This would require more sophisticated parameter interpolation
        # For now, implement as a crossfade
        await self._crossfade_transition(from_slot, to_slot, params)


class LivePerformanceEngine:
    """Main live performance engine"""
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.synthesizer = synthesizer
        self.audio_analyzer = AdvancedAudioAnalyzer()
        self.transitioner = PatternTransitioner(synthesizer)
        self.automation = PerformanceAutomation()
        
        # Performance state
        self.state = LivePerformanceState(
            mode=PerformanceMode.JAM_SESSION,
            state=PerformanceState.STOPPED,
            current_bpm=150.0,
            master_volume=1.0,
            crossfader_position=0.0,
            pattern_slots={},
            active_transitions=[],
            performance_start_time=0.0,
            total_patterns_played=0,
            audience_energy=0.5
        )
        
        # Initialize pattern slots (8 slots for live performance)
        for i in range(8):
            slot_id = f"slot_{i:02d}"
            self.state.pattern_slots[slot_id] = PatternSlot(slot_id=slot_id)
        
        # Performance statistics
        self.performance_stats = {
            "transitions_executed": 0,
            "patterns_played": 0,
            "performance_duration": 0.0,
            "audience_engagement": deque(maxlen=100),  # Rolling average
            "energy_curve": deque(maxlen=1000)
        }
        
        # Event queue for real-time performance
        self.event_queue = asyncio.Queue()
        self.is_running = False
        
    async def start_performance(self, mode: PerformanceMode = PerformanceMode.JAM_SESSION):
        """Start live performance"""
        self.state.mode = mode
        self.state.state = PerformanceState.PLAYING
        self.state.performance_start_time = time.time()
        self.is_running = True
        
        logger.info(f"Started live performance in {mode.value} mode")
        
        # Start performance loop
        asyncio.create_task(self._performance_loop())
        
        # Start audience simulation
        asyncio.create_task(self._audience_simulation())
    
    async def stop_performance(self):
        """Stop live performance"""
        self.state.state = PerformanceState.STOPPED
        self.is_running = False
        
        # Stop all active patterns
        for slot in self.state.pattern_slots.values():
            slot.is_active = False
        
        # Calculate final stats
        duration = time.time() - self.state.performance_start_time
        self.performance_stats["performance_duration"] = duration
        
        logger.info(f"Performance stopped. Duration: {duration:.1f}s, Patterns played: {self.state.total_patterns_played}")
    
    async def load_pattern_to_slot(self, slot_id: str, pattern: HardcorePattern):
        """Load a pattern into a performance slot"""
        if slot_id not in self.state.pattern_slots:
            raise ValueError(f"Invalid slot ID: {slot_id}")
        
        slot = self.state.pattern_slots[slot_id]
        slot.pattern = pattern
        
        logger.info(f"Loaded pattern '{pattern.name}' to slot {slot_id}")
    
    async def trigger_slot(self, slot_id: str, transition_params: Optional[TransitionParams] = None):
        """Trigger a pattern slot"""
        if slot_id not in self.state.pattern_slots:
            raise ValueError(f"Invalid slot ID: {slot_id}")
        
        target_slot = self.state.pattern_slots[slot_id]
        
        if not target_slot.pattern:
            logger.warning(f"No pattern loaded in slot {slot_id}")
            return
        
        # Find currently active slot
        active_slots = [slot for slot in self.state.pattern_slots.values() if slot.is_active]
        
        if not active_slots:
            # No active pattern, just start this one
            target_slot.activate()
            await self.synthesizer.play_pattern(target_slot.pattern)
            self.state.total_patterns_played += 1
            
        else:
            # Transition from active pattern
            from_slot = active_slots[0]  # Assume single active pattern for simplicity
            
            if transition_params is None:
                transition_params = TransitionParams(TransitionType.CROSSFADE)
            
            # Record automation if recording
            self.automation.record_event(
                event_type="trigger_slot",
                target_slot=slot_id,
                parameters={"transition": transition_params.transition_type.value}
            )
            
            # Execute transition
            success = await self.transitioner.transition(from_slot, target_slot, transition_params)
            
            if success:
                self.performance_stats["transitions_executed"] += 1
                self.state.total_patterns_played += 1
        
        logger.info(f"Triggered slot {slot_id}")
    
    async def set_crossfader(self, position: float):
        """Set crossfader position (-1.0 to 1.0)"""
        position = max(-1.0, min(1.0, position))
        self.state.crossfader_position = position
        
        # Apply crossfader to slot volumes
        # Assuming slots 0-3 on left, 4-7 on right
        left_volume = max(0.0, 1.0 + position) if position < 0 else max(0.0, 1.0 - position)
        right_volume = max(0.0, 1.0 - position) if position > 0 else max(0.0, 1.0 + position)
        
        for i in range(4):
            self.state.pattern_slots[f"slot_{i:02d}"].volume = left_volume
            self.state.pattern_slots[f"slot_{i+4:02d}"].volume = right_volume
        
        # Record automation
        self.automation.record_event(
            event_type="crossfader",
            target_slot="master",
            parameters={"position": position}
        )
    
    async def set_master_volume(self, volume: float):
        """Set master volume (0.0 to 1.0)"""
        self.state.master_volume = max(0.0, min(1.0, volume))
        
        # Apply to synthesizer
        # Implementation would depend on synthesizer backend
        
        self.automation.record_event(
            event_type="master_volume",
            target_slot="master",
            parameters={"volume": volume}
        )
    
    async def apply_effect(self, slot_id: str, effect_type: str, parameters: Dict[str, Any]):
        """Apply real-time effect to a slot"""
        if slot_id not in self.state.pattern_slots:
            return
        
        slot = self.state.pattern_slots[slot_id]
        
        # Apply effect based on type
        if effect_type == "filter_sweep":
            cutoff = parameters.get("cutoff", 1.0)
            slot.filter_cutoff = cutoff
            
        elif effect_type == "volume":
            volume = parameters.get("volume", 1.0)
            slot.volume = volume
        
        # Record automation
        self.automation.record_event(
            event_type="effect",
            target_slot=slot_id,
            parameters={"effect_type": effect_type, **parameters}
        )
        
        logger.debug(f"Applied {effect_type} to slot {slot_id}")
    
    async def _performance_loop(self):
        """Main performance loop"""
        while self.is_running:
            try:
                current_time = time.time() - self.state.performance_start_time
                
                # Process automation events
                events = self.automation.get_events_at_time(current_time)
                for event in events:
                    await self._execute_automation_event(event)
                    event.is_executed = True
                
                # Update audience energy based on current patterns
                await self._update_audience_energy()
                
                # Performance analytics
                self._update_performance_stats()
                
                await asyncio.sleep(0.02)  # 50Hz update rate
                
            except Exception as e:
                logger.error(f"Performance loop error: {e}")
    
    async def _execute_automation_event(self, event: PerformanceEvent):
        """Execute an automation event"""
        try:
            if event.event_type == "trigger_slot":
                await self.trigger_slot(event.target_slot)
                
            elif event.event_type == "crossfader":
                position = event.parameters.get("position", 0.0)
                await self.set_crossfader(position)
                
            elif event.event_type == "master_volume":
                volume = event.parameters.get("volume", 1.0)
                await self.set_master_volume(volume)
                
            elif event.event_type == "effect":
                effect_type = event.parameters.get("effect_type", "")
                await self.apply_effect(event.target_slot, effect_type, event.parameters)
            
        except Exception as e:
            logger.error(f"Failed to execute automation event: {e}")
    
    async def _update_audience_energy(self):
        """Update audience energy simulation"""
        # Simulate audience response based on current patterns
        active_patterns = sum(1 for slot in self.state.pattern_slots.values() if slot.is_active)
        
        base_energy = 0.3 + (active_patterns * 0.2)
        
        # Add some randomness and momentum
        import random
        energy_change = random.uniform(-0.05, 0.05)
        self.state.audience_energy = max(0.0, min(1.0, self.state.audience_energy + energy_change))
        
        # Boost energy during transitions
        if self.state.active_transitions:
            self.state.audience_energy = min(1.0, self.state.audience_energy + 0.1)
        
        # Record energy curve
        self.performance_stats["energy_curve"].append(self.state.audience_energy)
        self.performance_stats["audience_engagement"].append(self.state.audience_energy)
    
    async def _audience_simulation(self):
        """Simulate audience response and feedback"""
        while self.is_running:
            try:
                # Generate audience events based on energy
                if self.state.audience_energy > 0.8:
                    # High energy - audience goes wild
                    logger.info("🔥 Audience energy: HIGH - Crowd going wild!")
                elif self.state.audience_energy > 0.6:
                    # Medium energy - good response
                    logger.debug("⚡ Audience energy: MEDIUM - Good vibes")
                elif self.state.audience_energy < 0.3:
                    # Low energy - need to step it up
                    logger.debug("😴 Audience energy: LOW - Need more energy!")
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Audience simulation error: {e}")
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        self.performance_stats["patterns_played"] = self.state.total_patterns_played
        
        # Calculate average audience engagement
        if self.performance_stats["audience_engagement"]:
            avg_engagement = sum(self.performance_stats["audience_engagement"]) / len(self.performance_stats["audience_engagement"])
        else:
            avg_engagement = 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        current_time = time.time()
        duration = current_time - self.state.performance_start_time if self.is_running else 0
        
        return {
            "mode": self.state.mode.value,
            "state": self.state.state.value,
            "duration_seconds": duration,
            "patterns_played": self.state.total_patterns_played,
            "transitions_executed": self.performance_stats["transitions_executed"],
            "current_bpm": self.state.current_bpm,
            "audience_energy": self.state.audience_energy,
            "crossfader_position": self.state.crossfader_position,
            "master_volume": self.state.master_volume,
            "active_slots": [
                slot_id for slot_id, slot in self.state.pattern_slots.items()
                if slot.is_active
            ]
        }
    
    def get_slot_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all pattern slots"""
        return {
            slot_id: {
                "pattern_name": slot.pattern.name if slot.pattern else None,
                "is_active": slot.is_active,
                "volume": slot.volume,
                "filter_cutoff": slot.filter_cutoff,
                "is_muted": slot.is_muted,
                "is_soloed": slot.is_soloed,
                "play_count": slot.play_count
            }
            for slot_id, slot in self.state.pattern_slots.items()
        }


# Factory function
def create_live_performance_engine(synthesizer: AbstractSynthesizer) -> LivePerformanceEngine:
    """Create a live performance engine with all dependencies"""
    return LivePerformanceEngine(synthesizer)


if __name__ == "__main__":
    # Demo the live performance engine
    import asyncio
    from ..interfaces.synthesizer import MockSynthesizer
    from ..models.hardcore_models import HardcorePattern, SynthType
    
    async def demo_live_performance():
        # Create mock synthesizer
        synth = MockSynthesizer()
        
        # Create performance engine
        engine = create_live_performance_engine(synth)
        
        # Create test patterns
        patterns = [
            HardcorePattern(
                name="Gabber_Kick_180",
                bpm=180,
                pattern_data='s("bd:5").struct("x ~ x ~").shape(0.9)',
                synth_type=SynthType.GABBER_KICK,
                genre="gabber"
            ),
            HardcorePattern(
                name="Industrial_Loop",
                bpm=140,
                pattern_data='s("bd:9").room(0.8).struct("x ~ ~ ~")',
                synth_type=SynthType.INDUSTRIAL_KICK,
                genre="industrial"
            )
        ]
        
        # Load patterns into slots
        await engine.load_pattern_to_slot("slot_00", patterns[0])
        await engine.load_pattern_to_slot("slot_01", patterns[1])
        
        print("=== Live Performance Engine Demo ===")
        
        # Start performance
        await engine.start_performance(PerformanceMode.JAM_SESSION)
        print("Performance started")
        
        # Trigger first pattern
        await engine.trigger_slot("slot_00")
        print("Triggered slot 00")
        await asyncio.sleep(2)
        
        # Crossfade to second pattern
        transition_params = TransitionParams(
            transition_type=TransitionType.CROSSFADE,
            duration_beats=8.0
        )
        await engine.trigger_slot("slot_01", transition_params)
        print("Crossfaded to slot 01")
        await asyncio.sleep(3)
        
        # Apply some effects
        await engine.apply_effect("slot_01", "filter_sweep", {"cutoff": 0.3})
        print("Applied filter sweep")
        await asyncio.sleep(1)
        
        # Use crossfader
        await engine.set_crossfader(-0.5)  # Favor left side
        print("Set crossfader to -0.5")
        await asyncio.sleep(2)
        
        # Get performance stats
        stats = engine.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"- Duration: {stats['duration_seconds']:.1f}s")
        print(f"- Patterns played: {stats['patterns_played']}")
        print(f"- Audience energy: {stats['audience_energy']:.2f}")
        print(f"- Transitions: {stats['transitions_executed']}")
        
        # Stop performance
        await engine.stop_performance()
        print("\nPerformance stopped")
    
    # Run demo
    asyncio.run(demo_live_performance())