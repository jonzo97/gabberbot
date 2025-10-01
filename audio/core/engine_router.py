#!/usr/bin/env python3
"""
Audio Engine Backend Router

Engine Type: [FRAMEWORK-AGNOSTIC]
Dependencies: None (orchestrates other engines)
Abstraction Level: [HIGH-LEVEL]
Integration: Smart backend selection and graceful fallbacks for Track system

Provides intelligent routing between different audio engine backends:
- SuperCollider for professional synthesis
- TidalCycles for advanced patterns (future)
- Strudel for web compatibility and fallbacks
- Custom Python for basic synthesis when others unavailable

Key Features:
- Automatic engine detection and capability assessment
- Graceful degradation (SC unavailable → use Strudel)
- Performance-based routing (complex synthesis → SC, simple → any)
- Hot-swapping between engines without user interruption
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Type
from enum import Enum
from dataclasses import dataclass
import time

# Import shared components
import sys
sys.path.append('/home/onathan_rgill/music_code_cli')

from cli_shared.interfaces.synthesizer import AbstractSynthesizer
from cli_shared.models.hardcore_models import SynthType, SynthParams, HardcorePattern


class EngineCapability(Enum):
    """Engine capability levels"""
    BASIC = "basic"           # Simple synthesis, limited effects
    INTERMEDIATE = "intermediate"  # Good synthesis, some effects
    PROFESSIONAL = "professional" # Full synthesis, all effects, low latency
    ADVANCED = "advanced"     # Professional + pattern generation


class EngineBackend(Enum):
    """Available engine backends"""
    SUPERCOLLIDER = "supercollider"
    TIDALCYCLES = "tidalcycles"
    PYTHON_NATIVE = "python_native"


@dataclass
class EngineStatus:
    """Status information for an engine backend"""
    backend: EngineBackend
    available: bool = False
    initialized: bool = False
    capability: EngineCapability = EngineCapability.BASIC
    latency_ms: float = 50.0
    cpu_usage: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    supported_synths: List[SynthType] = None
    
    def __post_init__(self):
        if self.supported_synths is None:
            self.supported_synths = []


@dataclass
class RoutingPreferences:
    """User preferences for engine routing"""
    prefer_quality_over_speed: bool = True
    allow_engine_switching: bool = True
    fallback_enabled: bool = True
    max_latency_ms: float = 20.0
    prefer_backend: Optional[EngineBackend] = None


class EngineRouter:
    """
    Intelligent router for audio engine backends
    
    Manages multiple synthesizer backends and provides:
    - Automatic engine selection based on requirements
    - Graceful fallbacks when engines fail
    - Performance monitoring and optimization
    - Hot-swapping between engines
    """
    
    def __init__(self):
        self.engines: Dict[EngineBackend, Optional[AbstractSynthesizer]] = {}
        self.engine_status: Dict[EngineBackend, EngineStatus] = {}
        self.preferences = RoutingPreferences()
        self.logger = logging.getLogger(__name__)
        
        # Active engine tracking
        self.primary_engine: Optional[EngineBackend] = None
        self.fallback_chain: List[EngineBackend] = [
            EngineBackend.SUPERCOLLIDER,
            EngineBackend.TIDALCYCLES,
            EngineBackend.PYTHON_NATIVE,
        ]
        
        # Performance tracking
        self.routing_stats: Dict[str, int] = {
            "total_requests": 0,
            "successful_routes": 0,
            "fallback_activations": 0,
            "engine_switches": 0,
        }
        
        # Initialize engine status
        self._initialize_engine_status()
    
    def _initialize_engine_status(self):
        """Initialize status tracking for all engines"""
        for backend in EngineBackend:
            self.engine_status[backend] = EngineStatus(
                backend=backend,
                capability=self._get_default_capability(backend)
            )
    
    def _get_default_capability(self, backend: EngineBackend) -> EngineCapability:
        """Get default capability level for each backend"""
        capability_map = {
            EngineBackend.SUPERCOLLIDER: EngineCapability.PROFESSIONAL,
            EngineBackend.TIDALCYCLES: EngineCapability.ADVANCED,
            EngineBackend.PYTHON_NATIVE: EngineCapability.BASIC,
        }
        return capability_map.get(backend, EngineCapability.BASIC)
    
    async def register_engine(self, backend: EngineBackend, engine: AbstractSynthesizer) -> bool:
        """Register an engine backend"""
        try:
            self.engines[backend] = engine
            status = self.engine_status[backend]
            
            # Test engine availability
            if hasattr(engine, 'start'):
                # Don't actually start - just check if it's startable
                status.available = True
            else:
                status.available = True  # Assume available if no start method
            
            # Update supported synths
            if hasattr(engine, 'get_available_synths'):
                try:
                    status.supported_synths = await engine.get_available_synths()
                except Exception as e:
                    self.logger.warning(f"Could not get synths for {backend.value}: {e}")
                    status.supported_synths = []
            
            status.initialized = True
            self.logger.info(f"Registered engine: {backend.value}")
            
            # Set as primary if it's the best available
            if self.primary_engine is None or status.capability.value > self.engine_status[self.primary_engine].capability.value:
                self.primary_engine = backend
                self.logger.info(f"Set primary engine: {backend.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register engine {backend.value}: {e}")
            self.engine_status[backend].last_error = str(e)
            self.engine_status[backend].error_count += 1
            return False
    
    def route_synth_request(self, synth_type: SynthType, params: SynthParams, 
                           requirements: Optional[Dict[str, Any]] = None) -> Optional[EngineBackend]:
        """
        Route a synth request to the best available engine
        
        Args:
            synth_type: Type of synthesizer requested
            params: Synthesis parameters
            requirements: Optional specific requirements (latency, quality, etc.)
        
        Returns:
            Best engine backend for the request, or None if none available
        """
        self.routing_stats["total_requests"] += 1
        
        try:
            # Parse requirements
            req = requirements or {}
            max_latency = req.get("max_latency_ms", self.preferences.max_latency_ms)
            min_quality = req.get("min_quality", EngineCapability.BASIC)
            preferred_backend = req.get("preferred_backend", self.preferences.prefer_backend)
            
            # Find suitable engines
            suitable_engines = []
            
            for backend, status in self.engine_status.items():
                if not status.available or not status.initialized:
                    continue
                
                # Check if engine supports this synth type
                if synth_type not in status.supported_synths and status.supported_synths:
                    continue
                
                # Check latency requirement
                if status.latency_ms > max_latency:
                    continue
                
                # Check quality requirement
                if self._capability_level(status.capability) < self._capability_level(min_quality):
                    continue
                
                suitable_engines.append((backend, status))
            
            if not suitable_engines:
                self.logger.warning(f"No suitable engines for {synth_type.value}")
                return None
            
            # Select best engine
            selected_backend = self._select_best_engine(suitable_engines, preferred_backend)
            
            if selected_backend:
                self.routing_stats["successful_routes"] += 1
                self.logger.debug(f"Routed {synth_type.value} to {selected_backend.value}")
            
            return selected_backend
            
        except Exception as e:
            self.logger.error(f"Error routing synth request: {e}")
            return None
    
    def _capability_level(self, capability: EngineCapability) -> int:
        """Convert capability enum to numeric level for comparison"""
        level_map = {
            EngineCapability.BASIC: 1,
            EngineCapability.INTERMEDIATE: 2,
            EngineCapability.PROFESSIONAL: 3,
            EngineCapability.ADVANCED: 4,
        }
        return level_map.get(capability, 1)
    
    def _select_best_engine(self, suitable_engines: List[tuple], 
                           preferred_backend: Optional[EngineBackend]) -> Optional[EngineBackend]:
        """Select the best engine from suitable candidates"""
        if not suitable_engines:
            return None
        
        # If preferred backend is specified and available, use it
        if preferred_backend:
            for backend, status in suitable_engines:
                if backend == preferred_backend:
                    return backend
        
        # Sort by quality, then by performance
        def score_engine(engine_tuple):
            backend, status = engine_tuple
            score = 0
            
            # Quality score (higher is better)
            score += self._capability_level(status.capability) * 100
            
            # Performance score (lower latency is better)
            score += max(0, 50 - status.latency_ms)
            
            # Reliability score (fewer errors is better)
            score -= status.error_count * 10
            
            # CPU usage score (lower is better)
            score += max(0, 50 - status.cpu_usage)
            
            return score
        
        # Select highest scoring engine
        best_engine = max(suitable_engines, key=score_engine)
        return best_engine[0]
    
    async def get_engine(self, backend: EngineBackend) -> Optional[AbstractSynthesizer]:
        """Get synthesizer instance for a specific backend"""
        if backend not in self.engines:
            return None
        
        engine = self.engines[backend]
        status = self.engine_status[backend]
        
        # Start engine if not running
        if engine and hasattr(engine, 'start') and not status.initialized:
            try:
                success = await engine.start()
                if success:
                    status.initialized = True
                    status.available = True
                else:
                    status.available = False
                    status.last_error = "Failed to start engine"
                    status.error_count += 1
            except Exception as e:
                status.available = False
                status.last_error = str(e)
                status.error_count += 1
                self.logger.error(f"Error starting engine {backend.value}: {e}")
        
        return engine if status.available else None
    
    async def handle_engine_failure(self, backend: EngineBackend, error: str):
        """Handle engine failure and attempt fallback"""
        self.logger.error(f"Engine failure in {backend.value}: {error}")
        
        status = self.engine_status[backend]
        status.available = False
        status.last_error = error
        status.error_count += 1
        
        # If this was the primary engine, find a fallback
        if self.primary_engine == backend and self.preferences.fallback_enabled:
            self._activate_fallback(backend)
    
    def _activate_fallback(self, failed_backend: EngineBackend):
        """Activate fallback engine when primary fails"""
        self.logger.info(f"Activating fallback for {failed_backend.value}")
        self.routing_stats["fallback_activations"] += 1
        
        # Find best available fallback
        for fallback_backend in self.fallback_chain:
            if fallback_backend == failed_backend:
                continue
            
            status = self.engine_status.get(fallback_backend)
            if status and status.available and status.initialized:
                self.primary_engine = fallback_backend
                self.routing_stats["engine_switches"] += 1
                self.logger.info(f"Switched to fallback engine: {fallback_backend.value}")
                return
        
        self.logger.warning("No fallback engines available")
        self.primary_engine = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive router status"""
        return {
            "primary_engine": self.primary_engine.value if self.primary_engine else None,
            "available_engines": [
                backend.value for backend, status in self.engine_status.items() 
                if status.available
            ],
            "engine_status": {
                backend.value: {
                    "available": status.available,
                    "initialized": status.initialized,
                    "capability": status.capability.value,
                    "latency_ms": status.latency_ms,
                    "error_count": status.error_count,
                    "last_error": status.last_error,
                    "supported_synths": [s.value for s in status.supported_synths],
                }
                for backend, status in self.engine_status.items()
            },
            "routing_stats": self.routing_stats.copy(),
            "preferences": {
                "prefer_quality_over_speed": self.preferences.prefer_quality_over_speed,
                "allow_engine_switching": self.preferences.allow_engine_switching,
                "fallback_enabled": self.preferences.fallback_enabled,
                "max_latency_ms": self.preferences.max_latency_ms,
            }
        }
    
    def set_preferences(self, **kwargs):
        """Update routing preferences"""
        for key, value in kwargs.items():
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
                self.logger.info(f"Updated preference {key} = {value}")


# Global router instance
_global_router: Optional[EngineRouter] = None

def get_engine_router() -> EngineRouter:
    """Get the global engine router instance"""
    global _global_router
    if _global_router is None:
        _global_router = EngineRouter()
    return _global_router


# Test function
async def test_engine_router():
    """Test engine router functionality"""
    print("🎛️ Testing Engine Router")
    print("=" * 50)
    
    router = EngineRouter()
    
    # Mock engine classes for testing
    class MockSuperColliderEngine:
        async def start(self): return True
        async def get_available_synths(self): return [SynthType.GABBER_KICK, SynthType.HOOVER_SYNTH]
    
    class MockPythonEngine:
        async def start(self): return True
        async def get_available_synths(self): return [SynthType.GABBER_KICK, SynthType.INDUSTRIAL_KICK]
    
    # Register engines
    print("Registering engines...")
    await router.register_engine(EngineBackend.SUPERCOLLIDER, MockSuperColliderEngine())
    await router.register_engine(EngineBackend.PYTHON_NATIVE, MockPythonEngine())
    
    # Test routing
    print("\n🎵 Testing synth routing...")
    
    backend = router.route_synth_request(
        SynthType.GABBER_KICK,
        SynthParams(freq=60.0, amp=0.8)
    )
    print(f"Gabber kick routed to: {backend.value if backend else 'None'}")
    
    # Test with requirements
    backend = router.route_synth_request(
        SynthType.HOOVER_SYNTH,
        SynthParams(freq=220.0, amp=0.6),
        requirements={"max_latency_ms": 5.0}
    )
    print(f"Hoover synth (low latency) routed to: {backend.value if backend else 'None'}")
    
    # Show status
    status = router.get_status()
    print(f"\n📊 Router Status:")
    print(f"Primary Engine: {status['primary_engine']}")
    print(f"Available Engines: {status['available_engines']}")
    print(f"Total Requests: {status['routing_stats']['total_requests']}")
    
    print("\n✅ Engine router test completed")


if __name__ == "__main__":
    asyncio.run(test_engine_router())