#!/usr/bin/env python3
"""
Web-based Control Interface for Hardcore Music Production
Modern React-inspired FastAPI backend with real-time WebSocket communication
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
import numpy as np

from ..interfaces.synthesizer import AbstractSynthesizer
from ..models.hardcore_models import HardcorePattern, SynthParams, SynthType
from ..ai.conversation_engine import ConversationEngine
from ..production.conversational_production_engine import ConversationalProductionEngine
from ..performance.live_performance_engine import LivePerformanceEngine, TransitionType
from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer
from ..evolution.pattern_evolution_engine import PatternEvolutionEngine
from ..hardware.midi_controller_integration import HardwareMIDIIntegration
from ..benchmarking.performance_benchmark_suite import ComprehensiveBenchmarkSuite, BenchmarkSeverity


logger = logging.getLogger(__name__)


# Pydantic models for API
class PatternRequest(BaseModel):
    """Request to create a pattern"""
    description: str = Field(..., description="Natural language description of the pattern")
    bpm: Optional[int] = Field(None, ge=60, le=300, description="BPM (60-300)")
    genre: Optional[str] = Field(None, description="Genre (gabber, industrial, hardcore, etc.)")
    artist_style: Optional[str] = Field(None, description="Artist style reference")
    intensity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Intensity level (0.0-1.0)")


class PatternModifyRequest(BaseModel):
    """Request to modify an existing pattern"""
    pattern_id: str = Field(..., description="ID of pattern to modify")
    modification: str = Field(..., description="Natural language modification request")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Specific parameters to modify")


class AnalysisRequest(BaseModel):
    """Request for audio analysis"""
    pattern_id: Optional[str] = Field(None, description="Pattern ID to analyze")
    analysis_type: str = Field("complete", description="Type of analysis (basic, kick_dna, psychoacoustic, complete)")


class EvolutionRequest(BaseModel):
    """Request for pattern evolution"""
    pattern_id: str = Field(..., description="Base pattern ID")
    population_size: int = Field(20, ge=5, le=100, description="Population size for evolution")
    generations: int = Field(5, ge=1, le=20, description="Number of generations")
    mutation_rate: float = Field(0.1, ge=0.0, le=1.0, description="Mutation rate")


class PerformanceSlotRequest(BaseModel):
    """Request to load pattern into performance slot"""
    slot_id: str = Field(..., description="Performance slot ID (slot_00 to slot_07)")
    pattern_id: str = Field(..., description="Pattern ID to load")


class PerformanceTriggerRequest(BaseModel):
    """Request to trigger performance slot"""
    slot_id: str = Field(..., description="Performance slot ID to trigger")
    transition_type: Optional[str] = Field("crossfade", description="Transition type")
    transition_duration: Optional[float] = Field(4.0, description="Transition duration in beats")


class BenchmarkRequest(BaseModel):
    """Request to run benchmark tests"""
    test_names: Optional[List[str]] = Field(None, description="Specific tests to run (null for all)")
    severity: str = Field("moderate", description="Severity level (light, moderate, heavy, extreme, torture)")


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    session_id: Optional[str] = Field(None, description="Session ID")
    timestamp: float = Field(default_factory=time.time, description="Message timestamp")


class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "connected_at": time.time(),
            "last_activity": time.time()
        }
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
                self.session_data[session_id]["last_activity"] = time.time()
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Send message to all connected sessions"""
        disconnected_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
                self.session_data[session_id]["last_activity"] = time.time()
            except Exception as e:
                logger.error(f"Failed to broadcast to {session_id}: {e}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.active_connections)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        if not self.session_data:
            return {"active_sessions": 0}
        
        current_time = time.time()
        session_durations = [current_time - data["connected_at"] for data in self.session_data.values()]
        
        return {
            "active_sessions": len(self.active_connections),
            "average_session_duration": sum(session_durations) / len(session_durations),
            "longest_session_duration": max(session_durations) if session_durations else 0
        }


class HardcoreWebApp:
    """Main web application for hardcore music production"""
    
    def __init__(self,
                 synthesizer: AbstractSynthesizer,
                 conversation_engine: ConversationEngine,
                 production_engine: ConversationalProductionEngine,
                 performance_engine: LivePerformanceEngine,
                 audio_analyzer: AdvancedAudioAnalyzer,
                 evolution_engine: PatternEvolutionEngine,
                 hardware_integration: HardwareMIDIIntegration,
                 benchmark_suite: ComprehensiveBenchmarkSuite):
        
        self.synthesizer = synthesizer
        self.conversation_engine = conversation_engine
        self.production_engine = production_engine
        self.performance_engine = performance_engine
        self.audio_analyzer = audio_analyzer
        self.evolution_engine = evolution_engine
        self.hardware_integration = hardware_integration
        self.benchmark_suite = benchmark_suite
        
        # WebSocket manager
        self.connection_manager = ConnectionManager()
        
        # Pattern storage
        self.patterns: Dict[str, HardcorePattern] = {}
        self.audio_cache: Dict[str, np.ndarray] = {}
        
        # Performance tracking
        self.stats = {
            "patterns_created": 0,
            "api_requests": 0,
            "websocket_messages": 0,
            "active_sessions": 0,
            "uptime_start": time.time()
        }
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("🔥 Starting Hardcore Web App")
            await self._startup()
            yield
            # Shutdown
            logger.info("🛑 Shutting down Hardcore Web App")
            await self._shutdown()
        
        app = FastAPI(
            title="GABBERBOT Web Interface",
            description="Hardcore music production control interface",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # CORS middleware for frontend integration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Static files and templates
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        templates_dir = Path(__file__).parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None
        
        self._setup_routes(app)
        
        return app
    
    def _setup_routes(self, app: FastAPI):
        """Setup all API routes"""
        
        # Health check
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "uptime": time.time() - self.stats["uptime_start"],
                "patterns_created": self.stats["patterns_created"],
                "active_sessions": self.connection_manager.get_session_count()
            }
        
        # Main web interface
        @app.get("/", response_class=HTMLResponse)
        async def web_interface(request: Request):
            if self.templates:
                return self.templates.TemplateResponse("index.html", {"request": request})
            else:
                return HTMLResponse(self._generate_simple_html())
        
        # WebSocket endpoint
        @app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await self.connection_manager.connect(websocket, session_id)
            
            try:
                # Send welcome message
                await self.connection_manager.send_personal_message({
                    "type": "welcome",
                    "data": {
                        "session_id": session_id,
                        "server_info": {
                            "name": "GABBERBOT Web Interface",
                            "version": "1.0.0",
                            "capabilities": [
                                "pattern_creation", "audio_analysis", "pattern_evolution",
                                "live_performance", "hardware_control", "benchmarking"
                            ]
                        }
                    }
                }, session_id)
                
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    self.stats["websocket_messages"] += 1
                    
                    # Process message
                    response = await self._process_websocket_message(message_data, session_id)
                    
                    # Send response
                    await self.connection_manager.send_personal_message(response, session_id)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(session_id)
            except Exception as e:
                logger.error(f"WebSocket error for {session_id}: {e}")
                self.connection_manager.disconnect(session_id)
        
        # REST API endpoints
        
        @app.post("/api/patterns/create")
        async def create_pattern(request: PatternRequest):
            """Create a new pattern from natural language description"""
            self.stats["api_requests"] += 1
            
            try:
                # Use production engine to process request
                response = await self.production_engine.process_request(
                    user_input=request.description,
                    session_id=f"api_{int(time.time())}"
                )
                
                if response.success and response.pattern:
                    pattern_id = str(uuid.uuid4())
                    self.patterns[pattern_id] = response.pattern
                    self.stats["patterns_created"] += 1
                    
                    # Generate audio if possible
                    try:
                        audio_data = await self.synthesizer.play_pattern(response.pattern)
                        if audio_data is not None:
                            self.audio_cache[pattern_id] = audio_data
                    except Exception as e:
                        logger.warning(f"Failed to generate audio for pattern: {e}")
                    
                    return {
                        "success": True,
                        "pattern_id": pattern_id,
                        "pattern": {
                            "name": response.pattern.name,
                            "bpm": response.pattern.bpm,
                            "genre": response.pattern.genre,
                            "synth_type": response.pattern.synth_type.value
                        },
                        "message": response.message,
                        "code": response.code_generated,
                        "confidence": response.confidence
                    }
                else:
                    raise HTTPException(status_code=400, detail=response.message)
                    
            except Exception as e:
                logger.error(f"Pattern creation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/patterns/{pattern_id}/modify")
        async def modify_pattern(pattern_id: str, request: PatternModifyRequest):
            """Modify an existing pattern"""
            self.stats["api_requests"] += 1
            
            if pattern_id not in self.patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            
            try:
                # Set current pattern in a temporary session
                temp_session_id = f"modify_{pattern_id}_{int(time.time())}"
                
                # Create session and set current pattern
                session = self.production_engine._get_or_create_session(temp_session_id, "api_user")
                session.current_pattern = self.patterns[pattern_id]
                
                # Process modification request
                response = await self.production_engine.process_request(
                    user_input=request.modification,
                    session_id=temp_session_id
                )
                
                if response.success and response.pattern:
                    # Update stored pattern
                    self.patterns[pattern_id] = response.pattern
                    
                    # Update audio cache
                    try:
                        audio_data = await self.synthesizer.play_pattern(response.pattern)
                        if audio_data is not None:
                            self.audio_cache[pattern_id] = audio_data
                    except Exception as e:
                        logger.warning(f"Failed to generate audio for modified pattern: {e}")
                    
                    return {
                        "success": True,
                        "pattern": {
                            "name": response.pattern.name,
                            "bpm": response.pattern.bpm,
                            "genre": response.pattern.genre,
                            "synth_type": response.pattern.synth_type.value
                        },
                        "message": response.message,
                        "code": response.code_generated
                    }
                else:
                    raise HTTPException(status_code=400, detail=response.message)
                    
            except Exception as e:
                logger.error(f"Pattern modification failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/patterns/{pattern_id}/analyze")
        async def analyze_pattern(pattern_id: str, request: AnalysisRequest):
            """Analyze a pattern's audio characteristics"""
            self.stats["api_requests"] += 1
            
            if pattern_id not in self.patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            
            try:
                pattern = self.patterns[pattern_id]
                
                # Get or generate audio
                if pattern_id in self.audio_cache:
                    audio_data = self.audio_cache[pattern_id]
                else:
                    audio_data = await self.synthesizer.play_pattern(pattern)
                    if audio_data is not None:
                        self.audio_cache[pattern_id] = audio_data
                
                if audio_data is None:
                    raise HTTPException(status_code=500, detail="Failed to generate audio for analysis")
                
                analysis_results = {}
                
                # Perform requested analysis
                if request.analysis_type in ["basic", "complete"]:
                    basic_analysis = await self.audio_analyzer.analyze_pattern_dna(audio_data)
                    analysis_results["basic"] = basic_analysis.to_dict()
                
                if request.analysis_type in ["kick_dna", "complete"]:
                    if "kick" in pattern.synth_type.value.lower():
                        kick_analysis = await self.audio_analyzer.analyze_kick_dna(audio_data)
                        analysis_results["kick_dna"] = kick_analysis.to_dict()
                
                if request.analysis_type in ["psychoacoustic", "complete"]:
                    psycho_analysis = await self.audio_analyzer.analyze_psychoacoustic_properties(audio_data)
                    analysis_results["psychoacoustic"] = psycho_analysis
                
                return {
                    "success": True,
                    "pattern_id": pattern_id,
                    "analysis_type": request.analysis_type,
                    "results": analysis_results
                }
                
            except Exception as e:
                logger.error(f"Pattern analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/patterns/{pattern_id}/evolve")
        async def evolve_pattern(pattern_id: str, request: EvolutionRequest):
            """Evolve a pattern using genetic algorithms"""
            self.stats["api_requests"] += 1
            
            if pattern_id not in self.patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            
            try:
                base_pattern = self.patterns[pattern_id]
                
                # Generate initial population
                population = await self.evolution_engine.generate_population(
                    population_size=request.population_size,
                    base_pattern=base_pattern
                )
                
                # Evolve for specified generations
                for generation in range(request.generations):
                    population = await self.evolution_engine.evolve_generation(population)
                
                # Store evolved patterns
                evolved_pattern_ids = []
                for i, evolved_pattern in enumerate(population[:5]):  # Store top 5
                    evolved_id = f"{pattern_id}_evolved_{i}"
                    evolved_pattern.name = f"{base_pattern.name}_evolved_{i}"
                    self.patterns[evolved_id] = evolved_pattern
                    evolved_pattern_ids.append(evolved_id)
                
                return {
                    "success": True,
                    "base_pattern_id": pattern_id,
                    "evolved_pattern_ids": evolved_pattern_ids,
                    "generations": request.generations,
                    "population_size": request.population_size,
                    "best_fitness": (await self.evolution_engine.evaluate_fitness(population[0])).overall if population else 0.0
                }
                
            except Exception as e:
                logger.error(f"Pattern evolution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/patterns")
        async def list_patterns():
            """List all stored patterns"""
            return {
                "patterns": {
                    pattern_id: {
                        "name": pattern.name,
                        "bpm": pattern.bpm,
                        "genre": pattern.genre,
                        "synth_type": pattern.synth_type.value,
                        "has_audio": pattern_id in self.audio_cache
                    }
                    for pattern_id, pattern in self.patterns.items()
                }
            }
        
        @app.get("/api/patterns/{pattern_id}")
        async def get_pattern(pattern_id: str):
            """Get specific pattern details"""
            if pattern_id not in self.patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            
            pattern = self.patterns[pattern_id]
            return {
                "pattern_id": pattern_id,
                "pattern": {
                    "name": pattern.name,
                    "bpm": pattern.bpm,
                    "genre": pattern.genre,
                    "synth_type": pattern.synth_type.value,
                    "pattern_data": pattern.pattern_data
                },
                "has_audio": pattern_id in self.audio_cache
            }
        
        @app.delete("/api/patterns/{pattern_id}")
        async def delete_pattern(pattern_id: str):
            """Delete a pattern"""
            if pattern_id not in self.patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            
            del self.patterns[pattern_id]
            if pattern_id in self.audio_cache:
                del self.audio_cache[pattern_id]
            
            return {"success": True, "message": f"Pattern {pattern_id} deleted"}
        
        # Performance engine endpoints
        
        @app.post("/api/performance/slots/{slot_id}/load")
        async def load_performance_slot(slot_id: str, request: PerformanceSlotRequest):
            """Load pattern into performance slot"""
            if request.pattern_id not in self.patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            
            try:
                pattern = self.patterns[request.pattern_id]
                await self.performance_engine.load_pattern_to_slot(slot_id, pattern)
                
                return {
                    "success": True,
                    "slot_id": slot_id,
                    "pattern_id": request.pattern_id,
                    "pattern_name": pattern.name
                }
                
            except Exception as e:
                logger.error(f"Failed to load pattern to slot: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/performance/slots/{slot_id}/trigger")
        async def trigger_performance_slot(slot_id: str, request: PerformanceTriggerRequest):
            """Trigger performance slot"""
            try:
                from ..performance.live_performance_engine import TransitionParams, TransitionType
                
                # Convert transition type string to enum
                transition_type = TransitionType.CROSSFADE
                if request.transition_type:
                    try:
                        transition_type = TransitionType(request.transition_type)
                    except ValueError:
                        pass  # Use default
                
                transition_params = TransitionParams(
                    transition_type=transition_type,
                    duration_beats=request.transition_duration
                )
                
                await self.performance_engine.trigger_slot(slot_id, transition_params)
                
                return {
                    "success": True,
                    "slot_id": slot_id,
                    "transition_type": request.transition_type,
                    "transition_duration": request.transition_duration
                }
                
            except Exception as e:
                logger.error(f"Failed to trigger slot: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/performance/status")
        async def get_performance_status():
            """Get current performance status"""
            stats = self.performance_engine.get_performance_stats()
            slot_status = self.performance_engine.get_slot_status()
            
            return {
                "performance_stats": stats,
                "slot_status": slot_status
            }
        
        # Hardware integration endpoints
        
        @app.get("/api/hardware/status")
        async def get_hardware_status():
            """Get hardware controller status"""
            return self.hardware_integration.get_hardware_status()
        
        @app.post("/api/hardware/scan")
        async def scan_hardware():
            """Scan for hardware controllers"""
            try:
                ports = await self.hardware_integration.scan_for_controllers()
                connection_results = await self.hardware_integration.auto_connect_controllers()
                
                return {
                    "success": True,
                    "available_ports": ports,
                    "connection_results": {k.value: v for k, v in connection_results.items()}
                }
                
            except Exception as e:
                logger.error(f"Hardware scan failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Benchmarking endpoints
        
        @app.post("/api/benchmark/run")
        async def run_benchmark(request: BenchmarkRequest):
            """Run benchmark tests"""
            try:
                # Convert severity string to enum
                severity = BenchmarkSeverity.MODERATE
                try:
                    severity = BenchmarkSeverity(request.severity)
                except ValueError:
                    pass  # Use default
                
                if request.test_names:
                    # Run specific tests
                    results = {}
                    for test_name in request.test_names:
                        try:
                            result = await self.benchmark_suite.run_single_benchmark(test_name, severity)
                            results[test_name] = {
                                "success": result.success,
                                "execution_time": result.execution_time,
                                "error_message": result.error_message,
                                "metrics": [
                                    {"name": m.name, "value": m.value, "unit": m.unit}
                                    for m in result.metrics[:10]  # Limit for API response
                                ]
                            }
                        except Exception as e:
                            results[test_name] = {
                                "success": False,
                                "error_message": str(e)
                            }
                else:
                    # Run full suite
                    full_results = await self.benchmark_suite.run_full_benchmark_suite(severity)
                    results = {}
                    
                    for test_name, result in full_results.items():
                        if test_name == "_suite_info":
                            results[test_name] = result
                        else:
                            results[test_name] = {
                                "success": result.success,
                                "execution_time": result.execution_time,
                                "error_message": result.error_message,
                                "metrics": [
                                    {"name": m.name, "value": m.value, "unit": m.unit}
                                    for m in result.metrics[:10]
                                ]
                            }
                
                return {
                    "success": True,
                    "severity": request.severity,
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/benchmark/summary")
        async def get_benchmark_summary():
            """Get benchmark history summary"""
            return self.benchmark_suite.get_benchmark_summary()
        
        # System status and statistics
        
        @app.get("/api/system/stats")
        async def get_system_stats():
            """Get comprehensive system statistics"""
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Application metrics
            app_stats = {
                **self.stats,
                "active_sessions": self.connection_manager.get_session_count(),
                "stored_patterns": len(self.patterns),
                "cached_audio": len(self.audio_cache),
                "uptime": time.time() - self.stats["uptime_start"]
            }
            
            # Performance engine stats
            perf_stats = self.performance_engine.get_performance_stats() if self.performance_engine else {}
            
            # Hardware stats
            hw_stats = self.hardware_integration.get_hardware_status()
            
            return {
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_mb": memory.used / 1024 / 1024,
                    "memory_total_mb": memory.total / 1024 / 1024,
                    "disk_used_percent": disk.percent,
                    "disk_free_gb": disk.free / 1024 / 1024 / 1024
                },
                "application": app_stats,
                "performance_engine": perf_stats,
                "hardware": hw_stats,
                "websocket_sessions": self.connection_manager.get_session_stats()
            }
    
    async def _process_websocket_message(self, message_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process incoming WebSocket message"""
        try:
            msg_type = message_data.get("type", "unknown")
            data = message_data.get("data", {})
            
            if msg_type == "chat":
                # Process chat message through production engine
                user_input = data.get("message", "")
                response = await self.production_engine.process_request(
                    user_input=user_input,
                    session_id=session_id
                )
                
                result = {
                    "type": "chat_response",
                    "data": {
                        "success": response.success,
                        "message": response.message,
                        "confidence": response.confidence,
                        "execution_time_ms": response.execution_time_ms
                    }
                }
                
                if response.pattern:
                    pattern_id = str(uuid.uuid4())
                    self.patterns[pattern_id] = response.pattern
                    result["data"]["pattern_id"] = pattern_id
                    result["data"]["pattern"] = {
                        "name": response.pattern.name,
                        "bpm": response.pattern.bpm,
                        "genre": response.pattern.genre
                    }
                
                return result
            
            elif msg_type == "get_patterns":
                # Return list of patterns
                return {
                    "type": "patterns_list",
                    "data": {
                        "patterns": {
                            pattern_id: {
                                "name": pattern.name,
                                "bpm": pattern.bpm,
                                "genre": pattern.genre
                            }
                            for pattern_id, pattern in self.patterns.items()
                        }
                    }
                }
            
            elif msg_type == "get_performance_status":
                # Return performance status
                stats = self.performance_engine.get_performance_stats()
                return {
                    "type": "performance_status",
                    "data": stats
                }
            
            elif msg_type == "ping":
                return {
                    "type": "pong",
                    "data": {"timestamp": time.time()}
                }
            
            else:
                return {
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"}
                }
                
        except Exception as e:
            logger.error(f"WebSocket message processing failed: {e}")
            return {
                "type": "error",
                "data": {"message": str(e)}
            }
    
    def _generate_simple_html(self) -> str:
        """Generate simple HTML interface if templates not available"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>🔥 GABBERBOT Web Interface 🔥</title>
    <style>
        body { font-family: 'Courier New', monospace; background: #000; color: #00ff00; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; border: 2px solid #ff0000; padding: 20px; margin-bottom: 30px; }
        .section { border: 1px solid #333; padding: 20px; margin: 20px 0; }
        .button { background: #ff0000; color: #fff; padding: 10px 20px; border: none; cursor: pointer; font-weight: bold; }
        .button:hover { background: #ff3333; }
        .status { color: #ffff00; }
        .error { color: #ff0066; }
        .success { color: #00ff66; }
        .code { background: #001100; padding: 10px; font-family: monospace; border-left: 3px solid #00ff00; }
        input, textarea { background: #111; color: #00ff00; border: 1px solid #333; padding: 10px; width: 100%; }
        #chat { height: 400px; overflow-y: scroll; background: #001100; padding: 15px; border: 1px solid #333; }
        .message { margin: 10px 0; padding: 5px; }
        .user { color: #00ff00; }
        .bot { color: #ff6600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 GABBERBOT WEB INTERFACE 🔥</h1>
            <p>Hardcore Music Production Control Center</p>
        </div>
        
        <div class="section">
            <h2>🤖 AI Chat Interface</h2>
            <div id="chat"></div>
            <input type="text" id="chatInput" placeholder="Ask GABBERBOT to create hardcore patterns..." onkeypress="if(event.key==='Enter') sendChat()">
            <button class="button" onclick="sendChat()">Send</button>
        </div>
        
        <div class="section">
            <h2>📊 System Status</h2>
            <div id="status">Connecting to system...</div>
            <button class="button" onclick="updateStatus()">Refresh Status</button>
        </div>
        
        <div class="section">
            <h2>🎵 Pattern Library</h2>
            <div id="patterns">No patterns loaded</div>
            <button class="button" onclick="loadPatterns()">Load Patterns</button>
        </div>
        
        <div class="section">
            <h2>🎛️ Performance Engine</h2>
            <div id="performance">Performance engine status loading...</div>
            <button class="button" onclick="updatePerformanceStatus()">Update Performance Status</button>
        </div>
        
        <div class="section">
            <h2>⚡ Quick Actions</h2>
            <button class="button" onclick="createGabberKick()">Create Gabber Kick</button>
            <button class="button" onclick="createIndustrialLoop()">Create Industrial Loop</button>
            <button class="button" onclick="scanHardware()">Scan Hardware</button>
            <button class="button" onclick="runBenchmark()">Run Benchmark</button>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin + '/api';
        let ws = null;
        let sessionId = 'web_' + Math.random().toString(36).substr(2, 9);
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws/' + sessionId;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
                addChatMessage('System', 'Connected to GABBERBOT 🔥', 'success');
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                addChatMessage('System', 'Disconnected from GABBERBOT ❌', 'error');
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                addChatMessage('System', 'Connection error ⚠️', 'error');
            };
        }
        
        function handleWebSocketMessage(message) {
            if (message.type === 'chat_response') {
                const data = message.data;
                addChatMessage('GABBERBOT', data.message, data.success ? 'success' : 'error');
                
                if (data.pattern) {
                    addChatMessage('System', `Created pattern: ${data.pattern.name} (${data.pattern.bpm} BPM)`, 'status');
                }
            } else if (message.type === 'welcome') {
                addChatMessage('GABBERBOT', 'Welcome to the hardcore production interface! 🎵🔥', 'bot');
            }
        }
        
        function sendChat() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                addChatMessage('You', message, 'user');
                
                ws.send(JSON.stringify({
                    type: 'chat',
                    data: { message: message },
                    session_id: sessionId
                }));
                
                input.value = '';
            }
        }
        
        function addChatMessage(sender, text, className) {
            const chat = document.getElementById('chat');
            const message = document.createElement('div');
            message.className = 'message ' + className;
            message.innerHTML = `<strong>${sender}:</strong> ${text}`;
            chat.appendChild(message);
            chat.scrollTop = chat.scrollHeight;
        }
        
        async function updateStatus() {
            try {
                const response = await fetch(API_BASE + '/system/stats');
                const data = await response.json();
                
                document.getElementById('status').innerHTML = `
                    <div class="success">System Status: Online ✅</div>
                    <div>CPU: ${data.system.cpu_percent.toFixed(1)}%</div>
                    <div>Memory: ${data.system.memory_percent.toFixed(1)}%</div>
                    <div>Patterns: ${data.application.stored_patterns}</div>
                    <div>Active Sessions: ${data.application.active_sessions}</div>
                    <div>Uptime: ${(data.application.uptime / 60).toFixed(1)} minutes</div>
                `;
            } catch (error) {
                document.getElementById('status').innerHTML = '<div class="error">Failed to load status ❌</div>';
            }
        }
        
        async function loadPatterns() {
            try {
                const response = await fetch(API_BASE + '/patterns');
                const data = await response.json();
                
                if (Object.keys(data.patterns).length === 0) {
                    document.getElementById('patterns').innerHTML = '<div class="status">No patterns stored</div>';
                } else {
                    let html = '<div class="success">Stored Patterns:</div>';
                    for (const [id, pattern] of Object.entries(data.patterns)) {
                        html += `<div>• ${pattern.name} - ${pattern.bpm} BPM (${pattern.genre})</div>`;
                    }
                    document.getElementById('patterns').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('patterns').innerHTML = '<div class="error">Failed to load patterns ❌</div>';
            }
        }
        
        async function updatePerformanceStatus() {
            try {
                const response = await fetch(API_BASE + '/performance/status');
                const data = await response.json();
                
                const stats = data.performance_stats;
                document.getElementById('performance').innerHTML = `
                    <div>State: <span class="${stats.state === 'playing' ? 'success' : 'status'}">${stats.state}</span></div>
                    <div>BPM: ${stats.current_bpm}</div>
                    <div>Patterns Played: ${stats.patterns_played}</div>
                    <div>Audience Energy: ${(stats.audience_energy * 100).toFixed(1)}%</div>
                    <div>Active Slots: ${stats.active_slots.length}</div>
                `;
            } catch (error) {
                document.getElementById('performance').innerHTML = '<div class="error">Performance engine not available ❌</div>';
            }
        }
        
        async function createGabberKick() {
            try {
                const response = await fetch(API_BASE + '/patterns/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        description: 'Create a brutal gabber kick at 180 BPM with maximum crunch',
                        bpm: 180,
                        genre: 'gabber',
                        intensity: 0.9
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    addChatMessage('System', `Created: ${data.pattern.name} 🔥`, 'success');
                    loadPatterns();
                } else {
                    addChatMessage('System', 'Failed to create pattern ❌', 'error');
                }
            } catch (error) {
                addChatMessage('System', 'Request failed ❌', 'error');
            }
        }
        
        async function createIndustrialLoop() {
            try {
                const response = await fetch(API_BASE + '/patterns/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        description: 'Create a dark industrial loop with heavy reverb and metallic percussion',
                        bpm: 140,
                        genre: 'industrial',
                        intensity: 0.7
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    addChatMessage('System', `Created: ${data.pattern.name} ⚙️`, 'success');
                    loadPatterns();
                } else {
                    addChatMessage('System', 'Failed to create pattern ❌', 'error');
                }
            } catch (error) {
                addChatMessage('System', 'Request failed ❌', 'error');
            }
        }
        
        async function scanHardware() {
            try {
                const response = await fetch(API_BASE + '/hardware/scan', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    const connected = Object.values(data.connection_results).filter(Boolean).length;
                    addChatMessage('System', `Hardware scan complete: ${connected} controllers connected 🎛️`, 'success');
                } else {
                    addChatMessage('System', 'Hardware scan failed ❌', 'error');
                }
            } catch (error) {
                addChatMessage('System', 'Hardware scan request failed ❌', 'error');
            }
        }
        
        async function runBenchmark() {
            addChatMessage('System', 'Running performance benchmark... ⚡', 'status');
            
            try {
                const response = await fetch(API_BASE + '/benchmark/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        severity: 'light'
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    const results = Object.entries(data.results).filter(([k, v]) => k !== '_suite_info');
                    const passed = results.filter(([k, v]) => v.success).length;
                    addChatMessage('System', `Benchmark complete: ${passed}/${results.length} tests passed 📊`, 'success');
                } else {
                    addChatMessage('System', 'Benchmark failed ❌', 'error');
                }
            } catch (error) {
                addChatMessage('System', 'Benchmark request failed ❌', 'error');
            }
        }
        
        // Initialize the interface
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            updateStatus();
            loadPatterns();
            updatePerformanceStatus();
            
            // Auto-refresh status every 30 seconds
            setInterval(updateStatus, 30000);
        });
    </script>
</body>
</html>
        """
    
    async def _startup(self):
        """Application startup tasks"""
        try:
            # Start performance engine
            await self.performance_engine.start_performance()
            
            # Start hardware monitoring
            await self.hardware_integration.start_monitoring()
            
            # Connect to hardware controllers
            await self.hardware_integration.auto_connect_controllers()
            
            logger.info("Hardcore Web App startup completed successfully")
            
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    async def _shutdown(self):
        """Application shutdown tasks"""
        try:
            # Stop performance engine
            await self.performance_engine.stop_performance()
            
            # Stop hardware monitoring
            await self.hardware_integration.stop_monitoring()
            
            logger.info("Hardcore Web App shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def run(self, host: str = "localhost", port: int = 8000, **kwargs):
        """Run the web application"""
        logger.info(f"🔥 Starting GABBERBOT Web Interface on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )


# Factory function
def create_hardcore_web_app(
    synthesizer: AbstractSynthesizer,
    conversation_engine: ConversationEngine,
    production_engine: ConversationalProductionEngine,
    performance_engine: LivePerformanceEngine,
    audio_analyzer: AdvancedAudioAnalyzer,
    evolution_engine: PatternEvolutionEngine,
    hardware_integration: HardwareMIDIIntegration,
    benchmark_suite: ComprehensiveBenchmarkSuite
) -> HardcoreWebApp:
    """Create hardcore web app with all dependencies"""
    
    return HardcoreWebApp(
        synthesizer=synthesizer,
        conversation_engine=conversation_engine,
        production_engine=production_engine,
        performance_engine=performance_engine,
        audio_analyzer=audio_analyzer,
        evolution_engine=evolution_engine,
        hardware_integration=hardware_integration,
        benchmark_suite=benchmark_suite
    )


if __name__ == "__main__":
    # Demo the web application
    import asyncio
    from ..interfaces.synthesizer import MockSynthesizer
    from ..ai.conversation_engine import create_conversation_engine
    from ..production.conversational_production_engine import create_conversational_production_engine
    from ..performance.live_performance_engine import create_live_performance_engine
    from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer
    from ..evolution.pattern_evolution_engine import PatternEvolutionEngine
    from ..hardware.midi_controller_integration import create_hardware_integration
    from ..benchmarking.performance_benchmark_suite import create_comprehensive_benchmark_suite
    
    def demo_web_app():
        print("🔥 HARDCORE WEB APP DEMO 🔥")
        print("=" * 40)
        
        # Create all dependencies
        synth = MockSynthesizer()
        conv_engine = create_conversation_engine(synth)
        prod_engine = create_conversational_production_engine(synth)
        perf_engine = create_live_performance_engine(synth)
        audio_analyzer = AdvancedAudioAnalyzer()
        evolution_engine = PatternEvolutionEngine()
        hardware_integration = create_hardware_integration(synth, perf_engine)
        benchmark_suite = create_comprehensive_benchmark_suite(
            synth, conv_engine, prod_engine, audio_analyzer, evolution_engine, perf_engine
        )
        
        # Create web app
        web_app = create_hardcore_web_app(
            synthesizer=synth,
            conversation_engine=conv_engine,
            production_engine=prod_engine,
            performance_engine=perf_engine,
            audio_analyzer=audio_analyzer,
            evolution_engine=evolution_engine,
            hardware_integration=hardware_integration,
            benchmark_suite=benchmark_suite
        )
        
        print("🌐 Starting web server on http://localhost:8000")
        print("Features available:")
        print("• 🤖 AI Chat Interface")
        print("• 🎵 Pattern Creation & Management")
        print("• 🔬 Audio Analysis")
        print("• 🧬 Pattern Evolution")
        print("• 🎛️ Live Performance Control")
        print("• 🎧 Hardware MIDI Integration")
        print("• ⚡ Performance Benchmarking")
        print("• 📊 Real-time System Monitoring")
        print("• 🔌 WebSocket Real-time Communication")
        
        # Run the web application
        web_app.run(host="localhost", port=8000)
    
    demo_web_app()