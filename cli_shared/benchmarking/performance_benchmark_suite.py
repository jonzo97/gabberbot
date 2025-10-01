#!/usr/bin/env python3
"""
Comprehensive Backend Benchmarking Suite for Hardcore Music Production
Performance analysis, stress testing, and optimization profiling
"""

import asyncio
import gc
import logging
import multiprocessing
import os
import psutil
import statistics
import sys
import threading
import time
import tracemalloc
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import json
import numpy as np
from collections import defaultdict, deque
import resource

from ..interfaces.synthesizer import AbstractSynthesizer
from ..models.hardcore_models import HardcorePattern, SynthType
from ..ai.conversation_engine import ConversationEngine
from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer
from ..evolution.pattern_evolution_engine import PatternEvolutionEngine
from ..performance.live_performance_engine import LivePerformanceEngine
from ..production.conversational_production_engine import ConversationalProductionEngine


logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    LATENCY = "latency"                    # Response time measurements
    THROUGHPUT = "throughput"              # Operations per second
    MEMORY = "memory"                      # Memory usage and leaks
    CPU = "cpu"                           # CPU utilization
    CONCURRENCY = "concurrency"           # Multi-threading/processing
    STRESS = "stress"                     # High load conditions
    ENDURANCE = "endurance"               # Long-running stability
    REGRESSION = "regression"             # Performance over time
    SCALABILITY = "scalability"           # Performance vs load
    AUDIO_QUALITY = "audio_quality"       # Audio generation quality


class BenchmarkSeverity(Enum):
    LIGHT = "light"                       # Basic functionality tests
    MODERATE = "moderate"                 # Standard load tests
    HEAVY = "heavy"                      # High stress tests
    EXTREME = "extreme"                  # Maximum capability tests
    TORTURE = "torture"                  # Destructive stress tests


@dataclass
class BenchmarkMetric:
    """Single benchmark measurement"""
    name: str
    value: float
    unit: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    test_name: str
    benchmark_type: BenchmarkType
    severity: BenchmarkSeverity
    success: bool
    execution_time: float
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    error_message: Optional[str] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, name: str, value: float, unit: str, **context):
        """Add a metric to this benchmark result"""
        metric = BenchmarkMetric(
            name=name,
            value=value,
            unit=unit,
            context=context
        )
        self.metrics.append(metric)
    
    def get_metric(self, name: str) -> Optional[BenchmarkMetric]:
        """Get a specific metric by name"""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None


@dataclass
class SystemSnapshot:
    """System performance snapshot"""
    timestamp: float
    cpu_percent: float
    memory_used: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    thread_count: int
    process_count: int
    load_average: Tuple[float, float, float]
    
    @classmethod
    def capture(cls) -> 'SystemSnapshot':
        """Capture current system state"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            return cls(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_used=memory.used,
                memory_percent=memory.percent,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_sent=net_io.bytes_sent if net_io else 0,
                network_recv=net_io.bytes_recv if net_io else 0,
                thread_count=threading.active_count(),
                process_count=len(psutil.pids()),
                load_average=os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
            )
        except Exception as e:
            logger.error(f"Failed to capture system snapshot: {e}")
            return cls(
                timestamp=time.time(),
                cpu_percent=0.0, memory_used=0, memory_percent=0.0,
                disk_io_read=0, disk_io_write=0,
                network_sent=0, network_recv=0,
                thread_count=1, process_count=1,
                load_average=(0.0, 0.0, 0.0)
            )


class AbstractBenchmark(ABC):
    """Abstract base class for benchmark tests"""
    
    def __init__(self, name: str, benchmark_type: BenchmarkType, severity: BenchmarkSeverity):
        self.name = name
        self.benchmark_type = benchmark_type
        self.severity = severity
        self.system_snapshots: List[SystemSnapshot] = []
        
    @abstractmethod
    async def run_benchmark(self, **kwargs) -> BenchmarkResult:
        """Run the benchmark test"""
        pass
    
    def start_monitoring(self, interval: float = 0.5):
        """Start system monitoring"""
        def monitor():
            while getattr(self, '_monitoring', False):
                snapshot = SystemSnapshot.capture()
                self.system_snapshots.append(snapshot)
                time.sleep(interval)
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get aggregated system statistics"""
        if not self.system_snapshots:
            return {}
        
        cpu_values = [s.cpu_percent for s in self.system_snapshots]
        memory_values = [s.memory_percent for s in self.system_snapshots]
        
        return {
            "cpu_avg": statistics.mean(cpu_values),
            "cpu_max": max(cpu_values),
            "cpu_min": min(cpu_values),
            "memory_avg": statistics.mean(memory_values),
            "memory_max": max(memory_values),
            "memory_peak_mb": max(s.memory_used for s in self.system_snapshots) / 1024 / 1024,
            "sample_count": len(self.system_snapshots)
        }


class SynthesizerLatencyBenchmark(AbstractBenchmark):
    """Benchmark synthesizer latency performance"""
    
    def __init__(self, synthesizer: AbstractSynthesizer, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE):
        super().__init__(f"Synthesizer Latency ({severity.value})", BenchmarkType.LATENCY, severity)
        self.synthesizer = synthesizer
        
    async def run_benchmark(self, **kwargs) -> BenchmarkResult:
        start_time = time.time()
        result = BenchmarkResult(
            test_name=self.name,
            benchmark_type=self.benchmark_type,
            severity=self.severity,
            success=False,
            execution_time=0.0
        )
        
        try:
            self.start_monitoring()
            
            # Test parameters based on severity
            test_params = {
                BenchmarkSeverity.LIGHT: {"iterations": 10, "patterns": 1},
                BenchmarkSeverity.MODERATE: {"iterations": 50, "patterns": 5},
                BenchmarkSeverity.HEAVY: {"iterations": 100, "patterns": 10},
                BenchmarkSeverity.EXTREME: {"iterations": 200, "patterns": 20},
                BenchmarkSeverity.TORTURE: {"iterations": 500, "patterns": 50}
            }
            
            params = test_params[self.severity]
            iterations = params["iterations"]
            pattern_count = params["patterns"]
            
            # Create test patterns
            test_patterns = []
            for i in range(pattern_count):
                pattern = HardcorePattern(
                    name=f"benchmark_pattern_{i}",
                    bpm=150 + (i * 10),
                    pattern_data=f's("bd:{i%10}").struct("x ~ x ~").shape(0.{7+i%3})',
                    synth_type=SynthType.GABBER_KICK,
                    genre="gabber"
                )
                test_patterns.append(pattern)
            
            # Measure latencies
            latencies = []
            synthesis_latencies = []
            
            for i in range(iterations):
                pattern = test_patterns[i % len(test_patterns)]
                
                # Measure total latency
                start = time.perf_counter()
                try:
                    audio_data = await self.synthesizer.play_pattern(pattern)
                    end = time.perf_counter()
                    latency = (end - start) * 1000  # Convert to milliseconds
                    latencies.append(latency)
                except Exception as e:
                    logger.warning(f"Synthesis failed on iteration {i}: {e}")
                    continue
                
                # Measure synthesis-only latency
                synth_start = time.perf_counter()
                try:
                    await self.synthesizer.play_synth(SynthType.GABBER_KICK)
                    synth_end = time.perf_counter()
                    synth_latency = (synth_end - synth_start) * 1000
                    synthesis_latencies.append(synth_latency)
                except Exception as e:
                    logger.warning(f"Synth test failed on iteration {i}: {e}")
                
                # Brief pause to avoid overwhelming the system
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
            
            if latencies:
                result.add_metric("avg_latency", statistics.mean(latencies), "ms")
                result.add_metric("min_latency", min(latencies), "ms")
                result.add_metric("max_latency", max(latencies), "ms")
                result.add_metric("p95_latency", statistics.quantiles(latencies, n=20)[18], "ms")
                result.add_metric("p99_latency", statistics.quantiles(latencies, n=100)[98], "ms")
                result.add_metric("latency_stddev", statistics.stdev(latencies) if len(latencies) > 1 else 0, "ms")
            
            if synthesis_latencies:
                result.add_metric("avg_synth_latency", statistics.mean(synthesis_latencies), "ms")
                result.add_metric("min_synth_latency", min(synthesis_latencies), "ms")
                result.add_metric("max_synth_latency", max(synthesis_latencies), "ms")
            
            result.add_metric("total_iterations", len(latencies), "count")
            result.add_metric("success_rate", (len(latencies) / iterations) * 100, "percent")
            
            result.success = len(latencies) > iterations * 0.8  # 80% success threshold
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Synthesizer latency benchmark failed: {e}")
        finally:
            self.stop_monitoring()
            result.execution_time = time.time() - start_time
            result.system_info = self.get_system_stats()
        
        return result


class AIConversationThroughputBenchmark(AbstractBenchmark):
    """Benchmark AI conversation throughput"""
    
    def __init__(self, conversation_engine: ConversationEngine, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE):
        super().__init__(f"AI Conversation Throughput ({severity.value})", BenchmarkType.THROUGHPUT, severity)
        self.conversation_engine = conversation_engine
        
    async def run_benchmark(self, **kwargs) -> BenchmarkResult:
        start_time = time.time()
        result = BenchmarkResult(
            test_name=self.name,
            benchmark_type=self.benchmark_type,
            severity=self.severity,
            success=False,
            execution_time=0.0
        )
        
        try:
            self.start_monitoring()
            
            # Test parameters
            test_params = {
                BenchmarkSeverity.LIGHT: {"concurrent_sessions": 1, "requests_per_session": 5},
                BenchmarkSeverity.MODERATE: {"concurrent_sessions": 3, "requests_per_session": 10},
                BenchmarkSeverity.HEAVY: {"concurrent_sessions": 5, "requests_per_session": 20},
                BenchmarkSeverity.EXTREME: {"concurrent_sessions": 10, "requests_per_session": 30},
                BenchmarkSeverity.TORTURE: {"concurrent_sessions": 20, "requests_per_session": 50}
            }
            
            params = test_params[self.severity]
            concurrent_sessions = params["concurrent_sessions"]
            requests_per_session = params["requests_per_session"]
            
            # Test queries for conversation
            test_queries = [
                "make a gabber kick at 180 bpm",
                "make it harder and more distorted",
                "analyze the frequency spectrum",
                "set bpm to 200",
                "save as test_pattern",
                "create industrial loop",
                "apply reverb effect",
                "what are some variations?",
                "evolve this pattern",
                "export as wav file"
            ]
            
            # Track performance metrics
            response_times = []
            successful_requests = 0
            failed_requests = 0
            
            async def run_session(session_id: str, request_count: int):
                session_responses = []
                session_failures = 0
                
                for i in range(request_count):
                    query = test_queries[i % len(test_queries)]
                    
                    try:
                        start = time.perf_counter()
                        response = await self.conversation_engine.chat(session_id, query)
                        end = time.perf_counter()
                        
                        response_time = (end - start) * 1000
                        session_responses.append(response_time)
                        
                        if response.confidence > 0.3:  # Minimum confidence threshold
                            nonlocal successful_requests
                            successful_requests += 1
                        else:
                            session_failures += 1
                    
                    except Exception as e:
                        logger.warning(f"Request failed in session {session_id}: {e}")
                        session_failures += 1
                        nonlocal failed_requests
                        failed_requests += 1
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                
                return session_responses, session_failures
            
            # Run concurrent sessions
            session_tasks = []
            for i in range(concurrent_sessions):
                session_id = f"benchmark_session_{i}"
                task = run_session(session_id, requests_per_session)
                session_tasks.append(task)
            
            # Wait for all sessions to complete
            session_results = await asyncio.gather(*session_tasks, return_exceptions=True)
            
            # Aggregate results
            all_response_times = []
            total_failures = 0
            
            for session_result in session_results:
                if isinstance(session_result, Exception):
                    logger.error(f"Session failed: {session_result}")
                    total_failures += requests_per_session
                else:
                    responses, failures = session_result
                    all_response_times.extend(responses)
                    total_failures += failures
            
            response_times = all_response_times
            total_requests = concurrent_sessions * requests_per_session
            
            if response_times:
                result.add_metric("avg_response_time", statistics.mean(response_times), "ms")
                result.add_metric("min_response_time", min(response_times), "ms")
                result.add_metric("max_response_time", max(response_times), "ms")
                result.add_metric("p95_response_time", statistics.quantiles(response_times, n=20)[18], "ms")
                result.add_metric("p99_response_time", statistics.quantiles(response_times, n=100)[98], "ms")
                
                # Calculate throughput
                total_time = max(response_times) / 1000 if response_times else 1
                throughput = len(response_times) / total_time
                result.add_metric("throughput", throughput, "requests/sec")
            
            result.add_metric("total_requests", total_requests, "count")
            result.add_metric("successful_requests", successful_requests, "count")
            result.add_metric("failed_requests", failed_requests, "count")
            result.add_metric("success_rate", (successful_requests / total_requests) * 100, "percent")
            result.add_metric("concurrent_sessions", concurrent_sessions, "count")
            
            result.success = successful_requests > total_requests * 0.7  # 70% success threshold
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"AI conversation throughput benchmark failed: {e}")
        finally:
            self.stop_monitoring()
            result.execution_time = time.time() - start_time
            result.system_info = self.get_system_stats()
        
        return result


class MemoryLeakBenchmark(AbstractBenchmark):
    """Benchmark memory usage and detect leaks"""
    
    def __init__(self, production_engine: ConversationalProductionEngine, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE):
        super().__init__(f"Memory Leak Detection ({severity.value})", BenchmarkType.MEMORY, severity)
        self.production_engine = production_engine
        
    async def run_benchmark(self, **kwargs) -> BenchmarkResult:
        start_time = time.time()
        result = BenchmarkResult(
            test_name=self.name,
            benchmark_type=self.benchmark_type,
            severity=self.severity,
            success=False,
            execution_time=0.0
        )
        
        try:
            # Start memory tracing
            tracemalloc.start()
            initial_memory = psutil.Process().memory_info().rss
            
            self.start_monitoring()
            
            # Test parameters
            test_params = {
                BenchmarkSeverity.LIGHT: {"cycles": 5, "operations_per_cycle": 10},
                BenchmarkSeverity.MODERATE: {"cycles": 10, "operations_per_cycle": 20},
                BenchmarkSeverity.HEAVY: {"cycles": 20, "operations_per_cycle": 50},
                BenchmarkSeverity.EXTREME: {"cycles": 50, "operations_per_cycle": 100},
                BenchmarkSeverity.TORTURE: {"cycles": 100, "operations_per_cycle": 200}
            }
            
            params = test_params[self.severity]
            cycles = params["cycles"]
            operations_per_cycle = params["operations_per_cycle"]
            
            memory_samples = []
            session_id = "memory_test_session"
            
            # Test operations that might leak memory
            test_operations = [
                "create a gabber pattern at 175 bpm",
                "make it more aggressive",
                "analyze the audio",
                "evolve this pattern",
                "save as memory_test",
                "load pattern memory_test",
                "export audio",
                "suggest variations"
            ]
            
            for cycle in range(cycles):
                cycle_start_memory = psutil.Process().memory_info().rss
                
                # Perform operations
                for op in range(operations_per_cycle):
                    operation = test_operations[op % len(test_operations)]
                    
                    try:
                        await self.production_engine.process_request(
                            user_input=operation,
                            session_id=f"{session_id}_{cycle}_{op}"
                        )
                    except Exception as e:
                        logger.warning(f"Operation failed: {e}")
                    
                    # Sample memory every 10 operations
                    if op % 10 == 0:
                        current_memory = psutil.Process().memory_info().rss
                        memory_samples.append({
                            "cycle": cycle,
                            "operation": op,
                            "memory_mb": current_memory / 1024 / 1024,
                            "timestamp": time.time()
                        })
                
                cycle_end_memory = psutil.Process().memory_info().rss
                
                # Force garbage collection
                gc.collect()
                gc_memory = psutil.Process().memory_info().rss
                
                result.add_metric(f"cycle_{cycle}_memory_growth", 
                                (cycle_end_memory - cycle_start_memory) / 1024 / 1024, "MB",
                                cycle=cycle)
                result.add_metric(f"cycle_{cycle}_gc_freed", 
                                (cycle_end_memory - gc_memory) / 1024 / 1024, "MB",
                                cycle=cycle)
                
                # Brief pause between cycles
                await asyncio.sleep(0.5)
            
            final_memory = psutil.Process().memory_info().rss
            
            # Get memory trace statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Analyze memory growth
            if memory_samples:
                initial_sample = memory_samples[0]["memory_mb"]
                final_sample = memory_samples[-1]["memory_mb"]
                total_growth = final_sample - initial_sample
                
                # Calculate growth rate
                time_span = memory_samples[-1]["timestamp"] - memory_samples[0]["timestamp"]
                growth_rate = total_growth / time_span if time_span > 0 else 0
                
                result.add_metric("total_memory_growth", total_growth, "MB")
                result.add_metric("memory_growth_rate", growth_rate, "MB/sec")
                result.add_metric("peak_memory_traced", peak / 1024 / 1024, "MB")
                result.add_metric("final_memory_traced", current / 1024 / 1024, "MB")
            
            result.add_metric("initial_memory", initial_memory / 1024 / 1024, "MB")
            result.add_metric("final_memory", final_memory / 1024 / 1024, "MB")
            result.add_metric("net_memory_change", (final_memory - initial_memory) / 1024 / 1024, "MB")
            result.add_metric("total_cycles", cycles, "count")
            result.add_metric("operations_per_cycle", operations_per_cycle, "count")
            
            # Memory leak detection logic
            memory_growth_threshold = {
                BenchmarkSeverity.LIGHT: 50,      # 50MB
                BenchmarkSeverity.MODERATE: 100,  # 100MB
                BenchmarkSeverity.HEAVY: 200,     # 200MB
                BenchmarkSeverity.EXTREME: 500,   # 500MB
                BenchmarkSeverity.TORTURE: 1000   # 1GB
            }
            
            threshold = memory_growth_threshold[self.severity]
            memory_growth = (final_memory - initial_memory) / 1024 / 1024
            
            result.success = memory_growth < threshold
            
            if not result.success:
                result.add_metric("memory_leak_detected", 1, "bool")
                result.error_message = f"Potential memory leak: {memory_growth:.1f}MB growth exceeds {threshold}MB threshold"
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Memory leak benchmark failed: {e}")
            tracemalloc.stop()
        finally:
            self.stop_monitoring()
            result.execution_time = time.time() - start_time
            result.system_info = self.get_system_stats()
        
        return result


class AudioAnalysisStressBenchmark(AbstractBenchmark):
    """Stress test audio analysis performance"""
    
    def __init__(self, audio_analyzer: AdvancedAudioAnalyzer, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE):
        super().__init__(f"Audio Analysis Stress Test ({severity.value})", BenchmarkType.STRESS, severity)
        self.audio_analyzer = audio_analyzer
        
    async def run_benchmark(self, **kwargs) -> BenchmarkResult:
        start_time = time.time()
        result = BenchmarkResult(
            test_name=self.name,
            benchmark_type=self.benchmark_type,
            severity=self.severity,
            success=False,
            execution_time=0.0
        )
        
        try:
            self.start_monitoring()
            
            # Test parameters
            test_params = {
                BenchmarkSeverity.LIGHT: {"audio_samples": 10, "concurrent_tasks": 2, "sample_rate": 44100, "duration": 1.0},
                BenchmarkSeverity.MODERATE: {"audio_samples": 25, "concurrent_tasks": 4, "sample_rate": 44100, "duration": 2.0},
                BenchmarkSeverity.HEAVY: {"audio_samples": 50, "concurrent_tasks": 8, "sample_rate": 48000, "duration": 4.0},
                BenchmarkSeverity.EXTREME: {"audio_samples": 100, "concurrent_tasks": 12, "sample_rate": 96000, "duration": 8.0},
                BenchmarkSeverity.TORTURE: {"audio_samples": 200, "concurrent_tasks": 16, "sample_rate": 96000, "duration": 16.0}
            }
            
            params = test_params[self.severity]
            audio_samples = params["audio_samples"]
            concurrent_tasks = params["concurrent_tasks"]
            sample_rate = params["sample_rate"]
            duration = params["duration"]
            
            # Generate test audio data
            def generate_test_audio(sample_id: int) -> np.ndarray:
                samples = int(sample_rate * duration)
                t = np.linspace(0, duration, samples)
                
                # Generate complex waveform (simulating hardcore kick)
                frequency = 60 + (sample_id % 40)  # 60-100Hz range
                kick = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 10)
                
                # Add harmonics and noise
                harmonics = 0.3 * np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t * 15)
                noise = 0.1 * np.random.normal(0, 1, samples)
                
                return kick + harmonics + noise
            
            test_audio_data = []
            for i in range(audio_samples):
                audio = generate_test_audio(i)
                test_audio_data.append(audio)
            
            # Track analysis performance
            analysis_times = []
            successful_analyses = 0
            failed_analyses = 0
            
            # Analysis functions to test
            analysis_functions = [
                ("basic_analysis", self.audio_analyzer.analyze_pattern_dna),
                ("kick_dna", self.audio_analyzer.analyze_kick_dna),
                ("psychoacoustic", self.audio_analyzer.analyze_psychoacoustic_properties),
                ("spectral", self.audio_analyzer.analyze_spectral_features)
            ]
            
            async def analyze_audio_sample(audio_data: np.ndarray, analysis_type: str, analysis_func: Callable):
                try:
                    start = time.perf_counter()
                    
                    if analysis_type == "psychoacoustic":
                        # This function returns a dict, not an object
                        await analysis_func(audio_data)
                    else:
                        await analysis_func(audio_data)
                    
                    end = time.perf_counter()
                    analysis_time = (end - start) * 1000
                    
                    nonlocal successful_analyses, analysis_times
                    successful_analyses += 1
                    analysis_times.append(analysis_time)
                    
                    return analysis_time
                    
                except Exception as e:
                    logger.warning(f"Analysis {analysis_type} failed: {e}")
                    nonlocal failed_analyses
                    failed_analyses += 1
                    return None
            
            # Run concurrent analysis tasks
            semaphore = asyncio.Semaphore(concurrent_tasks)
            
            async def run_analysis_with_semaphore(audio_data, analysis_type, analysis_func):
                async with semaphore:
                    return await analyze_audio_sample(audio_data, analysis_type, analysis_func)
            
            # Create all analysis tasks
            all_tasks = []
            for audio_data in test_audio_data:
                for analysis_type, analysis_func in analysis_functions:
                    task = run_analysis_with_semaphore(audio_data, analysis_type, analysis_func)
                    all_tasks.append(task)
            
            # Execute all tasks
            task_results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Process results
            valid_times = [t for t in task_results if isinstance(t, (int, float)) and t is not None]
            analysis_times.extend(valid_times)
            
            total_analyses = len(all_tasks)
            successful_analyses = len(valid_times)
            failed_analyses = total_analyses - successful_analyses
            
            if analysis_times:
                result.add_metric("avg_analysis_time", statistics.mean(analysis_times), "ms")
                result.add_metric("min_analysis_time", min(analysis_times), "ms")
                result.add_metric("max_analysis_time", max(analysis_times), "ms")
                result.add_metric("p95_analysis_time", statistics.quantiles(analysis_times, n=20)[18], "ms")
                result.add_metric("p99_analysis_time", statistics.quantiles(analysis_times, n=100)[98], "ms")
                
                # Calculate throughput
                total_time = result.execution_time if result.execution_time > 0 else 1
                throughput = successful_analyses / total_time
                result.add_metric("analysis_throughput", throughput, "analyses/sec")
            
            result.add_metric("total_analyses", total_analyses, "count")
            result.add_metric("successful_analyses", successful_analyses, "count")
            result.add_metric("failed_analyses", failed_analyses, "count")
            result.add_metric("success_rate", (successful_analyses / total_analyses) * 100, "percent")
            result.add_metric("concurrent_tasks", concurrent_tasks, "count")
            result.add_metric("audio_samples", audio_samples, "count")
            result.add_metric("sample_rate", sample_rate, "Hz")
            result.add_metric("audio_duration", duration, "seconds")
            
            result.success = successful_analyses > total_analyses * 0.8  # 80% success threshold
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Audio analysis stress benchmark failed: {e}")
        finally:
            self.stop_monitoring()
            result.execution_time = time.time() - start_time
            result.system_info = self.get_system_stats()
        
        return result


class PatternEvolutionScalabilityBenchmark(AbstractBenchmark):
    """Test pattern evolution scalability"""
    
    def __init__(self, evolution_engine: PatternEvolutionEngine, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE):
        super().__init__(f"Pattern Evolution Scalability ({severity.value})", BenchmarkType.SCALABILITY, severity)
        self.evolution_engine = evolution_engine
        
    async def run_benchmark(self, **kwargs) -> BenchmarkResult:
        start_time = time.time()
        result = BenchmarkResult(
            test_name=self.name,
            benchmark_type=self.benchmark_type,
            severity=self.severity,
            success=False,
            execution_time=0.0
        )
        
        try:
            self.start_monitoring()
            
            # Test parameters - scaling population sizes
            test_params = {
                BenchmarkSeverity.LIGHT: {"max_population": 20, "generations": 3, "tests": 3},
                BenchmarkSeverity.MODERATE: {"max_population": 50, "generations": 5, "tests": 5},
                BenchmarkSeverity.HEAVY: {"max_population": 100, "generations": 10, "tests": 8},
                BenchmarkSeverity.EXTREME: {"max_population": 200, "generations": 15, "tests": 10},
                BenchmarkSeverity.TORTURE: {"max_population": 500, "generations": 20, "tests": 12}
            }
            
            params = test_params[self.severity]
            max_population = params["max_population"]
            generations = params["generations"]
            test_points = params["tests"]
            
            # Test different population sizes
            population_sizes = [int(max_population * (i + 1) / test_points) for i in range(test_points)]
            scalability_results = []
            
            for pop_size in population_sizes:
                logger.info(f"Testing population size: {pop_size}")
                
                try:
                    # Measure generation time
                    gen_start = time.perf_counter()
                    population = await self.evolution_engine.generate_population(
                        population_size=pop_size,
                        base_bpm=150
                    )
                    gen_end = time.perf_counter()
                    generation_time = (gen_end - gen_start) * 1000
                    
                    # Measure evolution time
                    evolution_times = []
                    for gen in range(generations):
                        evo_start = time.perf_counter()
                        population = await self.evolution_engine.evolve_generation(population)
                        evo_end = time.perf_counter()
                        evolution_time = (evo_end - evo_start) * 1000
                        evolution_times.append(evolution_time)
                    
                    avg_evolution_time = statistics.mean(evolution_times)
                    total_evolution_time = sum(evolution_times)
                    
                    # Measure fitness evaluation time
                    fitness_start = time.perf_counter()
                    fitness_scores = []
                    for pattern in population[:min(10, len(population))]:  # Sample for timing
                        fitness = await self.evolution_engine.evaluate_fitness(pattern)
                        fitness_scores.append(fitness.overall)
                    fitness_end = time.perf_counter()
                    
                    avg_fitness_time = ((fitness_end - fitness_start) * 1000) / len(fitness_scores) if fitness_scores else 0
                    
                    scalability_result = {
                        "population_size": pop_size,
                        "generation_time": generation_time,
                        "avg_evolution_time": avg_evolution_time,
                        "total_evolution_time": total_evolution_time,
                        "avg_fitness_time": avg_fitness_time,
                        "avg_fitness_score": statistics.mean(fitness_scores) if fitness_scores else 0.0,
                        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
                    }
                    scalability_results.append(scalability_result)
                    
                    # Add individual metrics
                    result.add_metric(f"pop_{pop_size}_generation_time", generation_time, "ms")
                    result.add_metric(f"pop_{pop_size}_avg_evolution_time", avg_evolution_time, "ms")
                    result.add_metric(f"pop_{pop_size}_total_time", total_evolution_time + generation_time, "ms")
                    result.add_metric(f"pop_{pop_size}_memory_mb", scalability_result["memory_usage"], "MB")
                    
                except Exception as e:
                    logger.error(f"Evolution failed for population size {pop_size}: {e}")
                    scalability_results.append({
                        "population_size": pop_size,
                        "error": str(e)
                    })
                
                # Force garbage collection between tests
                gc.collect()
                
                # Brief pause
                await asyncio.sleep(0.5)
            
            # Analyze scalability
            if len(scalability_results) >= 2:
                successful_results = [r for r in scalability_results if "error" not in r]
                
                if len(successful_results) >= 2:
                    # Calculate scaling factors
                    first_result = successful_results[0]
                    last_result = successful_results[-1]
                    
                    pop_scale_factor = last_result["population_size"] / first_result["population_size"]
                    time_scale_factor = last_result["total_evolution_time"] / first_result["total_evolution_time"] if first_result["total_evolution_time"] > 0 else 0
                    memory_scale_factor = last_result["memory_usage"] / first_result["memory_usage"] if first_result["memory_usage"] > 0 else 0
                    
                    # Calculate efficiency (lower is better)
                    time_efficiency = time_scale_factor / pop_scale_factor if pop_scale_factor > 0 else float('inf')
                    memory_efficiency = memory_scale_factor / pop_scale_factor if pop_scale_factor > 0 else float('inf')
                    
                    result.add_metric("population_scale_factor", pop_scale_factor, "ratio")
                    result.add_metric("time_scale_factor", time_scale_factor, "ratio")
                    result.add_metric("memory_scale_factor", memory_scale_factor, "ratio")
                    result.add_metric("time_efficiency", time_efficiency, "ratio")
                    result.add_metric("memory_efficiency", memory_efficiency, "ratio")
                    
                    # Performance assessment
                    result.success = (
                        len(successful_results) >= len(population_sizes) * 0.8 and  # 80% of tests passed
                        time_efficiency < 2.0 and  # Time doesn't scale worse than O(n^2)
                        memory_efficiency < 2.0     # Memory doesn't scale worse than O(n^2)
                    )
                    
                    if not result.success:
                        issues = []
                        if len(successful_results) < len(population_sizes) * 0.8:
                            issues.append("high failure rate")
                        if time_efficiency >= 2.0:
                            issues.append(f"poor time scaling ({time_efficiency:.2f})")
                        if memory_efficiency >= 2.0:
                            issues.append(f"poor memory scaling ({memory_efficiency:.2f})")
                        result.error_message = f"Scalability issues: {', '.join(issues)}"
            
            result.add_metric("total_population_sizes_tested", len(population_sizes), "count")
            result.add_metric("successful_tests", len([r for r in scalability_results if "error" not in r]), "count")
            result.add_metric("max_population_tested", max_population, "count")
            result.add_metric("generations_per_test", generations, "count")
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Pattern evolution scalability benchmark failed: {e}")
        finally:
            self.stop_monitoring()
            result.execution_time = time.time() - start_time
            result.system_info = self.get_system_stats()
        
        return result


class ComprehensiveBenchmarkSuite:
    """Complete benchmarking suite for hardcore music production system"""
    
    def __init__(self, 
                 synthesizer: AbstractSynthesizer,
                 conversation_engine: ConversationEngine,
                 production_engine: ConversationalProductionEngine,
                 audio_analyzer: AdvancedAudioAnalyzer,
                 evolution_engine: PatternEvolutionEngine,
                 performance_engine: Optional[LivePerformanceEngine] = None):
        
        self.synthesizer = synthesizer
        self.conversation_engine = conversation_engine
        self.production_engine = production_engine
        self.audio_analyzer = audio_analyzer
        self.evolution_engine = evolution_engine
        self.performance_engine = performance_engine
        
        # Initialize all benchmark tests
        self.benchmark_tests: Dict[str, AbstractBenchmark] = {}
        self.results_history: List[BenchmarkResult] = []
        
    def register_benchmarks(self, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE):
        """Register all available benchmark tests"""
        
        # Core performance benchmarks
        self.benchmark_tests["synthesizer_latency"] = SynthesizerLatencyBenchmark(
            self.synthesizer, severity
        )
        
        self.benchmark_tests["ai_conversation_throughput"] = AIConversationThroughputBenchmark(
            self.conversation_engine, severity
        )
        
        self.benchmark_tests["memory_leak_detection"] = MemoryLeakBenchmark(
            self.production_engine, severity
        )
        
        self.benchmark_tests["audio_analysis_stress"] = AudioAnalysisStressBenchmark(
            self.audio_analyzer, severity
        )
        
        self.benchmark_tests["pattern_evolution_scalability"] = PatternEvolutionScalabilityBenchmark(
            self.evolution_engine, severity
        )
        
        logger.info(f"Registered {len(self.benchmark_tests)} benchmark tests at {severity.value} severity")
    
    async def run_full_benchmark_suite(self, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE) -> Dict[str, BenchmarkResult]:
        """Run the complete benchmark suite"""
        logger.info(f"Starting full benchmark suite at {severity.value} severity")
        
        # Register benchmarks at specified severity
        self.register_benchmarks(severity)
        
        suite_start_time = time.time()
        results = {}
        
        # Capture initial system state
        initial_system_state = SystemSnapshot.capture()
        
        for test_name, benchmark in self.benchmark_tests.items():
            logger.info(f"Running benchmark: {test_name}")
            
            try:
                # Run the benchmark
                result = await benchmark.run_benchmark()
                results[test_name] = result
                self.results_history.append(result)
                
                # Log result summary
                status = "✅ PASSED" if result.success else "❌ FAILED"
                logger.info(f"  {status} - {result.execution_time:.2f}s - {len(result.metrics)} metrics")
                
                if result.error_message:
                    logger.warning(f"  Error: {result.error_message}")
                
                # Brief pause between benchmarks
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Benchmark {test_name} crashed: {e}")
                results[test_name] = BenchmarkResult(
                    test_name=test_name,
                    benchmark_type=benchmark.benchmark_type,
                    severity=severity,
                    success=False,
                    execution_time=0.0,
                    error_message=str(e)
                )
        
        suite_end_time = time.time()
        final_system_state = SystemSnapshot.capture()
        
        # Generate suite summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.success)
        suite_duration = suite_end_time - suite_start_time
        
        logger.info(f"Benchmark suite completed: {passed_tests}/{total_tests} passed in {suite_duration:.1f}s")
        
        # Add suite metadata to results
        suite_info = {
            "suite_start_time": suite_start_time,
            "suite_end_time": suite_end_time,
            "suite_duration": suite_duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "severity": severity.value,
            "initial_system_state": initial_system_state,
            "final_system_state": final_system_state
        }
        
        results["_suite_info"] = suite_info
        
        return results
    
    async def run_single_benchmark(self, test_name: str, severity: BenchmarkSeverity = BenchmarkSeverity.MODERATE) -> BenchmarkResult:
        """Run a single benchmark test"""
        if test_name not in self.benchmark_tests:
            self.register_benchmarks(severity)
        
        if test_name not in self.benchmark_tests:
            raise ValueError(f"Unknown benchmark test: {test_name}")
        
        benchmark = self.benchmark_tests[test_name]
        logger.info(f"Running single benchmark: {test_name} at {severity.value} severity")
        
        result = await benchmark.run_benchmark()
        self.results_history.append(result)
        
        status = "✅ PASSED" if result.success else "❌ FAILED"
        logger.info(f"Benchmark {test_name} {status} in {result.execution_time:.2f}s")
        
        return result
    
    def generate_performance_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate human-readable performance report"""
        report_lines = []
        
        report_lines.append("🔥 HARDCORE MUSIC PRODUCTION PERFORMANCE REPORT 🔥")
        report_lines.append("=" * 60)
        
        # Suite summary
        if "_suite_info" in results:
            suite_info = results["_suite_info"]
            report_lines.append(f"Suite Duration: {suite_info['suite_duration']:.1f}s")
            report_lines.append(f"Tests Passed: {suite_info['passed_tests']}/{suite_info['total_tests']}")
            report_lines.append(f"Success Rate: {suite_info['success_rate']:.1f}%")
            report_lines.append(f"Severity Level: {suite_info['severity'].upper()}")
            report_lines.append("")
        
        # Individual test results
        for test_name, result in results.items():
            if test_name == "_suite_info":
                continue
            
            status_emoji = "✅" if result.success else "❌"
            report_lines.append(f"{status_emoji} {result.test_name}")
            report_lines.append(f"   Execution Time: {result.execution_time:.2f}s")
            
            if result.error_message:
                report_lines.append(f"   Error: {result.error_message}")
            
            # Key metrics
            key_metrics = []
            for metric in result.metrics[:5]:  # Show top 5 metrics
                if metric.unit == "ms":
                    key_metrics.append(f"{metric.name}: {metric.value:.1f}ms")
                elif metric.unit == "percent":
                    key_metrics.append(f"{metric.name}: {metric.value:.1f}%")
                elif metric.unit == "MB":
                    key_metrics.append(f"{metric.name}: {metric.value:.1f}MB")
                else:
                    key_metrics.append(f"{metric.name}: {metric.value:.1f} {metric.unit}")
            
            if key_metrics:
                report_lines.append(f"   Key Metrics: {', '.join(key_metrics[:3])}")
            
            if result.system_info:
                sys_info = result.system_info
                if "cpu_avg" in sys_info:
                    report_lines.append(f"   Avg CPU: {sys_info['cpu_avg']:.1f}%")
                if "memory_peak_mb" in sys_info:
                    report_lines.append(f"   Peak Memory: {sys_info['memory_peak_mb']:.1f}MB")
            
            report_lines.append("")
        
        # Performance recommendations
        report_lines.append("🎯 PERFORMANCE RECOMMENDATIONS")
        report_lines.append("-" * 30)
        
        recommendations = self._generate_recommendations(results)
        for rec in recommendations:
            report_lines.append(f"• {rec}")
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self, results: Dict[str, BenchmarkResult]) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []
        
        for test_name, result in results.items():
            if test_name == "_suite_info" or not result.success:
                continue
            
            # Latency recommendations
            if "latency" in test_name.lower():
                latency_metric = result.get_metric("avg_latency")
                if latency_metric and latency_metric.value > 50:
                    recommendations.append(f"High latency detected ({latency_metric.value:.1f}ms) - consider audio buffer optimization")
            
            # Memory recommendations
            if "memory" in test_name.lower():
                memory_growth = result.get_metric("total_memory_growth")
                if memory_growth and memory_growth.value > 100:
                    recommendations.append(f"High memory growth ({memory_growth.value:.1f}MB) - review memory management")
            
            # Throughput recommendations
            if "throughput" in test_name.lower():
                success_rate = result.get_metric("success_rate")
                if success_rate and success_rate.value < 90:
                    recommendations.append(f"Low success rate ({success_rate.value:.1f}%) - investigate error handling")
            
            # CPU recommendations
            if result.system_info and "cpu_avg" in result.system_info:
                cpu_avg = result.system_info["cpu_avg"]
                if cpu_avg > 80:
                    recommendations.append(f"High CPU usage ({cpu_avg:.1f}%) - consider load balancing")
        
        if not recommendations:
            recommendations.append("All systems performing within acceptable parameters 🚀")
        
        return recommendations
    
    def export_results_json(self, results: Dict[str, BenchmarkResult], filename: str):
        """Export results to JSON file"""
        try:
            export_data = {}
            
            for test_name, result in results.items():
                if test_name == "_suite_info":
                    export_data[test_name] = result
                else:
                    export_data[test_name] = {
                        "test_name": result.test_name,
                        "benchmark_type": result.benchmark_type.value,
                        "severity": result.severity.value,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "error_message": result.error_message,
                        "metrics": [
                            {
                                "name": m.name,
                                "value": m.value,
                                "unit": m.unit,
                                "timestamp": m.timestamp,
                                "context": m.context
                            }
                            for m in result.metrics
                        ],
                        "system_info": result.system_info
                    }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Benchmark results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark runs"""
        if not self.results_history:
            return {"message": "No benchmark results available"}
        
        total_runs = len(self.results_history)
        successful_runs = sum(1 for r in self.results_history if r.success)
        
        # Group by test type
        test_type_stats = defaultdict(list)
        for result in self.results_history:
            test_type_stats[result.benchmark_type.value].append(result.success)
        
        type_success_rates = {}
        for test_type, successes in test_type_stats.items():
            success_rate = (sum(successes) / len(successes)) * 100
            type_success_rates[test_type] = success_rate
        
        return {
            "total_benchmark_runs": total_runs,
            "successful_runs": successful_runs,
            "overall_success_rate": (successful_runs / total_runs) * 100,
            "test_type_success_rates": type_success_rates,
            "last_run_time": max(r.metrics[0].timestamp for r in self.results_history if r.metrics) if self.results_history else None
        }


# Factory function
def create_comprehensive_benchmark_suite(
    synthesizer: AbstractSynthesizer,
    conversation_engine: ConversationEngine,
    production_engine: ConversationalProductionEngine,
    audio_analyzer: AdvancedAudioAnalyzer,
    evolution_engine: PatternEvolutionEngine,
    performance_engine: Optional[LivePerformanceEngine] = None
) -> ComprehensiveBenchmarkSuite:
    """Create comprehensive benchmark suite with all components"""
    
    return ComprehensiveBenchmarkSuite(
        synthesizer=synthesizer,
        conversation_engine=conversation_engine,
        production_engine=production_engine,
        audio_analyzer=audio_analyzer,
        evolution_engine=evolution_engine,
        performance_engine=performance_engine
    )


if __name__ == "__main__":
    # Demo the benchmark suite
    async def demo_benchmark_suite():
        from ..interfaces.synthesizer import MockSynthesizer
        from ..ai.conversation_engine import create_conversation_engine
        from ..production.conversational_production_engine import create_conversational_production_engine
        from ..analysis.advanced_audio_analyzer import AdvancedAudioAnalyzer
        from ..evolution.pattern_evolution_engine import PatternEvolutionEngine
        
        print("🔥 HARDCORE BENCHMARK SUITE DEMO 🔥")
        print("=" * 50)
        
        # Create all components
        synth = MockSynthesizer()
        conv_engine = create_conversation_engine(synth)
        prod_engine = create_conversational_production_engine(synth)
        audio_analyzer = AdvancedAudioAnalyzer()
        evolution_engine = PatternEvolutionEngine()
        
        # Create benchmark suite
        benchmark_suite = create_comprehensive_benchmark_suite(
            synthesizer=synth,
            conversation_engine=conv_engine,
            production_engine=prod_engine,
            audio_analyzer=audio_analyzer,
            evolution_engine=evolution_engine
        )
        
        # Run light benchmark suite for demo
        print("Running LIGHT severity benchmark suite...")
        results = await benchmark_suite.run_full_benchmark_suite(BenchmarkSeverity.LIGHT)
        
        # Generate and display report
        report = benchmark_suite.generate_performance_report(results)
        print("\n" + report)
        
        # Show summary
        summary = benchmark_suite.get_benchmark_summary()
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"Total runs: {summary['total_benchmark_runs']}")
        print(f"Success rate: {summary['overall_success_rate']:.1f}%")
        
        # Export results
        benchmark_suite.export_results_json(results, "/tmp/benchmark_results.json")
        print(f"\n💾 Results exported to /tmp/benchmark_results.json")
        
        print("\n🎯 BENCHMARK DEMO COMPLETED 🎯")
    
    # Run demo
    asyncio.run(demo_benchmark_suite())