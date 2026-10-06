#!/usr/bin/env python3
"""
BMAD Performance Monitor - Optimization and Monitoring System
@music-archivist (Keeper) - Phase 4 Performance System

Complete performance monitoring and optimization system for BMAD:
- Real-time performance monitoring during generation
- Automatic optimization recommendations
- Resource usage tracking and analysis
- Performance benchmarking and comparison
- Memory management and cleanup
- System health monitoring
"""

import os
import sys
import time
import json
import psutil
import asyncio
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import weakref
import gc
import tracemalloc

# Import BMAD components for monitoring
from bmad_hardcore_factory import BMADHardcoreFactory, BMADFactoryConfig, ProductionMode, QualityLevel


class PerformanceLevel(Enum):
    """Performance levels for optimization"""
    OPTIMAL = "optimal"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Types of system resources to monitor"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GENERATION_TIME = "generation_time"
    QUEUE_SIZE = "queue_size"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: datetime
    metric_type: ResourceType
    value: float
    unit: str
    threshold_warning: float = 0.0
    threshold_critical: float = 0.0
    
    def get_performance_level(self) -> PerformanceLevel:
        """Determine performance level based on thresholds"""
        if self.value >= self.threshold_critical:
            return PerformanceLevel.CRITICAL
        elif self.value >= self.threshold_warning:
            return PerformanceLevel.POOR
        elif self.value >= self.threshold_warning * 0.8:
            return PerformanceLevel.MODERATE
        elif self.value >= self.threshold_warning * 0.6:
            return PerformanceLevel.GOOD
        else:
            return PerformanceLevel.OPTIMAL


@dataclass
class GenerationMetrics:
    """Metrics for individual generation sessions"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Generation metrics
    tracks_generated: int = 0
    tracks_target: int = 0
    generation_time_seconds: float = 0.0
    tracks_per_hour: float = 0.0
    
    # Resource metrics
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Quality metrics
    quality_checks_passed: int = 0
    quality_checks_failed: int = 0
    error_count: int = 0
    
    # Configuration
    production_mode: str = ""
    quality_level: str = ""
    use_evolution: bool = False
    parallel_processing: bool = False
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-1)"""
        # Success rate component
        success_rate = self.tracks_generated / max(1, self.tracks_target)
        
        # Speed component (normalized)
        target_tracks_per_hour = 30.0  # Target baseline
        speed_score = min(1.0, self.tracks_per_hour / target_tracks_per_hour)
        
        # Quality component
        quality_rate = self.quality_checks_passed / max(1, self.quality_checks_passed + self.quality_checks_failed)
        
        # Error penalty
        error_penalty = max(0.0, 1.0 - (self.error_count * 0.1))
        
        # Combine components
        performance_score = (success_rate * 0.3 + speed_score * 0.3 + quality_rate * 0.3 + error_penalty * 0.1)
        return min(1.0, max(0.0, performance_score))


@dataclass
class SystemSnapshot:
    """System resource snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_usage_percent: float
    available_memory_mb: float
    active_processes: int
    generation_active: bool = False


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str
    priority: str  # high, medium, low
    title: str
    description: str
    impact: str
    implementation: str
    estimated_improvement: str


class BMADPerformanceMonitor:
    """
    BMAD Performance Monitor
    
    Provides comprehensive performance monitoring, optimization recommendations,
    and system health tracking for BMAD hardcore music generation.
    """
    
    def __init__(self, monitoring_interval: float = 1.0, output_directory: str = "bmad_performance_logs"):
        """Initialize performance monitor"""
        self.monitoring_interval = monitoring_interval
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Data storage
        self.performance_metrics: List[PerformanceMetric] = []
        self.generation_metrics: List[GenerationMetrics] = []
        self.system_snapshots: List[SystemSnapshot] = []
        
        # Current session tracking
        self.current_session: Optional[GenerationMetrics] = None
        self.session_start_snapshot: Optional[SystemSnapshot] = None
        
        # Performance thresholds
        self.thresholds = self._setup_performance_thresholds()
        
        # Memory tracking
        self.memory_tracker_active = False
        
        # Weak references to monitored objects
        self.monitored_objects: List[weakref.ref] = []
        
        self.logger.info("BMAD Performance Monitor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up performance monitor logging"""
        logger = logging.getLogger("bmad_performance")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = self.output_directory / f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _setup_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Set up performance thresholds"""
        return {
            "cpu": {
                "warning": 80.0,  # 80% CPU
                "critical": 95.0  # 95% CPU
            },
            "memory": {
                "warning": 1000.0,  # 1GB memory
                "critical": 2000.0  # 2GB memory
            },
            "memory_percent": {
                "warning": 80.0,  # 80% of system memory
                "critical": 95.0   # 95% of system memory
            },
            "generation_time": {
                "warning": 120.0,  # 2 minutes per track
                "critical": 300.0  # 5 minutes per track
            },
            "disk_usage": {
                "warning": 90.0,  # 90% disk usage
                "critical": 98.0   # 98% disk usage
            }
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Take system snapshot
                snapshot = self._take_system_snapshot()
                self.system_snapshots.append(snapshot)
                
                # Record performance metrics
                self._record_performance_metrics(snapshot)
                
                # Check for performance issues
                self._check_performance_alerts(snapshot)
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep until next monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_system_snapshot(self) -> SystemSnapshot:
        """Take current system resource snapshot"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = (memory.total - memory.available) / 1024 / 1024
            memory_percent = memory.percent
            available_memory_mb = memory.available / 1024 / 1024
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Process count
            active_processes = len(psutil.pids())
            
            return SystemSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                available_memory_mb=available_memory_mb,
                active_processes=active_processes,
                generation_active=self.current_session is not None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to take system snapshot: {e}")
            return SystemSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                available_memory_mb=0.0,
                active_processes=0
            )
    
    def _record_performance_metrics(self, snapshot: SystemSnapshot):
        """Record performance metrics from snapshot"""
        timestamp = snapshot.timestamp
        
        # CPU metric
        cpu_metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=ResourceType.CPU,
            value=snapshot.cpu_percent,
            unit="percent",
            threshold_warning=self.thresholds["cpu"]["warning"],
            threshold_critical=self.thresholds["cpu"]["critical"]
        )
        self.performance_metrics.append(cpu_metric)
        
        # Memory metric
        memory_metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=ResourceType.MEMORY,
            value=snapshot.memory_mb,
            unit="MB",
            threshold_warning=self.thresholds["memory"]["warning"],
            threshold_critical=self.thresholds["memory"]["critical"]
        )
        self.performance_metrics.append(memory_metric)
        
        # Disk metric
        disk_metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=ResourceType.DISK,
            value=snapshot.disk_usage_percent,
            unit="percent",
            threshold_warning=self.thresholds["disk_usage"]["warning"],
            threshold_critical=self.thresholds["disk_usage"]["critical"]
        )
        self.performance_metrics.append(disk_metric)
    
    def _check_performance_alerts(self, snapshot: SystemSnapshot):
        """Check for performance alerts"""
        alerts = []
        
        # CPU alert
        if snapshot.cpu_percent >= self.thresholds["cpu"]["critical"]:
            alerts.append(f"CRITICAL: CPU usage at {snapshot.cpu_percent:.1f}%")
        elif snapshot.cpu_percent >= self.thresholds["cpu"]["warning"]:
            alerts.append(f"WARNING: CPU usage at {snapshot.cpu_percent:.1f}%")
        
        # Memory alert
        if snapshot.memory_percent >= self.thresholds["memory_percent"]["critical"]:
            alerts.append(f"CRITICAL: Memory usage at {snapshot.memory_percent:.1f}%")
        elif snapshot.memory_percent >= self.thresholds["memory_percent"]["warning"]:
            alerts.append(f"WARNING: Memory usage at {snapshot.memory_percent:.1f}%")
        
        # Disk alert
        if snapshot.disk_usage_percent >= self.thresholds["disk_usage"]["critical"]:
            alerts.append(f"CRITICAL: Disk usage at {snapshot.disk_usage_percent:.1f}%")
        elif snapshot.disk_usage_percent >= self.thresholds["disk_usage"]["warning"]:
            alerts.append(f"WARNING: Disk usage at {snapshot.disk_usage_percent:.1f}%")
        
        # Log alerts
        for alert in alerts:
            if "CRITICAL" in alert:
                self.logger.critical(alert)
            else:
                self.logger.warning(alert)
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of data
        
        # Clean performance metrics
        self.performance_metrics = [m for m in self.performance_metrics if m.timestamp > cutoff_time]
        
        # Clean system snapshots
        self.system_snapshots = [s for s in self.system_snapshots if s.timestamp > cutoff_time]
        
        # Clean up weak references
        self.monitored_objects = [ref for ref in self.monitored_objects if ref() is not None]
    
    def start_generation_session(self, session_id: str, config: BMADFactoryConfig) -> GenerationMetrics:
        """Start monitoring a generation session"""
        if self.current_session:
            self.logger.warning(f"Ending previous session {self.current_session.session_id}")
            self.end_generation_session()
        
        self.current_session = GenerationMetrics(
            session_id=session_id,
            start_time=datetime.now(),
            production_mode=config.production_mode.value if hasattr(config.production_mode, 'value') else str(config.production_mode),
            quality_level=config.quality_level.value if hasattr(config.quality_level, 'value') else str(config.quality_level),
            use_evolution=config.use_evolution,
            parallel_processing=config.parallel_processing,
            tracks_target=getattr(config, 'track_count', 1)
        )
        
        self.session_start_snapshot = self._take_system_snapshot()
        
        # Start memory tracking if not already active
        if not self.memory_tracker_active:
            tracemalloc.start()
            self.memory_tracker_active = True
        
        self.logger.info(f"Started monitoring session: {session_id}")
        return self.current_session
    
    def end_generation_session(self) -> Optional[GenerationMetrics]:
        """End current generation session"""
        if not self.current_session:
            return None
        
        session = self.current_session
        session.end_time = datetime.now()
        session.generation_time_seconds = (session.end_time - session.start_time).total_seconds()
        
        # Calculate tracks per hour
        if session.generation_time_seconds > 0:
            session.tracks_per_hour = session.tracks_generated / (session.generation_time_seconds / 3600)
        
        # Get memory statistics if available
        if self.memory_tracker_active:
            try:
                current, peak = tracemalloc.get_traced_memory()
                session.peak_memory_mb = peak / 1024 / 1024
                tracemalloc.stop()
                self.memory_tracker_active = False
            except Exception as e:
                self.logger.warning(f"Failed to get memory statistics: {e}")
        
        # Calculate resource usage during session
        self._calculate_session_resource_usage(session)
        
        # Store completed session
        self.generation_metrics.append(session)
        self.current_session = None
        
        self.logger.info(f"Ended monitoring session: {session.session_id} (Performance score: {session.calculate_performance_score():.2f})")
        return session
    
    def _calculate_session_resource_usage(self, session: GenerationMetrics):
        """Calculate resource usage during session"""
        if not self.session_start_snapshot:
            return
        
        # Find snapshots during session
        session_snapshots = [
            s for s in self.system_snapshots
            if session.start_time <= s.timestamp <= (session.end_time or datetime.now())
        ]
        
        if not session_snapshots:
            return
        
        # Calculate CPU statistics
        cpu_values = [s.cpu_percent for s in session_snapshots]
        session.avg_cpu_percent = sum(cpu_values) / len(cpu_values)
        session.peak_cpu_percent = max(cpu_values)
        
        # Calculate memory statistics
        memory_values = [s.memory_mb for s in session_snapshots]
        session.avg_memory_mb = sum(memory_values) / len(memory_values)
        if not session.peak_memory_mb:  # Only set if not already set by tracemalloc
            session.peak_memory_mb = max(memory_values)
    
    def update_session_progress(self, tracks_generated: int, quality_passed: int = 0, quality_failed: int = 0, errors: int = 0):
        """Update current session progress"""
        if not self.current_session:
            return
        
        self.current_session.tracks_generated = tracks_generated
        self.current_session.quality_checks_passed = quality_passed
        self.current_session.quality_checks_failed = quality_failed
        self.current_session.error_count = errors
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_metrics = [m for m in self.performance_metrics if m.timestamp > cutoff_time]
        recent_sessions = [s for s in self.generation_metrics if s.start_time > cutoff_time]
        recent_snapshots = [s for s in self.system_snapshots if s.timestamp > cutoff_time]
        
        # Calculate summary statistics
        summary = {
            "time_period_hours": hours,
            "monitoring_active": self.monitoring_active,
            "total_sessions": len(recent_sessions),
            "total_tracks_generated": sum(s.tracks_generated for s in recent_sessions),
            "average_generation_time": 0.0,
            "average_performance_score": 0.0,
            "resource_usage": {},
            "performance_alerts": 0,
            "recommendations": []
        }
        
        # Session statistics
        if recent_sessions:
            summary["average_generation_time"] = sum(s.generation_time_seconds for s in recent_sessions) / len(recent_sessions)
            summary["average_performance_score"] = sum(s.calculate_performance_score() for s in recent_sessions) / len(recent_sessions)
        
        # Resource usage statistics
        if recent_snapshots:
            cpu_values = [s.cpu_percent for s in recent_snapshots]
            memory_values = [s.memory_mb for s in recent_snapshots]
            disk_values = [s.disk_usage_percent for s in recent_snapshots]
            
            summary["resource_usage"] = {
                "cpu": {
                    "average": sum(cpu_values) / len(cpu_values),
                    "peak": max(cpu_values),
                    "minimum": min(cpu_values)
                },
                "memory_mb": {
                    "average": sum(memory_values) / len(memory_values),
                    "peak": max(memory_values),
                    "minimum": min(memory_values)
                },
                "disk_percent": {
                    "average": sum(disk_values) / len(disk_values),
                    "peak": max(disk_values),
                    "minimum": min(disk_values)
                }
            }
        
        # Performance alerts
        critical_metrics = [m for m in recent_metrics if m.get_performance_level() == PerformanceLevel.CRITICAL]
        warning_metrics = [m for m in recent_metrics if m.get_performance_level() == PerformanceLevel.POOR]
        summary["performance_alerts"] = len(critical_metrics) + len(warning_metrics)
        
        # Generate recommendations
        summary["recommendations"] = self._generate_optimization_recommendations(recent_sessions, recent_snapshots)
        
        return summary
    
    def _generate_optimization_recommendations(self, sessions: List[GenerationMetrics], snapshots: List[SystemSnapshot]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []
        
        if not sessions:
            return recommendations
        
        # Analyze session performance
        avg_performance = sum(s.calculate_performance_score() for s in sessions) / len(sessions)
        avg_generation_time = sum(s.generation_time_seconds for s in sessions) / len(sessions)
        avg_tracks_per_hour = sum(s.tracks_per_hour for s in sessions) / len(sessions) if sessions else 0
        
        # CPU optimization recommendations
        if snapshots:
            avg_cpu = sum(s.cpu_percent for s in snapshots) / len(snapshots)
            peak_cpu = max(s.cpu_percent for s in snapshots)
            
            if avg_cpu > 80:
                recommendations.append(OptimizationRecommendation(
                    category="CPU",
                    priority="high",
                    title="High CPU Usage Detected",
                    description=f"Average CPU usage is {avg_cpu:.1f}%, which may slow generation",
                    impact="Reduced generation speed, system responsiveness issues",
                    implementation="Enable parallel processing, reduce quality level for testing, close other applications",
                    estimated_improvement="20-40% speed improvement"
                ))
            
            if peak_cpu > 95:
                recommendations.append(OptimizationRecommendation(
                    category="CPU",
                    priority="critical",
                    title="CPU Saturation",
                    description=f"Peak CPU usage reached {peak_cpu:.1f}%",
                    impact="System instability, generation failures",
                    implementation="Reduce concurrent generation, use draft quality, system upgrade needed",
                    estimated_improvement="Prevent system crashes"
                ))
        
        # Memory optimization recommendations
        if snapshots:
            avg_memory = sum(s.memory_mb for s in snapshots) / len(snapshots)
            peak_memory = max(s.memory_mb for s in snapshots)
            
            if avg_memory > 1000:  # 1GB
                recommendations.append(OptimizationRecommendation(
                    category="Memory",
                    priority="medium",
                    title="High Memory Usage",
                    description=f"Average memory usage is {avg_memory:.0f}MB",
                    impact="Slower generation, potential memory errors",
                    implementation="Set memory limits, enable garbage collection, reduce batch sizes",
                    estimated_improvement="10-20% speed improvement"
                ))
        
        # Generation speed recommendations
        if avg_generation_time > 180:  # 3 minutes
            recommendations.append(OptimizationRecommendation(
                category="Generation Speed",
                priority="medium",
                title="Slow Generation Performance",
                description=f"Average generation time is {avg_generation_time:.1f} seconds",
                impact="Low productivity, extended wait times",
                implementation="Use draft quality for testing, disable evolution for simple tracks, enable parallel processing",
                estimated_improvement="50-70% speed improvement"
            ))
        
        # Quality vs speed recommendations
        high_quality_sessions = [s for s in sessions if s.quality_level in ["professional", "mastered"]]
        if len(high_quality_sessions) / len(sessions) > 0.8 and avg_tracks_per_hour < 20:
            recommendations.append(OptimizationRecommendation(
                category="Quality vs Speed",
                priority="low",
                title="Optimize Quality/Speed Balance",
                description="Most generations use high quality settings with low throughput",
                impact="Longer development cycles",
                implementation="Use standard quality for development, reserve mastered for final releases",
                estimated_improvement="2-3x speed improvement for development"
            ))
        
        # Evolution system recommendations
        evolution_sessions = [s for s in sessions if s.use_evolution]
        if evolution_sessions and avg_generation_time > 120:
            avg_evolution_time = sum(s.generation_time_seconds for s in evolution_sessions) / len(evolution_sessions)
            if avg_evolution_time > avg_generation_time * 1.5:
                recommendations.append(OptimizationRecommendation(
                    category="Evolution System",
                    priority="low",
                    title="Evolution Impact on Speed",
                    description="Evolution system significantly increases generation time",
                    impact="Slower iteration during development",
                    implementation="Disable evolution for testing, reduce evolution generations, use evolution selectively",
                    estimated_improvement="30-50% speed improvement"
                ))
        
        return recommendations
    
    def generate_performance_report(self, hours: int = 24) -> str:
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary(hours)
        
        report = f"""# BMAD Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Time Period: {hours} hours

## Summary

- **Monitoring Status**: {'Active' if summary['monitoring_active'] else 'Inactive'}
- **Total Sessions**: {summary['total_sessions']}
- **Total Tracks Generated**: {summary['total_tracks_generated']}
- **Average Generation Time**: {summary['average_generation_time']:.1f} seconds
- **Average Performance Score**: {summary['average_performance_score']:.2f}
- **Performance Alerts**: {summary['performance_alerts']}

## Resource Usage

### CPU
- Average: {summary['resource_usage'].get('cpu', {}).get('average', 0):.1f}%
- Peak: {summary['resource_usage'].get('cpu', {}).get('peak', 0):.1f}%
- Minimum: {summary['resource_usage'].get('cpu', {}).get('minimum', 0):.1f}%

### Memory
- Average: {summary['resource_usage'].get('memory_mb', {}).get('average', 0):.1f} MB
- Peak: {summary['resource_usage'].get('memory_mb', {}).get('peak', 0):.1f} MB
- Minimum: {summary['resource_usage'].get('memory_mb', {}).get('minimum', 0):.1f} MB

### Disk
- Average: {summary['resource_usage'].get('disk_percent', {}).get('average', 0):.1f}%
- Peak: {summary['resource_usage'].get('disk_percent', {}).get('peak', 0):.1f}%

## Optimization Recommendations

"""
        
        if summary['recommendations']:
            for i, rec in enumerate(summary['recommendations'], 1):
                report += f"""
### {i}. {rec.title} ({rec.priority.upper()} priority)

- **Category**: {rec.category}
- **Description**: {rec.description}
- **Impact**: {rec.impact}
- **Implementation**: {rec.implementation}
- **Estimated Improvement**: {rec.estimated_improvement}
"""
        else:
            report += "No specific recommendations at this time. System performance is within acceptable ranges.\n"
        
        report += f"""
## Session Details

"""
        
        recent_sessions = [s for s in self.generation_metrics if s.start_time > datetime.now() - timedelta(hours=hours)]
        if recent_sessions:
            for session in recent_sessions[-5:]:  # Last 5 sessions
                report += f"""
### Session: {session.session_id}
- **Start Time**: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {session.generation_time_seconds:.1f}s
- **Tracks Generated**: {session.tracks_generated}/{session.tracks_target}
- **Performance Score**: {session.calculate_performance_score():.2f}
- **CPU Peak**: {session.peak_cpu_percent:.1f}%
- **Memory Peak**: {session.peak_memory_mb:.1f} MB
- **Configuration**: {session.production_mode}, {session.quality_level}
"""
        else:
            report += "No recent sessions to display.\n"
        
        return report
    
    async def save_performance_data(self):
        """Save performance data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save performance summary
        summary = self.get_performance_summary(24)
        summary_file = self.output_directory / f"performance_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed metrics
        if self.generation_metrics:
            metrics_data = [asdict(m) for m in self.generation_metrics]
            metrics_file = self.output_directory / f"generation_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        
        # Save performance report
        report = self.generate_performance_report(24)
        report_file = self.output_directory / f"performance_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Performance data saved to {self.output_directory}")
    
    def optimize_system_for_generation(self) -> Dict[str, Any]:
        """Perform automatic system optimization for generation"""
        optimizations = {
            "performed": [],
            "recommendations": [],
            "memory_cleaned": 0,
            "objects_collected": 0
        }
        
        # Force garbage collection
        collected = gc.collect()
        optimizations["objects_collected"] = collected
        optimizations["performed"].append(f"Garbage collection: {collected} objects")
        
        # Clean weak references
        initial_refs = len(self.monitored_objects)
        self.monitored_objects = [ref for ref in self.monitored_objects if ref() is not None]
        cleaned_refs = initial_refs - len(self.monitored_objects)
        if cleaned_refs > 0:
            optimizations["performed"].append(f"Cleaned {cleaned_refs} dead weak references")
        
        # System recommendations
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                optimizations["recommendations"].append("Consider closing other applications to free memory")
            
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 70:
                optimizations["recommendations"].append("System CPU usage is high, consider reducing parallel processing")
            
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                optimizations["recommendations"].append("Disk space is low, consider cleaning old output files")
                
        except Exception as e:
            self.logger.warning(f"Failed to get system recommendations: {e}")
        
        return optimizations
    
    def get_current_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            snapshot = self._take_system_snapshot()
            
            status = {
                "timestamp": snapshot.timestamp.isoformat(),
                "cpu_percent": snapshot.cpu_percent,
                "memory_mb": snapshot.memory_mb,
                "memory_percent": snapshot.memory_percent,
                "available_memory_mb": snapshot.available_memory_mb,
                "disk_usage_percent": snapshot.disk_usage_percent,
                "active_processes": snapshot.active_processes,
                "generation_active": snapshot.generation_active,
                "monitoring_active": self.monitoring_active,
                "performance_level": self._determine_overall_performance_level(snapshot),
                "current_session": self.current_session.session_id if self.current_session else None
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    def _determine_overall_performance_level(self, snapshot: SystemSnapshot) -> str:
        """Determine overall system performance level"""
        levels = []
        
        # CPU level
        if snapshot.cpu_percent >= self.thresholds["cpu"]["critical"]:
            levels.append(PerformanceLevel.CRITICAL)
        elif snapshot.cpu_percent >= self.thresholds["cpu"]["warning"]:
            levels.append(PerformanceLevel.POOR)
        else:
            levels.append(PerformanceLevel.GOOD)
        
        # Memory level
        if snapshot.memory_percent >= self.thresholds["memory_percent"]["critical"]:
            levels.append(PerformanceLevel.CRITICAL)
        elif snapshot.memory_percent >= self.thresholds["memory_percent"]["warning"]:
            levels.append(PerformanceLevel.POOR)
        else:
            levels.append(PerformanceLevel.GOOD)
        
        # Return worst level
        if PerformanceLevel.CRITICAL in levels:
            return PerformanceLevel.CRITICAL.value
        elif PerformanceLevel.POOR in levels:
            return PerformanceLevel.POOR.value
        else:
            return PerformanceLevel.GOOD.value


# Context manager for automatic session monitoring
class MonitoredGeneration:
    """Context manager for automatic performance monitoring during generation"""
    
    def __init__(self, monitor: BMADPerformanceMonitor, session_id: str, config: BMADFactoryConfig):
        self.monitor = monitor
        self.session_id = session_id
        self.config = config
        self.session = None
    
    def __enter__(self):
        self.session = self.monitor.start_generation_session(self.session_id, self.config)
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_generation_session()


# Global performance monitor instance
_global_monitor: Optional[BMADPerformanceMonitor] = None

def get_global_monitor() -> BMADPerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = BMADPerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor

def monitor_generation(session_id: str, config: BMADFactoryConfig) -> MonitoredGeneration:
    """Create monitored generation context"""
    monitor = get_global_monitor()
    return MonitoredGeneration(monitor, session_id, config)


# CLI Interface
def main():
    """Command line interface for performance monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BMAD Performance Monitor")
    parser.add_argument("command", choices=["start", "stop", "status", "report", "optimize"], help="Monitor command")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data for report")
    parser.add_argument("--save", action="store_true", help="Save performance data")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger("bmad_performance").setLevel(logging.DEBUG)
    
    monitor = BMADPerformanceMonitor()
    
    if args.command == "start":
        monitor.start_monitoring()
        print("🔍 Performance monitoring started")
        print("Use Ctrl+C to stop monitoring")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\n🛑 Performance monitoring stopped")
    
    elif args.command == "stop":
        monitor.stop_monitoring()
        print("🛑 Performance monitoring stopped")
    
    elif args.command == "status":
        status = monitor.get_current_system_status()
        print(f"\n🖥️  System Status:")
        print(f"CPU: {status.get('cpu_percent', 0):.1f}%")
        print(f"Memory: {status.get('memory_mb', 0):.1f} MB ({status.get('memory_percent', 0):.1f}%)")
        print(f"Disk: {status.get('disk_usage_percent', 0):.1f}%")
        print(f"Performance Level: {status.get('performance_level', 'unknown')}")
        print(f"Monitoring: {'Active' if status.get('monitoring_active') else 'Inactive'}")
        print(f"Generation Active: {'Yes' if status.get('generation_active') else 'No'}")
    
    elif args.command == "report":
        report = monitor.generate_performance_report(args.hours)
        print(report)
        
        if args.save:
            asyncio.run(monitor.save_performance_data())
            print(f"\n💾 Performance data saved to {monitor.output_directory}")
    
    elif args.command == "optimize":
        optimizations = monitor.optimize_system_for_generation()
        print(f"\n⚡ System Optimization Complete:")
        
        if optimizations["performed"]:
            print("Performed:")
            for action in optimizations["performed"]:
                print(f"  ✅ {action}")
        
        if optimizations["recommendations"]:
            print("Recommendations:")
            for rec in optimizations["recommendations"]:
                print(f"  💡 {rec}")
        
        if not optimizations["performed"] and not optimizations["recommendations"]:
            print("  ✅ System is already optimized")


if __name__ == "__main__":
    main()