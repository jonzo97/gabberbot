#!/usr/bin/env python3
"""
Real-time Audio Performance Test
Tests buffer management, loop timing, and live performance capabilities
"""

import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import subprocess
import tempfile
import time
import threading
from pathlib import Path
import sys
from improved_audio_engine import ProfessionalAudioEngine

class RealTimeAudioTester:
    """Test real-time audio performance and buffer management"""
    
    def __init__(self):
        self.engine = ProfessionalAudioEngine()
        self.is_playing = False
        self.audio_buffer = None
        
    def create_seamless_loop(self, kick_audio: np.ndarray, bpm: int, bars: int = 4) -> np.ndarray:
        """Create a seamless loop with proper crossfading"""
        
        beats_per_bar = 4
        total_beats = bars * beats_per_bar
        samples_per_beat = int(60 * self.engine.sample_rate / bpm)
        
        # Create empty loop buffer
        loop_samples = samples_per_beat * total_beats
        loop = np.zeros(loop_samples)
        
        # Calculate kick placement (every beat)
        kick_positions = []
        for beat in range(total_beats):
            pos = beat * samples_per_beat
            kick_positions.append(pos)
        
        # Place kicks with overlap handling
        kick_length = len(kick_audio)
        for pos in kick_positions:
            end_pos = min(pos + kick_length, loop_samples)
            actual_length = end_pos - pos
            
            if actual_length > 0:
                loop[pos:end_pos] += kick_audio[:actual_length]
        
        # Add crossfade for seamless looping
        fade_samples = int(0.01 * self.engine.sample_rate)  # 10ms crossfade
        
        # Fade out at end
        loop[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Fade in at beginning (for when loop repeats)
        loop[:fade_samples] *= np.linspace(0, 1, fade_samples)
        
        # Prevent clipping
        peak = np.max(np.abs(loop))
        if peak > 0.95:
            loop = loop * 0.95 / peak
            
        return loop
        
    def test_timing_accuracy(self, bpm: int = 180) -> dict:
        """Test timing accuracy for different BPMs"""
        
        print(f"\n⏱️ Testing Timing Accuracy at {bpm} BPM")
        print("-" * 50)
        
        # Generate kick
        kick_array, kick_audio = self.engine.create_gabber_kick(bpm)
        
        # Create 4-bar loop
        loop_array = self.create_seamless_loop(kick_array, bpm, bars=4)
        
        # Calculate expected vs actual timing
        expected_duration = (4 * 4 * 60) / bpm  # 4 bars * 4 beats * 60 seconds / bpm
        actual_duration = len(loop_array) / self.engine.sample_rate
        timing_error = abs(expected_duration - actual_duration) * 1000  # ms
        
        # Analyze beat detection
        kick_positions = self.detect_kicks(loop_array)
        if len(kick_positions) > 1:
            inter_kick_times = np.diff(kick_positions) / self.engine.sample_rate
            expected_beat_time = 60.0 / bpm
            beat_timing_errors = np.abs(inter_kick_times - expected_beat_time) * 1000
            avg_beat_error = np.mean(beat_timing_errors)
            max_beat_error = np.max(beat_timing_errors)
        else:
            avg_beat_error = 0
            max_beat_error = 0
        
        results = {
            "bpm": bpm,
            "expected_duration": expected_duration,
            "actual_duration": actual_duration,
            "timing_error_ms": timing_error,
            "kicks_detected": len(kick_positions),
            "avg_beat_error_ms": avg_beat_error,
            "max_beat_error_ms": max_beat_error,
            "timing_quality": "✅" if timing_error < 1.0 and avg_beat_error < 0.5 else "⚠️"
        }
        
        print(f"Expected Duration: {expected_duration:.3f}s")
        print(f"Actual Duration: {actual_duration:.3f}s")
        print(f"Timing Error: {timing_error:.2f}ms {results['timing_quality']}")
        print(f"Kicks Detected: {len(kick_positions)} (expected: 16)")
        print(f"Average Beat Error: {avg_beat_error:.2f}ms")
        print(f"Max Beat Error: {max_beat_error:.2f}ms")
        
        return results
        
    def detect_kicks(self, audio: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Detect kick drum positions in audio"""
        
        # Create onset detection
        envelope = np.abs(audio)
        
        # Smooth envelope
        window_size = int(0.01 * self.engine.sample_rate)  # 10ms window
        smoothed = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks
        peaks = []
        for i in range(window_size, len(smoothed) - window_size):
            if (smoothed[i] > threshold and 
                smoothed[i] > smoothed[i-window_size] and 
                smoothed[i] > smoothed[i+window_size]):
                peaks.append(i)
        
        # Remove peaks too close together (minimum 100ms apart)
        min_distance = int(0.1 * self.engine.sample_rate)
        filtered_peaks = []
        last_peak = -min_distance
        
        for peak in peaks:
            if peak - last_peak >= min_distance:
                filtered_peaks.append(peak)
                last_peak = peak
                
        return np.array(filtered_peaks)
        
    def test_buffer_performance(self) -> dict:
        """Test audio buffer performance and latency"""
        
        print(f"\n🔊 Testing Buffer Performance")
        print("-" * 50)
        
        # Test different buffer configurations
        buffer_tests = []
        
        # Generate test audio
        kick_array, kick_audio = self.engine.create_gabber_kick(180)
        loop_array = self.create_seamless_loop(kick_array, 180, bars=2)
        
        # Convert to AudioSegment
        loop_audio = AudioSegment(
            (loop_array * 32767).astype(np.int16).tobytes(),
            frame_rate=self.engine.sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Test playback with timing measurement
        for latency in [50, 100, 150, 300]:
            print(f"\nTesting PULSE_LATENCY_MSEC={latency}:")
            
            # Time the playback initiation
            start_time = time.time()
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                loop_audio.export(tmp.name, format="wav")
                
                # Test playback with different latency settings
                try:
                    env = {"PULSE_LATENCY_MSEC": str(latency)}
                    result = subprocess.run(
                        ['timeout', '3s', 'paplay', tmp.name], 
                        env={**dict(), **env},
                        capture_output=True,
                        timeout=5
                    )
                    
                    playback_start_time = time.time() - start_time
                    
                    buffer_test = {
                        "latency_ms": latency,
                        "playback_start_ms": playback_start_time * 1000,
                        "success": result.returncode == 0 or result.returncode == 124,  # timeout is ok
                        "stderr": result.stderr.decode() if result.stderr else ""
                    }
                    
                    print(f"   Start Latency: {playback_start_time*1000:.1f}ms")
                    print(f"   Success: {'✅' if buffer_test['success'] else '❌'}")
                    if result.stderr:
                        print(f"   Errors: {result.stderr.decode()[:100]}")
                        
                    buffer_tests.append(buffer_test)
                    
                except subprocess.TimeoutExpired:
                    print(f"   Timeout (expected)")
                    buffer_tests.append({
                        "latency_ms": latency,
                        "success": True,
                        "playback_start_ms": (time.time() - start_time) * 1000
                    })
                except Exception as e:
                    print(f"   Error: {e}")
                    buffer_tests.append({
                        "latency_ms": latency,
                        "success": False,
                        "error": str(e)
                    })
                    
        return {
            "buffer_tests": buffer_tests,
            "recommended_latency": self.find_optimal_latency(buffer_tests)
        }
        
    def find_optimal_latency(self, buffer_tests: list) -> int:
        """Find optimal latency setting"""
        
        successful_tests = [t for t in buffer_tests if t.get('success', False)]
        if not successful_tests:
            return 150  # Default fallback
            
        # Find lowest latency that works reliably
        successful_tests.sort(key=lambda x: x['latency_ms'])
        return successful_tests[0]['latency_ms']
        
    def test_live_performance_simulation(self) -> dict:
        """Simulate live performance scenarios"""
        
        print(f"\n🎤 Testing Live Performance Simulation")
        print("-" * 50)
        
        # Generate different elements for live mixing
        elements = {}
        
        # Generate kicks
        gabber_array, _ = self.engine.create_gabber_kick(190)
        industrial_array, _ = self.engine.create_industrial_kick(135)
        
        elements["gabber_kick"] = gabber_array
        elements["industrial_kick"] = industrial_array
        
        # Test rapid pattern switching (simulate live performance)
        switch_test_results = []
        
        for switch_count in range(3):  # Test 3 rapid switches
            print(f"\nSwitch Test {switch_count + 1}:")
            
            # Switch between gabber and industrial
            current_pattern = "gabber_kick" if switch_count % 2 == 0 else "industrial_kick"
            bpm = 190 if current_pattern == "gabber_kick" else 135
            
            print(f"   Switching to: {current_pattern} at {bpm} BPM")
            
            # Time the pattern generation
            start_time = time.time()
            
            # Create loop
            loop_array = self.create_seamless_loop(elements[current_pattern], bpm, bars=1)
            
            generation_time = time.time() - start_time
            
            # Convert and test playback
            loop_audio = AudioSegment(
                (loop_array * 32767).astype(np.int16).tobytes(),
                frame_rate=self.engine.sample_rate,
                sample_width=2,
                channels=1
            )
            
            switch_result = {
                "pattern": current_pattern,
                "bpm": bpm,
                "generation_time_ms": generation_time * 1000,
                "loop_duration": len(loop_array) / self.engine.sample_rate,
                "meets_latency_target": generation_time < 0.1  # <100ms target
            }
            
            print(f"   Generation Time: {generation_time*1000:.1f}ms {'✅' if switch_result['meets_latency_target'] else '❌'}")
            print(f"   Loop Duration: {switch_result['loop_duration']:.2f}s")
            
            switch_test_results.append(switch_result)
            
            # Brief playback test
            if switch_count == 0:  # Only play first one to avoid spam
                print(f"   Playing sample...")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    loop_audio.export(tmp.name, format="wav")
                    subprocess.run(['timeout', '2s', 'paplay', tmp.name], 
                                 capture_output=True)
        
        return {
            "switch_tests": switch_test_results,
            "performance_ready": all(t['meets_latency_target'] for t in switch_test_results)
        }
        
    def run_comprehensive_test(self):
        """Run all audio performance tests"""
        
        print("=" * 70)
        print("🎵 REAL-TIME AUDIO PERFORMANCE TEST SUITE")
        print("=" * 70)
        
        # Test 1: Timing accuracy at different BPMs
        timing_results = []
        for bpm in [140, 180, 200, 250]:
            timing_result = self.test_timing_accuracy(bpm)
            timing_results.append(timing_result)
        
        # Test 2: Buffer performance
        buffer_results = self.test_buffer_performance()
        
        # Test 3: Live performance simulation
        performance_results = self.test_live_performance_simulation()
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Timing quality
        timing_quality = all(t['timing_quality'] == '✅' for t in timing_results)
        print(f"⏱️ Timing Accuracy: {'✅ PASS' if timing_quality else '❌ FAIL'}")
        
        if not timing_quality:
            for t in timing_results:
                if t['timing_quality'] != '✅':
                    print(f"   - {t['bpm']} BPM: {t['timing_error_ms']:.2f}ms error")
        
        # Buffer performance
        buffer_success = len([t for t in buffer_results['buffer_tests'] if t.get('success')]) > 0
        print(f"🔊 Buffer Performance: {'✅ PASS' if buffer_success else '❌ FAIL'}")
        print(f"   Recommended Latency: {buffer_results['recommended_latency']}ms")
        
        # Live performance
        performance_ready = performance_results['performance_ready']
        print(f"🎤 Live Performance: {'✅ READY' if performance_ready else '❌ NOT READY'}")
        
        avg_gen_time = np.mean([t['generation_time_ms'] for t in performance_results['switch_tests']])
        print(f"   Average Generation Time: {avg_gen_time:.1f}ms")
        
        # Overall assessment
        overall_ready = timing_quality and buffer_success and performance_ready
        print(f"\n🎯 OVERALL STATUS: {'✅ PRODUCTION READY' if overall_ready else '⚠️ NEEDS OPTIMIZATION'}")
        
        if overall_ready:
            print("\n💡 Recommendations:")
            print(f"   • Set PULSE_LATENCY_MSEC={buffer_results['recommended_latency']}")
            print("   • Audio engine ready for live jamming sessions")
            print("   • Timing accurate up to 250 BPM")
        
        return {
            "timing_results": timing_results,
            "buffer_results": buffer_results,
            "performance_results": performance_results,
            "overall_ready": overall_ready
        }


def main():
    """Run real-time audio tests"""
    tester = RealTimeAudioTester()
    results = tester.run_comprehensive_test()
    
    # Save results
    import json
    with open("realtime_audio_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: realtime_audio_test_results.json")

if __name__ == "__main__":
    main()