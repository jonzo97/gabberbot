#!/usr/bin/env python3
"""
Audio Engine Test Harness
Interactive testing and validation of hardcore audio synthesis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
from pydub.generators import Sine
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import our existing synthesis code
sys.path.insert(0, str(Path(__file__).parent))
from cli_strudel.music_assistant import HardcorePatternGenerator

class AudioEngineTestSuite:
    """Comprehensive audio engine testing and validation"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.generator = HardcorePatternGenerator()
        self.test_results = []
        
    def analyze_audio(self, audio: np.ndarray, name: str = "audio") -> Dict:
        """Comprehensive audio analysis"""
        
        # Basic metrics
        duration = len(audio) / self.sample_rate
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        # Frequency analysis
        fft_result = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        
        # Find fundamental frequency
        peak_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
        fundamental_freq = freqs[peak_idx]
        
        # Frequency bands analysis
        bands = {
            "sub": (20, 80),
            "bass": (80, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 5000),
            "high": (5000, 12000)
        }
        
        band_energy = {}
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs < high)
            band_energy[band_name] = np.mean(magnitude[band_mask])
            
        # Attack time measurement
        envelope = np.abs(audio)
        peak_level = np.max(envelope)
        attack_threshold = peak_level * 0.9
        attack_samples = np.argmax(envelope >= attack_threshold)
        attack_time = attack_samples / self.sample_rate
        
        # Harmonic distortion
        if fundamental_freq > 0:
            harmonic_freqs = [fundamental_freq * i for i in range(2, 6)]
            harmonics = {}
            for i, h_freq in enumerate(harmonic_freqs, 2):
                h_idx = np.argmin(np.abs(freqs - h_freq))
                harmonics[f"h{i}"] = magnitude[h_idx] / magnitude[peak_idx]
        else:
            harmonics = {}
            
        return {
            "name": name,
            "duration": duration,
            "peak": peak,
            "rms": rms,
            "fundamental_freq": fundamental_freq,
            "band_energy": band_energy,
            "attack_time": attack_time * 1000,  # Convert to ms
            "harmonics": harmonics,
            "dc_offset": np.mean(audio),
            "clipping": np.sum(np.abs(audio) >= 0.99) / len(audio) * 100  # % clipped samples
        }
        
    def validate_kick_quality(self, kick_audio: np.ndarray, genre: str = "gabber") -> Tuple[bool, List[str]]:
        """Validate kick drum quality against specifications"""
        
        analysis = self.analyze_audio(kick_audio, f"{genre}_kick")
        issues = []
        passed = True
        
        # Genre-specific criteria
        criteria = {
            "gabber": {
                "fundamental_range": (40, 80),
                "attack_time_max": 2.0,  # ms
                "min_harmonics": 2,
                "peak_min": 0.7
            },
            "industrial": {
                "fundamental_range": (35, 70),
                "attack_time_max": 3.0,
                "min_harmonics": 1,
                "peak_min": 0.6
            },
            "hardcore": {
                "fundamental_range": (45, 90),
                "attack_time_max": 1.5,
                "min_harmonics": 3,
                "peak_min": 0.8
            }
        }
        
        spec = criteria.get(genre, criteria["gabber"])
        
        # Check fundamental frequency
        if not (spec["fundamental_range"][0] <= analysis["fundamental_freq"] <= spec["fundamental_range"][1]):
            issues.append(f"Fundamental frequency {analysis['fundamental_freq']:.1f}Hz outside range {spec['fundamental_range']}")
            passed = False
            
        # Check attack time
        if analysis["attack_time"] > spec["attack_time_max"]:
            issues.append(f"Attack time {analysis['attack_time']:.2f}ms exceeds maximum {spec['attack_time_max']}ms")
            passed = False
            
        # Check harmonics
        strong_harmonics = sum(1 for h in analysis["harmonics"].values() if h > 0.1)
        if strong_harmonics < spec["min_harmonics"]:
            issues.append(f"Only {strong_harmonics} strong harmonics, need at least {spec['min_harmonics']}")
            passed = False
            
        # Check peak level
        if analysis["peak"] < spec["peak_min"]:
            issues.append(f"Peak level {analysis['peak']:.2f} below minimum {spec['peak_min']}")
            passed = False
            
        # Check for artifacts
        if abs(analysis["dc_offset"]) > 0.01:
            issues.append(f"DC offset detected: {analysis['dc_offset']:.4f}")
            passed = False
            
        if analysis["clipping"] > 0.1:
            issues.append(f"Clipping detected: {analysis['clipping']:.2f}% of samples")
            passed = False
            
        return passed, issues
        
    def generate_test_kicks(self) -> Dict[str, AudioSegment]:
        """Generate all kick types for testing"""
        
        kicks = {}
        
        print("\n🥁 Generating Test Kicks...")
        print("=" * 50)
        
        # Test 1: Gabber Kick
        print("\n1. Gabber Kick (190 BPM)")
        gabber = self.generator.create_gabber_kick(190)
        kicks["gabber"] = gabber
        
        # Analyze
        gabber_array = np.array(gabber.get_array_of_samples())
        analysis = self.analyze_audio(gabber_array, "gabber")
        passed, issues = self.validate_kick_quality(gabber_array, "gabber")
        
        print(f"   Fundamental: {analysis['fundamental_freq']:.1f} Hz")
        print(f"   Attack Time: {analysis['attack_time']:.2f} ms")
        print(f"   Peak Level: {analysis['peak']:.3f}")
        print(f"   Harmonics: {len([h for h in analysis['harmonics'].values() if h > 0.1])}")
        
        if passed:
            print("   ✅ Quality Validation: PASSED")
        else:
            print("   ❌ Quality Issues:")
            for issue in issues:
                print(f"      - {issue}")
                
        # Test 2: Industrial Kick
        print("\n2. Industrial Kick (135 BPM)")
        industrial = self.generator.create_industrial_kick(135)
        kicks["industrial"] = industrial
        
        industrial_array = np.array(industrial.get_array_of_samples())
        analysis = self.analyze_audio(industrial_array, "industrial")
        passed, issues = self.validate_kick_quality(industrial_array, "industrial")
        
        print(f"   Fundamental: {analysis['fundamental_freq']:.1f} Hz")
        print(f"   Attack Time: {analysis['attack_time']:.2f} ms")
        print(f"   Peak Level: {analysis['peak']:.3f}")
        print(f"   Sub Energy: {analysis['band_energy']['sub']:.3f}")
        
        if passed:
            print("   ✅ Quality Validation: PASSED")
        else:
            print("   ❌ Quality Issues:")
            for issue in issues:
                print(f"      - {issue}")
                
        # Test 3: Uptempo Kick
        print("\n3. Uptempo/Hardcore Kick (210 BPM)")
        uptempo = self.generator.create_uptempo_kick(210)
        kicks["uptempo"] = uptempo
        
        uptempo_array = np.array(uptempo.get_array_of_samples())
        analysis = self.analyze_audio(uptempo_array, "hardcore")
        passed, issues = self.validate_kick_quality(uptempo_array, "hardcore")
        
        print(f"   Fundamental: {analysis['fundamental_freq']:.1f} Hz")
        print(f"   Attack Time: {analysis['attack_time']:.2f} ms")
        print(f"   Peak Level: {analysis['peak']:.3f}")
        print(f"   High-Mid Energy: {analysis['band_energy']['high_mid']:.3f}")
        
        if passed:
            print("   ✅ Quality Validation: PASSED")
        else:
            print("   ❌ Quality Issues:")
            for issue in issues:
                print(f"      - {issue}")
                
        return kicks
        
    def test_effects_chain(self, base_audio: AudioSegment) -> Dict[str, AudioSegment]:
        """Test effects processing chain"""
        
        print("\n🎛️ Testing Effects Chain...")
        print("=" * 50)
        
        effects_results = {}
        base_array = np.array(base_audio.get_array_of_samples())
        
        # Test distortion levels
        print("\n1. Distortion Levels:")
        for drive in [1.5, 2.5, 4.0]:
            distorted = self.generator.apply_analog_distortion(base_array, drive)
            audio_seg = AudioSegment(
                distorted.astype(np.int16).tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            effects_results[f"distortion_{drive}"] = audio_seg
            
            analysis = self.analyze_audio(distorted, f"distortion_{drive}")
            print(f"   Drive {drive}: Peak={analysis['peak']:.3f}, Harmonics={len(analysis['harmonics'])}")
            
        # Test compression ratios
        print("\n2. Compression Ratios:")
        for ratio in [4.0, 8.0, 12.0]:
            compressed = self.generator.apply_compression(base_array, ratio, threshold=-15.0)
            audio_seg = AudioSegment(
                compressed.astype(np.int16).tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            effects_results[f"compression_{ratio}"] = audio_seg
            
            analysis = self.analyze_audio(compressed, f"compression_{ratio}")
            print(f"   Ratio {ratio}:1: RMS={analysis['rms']:.3f}, Peak={analysis['peak']:.3f}")
            
        # Test combined effects
        print("\n3. Combined Effects Chain:")
        # Typical gabber chain
        processed = base_array
        processed = self.generator.apply_analog_distortion(processed, 2.5)
        processed = self.generator.apply_compression(processed, 8.0, -12.0)
        processed = self.generator.apply_eq_boost(
            AudioSegment(processed.astype(np.int16).tobytes(), 
                        frame_rate=self.sample_rate, sample_width=2, channels=1),
            freq_hz=3000, gain_db=2.0
        )
        
        effects_results["gabber_chain"] = processed
        
        return effects_results
        
    def test_pattern_generation(self):
        """Test pattern/loop generation"""
        
        print("\n🎵 Testing Pattern Generation...")
        print("=" * 50)
        
        patterns = {}
        
        # Test different pattern types
        test_patterns = [
            ("gabber_loop", 190, 2),
            ("industrial_loop", 135, 4),
            ("clean_909", 140, 2)
        ]
        
        for pattern_name, bpm, bars in test_patterns:
            print(f"\n{pattern_name}: {bpm} BPM, {bars} bars")
            
            if "gabber" in pattern_name:
                loop = self.generator.create_gabber_loop(bpm, bars)
            elif "industrial" in pattern_name:
                # Create industrial pattern (you may need to implement this)
                kick = self.generator.create_industrial_kick(bpm)
                loop = self.create_simple_loop(kick, bpm, bars)
            else:
                kick = self.generator.sample_manager.create_909_style_kick()
                loop = self.create_simple_loop(kick, bpm, bars)
                
            patterns[pattern_name] = loop
            
            # Analyze loop
            loop_array = np.array(loop.get_array_of_samples())
            duration = len(loop_array) / self.sample_rate
            expected_duration = (bars * 4 * 60) / bpm  # bars * beats_per_bar * 60 / bpm
            
            print(f"   Duration: {duration:.2f}s (expected: {expected_duration:.2f}s)")
            print(f"   Timing Accuracy: {abs(duration - expected_duration) * 1000:.1f}ms deviation")
            
        return patterns
        
    def create_simple_loop(self, kick: AudioSegment, bpm: int, bars: int) -> AudioSegment:
        """Create a simple kick loop"""
        beats_per_bar = 4
        total_beats = bars * beats_per_bar
        ms_per_beat = 60000 / bpm
        
        loop = AudioSegment.silent(duration=int(total_beats * ms_per_beat))
        
        for beat in range(total_beats):
            position_ms = int(beat * ms_per_beat)
            loop = loop.overlay(kick, position=position_ms)
            
        return loop
        
    def save_test_results(self, audio_dict: Dict[str, AudioSegment], prefix: str = "test"):
        """Save audio files for manual inspection"""
        
        output_dir = Path("audio_tests")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving test audio to {output_dir}/")
        
        for name, audio in audio_dict.items():
            filename = output_dir / f"{prefix}_{name}.wav"
            audio.export(filename, format="wav")
            print(f"   Saved: {filename}")
            
    def play_audio(self, audio: AudioSegment, name: str = "audio"):
        """Play audio through system speakers"""
        
        print(f"\n▶️ Playing: {name}")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio.export(temp_file.name, format="wav")
            
            # Try multiple players for compatibility
            for player_cmd in [['paplay', temp_file.name], 
                             ['aplay', temp_file.name],
                             ['ffplay', '-nodisp', '-autoexit', temp_file.name]]:
                try:
                    subprocess.run(player_cmd, check=True, capture_output=True)
                    break
                except:
                    continue
                    
            os.unlink(temp_file.name)
            
    def create_spectrum_plot(self, audio: AudioSegment, name: str = "audio"):
        """Create frequency spectrum visualization"""
        
        audio_array = np.array(audio.get_array_of_samples())
        
        # Compute spectrum
        freqs, psd = signal.welch(audio_array, self.sample_rate, nperseg=1024)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.semilogy(freqs, psd)
        plt.title(f'Frequency Spectrum - {name}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid(True)
        plt.xlim(0, 5000)
        
        # Waveform
        plt.subplot(2, 1, 2)
        time = np.arange(len(audio_array)) / self.sample_rate
        plt.plot(time[:self.sample_rate//10], audio_array[:self.sample_rate//10])  # First 100ms
        plt.title('Waveform (first 100ms)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.tight_layout()
        
        output_dir = Path("audio_tests")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{name}_spectrum.png", dpi=150)
        print(f"   Saved spectrum plot: {output_dir}/{name}_spectrum.png")
        plt.close()


def run_interactive_tests():
    """Run interactive audio engine tests"""
    
    print("=" * 60)
    print("🎵 HARDCORE AUDIO ENGINE TEST SUITE")
    print("=" * 60)
    
    tester = AudioEngineTestSuite()
    
    # Test menu
    while True:
        print("\n📋 Test Menu:")
        print("1. Generate & Validate All Kicks")
        print("2. Test Effects Chain")
        print("3. Test Pattern Generation")
        print("4. Full Test Suite")
        print("5. Play Last Generated Audio")
        print("6. Generate Spectrum Plots")
        print("Q. Quit")
        
        choice = input("\nSelect test: ").strip().lower()
        
        if choice == '1':
            kicks = tester.generate_test_kicks()
            tester.save_test_results(kicks, "kick")
            
            # Play a sample
            play_choice = input("\nPlay a kick? (gabber/industrial/uptempo/n): ").strip().lower()
            if play_choice in kicks:
                tester.play_audio(kicks[play_choice], play_choice)
                
        elif choice == '2':
            # Generate base kick for effects testing
            print("\nGenerating base kick for effects testing...")
            base_kick = tester.generator.create_gabber_kick(180)
            effects = tester.test_effects_chain(base_kick)
            tester.save_test_results(effects, "effects")
            
        elif choice == '3':
            patterns = tester.test_pattern_generation()
            tester.save_test_results(patterns, "pattern")
            
        elif choice == '4':
            print("\n🔥 Running Full Test Suite...")
            
            # All tests
            kicks = tester.generate_test_kicks()
            base_kick = tester.generator.create_gabber_kick(180)
            effects = tester.test_effects_chain(base_kick)
            patterns = tester.test_pattern_generation()
            
            # Save everything
            tester.save_test_results(kicks, "kick")
            tester.save_test_results(effects, "effects")
            tester.save_test_results(patterns, "pattern")
            
            print("\n✅ Full test suite complete!")
            print(f"   Audio files saved to: audio_tests/")
            
        elif choice == '5':
            # Play last generated audio
            test_files = list(Path("audio_tests").glob("*.wav"))
            if test_files:
                latest = max(test_files, key=os.path.getctime)
                audio = AudioSegment.from_wav(latest)
                tester.play_audio(audio, latest.stem)
            else:
                print("No test audio found. Run a test first.")
                
        elif choice == '6':
            # Generate spectrum plots
            test_files = list(Path("audio_tests").glob("*.wav"))[:5]  # Last 5 files
            for file in test_files:
                audio = AudioSegment.from_wav(file)
                tester.create_spectrum_plot(audio, file.stem)
            print(f"\n✅ Spectrum plots saved to audio_tests/")
            
        elif choice == 'q':
            break
            
        else:
            print("Invalid choice")
            
    print("\n👋 Test suite complete!")


if __name__ == "__main__":
    run_interactive_tests()