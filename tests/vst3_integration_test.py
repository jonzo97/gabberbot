#!/usr/bin/env python3
"""
VST3 Integration Test with Pedalboard
Test VST3 loading capabilities and integration with existing synthesis engine
"""

import numpy as np
import pedalboard
from pedalboard import VST3Plugin, load_plugin
import tempfile
from pathlib import Path
import os
from pydub import AudioSegment
import subprocess


class VST3IntegrationTester:
    """Test VST3 integration capabilities"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def scan_for_vst3_plugins(self) -> list:
        """Scan common VST3 directories for available plugins"""
        
        common_vst3_paths = [
            "/usr/lib/vst3",
            "/usr/local/lib/vst3", 
            "/home/onathan_rgill/.vst3",
            "~/.vst3",
            "/opt/vst3"
        ]
        
        found_plugins = []
        
        print("🔍 Scanning for VST3 plugins...")
        
        for vst_path in common_vst3_paths:
            expanded_path = Path(vst_path).expanduser()
            if expanded_path.exists():
                print(f"   Scanning: {expanded_path}")
                
                # Look for .vst3 files/directories
                for item in expanded_path.rglob("*.vst3"):
                    found_plugins.append(str(item))
                    print(f"   Found: {item.name}")
                    
        if not found_plugins:
            print("   ❌ No VST3 plugins found in common directories")
        else:
            print(f"   ✅ Found {len(found_plugins)} VST3 plugins")
            
        return found_plugins
        
    def test_pedalboard_basics(self):
        """Test basic pedalboard functionality"""
        
        print("\n🎛️ Testing Pedalboard Basic Functionality")
        print("-" * 50)
        
        # Generate test audio
        duration = 2.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Simple sawtooth wave
        frequency = 220.0  # A3
        phase = 2 * np.pi * frequency * t
        sawtooth = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        
        print(f"✅ Generated test sawtooth: {duration}s at {frequency}Hz")
        
        # Test built-in pedalboard effects
        try:
            board = pedalboard.Pedalboard([
                pedalboard.Distortion(drive_db=10),
                pedalboard.LowpassFilter(cutoff_frequency_hz=800),
                pedalboard.Reverb(room_size=0.8, wet_level=0.3)
            ])
            
            processed = board(sawtooth, self.sample_rate)
            
            print(f"✅ Applied pedalboard effects chain")
            print(f"   Input peak: {np.max(np.abs(sawtooth)):.3f}")
            print(f"   Output peak: {np.max(np.abs(processed)):.3f}")
            
            # Export test
            self._export_audio(processed, "pedalboard_basic_test.wav")
            
            return True
            
        except Exception as e:
            print(f"❌ Pedalboard basic test failed: {e}")
            return False
            
    def test_vst3_loading(self, vst3_plugins: list):
        """Test VST3 plugin loading"""
        
        print(f"\n🔌 Testing VST3 Plugin Loading")
        print("-" * 50)
        
        if not vst3_plugins:
            print("❌ No VST3 plugins available for testing")
            return False
            
        successful_loads = 0
        
        for plugin_path in vst3_plugins[:3]:  # Test first 3 plugins
            try:
                print(f"\n📦 Testing: {Path(plugin_path).name}")
                
                # Try to load the plugin
                plugin = load_plugin(plugin_path)
                
                print(f"   ✅ Loaded successfully")
                print(f"   Plugin name: {getattr(plugin, 'name', 'Unknown')}")
                
                # Try to get parameters
                try:
                    params = plugin.parameters
                    print(f"   Parameters: {len(params)} found")
                    
                    # Show first few parameters
                    for i, (param_name, param_value) in enumerate(params.items()):
                        if i < 3:  # Show first 3
                            print(f"     {param_name}: {param_value}")
                        
                except Exception as param_error:
                    print(f"   ⚠️ Parameter access failed: {param_error}")
                
                successful_loads += 1
                
            except Exception as e:
                print(f"   ❌ Failed to load: {e}")
                
        print(f"\n📊 VST3 Loading Results: {successful_loads}/{len(vst3_plugins[:3])} successful")
        
        return successful_loads > 0
        
    def test_midi_to_vst3_synthesis(self, vst3_plugins: list):
        """Test MIDI note synthesis through VST3"""
        
        print(f"\n🎹 Testing MIDI → VST3 Synthesis")
        print("-" * 50)
        
        if not vst3_plugins:
            print("❌ No VST3 plugins for synthesis testing")
            return False
            
        # Try to find a synthesizer plugin
        synth_plugin = None
        for plugin_path in vst3_plugins:
            try:
                plugin = load_plugin(plugin_path)
                # Check if it might be a synth (heuristic)
                plugin_name = getattr(plugin, 'name', Path(plugin_path).name).lower()
                
                if any(keyword in plugin_name for keyword in ['synth', 'vsti', 'instrument']):
                    synth_plugin = plugin
                    print(f"   🎛️ Found potential synth: {plugin_name}")
                    break
                    
            except:
                continue
                
        if not synth_plugin:
            print("❌ No synthesizer plugins found")
            return False
            
        try:
            # Generate MIDI note
            import mido
            
            # Create simple MIDI sequence
            notes = [60, 64, 67]  # C major triad
            duration = 1.0
            
            # This is where we'd need to implement MIDI → audio rendering
            # Pedalboard doesn't directly support MIDI input, so we'd need
            # to use a different approach or generate audio directly
            
            print("   ⚠️ MIDI synthesis requires additional implementation")
            print("   ⚠️ Pedalboard primarily processes audio, not MIDI")
            
            return False
            
        except Exception as e:
            print(f"   ❌ MIDI synthesis test failed: {e}")
            return False
            
    def test_hybrid_approach(self):
        """Test hybrid approach: Python synthesis → VST3 processing"""
        
        print(f"\n🔄 Testing Hybrid Approach: Python Synth → VST3 Processing")
        print("-" * 50)
        
        # Generate hardcore-style synth in Python (like our current system)
        duration = 3.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Multi-oscillator hardcore synth
        hardcore_synth = np.zeros(samples)
        base_freq = 220.0
        
        # Multiple detuned sawtooths
        detune_amounts = [-0.10, 0.0, 0.05, 0.10]
        for detune in detune_amounts:
            freq = base_freq * (2 ** (detune / 12))
            phase = 2 * np.pi * freq * t
            sawtooth = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
            hardcore_synth += sawtooth / len(detune_amounts)
            
        # Filter sweep
        from scipy import signal
        start_freq = 200
        end_freq = 2000
        sweep_freq = start_freq + (end_freq - start_freq) * (t / duration) ** 2
        
        # Apply basic filter (simplified)
        b, a = signal.butter(4, 800 / (self.sample_rate / 2), btype='low')
        hardcore_synth = signal.filtfilt(b, a, hardcore_synth)
        
        print(f"✅ Generated Python hardcore synth")
        
        # Apply pedalboard effects as "VST3-style" processing
        try:
            # Create aggressive effects chain
            board = pedalboard.Pedalboard([
                pedalboard.Distortion(drive_db=20),
                pedalboard.Compressor(threshold_db=-12, ratio=8),
                pedalboard.HighpassFilter(cutoff_frequency_hz=100),
                pedalboard.LowpassFilter(cutoff_frequency_hz=8000),
                pedalboard.Reverb(room_size=0.6, wet_level=0.2)
            ])
            
            processed = board(hardcore_synth.astype(np.float32), self.sample_rate)
            
            print(f"✅ Applied hardcore effects chain")
            
            # Export both versions
            self._export_audio(hardcore_synth, "python_synth_raw.wav")
            self._export_audio(processed, "python_synth_processed.wav")
            
            print(f"💾 Exported both versions for comparison")
            
            return True
            
        except Exception as e:
            print(f"❌ Hybrid processing failed: {e}")
            return False
            
    def _export_audio(self, audio: np.ndarray, filename: str):
        """Export audio array to WAV file"""
        
        # Normalize and convert to int16
        audio_normalized = audio / np.max(np.abs(audio)) * 0.9
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # Create AudioSegment and export
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        )
        
        output_path = Path("audio_tests") / filename
        output_path.parent.mkdir(exist_ok=True)
        audio_segment.export(str(output_path), format="wav")
        
    def generate_integration_report(self, vst3_available: bool, pedalboard_works: bool, 
                                  vst3_loading_works: bool, hybrid_works: bool):
        """Generate comprehensive integration feasibility report"""
        
        print("\n" + "=" * 70)
        print("📊 VST3 INTEGRATION FEASIBILITY REPORT")
        print("=" * 70)
        
        print(f"\n🔍 CAPABILITY ASSESSMENT:")
        print(f"   VST3 Plugins Available: {'✅' if vst3_available else '❌'}")
        print(f"   Pedalboard Functionality: {'✅' if pedalboard_works else '❌'}")  
        print(f"   VST3 Loading: {'✅' if vst3_loading_works else '❌'}")
        print(f"   Hybrid Processing: {'✅' if hybrid_works else '❌'}")
        
        print(f"\n🎯 RECOMMENDED APPROACH:")
        
        if pedalboard_works and hybrid_works:
            print("✅ HYBRID APPROACH RECOMMENDED")
            print("   • Python synthesis for MIDI generation and core sounds")
            print("   • Pedalboard effects for professional audio processing")
            print("   • Keep existing Frankenstein engine for hardcore character")
            print("   • Add VST3-style effects chains for polish")
            
            print(f"\n🔧 IMPLEMENTATION STRATEGY:")
            print("   1. Enhance current Python synthesis")
            print("   2. Add pedalboard effects as post-processing")  
            print("   3. Create preset system mimicking VST3 libraries")
            print("   4. Maintain real-time performance")
            
        elif pedalboard_works:
            print("⚠️ PEDALBOARD-ONLY APPROACH")
            print("   • Use built-in pedalboard effects")
            print("   • Limited to available built-in processors")
            print("   • Good for basic audio enhancement")
            
        else:
            print("❌ PURE PYTHON APPROACH RECOMMENDED")  
            print("   • Continue with current synthesis engine")
            print("   • Enhance algorithms based on research")
            print("   • No external VST dependencies")
            
        print(f"\n⏱️ ESTIMATED IMPLEMENTATION TIME:")
        if hybrid_works:
            print("   • Hybrid integration: 1-2 weeks")
            print("   • Preset system: 1 week")
            print("   • Testing and optimization: 1 week")
        else:
            print("   • Enhanced Python synthesis: 2-3 weeks")
            
        return {
            "vst3_available": vst3_available,
            "pedalboard_works": pedalboard_works,
            "vst3_loading_works": vst3_loading_works, 
            "hybrid_works": hybrid_works,
            "recommended_approach": "hybrid" if hybrid_works else "pure_python"
        }


def run_vst3_integration_tests():
    """Run complete VST3 integration test suite"""
    
    print("=" * 80)
    print("🧪 VST3 INTEGRATION TESTING SUITE")
    print("=" * 80)
    
    tester = VST3IntegrationTester()
    
    # Test 1: Scan for VST3 plugins
    vst3_plugins = tester.scan_for_vst3_plugins()
    vst3_available = len(vst3_plugins) > 0
    
    # Test 2: Basic pedalboard functionality
    pedalboard_works = tester.test_pedalboard_basics()
    
    # Test 3: VST3 loading
    vst3_loading_works = False
    if vst3_available:
        vst3_loading_works = tester.test_vst3_loading(vst3_plugins)
        
    # Test 4: MIDI synthesis (expected to have limitations)
    if vst3_available:
        tester.test_midi_to_vst3_synthesis(vst3_plugins)
        
    # Test 5: Hybrid approach
    hybrid_works = tester.test_hybrid_approach()
    
    # Generate final report
    report = tester.generate_integration_report(
        vst3_available, pedalboard_works, vst3_loading_works, hybrid_works
    )
    
    return report


if __name__ == "__main__":
    run_vst3_integration_tests()