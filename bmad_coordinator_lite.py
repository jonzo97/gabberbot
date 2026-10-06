#!/usr/bin/env python3
"""
BMAD Music Coordinator Lite - Real Music Generation without heavy dependencies

A lightweight version of the BMAD coordinator that generates real MIDI files
and demonstrates the complete workflow using only standard library components.
Perfect for immediate testing and demonstration.
"""

import os
import time
import random
import math
import wave
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import existing infrastructure
from cli_shared.generators.acid_bassline import AcidBasslineGenerator, create_hardcore_acid_line
from cli_shared.generators.tuned_kick import TunedKickGenerator, create_frenchcore_kicks
from cli_shared.models.midi_clips import MIDIClip, create_empty_midi_clip


class HardcoreStyle(Enum):
    """Hardcore music styles"""
    ROTTERDAM_GABBER = "rotterdam_gabber"
    FRENCHCORE = "frenchcore"
    UK_HARDCORE = "uk_hardcore"
    BERLIN_INDUSTRIAL = "berlin_industrial"


@dataclass
class BMadTrackConfig:
    """Configuration for BMAD track generation"""
    style: HardcoreStyle = HardcoreStyle.ROTTERDAM_GABBER
    bpm: float = 180.0
    length_bars: float = 16.0
    key: str = "A_minor"
    seed: Optional[int] = None


class BMadMusicAnalyst:
    """@music-analyst (Nexus) - Pattern recognition for hardcore"""
    
    def __init__(self):
        self.name = "Nexus"
        self.role = "@music-analyst"
    
    def analyze_hardcore_patterns(self, config: BMadTrackConfig) -> Dict[str, Any]:
        """Analyze hardcore patterns and generate parameters"""
        
        return {
            "pattern_complexity": self._get_complexity(config.style),
            "rhythmic_patterns": self._get_rhythm_patterns(config.style),
            "energy_curve": self._generate_energy_curve(config.length_bars),
            "breakdown_points": [8.0, 12.0] if config.length_bars >= 16 else []
        }
    
    def _get_complexity(self, style: HardcoreStyle) -> Dict[str, float]:
        """Get complexity factors for style"""
        complexity_map = {
            HardcoreStyle.ROTTERDAM_GABBER: {"kick": 0.7, "bass": 0.6},
            HardcoreStyle.FRENCHCORE: {"kick": 0.9, "bass": 0.8},
            HardcoreStyle.UK_HARDCORE: {"kick": 0.8, "bass": 0.7},
            HardcoreStyle.BERLIN_INDUSTRIAL: {"kick": 0.6, "bass": 0.8}
        }
        return complexity_map.get(style, {"kick": 0.7, "bass": 0.6})
    
    def _get_rhythm_patterns(self, style: HardcoreStyle) -> Dict[str, str]:
        """Get rhythm patterns for style"""
        patterns = {
            HardcoreStyle.ROTTERDAM_GABBER: {
                "kick": "x ~ x ~ x ~ x ~",
                "bass": "~ x ~ x ~ x ~ x"
            },
            HardcoreStyle.FRENCHCORE: {
                "kick": "x x ~ x x ~ x x",
                "bass": "~ ~ x ~ ~ x ~ ~"
            },
            HardcoreStyle.UK_HARDCORE: {
                "kick": "x ~ ~ x x ~ ~ x",
                "bass": "~ x x ~ ~ x x ~"
            },
            HardcoreStyle.BERLIN_INDUSTRIAL: {
                "kick": "x ~ ~ ~ x ~ ~ x",
                "bass": "~ x x ~ ~ x x ~"
            }
        }
        return patterns.get(style, patterns[HardcoreStyle.ROTTERDAM_GABBER])
    
    def _generate_energy_curve(self, length_bars: float) -> List[float]:
        """Generate energy progression"""
        points = int(length_bars / 4)
        if points <= 0:
            return [0.8]
        
        curve = []
        for i in range(points):
            progress = i / max(1, points - 1)
            if progress < 0.3:  # Build
                energy = 0.4 + progress * 2.0
            elif progress < 0.7:  # Peak
                energy = 1.0
            else:  # Breakdown
                energy = 1.0 - (progress - 0.7) * 1.5
            curve.append(max(0.3, min(1.0, energy)))
        
        return curve


class BMadMusicProducer:
    """@music-producer (Raven) - MIDI generation using real generators"""
    
    def __init__(self):
        self.name = "Raven"
        self.role = "@music-producer"
    
    def generate_track_midi(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> Dict[str, MIDIClip]:
        """Generate MIDI using existing generators"""
        
        clips = {}
        
        # Generate kick pattern
        clips["kick"] = self._generate_kick_midi(config, analysis)
        
        # Generate bassline
        clips["bassline"] = self._generate_bassline_midi(config, analysis)
        
        return clips
    
    def _generate_kick_midi(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> MIDIClip:
        """Generate kick MIDI using TunedKickGenerator"""
        
        patterns = analysis["rhythmic_patterns"]
        
        generator = TunedKickGenerator(
            root_note="C1",
            pattern=patterns["kick"],
            tuning="pentatonic"
        )
        
        return generator.generate(config.length_bars, config.bpm)
    
    def _generate_bassline_midi(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> MIDIClip:
        """Generate bassline MIDI using AcidBasslineGenerator"""
        
        if config.style == HardcoreStyle.FRENCHCORE:
            return create_hardcore_acid_line(config.length_bars, config.bpm)
        else:
            generator = AcidBasslineGenerator(scale=config.key.replace("_", " "))
            return generator.generate(config.length_bars, config.bpm)


class BMadSoundDesigner:
    """@sound-designer (Void) - Basic audio synthesis using math"""
    
    def __init__(self):
        self.name = "Void"
        self.role = "@sound-designer"
        self.sample_rate = 44100
    
    def generate_basic_audio(self, midi_clips: Dict[str, MIDIClip], config: BMadTrackConfig) -> Dict[str, List[float]]:
        """Generate basic audio from MIDI using simple synthesis"""
        
        audio_clips = {}
        
        # Generate kick audio
        if "kick" in midi_clips:
            audio_clips["kick"] = self._synthesize_kick(midi_clips["kick"], config)
        
        # Generate bass audio  
        if "bassline" in midi_clips:
            audio_clips["bassline"] = self._synthesize_bass(midi_clips["bassline"], config)
        
        return audio_clips
    
    def _synthesize_kick(self, midi_clip: MIDIClip, config: BMadTrackConfig) -> List[float]:
        """Basic kick synthesis using sine + envelope"""
        
        duration_sec = midi_clip.get_total_beats() * 60.0 / config.bpm
        samples = int(duration_sec * self.sample_rate)
        audio = [0.0] * samples
        
        for note in midi_clip.notes:
            # FIXED: Correct beat-to-sample conversion (removed multiplication by 4)
            # Formula: samples = beats * (60/bpm) * sample_rate
            start_sample = int(note.start_time * (60.0 / config.bpm) * self.sample_rate)
            note_duration = 0.2  # 200ms kick
            note_samples = int(note_duration * self.sample_rate)
            
            # Generate kick wave
            frequency = 60.0  # Base kick frequency
            for i in range(min(note_samples, len(audio) - start_sample)):
                t = i / self.sample_rate
                # Frequency sweep down
                freq = frequency * (1.0 - t * 3)
                # Exponential envelope
                envelope = math.exp(-t * 15)
                # Sine wave with some harmonics
                wave = math.sin(2 * math.pi * freq * t) * 0.8
                wave += math.sin(2 * math.pi * freq * 2 * t) * 0.3
                
                sample_idx = start_sample + i
                if sample_idx < len(audio):
                    audio[sample_idx] += wave * envelope * (note.velocity / 127.0)
        
        return audio
    
    def _synthesize_bass(self, midi_clip: MIDIClip, config: BMadTrackConfig) -> List[float]:
        """Basic bass synthesis using sawtooth wave"""
        
        duration_sec = midi_clip.get_total_beats() * 60.0 / config.bpm
        samples = int(duration_sec * self.sample_rate)
        audio = [0.0] * samples
        
        for note in midi_clip.notes:
            # FIXED: Correct beat-to-sample conversion (removed multiplication by 4)
            start_sample = int(note.start_time * (60.0 / config.bpm) * self.sample_rate)
            note_duration = note.duration * (60.0 / config.bpm)
            note_samples = int(note_duration * self.sample_rate)
            
            # Convert MIDI note to frequency
            frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
            
            for i in range(min(note_samples, len(audio) - start_sample)):
                t = i / self.sample_rate
                # Sawtooth wave
                phase = (frequency * t) % 1.0
                wave = (2 * phase - 1) * 0.5
                
                # Simple envelope
                envelope = max(0, 1.0 - t / note_duration) if note_duration > 0 else 0
                
                sample_idx = start_sample + i
                if sample_idx < len(audio):
                    audio[sample_idx] += wave * envelope * (note.velocity / 127.0) * 0.7
        
        return audio


class BMadMixEngineer:
    """@mix-engineer (Phoenix) - Basic mixing and WAV export"""
    
    def __init__(self):
        self.name = "Phoenix"
        self.role = "@mix-engineer"
        self.sample_rate = 44100
    
    def create_final_mix(self, audio_clips: Dict[str, List[float]]) -> List[float]:
        """Mix audio clips together"""
        
        if not audio_clips:
            return []
        
        # Find longest clip
        max_length = max(len(clip) for clip in audio_clips.values())
        
        # Create mix
        mix = [0.0] * max_length
        
        for name, audio in audio_clips.items():
            # Track levels
            level = 1.0 if name == "kick" else 0.8
            
            for i in range(min(len(audio), len(mix))):
                mix[i] += audio[i] * level
        
        # Simple limiting
        max_val = max(abs(s) for s in mix) if mix else 1.0
        if max_val > 1.0:
            mix = [s / max_val * 0.9 for s in mix]
        
        return mix
    
    def export_wav(self, audio: List[float], filepath: str) -> bool:
        """Export audio as WAV file using standard library"""
        
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                
                # Convert to 16-bit integers
                audio_data = b''
                for sample in audio:
                    # Clamp and convert
                    clamped = max(-1.0, min(1.0, sample))
                    int_sample = int(clamped * 32767)
                    audio_data += struct.pack('<h', int_sample)
                
                wav_file.writeframes(audio_data)
            
            return True
        except Exception as e:
            print(f"Error exporting WAV: {e}")
            return False


class BMadCoordinatorLite:
    """Lightweight BMAD Music Coordinator - Real music generation"""
    
    def __init__(self, output_dir: str = "bmad_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize agents
        self.analyst = BMadMusicAnalyst()
        self.producer = BMadMusicProducer()
        self.sound_designer = BMadSoundDesigner()
        self.mix_engineer = BMadMixEngineer()
        
        print("🎛️ BMAD Coordinator Lite initialized")
        print(f"   Output: {self.output_dir.absolute()}")
    
    def generate_hardcore_track(self, config: BMadTrackConfig) -> str:
        """Generate a complete hardcore track"""
        
        # Create session directory
        session_id = f"bmad_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        print(f"\n🎵 Generating hardcore track: {session_id}")
        print(f"   Style: {config.style.name}")
        print(f"   BPM: {config.bpm}")
        print(f"   Length: {config.length_bars} bars")
        
        try:
            # Set seed for reproducibility
            if config.seed:
                random.seed(config.seed)
            
            # Step 1: Analysis
            print(f"\n🔍 {self.analyst.role} ({self.analyst.name}) analyzing patterns...")
            analysis = self.analyst.analyze_hardcore_patterns(config)
            
            # Step 2: MIDI Generation
            print(f"\n🎹 {self.producer.role} ({self.producer.name}) generating MIDI...")
            midi_clips = self.producer.generate_track_midi(config, analysis)
            
            # Export MIDI files
            midi_files = []
            for name, clip in midi_clips.items():
                midi_path = session_dir / f"{name}.mid"
                if clip.save_midi_file(str(midi_path)):
                    midi_files.append(midi_path.name)
                    print(f"   ✅ {midi_path.name}")
            
            # Step 3: Audio Synthesis
            print(f"\n🎛️ {self.sound_designer.role} ({self.sound_designer.name}) synthesizing audio...")
            audio_clips = self.sound_designer.generate_basic_audio(midi_clips, config)
            
            # Export individual audio tracks
            audio_files = []
            for name, audio in audio_clips.items():
                if audio:
                    audio_path = session_dir / f"{name}_track.wav"
                    if self.mix_engineer.export_wav(audio, str(audio_path)):
                        audio_files.append(audio_path.name)
                        print(f"   ✅ {audio_path.name}")
            
            # Step 4: Final Mix
            print(f"\n🎚️ {self.mix_engineer.role} ({self.mix_engineer.name}) creating final mix...")
            final_mix = self.mix_engineer.create_final_mix(audio_clips)
            
            # Export final track
            final_path = session_dir / f"{session_id}_final.wav"
            if self.mix_engineer.export_wav(final_mix, str(final_path)):
                print(f"   ✅ {final_path.name}")
            
            # Create info file
            info_path = session_dir / "track_info.txt"
            with open(info_path, 'w') as f:
                f.write(f"BMAD Hardcore Track Generation\n")
                f.write(f"==============================\n\n")
                f.write(f"Session: {session_id}\n")
                f.write(f"Style: {config.style.name}\n")
                f.write(f"BPM: {config.bpm}\n")
                f.write(f"Length: {config.length_bars} bars\n")
                f.write(f"Key: {config.key}\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"MIDI Files:\n")
                for file in midi_files:
                    f.write(f"  • {file}\n")
                f.write(f"\nAudio Files:\n")
                for file in audio_files:
                    f.write(f"  • {file}\n")
                f.write(f"  • {final_path.name}\n\n")
                f.write(f"BMAD Agents:\n")
                f.write(f"  • {self.analyst.role} ({self.analyst.name})\n")
                f.write(f"  • {self.producer.role} ({self.producer.name})\n") 
                f.write(f"  • {self.sound_designer.role} ({self.sound_designer.name})\n")
                f.write(f"  • {self.mix_engineer.role} ({self.mix_engineer.name})\n")
            
            print(f"\n🎉 Track generation complete!")
            print(f"   Session: {session_id}")
            print(f"   Output: {session_dir.name}")
            
            return session_id
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise
    
    def generate_hardcore_collection(self, count: int = 3) -> List[str]:
        """Generate multiple hardcore tracks"""
        
        print(f"\n🏭 BMAD Collection Generation - {count} tracks")
        print("=" * 50)
        
        sessions = []
        styles = [HardcoreStyle.ROTTERDAM_GABBER, HardcoreStyle.FRENCHCORE, HardcoreStyle.UK_HARDCORE]
        bpms = [170, 180, 190, 200]
        lengths = [16, 24, 32]
        
        for i in range(count):
            print(f"\n📀 Track {i + 1}/{count}")
            
            config = BMadTrackConfig(
                style=random.choice(styles),
                bpm=random.choice(bpms),
                length_bars=random.choice(lengths),
                key=random.choice(["A_minor", "E_minor"]),
                seed=1000 + i
            )
            
            try:
                session_id = self.generate_hardcore_track(config)
                sessions.append(session_id)
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        print(f"\n🎊 Collection complete! Generated {len(sessions)} tracks")
        return sessions


def demo_bmad_lite():
    """Demonstrate the BMAD lite coordinator"""
    
    print("🎛️ BMAD Music Coordinator Lite - Demo")
    print("=" * 45)
    
    coordinator = BMadCoordinatorLite()
    
    # Generate a gabber track
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=180.0,
        length_bars=16.0,
        key="A_minor",
        seed=42
    )
    
    session_id = coordinator.generate_hardcore_track(config)
    
    print(f"\n✨ Demo complete!")
    print(f"   Check the generated files in: bmad_output/{session_id}/")
    print(f"   Play {session_id}_final.wav to hear your hardcore track!")


def quick_gabber_track():
    """Quick function to generate a gabber track"""
    coordinator = BMadCoordinatorLite()
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=180.0,
        length_bars=24.0,
        seed=random.randint(1000, 9999)
    )
    return coordinator.generate_hardcore_track(config)


def quick_frenchcore_track():
    """Quick function to generate a frenchcore track"""
    coordinator = BMadCoordinatorLite()
    config = BMadTrackConfig(
        style=HardcoreStyle.FRENCHCORE,
        bpm=200.0,
        length_bars=20.0,
        seed=random.randint(1000, 9999)
    )
    return coordinator.generate_hardcore_track(config)


if __name__ == "__main__":
    demo_bmad_lite()