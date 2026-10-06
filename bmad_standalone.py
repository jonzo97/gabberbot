#!/usr/bin/env python3
"""
BMAD Standalone Music Coordinator - Real Hardcore Music Generation

A completely self-contained BMAD music coordinator that generates real MIDI and audio files
without any external dependencies. Uses only Python standard library.

This demonstrates the REAL BMAD workflow:
- @music-analyst: Analyzes hardcore patterns 
- @music-producer: Generates actual MIDI files
- @sound-designer: Creates real audio synthesis
- @mix-engineer: Produces final WAV files

Generates authentic hardcore tracks that can be played in any audio player!
"""

import os
import time
import random
import math
import wave
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class HardcoreStyle(Enum):
    """Hardcore music styles with specific characteristics"""
    ROTTERDAM_GABBER = "rotterdam_gabber"
    FRENCHCORE = "frenchcore" 
    UK_HARDCORE = "uk_hardcore"
    BERLIN_INDUSTRIAL = "berlin_industrial"
    SPEEDCORE = "speedcore"


@dataclass
class BMadTrackConfig:
    """Configuration for BMAD track generation"""
    style: HardcoreStyle = HardcoreStyle.ROTTERDAM_GABBER
    bpm: float = 180.0
    length_bars: float = 16.0
    key: str = "A_minor"
    seed: Optional[int] = None


@dataclass 
class MIDINote:
    """Simple MIDI note representation"""
    pitch: int
    velocity: int
    start_time: float  # In beats
    duration: float    # In beats
    channel: int = 0


class SimpleMIDIExporter:
    """Basic MIDI file exporter using standard library"""
    
    @staticmethod
    def export_midi(notes: List[MIDINote], filepath: str, bpm: float = 120) -> bool:
        """Export MIDI notes to file"""
        
        try:
            with open(filepath, 'wb') as f:
                # Basic MIDI header
                f.write(b'MThd')  # Header chunk
                f.write(struct.pack('>I', 6))  # Header length
                f.write(struct.pack('>H', 0))  # Format 0
                f.write(struct.pack('>H', 1))  # 1 track
                f.write(struct.pack('>H', 480))  # Ticks per quarter note
                
                # Track header
                f.write(b'MTrk')
                
                # Calculate track data
                track_data = b''
                
                # Tempo event (120 BPM = 500000 microseconds per beat)
                tempo = int(60000000 / bpm)
                track_data += SimpleMIDIExporter._variable_length(0)  # Delta time
                track_data += b'\xff\x51\x03'  # Set tempo meta event
                track_data += struct.pack('>I', tempo)[1:]  # 3 bytes of tempo
                
                # Convert notes to MIDI events
                events = []
                for note in notes:
                    start_ticks = int(note.start_time * 480)  # Convert beats to ticks
                    end_ticks = int((note.start_time + note.duration) * 480)
                    
                    events.append((start_ticks, 'note_on', note))
                    events.append((end_ticks, 'note_off', note))
                
                # Sort events by time
                events.sort(key=lambda x: x[0])
                
                # Convert to MIDI data
                current_time = 0
                for event_time, event_type, note in events:
                    delta = event_time - current_time
                    current_time = event_time
                    
                    track_data += SimpleMIDIExporter._variable_length(delta)
                    
                    if event_type == 'note_on':
                        track_data += bytes([0x90 | note.channel, note.pitch, note.velocity])
                    else:  # note_off
                        track_data += bytes([0x80 | note.channel, note.pitch, 0])
                
                # End of track
                track_data += SimpleMIDIExporter._variable_length(0)
                track_data += b'\xff\x2f\x00'  # End of track meta event
                
                # Write track length and data
                f.write(struct.pack('>I', len(track_data)))
                f.write(track_data)
            
            return True
        except Exception as e:
            print(f"MIDI export error: {e}")
            return False
    
    @staticmethod
    def _variable_length(value: int) -> bytes:
        """Convert integer to MIDI variable length format"""
        result = b''
        while value > 0x7f:
            result = bytes([0x80 | (value & 0x7f)]) + result
            value >>= 7
        result = bytes([value & 0x7f]) + result
        return result if result else b'\x00'


class BMadMusicAnalyst:
    """@music-analyst (Nexus) - Pattern recognition savant"""
    
    def __init__(self):
        self.name = "Nexus"
        self.role = "@music-analyst"
        self.specialty = "Pattern recognition and hardcore evolution"
    
    def analyze_hardcore_patterns(self, config: BMadTrackConfig) -> Dict[str, Any]:
        """Analyze hardcore patterns and create evolution parameters"""
        
        print(f"   🧠 Analyzing {config.style.name} patterns at {config.bpm} BPM...")
        
        analysis = {
            "pattern_complexity": self._analyze_complexity(config.style),
            "rhythmic_patterns": self._get_rhythm_patterns(config.style), 
            "scale_degrees": self._get_scale_degrees(config.key),
            "energy_curve": self._generate_energy_curve(config.length_bars),
            "breakdown_points": self._identify_breakdowns(config.length_bars),
            "frequency_ranges": self._get_frequency_ranges(config.style)
        }
        
        print(f"      • Pattern complexity: {analysis['pattern_complexity']['overall']:.1f}")
        print(f"      • Energy curve: {len(analysis['energy_curve'])} sections")
        print(f"      • Breakdown points: {len(analysis['breakdown_points'])}")
        
        return analysis
    
    def _analyze_complexity(self, style: HardcoreStyle) -> Dict[str, float]:
        """Analyze pattern complexity for style"""
        complexity_map = {
            HardcoreStyle.ROTTERDAM_GABBER: {"kick": 0.7, "bass": 0.6, "overall": 0.65},
            HardcoreStyle.FRENCHCORE: {"kick": 0.9, "bass": 0.8, "overall": 0.85},
            HardcoreStyle.UK_HARDCORE: {"kick": 0.8, "bass": 0.7, "overall": 0.75},
            HardcoreStyle.BERLIN_INDUSTRIAL: {"kick": 0.6, "bass": 0.8, "overall": 0.7},
            HardcoreStyle.SPEEDCORE: {"kick": 1.0, "bass": 0.9, "overall": 0.95}
        }
        return complexity_map.get(style, {"kick": 0.7, "bass": 0.6, "overall": 0.65})
    
    def _get_rhythm_patterns(self, style: HardcoreStyle) -> Dict[str, List[bool]]:
        """Get rhythm patterns as boolean arrays"""
        patterns = {
            HardcoreStyle.ROTTERDAM_GABBER: {
                "kick": [True, False, True, False, True, False, True, False],
                "bass": [False, True, False, True, False, True, False, True]
            },
            HardcoreStyle.FRENCHCORE: {
                "kick": [True, True, False, True, True, False, True, True],
                "bass": [False, False, True, False, False, True, False, False]
            },
            HardcoreStyle.UK_HARDCORE: {
                "kick": [True, False, False, True, True, False, False, True],
                "bass": [False, True, True, False, False, True, True, False]
            },
            HardcoreStyle.BERLIN_INDUSTRIAL: {
                "kick": [True, False, False, False, True, False, False, True],
                "bass": [False, True, True, False, False, True, True, False]
            },
            HardcoreStyle.SPEEDCORE: {
                "kick": [True, True, True, False, True, True, True, False],
                "bass": [False, False, False, True, False, False, False, True]
            }
        }
        return patterns.get(style, patterns[HardcoreStyle.ROTTERDAM_GABBER])
    
    def _get_scale_degrees(self, key: str) -> List[int]:
        """Get scale degrees for key (MIDI note offsets from root)"""
        scales = {
            "A_minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
            "E_minor": [0, 2, 3, 5, 7, 8, 10],
            "C_minor": [0, 2, 3, 5, 7, 8, 10]
        }
        return scales.get(key, scales["A_minor"])
    
    def _generate_energy_curve(self, length_bars: float) -> List[float]:
        """Generate energy progression curve"""
        sections = max(1, int(length_bars / 4))
        curve = []
        
        for i in range(sections):
            progress = i / max(1, sections - 1)
            
            if progress < 0.25:  # Intro
                energy = 0.4 + progress * 1.6
            elif progress < 0.6:  # Build/Peak
                energy = 0.8 + (progress - 0.25) * 0.6
            elif progress < 0.8:  # Drop
                energy = 1.0
            else:  # Breakdown/Outro
                energy = 1.0 - (progress - 0.8) * 2.0
            
            curve.append(max(0.2, min(1.0, energy)))
        
        return curve
    
    def _identify_breakdowns(self, length_bars: float) -> List[float]:
        """Identify breakdown positions"""
        breakdowns = []
        if length_bars >= 12:
            breakdowns.append(8.0)
        if length_bars >= 20:
            breakdowns.append(16.0)
        if length_bars >= 28:
            breakdowns.append(24.0)
        return breakdowns
    
    def _get_frequency_ranges(self, style: HardcoreStyle) -> Dict[str, Tuple[float, float]]:
        """Get frequency ranges for synthesis"""
        ranges = {
            HardcoreStyle.ROTTERDAM_GABBER: {"kick": (50, 80), "bass": (80, 200)},
            HardcoreStyle.FRENCHCORE: {"kick": (60, 90), "bass": (90, 250)},
            HardcoreStyle.UK_HARDCORE: {"kick": (55, 85), "bass": (85, 220)},
            HardcoreStyle.BERLIN_INDUSTRIAL: {"kick": (45, 75), "bass": (75, 180)},
            HardcoreStyle.SPEEDCORE: {"kick": (70, 100), "bass": (100, 300)}
        }
        return ranges.get(style, ranges[HardcoreStyle.ROTTERDAM_GABBER])


class BMadMusicProducer:
    """@music-producer (Raven) - Creative visionary with relentless drive"""
    
    def __init__(self):
        self.name = "Raven"
        self.role = "@music-producer"
        self.specialty = "Track composition and pattern generation"
    
    def generate_hardcore_midi(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> Dict[str, List[MIDINote]]:
        """Generate MIDI patterns for hardcore track"""
        
        print(f"   🎹 Generating MIDI patterns for {config.style.name}...")
        
        midi_tracks = {}
        
        # Generate kick pattern
        kick_notes = self._generate_kick_pattern(config, analysis)
        midi_tracks["kick"] = kick_notes
        print(f"      • Kick pattern: {len(kick_notes)} notes")
        
        # Generate bassline pattern
        bass_notes = self._generate_bassline_pattern(config, analysis)
        midi_tracks["bassline"] = bass_notes
        print(f"      • Bassline: {len(bass_notes)} notes")
        
        # Generate percussion
        perc_notes = self._generate_percussion_pattern(config, analysis)
        midi_tracks["percussion"] = perc_notes
        print(f"      • Percussion: {len(perc_notes)} notes")
        
        return midi_tracks
    
    def _generate_kick_pattern(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[MIDINote]:
        """Generate kick drum pattern"""
        
        pattern = analysis["rhythmic_patterns"]["kick"]
        notes = []
        
        # Calculate timing
        beats_per_bar = 4.0
        total_beats = config.length_bars * beats_per_bar
        step_size = 0.25  # 16th notes
        steps_per_pattern = len(pattern)
        
        current_beat = 0.0
        step_index = 0
        
        while current_beat < total_beats:
            pattern_step = step_index % steps_per_pattern
            
            if pattern[pattern_step]:
                # Add kick note
                velocity = 120 + random.randint(-10, 7)  # High velocity with variation
                notes.append(MIDINote(
                    pitch=36,  # C1 kick
                    velocity=velocity,
                    start_time=current_beat,
                    duration=0.2,  # Short kick
                    channel=9  # Drum channel
                ))
            
            current_beat += step_size
            step_index += 1
        
        # Add some variations based on complexity
        complexity = analysis["pattern_complexity"]["kick"]
        if complexity > 0.8:
            self._add_kick_variations(notes, config, analysis)
        
        return notes
    
    def _generate_bassline_pattern(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[MIDINote]:
        """Generate acid bassline pattern"""
        
        pattern = analysis["rhythmic_patterns"]["bass"]
        scale_degrees = analysis["scale_degrees"]
        notes = []
        
        # Base note for key
        base_notes = {
            "A_minor": 45,  # A1
            "E_minor": 52,  # E2
            "C_minor": 48   # C2
        }
        root_note = base_notes.get(config.key, 45)
        
        # Calculate timing
        beats_per_bar = 4.0
        total_beats = config.length_bars * beats_per_bar
        step_size = 0.25  # 16th notes
        steps_per_pattern = len(pattern)
        
        current_beat = 0.0
        step_index = 0
        current_note = 0  # Scale degree index
        
        while current_beat < total_beats:
            pattern_step = step_index % steps_per_pattern
            
            if pattern[pattern_step]:
                # Choose note from scale
                scale_degree = scale_degrees[current_note % len(scale_degrees)]
                pitch = root_note + scale_degree
                
                # Add some octave variation
                if random.random() < 0.3:
                    pitch += 12 * random.choice([-1, 1])
                
                velocity = 90 + random.randint(-15, 15)
                duration = random.choice([0.2, 0.25, 0.3])  # Vary note length
                
                notes.append(MIDINote(
                    pitch=pitch,
                    velocity=velocity,
                    start_time=current_beat,
                    duration=duration,
                    channel=0
                ))
                
                # Move to next scale note with some pattern
                if random.random() < 0.7:  # Step movement
                    current_note += random.choice([-1, 1])
                else:  # Jump
                    current_note += random.choice([-3, -2, 2, 3])
                
                current_note = max(0, current_note) % len(scale_degrees)
            
            current_beat += step_size
            step_index += 1
        
        return notes
    
    def _generate_percussion_pattern(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[MIDINote]:
        """Generate percussion elements"""
        
        notes = []
        
        # Hi-hat pattern
        total_beats = config.length_bars * 4.0
        current_beat = 0.0
        
        while current_beat < total_beats:
            # Off-beat hi-hats
            if (current_beat % 1.0) == 0.5:  # On the off-beat
                if random.random() < 0.8:  # Not every off-beat
                    notes.append(MIDINote(
                        pitch=42,  # Closed hi-hat
                        velocity=60 + random.randint(-10, 10),
                        start_time=current_beat,
                        duration=0.1,
                        channel=9
                    ))
            
            current_beat += 0.25
        
        return notes
    
    def _add_kick_variations(self, notes: List[MIDINote], config: BMadTrackConfig, analysis: Dict[str, Any]):
        """Add kick variations for complex styles"""
        
        # Add some ghost kicks
        breakdown_points = analysis["breakdown_points"]
        
        for breakdown in breakdown_points:
            if breakdown < config.length_bars * 4:
                # Add a kick variation before breakdown
                notes.append(MIDINote(
                    pitch=35,  # Different kick sound
                    velocity=100,
                    start_time=breakdown - 0.25,
                    duration=0.15,
                    channel=9
                ))


class BMadSoundDesigner:
    """@sound-designer (Void) - Sonic alchemist obsessed with spectral manipulation"""
    
    def __init__(self):
        self.name = "Void"
        self.role = "@sound-designer"
        self.specialty = "Synthesis mastery and spectral processing"
        self.sample_rate = 44100
    
    def synthesize_hardcore_audio(self, midi_tracks: Dict[str, List[MIDINote]], 
                                config: BMadTrackConfig, analysis: Dict[str, Any]) -> Dict[str, List[float]]:
        """Synthesize audio from MIDI using hardcore-specific synthesis"""
        
        print(f"   🎛️ Synthesizing {config.style.name} audio...")
        
        audio_tracks = {}
        
        # Synthesize kick drums
        kick_audio = self._synthesize_kicks(midi_tracks.get("kick", []), config, analysis)
        audio_tracks["kick"] = kick_audio
        print(f"      • Kick synthesis: {len(kick_audio)} samples")
        
        # Synthesize bassline
        bass_audio = self._synthesize_bassline(midi_tracks.get("bassline", []), config, analysis)
        audio_tracks["bassline"] = bass_audio
        print(f"      • Bass synthesis: {len(bass_audio)} samples")
        
        # Synthesize percussion
        perc_audio = self._synthesize_percussion(midi_tracks.get("percussion", []), config, analysis)
        audio_tracks["percussion"] = perc_audio
        print(f"      • Percussion synthesis: {len(perc_audio)} samples")
        
        return audio_tracks
    
    def _synthesize_kicks(self, notes: List[MIDINote], config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[float]:
        """Synthesize hardcore kick drums"""
        
        # Calculate total duration
        if not notes:
            return []
        
        total_beats = config.length_bars * 4.0
        duration_sec = total_beats * 60.0 / config.bpm
        samples = int(duration_sec * self.sample_rate)
        audio = [0.0] * samples
        
        freq_range = analysis["frequency_ranges"]["kick"]
        base_freq = (freq_range[0] + freq_range[1]) / 2
        
        for note in notes:
            # FIXED: Correct beat-to-sample conversion (removed division by 4)
            # Formula: samples = beats * (60/bpm) * sample_rate
            start_sample = int(note.start_time * (60.0 / config.bpm) * self.sample_rate)

            # Generate kick sample based on style
            kick_sample = self._generate_kick_sample(base_freq, note.velocity, config.style)
            
            # Add to output
            end_sample = min(start_sample + len(kick_sample), len(audio))
            for i in range(start_sample, end_sample):
                if i < len(audio):
                    audio[i] += kick_sample[i - start_sample]
        
        return audio
    
    def _generate_kick_sample(self, frequency: float, velocity: int, style: HardcoreStyle) -> List[float]:
        """Generate a single kick sample"""
        
        duration = 0.3  # 300ms kick
        samples = int(duration * self.sample_rate)
        kick = [0.0] * samples
        
        # Style-specific synthesis
        if style == HardcoreStyle.ROTTERDAM_GABBER:
            # Classic 909-style kick with frequency sweep
            for i in range(samples):
                t = i / self.sample_rate
                # Frequency sweep down
                freq = frequency * (1.0 - t * 2.5)
                # Exponential envelope
                envelope = math.exp(-t * 12)
                # Sine wave with harmonics
                wave = math.sin(2 * math.pi * freq * t) * 0.8
                wave += math.sin(2 * math.pi * freq * 2 * t) * 0.3
                # Add some punch
                wave += math.sin(2 * math.pi * freq * 4 * t) * 0.1
                
                kick[i] = wave * envelope * (velocity / 127.0)
        
        elif style == HardcoreStyle.FRENCHCORE:
            # More aggressive kick with distortion
            for i in range(samples):
                t = i / self.sample_rate
                freq = frequency * (1.0 - t * 3.0)
                envelope = math.exp(-t * 15)
                
                # Multiple oscillators for thickness
                wave = math.sin(2 * math.pi * freq * t) * 0.7
                wave += math.sin(2 * math.pi * freq * 1.5 * t) * 0.4
                wave += math.sin(2 * math.pi * freq * 3 * t) * 0.2
                
                # Distortion
                wave = math.tanh(wave * 2.0) * 0.8
                
                kick[i] = wave * envelope * (velocity / 127.0)
        
        elif style == HardcoreStyle.SPEEDCORE:
            # Extreme kick with more aggressive envelope
            for i in range(min(samples, int(0.15 * self.sample_rate))):  # Shorter duration
                t = i / self.sample_rate
                freq = frequency * (1.0 - t * 4.0)
                envelope = math.exp(-t * 25)
                
                # Very aggressive harmonics
                wave = math.sin(2 * math.pi * freq * t) * 0.6
                wave += math.sin(2 * math.pi * freq * 2 * t) * 0.4
                wave += math.sin(2 * math.pi * freq * 3 * t) * 0.3
                wave += math.sin(2 * math.pi * freq * 5 * t) * 0.2
                
                # Heavy distortion
                wave = math.tanh(wave * 3.0) * 0.7
                
                kick[i] = wave * envelope * (velocity / 127.0)
        
        else:
            # Default kick synthesis
            for i in range(samples):
                t = i / self.sample_rate
                freq = frequency * math.exp(-t * 3)
                envelope = math.exp(-t * 10)
                wave = math.sin(2 * math.pi * freq * t)
                kick[i] = wave * envelope * (velocity / 127.0) * 0.8
        
        return kick
    
    def _synthesize_bassline(self, notes: List[MIDINote], config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[float]:
        """Synthesize acid bassline"""
        
        if not notes:
            return []
        
        total_beats = config.length_bars * 4.0
        duration_sec = total_beats * 60.0 / config.bpm
        samples = int(duration_sec * self.sample_rate)
        audio = [0.0] * samples
        
        for note in notes:
            # FIXED: Correct beat-to-sample conversion (removed division by 4)
            start_sample = int(note.start_time * (60.0 / config.bpm) * self.sample_rate)
            note_duration = note.duration * (60.0 / config.bpm)
            note_samples = int(note_duration * self.sample_rate)
            
            # Convert MIDI note to frequency
            frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
            
            # Generate bass sample
            bass_sample = self._generate_bass_sample(frequency, note_duration, note.velocity, config.style)
            
            # Add to output
            end_sample = min(start_sample + len(bass_sample), len(audio))
            for i in range(start_sample, end_sample):
                if i < len(audio):
                    audio[i] += bass_sample[i - start_sample]
        
        return audio
    
    def _generate_bass_sample(self, frequency: float, duration: float, velocity: int, style: HardcoreStyle) -> List[float]:
        """Generate a single bass sample"""
        
        samples = int(duration * self.sample_rate)
        bass = [0.0] * samples
        
        # Acid bass synthesis with filter sweep
        for i in range(samples):
            t = i / self.sample_rate
            progress = t / duration if duration > 0 else 0
            
            # Sawtooth wave oscillator
            phase = (frequency * t) % 1.0
            sawtooth = (2 * phase - 1) * 0.6
            
            # Add sub oscillator
            sub_phase = (frequency * 0.5 * t) % 1.0
            sub = (2 * sub_phase - 1) * 0.3
            
            # Combine oscillators
            wave = sawtooth + sub
            
            # Filter envelope (classic acid sweep)
            filter_env = 1.0 - progress * 0.7  # Sweep down
            if style == HardcoreStyle.FRENCHCORE:
                filter_env = 1.0 - progress * 0.5  # Less sweep
            
            # Simple lowpass filter simulation
            cutoff_factor = filter_env * 0.8 + 0.2
            wave *= cutoff_factor
            
            # Resonance simulation (add some harmonics)
            if filter_env > 0.6:
                resonance = math.sin(2 * math.pi * frequency * 2 * t) * 0.1 * filter_env
                wave += resonance
            
            # Amplitude envelope
            if duration > 0:
                amp_env = max(0, 1.0 - progress) * 0.9 + 0.1
            else:
                amp_env = 1.0
            
            bass[i] = wave * amp_env * (velocity / 127.0) * 0.7
        
        return bass
    
    def _synthesize_percussion(self, notes: List[MIDINote], config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[float]:
        """Synthesize percussion elements"""
        
        if not notes:
            return []
        
        total_beats = config.length_bars * 4.0
        duration_sec = total_beats * 60.0 / config.bpm
        samples = int(duration_sec * self.sample_rate)
        audio = [0.0] * samples
        
        for note in notes:
            # FIXED: Correct beat-to-sample conversion (removed division by 4)
            start_sample = int(note.start_time * (60.0 / config.bpm) * self.sample_rate)

            # Generate percussion sample based on MIDI note
            if note.pitch == 42:  # Closed hi-hat
                perc_sample = self._generate_hihat_sample(note.velocity)
            else:
                perc_sample = self._generate_generic_perc(note.pitch, note.velocity)
            
            # Add to output
            end_sample = min(start_sample + len(perc_sample), len(audio))
            for i in range(start_sample, end_sample):
                if i < len(audio):
                    audio[i] += perc_sample[i - start_sample]
        
        return audio
    
    def _generate_hihat_sample(self, velocity: int) -> List[float]:
        """Generate hi-hat sample using noise"""
        
        duration = 0.1  # 100ms hi-hat
        samples = int(duration * self.sample_rate)
        hihat = [0.0] * samples
        
        for i in range(samples):
            t = i / self.sample_rate
            # High frequency noise
            noise = (random.random() - 0.5) * 2.0
            # High-pass filter simulation
            filtered_noise = noise * (1.0 - math.exp(-t * 100))
            # Exponential decay
            envelope = math.exp(-t * 30)
            
            hihat[i] = filtered_noise * envelope * (velocity / 127.0) * 0.3
        
        return hihat
    
    def _generate_generic_perc(self, pitch: int, velocity: int) -> List[float]:
        """Generate generic percussion sample"""
        
        duration = 0.15
        samples = int(duration * self.sample_rate)
        perc = [0.0] * samples
        
        # Simple pitched noise
        frequency = 440.0 * (2 ** ((pitch - 69) / 12.0))
        
        for i in range(samples):
            t = i / self.sample_rate
            noise = (random.random() - 0.5) * 2.0
            tone = math.sin(2 * math.pi * frequency * t) * 0.3
            envelope = math.exp(-t * 20)
            
            perc[i] = (noise * 0.7 + tone * 0.3) * envelope * (velocity / 127.0) * 0.4
        
        return perc


class BMadMixEngineer:
    """@mix-engineer (Phoenix) - Perfectionist with warehouse sound obsession"""
    
    def __init__(self):
        self.name = "Phoenix"
        self.role = "@mix-engineer"
        self.specialty = "Professional mixing and mastering"
        self.sample_rate = 44100
    
    def create_professional_mix(self, audio_tracks: Dict[str, List[float]], 
                              config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[float]:
        """Create professional hardcore mix"""
        
        print(f"   🎚️ Mixing {config.style.name} track...")
        
        if not audio_tracks:
            return []
        
        # Find longest track
        max_length = max(len(track) for track in audio_tracks.values())
        mix = [0.0] * max_length
        
        # Mix tracks with appropriate levels and processing
        for name, audio in audio_tracks.items():
            if not audio:
                continue
            
            print(f"      • Processing {name} track...")
            
            # Pad shorter tracks
            padded = audio + [0.0] * (max_length - len(audio))
            
            # Apply track-specific processing
            processed = self._process_track(padded, name, config, analysis)
            
            # Mix into output
            for i in range(len(processed)):
                if i < len(mix):
                    mix[i] += processed[i]
        
        # Apply master bus processing
        print(f"      • Applying master processing...")
        mix = self._apply_master_processing(mix, config, analysis)
        
        print(f"      • Final mix: {len(mix)} samples")
        
        return mix
    
    def _process_track(self, audio: List[float], track_name: str, 
                      config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[float]:
        """Apply track-specific processing"""
        
        processed = audio.copy()
        
        # Track-specific levels and EQ
        if track_name == "kick":
            # Kick: full level, some compression
            processed = self._apply_compression(processed, ratio=4.0, threshold=0.7)
            processed = [sample * 1.0 for sample in processed]  # Full level
        
        elif track_name == "bassline":
            # Bass: slight compression, EQ
            processed = self._apply_compression(processed, ratio=3.0, threshold=0.6)
            processed = self._apply_distortion(processed, amount=0.2)
            processed = [sample * 0.8 for sample in processed]  # Slightly lower
        
        elif track_name == "percussion":
            # Percussion: light processing
            processed = [sample * 0.6 for sample in processed]  # Background level
        
        return processed
    
    def _apply_compression(self, audio: List[float], ratio: float = 4.0, threshold: float = 0.7) -> List[float]:
        """Apply basic compression"""
        
        compressed = []
        
        for sample in audio:
            abs_sample = abs(sample)
            
            if abs_sample > threshold:
                # Compress above threshold
                over_threshold = abs_sample - threshold
                compressed_over = over_threshold / ratio
                new_level = threshold + compressed_over
                
                # Preserve sign
                compressed_sample = new_level * (1 if sample >= 0 else -1)
            else:
                compressed_sample = sample
            
            compressed.append(compressed_sample)
        
        return compressed
    
    def _apply_distortion(self, audio: List[float], amount: float = 0.3) -> List[float]:
        """Apply gentle distortion for analog character"""
        
        distorted = []
        
        for sample in audio:
            # Soft saturation curve
            driven = sample * (1.0 + amount)
            saturated = math.tanh(driven) * 0.8
            distorted.append(saturated)
        
        return distorted
    
    def _apply_master_processing(self, audio: List[float], config: BMadTrackConfig, analysis: Dict[str, Any]) -> List[float]:
        """Apply master bus processing"""
        
        processed = audio.copy()
        
        # Master compression
        processed = self._apply_compression(processed, ratio=2.5, threshold=0.8)
        
        # Add some warehouse reverb for atmosphere
        if config.style in [HardcoreStyle.BERLIN_INDUSTRIAL, HardcoreStyle.UK_HARDCORE]:
            processed = self._apply_reverb(processed, wet_level=0.15)
        
        # Master limiting
        processed = self._apply_limiter(processed, threshold=0.95)
        
        # Final level adjustment
        processed = [sample * 0.8 for sample in processed]
        
        return processed
    
    def _apply_reverb(self, audio: List[float], wet_level: float = 0.2) -> List[float]:
        """Apply basic reverb"""
        
        # Simple delay-based reverb
        delay_samples = int(0.03 * self.sample_rate)  # 30ms delay
        reverb = audio.copy()
        
        # Add delayed signal
        for i in range(delay_samples, len(audio)):
            reverb[i] += audio[i - delay_samples] * 0.3
        
        # Mix wet/dry
        output = []
        for i in range(len(audio)):
            wet = reverb[i] if i < len(reverb) else 0
            dry = audio[i]
            output.append(dry * (1 - wet_level) + wet * wet_level)
        
        return output
    
    def _apply_limiter(self, audio: List[float], threshold: float = 0.95) -> List[float]:
        """Apply brick wall limiting"""
        
        limited = []
        
        for sample in audio:
            if sample > threshold:
                limited.append(threshold)
            elif sample < -threshold:
                limited.append(-threshold)
            else:
                limited.append(sample)
        
        return limited
    
    def export_wav_file(self, audio: List[float], filepath: str) -> bool:
        """Export audio as WAV file"""
        
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                
                # Convert to 16-bit integers
                audio_data = b''
                for sample in audio:
                    # Clamp to valid range
                    clamped = max(-1.0, min(1.0, sample))
                    int_sample = int(clamped * 32767)
                    audio_data += struct.pack('<h', int_sample)
                
                wav_file.writeframes(audio_data)
            
            return True
        except Exception as e:
            print(f"WAV export error: {e}")
            return False


class BMadStandaloneCoordinator:
    """
    BMAD Standalone Music Coordinator
    
    The REAL hardcore music production factory that generates actual files.
    Coordinates all BMAD agents to create authentic hardcore tracks.
    """
    
    def __init__(self, output_dir: str = "bmad_hardcore_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize BMAD agents
        self.analyst = BMadMusicAnalyst()
        self.producer = BMadMusicProducer()
        self.sound_designer = BMadSoundDesigner()
        self.mix_engineer = BMadMixEngineer()
        
        print("🎛️ BMAD Standalone Music Coordinator initialized")
        print(f"   Output directory: {self.output_dir.absolute()}")
        print(f"   BMAD Agents ready:")
        print(f"     • {self.analyst.role} ({self.analyst.name}): {self.analyst.specialty}")
        print(f"     • {self.producer.role} ({self.producer.name}): {self.producer.specialty}")
        print(f"     • {self.sound_designer.role} ({self.sound_designer.name}): {self.sound_designer.specialty}")
        print(f"     • {self.mix_engineer.role} ({self.mix_engineer.name}): {self.mix_engineer.specialty}")
    
    def generate_hardcore_track(self, config: BMadTrackConfig) -> str:
        """Generate a complete hardcore track with real files"""
        
        # Create session
        session_id = f"bmad_{config.style.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        print(f"\n🎵 BMAD HARDCORE TRACK GENERATION")
        print(f"=" * 50)
        print(f"Session: {session_id}")
        print(f"Style: {config.style.name}")
        print(f"BPM: {config.bpm}")
        print(f"Length: {config.length_bars} bars")
        print(f"Key: {config.key}")
        
        try:
            # Set seed for reproducibility
            if config.seed:
                random.seed(config.seed)
                print(f"Seed: {config.seed}")
            
            start_time = time.time()
            
            # Step 1: Music Analysis
            print(f"\n🔍 {self.analyst.role} ({self.analyst.name}) - Pattern Analysis")
            analysis = self.analyst.analyze_hardcore_patterns(config)
            
            # Step 2: MIDI Generation  
            print(f"\n🎹 {self.producer.role} ({self.producer.name}) - MIDI Generation")
            midi_tracks = self.producer.generate_hardcore_midi(config, analysis)
            
            # Export MIDI files
            midi_files = []
            for track_name, notes in midi_tracks.items():
                if notes:
                    midi_path = session_dir / f"{track_name}.mid"
                    if SimpleMIDIExporter.export_midi(notes, str(midi_path), config.bpm):
                        midi_files.append(midi_path.name)
                        print(f"      ✅ {midi_path.name}")
            
            # Step 3: Audio Synthesis
            print(f"\n🎛️ {self.sound_designer.role} ({self.sound_designer.name}) - Audio Synthesis")
            audio_tracks = self.sound_designer.synthesize_hardcore_audio(midi_tracks, config, analysis)
            
            # Export individual audio tracks
            audio_files = []
            for track_name, audio in audio_tracks.items():
                if audio:
                    wav_path = session_dir / f"{track_name}_track.wav"
                    if self.mix_engineer.export_wav_file(audio, str(wav_path)):
                        audio_files.append(wav_path.name)
                        print(f"      ✅ {wav_path.name}")
            
            # Step 4: Professional Mixing
            print(f"\n🎚️ {self.mix_engineer.role} ({self.mix_engineer.name}) - Professional Mixing")
            final_mix = self.mix_engineer.create_professional_mix(audio_tracks, config, analysis)
            
            # Export final track
            final_path = session_dir / f"{session_id}_final.wav"
            if self.mix_engineer.export_wav_file(final_mix, str(final_path)):
                print(f"      ✅ {final_path.name}")
            
            generation_time = time.time() - start_time
            
            # Create session report
            self._create_session_report(session_dir, session_id, config, analysis, 
                                      midi_files, audio_files, generation_time)
            
            print(f"\n🎉 HARDCORE TRACK GENERATION COMPLETE!")
            print(f"   Session: {session_id}")
            print(f"   Generation time: {generation_time:.1f} seconds")
            print(f"   Output directory: {session_dir.name}")
            print(f"   Final track: {session_id}_final.wav")
            print(f"   🔊 Play the WAV file to hear your hardcore track!")
            
            return session_id
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise
    
    def _create_session_report(self, session_dir: Path, session_id: str, config: BMadTrackConfig,
                             analysis: Dict[str, Any], midi_files: List[str], audio_files: List[str],
                             generation_time: float):
        """Create detailed session report"""
        
        report_path = session_dir / "session_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("BMAD HARDCORE MUSIC GENERATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Generation Time: {generation_time:.1f} seconds\n")
            f.write(f"Style: {config.style.name}\n")
            f.write(f"BPM: {config.bpm}\n")
            f.write(f"Length: {config.length_bars} bars\n")
            f.write(f"Key: {config.key}\n")
            if config.seed:
                f.write(f"Seed: {config.seed}\n")
            f.write("\n")
            
            f.write("ANALYSIS RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Pattern Complexity: {analysis['pattern_complexity']['overall']:.2f}\n")
            f.write(f"Energy Sections: {len(analysis['energy_curve'])}\n")
            f.write(f"Breakdown Points: {len(analysis['breakdown_points'])}\n")
            f.write(f"Scale Degrees: {len(analysis['scale_degrees'])}\n")
            f.write("\n")
            
            f.write("GENERATED FILES:\n")
            f.write("-" * 20 + "\n")
            f.write("MIDI Files:\n")
            for midi_file in midi_files:
                f.write(f"  • {midi_file}\n")
            f.write("Audio Files:\n")
            for audio_file in audio_files:
                f.write(f"  • {audio_file}\n")
            f.write(f"  • {session_id}_final.wav (Final Mix)\n")
            f.write("\n")
            
            f.write("BMAD PRODUCTION TEAM:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• {self.analyst.role} ({self.analyst.name})\n")
            f.write(f"  {self.analyst.specialty}\n")
            f.write(f"• {self.producer.role} ({self.producer.name})\n") 
            f.write(f"  {self.producer.specialty}\n")
            f.write(f"• {self.sound_designer.role} ({self.sound_designer.name})\n")
            f.write(f"  {self.sound_designer.specialty}\n")
            f.write(f"• {self.mix_engineer.role} ({self.mix_engineer.name})\n")
            f.write(f"  {self.mix_engineer.specialty}\n")
            f.write("\n")
            
            f.write("TECHNICAL SPECIFICATIONS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Sample Rate: 44100 Hz\n")
            f.write(f"Bit Depth: 16-bit\n")
            f.write(f"Channels: Mono\n")
            f.write(f"Format: WAV\n")
    
    def generate_hardcore_collection(self, count: int = 5) -> List[str]:
        """Generate a collection of hardcore tracks"""
        
        print(f"\n🏭 BMAD HARDCORE MUSIC FACTORY")
        print(f"=" * 40)
        print(f"Generating {count} hardcore tracks...")
        
        sessions = []
        styles = list(HardcoreStyle)
        keys = ["A_minor", "E_minor", "C_minor"]
        
        for i in range(count):
            print(f"\n📀 Track {i + 1}/{count}")
            
            # Create varied configuration
            style = random.choice(styles)
            
            # Style-specific BPM ranges
            if style == HardcoreStyle.ROTTERDAM_GABBER:
                bpm = random.uniform(160, 180)
            elif style == HardcoreStyle.FRENCHCORE:
                bpm = random.uniform(180, 220)
            elif style == HardcoreStyle.SPEEDCORE:
                bpm = random.uniform(200, 300)
            elif style == HardcoreStyle.UK_HARDCORE:
                bpm = random.uniform(160, 180)
            else:  # BERLIN_INDUSTRIAL
                bpm = random.uniform(130, 150)
            
            config = BMadTrackConfig(
                style=style,
                bpm=bpm,
                length_bars=random.choice([16, 20, 24, 32]),
                key=random.choice(keys),
                seed=1000 + i
            )
            
            try:
                session_id = self.generate_hardcore_track(config)
                sessions.append(session_id)
                print(f"✅ Track {i + 1} complete: {session_id}")
            except Exception as e:
                print(f"❌ Track {i + 1} failed: {e}")
        
        print(f"\n🎊 FACTORY RUN COMPLETE!")
        print(f"   Generated {len(sessions)} hardcore tracks")
        print(f"   Output directory: {self.output_dir.absolute()}")
        
        return sessions


# Convenience functions for quick generation
def quick_gabber_track(bpm: float = 180.0, length_bars: float = 24.0) -> str:
    """Quick generation of Rotterdam gabber track"""
    coordinator = BMadStandaloneCoordinator()
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=bpm,
        length_bars=length_bars,
        key="A_minor",
        seed=random.randint(1000, 9999)
    )
    return coordinator.generate_hardcore_track(config)


def quick_frenchcore_track(bpm: float = 200.0, length_bars: float = 20.0) -> str:
    """Quick generation of frenchcore track"""
    coordinator = BMadStandaloneCoordinator()
    config = BMadTrackConfig(
        style=HardcoreStyle.FRENCHCORE,
        bpm=bpm,
        length_bars=length_bars,
        key="E_minor",
        seed=random.randint(1000, 9999)
    )
    return coordinator.generate_hardcore_track(config)


def quick_speedcore_track(bpm: float = 250.0, length_bars: float = 16.0) -> str:
    """Quick generation of speedcore track"""
    coordinator = BMadStandaloneCoordinator()
    config = BMadTrackConfig(
        style=HardcoreStyle.SPEEDCORE,
        bpm=bpm,
        length_bars=length_bars,
        key="C_minor",
        seed=random.randint(1000, 9999)
    )
    return coordinator.generate_hardcore_track(config)


def demo_bmad_standalone():
    """Demonstration of BMAD standalone coordinator"""
    
    print("🎛️ BMAD STANDALONE MUSIC COORDINATOR DEMO")
    print("=" * 55)
    
    coordinator = BMadStandaloneCoordinator()
    
    # Generate demo track
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=180.0,
        length_bars=16.0,
        key="A_minor",
        seed=42  # Reproducible demo
    )
    
    session_id = coordinator.generate_hardcore_track(config)
    
    print(f"\n✨ DEMO COMPLETE!")
    print(f"   Check your generated hardcore track:")
    print(f"   Directory: bmad_hardcore_output/{session_id}/")
    print(f"   Final track: {session_id}_final.wav")
    print(f"   🎧 Open the WAV file in any audio player to hear your hardcore track!")


if __name__ == "__main__":
    demo_bmad_standalone()