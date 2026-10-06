#!/usr/bin/env python3
"""
Simple BMAD Test - Generate a hardcore track without emojis for Windows compatibility
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
    ROTTERDAM_GABBER = "rotterdam_gabber"
    FRENCHCORE = "frenchcore"


@dataclass
class BMadTrackConfig:
    style: HardcoreStyle = HardcoreStyle.ROTTERDAM_GABBER
    bpm: float = 180.0
    length_bars: float = 8.0
    key: str = "A_minor"
    seed: Optional[int] = None


@dataclass 
class MIDINote:
    pitch: int
    velocity: int
    start_time: float
    duration: float
    channel: int = 0


class SimpleMIDIExporter:
    @staticmethod
    def export_midi(notes: List[MIDINote], filepath: str, bpm: float = 120) -> bool:
        try:
            with open(filepath, 'wb') as f:
                # Basic MIDI header
                f.write(b'MThd')
                f.write(struct.pack('>I', 6))
                f.write(struct.pack('>H', 0))
                f.write(struct.pack('>H', 1))
                f.write(struct.pack('>H', 480))
                
                # Track header
                f.write(b'MTrk')
                
                # Calculate track data
                track_data = b''
                
                # Tempo event
                tempo = int(60000000 / bpm)
                track_data += SimpleMIDIExporter._variable_length(0)
                track_data += b'\xff\x51\x03'
                track_data += struct.pack('>I', tempo)[1:]
                
                # Convert notes to MIDI events
                events = []
                for note in notes:
                    start_ticks = int(note.start_time * 480)
                    end_ticks = int((note.start_time + note.duration) * 480)
                    
                    events.append((start_ticks, 'note_on', note))
                    events.append((end_ticks, 'note_off', note))
                
                events.sort(key=lambda x: x[0])
                
                current_time = 0
                for event_time, event_type, note in events:
                    delta = event_time - current_time
                    current_time = event_time
                    
                    track_data += SimpleMIDIExporter._variable_length(delta)
                    
                    if event_type == 'note_on':
                        track_data += bytes([0x90 | note.channel, note.pitch, note.velocity])
                    else:
                        track_data += bytes([0x80 | note.channel, note.pitch, 0])
                
                track_data += SimpleMIDIExporter._variable_length(0)
                track_data += b'\xff\x2f\x00'
                
                f.write(struct.pack('>I', len(track_data)))
                f.write(track_data)
            
            return True
        except Exception as e:
            print(f"MIDI export error: {e}")
            return False
    
    @staticmethod
    def _variable_length(value: int) -> bytes:
        result = b''
        while value > 0x7f:
            result = bytes([0x80 | (value & 0x7f)]) + result
            value >>= 7
        result = bytes([value & 0x7f]) + result
        return result if result else b'\x00'


class BMadMusicProducer:
    def __init__(self):
        self.name = "Raven"
    
    def generate_hardcore_midi(self, config: BMadTrackConfig) -> Dict[str, List[MIDINote]]:
        print(f"   Generating MIDI for {config.style.name}...")
        
        midi_tracks = {}
        
        # Generate kick pattern
        kick_notes = self._generate_kick_pattern(config)
        midi_tracks["kick"] = kick_notes
        print(f"      Kick pattern: {len(kick_notes)} notes")
        
        # Generate bassline
        bass_notes = self._generate_bassline_pattern(config)
        midi_tracks["bassline"] = bass_notes
        print(f"      Bassline: {len(bass_notes)} notes")
        
        return midi_tracks
    
    def _generate_kick_pattern(self, config: BMadTrackConfig) -> List[MIDINote]:
        pattern = [True, False, True, False, True, False, True, False]  # Basic gabber pattern
        notes = []
        
        total_beats = config.length_bars * 4.0
        step_size = 0.25
        
        current_beat = 0.0
        step_index = 0
        
        while current_beat < total_beats:
            pattern_step = step_index % len(pattern)
            
            if pattern[pattern_step]:
                velocity = 120 + random.randint(-10, 7)
                notes.append(MIDINote(
                    pitch=36,  # C1 kick
                    velocity=velocity,
                    start_time=current_beat,
                    duration=0.2,
                    channel=9
                ))
            
            current_beat += step_size
            step_index += 1
        
        return notes
    
    def _generate_bassline_pattern(self, config: BMadTrackConfig) -> List[MIDINote]:
        pattern = [False, True, False, True, False, True, False, True]
        scale_degrees = [0, 2, 3, 5, 7, 8, 10]  # A minor
        notes = []
        
        root_note = 45  # A1
        total_beats = config.length_bars * 4.0
        step_size = 0.25
        
        current_beat = 0.0
        step_index = 0
        current_note = 0
        
        while current_beat < total_beats:
            pattern_step = step_index % len(pattern)
            
            if pattern[pattern_step]:
                scale_degree = scale_degrees[current_note % len(scale_degrees)]
                pitch = root_note + scale_degree
                
                velocity = 90 + random.randint(-15, 15)
                duration = random.choice([0.2, 0.25, 0.3])
                
                notes.append(MIDINote(
                    pitch=pitch,
                    velocity=velocity,
                    start_time=current_beat,
                    duration=duration,
                    channel=0
                ))
                
                current_note += random.choice([-1, 1])
                current_note = max(0, current_note) % len(scale_degrees)
            
            current_beat += step_size
            step_index += 1
        
        return notes


class BMadSoundDesigner:
    def __init__(self):
        self.name = "Void"
        self.sample_rate = 44100
    
    def synthesize_hardcore_audio(self, midi_tracks: Dict[str, List[MIDINote]], config: BMadTrackConfig) -> Dict[str, List[float]]:
        print(f"   Synthesizing {config.style.name} audio...")
        
        audio_tracks = {}
        
        if "kick" in midi_tracks:
            kick_audio = self._synthesize_kicks(midi_tracks["kick"], config)
            audio_tracks["kick"] = kick_audio
            print(f"      Kick synthesis: {len(kick_audio)} samples")
        
        if "bassline" in midi_tracks:
            bass_audio = self._synthesize_bassline(midi_tracks["bassline"], config)
            audio_tracks["bassline"] = bass_audio
            print(f"      Bass synthesis: {len(bass_audio)} samples")
        
        return audio_tracks
    
    def _synthesize_kicks(self, notes: List[MIDINote], config: BMadTrackConfig) -> List[float]:
        if not notes:
            return []
        
        total_beats = config.length_bars * 4.0
        duration_sec = total_beats * 60.0 / config.bpm
        samples = int(duration_sec * self.sample_rate)
        audio = [0.0] * samples
        
        for note in notes:
            # note.start_time is already in BEATS, not 16th notes!
            seconds_per_beat = 60.0 / config.bpm
            start_sample = int(note.start_time * seconds_per_beat * self.sample_rate)
            kick_sample = self._generate_kick_sample(150.0, note.velocity)  # Proper kick frequency
            
            end_sample = min(start_sample + len(kick_sample), len(audio))
            for i in range(start_sample, end_sample):
                if i < len(audio):
                    audio[i] += kick_sample[i - start_sample]
        
        return audio
    
    def _generate_kick_sample(self, frequency: float, velocity: int) -> List[float]:
        duration = 0.15  # Shorter, punchier kick
        samples = int(duration * self.sample_rate)
        kick = [0.0] * samples
        
        for i in range(samples):
            t = i / self.sample_rate
            # Hardcore 909-style pitch envelope
            freq = frequency * math.exp(-t * 35)  # Fast pitch drop
            envelope = math.exp(-t * 25)  # Fast decay for punch
            
            # Layer 1: Main punch
            punch = math.sin(2 * math.pi * freq * t) * 0.9
            # Layer 2: Sub harmonics
            sub = math.sin(2 * math.pi * 50 * t) * 0.5 * envelope
            
            wave = punch * envelope + sub
            
            # Soft clip for distortion
            if abs(wave) > 0.8:
                wave = 0.8 * (1 if wave > 0 else -1)
            
            kick[i] = wave * (velocity / 127.0) * 1.5
        
        return kick
    
    def _synthesize_bassline(self, notes: List[MIDINote], config: BMadTrackConfig) -> List[float]:
        if not notes:
            return []
        
        total_beats = config.length_bars * 4.0
        duration_sec = total_beats * 60.0 / config.bpm
        samples = int(duration_sec * self.sample_rate)
        audio = [0.0] * samples
        
        for note in notes:
            # note.start_time is already in BEATS, not 16th notes!
            seconds_per_beat = 60.0 / config.bpm
            start_sample = int(note.start_time * seconds_per_beat * self.sample_rate)
            note_duration = note.duration * seconds_per_beat
            
            frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
            bass_sample = self._generate_bass_sample(frequency, note_duration, note.velocity)
            
            end_sample = min(start_sample + len(bass_sample), len(audio))
            for i in range(start_sample, end_sample):
                if i < len(audio):
                    audio[i] += bass_sample[i - start_sample]
        
        return audio
    
    def _generate_bass_sample(self, frequency: float, duration: float, velocity: int) -> List[float]:
        samples = int(duration * self.sample_rate)
        bass = [0.0] * samples
        
        for i in range(samples):
            t = i / self.sample_rate
            progress = t / duration if duration > 0 else 0
            
            phase = (frequency * t) % 1.0
            sawtooth = (2 * phase - 1) * 0.6
            
            sub_phase = (frequency * 0.5 * t) % 1.0
            sub = (2 * sub_phase - 1) * 0.3
            
            wave = sawtooth + sub
            
            filter_env = 1.0 - progress * 0.7
            cutoff_factor = filter_env * 0.8 + 0.2
            wave *= cutoff_factor
            
            amp_env = max(0, 1.0 - progress) * 0.9 + 0.1 if duration > 0 else 1.0
            
            bass[i] = wave * amp_env * (velocity / 127.0) * 0.7
        
        return bass


class BMadMixEngineer:
    def __init__(self):
        self.name = "Phoenix"
        self.sample_rate = 44100
    
    def create_professional_mix(self, audio_tracks: Dict[str, List[float]], config: BMadTrackConfig) -> List[float]:
        print(f"   Mixing {config.style.name} track...")
        
        if not audio_tracks:
            return []
        
        max_length = max(len(track) for track in audio_tracks.values())
        mix = [0.0] * max_length
        
        for name, audio in audio_tracks.items():
            if not audio:
                continue
            
            print(f"      Processing {name} track...")
            padded = audio + [0.0] * (max_length - len(audio))
            
            # Track levels
            if name == "kick":
                level = 1.0
            elif name == "bassline":
                level = 0.8
            else:
                level = 0.6
            
            for i in range(len(padded)):
                if i < len(mix):
                    mix[i] += padded[i] * level
        
        # Simple limiting
        max_val = max(abs(s) for s in mix) if mix else 1.0
        if max_val > 1.0:
            mix = [s / max_val * 0.9 for s in mix]
        
        print(f"      Final mix: {len(mix)} samples")
        return mix
    
    def export_wav_file(self, audio: List[float], filepath: str) -> bool:
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                
                audio_data = b''
                for sample in audio:
                    clamped = max(-1.0, min(1.0, sample))
                    int_sample = int(clamped * 32767)
                    audio_data += struct.pack('<h', int_sample)
                
                wav_file.writeframes(audio_data)
            
            return True
        except Exception as e:
            print(f"WAV export error: {e}")
            return False


class BMadSimpleCoordinator:
    def __init__(self, output_dir: str = "bmad_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.producer = BMadMusicProducer()
        self.sound_designer = BMadSoundDesigner()
        self.mix_engineer = BMadMixEngineer()
        
        print("BMAD Simple Coordinator initialized")
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def generate_hardcore_track(self, config: BMadTrackConfig) -> str:
        session_id = f"bmad_{config.style.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        print(f"\nBMAD HARDCORE TRACK GENERATION")
        print(f"Session: {session_id}")
        print(f"Style: {config.style.name}")
        print(f"BPM: {config.bpm}")
        print(f"Length: {config.length_bars} bars")
        
        try:
            if config.seed:
                random.seed(config.seed)
            
            # Generate MIDI
            print(f"\nMusic Producer ({self.producer.name}) - MIDI Generation")
            midi_tracks = self.producer.generate_hardcore_midi(config)
            
            # Export MIDI files
            for track_name, notes in midi_tracks.items():
                if notes:
                    midi_path = session_dir / f"{track_name}.mid"
                    if SimpleMIDIExporter.export_midi(notes, str(midi_path), config.bpm):
                        print(f"      MIDI exported: {midi_path.name}")
            
            # Synthesize audio
            print(f"\nSound Designer ({self.sound_designer.name}) - Audio Synthesis")
            audio_tracks = self.sound_designer.synthesize_hardcore_audio(midi_tracks, config)
            
            # Export individual tracks
            for track_name, audio in audio_tracks.items():
                if audio:
                    wav_path = session_dir / f"{track_name}_track.wav"
                    if self.mix_engineer.export_wav_file(audio, str(wav_path)):
                        print(f"      Audio exported: {wav_path.name}")
            
            # Create final mix
            print(f"\nMix Engineer ({self.mix_engineer.name}) - Professional Mixing")
            final_mix = self.mix_engineer.create_professional_mix(audio_tracks, config)
            
            # Export final track
            final_path = session_dir / f"{session_id}_final.wav"
            if self.mix_engineer.export_wav_file(final_mix, str(final_path)):
                print(f"      Final track exported: {final_path.name}")
            
            print(f"\nHARDCORE TRACK GENERATION COMPLETE!")
            print(f"Session: {session_id}")
            print(f"Output: {session_dir.name}")
            print(f"Final track: {session_id}_final.wav")
            
            return session_id
            
        except Exception as e:
            print(f"Generation failed: {e}")
            raise


def test_bmad_simple():
    """Simple test of BMAD coordinator"""
    print("BMAD Simple Test - Generating Hardcore Track")
    print("=" * 45)
    
    coordinator = BMadSimpleCoordinator()
    
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=180.0,
        length_bars=8.0,
        key="A_minor",
        seed=42
    )
    
    session_id = coordinator.generate_hardcore_track(config)
    
    print(f"\nTest complete!")
    print(f"Check bmad_output/{session_id}/ for your hardcore track!")
    return session_id


if __name__ == "__main__":
    test_bmad_simple()