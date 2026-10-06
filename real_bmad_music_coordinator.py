#!/usr/bin/env python3
"""
Real BMAD Music Coordinator - Hardcore Music Production Factory

This is the REAL music production coordinator that generates actual MIDI and audio files
using the existing infrastructure. Replaces fake simulation with authentic hardcore
music generation coordinated by BMAD agents.

Features:
- @music-analyst: Analyzes hardcore patterns and creates evolution parameters
- @music-producer: Uses AcidBasslineGenerator and TunedKickGenerator for real MIDI
- @sound-designer: Applies real effects chains and synthesis using existing engines
- @mix-engineer: Creates complete professional tracks with real audio export

Generates actual files that can be played in real audio players!
"""

import os
import time
import random
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# Import existing infrastructure components
from cli_shared.generators.acid_bassline import (
    AcidBasslineGenerator, create_hardcore_acid_line, create_classic_acid_line
)
from cli_shared.generators.tuned_kick import (
    TunedKickGenerator, create_frenchcore_kicks, create_hardstyle_kicks
)
from cli_shared.models.midi_clips import MIDIClip, TriggerClip, create_empty_midi_clip
from audio.core.track import Track, TrackCollection, PatternControlSource, KickAudioSource
from audio.synthesis.oscillators import (
    gabber_oscillator_bank, industrial_oscillator_bank, terrorcore_oscillator_bank
)
from audio.synthesis.fm_engine import FMSynthEngineExtended, FMAlgorithm
from audio.effects.distortion import (
    rotterdam_doorlussen, apply_rotterdam_gabber_distortion, 
    apply_industrial_hardcore_distortion, alpha_juno_distortion
)
from audio.effects.dynamics import (
    apply_gabber_compression, apply_industrial_compression, apply_hardcore_limiter
)
from audio.effects.spatial import warehouse_reverb, industrial_reverb, apply_eighth_note_delay
from audio.parameters.synthesis_constants import (
    HardcoreConstants, SynthesisParams, HARDCORE_STYLE_PRESETS, HardcoreStyle
)

# Audio export
import scipy.io.wavfile as wavfile


@dataclass
class BMadTrackConfig:
    """Configuration for BMAD track generation"""
    style: HardcoreStyle = HardcoreStyle.ROTTERDAM_GABBER
    bpm: float = 180.0
    length_bars: float = 32.0
    key: str = "A_minor"
    
    # Track structure
    has_intro: bool = True
    has_buildup: bool = True
    has_drop: bool = True
    has_breakdown: bool = True
    has_outro: bool = True
    
    # Generation settings
    seed: Optional[int] = None
    variation_factor: float = 0.7  # How much variation to add


@dataclass
class BMadGenerationSession:
    """A complete BMAD music generation session"""
    session_id: str
    config: BMadTrackConfig
    output_dir: Path
    
    # Generated content
    midi_clips: Dict[str, MIDIClip] = field(default_factory=dict)
    audio_clips: Dict[str, np.ndarray] = field(default_factory=dict)
    final_track: Optional[np.ndarray] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    generation_log: List[str] = field(default_factory=list)


class BMadMusicAnalyst:
    """@music-analyst (Nexus) - Pattern recognition and hardcore evolution"""
    
    def __init__(self):
        self.name = "Nexus"
        self.role = "@music-analyst"
        self.specialty = "Pattern recognition savant"
    
    def analyze_hardcore_pattern_structure(self, config: BMadTrackConfig) -> Dict[str, Any]:
        """Analyze hardcore patterns and create evolution parameters"""
        
        analysis = {
            "pattern_complexity": self._get_pattern_complexity(config.style),
            "rhythmic_elements": self._analyze_rhythmic_elements(config.style),
            "harmonic_progression": self._analyze_harmonic_structure(config.key),
            "energy_curve": self._generate_energy_curve(config.length_bars),
            "breakdown_points": self._identify_breakdown_points(config.length_bars)
        }
        
        return analysis
    
    def _get_pattern_complexity(self, style: HardcoreStyle) -> Dict[str, float]:
        """Determine pattern complexity based on hardcore style"""
        complexity_map = {
            HardcoreStyle.ROTTERDAM_GABBER: {"kick": 0.7, "bassline": 0.6, "lead": 0.5},
            HardcoreStyle.FRENCHCORE: {"kick": 0.9, "bassline": 0.8, "lead": 0.7},
            HardcoreStyle.SPEEDCORE: {"kick": 1.0, "bassline": 0.9, "lead": 0.8},
            HardcoreStyle.BERLIN_INDUSTRIAL: {"kick": 0.6, "bassline": 0.7, "lead": 0.8},
            HardcoreStyle.UK_HARDCORE: {"kick": 0.8, "bassline": 0.7, "lead": 0.9},
            HardcoreStyle.TERRORCORE: {"kick": 1.0, "bassline": 1.0, "lead": 1.0}
        }
        
        return complexity_map.get(style, {"kick": 0.7, "bassline": 0.6, "lead": 0.5})
    
    def _analyze_rhythmic_elements(self, style: HardcoreStyle) -> Dict[str, str]:
        """Analyze rhythmic patterns for style"""
        patterns = {
            HardcoreStyle.ROTTERDAM_GABBER: {
                "kick_pattern": "x ~ x ~ x ~ x ~",
                "bass_pattern": "~ x ~ x ~ x ~ x",
                "hat_pattern": "~ ~ x ~ ~ ~ x ~"
            },
            HardcoreStyle.FRENCHCORE: {
                "kick_pattern": "x x ~ x x ~ x x",
                "bass_pattern": "~ ~ x ~ ~ x ~ ~",
                "hat_pattern": "~ x ~ x ~ x ~ x"
            },
            HardcoreStyle.BERLIN_INDUSTRIAL: {
                "kick_pattern": "x ~ ~ ~ x ~ ~ x",
                "bass_pattern": "~ x x ~ ~ x x ~",
                "hat_pattern": "~ ~ ~ x ~ ~ ~ x"
            }
        }
        
        return patterns.get(style, patterns[HardcoreStyle.ROTTERDAM_GABBER])
    
    def _analyze_harmonic_structure(self, key: str) -> List[str]:
        """Generate harmonic progression for key"""
        progressions = {
            "A_minor": ["Am", "F", "C", "G", "Am", "Dm", "G", "Am"],
            "E_minor": ["Em", "C", "G", "D", "Em", "Am", "D", "Em"],
            "C_minor": ["Cm", "Ab", "Eb", "Bb", "Cm", "Fm", "Bb", "Cm"]
        }
        
        return progressions.get(key, progressions["A_minor"])
    
    def _generate_energy_curve(self, length_bars: float) -> List[float]:
        """Generate energy curve for track structure"""
        # Classic hardcore energy curve: intro -> buildup -> drop -> breakdown -> outro
        total_points = int(length_bars / 4)  # One point per 4 bars
        
        energy = []
        for i in range(total_points):
            progress = i / max(1, total_points - 1)
            
            if progress < 0.2:  # Intro
                energy.append(0.3 + progress * 1.5)
            elif progress < 0.4:  # Buildup
                energy.append(0.6 + (progress - 0.2) * 2.0)
            elif progress < 0.7:  # Drop
                energy.append(1.0)
            elif progress < 0.9:  # Breakdown
                energy.append(1.0 - (progress - 0.7) * 2.5)
            else:  # Outro
                energy.append(0.5 - (progress - 0.9) * 5.0)
        
        return [max(0.1, min(1.0, e)) for e in energy]
    
    def _identify_breakdown_points(self, length_bars: float) -> List[float]:
        """Identify where breakdowns should occur"""
        breakdown_points = []
        
        # Standard hardcore breakdown positions
        if length_bars >= 16:
            breakdown_points.append(16.0)  # First breakdown
        if length_bars >= 24:
            breakdown_points.append(24.0)  # Second breakdown
        if length_bars >= 28:
            breakdown_points.append(length_bars - 4)  # Outro breakdown
            
        return breakdown_points


class BMadMusicProducer:
    """@music-producer (Raven) - Track composition using real MIDI generators"""
    
    def __init__(self):
        self.name = "Raven"
        self.role = "@music-producer"
        self.specialty = "Creative visionary with relentless drive"
    
    def generate_hardcore_track_midi(self, config: BMadTrackConfig, 
                                   analysis: Dict[str, Any]) -> Dict[str, MIDIClip]:
        """Generate complete MIDI track using existing generators"""
        
        clips = {}
        
        # Generate kick pattern using TunedKickGenerator
        clips["kick"] = self._generate_kick_pattern(config, analysis)
        
        # Generate bassline using AcidBasslineGenerator
        clips["bassline"] = self._generate_bassline_pattern(config, analysis)
        
        # Generate lead/hoover patterns
        clips["lead"] = self._generate_lead_pattern(config, analysis)
        
        # Generate percussion elements
        clips["percussion"] = self._generate_percussion_pattern(config, analysis)
        
        return clips
    
    def _generate_kick_pattern(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> MIDIClip:
        """Generate kick pattern using TunedKickGenerator"""
        
        rhythmic = analysis["rhythmic_elements"]
        complexity = analysis["pattern_complexity"]["kick"]
        
        if config.style == HardcoreStyle.FRENCHCORE:
            return create_frenchcore_kicks("C1", config.length_bars, config.bpm)
        elif config.style == HardcoreStyle.UK_HARDCORE:
            return create_hardstyle_kicks("C1", config.length_bars, config.bpm)
        else:
            # Use Rotterdam gabber style
            generator = TunedKickGenerator(
                root_note="C1",
                pattern=rhythmic["kick_pattern"],
                tuning="pentatonic"
            )
            return generator.generate(config.length_bars, config.bpm)
    
    def _generate_bassline_pattern(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> MIDIClip:
        """Generate bassline using AcidBasslineGenerator"""
        
        complexity = analysis["pattern_complexity"]["bassline"]
        
        if complexity > 0.8:
            return create_hardcore_acid_line(config.length_bars, config.bpm)
        else:
            return create_classic_acid_line(config.length_bars, config.bpm)
    
    def _generate_lead_pattern(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> MIDIClip:
        """Generate lead/hoover pattern"""
        
        # Create simple lead pattern for now
        # In a full implementation, this would use hoover generators
        clip = create_empty_midi_clip("lead", config.length_bars, config.bpm)
        clip.tags = ["lead", "hoover", "hardcore"]
        
        return clip
    
    def _generate_percussion_pattern(self, config: BMadTrackConfig, analysis: Dict[str, Any]) -> MIDIClip:
        """Generate percussion elements"""
        
        # Create percussion pattern
        clip = create_empty_midi_clip("percussion", config.length_bars, config.bpm)
        clip.tags = ["percussion", "hats", "hardcore"]
        
        return clip


class BMadSoundDesigner:
    """@sound-designer (Void) - Sonic alchemist using real synthesis engines"""
    
    def __init__(self):
        self.name = "Void"
        self.role = "@sound-designer"
        self.specialty = "Spectral manipulation master"
        self.fm_engine = FMSynthEngineExtended(sample_rate=HardcoreConstants.SAMPLE_RATE_44K)
    
    def synthesize_track_audio(self, midi_clips: Dict[str, MIDIClip], 
                             config: BMadTrackConfig) -> Dict[str, np.ndarray]:
        """Synthesize audio from MIDI clips using real synthesis engines"""
        
        audio_clips = {}
        
        # Synthesize kick using oscillator bank
        audio_clips["kick"] = self._synthesize_kick_audio(midi_clips["kick"], config)
        
        # Synthesize bassline using FM synthesis
        audio_clips["bassline"] = self._synthesize_bassline_audio(midi_clips["bassline"], config)
        
        # Synthesize lead using FM synthesis
        if "lead" in midi_clips:
            audio_clips["lead"] = self._synthesize_lead_audio(midi_clips["lead"], config)
        
        return audio_clips
    
    def _synthesize_kick_audio(self, midi_clip: MIDIClip, config: BMadTrackConfig) -> np.ndarray:
        """Synthesize kick drum using oscillator banks"""
        
        duration_samples = int((midi_clip.get_total_beats() * 60.0 / config.bpm) * HardcoreConstants.SAMPLE_RATE_44K)
        output = np.zeros(duration_samples)
        
        # Use style-specific oscillator
        if config.style == HardcoreStyle.ROTTERDAM_GABBER:
            synth_func = gabber_oscillator_bank
        elif config.style == HardcoreStyle.BERLIN_INDUSTRIAL:
            synth_func = industrial_oscillator_bank
        elif config.style == HardcoreStyle.TERRORCORE:
            synth_func = terrorcore_oscillator_bank
        else:
            synth_func = gabber_oscillator_bank
        
        # Generate kick samples for each note
        for note in midi_clip.notes:
            # FIXED: Correct beat-to-sample conversion (removed multiplication by 4)
            # Formula: samples = beats * (60/bpm) * sample_rate
            start_sample = int(note.start_time * (60.0 / config.bpm) * HardcoreConstants.SAMPLE_RATE_44K)

            # Generate kick sample
            frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
            duration_ms = int(note.duration * (60.0 / config.bpm) * 1000)
            
            kick_sample = synth_func(frequency, duration_ms)
            
            # Apply velocity
            kick_sample *= (note.velocity / 127.0)
            
            # Add to output
            end_sample = min(start_sample + len(kick_sample), len(output))
            if start_sample < len(output):
                output[start_sample:end_sample] += kick_sample[:end_sample - start_sample]
        
        return output
    
    def _synthesize_bassline_audio(self, midi_clip: MIDIClip, config: BMadTrackConfig) -> np.ndarray:
        """Synthesize bassline using FM synthesis"""
        
        # Load acid bass preset
        self.fm_engine.load_preset("acid_bass")
        
        duration_samples = int((midi_clip.get_total_beats() * 60.0 / config.bpm) * HardcoreConstants.SAMPLE_RATE_44K)
        output = np.zeros(duration_samples)
        
        # Generate FM synthesis for each note
        for note in midi_clip.notes:
            # FIXED: Correct beat-to-sample conversion (removed multiplication by 4)
            start_sample = int(note.start_time * (60.0 / config.bpm) * HardcoreConstants.SAMPLE_RATE_44K)
            duration_seconds = note.duration * (60.0 / config.bpm)
            note_samples = int(duration_seconds * HardcoreConstants.SAMPLE_RATE_44K)
            
            # Generate FM synthesis
            frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
            self.fm_engine.note_on(frequency, note.velocity / 127.0)
            
            fm_samples = self.fm_engine.generate_samples(note_samples)
            self.fm_engine.note_off()
            
            # Add to output
            end_sample = min(start_sample + len(fm_samples), len(output))
            if start_sample < len(output):
                output[start_sample:end_sample] += fm_samples[:end_sample - start_sample]
        
        return output
    
    def _synthesize_lead_audio(self, midi_clip: MIDIClip, config: BMadTrackConfig) -> np.ndarray:
        """Synthesize lead using FM synthesis"""
        
        # Load hoover preset
        self.fm_engine.load_preset("classic_hoover")
        
        duration_samples = int((midi_clip.get_total_beats() * 60.0 / config.bpm) * HardcoreConstants.SAMPLE_RATE_44K)
        output = np.zeros(duration_samples)
        
        # For now, return silence as lead is empty
        # In full implementation, this would generate hoover sounds
        
        return output
    
    def apply_hardcore_effects_chain(self, audio: np.ndarray, 
                                   track_type: str, config: BMadTrackConfig) -> np.ndarray:
        """Apply appropriate effects chain based on track type and style"""
        
        if len(audio) == 0:
            return audio
            
        processed = audio.copy()
        
        # Apply track-specific effects
        if track_type == "kick":
            processed = self._apply_kick_effects(processed, config)
        elif track_type == "bassline":
            processed = self._apply_bassline_effects(processed, config)
        elif track_type == "lead":
            processed = self._apply_lead_effects(processed, config)
        
        return processed
    
    def _apply_kick_effects(self, audio: np.ndarray, config: BMadTrackConfig) -> np.ndarray:
        """Apply kick-specific effects"""
        
        # Style-specific distortion
        if config.style == HardcoreStyle.ROTTERDAM_GABBER:
            audio = apply_rotterdam_gabber_distortion(audio)
        elif config.style == HardcoreStyle.BERLIN_INDUSTRIAL:
            audio = apply_industrial_hardcore_distortion(audio)
        
        # Compression
        audio = apply_gabber_compression(audio)
        
        return audio
    
    def _apply_bassline_effects(self, audio: np.ndarray, config: BMadTrackConfig) -> np.ndarray:
        """Apply bassline-specific effects"""
        
        # Acid distortion
        audio = rotterdam_doorlussen(audio, stages=2, drive_per_stage=1.8)
        
        # Compression
        audio = apply_industrial_compression(audio)
        
        return audio
    
    def _apply_lead_effects(self, audio: np.ndarray, config: BMadTrackConfig) -> np.ndarray:
        """Apply lead-specific effects"""
        
        # Hoover distortion
        audio = alpha_juno_distortion(audio)
        
        return audio


class BMadMixEngineer:
    """@mix-engineer (Phoenix) - Professional mixing and track assembly"""
    
    def __init__(self):
        self.name = "Phoenix"
        self.role = "@mix-engineer"
        self.specialty = "Warehouse-optimized mixing perfectionist"
    
    def create_final_mix(self, audio_clips: Dict[str, np.ndarray], 
                        config: BMadTrackConfig) -> np.ndarray:
        """Create final professional mix from audio clips"""
        
        # Find the longest clip to determine mix length
        max_length = max(len(clip) for clip in audio_clips.values() if len(clip) > 0)
        if max_length == 0:
            return np.array([])
        
        # Create mix bus
        mix = np.zeros(max_length)
        
        # Mix each track with appropriate levels
        for track_name, audio in audio_clips.items():
            if len(audio) == 0:
                continue
                
            # Pad shorter tracks
            if len(audio) < max_length:
                padded = np.pad(audio, (0, max_length - len(audio)))
            else:
                padded = audio[:max_length]
            
            # Apply track-specific mixing
            processed = self._apply_track_mixing(padded, track_name, config)
            
            # Add to mix
            mix += processed
        
        # Apply master bus processing
        mix = self._apply_master_bus_processing(mix, config)
        
        return mix
    
    def _apply_track_mixing(self, audio: np.ndarray, track_name: str, 
                          config: BMadTrackConfig) -> np.ndarray:
        """Apply track-specific mixing (EQ, compression, levels)"""
        
        # Track-specific levels
        levels = {
            "kick": 1.0,
            "bassline": 0.8,
            "lead": 0.7,
            "percussion": 0.6
        }
        
        level = levels.get(track_name, 0.5)
        return audio * level
    
    def _apply_master_bus_processing(self, audio: np.ndarray, 
                                   config: BMadTrackConfig) -> np.ndarray:
        """Apply master bus processing for professional sound"""
        
        if len(audio) == 0:
            return audio
        
        # Master compression
        audio = apply_gabber_compression(audio)
        
        # Spatial processing
        if config.style == HardcoreStyle.BERLIN_INDUSTRIAL:
            audio = industrial_reverb(audio, wet_level=0.2)
        else:
            audio = warehouse_reverb(audio, wet_level=0.15)
        
        # Add delay for movement
        audio = apply_eighth_note_delay(audio, config.bpm)
        
        # Final limiting
        audio = apply_hardcore_limiter(audio)
        
        # Master level
        audio *= 0.8
        
        return audio
    
    def export_audio_file(self, audio: np.ndarray, filepath: str) -> bool:
        """Export audio as WAV file"""
        
        try:
            # Ensure audio is in valid range
            audio = np.clip(audio, -1.0, 1.0)
            
            # Convert to 16-bit integer
            audio_int = (audio * 32767).astype(np.int16)
            
            # Export as WAV
            wavfile.write(filepath, HardcoreConstants.SAMPLE_RATE_44K, audio_int)
            
            return True
        except Exception as e:
            print(f"Error exporting audio: {e}")
            return False


class RealBMADMusicCoordinator:
    """
    Real BMAD Music Coordinator - The music production factory
    
    Coordinates all BMAD agents to generate real hardcore music files.
    This is the main orchestrator that creates actual MIDI and audio files.
    """
    
    def __init__(self, output_base_dir: str = "bmad_output"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        
        # Initialize BMAD agents
        self.analyst = BMadMusicAnalyst()
        self.producer = BMadMusicProducer()
        self.sound_designer = BMadSoundDesigner()
        self.mix_engineer = BMadMixEngineer()
        
        # Session tracking
        self.active_sessions: Dict[str, BMadGenerationSession] = {}
        
        print("🎛️ Real BMAD Music Coordinator initialized")
        print(f"   Output directory: {self.output_base_dir.absolute()}")
        print(f"   Agents ready: {self.analyst.name}, {self.producer.name}, {self.sound_designer.name}, {self.mix_engineer.name}")
    
    def generate_hardcore_track(self, config: BMadTrackConfig) -> BMadGenerationSession:
        """Generate a complete hardcore track with real files"""
        
        # Create session
        session_id = f"bmad_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        session_dir = self.output_base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        session = BMadGenerationSession(
            session_id=session_id,
            config=config,
            output_dir=session_dir
        )
        
        self.active_sessions[session_id] = session
        
        try:
            # Set random seed if provided
            if config.seed:
                random.seed(config.seed)
                np.random.seed(config.seed)
            
            print(f"\n🎵 Starting BMAD hardcore track generation")
            print(f"   Session: {session_id}")
            print(f"   Style: {config.style.name}")
            print(f"   BPM: {config.bpm}")
            print(f"   Length: {config.length_bars} bars")
            
            # Step 1: Music Analysis
            print(f"\n🔍 {self.analyst.role} ({self.analyst.name}) analyzing hardcore patterns...")
            analysis = self.analyst.analyze_hardcore_pattern_structure(config)
            session.generation_log.append(f"Analysis complete - {len(analysis)} pattern elements identified")
            
            # Step 2: MIDI Generation
            print(f"\n🎹 {self.producer.role} ({self.producer.name}) generating MIDI patterns...")
            midi_clips = self.producer.generate_hardcore_track_midi(config, analysis)
            session.midi_clips = midi_clips
            session.generation_log.append(f"MIDI generation complete - {len(midi_clips)} clips created")
            
            # Export MIDI files
            for name, clip in midi_clips.items():
                midi_path = session_dir / f"{name}.mid"
                if clip.save_midi_file(str(midi_path)):
                    session.generation_log.append(f"MIDI exported: {midi_path.name}")
                    print(f"   ✅ MIDI exported: {midi_path.name}")
            
            # Step 3: Audio Synthesis
            print(f"\n🎛️ {self.sound_designer.role} ({self.sound_designer.name}) synthesizing audio...")
            audio_clips = self.sound_designer.synthesize_track_audio(midi_clips, config)
            
            # Apply effects to each track
            for name, audio in audio_clips.items():
                audio_clips[name] = self.sound_designer.apply_hardcore_effects_chain(audio, name, config)
            
            session.audio_clips = audio_clips
            session.generation_log.append(f"Audio synthesis complete - {len(audio_clips)} audio tracks")
            
            # Step 4: Mixing and Mastering
            print(f"\n🎚️ {self.mix_engineer.role} ({self.mix_engineer.name}) creating final mix...")
            final_mix = self.mix_engineer.create_final_mix(audio_clips, config)
            session.final_track = final_mix
            
            # Export final WAV
            final_path = session_dir / f"{session_id}_final.wav"
            if self.mix_engineer.export_audio_file(final_mix, str(final_path)):
                session.generation_log.append(f"Final mix exported: {final_path.name}")
                print(f"   ✅ Final track exported: {final_path.name}")
            
            # Export individual tracks
            for name, audio in audio_clips.items():
                if len(audio) > 0:
                    track_path = session_dir / f"{name}_track.wav"
                    if self.mix_engineer.export_audio_file(audio, str(track_path)):
                        print(f"   ✅ Track exported: {track_path.name}")
            
            # Generate session report
            self._generate_session_report(session)
            
            print(f"\n🎉 BMAD track generation complete!")
            print(f"   Session ID: {session_id}")
            print(f"   Files in: {session_dir.name}")
            print(f"   Final track: {session_id}_final.wav")
            
            return session
            
        except Exception as e:
            session.generation_log.append(f"ERROR: {str(e)}")
            print(f"❌ Generation failed: {e}")
            raise
    
    def _generate_session_report(self, session: BMadGenerationSession):
        """Generate a report for the generation session"""
        
        report_path = session.output_dir / "session_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("BMAD HARDCORE MUSIC GENERATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Session ID: {session.session_id}\n")
            f.write(f"Generated: {session.created_at}\n")
            f.write(f"Style: {session.config.style.name}\n")
            f.write(f"BPM: {session.config.bpm}\n")
            f.write(f"Length: {session.config.length_bars} bars\n")
            f.write(f"Key: {session.config.key}\n\n")
            
            f.write("GENERATION LOG:\n")
            f.write("-" * 20 + "\n")
            for log_entry in session.generation_log:
                f.write(f"• {log_entry}\n")
            
            f.write("\nGENERATED FILES:\n")
            f.write("-" * 20 + "\n")
            for file_path in session.output_dir.glob("*"):
                if file_path.is_file() and file_path.name != "session_report.txt":
                    f.write(f"• {file_path.name}\n")
            
            f.write("\nBMAD AGENTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• {self.analyst.role} ({self.analyst.name}): {self.analyst.specialty}\n")
            f.write(f"• {self.producer.role} ({self.producer.name}): {self.producer.specialty}\n")
            f.write(f"• {self.sound_designer.role} ({self.sound_designer.name}): {self.sound_designer.specialty}\n")
            f.write(f"• {self.mix_engineer.role} ({self.mix_engineer.name}): {self.mix_engineer.specialty}\n")
    
    def generate_hardcore_collection(self, num_tracks: int = 5, 
                                   base_config: Optional[BMadTrackConfig] = None) -> List[BMadGenerationSession]:
        """Generate a collection of hardcore tracks"""
        
        if base_config is None:
            base_config = BMadTrackConfig()
        
        sessions = []
        
        print(f"\n🏭 BMAD Hardcore Music Factory - Generating {num_tracks} tracks")
        print("=" * 60)
        
        for i in range(num_tracks):
            print(f"\n📀 Track {i + 1}/{num_tracks}")
            
            # Create variation of base config
            config = BMadTrackConfig(
                style=random.choice(list(HardcoreStyle)),
                bpm=base_config.bpm + random.uniform(-10, 10),
                length_bars=base_config.length_bars + random.choice([-8, -4, 0, 4, 8]),
                key=random.choice(["A_minor", "E_minor", "C_minor"]),
                seed=random.randint(1000, 9999)
            )
            
            try:
                session = self.generate_hardcore_track(config)
                sessions.append(session)
            except Exception as e:
                print(f"❌ Failed to generate track {i + 1}: {e}")
                continue
        
        print(f"\n🎊 Collection complete! Generated {len(sessions)} tracks")
        print(f"   Output directory: {self.output_base_dir.absolute()}")
        
        return sessions
    
    def run_overnight_factory(self, target_tracks: int = 50):
        """Run overnight generation for extensive music creation"""
        
        print(f"\n🌙 BMAD Overnight Factory Mode - Target: {target_tracks} tracks")
        print("   This will run continuously generating hardcore music...")
        
        styles = list(HardcoreStyle)
        keys = ["A_minor", "E_minor", "C_minor", "D_minor"]
        bpm_ranges = {
            HardcoreStyle.ROTTERDAM_GABBER: (160, 180),
            HardcoreStyle.FRENCHCORE: (180, 220),
            HardcoreStyle.SPEEDCORE: (200, 300),
            HardcoreStyle.BERLIN_INDUSTRIAL: (130, 150),
            HardcoreStyle.UK_HARDCORE: (160, 180),
            HardcoreStyle.TERRORCORE: (250, 400)
        }
        
        generated_count = 0
        
        while generated_count < target_tracks:
            try:
                # Random configuration
                style = random.choice(styles)
                bpm_min, bpm_max = bpm_ranges[style]
                
                config = BMadTrackConfig(
                    style=style,
                    bpm=random.uniform(bpm_min, bpm_max),
                    length_bars=random.choice([16, 24, 32, 40]),
                    key=random.choice(keys),
                    seed=random.randint(1000, 99999)
                )
                
                session = self.generate_hardcore_track(config)
                generated_count += 1
                
                print(f"✅ Progress: {generated_count}/{target_tracks} tracks completed")
                
                # Brief pause between tracks
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Track generation failed: {e}")
                print("   Continuing with next track...")
                time.sleep(5)
        
        print(f"\n🎉 Overnight factory complete! Generated {generated_count} hardcore tracks")


# Convenience functions for quick generation
def generate_quick_gabber_track(bpm: float = 180.0, length_bars: float = 32.0) -> BMadGenerationSession:
    """Quick generation of a Rotterdam gabber track"""
    coordinator = RealBMADMusicCoordinator()
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=bpm,
        length_bars=length_bars,
        key="A_minor"
    )
    return coordinator.generate_hardcore_track(config)


def generate_quick_frenchcore_track(bpm: float = 200.0, length_bars: float = 24.0) -> BMadGenerationSession:
    """Quick generation of a frenchcore track"""
    coordinator = RealBMADMusicCoordinator()
    config = BMadTrackConfig(
        style=HardcoreStyle.FRENCHCORE,
        bpm=bpm,
        length_bars=length_bars,
        key="E_minor"
    )
    return coordinator.generate_hardcore_track(config)


def run_bmad_demo():
    """Run a demonstration of the BMAD music coordinator"""
    print("🎛️ BMAD Real Music Coordinator Demo")
    print("=" * 50)
    
    coordinator = RealBMADMusicCoordinator()
    
    # Generate a single track
    config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=180.0,
        length_bars=16.0,
        key="A_minor",
        seed=12345
    )
    
    session = coordinator.generate_hardcore_track(config)
    
    print(f"\n🎵 Demo complete!")
    print(f"   Generated session: {session.session_id}")
    print(f"   Check output in: {session.output_dir}")


if __name__ == "__main__":
    run_bmad_demo()