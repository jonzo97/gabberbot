"""
AudioService - MIDIClip to WAV Rendering Service.

Consolidates all audio synthesis logic into a single service that takes
MIDIClip objects and renders them to high-quality WAV files.
"""

import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass

from ..models.core import MIDIClip, MIDINote
from ..models.config import Settings
from ..audio.synthesis import HardcoreSynthesizer, KickSynthesizer, AcidSynthesizer


@dataclass
class AudioConfig:
    """Configuration for audio rendering."""
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 1  # Mono for hardcore music
    buffer_size: int = 512
    output_format: str = "WAV"


@dataclass
class RenderProgress:
    """Progress information during audio rendering."""
    stage: str
    progress: float  # 0.0 to 1.0
    message: str


class AudioService:
    """
    Service for rendering MIDIClip objects to high-quality WAV files.
    
    Consolidates all synthesis logic and provides the final step in the
    text → MIDI → audio pipeline.
    """
    
    def __init__(
        self, 
        settings: Settings,
        audio_config: Optional[AudioConfig] = None,
        progress_callback: Optional[Callable[[RenderProgress], None]] = None
    ):
        """Initialize AudioService with configuration."""
        self.settings = settings
        self.audio_config = audio_config or AudioConfig()
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        
        # Initialize synthesizers
        self.hardcore_synth = HardcoreSynthesizer(self.audio_config.sample_rate)
        self.kick_synth = KickSynthesizer(self.audio_config.sample_rate)
        self.acid_synth = AcidSynthesizer(self.audio_config.sample_rate)
        
        # Synthesis stats
        self.render_stats = {
            'total_renders': 0,
            'successful_renders': 0,
            'avg_render_time': 0.0,
            'total_duration_rendered': 0.0
        }
    
    def render_to_wav(
        self, 
        clip: MIDIClip, 
        output_path: Union[str, Path],
        audio_config: Optional[AudioConfig] = None
    ) -> Path:
        """
        Render a MIDIClip to a WAV file.
        
        Args:
            clip: MIDIClip object to render
            output_path: Path where WAV file should be saved
            audio_config: Optional audio configuration override
            
        Returns:
            Path to the generated WAV file
            
        Raises:
            AudioRenderError: If rendering fails
        """
        import time
        start_time = time.time()
        self.render_stats['total_renders'] += 1
        
        try:
            config = audio_config or self.audio_config
            output_path = Path(output_path)
            
            self._report_progress("initialization", 0.0, "Initializing audio rendering...")
            
            # Calculate audio duration
            audio_duration = self._calculate_audio_duration(clip)
            self.logger.info(f"Rendering {audio_duration:.2f}s of audio for clip '{clip.name}'")
            
            self._report_progress("synthesis", 0.2, "Synthesizing audio...")
            
            # Generate audio samples
            audio_samples = self._synthesize_clip(clip, config)
            
            self._report_progress("processing", 0.7, "Processing and mixing...")
            
            # Apply final processing
            audio_samples = self._apply_final_processing(audio_samples, clip, config)
            
            self._report_progress("export", 0.9, "Exporting WAV file...")
            
            # Export to WAV
            self._export_wav(audio_samples, output_path, config)
            
            # Update stats
            render_time = time.time() - start_time
            self._update_render_stats(render_time, audio_duration)
            
            self._report_progress("complete", 1.0, f"Audio rendered to {output_path}")
            self.logger.info(f"Successfully rendered audio to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to render audio: {e}")
            raise AudioRenderError(f"Audio rendering failed: {e}") from e
    
    def _synthesize_clip(self, clip: MIDIClip, config: AudioConfig) -> np.ndarray:
        """Synthesize audio for a complete MIDIClip."""
        # Calculate total samples needed
        duration_seconds = self._calculate_audio_duration(clip)
        total_samples = int(duration_seconds * config.sample_rate)
        
        # Initialize audio buffer
        audio_buffer = np.zeros(total_samples, dtype=np.float32)
        
        # Group notes by synthesis type based on pitch ranges and clip metadata
        note_groups = self._group_notes_by_synthesis_type(clip)
        
        # Synthesize each group with appropriate synthesizer
        for synth_type, notes in note_groups.items():
            if not notes:
                continue
                
            self.logger.debug(f"Synthesizing {len(notes)} notes with {synth_type}")
            
            if synth_type == "kick":
                samples = self._synthesize_kick_notes(notes, clip, config)
            elif synth_type == "acid":
                samples = self._synthesize_acid_notes(notes, clip, config)
            else:  # hardcore/general
                samples = self._synthesize_hardcore_notes(notes, clip, config)
            
            # Mix into main buffer
            if len(samples) <= len(audio_buffer):
                audio_buffer[:len(samples)] += samples
            else:
                # Truncate if samples are longer than buffer
                audio_buffer += samples[:len(audio_buffer)]
        
        return audio_buffer
    
    def _group_notes_by_synthesis_type(self, clip: MIDIClip) -> Dict[str, List[MIDINote]]:
        """Group notes by appropriate synthesis type based on pitch ranges."""
        groups = {"kick": [], "acid": [], "hardcore": []}
        
        for note in clip.notes:
            if note.pitch <= 40:  # Kick drum range
                groups["kick"].append(note)
            elif 40 < note.pitch <= 80:  # Bass range - could be acid
                # Check if clip suggests acid synthesis
                if "acid" in clip.name.lower() or "bass" in clip.name.lower():
                    groups["acid"].append(note)
                else:
                    groups["hardcore"].append(note)
            else:  # Higher range - leads, riffs
                groups["hardcore"].append(note)
        
        return groups
    
    def _synthesize_kick_notes(
        self, notes: List[MIDINote], clip: MIDIClip, config: AudioConfig
    ) -> np.ndarray:
        """Synthesize kick drum notes with hardcore characteristics."""
        duration_seconds = self._calculate_audio_duration(clip)
        total_samples = int(duration_seconds * config.sample_rate)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)
        
        for note in notes:
            # Convert time to samples
            start_sample = int(note.start_time * (60.0 / clip.bpm) * config.sample_rate)
            
            # Generate hardcore kick
            kick_samples = self.kick_synth.generate_kick(
                duration=note.duration * (60.0 / clip.bpm),
                velocity=note.velocity,
                pitch=note.pitch
            )
            
            # Add to buffer
            end_sample = start_sample + len(kick_samples)
            if end_sample <= total_samples:
                audio_buffer[start_sample:end_sample] += kick_samples
        
        return audio_buffer
    
    def _synthesize_acid_notes(
        self, notes: List[MIDINote], clip: MIDIClip, config: AudioConfig
    ) -> np.ndarray:
        """Synthesize acid bassline notes with TB-303 characteristics."""
        duration_seconds = self._calculate_audio_duration(clip)
        total_samples = int(duration_seconds * config.sample_rate)
        
        # Acid synthesis works best with continuous patterns
        return self.acid_synth.generate_sequence(
            notes=notes,
            bpm=clip.bpm,
            total_duration=duration_seconds
        )
    
    def _synthesize_hardcore_notes(
        self, notes: List[MIDINote], clip: MIDIClip, config: AudioConfig
    ) -> np.ndarray:
        """Synthesize general hardcore notes (riffs, leads, etc.)."""
        duration_seconds = self._calculate_audio_duration(clip)
        total_samples = int(duration_seconds * config.sample_rate)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)
        
        for note in notes:
            # Convert time to samples
            start_sample = int(note.start_time * (60.0 / clip.bpm) * config.sample_rate)
            note_duration = note.duration * (60.0 / clip.bpm)
            
            # Generate hardcore sound
            note_samples = self.hardcore_synth.generate_note(
                pitch=note.pitch,
                duration=note_duration,
                velocity=note.velocity
            )
            
            # Add to buffer
            end_sample = start_sample + len(note_samples)
            if end_sample <= total_samples:
                audio_buffer[start_sample:end_sample] += note_samples
        
        return audio_buffer
    
    def _apply_final_processing(
        self, audio: np.ndarray, clip: MIDIClip, config: AudioConfig
    ) -> np.ndarray:
        """Apply final processing and effects to the audio."""
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9  # Leave some headroom
        
        # Apply hardcore-style compression and limiting
        audio = self._apply_hardcore_processing(audio, clip)
        
        return audio
    
    def _apply_hardcore_processing(self, audio: np.ndarray, clip: MIDIClip) -> np.ndarray:
        """Apply hardcore-specific processing (compression, distortion, etc.)."""
        # Simple compression simulation
        threshold = 0.7
        ratio = 4.0
        
        # Apply compression
        compressed = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
            audio
        )
        
        # Light distortion for hardcore edge
        distorted = np.tanh(compressed * 1.2)
        
        return distorted
    
    def _export_wav(
        self, audio: np.ndarray, output_path: Path, config: AudioConfig
    ) -> None:
        """Export audio samples to WAV file."""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to appropriate bit depth
        if config.bit_depth == 16:
            audio_int = (audio * 32767).astype(np.int16)
        elif config.bit_depth == 24:
            audio_int = (audio * 8388607).astype(np.int32)
        else:  # 32-bit float
            audio_int = audio.astype(np.float32)
        
        # Write WAV file
        sf.write(
            output_path,
            audio_int,
            config.sample_rate,
            subtype=f'PCM_{config.bit_depth}' if config.bit_depth <= 24 else 'FLOAT'
        )
    
    def _calculate_audio_duration(self, clip: MIDIClip) -> float:
        """Calculate the total duration of audio to render in seconds."""
        if not clip.notes:
            return clip.length_bars * (60.0 / clip.bpm) * 4  # Assume 4/4 time
        
        # Find the latest note end time
        latest_end = max(note.start_time + note.duration for note in clip.notes)
        
        # Convert beats to seconds
        duration_seconds = latest_end * (60.0 / clip.bpm)
        
        # Ensure minimum duration based on clip length
        min_duration = clip.length_bars * (60.0 / clip.bpm) * 4
        
        return max(duration_seconds, min_duration)
    
    def _report_progress(self, stage: str, progress: float, message: str) -> None:
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(RenderProgress(stage, progress, message))
    
    def _update_render_stats(self, render_time: float, audio_duration: float) -> None:
        """Update rendering statistics."""
        self.render_stats['successful_renders'] += 1
        total = self.render_stats['total_renders']
        current_avg = self.render_stats['avg_render_time']
        
        # Update running average
        self.render_stats['avg_render_time'] = (
            (current_avg * (total - 1) + render_time) / total
        )
        self.render_stats['total_duration_rendered'] += audio_duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        return self.render_stats.copy()
    
    def is_available(self) -> bool:
        """Check if audio service is available."""
        return True  # Always available with software synthesis


class AudioRenderError(Exception):
    """Raised when audio rendering fails."""
    pass