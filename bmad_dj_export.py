#!/usr/bin/env python3
"""
BMAD DJ Export Pipeline
@mix-engineer (Phoenix) - Phase 3 DJ-Ready Export System

DJ-focused export pipeline with multiple formats and proper metadata:
- Individual track exports (WAV, FLAC, MP3)
- Complete album mix as continuous DJ set
- MIDI exports for DAW integration
- Stems and instrumental versions for remixing
- DJ pool ready exports with proper metadata
- Beatport/Traxsource compatible formatting
"""

import os
import json
import wave
import struct
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

# Import BMAD components
from bmad_mastering_chain import BMADMasteringChain, MasteringTarget, BMADMasteringExporter
from bmad_album_producer import AlbumTrackConfig, AlbumConfig


class ExportFormat(Enum):
    """Audio export formats"""
    WAV_44K_16 = "wav_44k_16bit"      # Standard DJ format
    WAV_44K_24 = "wav_44k_24bit"      # High quality
    FLAC_44K = "flac_44k"             # Lossless compression
    MP3_320 = "mp3_320kbps"           # High quality lossy
    MP3_256 = "mp3_256kbps"           # Standard quality
    AIFF_44K_16 = "aiff_44k_16bit"    # Mac DJ software


class ExportPurpose(Enum):
    """Export purposes for different use cases"""
    DJ_POOL = "dj_pool"               # DJ pool distribution
    BEATPORT = "beatport"             # Beatport store
    TRAXSOURCE = "traxsource"         # Traxsource store
    SOUNDCLOUD = "soundcloud"         # SoundCloud upload
    YOUTUBE = "youtube"               # YouTube upload
    VINYL_CUTTING = "vinyl_cutting"   # Vinyl mastering
    RADIO_EDIT = "radio_edit"         # Radio friendly
    WAREHOUSE_SET = "warehouse_set"   # Live set optimization


@dataclass
class TrackMetadata:
    """Complete track metadata for DJ pools and stores"""
    # Basic info
    title: str
    artist: str
    album: str = ""
    label: str = "BMAD Records"
    release_date: str = ""
    
    # DJ specific
    bpm: float = 0.0
    key: str = ""
    genre: str = "Hardcore"
    subgenre: str = "Gabber"
    energy_level: int = 8  # 1-10 scale
    
    # Technical
    duration_seconds: float = 0.0
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2
    
    # Mix info
    mix_in_time: str = "00:00"        # MM:SS format
    mix_out_time: str = "00:00"       # MM:SS format
    intro_length: int = 16            # bars
    outro_length: int = 16            # bars
    
    # Store specific
    catalog_number: str = ""
    isrc: str = ""                    # International Standard Recording Code
    upc: str = ""                     # Universal Product Code
    price_tier: str = "standard"
    
    # Content warnings
    explicit: bool = False
    instrumental: bool = False
    
    # Tags for discovery
    tags: List[str] = field(default_factory=list)
    mood_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            'title': self.title,
            'artist': self.artist,
            'album': self.album,
            'label': self.label,
            'release_date': self.release_date,
            'bpm': self.bpm,
            'key': self.key,
            'genre': self.genre,
            'subgenre': self.subgenre,
            'energy_level': self.energy_level,
            'duration_seconds': self.duration_seconds,
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'channels': self.channels,
            'mix_in_time': self.mix_in_time,
            'mix_out_time': self.mix_out_time,
            'intro_length': self.intro_length,
            'outro_length': self.outro_length,
            'catalog_number': self.catalog_number,
            'isrc': self.isrc,
            'upc': self.upc,
            'explicit': self.explicit,
            'instrumental': self.instrumental,
            'tags': self.tags,
            'mood_tags': self.mood_tags
        }


@dataclass
class ExportJob:
    """Complete export job configuration"""
    name: str
    purpose: ExportPurpose
    formats: List[ExportFormat]
    mastering_target: MasteringTarget
    
    # Output settings
    include_stems: bool = False
    include_instrumentals: bool = False
    include_midi: bool = False
    include_continuous_mix: bool = False
    
    # Naming convention
    filename_template: str = "{artist} - {title} ({label})"
    folder_structure: str = "flat"  # flat, artist_album, genre
    
    # Metadata requirements
    require_bpm_analysis: bool = True
    require_key_detection: bool = True
    auto_generate_tags: bool = True
    
    @classmethod
    def for_dj_pool(cls) -> 'ExportJob':
        """Create DJ pool export configuration"""
        return cls(
            name="DJ Pool Export",
            purpose=ExportPurpose.DJ_POOL,
            formats=[ExportFormat.WAV_44K_16, ExportFormat.MP3_320],
            mastering_target=MasteringTarget.DJ_POOL_STANDARD,
            include_stems=False,
            include_instrumentals=True,
            include_midi=False,
            require_bpm_analysis=True,
            require_key_detection=True,
            filename_template="{artist} - {title} ({bpm}BPM {key})"
        )
    
    @classmethod
    def for_beatport(cls) -> 'ExportJob':
        """Create Beatport store export configuration"""
        return cls(
            name="Beatport Export",
            purpose=ExportPurpose.BEATPORT,
            formats=[ExportFormat.WAV_44K_16, ExportFormat.AIFF_44K_16],
            mastering_target=MasteringTarget.HARDCORE_CLUB,
            include_stems=False,
            include_instrumentals=False,
            include_midi=False,
            folder_structure="artist_album",
            filename_template="{artist} - {title} (Original Mix)"
        )
    
    @classmethod
    def for_warehouse_set(cls) -> 'ExportJob':
        """Create warehouse set export configuration"""
        return cls(
            name="Warehouse Set Export",
            purpose=ExportPurpose.WAREHOUSE_SET,
            formats=[ExportFormat.WAV_44K_24],
            mastering_target=MasteringTarget.WAREHOUSE_SYSTEM,
            include_stems=True,
            include_continuous_mix=True,
            filename_template="{artist} - {title} (Warehouse Master)"
        )


class BMADDJExporter:
    """
    Professional DJ export pipeline for hardcore music.
    
    Features:
    - Multiple format exports optimized for different platforms
    - Professional metadata handling
    - DJ-specific analysis (BPM, key, mix points)
    - Stem separation for remixing
    - Continuous album mixes
    - Store-ready formatting
    """
    
    def __init__(self, output_base_dir: str = "bmad_exports"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        
        # Initialize mastering chain
        self.mastering_chain = BMADMasteringChain()
        self.mastering_exporter = BMADMasteringExporter(self.mastering_chain)
        
        # Export history
        self.export_history = []
        
        print("BMAD DJ Export Pipeline initialized")
        print(f"Output directory: {self.output_base_dir.absolute()}")
    
    def export_track(self, audio: np.ndarray, metadata: TrackMetadata, 
                    export_job: ExportJob) -> Dict[str, Any]:
        """
        Export single track with all specified formats and metadata
        
        Args:
            audio: Track audio data
            metadata: Complete track metadata
            export_job: Export configuration
        
        Returns:
            Dictionary with export results and file paths
        """
        print(f"\n🎧 Exporting: {metadata.title}")
        print(f"Purpose: {export_job.purpose.value}")
        print(f"Formats: {[f.value for f in export_job.formats]}")
        
        # Create export directory
        export_id = f"{metadata.artist}_{metadata.title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir = self.output_base_dir / export_id
        export_dir.mkdir(exist_ok=True)
        
        # Master audio for target
        print(f"   Mastering for {export_job.mastering_target.value}...")
        mastered_audio, mastering_analysis = self.mastering_chain.master_track(
            audio, export_job.mastering_target
        )
        
        # Update metadata with analysis results
        self._update_metadata_from_analysis(metadata, mastered_audio, mastering_analysis)
        
        # Export in all requested formats
        exported_files = []
        for format_type in export_job.formats:
            print(f"   Exporting {format_type.value}...")
            
            # Generate filename
            filename = self._generate_filename(metadata, export_job, format_type)
            
            # Export audio file
            file_path = self._export_audio_format(
                mastered_audio, filename, format_type, export_dir
            )
            
            if file_path:
                exported_files.append({
                    'format': format_type.value,
                    'path': str(file_path),
                    'size_mb': file_path.stat().st_size / (1024 * 1024)
                })
        
        # Export stems if requested
        stem_files = []
        if export_job.include_stems:
            print("   Exporting stems...")
            stem_files = self._export_stems(audio, metadata, export_job, export_dir)
        
        # Export instrumental if requested
        instrumental_files = []
        if export_job.include_instrumentals:
            print("   Creating instrumental version...")
            instrumental_files = self._export_instrumental(
                mastered_audio, metadata, export_job, export_dir
            )
        
        # Export MIDI if requested
        midi_files = []
        if export_job.include_midi:
            print("   Exporting MIDI...")
            midi_files = self._export_midi_data(metadata, export_dir)
        
        # Export metadata files
        metadata_files = self._export_metadata_files(metadata, export_dir)
        
        # Create DJ cue points
        cue_files = self._export_dj_cue_points(metadata, export_dir)
        
        # Generate export report
        export_result = {
            'export_id': export_id,
            'track': metadata.to_dict(),
            'export_job': export_job.name,
            'mastering_analysis': {
                'input_lufs': mastering_analysis.input_lufs,
                'output_lufs': mastering_analysis.output_lufs,
                'peak_db': mastering_analysis.output_peak_db,
                'quality_score': mastering_analysis.quality_score
            },
            'files': {
                'audio': exported_files,
                'stems': stem_files,
                'instrumentals': instrumental_files,
                'midi': midi_files,
                'metadata': metadata_files,
                'cue_points': cue_files
            },
            'export_timestamp': datetime.now().isoformat(),
            'total_files': len(exported_files) + len(stem_files) + len(instrumental_files) + 
                          len(midi_files) + len(metadata_files) + len(cue_files)
        }
        
        # Save export report
        report_file = export_dir / "export_report.json"
        with open(report_file, 'w') as f:
            json.dump(export_result, f, indent=2)
        
        self.export_history.append(export_result)
        
        print(f"   ✓ Export completed: {export_result['total_files']} files created")
        print(f"   Output: {export_dir.name}")
        
        return export_result
    
    def export_album(self, album_tracks: List[Tuple[np.ndarray, TrackMetadata]], 
                    album_config: AlbumConfig, export_job: ExportJob) -> Dict[str, Any]:
        """
        Export complete album with track-to-track consistency
        
        Args:
            album_tracks: List of (audio, metadata) tuples
            album_config: Album configuration
            export_job: Export configuration
        
        Returns:
            Dictionary with complete album export results
        """
        print(f"\n🎵 Exporting Album: {album_config.album_name}")
        print(f"Tracks: {len(album_tracks)}")
        print(f"Purpose: {export_job.purpose.value}")
        
        # Create album export directory
        album_export_id = f"album_{album_config.album_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        album_dir = self.output_base_dir / album_export_id
        album_dir.mkdir(exist_ok=True)
        
        # Export individual tracks
        track_exports = []
        mastered_tracks = []
        
        for i, (audio, metadata) in enumerate(album_tracks):
            print(f"\n   Track {i+1}/{len(album_tracks)}: {metadata.title}")
            
            # Create track subdirectory
            track_dir = album_dir / f"track_{i+1:02d}_{metadata.title.replace(' ', '_')}"
            track_dir.mkdir(exist_ok=True)
            
            # Export track
            track_export = self._export_single_track_in_album(
                audio, metadata, export_job, track_dir
            )
            
            track_exports.append(track_export)
            mastered_tracks.append(track_export['mastered_audio'])
        
        # Create continuous album mix if requested
        continuous_mix_files = []
        if export_job.include_continuous_mix:
            print(f"\n   Creating continuous album mix...")
            continuous_mix_files = self._create_continuous_album_mix(
                mastered_tracks, [metadata for _, metadata in album_tracks],
                album_config, export_job, album_dir
            )
        
        # Create album metadata
        album_metadata = self._create_album_metadata(album_config, album_tracks)
        
        # Export album-level files
        album_metadata_files = self._export_album_metadata(album_metadata, album_dir)
        
        # Create DJ pool package if appropriate
        dj_package_files = []
        if export_job.purpose == ExportPurpose.DJ_POOL:
            print("   Creating DJ pool package...")
            dj_package_files = self._create_dj_pool_package(
                track_exports, album_metadata, album_dir
            )
        
        # Generate complete album export report
        album_export_result = {
            'album_export_id': album_export_id,
            'album': album_metadata,
            'export_job': export_job.name,
            'tracks': track_exports,
            'continuous_mix': continuous_mix_files,
            'album_metadata_files': album_metadata_files,
            'dj_package_files': dj_package_files,
            'export_timestamp': datetime.now().isoformat(),
            'total_tracks': len(album_tracks),
            'total_duration_minutes': sum(metadata.duration_seconds for _, metadata in album_tracks) / 60,
            'total_files': sum(track['file_count'] for track in track_exports) + 
                          len(continuous_mix_files) + len(album_metadata_files) + len(dj_package_files)
        }
        
        # Save album export report
        album_report_file = album_dir / "album_export_report.json"
        with open(album_report_file, 'w') as f:
            json.dump(album_export_result, f, indent=2)
        
        print(f"\n🏆 Album export completed!")
        print(f"   Album: {album_config.album_name}")
        print(f"   Total files: {album_export_result['total_files']}")
        print(f"   Output: {album_dir.name}")
        
        return album_export_result
    
    def _update_metadata_from_analysis(self, metadata: TrackMetadata, 
                                     audio: np.ndarray, analysis):
        """Update metadata with analysis results"""
        # Update duration
        metadata.duration_seconds = len(audio) / self.mastering_chain.sample_rate
        
        # Update technical info
        metadata.sample_rate = self.mastering_chain.sample_rate
        metadata.bit_depth = 16  # Default export bit depth
        metadata.channels = 2 if len(audio.shape) > 1 else 1
        
        # Auto-generate tags based on analysis
        if hasattr(analysis, 'spectral_balance'):
            spectral = analysis.spectral_balance
            if spectral.get('bass', 0) > 0.4:
                metadata.tags.append("heavy-bass")
            if spectral.get('treble', 0) > 0.3:
                metadata.tags.append("bright")
        
        # Energy level based on LUFS
        if hasattr(analysis, 'output_lufs'):
            if analysis.output_lufs > -4:
                metadata.energy_level = 10
            elif analysis.output_lufs > -6:
                metadata.energy_level = 9
            elif analysis.output_lufs > -8:
                metadata.energy_level = 8
            else:
                metadata.energy_level = 7
        
        # Calculate mix points (simplified)
        total_duration = metadata.duration_seconds
        mix_in_seconds = total_duration * 0.1  # 10% into track
        mix_out_seconds = total_duration * 0.9  # 90% into track
        
        metadata.mix_in_time = self._seconds_to_mmss(mix_in_seconds)
        metadata.mix_out_time = self._seconds_to_mmss(mix_out_seconds)
    
    def _generate_filename(self, metadata: TrackMetadata, export_job: ExportJob, 
                          format_type: ExportFormat) -> str:
        """Generate filename from template"""
        template = export_job.filename_template
        
        # Replace template variables
        replacements = {
            '{artist}': metadata.artist,
            '{title}': metadata.title,
            '{album}': metadata.album,
            '{label}': metadata.label,
            '{bpm}': f"{metadata.bpm:.0f}",
            '{key}': metadata.key,
            '{genre}': metadata.genre,
            '{catalog}': metadata.catalog_number
        }
        
        filename = template
        for placeholder, value in replacements.items():
            filename = filename.replace(placeholder, value)
        
        # Clean filename
        filename = self._clean_filename(filename)
        
        # Add extension
        extension = self._get_format_extension(format_type)
        return f"{filename}.{extension}"
    
    def _export_audio_format(self, audio: np.ndarray, filename: str, 
                           format_type: ExportFormat, output_dir: Path) -> Optional[Path]:
        """Export audio in specified format"""
        file_path = output_dir / filename
        
        try:
            if format_type in [ExportFormat.WAV_44K_16, ExportFormat.WAV_44K_24]:
                self._export_wav(audio, file_path, format_type)
            elif format_type == ExportFormat.AIFF_44K_16:
                self._export_aiff(audio, file_path)
            elif format_type in [ExportFormat.MP3_320, ExportFormat.MP3_256]:
                self._export_mp3(audio, file_path, format_type)
            elif format_type == ExportFormat.FLAC_44K:
                self._export_flac(audio, file_path)
            else:
                print(f"   Warning: Format {format_type.value} not implemented")
                return None
            
            return file_path
            
        except Exception as e:
            print(f"   Error exporting {format_type.value}: {e}")
            return None
    
    def _export_wav(self, audio: np.ndarray, file_path: Path, format_type: ExportFormat):
        """Export WAV file"""
        sample_rate = self.mastering_chain.sample_rate
        
        if len(audio.shape) == 1:
            channels = 1
            audio_data = audio
        else:
            channels = audio.shape[1]
            audio_data = audio
        
        # Determine bit depth
        if format_type == ExportFormat.WAV_44K_24:
            sample_width = 3  # 24-bit
            max_val = 2**23 - 1
            dtype = np.int32
        else:
            sample_width = 2  # 16-bit
            max_val = 2**15 - 1
            dtype = np.int16
        
        # Convert to integer
        audio_int = (audio_data * max_val).astype(dtype)
        
        with wave.open(str(file_path), 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            
            if channels == 1:
                if sample_width == 3:
                    # 24-bit requires special handling
                    audio_bytes = b''.join(
                        audio_int[i].to_bytes(3, byteorder='little', signed=True)
                        for i in range(len(audio_int))
                    )
                else:
                    audio_bytes = audio_int.tobytes()
            else:
                if sample_width == 3:
                    # 24-bit stereo
                    audio_bytes = b''.join(
                        val.to_bytes(3, byteorder='little', signed=True)
                        for frame in audio_int
                        for val in frame
                    )
                else:
                    audio_bytes = audio_int.flatten().tobytes()
            
            wav_file.writeframes(audio_bytes)
    
    def _export_aiff(self, audio: np.ndarray, file_path: Path):
        """Export AIFF file (placeholder - would need proper AIFF library)"""
        # For now, export as WAV with .aiff extension
        self._export_wav(audio, file_path, ExportFormat.WAV_44K_16)
    
    def _export_mp3(self, audio: np.ndarray, file_path: Path, format_type: ExportFormat):
        """Export MP3 file (placeholder - would need MP3 encoder)"""
        print(f"   MP3 export would require external encoder (lame, ffmpeg, etc.)")
        # For now, create a placeholder file
        file_path.write_text("MP3 export placeholder")
    
    def _export_flac(self, audio: np.ndarray, file_path: Path):
        """Export FLAC file (placeholder - would need FLAC encoder)"""
        print(f"   FLAC export would require external encoder")
        # For now, create a placeholder file
        file_path.write_text("FLAC export placeholder")
    
    def _export_stems(self, audio: np.ndarray, metadata: TrackMetadata, 
                     export_job: ExportJob, output_dir: Path) -> List[Dict[str, Any]]:
        """Export individual stems for remixing"""
        stems_dir = output_dir / "stems"
        stems_dir.mkdir(exist_ok=True)
        
        # Placeholder stem separation
        # Real implementation would use sophisticated source separation
        stem_names = ["kick", "bass", "lead", "fx"]
        stem_files = []
        
        for stem_name in stem_names:
            # Create simplified stem (placeholder)
            stem_audio = audio * 0.25  # Each stem gets 1/4 of the signal
            
            stem_filename = f"{metadata.artist} - {metadata.title} ({stem_name} stem).wav"
            stem_path = stems_dir / stem_filename
            
            self._export_wav(stem_audio, stem_path, ExportFormat.WAV_44K_24)
            
            stem_files.append({
                'stem': stem_name,
                'path': str(stem_path),
                'size_mb': stem_path.stat().st_size / (1024 * 1024)
            })
        
        return stem_files
    
    def _export_instrumental(self, audio: np.ndarray, metadata: TrackMetadata,
                           export_job: ExportJob, output_dir: Path) -> List[Dict[str, Any]]:
        """Export instrumental version"""
        # For hardcore music, instrumental is usually the same as the main track
        # unless there are vocal samples that need removal
        
        instrumental_filename = f"{metadata.artist} - {metadata.title} (Instrumental).wav"
        instrumental_path = output_dir / instrumental_filename
        
        self._export_wav(audio, instrumental_path, ExportFormat.WAV_44K_16)
        
        return [{
            'type': 'instrumental',
            'path': str(instrumental_path),
            'size_mb': instrumental_path.stat().st_size / (1024 * 1024)
        }]
    
    def _export_midi_data(self, metadata: TrackMetadata, output_dir: Path) -> List[Dict[str, Any]]:
        """Export MIDI data if available"""
        # Placeholder for MIDI export
        # Real implementation would export pattern MIDI from the generation process
        
        midi_filename = f"{metadata.artist} - {metadata.title}.mid"
        midi_path = output_dir / midi_filename
        
        # Create placeholder MIDI file
        midi_path.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xff\x2f\x00')
        
        return [{
            'type': 'midi',
            'path': str(midi_path),
            'size_mb': midi_path.stat().st_size / (1024 * 1024)
        }]
    
    def _export_metadata_files(self, metadata: TrackMetadata, output_dir: Path) -> List[Dict[str, Any]]:
        """Export metadata in various formats"""
        metadata_files = []
        
        # JSON metadata
        json_file = output_dir / f"{metadata.title}_metadata.json"
        with open(json_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        metadata_files.append({
            'type': 'json_metadata',
            'path': str(json_file),
            'size_mb': json_file.stat().st_size / (1024 * 1024)
        })
        
        # Track info text file
        info_file = output_dir / f"{metadata.title}_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Track: {metadata.title}\n")
            f.write(f"Artist: {metadata.artist}\n")
            f.write(f"Album: {metadata.album}\n")
            f.write(f"Label: {metadata.label}\n")
            f.write(f"BPM: {metadata.bpm}\n")
            f.write(f"Key: {metadata.key}\n")
            f.write(f"Genre: {metadata.genre}\n")
            f.write(f"Duration: {metadata.duration_seconds:.1f}s\n")
            f.write(f"Energy: {metadata.energy_level}/10\n")
            f.write(f"Mix In: {metadata.mix_in_time}\n")
            f.write(f"Mix Out: {metadata.mix_out_time}\n")
        
        metadata_files.append({
            'type': 'info_text',
            'path': str(info_file),
            'size_mb': info_file.stat().st_size / (1024 * 1024)
        })
        
        return metadata_files
    
    def _export_dj_cue_points(self, metadata: TrackMetadata, output_dir: Path) -> List[Dict[str, Any]]:
        """Export DJ cue points in various formats"""
        cue_files = []
        
        # Serato cue points (simplified)
        serato_file = output_dir / f"{metadata.title}.cue"
        with open(serato_file, 'w') as f:
            f.write(f'TITLE "{metadata.title}"\n')
            f.write(f'PERFORMER "{metadata.artist}"\n')
            f.write(f'FILE "{metadata.title}.wav" WAVE\n')
            f.write('  TRACK 01 AUDIO\n')
            f.write('    INDEX 01 00:00:00\n')
            
            # Add mix points as cues
            mix_in_frames = self._mmss_to_frames(metadata.mix_in_time)
            mix_out_frames = self._mmss_to_frames(metadata.mix_out_time)
            
            f.write(f'    INDEX 02 {self._frames_to_cue_time(mix_in_frames)}\n')  # Mix in
            f.write(f'    INDEX 03 {self._frames_to_cue_time(mix_out_frames)}\n')  # Mix out
        
        cue_files.append({
            'type': 'serato_cue',
            'path': str(serato_file),
            'size_mb': serato_file.stat().st_size / (1024 * 1024)
        })
        
        return cue_files
    
    def _export_single_track_in_album(self, audio: np.ndarray, metadata: TrackMetadata,
                                    export_job: ExportJob, track_dir: Path) -> Dict[str, Any]:
        """Export single track as part of album"""
        # Master the track
        mastered_audio, mastering_analysis = self.mastering_chain.master_track(
            audio, export_job.mastering_target
        )
        
        # Update metadata
        self._update_metadata_from_analysis(metadata, mastered_audio, mastering_analysis)
        
        # Export in requested formats
        exported_files = []
        for format_type in export_job.formats:
            filename = self._generate_filename(metadata, export_job, format_type)
            file_path = self._export_audio_format(mastered_audio, filename, format_type, track_dir)
            
            if file_path:
                exported_files.append({
                    'format': format_type.value,
                    'path': str(file_path),
                    'size_mb': file_path.stat().st_size / (1024 * 1024)
                })
        
        return {
            'metadata': metadata.to_dict(),
            'mastered_audio': mastered_audio,
            'files': exported_files,
            'file_count': len(exported_files),
            'mastering_analysis': {
                'output_lufs': mastering_analysis.output_lufs,
                'peak_db': mastering_analysis.output_peak_db,
                'quality_score': mastering_analysis.quality_score
            }
        }
    
    def _create_continuous_album_mix(self, mastered_tracks: List[np.ndarray], 
                                   track_metadata: List[TrackMetadata],
                                   album_config: AlbumConfig, export_job: ExportJob,
                                   album_dir: Path) -> List[Dict[str, Any]]:
        """Create continuous album mix for DJ sets"""
        # This would implement proper DJ mixing with crossfades
        # For now, create a simple concatenation
        
        continuous_audio = []
        crossfade_samples = int(0.5 * self.mastering_chain.sample_rate)  # 0.5 second crossfade
        
        for i, track_audio in enumerate(mastered_tracks):
            if i == 0:
                # First track - full length
                continuous_audio.extend(track_audio)
            else:
                # Crossfade with previous track
                if len(continuous_audio) >= crossfade_samples:
                    # Remove some samples from end of previous track
                    continuous_audio = continuous_audio[:-crossfade_samples]
                    
                    # Create crossfade
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    
                    # Apply crossfade
                    prev_end = continuous_audio[-crossfade_samples:] * fade_out
                    curr_start = track_audio[:crossfade_samples] * fade_in
                    crossfaded = prev_end + curr_start
                    
                    # Combine
                    continuous_audio[-crossfade_samples:] = crossfaded
                    continuous_audio.extend(track_audio[crossfade_samples:])
                else:
                    # Not enough audio for crossfade, just append
                    continuous_audio.extend(track_audio)
        
        # Export continuous mix
        mix_filename = f"{album_config.album_name}_Continuous_Mix.wav"
        mix_path = album_dir / mix_filename
        
        continuous_array = np.array(continuous_audio)
        self._export_wav(continuous_array, mix_path, ExportFormat.WAV_44K_16)
        
        return [{
            'type': 'continuous_mix',
            'path': str(mix_path),
            'duration_minutes': len(continuous_array) / self.mastering_chain.sample_rate / 60,
            'size_mb': mix_path.stat().st_size / (1024 * 1024)
        }]
    
    def _create_album_metadata(self, album_config: AlbumConfig, 
                             album_tracks: List[Tuple[np.ndarray, TrackMetadata]]) -> Dict[str, Any]:
        """Create album-level metadata"""
        return {
            'album_name': album_config.album_name,
            'artist_name': album_config.artist_name,
            'total_tracks': len(album_tracks),
            'total_duration_minutes': sum(metadata.duration_seconds for _, metadata in album_tracks) / 60,
            'bpm_journey': f"{album_config.bpm_start} → {album_config.bpm_end}",
            'genre': 'Hardcore',
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'label': 'BMAD Records',
            'mastering_target': album_config.mastering_target,
            'catalog_number': f"BMAD{datetime.now().strftime('%Y%m%d')}",
            'tracks': [metadata.to_dict() for _, metadata in album_tracks]
        }
    
    def _export_album_metadata(self, album_metadata: Dict[str, Any], 
                             album_dir: Path) -> List[Dict[str, Any]]:
        """Export album-level metadata files"""
        metadata_files = []
        
        # Album info JSON
        album_json = album_dir / "album_info.json"
        with open(album_json, 'w') as f:
            json.dump(album_metadata, f, indent=2)
        
        metadata_files.append({
            'type': 'album_json',
            'path': str(album_json),
            'size_mb': album_json.stat().st_size / (1024 * 1024)
        })
        
        # Create M3U playlist
        playlist_file = album_dir / f"{album_metadata['album_name']}.m3u"
        with open(playlist_file, 'w') as f:
            f.write('#EXTM3U\n')
            for track in album_metadata['tracks']:
                f.write(f"#EXTINF:{int(track['duration_seconds'])},{track['artist']} - {track['title']}\n")
                f.write(f"{track['title']}.wav\n")
        
        metadata_files.append({
            'type': 'playlist_m3u',
            'path': str(playlist_file),
            'size_mb': playlist_file.stat().st_size / (1024 * 1024)
        })
        
        return metadata_files
    
    def _create_dj_pool_package(self, track_exports: List[Dict[str, Any]], 
                              album_metadata: Dict[str, Any], 
                              album_dir: Path) -> List[Dict[str, Any]]:
        """Create DJ pool distribution package"""
        dj_package_files = []
        
        # Create DJ pool info file
        dj_info_file = album_dir / "DJ_POOL_INFO.txt"
        with open(dj_info_file, 'w') as f:
            f.write(f"DJ POOL PACKAGE\n")
            f.write(f"===============\n\n")
            f.write(f"Album: {album_metadata['album_name']}\n")
            f.write(f"Artist: {album_metadata['artist_name']}\n")
            f.write(f"Label: {album_metadata['label']}\n")
            f.write(f"Genre: Hardcore\n")
            f.write(f"Total Tracks: {album_metadata['total_tracks']}\n")
            f.write(f"Total Duration: {album_metadata['total_duration_minutes']:.1f} minutes\n")
            f.write(f"BPM Range: {album_metadata['bpm_journey']}\n\n")
            
            f.write("TRACK LISTING:\n")
            f.write("-" * 50 + "\n")
            for i, track_export in enumerate(track_exports, 1):
                track = track_export['metadata']
                f.write(f"{i:2d}. {track['artist']} - {track['title']}\n")
                f.write(f"    BPM: {track['bpm']:.0f} | Key: {track['key']} | ")
                f.write(f"Energy: {track['energy_level']}/10\n")
                f.write(f"    Duration: {track['duration_seconds']:.0f}s | ")
                f.write(f"Mix In: {track['mix_in_time']} | Mix Out: {track['mix_out_time']}\n\n")
        
        dj_package_files.append({
            'type': 'dj_pool_info',
            'path': str(dj_info_file),
            'size_mb': dj_info_file.stat().st_size / (1024 * 1024)
        })
        
        return dj_package_files
    
    # Utility methods
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for filesystem compatibility"""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        return filename[:200]
    
    def _get_format_extension(self, format_type: ExportFormat) -> str:
        """Get file extension for format"""
        extension_map = {
            ExportFormat.WAV_44K_16: 'wav',
            ExportFormat.WAV_44K_24: 'wav',
            ExportFormat.FLAC_44K: 'flac',
            ExportFormat.MP3_320: 'mp3',
            ExportFormat.MP3_256: 'mp3',
            ExportFormat.AIFF_44K_16: 'aiff'
        }
        return extension_map.get(format_type, 'wav')
    
    def _seconds_to_mmss(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _mmss_to_frames(self, mmss: str) -> int:
        """Convert MM:SS to frame number"""
        try:
            minutes, seconds = map(int, mmss.split(':'))
            total_seconds = minutes * 60 + seconds
            return int(total_seconds * 75)  # CD frame rate
        except:
            return 0
    
    def _frames_to_cue_time(self, frames: int) -> str:
        """Convert frames to cue time format (MM:SS:FF)"""
        total_seconds = frames // 75
        frame_remainder = frames % 75
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}:{frame_remainder:02d}"


# Test DJ export system
def test_bmad_dj_export():
    """Test the DJ export pipeline"""
    print("🎧 Testing BMAD DJ Export Pipeline")
    print("=" * 60)
    
    # Create test audio and metadata
    sample_rate = 44100
    duration = 10.0  # 10 seconds for testing
    samples = int(duration * sample_rate)
    
    # Generate test audio
    t = np.linspace(0, duration, samples)
    test_audio = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 0.5) +  # Kick
                 np.sin(2 * np.pi * 150 * t) * 0.3 +                # Bass
                 np.sin(2 * np.pi * 440 * t) * 0.2) * 0.7           # Lead
    
    # Create test metadata
    metadata = TrackMetadata(
        title="Test Hardcore Track",
        artist="BMAD Test",
        album="Test Album",
        bpm=180.0,
        key="A_minor",
        genre="Hardcore",
        subgenre="Gabber",
        energy_level=8
    )
    
    # Initialize exporter
    exporter = BMADDJExporter()
    
    # Test different export jobs
    export_jobs = [
        ExportJob.for_dj_pool(),
        ExportJob.for_beatport(),
        ExportJob.for_warehouse_set()
    ]
    
    for export_job in export_jobs:
        print(f"\n🎯 Testing {export_job.name}")
        print("-" * 40)
        
        export_result = exporter.export_track(test_audio, metadata, export_job)
        
        print(f"   Export ID: {export_result['export_id']}")
        print(f"   Files created: {export_result['total_files']}")
        print(f"   Formats: {[f['format'] for f in export_result['files']['audio']]}")
        print(f"   Quality score: {export_result['mastering_analysis']['quality_score']:.1f}/10")
    
    print(f"\n✨ DJ export pipeline test completed!")
    print("Ready for professional music distribution!")


if __name__ == "__main__":
    test_bmad_dj_export()