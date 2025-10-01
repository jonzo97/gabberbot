#!/usr/bin/env python3
"""
MIDI Clip-Based AI Tools for Hardcore Music Production

Unified tools that work with standard MIDI clips instead of scattered specific tools.
Provides maximum reusability and composability for AI conversation.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from ..models.midi_clips import MIDIClip, TriggerClip, MIDINote, Trigger
from ..generators.acid_bassline import AcidBasslineGenerator, create_hardcore_acid_line
from ..generators.tuned_kick import TunedKickGenerator, create_frenchcore_kicks
from ..interfaces.synthesizer import AbstractSynthesizer


@dataclass
class ClipToolResult:
    """Result of a clip tool execution"""
    success: bool
    clip: Optional[Union[MIDIClip, TriggerClip]] = None
    data: Any = None
    error: str = None
    audio_path: Optional[str] = None
    tidal_pattern: Optional[str] = None
    

class ClipType(Enum):
    """Types of clips that can be generated"""
    ACID_BASSLINE = "acid_bassline"
    RIFF = "riff"
    CHORD_PROGRESSION = "chord_progression" 
    ARPEGGIO = "arpeggio"
    TUNED_KICK = "tuned_kick"
    DRUM_PATTERN = "drum_pattern"
    KICK_PATTERN = "kick_pattern"
    HIHAT_PATTERN = "hihat_pattern"


class CreateMIDIClipTool:
    """
    Unified tool for creating any type of melodic MIDI clip
    
    Replaces separate tools for riffs, basslines, chords, arpeggios, etc.
    Uses pattern generators under the hood.
    """
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.name = "create_midi_clip"
        self.description = "Create melodic MIDI clips (acid basslines, riffs, chord progressions, arpeggios, tuned kicks)"
        self.synthesizer = synthesizer
        
        # Initialize available generators
        self.generators = {
            ClipType.ACID_BASSLINE: self._create_acid_bassline,
            ClipType.TUNED_KICK: self._create_tuned_kick,
            # TODO: Add more generators as they're built
            # ClipType.RIFF: self._create_riff,
            # ClipType.CHORD_PROGRESSION: self._create_chord_progression,
            # ClipType.ARPEGGIO: self._create_arpeggio,
        }
    
    async def execute(self, clip_type: str, length_bars: float = 4.0, bpm: float = 180.0,
                     **parameters) -> ClipToolResult:
        """Create MIDI clip of specified type"""
        try:
            # Parse clip type
            try:
                clip_enum = ClipType(clip_type.lower())
            except ValueError:
                return ClipToolResult(
                    success=False,
                    error=f"Unknown clip type: {clip_type}. Available: {[t.value for t in ClipType]}"
                )
            
            # Check if generator available
            if clip_enum not in self.generators:
                return ClipToolResult(
                    success=False,
                    error=f"Generator for {clip_type} not yet implemented"
                )
            
            # Generate clip
            generator_func = self.generators[clip_enum]
            clip = generator_func(length_bars, bpm, **parameters)
            
            # Convert to TidalCycles pattern
            tidal_pattern = clip.to_tidal_pattern()
            
            # Generate audio if possible
            audio_path = None
            try:
                if hasattr(self.synthesizer, 'generate_audio'):
                    # Convert MIDI clip to HardcorePattern for synthesis
                    from ..models.hardcore_models import HardcorePattern, SynthType
                    pattern = HardcorePattern(
                        name=clip.name,
                        bpm=bpm,
                        pattern_data=clip.to_pattern_string() if hasattr(clip, 'to_pattern_string') else tidal_pattern,
                        synth_type=SynthType.ACID_BASS if 'acid' in clip_type else SynthType.GABBER_KICK,
                        genre="hardcore"
                    )
                    
                    audio_data = await self.synthesizer.generate_audio(pattern, duration=8.0)
                    
                    # Save audio
                    audio_path = f"./exports/{clip.name}_{int(time.time())}.wav"
                    await self.synthesizer.save_audio(audio_data, audio_path)
            except Exception as audio_error:
                # Audio generation failed, but clip creation succeeded
                pass
            
            return ClipToolResult(
                success=True,
                clip=clip,
                data={
                    "name": clip.name,
                    "type": clip_type,
                    "length_bars": length_bars,
                    "bpm": bpm,
                    "notes": len(clip.notes) if hasattr(clip, 'notes') else 0,
                    "parameters": parameters
                },
                audio_path=audio_path,
                tidal_pattern=tidal_pattern
            )
            
        except Exception as e:
            return ClipToolResult(success=False, error=str(e))
    
    def _create_acid_bassline(self, length_bars: float, bpm: float, **params) -> MIDIClip:
        """Create acid bassline using generator"""
        scale = params.get('scale', 'A_minor')
        accent_pattern = params.get('accent_pattern', 'x ~ ~ x ~ ~ x ~')
        style = params.get('style', 'classic')
        
        if style == 'hardcore':
            return create_hardcore_acid_line(length_bars, bpm)
        else:
            generator = AcidBasslineGenerator(scale=scale, accent_pattern=accent_pattern)
            return generator.generate(length_bars, bpm)
    
    def _create_tuned_kick(self, length_bars: float, bpm: float, **params) -> MIDIClip:
        """Create tuned kick pattern using generator"""
        root_note = params.get('root_note', 'C1')
        pattern = params.get('pattern', 'x ~ x ~ x ~ x ~')
        tuning = params.get('tuning', 'pentatonic')
        style = params.get('style', 'frenchcore')
        
        if style == 'frenchcore':
            return create_frenchcore_kicks(root_note, length_bars, bpm)
        else:
            generator = TunedKickGenerator(root_note=root_note, pattern=pattern, tuning=tuning)
            return generator.generate(length_bars, bpm)


class CreateTriggerClipTool:
    """
    Unified tool for creating trigger-based clips (drums, percussion)
    """
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.name = "create_trigger_clip"
        self.description = "Create trigger-based clips for drums and percussion"
        self.synthesizer = synthesizer
    
    async def execute(self, pattern: str = "x ~ x ~ x ~ x ~", 
                     sample_id: str = "kick", length_bars: float = 1.0,
                     bpm: float = 180.0, **parameters) -> ClipToolResult:
        """Create trigger clip with specified pattern"""
        try:
            # Create trigger clip
            clip = TriggerClip(
                name=f"{sample_id}_pattern_{int(time.time())}",
                length_bars=length_bars,
                bpm=bpm
            )
            
            # Parse pattern and add triggers
            pattern_clean = pattern.replace(" ", "")
            step_size = 0.25  # 16th notes
            
            for i, char in enumerate(pattern_clean):
                if char == 'x' or char == 'X':
                    trigger = Trigger(
                        sample_id=sample_id,
                        velocity=127 if char == 'X' else 100,  # Capital X = accent
                        start_time=i * step_size
                    )
                    clip.add_trigger(trigger)
            
            # Convert to TidalCycles
            tidal_pattern = clip.to_tidal_pattern()
            
            return ClipToolResult(
                success=True,
                clip=clip,
                data={
                    "name": clip.name,
                    "pattern": pattern,
                    "sample_id": sample_id,
                    "triggers": len(clip.triggers)
                },
                tidal_pattern=tidal_pattern
            )
            
        except Exception as e:
            return ClipToolResult(success=False, error=str(e))


class ManipulateClipTool:
    """
    Tool for manipulating existing clips (transpose, quantize, etc.)
    """
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.name = "manipulate_clip"
        self.description = "Manipulate existing clips (transpose, quantize, modify)"
        self.synthesizer = synthesizer
        self.session_clips: Dict[str, Union[MIDIClip, TriggerClip]] = {}
    
    async def execute(self, operation: str, clip_name: Optional[str] = None,
                     current_clip: Optional[Union[MIDIClip, TriggerClip]] = None,
                     **parameters) -> ClipToolResult:
        """Manipulate clip with specified operation"""
        try:
            # Get clip to operate on
            clip = current_clip
            if clip_name and clip_name in self.session_clips:
                clip = self.session_clips[clip_name]
            
            if clip is None:
                return ClipToolResult(
                    success=False,
                    error="No clip specified or found. Create a clip first."
                )
            
            # Apply operation
            if operation == "transpose":
                semitones = parameters.get('semitones', 0)
                if hasattr(clip, 'transpose'):
                    modified_clip = clip.transpose(semitones)
                    return ClipToolResult(
                        success=True,
                        clip=modified_clip,
                        data={"operation": "transpose", "semitones": semitones}
                    )
            
            elif operation == "quantize":
                grid = parameters.get('grid', 0.25)
                if hasattr(clip, 'quantize'):
                    clip.quantize(grid)
                    return ClipToolResult(
                        success=True,
                        clip=clip,
                        data={"operation": "quantize", "grid": grid}
                    )
            
            else:
                return ClipToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            return ClipToolResult(success=False, error=str(e))
    
    def store_clip(self, name: str, clip: Union[MIDIClip, TriggerClip]):
        """Store clip in session for later manipulation"""
        self.session_clips[name] = clip


class ExportClipTool:
    """
    Tool for exporting clips to different formats
    """
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.name = "export_clip"
        self.description = "Export clips to MIDI files, TidalCycles patterns, or audio"
        self.synthesizer = synthesizer
    
    async def execute(self, export_format: str, filename: Optional[str] = None,
                     clip: Optional[Union[MIDIClip, TriggerClip]] = None,
                     **parameters) -> ClipToolResult:
        """Export clip to specified format"""
        try:
            if clip is None:
                return ClipToolResult(
                    success=False,
                    error="No clip provided for export"
                )
            
            if filename is None:
                filename = f"{clip.name}_export_{int(time.time())}"
            
            if export_format.lower() == "midi":
                # Export to MIDI file
                if hasattr(clip, 'save_midi_file'):
                    filepath = f"./exports/{filename}.mid"
                    success = clip.save_midi_file(filepath)
                    
                    return ClipToolResult(
                        success=success,
                        data={"format": "midi", "filepath": filepath}
                    )
            
            elif export_format.lower() == "tidal":
                # Export TidalCycles pattern
                pattern = clip.to_tidal_pattern()
                filepath = f"./exports/{filename}.tidal"
                
                with open(filepath, 'w') as f:
                    f.write(pattern)
                
                return ClipToolResult(
                    success=True,
                    data={"format": "tidal", "filepath": filepath, "pattern": pattern}
                )
            
            else:
                return ClipToolResult(
                    success=False,
                    error=f"Unknown export format: {export_format}"
                )
                
        except Exception as e:
            return ClipToolResult(success=False, error=str(e))


class LoadClipTool:
    """
    Tool for loading clips from library or files
    """
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.name = "load_clip"
        self.description = "Load clips from library or MIDI files"
        self.synthesizer = synthesizer
        self.library_path = "./library/clips/"
        os.makedirs(self.library_path, exist_ok=True)
    
    async def execute(self, source: str, name: Optional[str] = None,
                     **parameters) -> ClipToolResult:
        """Load clip from source"""
        try:
            if source == "library":
                # Load from library
                if not name:
                    return ClipToolResult(
                        success=False,
                        error="Clip name required for library loading"
                    )
                
                filepath = os.path.join(self.library_path, f"{name}.json")
                if not os.path.exists(filepath):
                    return ClipToolResult(
                        success=False,
                        error=f"Clip '{name}' not found in library"
                    )
                
                with open(filepath, 'r') as f:
                    clip_data = json.load(f)
                
                # Determine clip type and reconstruct
                if 'notes' in clip_data:
                    clip = MIDIClip.from_dict(clip_data)
                else:
                    clip = TriggerClip.from_dict(clip_data)
                
                return ClipToolResult(
                    success=True,
                    clip=clip,
                    data={"source": "library", "name": name}
                )
            
            else:
                return ClipToolResult(
                    success=False,
                    error=f"Unknown source: {source}"
                )
                
        except Exception as e:
            return ClipToolResult(success=False, error=str(e))


class SaveClipTool:
    """
    Tool for saving clips to library
    """
    
    def __init__(self, synthesizer: AbstractSynthesizer):
        self.name = "save_clip"
        self.description = "Save clips to library for later use"
        self.synthesizer = synthesizer
        self.library_path = "./library/clips/"
        os.makedirs(self.library_path, exist_ok=True)
    
    async def execute(self, name: str, clip: Optional[Union[MIDIClip, TriggerClip]] = None,
                     tags: List[str] = None, **parameters) -> ClipToolResult:
        """Save clip to library"""
        try:
            if clip is None:
                return ClipToolResult(
                    success=False,
                    error="No clip provided for saving"
                )
            
            # Add tags if provided
            if tags:
                clip.tags.extend(tags)
            
            # Save to library
            clip_data = clip.to_dict()
            filepath = os.path.join(self.library_path, f"{name}.json")
            
            with open(filepath, 'w') as f:
                json.dump(clip_data, f, indent=2)
            
            return ClipToolResult(
                success=True,
                data={
                    "name": name,
                    "filepath": filepath,
                    "type": "MIDIClip" if hasattr(clip, 'notes') else "TriggerClip"
                }
            )
            
        except Exception as e:
            return ClipToolResult(success=False, error=str(e))


# Factory function to create all clip tools
def create_clip_tools(synthesizer: AbstractSynthesizer) -> Dict[str, Any]:
    """Create all clip-based tools for AI conversation"""
    return {
        "create_midi_clip": CreateMIDIClipTool(synthesizer),
        "create_trigger_clip": CreateTriggerClipTool(synthesizer), 
        "manipulate_clip": ManipulateClipTool(synthesizer),
        "export_clip": ExportClipTool(synthesizer),
        "load_clip": LoadClipTool(synthesizer),
        "save_clip": SaveClipTool(synthesizer),
    }