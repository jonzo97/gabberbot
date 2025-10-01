#!/usr/bin/env python3
"""
Hardcore Music Prototyper - Main Entry Point

Complete text-to-audio pipeline: Takes natural language prompts and generates
high-quality WAV files of hardcore, gabber, and industrial music.

Usage:
    python main.py "create 180 BPM gabber kick pattern"
    python main.py "make acid bassline at 160 BPM in A minor"
    python main.py "generate hardcore riff with heavy distortion"
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.models.config import Settings
from src.services.generation_service import GenerationService
from src.services.audio_service import AudioService, AudioConfig, RenderProgress
from src.utils.env import load_settings


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def progress_callback(progress: RenderProgress) -> None:
    """Display progress updates to the user."""
    progress_bar = "█" * int(progress.progress * 20) + "░" * (20 - int(progress.progress * 20))
    print(f"\r[{progress_bar}] {progress.progress*100:.0f}% - {progress.message}", end="", flush=True)
    
    if progress.progress >= 1.0:
        print()  # New line when complete


async def generate_music(
    prompt: str,
    output_path: Optional[Path] = None,
    settings: Optional[Settings] = None,
    verbose: bool = False
) -> Path:
    """
    Complete text-to-audio generation pipeline.
    
    Args:
        prompt: Natural language description of desired music
        output_path: Optional output file path
        settings: Optional configuration settings
        verbose: Enable verbose logging
        
    Returns:
        Path to generated WAV file
    """
    # Load settings
    if settings is None:
        settings = load_settings()
    
    # Set up output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        output_path = Path(f"output/hardcore_{safe_prompt}_{timestamp}.wav")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🎵 Hardcore Music Prototyper")
    print(f"📝 Prompt: \"{prompt}\"")
    print(f"📁 Output: {output_path}")
    print()
    
    # Initialize services
    print("🔧 Initializing AI generation service...")
    generation_service = GenerationService(settings)
    
    print("🔊 Initializing audio synthesis service...")
    audio_config = AudioConfig(
        sample_rate=44100,
        bit_depth=16,
        channels=1
    )
    audio_service = AudioService(
        settings, 
        audio_config=audio_config,
        progress_callback=progress_callback
    )
    
    # Generate MIDI pattern from text
    print("🤖 Generating musical pattern from prompt...")
    try:
        midi_clip = await generation_service.text_to_midi(prompt)
        print(f"✅ Generated {len(midi_clip.notes)} notes at {midi_clip.bpm} BPM")
        
        if verbose:
            print(f"   📊 Clip details:")
            print(f"      Name: {midi_clip.name}")
            print(f"      Key: {midi_clip.key}")
            print(f"      Length: {midi_clip.length_bars} bars")
            print(f"      Notes: {len(midi_clip.notes)}")
    
    except Exception as e:
        print(f"❌ Failed to generate musical pattern: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Render audio from MIDI
    print("🎛️  Rendering high-quality audio...")
    try:
        output_file = audio_service.render_to_wav(midi_clip, output_path)
        print(f"✅ Audio rendered successfully!")
        print(f"🎵 File saved: {output_file}")
        
        # Show stats
        gen_stats = generation_service.get_stats()
        audio_stats = audio_service.get_stats()
        
        print(f"\n📊 Generation Statistics:")
        print(f"   🤖 AI Generation: {gen_stats['successful_ai']} successful, {gen_stats['successful_fallback']} fallback")
        print(f"   🔊 Audio Renders: {audio_stats['successful_renders']} successful")
        print(f"   ⏱️  Average render time: {audio_stats['avg_render_time']:.2f}s")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Failed to render audio: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the hardcore music prototyper."""
    parser = argparse.ArgumentParser(
        description="Hardcore Music Prototyper - Generate hardcore music from text",
        epilog="Examples:\n"
               "  python main.py \"create 180 BPM gabber kick pattern\"\n"
               "  python main.py \"make acid bassline at 160 BPM in A minor\"\n"
               "  python main.py \"generate hardcore riff with heavy distortion\"",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "prompt",
        help="Natural language description of the music to generate"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output WAV file path (default: auto-generated in output/ directory)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging and detailed output"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        choices=[22050, 44100, 48000, 96000],
        help="Audio sample rate (default: 44100)"
    )
    
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=16,
        choices=[16, 24, 32],
        help="Audio bit depth (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        print("💡 Make sure your .env file is configured correctly")
        sys.exit(1)
    
    # Run generation
    try:
        output_file = asyncio.run(generate_music(
            prompt=args.prompt,
            output_path=args.output,
            settings=settings,
            verbose=args.verbose
        ))
        
        print(f"\n🎉 Success! Your hardcore music is ready:")
        print(f"🎵 {output_file}")
        print(f"\n💿 Load this WAV file into your DAW or play it with any audio player")
        print(f"🔥 Ready to destroy sound systems! 💀")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Generation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()