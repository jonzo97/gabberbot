#!/usr/bin/env python3
"""
Tempo Fix Validation Script

Generate test audio at 180 BPM to verify the tempo fix works correctly.
Creates both MIDI file and click track for manual validation.
"""

import sys
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, '.')

from cli_shared.models.midi_clips import MIDIClip, MIDINote
from tests.utils.tempo_validation import (
    generate_click_track,
    extract_bpm_from_midi,
    validate_tempo_accuracy
)
import numpy as np
import wave


def save_audio_wav(audio: np.ndarray, filepath: str, sample_rate: int = 44100):
    """Save audio samples to WAV file"""
    # Normalize audio
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to 16-bit PCM
    audio_int = (audio * 32767).astype(np.int16)

    # Save as WAV
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())


def main():
    print("=" * 60)
    print("TEMPO FIX VALIDATION - Story Phase 1 Consolidation 001")
    print("=" * 60)
    print()

    bpm = 180.0
    duration_bars = 4.0
    sample_rate = 44100

    print(f"Test Parameters:")
    print(f"  BPM: {bpm}")
    print(f"  Duration: {duration_bars} bars ({int(duration_bars * 4)} beats)")
    print(f"  Sample Rate: {sample_rate} Hz")
    print()

    # Create test MIDI clip
    print("1. Creating MIDI test clip...")
    clip = MIDIClip(name="tempo_fix_test_180bpm", length_bars=duration_bars, bpm=bpm)

    # Add 4-on-floor kick pattern
    for bar in range(int(duration_bars)):
        for beat in range(4):
            clip.add_note(MIDINote(
                pitch=36,  # C1 (kick)
                velocity=100,
                start_time=float(bar * 4 + beat),
                duration=0.5
            ))

    print(f"   Added {len(clip.notes)} kick notes")
    print()

    # Export MIDI file
    print("2. Exporting MIDI file...")
    midi_path = "tempo_fix_test_180bpm.mid"
    success = clip.save_midi_file(midi_path)

    if success:
        print(f"   ✓ MIDI exported: {midi_path}")

        # Validate MIDI tempo
        exported_bpm = extract_bpm_from_midi(midi_path)
        if exported_bpm:
            print(f"   ✓ Extracted BPM: {exported_bpm}")

            if validate_tempo_accuracy(bpm, exported_bpm, tolerance_percent=1.0):
                print(f"   ✓ Tempo validation: PASS (within 1% tolerance)")
            else:
                print(f"   ✗ Tempo validation: FAIL (outside 1% tolerance)")
        else:
            print(f"   ✗ Could not extract BPM from MIDI")
    else:
        print(f"   ✗ MIDI export failed")
    print()

    # Generate click track
    print("3. Generating click track audio...")
    click_audio = generate_click_track(bpm, duration_bars, sample_rate)

    # Calculate expected duration
    total_beats = duration_bars * 4
    expected_duration_sec = total_beats * (60.0 / bpm)
    expected_samples = int(expected_duration_sec * sample_rate)

    print(f"   Expected duration: {expected_duration_sec:.3f} seconds ({expected_samples} samples)")
    print(f"   Actual duration: {len(click_audio) / sample_rate:.3f} seconds ({len(click_audio)} samples)")

    if abs(len(click_audio) - expected_samples) <= 1:
        print(f"   ✓ Duration validation: PASS")
    else:
        print(f"   ✗ Duration validation: FAIL")
    print()

    # Save click track
    print("4. Saving click track WAV file...")
    click_path = "tempo_fix_test_180bpm_click.wav"
    save_audio_wav(click_audio, click_path, sample_rate)
    print(f"   ✓ Click track saved: {click_path}")
    print()

    # Validation summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print()
    print("Files generated for manual validation:")
    print(f"  1. {midi_path} - Load in DAW to verify 180 BPM playback")
    print(f"  2. {click_path} - Listen to verify steady 180 BPM clicks")
    print()
    print("Manual Validation Steps:")
    print("  1. Load MIDI file in your DAW (Ableton, FL Studio, etc.)")
    print("  2. Verify the DAW shows 180 BPM")
    print("  3. Listen to MIDI playback - should be steady kicks at 180 BPM")
    print("  4. Listen to click track - should be steady clicks at 180 BPM")
    print("  5. Both should sound identical in tempo")
    print()
    print("Expected Result:")
    print("  - Kicks/clicks should occur every 0.333 seconds (14700 samples)")
    print("  - 4 bars = 16 beats = 5.333 seconds total duration")
    print("  - If music sounds 4x too fast or slow, the bug still exists")
    print()

    # Timing verification
    print("TIMING VERIFICATION:")
    print()
    seconds_per_beat = 60.0 / bpm
    for beat in [0, 1, 2, 4, 8, 16]:
        sample_pos = int(beat * seconds_per_beat * sample_rate)
        time_sec = beat * seconds_per_beat
        print(f"  Beat {beat:2d}: {sample_pos:6d} samples = {time_sec:.3f} seconds")
    print()

    print("✓ Validation script complete!")
    print()


if __name__ == "__main__":
    main()
