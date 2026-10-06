#!/usr/bin/env python3
"""
Tempo Validation Utilities

Utilities for extracting and validating tempo from MIDI files and audio.
Used to verify tempo accuracy in test suite.
"""

import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    mido = None


def extract_bpm_from_midi(filepath: str) -> Optional[float]:
    """
    Extract BPM from MIDI file.

    Args:
        filepath: Path to MIDI file

    Returns:
        BPM value if found, None otherwise
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido library required for MIDI BPM extraction")

    try:
        mid = mido.MidiFile(filepath)

        # Look for tempo meta message in tracks
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    # Tempo is in microseconds per beat
                    bpm = 60_000_000 / msg.tempo
                    return float(bpm)

        return None
    except Exception as e:
        print(f"Error extracting BPM from MIDI: {e}")
        return None


def extract_note_timings_from_midi(filepath: str) -> list:
    """
    Extract note timing information from MIDI file.

    Returns list of (start_time_beats, duration_beats, pitch, velocity) tuples.
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido library required for MIDI note extraction")

    try:
        mid = mido.MidiFile(filepath)
        ticks_per_beat = mid.ticks_per_beat

        notes = []
        current_ticks = 0
        active_notes = {}  # pitch -> (start_ticks, velocity)

        for track in mid.tracks:
            current_ticks = 0
            for msg in track:
                current_ticks += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_ticks, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_ticks, velocity = active_notes.pop(msg.note)
                        duration_ticks = current_ticks - start_ticks

                        # Convert ticks to beats
                        start_beats = start_ticks / ticks_per_beat
                        duration_beats = duration_ticks / ticks_per_beat

                        notes.append((start_beats, duration_beats, msg.note, velocity))

        return sorted(notes, key=lambda x: x[0])
    except Exception as e:
        print(f"Error extracting note timings from MIDI: {e}")
        return []


def detect_bpm_from_audio(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """
    Detect BPM from audio using autocorrelation.

    This is a simple onset-based tempo detection.
    For production use, librosa.beat.tempo() is more accurate.

    Args:
        audio: Audio samples (mono)
        sample_rate: Sample rate in Hz

    Returns:
        Estimated BPM or None if detection fails
    """
    # Simple energy-based onset detection
    # Calculate frame energy
    frame_size = 2048
    hop_size = 512

    num_frames = (len(audio) - frame_size) // hop_size
    energy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = audio[start:end]
        energy[i] = np.sum(frame ** 2)

    # Detect peaks in energy (onsets)
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    onsets = []

    for i in range(1, len(energy) - 1):
        if energy[i] > threshold and energy[i] > energy[i-1] and energy[i] > energy[i+1]:
            onset_time = i * hop_size / sample_rate
            onsets.append(onset_time)

    if len(onsets) < 2:
        return None

    # Calculate inter-onset intervals
    intervals = np.diff(onsets)

    if len(intervals) == 0:
        return None

    # Use median interval as beat duration
    beat_duration = np.median(intervals)

    if beat_duration <= 0:
        return None

    bpm = 60.0 / beat_duration

    # Sanity check - reasonable BPM range
    if bpm < 60 or bpm > 300:
        return None

    return float(bpm)


def generate_click_track(bpm: float, duration_bars: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a simple click track at specified BPM.

    Args:
        bpm: Tempo in beats per minute
        duration_bars: Duration in bars (4 beats per bar)
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples for click track
    """
    beats_per_bar = 4
    total_beats = duration_bars * beats_per_bar
    seconds_per_beat = 60.0 / bpm
    total_duration = total_beats * seconds_per_beat

    total_samples = int(total_duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Generate click at each beat
    click_duration = 0.01  # 10ms click
    click_samples = int(click_duration * sample_rate)

    for beat in range(int(total_beats)):
        start_sample = int(beat * seconds_per_beat * sample_rate)

        if start_sample + click_samples <= total_samples:
            # Generate click tone (1000 Hz sine wave)
            t = np.linspace(0, click_duration, click_samples)
            click = np.sin(2 * np.pi * 1000 * t) * 0.5

            # Apply envelope
            envelope = np.exp(-t * 100)
            click *= envelope

            audio[start_sample:start_sample + click_samples] = click

    return audio


def validate_tempo_accuracy(expected_bpm: float, actual_bpm: float, tolerance_percent: float = 1.0) -> bool:
    """
    Validate that actual BPM is within tolerance of expected BPM.

    Args:
        expected_bpm: Expected BPM value
        actual_bpm: Measured BPM value
        tolerance_percent: Allowed error percentage (default 1%)

    Returns:
        True if within tolerance, False otherwise
    """
    if actual_bpm is None:
        return False

    error_percent = abs(actual_bpm - expected_bpm) / expected_bpm * 100
    return error_percent <= tolerance_percent


def calculate_beat_to_sample(beat_time: float, bpm: float, sample_rate: int) -> int:
    """
    Calculate sample position from beat time.

    This is the CORRECT formula that should be used throughout the codebase.

    Args:
        beat_time: Time in beats
        bpm: Tempo in beats per minute
        sample_rate: Audio sample rate in Hz

    Returns:
        Sample position (integer)
    """
    seconds_per_beat = 60.0 / bpm
    time_seconds = beat_time * seconds_per_beat
    sample_position = int(time_seconds * sample_rate)
    return sample_position


def calculate_sample_to_beat(sample_position: int, bpm: float, sample_rate: int) -> float:
    """
    Calculate beat time from sample position.

    Args:
        sample_position: Sample index
        bpm: Tempo in beats per minute
        sample_rate: Audio sample rate in Hz

    Returns:
        Time in beats (float)
    """
    time_seconds = sample_position / sample_rate
    seconds_per_beat = 60.0 / bpm
    beat_time = time_seconds / seconds_per_beat
    return beat_time


def verify_note_timing(expected_beat: float, actual_sample: int, bpm: float, sample_rate: int,
                       tolerance_samples: int = 10) -> Tuple[bool, int]:
    """
    Verify that a note's sample position matches expected beat timing.

    Args:
        expected_beat: Expected beat position
        actual_sample: Actual sample position in audio
        bpm: Tempo in BPM
        sample_rate: Audio sample rate
        tolerance_samples: Allowed error in samples

    Returns:
        (is_correct, error_samples) tuple
    """
    expected_sample = calculate_beat_to_sample(expected_beat, bpm, sample_rate)
    error_samples = abs(actual_sample - expected_sample)
    is_correct = error_samples <= tolerance_samples

    return is_correct, error_samples
