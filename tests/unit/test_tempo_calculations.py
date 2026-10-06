#!/usr/bin/env python3
"""
Unit Tests for Tempo Calculations

Tests the core tempo conversion formulas to ensure correct timing.
These tests verify the fix for the 4x tempo bug.
"""

import pytest
import numpy as np
from tests.utils.tempo_validation import (
    calculate_beat_to_sample,
    calculate_sample_to_beat,
    validate_tempo_accuracy,
    verify_note_timing
)


class TestBeatToSampleConversion:
    """Test beat to sample conversion formula"""

    def test_180_bpm_one_beat(self):
        """At 180 BPM, 1 beat should be exactly 14700 samples at 44.1kHz"""
        bpm = 180
        sample_rate = 44100
        beat_time = 1.0

        # Expected: 1 beat * (60/180) seconds/beat * 44100 samples/second = 14700 samples
        expected_samples = 14700

        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == expected_samples, \
            f"At 180 BPM, 1 beat should be {expected_samples} samples, got {actual_samples}"

    def test_180_bpm_four_beats(self):
        """At 180 BPM, 4 beats (1 bar) should be 58800 samples"""
        bpm = 180
        sample_rate = 44100
        beat_time = 4.0

        # Expected: 4 beats * (60/180) * 44100 = 58800 samples
        expected_samples = 58800

        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == expected_samples

    def test_120_bpm_one_beat(self):
        """At 120 BPM, 1 beat should be 22050 samples"""
        bpm = 120
        sample_rate = 44100
        beat_time = 1.0

        # Expected: 1 * (60/120) * 44100 = 22050 samples
        expected_samples = 22050

        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == expected_samples

    def test_200_bpm_one_beat(self):
        """At 200 BPM, 1 beat should be 13230 samples"""
        bpm = 200
        sample_rate = 44100
        beat_time = 1.0

        # Expected: 1 * (60/200) * 44100 = 13230 samples
        expected_samples = 13230

        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == expected_samples

    def test_zero_beat(self):
        """Beat 0 should always be sample 0"""
        bpm = 180
        sample_rate = 44100
        beat_time = 0.0

        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == 0

    def test_fractional_beat(self):
        """Test fractional beat values (e.g., 1.5 beats)"""
        bpm = 180
        sample_rate = 44100
        beat_time = 1.5

        # Expected: 1.5 * (60/180) * 44100 = 22050 samples
        expected_samples = 22050

        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == expected_samples


class TestSampleToBeatConversion:
    """Test sample to beat conversion (inverse operation)"""

    def test_round_trip_conversion(self):
        """Converting beat->sample->beat should return original value"""
        bpm = 180
        sample_rate = 44100
        original_beat = 2.5

        sample = calculate_beat_to_sample(original_beat, bpm, sample_rate)
        beat = calculate_sample_to_beat(sample, bpm, sample_rate)

        assert abs(beat - original_beat) < 0.001, \
            f"Round trip conversion failed: {original_beat} -> {sample} -> {beat}"

    def test_14700_samples_is_one_beat(self):
        """At 180 BPM, 14700 samples should be exactly 1 beat"""
        bpm = 180
        sample_rate = 44100
        sample_position = 14700

        beat = calculate_sample_to_beat(sample_position, bpm, sample_rate)

        assert abs(beat - 1.0) < 0.001


class TestTempoAccuracyValidation:
    """Test tempo accuracy validation helper"""

    def test_exact_match(self):
        """Exact BPM match should validate"""
        expected = 180.0
        actual = 180.0
        tolerance = 1.0

        result = validate_tempo_accuracy(expected, actual, tolerance)

        assert result is True

    def test_within_tolerance(self):
        """BPM within 1% tolerance should validate"""
        expected = 180.0
        actual = 181.0  # 0.56% error
        tolerance = 1.0

        result = validate_tempo_accuracy(expected, actual, tolerance)

        assert result is True

    def test_outside_tolerance(self):
        """BPM outside tolerance should fail"""
        expected = 180.0
        actual = 185.0  # 2.78% error
        tolerance = 1.0

        result = validate_tempo_accuracy(expected, actual, tolerance)

        assert result is False

    def test_none_actual_bpm(self):
        """None actual BPM should fail validation"""
        expected = 180.0
        actual = None
        tolerance = 1.0

        result = validate_tempo_accuracy(expected, actual, tolerance)

        assert result is False


class TestNoteTimingVerification:
    """Test note timing verification"""

    def test_correct_timing(self):
        """Note at correct sample position should verify"""
        expected_beat = 1.0
        bpm = 180
        sample_rate = 44100

        # Correct sample position for beat 1 at 180 BPM
        actual_sample = 14700

        is_correct, error = verify_note_timing(expected_beat, actual_sample, bpm, sample_rate)

        assert is_correct is True
        assert error == 0

    def test_4x_too_fast_bug(self):
        """Detect the 4x too fast bug (division by 4 error)"""
        expected_beat = 1.0
        bpm = 180
        sample_rate = 44100

        # Buggy calculation: note.start_time * 60.0 / bpm * sample_rate / 4.0
        # = 1.0 * 60.0 / 180.0 * 44100 / 4.0 = 3675 samples (WRONG!)
        buggy_sample = 3675

        is_correct, error = verify_note_timing(expected_beat, buggy_sample, bpm, sample_rate,
                                                tolerance_samples=10)

        assert is_correct is False, "4x division bug should be detected"
        assert error == 11025, f"Error should be 14700 - 3675 = 11025, got {error}"

    def test_4x_too_slow_bug(self):
        """Detect the 4x too slow bug (multiplication by 4 error)"""
        expected_beat = 1.0
        bpm = 180
        sample_rate = 44100

        # Buggy calculation: note.start_time * 60.0 / bpm * 4 * sample_rate
        # = 1.0 * 60.0 / 180.0 * 4 * 44100 = 58800 samples (WRONG!)
        buggy_sample = 58800

        is_correct, error = verify_note_timing(expected_beat, buggy_sample, bpm, sample_rate,
                                                tolerance_samples=10)

        assert is_correct is False, "4x multiplication bug should be detected"
        assert error == 44100, f"Error should be 58800 - 14700 = 44100, got {error}"


class TestMIDITempoConversion:
    """Test MIDI tempo meta message conversion"""

    def test_bpm_to_microseconds(self):
        """Test BPM to microseconds per quarter note conversion"""
        test_cases = [
            (120, 500000),   # 120 BPM = 500000 microseconds per beat
            (180, 333333),   # 180 BPM = 333333 microseconds per beat (rounded)
            (200, 300000),   # 200 BPM = 300000 microseconds per beat
            (60, 1000000),   # 60 BPM = 1000000 microseconds per beat
        ]

        for bpm, expected_microseconds in test_cases:
            # MIDI tempo formula: microseconds = 60,000,000 / BPM
            actual_microseconds = int(60_000_000 / bpm)

            # Allow small rounding error
            error = abs(actual_microseconds - expected_microseconds)
            assert error <= 1, \
                f"BPM {bpm} should convert to {expected_microseconds} microseconds, got {actual_microseconds}"

    def test_microseconds_to_bpm(self):
        """Test microseconds to BPM conversion"""
        test_cases = [
            (500000, 120),
            (333333, 180),
            (300000, 200),
        ]

        for microseconds, expected_bpm in test_cases:
            # Inverse formula: BPM = 60,000,000 / microseconds
            actual_bpm = 60_000_000 / microseconds

            # Allow 1% error due to rounding
            assert validate_tempo_accuracy(expected_bpm, actual_bpm, tolerance_percent=1.0)


class TestMIDITicksConversion:
    """Test MIDI ticks per quarter note (TPQN) conversion"""

    def test_beat_to_ticks_480_tpqn(self):
        """Standard MIDI uses 480 ticks per quarter note"""
        tpqn = 480
        test_cases = [
            (0.0, 0),      # Beat 0 = tick 0
            (1.0, 480),    # Beat 1 = tick 480
            (2.0, 960),    # Beat 2 = tick 960
            (4.0, 1920),   # Bar 1 = tick 1920
            (0.5, 240),    # Half beat = 240 ticks
            (0.25, 120),   # Quarter beat (16th note) = 120 ticks
        ]

        for beat, expected_ticks in test_cases:
            actual_ticks = int(beat * tpqn)
            assert actual_ticks == expected_ticks, \
                f"Beat {beat} should be {expected_ticks} ticks, got {actual_ticks}"

    def test_ticks_to_beat_480_tpqn(self):
        """Convert ticks back to beats"""
        tpqn = 480
        test_cases = [
            (0, 0.0),
            (480, 1.0),
            (960, 2.0),
            (240, 0.5),
            (120, 0.25),
        ]

        for ticks, expected_beat in test_cases:
            actual_beat = ticks / tpqn
            assert abs(actual_beat - expected_beat) < 0.001, \
                f"Ticks {ticks} should be beat {expected_beat}, got {actual_beat}"


class TestHardcoreBPMRange:
    """Test tempo calculations across hardcore BPM range (120-220)"""

    @pytest.mark.parametrize("bpm", [120, 150, 180, 200, 220])
    def test_one_beat_conversion(self, bpm):
        """Test that 1 beat converts correctly across BPM range"""
        sample_rate = 44100
        beat_time = 1.0

        # Calculate expected samples: beat * (60/BPM) * sample_rate
        expected_samples = int(beat_time * (60.0 / bpm) * sample_rate)

        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == expected_samples, \
            f"At {bpm} BPM, 1 beat conversion failed"

    @pytest.mark.parametrize("bpm", [120, 150, 180, 200, 220])
    def test_four_bar_phrase(self, bpm):
        """Test that a 4-bar phrase (16 beats) converts correctly"""
        sample_rate = 44100
        beat_time = 16.0  # 4 bars * 4 beats

        expected_samples = int(beat_time * (60.0 / bpm) * sample_rate)
        actual_samples = calculate_beat_to_sample(beat_time, bpm, sample_rate)

        assert actual_samples == expected_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
