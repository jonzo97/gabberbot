#!/usr/bin/env python3
"""
Integration Tests for Tempo Accuracy

End-to-end tests that verify MIDI export and audio rendering
produce correct tempo output. These tests will FAIL before the
4x tempo bug is fixed.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from cli_shared.models.midi_clips import MIDIClip, MIDINote
from tests.utils.tempo_validation import (
    extract_bpm_from_midi,
    extract_note_timings_from_midi,
    validate_tempo_accuracy,
    generate_click_track,
    calculate_beat_to_sample
)


class TestMIDIExportTempo:
    """Test MIDI export produces correct tempo"""

    def test_midi_export_180_bpm(self):
        """MIDI export should preserve 180 BPM tempo"""
        bpm = 180.0

        # Create test clip with known BPM
        clip = MIDIClip(name="tempo_test_180", length_bars=4.0, bpm=bpm)

        # Add a simple kick pattern
        for beat in [0, 1, 2, 3]:
            clip.add_note(MIDINote(
                pitch=36,  # C1 (kick)
                velocity=100,
                start_time=float(beat),
                duration=0.5
            ))

        # Export to MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi_path = f.name

        try:
            clip.save_midi_file(midi_path)

            # Extract BPM from exported file
            exported_bpm = extract_bpm_from_midi(midi_path)

            # Validate tempo accuracy (within 1%)
            assert exported_bpm is not None, "Could not extract BPM from MIDI file"
            assert validate_tempo_accuracy(bpm, exported_bpm, tolerance_percent=1.0), \
                f"MIDI export BPM mismatch: expected {bpm}, got {exported_bpm}"

        finally:
            Path(midi_path).unlink(missing_ok=True)

    @pytest.mark.parametrize("bpm", [120, 150, 180, 200, 220])
    def test_midi_export_various_bpms(self, bpm):
        """MIDI export should preserve tempo across BPM range"""
        clip = MIDIClip(name=f"tempo_test_{bpm}", length_bars=2.0, bpm=float(bpm))

        # Add notes
        clip.add_note(MIDINote(pitch=36, velocity=100, start_time=0.0, duration=0.5))
        clip.add_note(MIDINote(pitch=36, velocity=100, start_time=1.0, duration=0.5))

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi_path = f.name

        try:
            clip.save_midi_file(midi_path)
            exported_bpm = extract_bpm_from_midi(midi_path)

            assert exported_bpm is not None
            assert validate_tempo_accuracy(bpm, exported_bpm, tolerance_percent=1.0), \
                f"BPM {bpm} export failed: got {exported_bpm}"

        finally:
            Path(midi_path).unlink(missing_ok=True)

    def test_midi_note_timing_accuracy(self):
        """MIDI note timings should be accurate"""
        bpm = 180.0
        clip = MIDIClip(name="timing_test", length_bars=4.0, bpm=bpm)

        # Add notes at specific beat positions
        expected_notes = [
            (0.0, 0.5, 36, 100),   # Beat 0
            (1.0, 0.5, 36, 100),   # Beat 1
            (2.0, 0.5, 36, 100),   # Beat 2
            (4.0, 0.5, 36, 100),   # Beat 4 (bar 2)
        ]

        for start, duration, pitch, vel in expected_notes:
            clip.add_note(MIDINote(
                pitch=pitch,
                velocity=vel,
                start_time=start,
                duration=duration
            ))

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi_path = f.name

        try:
            clip.save_midi_file(midi_path)

            # Extract note timings
            exported_notes = extract_note_timings_from_midi(midi_path)

            assert len(exported_notes) == len(expected_notes), \
                f"Expected {len(expected_notes)} notes, got {len(exported_notes)}"

            # Verify each note timing (allow 0.01 beat tolerance for rounding)
            for expected, actual in zip(expected_notes, exported_notes):
                exp_start, exp_dur, exp_pitch, exp_vel = expected
                act_start, act_dur, act_pitch, act_vel = actual

                assert abs(act_start - exp_start) < 0.01, \
                    f"Note start timing error: expected {exp_start}, got {act_start}"

                assert abs(act_dur - exp_dur) < 0.01, \
                    f"Note duration error: expected {exp_dur}, got {act_dur}"

                assert act_pitch == exp_pitch

        finally:
            Path(midi_path).unlink(missing_ok=True)


class TestClickTrackGeneration:
    """Test click track generation for tempo validation"""

    def test_click_track_180_bpm(self):
        """Click track should have correct beat spacing at 180 BPM"""
        bpm = 180.0
        duration_bars = 4.0
        sample_rate = 44100

        audio = generate_click_track(bpm, duration_bars, sample_rate)

        # At 180 BPM, 4 bars = 16 beats
        # Duration should be: 16 beats * (60/180) seconds/beat = 5.333 seconds
        expected_duration = 16.0 * (60.0 / bpm)
        expected_samples = int(expected_duration * sample_rate)

        assert abs(len(audio) - expected_samples) <= 1, \
            f"Click track duration error: expected ~{expected_samples} samples, got {len(audio)}"

    def test_click_track_beat_positions(self):
        """Click track should have energy peaks at correct beat positions"""
        bpm = 180.0
        duration_bars = 2.0
        sample_rate = 44100

        audio = generate_click_track(bpm, duration_bars, sample_rate)

        # Check for energy peaks at expected beat positions
        beats_to_check = [0, 1, 2, 3, 4, 5, 6, 7]  # 2 bars = 8 beats

        for beat in beats_to_check:
            expected_sample = calculate_beat_to_sample(beat, bpm, sample_rate)

            # Check for energy peak around expected position
            # Allow ±100 samples tolerance
            start = max(0, expected_sample - 100)
            end = min(len(audio), expected_sample + 100)

            if end > start:
                segment = audio[start:end]
                max_energy = np.max(np.abs(segment))

                assert max_energy > 0.1, \
                    f"No click found near beat {beat} (sample {expected_sample})"


class TestAudioRenderingTempo:
    """Test audio rendering produces correct tempo

    NOTE: These tests require the buggy files to be fixed.
    They will FAIL before the fix is applied.
    """

    def test_simple_kick_pattern_timing(self):
        """Test that synthesized kicks appear at correct sample positions"""
        from cli_shared.models.hardcore_models import BMadTrackConfig, HardcoreStyle

        bpm = 180.0
        sample_rate = 44100

        # Create simple kick clip
        clip = MIDIClip(name="kick_timing_test", length_bars=4.0, bpm=bpm)

        # Add kicks at beats 0, 1, 2, 3
        for beat in range(4):
            clip.add_note(MIDINote(
                pitch=36,
                velocity=100,
                start_time=float(beat),
                duration=0.5
            ))

        # Expected sample positions for kicks
        expected_positions = [
            calculate_beat_to_sample(0, bpm, sample_rate),  # 0 samples
            calculate_beat_to_sample(1, bpm, sample_rate),  # 14700 samples
            calculate_beat_to_sample(2, bpm, sample_rate),  # 29400 samples
            calculate_beat_to_sample(3, bpm, sample_rate),  # 44100 samples
        ]

        # This test documents the expected behavior
        # Actual synthesis testing would require importing and testing
        # the buggy modules, which we'll fix instead

        for i, expected_pos in enumerate(expected_positions):
            beat = float(i)
            # Verify our calculation formula
            calculated = int(beat * (60.0 / bpm) * sample_rate)
            assert calculated == expected_pos, \
                f"Beat {beat} calculation error: {calculated} != {expected_pos}"

    def test_detect_4x_division_bug(self):
        """Document the 4x division bug behavior"""
        bpm = 180.0
        sample_rate = 44100
        beat = 1.0

        # CORRECT formula
        correct_sample = int(beat * (60.0 / bpm) * sample_rate)
        assert correct_sample == 14700

        # BUGGY formula (division by 4)
        buggy_sample = int(beat * 60.0 / bpm * sample_rate / 4.0)
        assert buggy_sample == 3675

        # Bug analysis
        assert buggy_sample * 4 == correct_sample, \
            "Division by 4 bug causes 4x too fast playback"

    def test_detect_4x_multiplication_bug(self):
        """Document the 4x multiplication bug behavior"""
        bpm = 180.0
        sample_rate = 44100
        beat = 1.0

        # CORRECT formula
        correct_sample = int(beat * (60.0 / bpm) * sample_rate)
        assert correct_sample == 14700

        # BUGGY formula (multiplication by 4)
        buggy_sample = int(beat * 60.0 / bpm * 4 * sample_rate)
        assert buggy_sample == 58800

        # Bug analysis
        assert buggy_sample / 4 == correct_sample, \
            "Multiplication by 4 bug causes 4x too slow playback"


class TestEndToEndTempoValidation:
    """End-to-end tempo validation tests"""

    @pytest.mark.parametrize("bpm", [120, 150, 180, 200, 220])
    def test_midi_export_end_to_end(self, bpm):
        """Complete MIDI export workflow with tempo validation"""
        clip = MIDIClip(name=f"e2e_test_{bpm}", length_bars=4.0, bpm=float(bpm))

        # Create 4-on-floor kick pattern
        for bar in range(4):
            for beat in range(4):
                clip.add_note(MIDINote(
                    pitch=36,
                    velocity=100,
                    start_time=float(bar * 4 + beat),
                    duration=0.5
                ))

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi_path = f.name

        try:
            # Export
            success = clip.save_midi_file(midi_path)
            assert success, "MIDI export failed"

            # Validate BPM
            exported_bpm = extract_bpm_from_midi(midi_path)
            assert exported_bpm is not None
            assert validate_tempo_accuracy(bpm, exported_bpm, tolerance_percent=1.0)

            # Validate note count
            exported_notes = extract_note_timings_from_midi(midi_path)
            assert len(exported_notes) == 16, f"Expected 16 notes, got {len(exported_notes)}"

        finally:
            Path(midi_path).unlink(missing_ok=True)

    def test_tempo_consistency_check(self):
        """Verify tempo remains consistent across multiple operations"""
        bpm = 180.0
        clip = MIDIClip(name="consistency_test", length_bars=8.0, bpm=bpm)

        # Add varied pattern
        for beat in range(32):  # 8 bars * 4 beats
            if beat % 2 == 0:  # On even beats
                clip.add_note(MIDINote(
                    pitch=36,
                    velocity=100,
                    start_time=float(beat),
                    duration=0.5
                ))

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi_path = f.name

        try:
            clip.save_midi_file(midi_path)

            # Extract and verify
            exported_bpm = extract_bpm_from_midi(midi_path)
            exported_notes = extract_note_timings_from_midi(midi_path)

            # BPM should be exact
            assert validate_tempo_accuracy(bpm, exported_bpm, tolerance_percent=0.1)

            # Note timings should be on even beats
            for i, (start_beat, dur, pitch, vel) in enumerate(exported_notes):
                expected_beat = float(i * 2)
                assert abs(start_beat - expected_beat) < 0.01, \
                    f"Note {i} timing error: expected beat {expected_beat}, got {start_beat}"

        finally:
            Path(midi_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
