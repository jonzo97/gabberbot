# 🎵 Track System User Guide

## Quick Start

The new Track system follows professional DAW principles. Each track has:
- **Control Source**: What triggers the track (pattern, MIDI, manual)
- **Audio Source**: What generates the sound (kick synth, Strudel, SuperCollider)
- **Effects Chain**: What processes the sound (distortion, compression, reverb)
- **Mixer**: Volume, pan, mute/solo controls

## Basic Track Creation

### Simple Kick Track
```python
from audio import Track, PatternControlSource, KickAudioSource

# Create track
kick_track = Track("My Kick")

# Set pattern trigger
kick_track.set_control_source(PatternControlSource("x ~ x ~ x ~ x ~"))

# Set kick synthesizer  
kick_track.set_audio_source(KickAudioSource(frequency=60.0, duration_ms=400))

# Add effects
kick_track.effects_chain.add_distortion("gabber", drive=2.5)
kick_track.effects_chain.add_compression(ratio=8.0, threshold_db=-10)

# Set mixer
kick_track.volume = 0.8
kick_track.pan = 0.0  # Center
```

### Multi-Track Session
```python
from audio import TrackCollection

# Create session
session = TrackCollection("My Session")

# Add tracks
session.add_track(kick_track)
session.add_track(lead_track)
session.add_track(bass_track)

# Render audio
from audio.parameters.synthesis_constants import SynthesisParams

params = SynthesisParams(frequency=60.0, bpm=170, brutality=0.8)
audio_step = session.render_step(step=0, bpm=170, params=params)
```

## Engine Backends

### Using Different Synthesis Engines

```python
from cli_strudel.synthesis.fm_synthesizer import StrudelFMSynth
from cli_sc.core.supercollider_synthesizer import SuperColliderSynth
from audio import KickAudioSource

# Strudel-based FM synthesis
strudel_track = Track("FM Lead")
strudel_track.set_audio_source(StrudelFMSynth())  # [STRUDEL-BASED]

# SuperCollider server synthesis
sc_track = Track("SC Bass") 
sc_track.set_audio_source(SuperColliderSynth())  # [SUPERCOLLIDER-BASED]

# Custom Python synthesis
python_track = Track("Python Kick")
python_track.set_audio_source(KickAudioSource())  # [CUSTOM-PYTHON]

# All work the same way in the Track system!
```

## Control Sources

### Pattern-Based Control
```python
# Standard patterns
kick_pattern = PatternControlSource("x ~ x ~ x ~ x ~")  # 4/4 kick
hihat_pattern = PatternControlSource("~ x ~ x ~ x ~ x")  # Off-beat hihat
complex_pattern = PatternControlSource("x ~ [x x] ~ x ~ ~ x")  # Complex

# Set to track
track.set_control_source(kick_pattern)
```

### MIDI Control
```python
# MIDI note triggering (future feature)
from audio import MidiControlSource

midi_control = MidiControlSource(midi_note=36)  # C1 kick
track.set_control_source(midi_control)
```

## Effects Chains

### Rotterdam Gabber Effects
```python
track.effects_chain.add_distortion("gabber", drive=2.5)  # Rotterdam doorlussen
track.effects_chain.add_compression(ratio=8.0, threshold_db=-10)
track.effects_chain.add_reverb("warehouse", wet_level=0.2)

# Or use preset
track.add_gabber_effects()
```

### Industrial Effects
```python
track.effects_chain.add_distortion("industrial", metallic_factor=0.6)
track.effects_chain.add_compression(ratio=4.0, threshold_db=-15)  
track.effects_chain.add_reverb("warehouse", wet_level=0.4)
```

### Custom Effects Chain
```python
# Build your own chain
track.effects_chain.add_distortion("alpha_juno", drive=15)
track.effects_chain.add_compression(ratio=6.0, threshold_db=-12)
track.effects_chain.add_reverb("industrial", wet_level=0.3)
```

## Preset Patterns

### Quick Track Setup Methods
```python
# Rotterdam gabber kick
track.add_kick_pattern("x ~ x ~ x ~ x ~", frequency=55.0, duration_ms=600)
track.add_gabber_effects()

# Industrial kick  
track.add_kick_pattern("x ~ ~ ~ x ~ x ~", frequency=45.0, duration_ms=800)
track.effects_chain.add_compression(ratio=4.0, threshold_db=-15)
track.effects_chain.add_reverb("warehouse", wet_level=0.4)
```

## Advanced Usage

### Engine Swapping
```python
# Start with Custom Python
track.set_audio_source(KickAudioSource(frequency=60.0))

# Switch to Strudel for more complex synthesis
track.set_audio_source(StrudelFMSynth(preset="HOOVER"))

# Switch to SuperCollider for professional quality
track.set_audio_source(SuperColliderSynth(synth_def="hardcore_kick"))

# Same track, different engines - seamless!
```

### Performance Optimization
```python
# Disable tracks to save CPU
track.enabled = False

# Mute/solo for mixing
track.muted = True
track.soloed = True

# Adjust sample rate for quality vs performance
track.sample_rate = 48000  # Higher quality
track.sample_rate = 22050  # Lower CPU usage
```

## Integration with Legacy Code

The Track system is designed to gradually replace the old engines/ code:

```python
# Old way (engines/ spaghetti - ELIMINATED)
# engine = ProfessionalHardcoreEngine()  # NO LONGER EXISTS

# New way (clean Track system)
session = TrackCollection("Professional Session")
session.add_track(gabber_kick_track)
session.add_track(hoover_lead_track)
audio = session.render_step(0, bpm=180, params=params)
```

## Troubleshooting

### Common Issues

**Import Errors**: Make sure to use absolute imports from project root
```python
from audio import Track  # ✅ Correct
from .audio import Track  # ❌ Wrong
```

**Missing Effects**: Check that effects chain is properly configured
```python
track.effects_chain.add_distortion("gabber")  # Effects must be added explicitly
```

**No Audio Output**: Verify control source is triggering
```python
# Check pattern
should_trigger = track.control_source.should_trigger(step=0, bpm=170)
print(should_trigger)  # Should be True for first step of "x ~ x ~" pattern
```

## See Also

- `examples/track_architecture_demo.py` - Working examples
- `docs/architecture/track_architecture.md` - Technical details  
- `docs/development/coding_standards.md` - Implementation standards