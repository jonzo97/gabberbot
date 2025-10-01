# 🧪 GABBERBOT Testing Guide

## Current Status

**✅ Working Components (No Dependencies)**
- Core models and data structures
- Mock synthesizer for testing
- Performance engine (basic functionality)
- Pattern creation and management
- Basic audio generation

**⚠️ Limited Functionality (Missing Dependencies)**
- AI conversation engine (needs: openai, anthropic)
- Audio analysis (needs: librosa, scipy, scikit-learn)
- Hardware MIDI (needs: python-rtmidi, mido)
- Web interface (needs: fastapi, uvicorn, pydantic)

## Quick Start Testing

### 1. Basic Functionality Test
```bash
# Run the simplified test to check core systems
python3 test_simplified.py
```

Expected output: 3/5 systems operational (Core Models, Mock Synthesizer, Performance Engine)

### 2. Install Minimal Dependencies
To get more features working, install these packages:

```bash
# For AI features (choose based on what API keys you have)
pip install openai  # If you have OpenAI API key
pip install anthropic  # If you have Anthropic API key

# For audio analysis
pip install numpy scipy  # Basic audio processing
pip install librosa  # Advanced audio analysis (optional, larger)

# For web interface
pip install fastapi uvicorn pydantic

# For hardware MIDI (optional)
pip install mido python-rtmidi
```

### 3. Test With API Keys
If you have API keys, you can test the AI features:

```python
# test_with_ai.py
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'

from cli_shared.ai.conversation_engine import create_conversation_engine
from cli_shared.interfaces.synthesizer import MockSynthesizer

synth = MockSynthesizer()
engine = create_conversation_engine(synth, openai_key=os.environ['OPENAI_API_KEY'])

# Now test with: "Make a gabber kick at 180 BPM"
```

## Component-Specific Tests

### Test Pattern Creation
```python
from cli_shared.models.hardcore_models import HardcorePattern, SynthType

pattern = HardcorePattern(
    name="test_pattern",
    bpm=180,
    pattern_data='s("bd:5").struct("x ~ x ~")',
    synth_type=SynthType.GABBER_KICK,
    genre="gabber"
)
print(f"Created: {pattern.name} at {pattern.bpm} BPM")
```

### Test Mock Synthesizer
```python
import asyncio
from cli_shared.interfaces.synthesizer import MockSynthesizer

async def test_synth():
    synth = MockSynthesizer()
    audio = await synth.play_pattern(pattern)
    print(f"Generated {len(audio)} audio samples")

asyncio.run(test_synth())
```

### Test Performance Engine
```python
from cli_shared.performance.live_performance_engine import LivePerformanceEngine

async def test_performance():
    engine = LivePerformanceEngine(synth)
    await engine.load_pattern_to_slot("slot_00", pattern)
    await engine.trigger_slot("slot_00")
    stats = engine.get_performance_stats()
    print(f"Performance state: {stats['state']}")

asyncio.run(test_performance())
```

## Testing Without External Dependencies

The system is designed to work with mock implementations when dependencies are missing:

1. **MockSynthesizer**: Generates simple sine waves instead of real audio
2. **Fallback Audio Analysis**: Basic numpy-based analysis when librosa unavailable
3. **No-API Mode**: Can test conversation engine structure without actual AI calls
4. **Simulated Hardware**: MIDI controller simulation for testing without hardware

## Progressive Testing Strategy

### Level 1: Core Functionality (No Dependencies)
- ✅ Import all modules
- ✅ Create patterns and models
- ✅ Use mock synthesizer
- ✅ Basic performance engine

### Level 2: Basic Dependencies (numpy, scipy)
- ✅ Basic audio analysis
- ✅ Pattern evolution (simplified)
- ✅ Audio utilities

### Level 3: AI Features (openai/anthropic)
- ✅ Natural language pattern creation
- ✅ Conversational interface
- ✅ Style transfer

### Level 4: Full Features (all dependencies)
- ✅ Advanced audio analysis with librosa
- ✅ Web interface with FastAPI
- ✅ Hardware MIDI control
- ✅ Complete benchmarking suite

## Common Issues and Solutions

### Issue: "No module named 'openai'"
**Solution**: AI features need API libraries. Either:
- Install: `pip install openai`
- Or test without AI features using mock components

### Issue: "No module named 'librosa'"
**Solution**: Advanced audio analysis needs librosa. Either:
- Install: `pip install librosa`
- Or use basic numpy-based analysis

### Issue: "python-rtmidi not installed"
**Solution**: This is just a warning. MIDI features optional:
- Install: `pip install python-rtmidi mido`
- Or ignore if not using hardware controllers

### Issue: Pattern creation fails
**Solution**: Make sure to include all required fields:
```python
pattern = HardcorePattern(
    name="required",
    bpm=180,  # required
    pattern_data="optional but recommended",
    synth_type=SynthType.GABBER_KICK,  # optional
    genre="gabber"  # required
)
```

## Performance Testing

For performance benchmarking without full dependencies:

```python
import time
import asyncio
from cli_shared.interfaces.synthesizer import MockSynthesizer

async def benchmark_mock():
    synth = MockSynthesizer()
    start = time.time()
    
    for i in range(100):
        audio = await synth.play_pattern(pattern)
    
    elapsed = time.time() - start
    print(f"100 patterns in {elapsed:.2f}s = {100/elapsed:.1f} patterns/sec")

asyncio.run(benchmark_mock())
```

## Next Steps

1. **Start with test_simplified.py** to verify core functionality
2. **Install minimal dependencies** based on what features you want
3. **Add API keys** if you want to test AI features
4. **Run component-specific tests** to explore individual systems
5. **Use the web interface** if you install FastAPI: `python3 -m web_interface.hardcore_web_app`

## Summary

The GABBERBOT system is designed to be modular and testable even without all dependencies. Start with the core functionality tests, then progressively add dependencies based on what features you want to explore. The mock implementations allow testing the architecture and flow even without real audio hardware or AI APIs.