# Develop Story Task - Developer Workflow

## Purpose
Implement story requirements systematically while following architectural patterns and maintaining code quality for the Hardcore Music Production System.

## Pre-Development Checklist

Before starting implementation:
- [ ] Story file loaded and all sections reviewed
- [ ] Acceptance criteria understood
- [ ] Architectural guidance clear
- [ ] Existing code references identified
- [ ] Dependencies available
- [ ] Testing requirements understood

## Development Workflow

### 1. Story Analysis

#### Review Story Sections:
- **Requirements**: Understand functional and non-functional needs
- **Architecture**: Note patterns and constraints to follow
- **Existing Code**: Identify components to leverage
- **Dev Notes**: Review implementation hints
- **Testing**: Understand validation requirements

#### Check Dependencies:
```python
# Verify required stories are complete
# Check external dependencies are available
# Ensure no blocking issues
```

### 2. Implementation Planning

#### Task Breakdown:
Review the tasks section and create mental model:
1. Setup and prerequisites
2. Core implementation 
3. Testing and validation
4. Documentation and cleanup

#### Architecture Alignment:
From `bmad-planning/02-architecture-spec.md`:
- Multi-process architecture (Main, Background, Real-time)
- MVC pattern for components
- Separation of generation and performance pipelines
- Event-driven communication

### 3. Core Implementation

#### For Each Task:

##### Setup Phase:
```python
# 1. Create/modify file structure
# 2. Import required dependencies
# 3. Set up class/module structure
```

##### Implementation Phase:
```python
# Follow patterns from architecture spec
# Use existing components from cli_shared/
# Apply CLAUDE.md coding standards
```

##### Key Implementation Patterns:

**MIDI Generation:**
```python
from cli_shared.models.midi_clips import MIDIClip, MIDINote

# Always use MIDIClip for musical data
clip = MIDIClip(name="pattern", length_bars=4, bpm=180)
clip.add_note(MIDINote(pitch=36, velocity=127, start_time=0, duration=0.25))
```

**Pattern Generators:**
```python
from cli_shared.generators import AcidBasslineGenerator

# Extend existing generators
generator = AcidBasslineGenerator(
    scale="A_minor",
    accent_pattern="x ~ ~ x"
)
clip = generator.generate(length_bars=4, bpm=180)
```

**AI Integration:**
```python
# Consolidate into intelligent_music_agent_v2.py
from intelligent_music_agent_v2 import IntelligentMusicAgentV2

agent = IntelligentMusicAgentV2()
params = agent.text_to_params(user_input)
```

**Audio Synthesis:**
```python
from cli_shared.interfaces.synthesizer import AbstractSynthesizer

class NewSynthesizer(AbstractSynthesizer):
    # Implement required interface methods
    pass
```

### 4. Testing Implementation

#### Unit Tests:
```python
# Test each new function/class
# Aim for 90% coverage on core modules
# Use pytest framework
```

#### Integration Tests:
```python
# Test component interactions
# Verify data flow between modules
# Check error handling
```

#### Music-Specific Tests:
```python
# Validate BPM accuracy
# Check note timing and velocity
# Verify genre-appropriate parameters
# Test audio quality (no clipping)
```

### 5. Validation

#### Against Acceptance Criteria:
- [ ] Each criterion from story met
- [ ] All functional requirements working
- [ ] Non-functional requirements satisfied
- [ ] Edge cases handled

#### Against Architecture:
- [ ] Patterns correctly implemented
- [ ] Separation of concerns maintained
- [ ] Performance requirements met (<20ms for real-time)
- [ ] No architectural violations

### 6. Story File Updates

**ONLY update these sections:**

#### Task Checkboxes:
```yaml
tasks:
  - [x] Setup and prerequisites  # Mark complete
    - [x] Subtask 1
    - [x] Subtask 2
```

#### Dev Agent Record:
```yaml
dev_agent_record:
  agent: Morgan
  started: 2024-01-15T10:00:00Z
  completed: 2024-01-15T16:00:00Z
  
  implementation_notes: |
    - Used existing MIDIClip model for pattern storage
    - Extended AcidBasslineGenerator for new pattern type
    - Implemented caching for performance improvement
    
  debug_log: |
    - Issue: Import error with numpy
      Solution: Added numpy to requirements.txt
    
  file_list:
    created: 
      - cli_shared/generators/new_pattern.py
    modified:
      - intelligent_music_agent_v2.py
      - requirements.txt
    
  change_log:
    - date: 2024-01-15
      change: Added new pattern generator
      reason: Story requirement for additional pattern types
```

#### Status Update:
```yaml
status: Ready for Review  # When all tasks complete
```

### 7. Code Quality Checklist

Before marking complete:
- [ ] All tests passing
- [ ] Code follows CLAUDE.md standards
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] No magic numbers (use constants)
- [ ] Error handling implemented
- [ ] Performance validated
- [ ] No code duplication

## Music Production Specific Guidelines

### Audio Processing:
- Maintain sample rate consistency (44100/48000 Hz)
- Handle buffer sizes appropriately
- Prevent clipping (normalize output)
- Use appropriate bit depth

### MIDI Handling:
- Validate MIDI ranges (0-127)
- Handle timing accurately (use beats, not seconds)
- Preserve velocity dynamics
- Support all required MIDI channels

### Synthesis Parameters:
From CLAUDE.md hardcore parameters:
```python
HARDCORE_PARAMS = {
    'kick_sub_freqs': [41.2, 82.4, 123.6],
    'detune_cents': [-19, -10, -5, 0, 5, 10, 19, 29],
    'distortion_db': 15,
    'highpass_hz': 120,
    'compression_ratio': 8
}
```

### Genre Authenticity:
- Gabber: 150-200 BPM, extreme kick distortion
- Industrial: 130-150 BPM, metallic reverb
- Hardcore: 180-250 BPM, heavy compression

## Blocking Conditions

Stop and ask for help if:
- Requirements ambiguous after checking story and docs
- Architecture pattern unclear
- Performance requirement not achievable
- Regression test failing
- External dependency missing
- Technical debt blocking progress

## Completion Criteria

Story is complete when:
- [ ] All tasks marked [x]
- [ ] All tests passing
- [ ] Acceptance criteria validated
- [ ] Code review ready
- [ ] File list updated
- [ ] Status set to "Ready for Review"

## Common Implementation Patterns

### Async Operations:
```python
# For background generation tasks
async def generate_pattern(params):
    # Long-running generation
    return result
```

### IPC Communication:
```python
# Between main and background processes
from multiprocessing import Queue
job_queue = Queue()
result_queue = Queue()
```

### OSC Messaging:
```python
# To SuperCollider audio engine
from pythonosc import udp_client
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
client.send_message("/play", [note, velocity, duration])
```

## Success Metrics

Well-implemented story has:
- Clean, readable code
- Comprehensive tests
- Met all requirements
- Followed architecture
- Good performance
- Proper documentation
- No technical debt added