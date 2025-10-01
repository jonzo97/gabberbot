# Review Story Task - QA Workflow

## Purpose
Validate story implementation against requirements, architecture, and quality standards for the Hardcore Music Production System.

## Review Workflow

### 1. Story Loading and Preparation

#### Load Story File:
```bash
# Load the story file for review
docs/bmad-development/stories/story-{phase}-{epic}-{sequence}.yaml
```

#### Review Context:
- Read story requirements and acceptance criteria
- Review architectural guidance
- Check testing requirements
- Note implementation details in Dev Agent Record

#### Gather References:
- `bmad-planning/02-architecture-spec.md` - Architecture patterns
- `bmad-planning/03-prd.md` - Product requirements
- `development/validation-testing-strategy.md` - Testing standards
- `CLAUDE.md` - Coding standards

### 2. Code Review

#### Structural Review:

**File Organization:**
- [ ] Files in correct directories
- [ ] Naming conventions followed
- [ ] Module structure logical

**Code Quality:**
```python
# Check for:
- Type hints on all functions
- Comprehensive docstrings
- No magic numbers
- DRY principle followed
- Single responsibility per function
```

**Architecture Compliance:**
- [ ] Patterns from Architecture Spec followed
- [ ] Multi-process architecture respected
- [ ] Separation of concerns maintained
- [ ] Event-driven communication used appropriately

#### Implementation Review:

**Requirements Coverage:**
- [ ] All functional requirements implemented
- [ ] Non-functional requirements met
- [ ] Edge cases handled
- [ ] Error handling comprehensive

**Music Domain Specifics:**
```python
# Verify:
- BPM handling (150-250 range)
- MIDI data using MIDIClip model
- Genre-appropriate parameters
- Audio quality standards
```

### 3. Testing Validation

#### Test Execution:

**Unit Tests:**
```bash
# Run unit tests with coverage
pytest tests/unit/ --cov=cli_shared --cov-report=term-missing

# Verify:
- [ ] 90% coverage for core modules
- [ ] All new code has tests
- [ ] Tests are meaningful (not just coverage)
```

**Integration Tests:**
```bash
# Run integration tests
pytest tests/integration/ -v

# Check:
- [ ] Component interactions work
- [ ] Data flows correctly
- [ ] No integration breaks
```

**Performance Tests:**
```python
# For real-time components
- Audio latency < 20ms
- Generation time < 500ms for complex
- Memory usage reasonable
- No memory leaks
```

#### Music-Specific Testing:

**Audio Quality:**
```python
# Test audio output for:
- No clipping or artifacts
- Proper frequency response
- Genre-appropriate sound
- Correct BPM and timing
```

**MIDI Validation:**
```python
# Verify MIDI data:
- Note values in range (0-127)
- Timing accurate to grid
- Velocity dynamics preserved
- Pattern structure correct
```

### 4. Acceptance Criteria Validation

For each criterion in story:
- [ ] Criterion met completely
- [ ] Measurable and verifiable
- [ ] Tested appropriately
- [ ] Edge cases considered

### 5. Regression Testing

#### Run Full Suite:
```bash
# Execute complete test suite
pytest tests/ --run-slow

# Ensure:
- [ ] No existing functionality broken
- [ ] All previous tests still pass
- [ ] Performance not degraded
```

#### Manual Testing:
For Phase-specific features:

**Phase 1 (Prototyper):**
```bash
python main.py "create 180 BPM hardcore kick pattern"
# Verify WAV file generated with correct pattern
```

**Phase 2 (Instrument):**
```python
# Test TUI responsiveness
# Verify multi-track mixing
# Check real-time playback
```

**Phase 3 (Partner):**
```python
# Test groove application
# Verify MIDI hardware response
# Check user preference learning
```

**Phase 4 (Studio):**
```python
# Test arrangement timeline
# Verify VST loading
# Check stem export quality
```

### 6. Quality Assessment

#### Code Metrics:
```python
# Analyze:
- Cyclomatic complexity < 10
- Function length < 50 lines
- Class cohesion high
- Coupling low
```

#### Documentation:
- [ ] All public functions documented
- [ ] Complex logic explained
- [ ] README updated if needed
- [ ] API changes documented

#### Security:
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] No SQL injection risks
- [ ] File operations safe

### 7. QA Results Documentation

Update ONLY the QA Results section of story:

```yaml
qa_results:
  reviewer: Casey
  review_date: 2024-01-15
  verdict: Pass  # Pass | Fail | Conditional Pass
  
  test_execution:
    unit_tests: Pass - 92% coverage
    integration_tests: Pass - All scenarios working
    regression_tests: Pass - No breaks detected
    performance_tests: Pass - Latency 15ms (< 20ms requirement)
    
  architectural_compliance:
    patterns_followed: Yes - MVC and multi-process correctly implemented
    constraints_met: Yes - All architectural constraints satisfied
    
  audio_validation:
    quality: Excellent - No artifacts, proper dynamics
    genre_authenticity: Confirmed - Hardcore kick meets standards
    
  issues_found:
    critical: []  # Blocks acceptance
    major: 
      - Missing error handling for invalid BPM values
    minor:
      - Could optimize pattern generation caching
    
  improvements_suggested: |
    - Add memoization to pattern generation for common requests
    - Consider implementing batch MIDI export
    - Document the new generator parameters in user guide
    
  final_notes: |
    Implementation meets all requirements with minor improvements needed.
    Code quality is high with good test coverage.
    Ready for production after addressing major issue.
```

### 8. Verdict Decision Criteria

#### Pass:
- All acceptance criteria met
- Tests passing with good coverage
- Architecture compliance confirmed
- No critical issues
- Performance requirements met

#### Conditional Pass:
- Minor issues that don't block functionality
- Can be addressed in follow-up
- Document conditions for full pass

#### Fail:
- Acceptance criteria not met
- Critical bugs or issues
- Architecture violations
- Performance below requirements
- Regression test failures

## Music Production Quality Standards

### Audio Requirements:
- **Frequency Response**: 20Hz-20kHz for full range
- **Dynamic Range**: Appropriate for genre (limited for hardcore)
- **Noise Floor**: < -60dB
- **THD**: < 1% at nominal levels

### MIDI Accuracy:
- **Timing**: ±1ms accuracy
- **Velocity**: Full 0-127 range support
- **Note Length**: Minimum 1/64 note resolution
- **Tempo**: Stable at all BPMs

### Genre Authenticity:

**Hardcore/Gabber:**
- Kick drum dominance
- Distortion appropriate
- BPM accurate (150-200)
- Rotterdam/Berlin style recognition

**Industrial:**
- Metallic textures present
- Warehouse reverb character
- Dark atmosphere maintained

## Review Checklist Summary

Before completing review:
- [ ] Code review complete
- [ ] All tests executed
- [ ] Acceptance criteria validated
- [ ] Regression suite passed
- [ ] Performance validated
- [ ] Audio quality confirmed
- [ ] Architecture compliance checked
- [ ] QA Results section updated
- [ ] Verdict decided and justified

## Common Issues to Check

### Performance:
- Memory leaks in audio processing
- Blocking operations in main thread
- Inefficient pattern generation
- Large buffer allocations

### Music Domain:
- Incorrect BPM calculations
- MIDI timing drift
- Audio buffer underruns
- Synthesis parameter ranges

### Architecture:
- Coupling between modules
- Bypassing abstraction layers
- Direct file I/O in wrong layer
- Synchronous operations in async context

## Success Criteria

A thorough review provides:
- Confidence in code quality
- Verification of requirements
- Constructive feedback
- Clear path forward
- Documentation of issues
- Learning for future stories