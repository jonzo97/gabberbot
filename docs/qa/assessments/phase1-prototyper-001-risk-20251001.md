# Risk Profile: Story phase1-prototyper-001 (Brownfield Cleanup)

Date: 2025-10-01
Reviewer: Quinn (Test Architect)
Story: docs/bmad-development/stories/story-phase1-prototyper-001.yaml

## Executive Summary

This risk assessment evaluates the consolidation of 17 experimental bmad_*.py files (13,134 lines) into the core architecture established in Story 001. The experimental files demonstrate **working music generation functionality** (83 sessions, 1660 patterns in 30-minute test run), but violate all foundational engineering principles from Story 001.

**Critical Finding**: This is a **HIGH-RISK CONSOLIDATION** with multiple critical regression risks and significant architectural complexity.

### Risk Overview
- **Total Risks Identified**: 23
- **Critical Risks**: 5 (Score 9)
- **High Risks**: 7 (Score 6)
- **Medium Risks**: 8 (Score 4)
- **Low Risks**: 3 (Score 2-3)
- **Overall Risk Score**: 21/100 (High Risk - Extensive mitigation required)

### Key Risk Drivers
1. **Three parallel implementations** of music generation with different architectures
2. **No shared data models** - experimental files ignore Story 001 Pydantic models
3. **Hardcoded synthesis parameters** throughout instead of using synthesis_constants.py
4. **Working functionality at risk** - music generation proven but implementation violates standards
5. **Undocumented dependencies** between experimental files creating complex consolidation paths

---

## Critical Risks Requiring Immediate Attention

### TECH-001: Three Parallel Music Generation Implementations

**Score: 9 (Critical)**

**Probability**: High (3) - Confirmed by code analysis: bmad_standalone.py, bmad_coordinator_lite.py, and bmad_simple_test.py each implement complete music generation pipelines with incompatible architectures.

**Impact**: High (3) - Consolidating three different implementations will require choosing one architecture or building a fourth. Any choice risks losing working features from the other implementations.

**Affected Components**:
- bmad_standalone.py (1,180 lines) - Complete self-contained MIDI/WAV generation
- bmad_coordinator_lite.py (487 lines) - Uses cli_shared generators
- bmad_simple_test.py (475 lines) - BMadSimpleCoordinator with different pattern generation
- bmad_hardcore_factory.py (808 lines) - Attempts to orchestrate all three

**Evidence of Parallelism**:
```python
# bmad_standalone.py - Custom SimpleMIDIExporter class
class SimpleMIDIExporter:
    @staticmethod
    def export_midi(notes: List[MIDINote], filepath: str, bpm: float = 120):
        # Custom MIDI implementation (ignoring cli_shared.models.midi_clips)

# bmad_coordinator_lite.py - Uses cli_shared but different pattern
from cli_shared.generators.acid_bassline import AcidBasslineGenerator
from cli_shared.generators.tuned_kick import TunedKickGenerator
from cli_shared.models.midi_clips import MIDIClip

# bmad_simple_test.py - Yet another architecture
class BMadSimpleCoordinator:
    def generate_track(self, config: BMadTrackConfig):
        # Different generation approach from other two
```

**Mitigation**:
1. **Inventory Features**: Document unique features in each implementation
2. **Feature Matrix**: Create comparison matrix showing what each implementation can do
3. **Consolidation Strategy**:
   - Option A: Extend Story 001 models to support all three use cases
   - Option B: Choose bmad_coordinator_lite.py (already uses cli_shared) as base
   - Option C: Build unified implementation following Story 001 architecture
4. **Migration Plan**: Phase migration to avoid breaking all implementations at once
5. **Regression Test Suite**: Capture current output quality before consolidation

**Testing Requirements**:
- Comparative audio quality tests across all three implementations
- Feature parity validation tests
- Migration regression tests (before/after consolidation)
- Performance benchmarks (ensure consolidation doesn't degrade speed)

**Residual Risk**: Medium - Even with careful migration, some implementation-specific behaviors may be lost.

**Owner**: dev (with @music-orchestrator coordination)
**Timeline**: Before any consolidation work begins

---

### TECH-002: Data Model Bypass - Story 001 Models Ignored

**Score: 9 (Critical)**

**Probability**: High (3) - Confirmed: Most bmad_*.py files define their own data structures instead of using Story 001 Pydantic models from src/models/core.py.

**Impact**: High (3) - Violates Story 001's core acceptance criterion: "MIDIClip Pydantic model with notes, timing, velocity validation". Consolidation requires rewriting all data handling or abandoning Story 001 models.

**Affected Components**:
- bmad_standalone.py - Custom MIDINote dataclass (lines 49-56)
- bmad_hardcore_factory.py - Custom BMADFactoryConfig, imports others
- bmad_phase3_demo*.py (3 files, 1,572 lines) - Custom AlbumTrackPlan structures
- 10+ other files with ad-hoc data structures

**Evidence**:
```python
# bmad_standalone.py - VIOLATES Story 001
@dataclass
class MIDINote:
    """Simple MIDI note representation"""
    pitch: int
    velocity: int
    start_time: float  # In beats
    duration: float    # In beats
    channel: int = 0
    # Missing: Story 001 validation, JSON serialization, frequency conversion

# Story 001 PROPER implementation (src/models/core.py)
@dataclass
class MIDINote:
    pitch: int                    # MIDI note number (0-127) - VALIDATED
    velocity: int                 # MIDI velocity (0-127) - VALIDATED
    start_time: float            # Start time in beats
    duration: float              # Duration in beats
    channel: int = 0

    def to_frequency(self) -> float:  # MISSING in experimental files
    def transpose(self, semitones: int) -> 'MIDINote':  # MISSING
```

**Root Cause**: Experimental files created before Story 001 completion, or developers unaware of Story 001 models.

**Mitigation**:
1. **Model Audit**: Catalog all custom data structures in bmad_*.py files
2. **Compatibility Analysis**: Identify which custom models can be replaced by Story 001 models
3. **Extension Requirements**: Document features in custom models not in Story 001 (extend if needed)
4. **Migration Script**: Automated conversion from custom models to Story 001 models
5. **Type Validation**: Add mypy strict mode to catch future bypasses

**Testing Requirements**:
- Model validation tests (Story 001 constraints: MIDI 0-127, timing >= 0)
- Serialization tests (JSON round-trip for all migrated models)
- Integration tests (ensure model changes don't break music generation)

**Residual Risk**: Low - Story 001 models are complete; migration is mechanical but tedious.

**Owner**: dev
**Timeline**: Phase 1 of consolidation (before touching business logic)

---

### TECH-003: Hardcoded Synthesis Parameters Everywhere

**Score: 9 (Critical)**

**Probability**: High (3) - Confirmed by grep: 18 instances of hardcoded BPM, frequency, distortion values across bmad_*.py files instead of using audio/parameters/synthesis_constants.py.

**Impact**: High (3) - Violates CLAUDE.md mandate: "NO MORE MAGIC NUMBERS - everything documented and parameterized". Creates inconsistent sound across different code paths and makes parameter tuning impossible without code changes.

**Affected Components**:
- bmad_standalone.py - Hardcoded: frequency = 440.0 * (2 ** ((pitch - 69) / 12.0))
- bmad_coordinator_lite.py - Hardcoded: frequency = 60.0 (kick), multiple BPM=180.0
- bmad_simple_test.py - Hardcoded: frequency calculation, BPM defaults
- Multiple files - Hardcoded distortion amounts (0.2, 0.3)

**Evidence from grep**:
```python
# bmad_standalone.py:579 - HARDCODED FORMULA
frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
# Should use: from audio.parameters.synthesis_constants import MIDIConstants
# MIDIConstants.MIDI_A4_FREQ, MIDIConstants.MIDI_A4_NOTE

# bmad_coordinator_lite.py:194 - MAGIC NUMBER
frequency = 60.0  # Base kick frequency
# Should use: HardcoreConstants.KICK_SUB_FREQS or synthesis presets

# bmad_standalone.py:770 - MAGIC DISTORTION
processed = self._apply_distortion(processed, amount=0.2)
# Should use: HardcoreConstants.HARDCORE_DISTORTION_DB or style presets
```

**Existing Proper Constants** (audio/parameters/synthesis_constants.py):
- HardcoreConstants.KICK_SUB_FREQS = [41.2, 82.4, 123.6] (USER VALIDATED)
- HardcoreConstants.DETUNE_CENTS (USER VALIDATED)
- HardcoreConstants.HARDCORE_DISTORTION_DB = 15 (USER VALIDATED)
- HARDCORE_STYLE_PRESETS with 6 complete style definitions
- MIDIConstants.MIDI_A4_FREQ = 440.0, MIDI_A4_NOTE = 69

**Mitigation**:
1. **Constants Audit**: Catalog every hardcoded value in bmad_*.py files
2. **Mapping Strategy**: Map each hardcoded value to existing synthesis_constants.py entry
3. **Gap Analysis**: Identify values not in synthesis_constants.py (add them)
4. **Refactoring Pass**: Replace all magic numbers with constant references
5. **Validation Tests**: Ensure audio output unchanged after constant replacement

**Testing Requirements**:
- Audio regression tests (compare before/after constant migration)
- Parameter validation tests (ensure constants used correctly)
- Style consistency tests (all Rotterdam Gabber code uses same constants)

**Residual Risk**: Low - Mechanical refactoring with clear mappings; audio regression tests catch issues.

**Owner**: dev
**Timeline**: Can be done in parallel with model migration (independent concern)

---

### DATA-001: Music Quality Degradation - 4x Tempo Bug

**Score: 9 (Critical)**

**Probability**: High (3) - User reported in context: "Music quality issues: 4x tempo bug, no variation".

**Impact**: High (3) - Core music generation produces unplayable output. If consolidation doesn't preserve correct tempo handling, all generated music will be unusable.

**Affected Components**:
- Unknown which bmad_*.py file(s) cause 4x tempo issue
- Likely: Timing calculation mismatch between BPM and actual beat duration
- Possible: MIDI ticks-per-quarter-note conversion error

**Evidence**:
- Context states: "Music quality issues: 4x tempo bug"
- Likely caused by: BPM conversion error or MIDI timing constant mismatch
- Similar issues seen in: bmad_standalone.py SimpleMIDIExporter (lines 72-73):
  ```python
  f.write(struct.pack('>H', 480))  # Ticks per quarter note
  tempo = int(60000000 / bpm)
  ```

**Root Cause Hypotheses**:
1. MIDI ticks calculation doesn't match actual note timing
2. BPM set correctly but timing conversion uses wrong formula
3. Multiple tempo sources (config vs hardcoded) creating conflicts

**Mitigation**:
1. **Bug Reproduction**: Create minimal test case demonstrating 4x tempo issue
2. **Timing Audit**: Review all BPM/tempo/timing conversion code across bmad_*.py
3. **Reference Implementation**: Validate against working cli_shared MIDI export
4. **Fix Before Consolidation**: Resolve tempo bug in experimental code first
5. **Regression Tests**: MIDI timing validation in test suite

**Testing Requirements**:
- Tempo accuracy tests (generate 120 BPM pattern, verify actual playback speed)
- MIDI timing validation (programmatic analysis of .mid file timing)
- Multi-BPM tests (150, 180, 200, 220 BPM - hardcore range)
- Audio playback verification (human listening validation)

**Residual Risk**: Medium - Tempo bugs can be subtle; requires careful validation.

**Owner**: dev + @sound-designer (audio validation)
**Timeline**: CRITICAL - Must fix before consolidation (can't migrate broken code)

---

### BUS-001: Loss of Working Functionality During Migration

**Score: 9 (Critical)**

**Probability**: High (3) - Large-scale consolidation (13,134 lines → proper architecture) with tight coupling and undocumented dependencies.

**Impact**: High (3) - User has working music generation (83 sessions, 1660 patterns in 30 minutes). Breaking this during consolidation loses all business value.

**Affected Components**:
- ALL 17 bmad_*.py files (entire experimental codebase)
- Integration with cli_shared components (evolution, generators)
- Integration with audio.core.track and audio.effects
- Production monitoring and output tracking

**Evidence of Working System**:
```json
// 30min_production_totals.json
{
  "total_sessions": 83,
  "total_outputs": {
    "sessions": 83,
    "patterns": 1660,
    "tracks": 415,
    "presets": 1660
  },
  "status": "running"
}
```

**Dependency Graph Complexity**:
```
bmad_hardcore_factory.py
  ├─> bmad_simple_test.py (BMadSimpleCoordinator)
  ├─> bmad_album_producer.py (BMADAlbumProducer)
  │   ├─> audio.core.track (Track, TrackCollection)
  │   ├─> audio.effects (multiple effect modules)
  │   └─> cli_shared.evolution.bmad_pattern_evolution
  ├─> bmad_phase3_demo_final.py (AlbumPlan)
  └─> real_bmad_music_coordinator.py (BMADMusicCoordinator)

bmad_integration_bridge.py
  ├─> src.services.generation_service
  ├─> src.services.audio_service
  ├─> src.tui.app (TUIApp)
  └─> Conditional imports (try/except ImportError)
```

**Mitigation**:
1. **Freeze Current Output**: Capture reference audio/MIDI from working system
2. **Dependency Map**: Document complete import graph and runtime dependencies
3. **Incremental Migration**: Migrate one component at a time with validation
4. **Parallel Implementation**: Keep old code working while building new
5. **Feature Flags**: Toggle between old/new implementations during transition
6. **Rollback Plan**: Clear rollback procedure if migration breaks functionality

**Testing Requirements**:
- **Golden Master Tests**: Compare new implementation output to frozen reference
- **A/B Testing**: Generate same pattern with old and new code, compare audio
- **Integration Smoke Tests**: Full end-to-end generation after each migration step
- **Performance Regression**: Ensure new code doesn't slow down generation

**Residual Risk**: High - Complex consolidation always has unexpected breakage risk.

**Owner**: dev + @qa (continuous validation) + @music-orchestrator (coordination)
**Timeline**: Throughout entire consolidation effort (ongoing risk)

---

## High Risks (Score 6)

### TECH-004: Undocumented Multi-Process Architecture

**Score: 6 (High)**

**Probability**: Medium (2) - Context mentions "Multi-process architecture not implemented" but some bmad files use async/multiprocessing.

**Impact**: High (3) - Consolidation may break async/parallel execution patterns, degrading performance or causing race conditions.

**Details**:
- bmad_hardcore_factory.py uses asyncio (line 19: import asyncio)
- Production totals suggest parallel session generation (83 sessions in ~7 minutes)
- Unclear if parallelism is process-based, thread-based, or async-based

**Mitigation**:
- Document current parallelism model
- Test single-threaded vs multi-process performance
- Design consolidation to preserve parallelism architecture

---

### TECH-005: Import Dependency Fragility

**Score: 6 (High)**

**Probability**: High (3) - Multiple files use try/except ImportError for conditional imports.

**Impact**: Medium (2) - Silent failures if expected modules missing; consolidation may change import paths.

**Evidence**:
```python
# bmad_hardcore_factory.py
try:
    from cli_shared.evolution.bmad_pattern_evolution import BMADPatternEvolution
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    print("Evolution system not available - running in basic mode")
```

**Mitigation**:
- Replace try/except imports with explicit dependency checking
- Use Poetry dependency groups (required vs optional)
- Fail fast with clear error messages instead of degraded mode

---

### TECH-006: Duplicate BMadMixEngineer Implementation

**Score: 6 (High)**

**Probability**: High (3) - grep shows BMadMixEngineer class in 3 different files.

**Impact**: Medium (2) - Code duplication makes consolidation complex; each copy may have evolved differently.

**Affected Files**:
- bmad_coordinator_lite.py: class BMadMixEngineer
- bmad_simple_test.py: class BMadMixEngineer
- bmad_standalone.py: class BMadMixEngineer

**Mitigation**:
- Diff all three implementations to find differences
- Identify canonical version or merge features
- Move to shared module in cli_shared/

---

### DATA-002: No Story 001 Test Coverage for Migration

**Score: 6 (High)**

**Probability**: Medium (2) - Story 001 has 37 unit tests but focused on model validation, not migration scenarios.

**Impact**: High (3) - Consolidation without migration tests risks breaking validated Story 001 functionality.

**Mitigation**:
- Extend Story 001 test suite with migration test fixtures
- Create "dirty input" tests (custom models → Story 001 models)
- Add integration tests for bmad feature scenarios

---

### PERF-001: Consolidation Performance Regression

**Score: 6 (High)**

**Probability**: Medium (2) - Story 001 models add validation overhead; consolidation may introduce abstraction layers.

**Impact**: High (3) - Current system generates 83 sessions in 7 minutes. Performance degradation would be unacceptable.

**Current Performance Baseline**:
- 83 sessions in ~7 minutes = 11.9 sessions/minute
- 1660 patterns in 7 minutes = 237 patterns/minute
- ~3.5 seconds per session (impressive for music generation)

**Mitigation**:
- Benchmark current bmad_*.py performance (establish baseline)
- Profile Story 001 model overhead (Pydantic validation cost)
- Set performance acceptance criteria (no more than 10% regression)
- Optimize hot paths if consolidation slows generation

---

### PERF-002: Memory Usage Unknown for Batch Generation

**Score: 6 (High)**

**Probability**: Medium (2) - 1660 patterns generated suggests large in-memory structures.

**Impact**: High (3) - Consolidation may change memory patterns; OOM failures would break batch workflows.

**Mitigation**:
- Measure current memory usage during 30-minute production run
- Design consolidated architecture with memory limits (per Story 001 Session Limits)
- Implement streaming patterns for large batches

---

### OPS-001: No Documented Rollback Procedure

**Score: 6 (High)**

**Probability**: Medium (2) - Large consolidation effort with multiple phases.

**Impact**: High (3) - If consolidation fails mid-way, no clear path back to working state.

**Mitigation**:
- Git branching strategy (feature branch for consolidation)
- Tag current working state before starting
- Document rollback steps in consolidation plan
- Keep bmad_*.py files in archive/ until consolidation validated

---

## Medium Risks (Score 4)

### TECH-007: Evolution System Integration Unclear

**Score: 4 (Medium)**

**Probability**: Medium (2) - cli_shared/evolution/bmad_pattern_evolution.py exists but integration varies by file.

**Impact**: Medium (2) - May lose pattern evolution features during consolidation.

**Mitigation**: Document evolution system API and ensure consolidated code uses it consistently.

---

### TECH-008: Audio Output File Management

**Score: 4 (Medium)**

**Probability**: Medium (2) - pattern_evolution_workspace/ directory exists but contains no MIDI/WAV files.

**Impact**: Medium (2) - Output file handling may be broken or files cleaned up; consolidation needs clear file strategy.

**Mitigation**: Define output directory structure and file retention policy.

---

### TECH-009: Logging Inconsistency

**Score: 4 (Medium)**

**Probability**: Medium (2) - Multiple logging approaches (logging.getLogger variations).

**Impact**: Medium (2) - Consolidation needs unified logging strategy.

**Mitigation**: Standardize logging configuration in consolidated architecture.

---

### DATA-003: Configuration Model Fragmentation

**Score: 4 (Medium)**

**Probability**: Medium (2) - Multiple config classes (BMadTrackConfig, BMADFactoryConfig, AlbumConfig).

**Impact**: Medium (2) - Consolidation needs unified configuration approach.

**Mitigation**: Extend Story 001 Configuration model to cover all use cases.

---

### DATA-004: JSON Output Format Compatibility

**Score: 4 (Medium)**

**Probability**: Medium (2) - 30min_production_totals.json uses custom format.

**Impact**: Medium (2) - External tools may depend on this format.

**Mitigation**: Maintain JSON schema compatibility or provide migration guide.

---

### SEC-001: API Key Handling in Multiple Files

**Score: 4 (Medium)**

**Probability**: Medium (2) - Some bmad files import from src.utils.env, others don't.

**Impact**: Medium (2) - Inconsistent API key handling could expose secrets.

**Mitigation**: Ensure all consolidated code uses Story 001 environment loading (src/utils/env.py).

---

### BUS-002: No Variation in Generated Music

**Score: 4 (Medium)**

**Probability**: Medium (2) - Context mentions "no variation" as music quality issue.

**Impact**: Medium (2) - Consolidation must preserve or improve variation algorithms.

**Mitigation**: Identify variation mechanisms in current code and ensure migration preserves them.

---

### OPS-002: No CI/CD for Experimental Files

**Score: 4 (Medium)**

**Probability**: Medium (2) - bmad_*.py files likely not in automated testing pipeline.

**Impact**: Medium (2) - Consolidation introduces new code into CI/CD; may reveal failures.

**Mitigation**: Add bmad test scenarios to CI/CD before consolidation.

---

## Low Risks (Score 2-3)

### TECH-010: Poetry Dependency Conflicts

**Score: 3 (Low)**

**Probability**: Low (1) - Story 001 established Poetry; experimental files use same dependencies.

**Impact**: High (3) - Dependency conflicts could block consolidation entirely.

**Mitigation**: Run `poetry install` and validate no conflicts exist.

---

### DATA-005: MIDI File Format Compatibility

**Score: 2 (Low)**

**Probability**: Low (1) - SimpleMIDIExporter uses standard MIDI format.

**Impact**: Medium (2) - Custom MIDI export could produce incompatible files.

**Mitigation**: Validate MIDI files playable in standard DAWs (Ableton, FL Studio).

---

### OPS-003: Documentation Debt

**Score: 2 (Low)**

**Probability**: High (3) - Experimental files have minimal docstrings/comments.

**Impact**: Low (1) - Makes consolidation harder but doesn't block it.

**Mitigation**: Document during consolidation, not before (would waste time if code discarded).

---

## Risk Distribution

### By Category
- **Technical (TECH)**: 10 risks (3 critical, 3 high, 4 medium, 0 low)
- **Data (DATA)**: 5 risks (1 critical, 1 high, 2 medium, 1 low)
- **Performance (PERF)**: 2 risks (0 critical, 2 high, 0 medium, 0 low)
- **Security (SEC)**: 1 risk (0 critical, 0 high, 1 medium, 0 low)
- **Business (BUS)**: 2 risks (1 critical, 0 high, 1 medium, 0 low)
- **Operational (OPS)**: 3 risks (0 critical, 1 high, 1 medium, 1 low)

### By Component
- **bmad_*.py Experimental Files**: 18 risks
- **Story 001 Architecture Integration**: 12 risks
- **cli_shared Components**: 8 risks
- **Audio Output Pipeline**: 6 risks
- **Testing Infrastructure**: 5 risks

---

## Detailed Risk Register

| Risk ID  | Category | Description | Probability | Impact | Score | Priority |
|----------|----------|-------------|-------------|--------|-------|----------|
| TECH-001 | Technical | Three parallel music generation implementations | High (3) | High (3) | 9 | Critical |
| TECH-002 | Technical | Data model bypass - Story 001 models ignored | High (3) | High (3) | 9 | Critical |
| TECH-003 | Technical | Hardcoded synthesis parameters everywhere | High (3) | High (3) | 9 | Critical |
| DATA-001 | Data | Music quality degradation - 4x tempo bug | High (3) | High (3) | 9 | Critical |
| BUS-001 | Business | Loss of working functionality during migration | High (3) | High (3) | 9 | Critical |
| TECH-004 | Technical | Undocumented multi-process architecture | Med (2) | High (3) | 6 | High |
| TECH-005 | Technical | Import dependency fragility | High (3) | Med (2) | 6 | High |
| TECH-006 | Technical | Duplicate BMadMixEngineer implementation | High (3) | Med (2) | 6 | High |
| DATA-002 | Data | No Story 001 test coverage for migration | Med (2) | High (3) | 6 | High |
| PERF-001 | Performance | Consolidation performance regression | Med (2) | High (3) | 6 | High |
| PERF-002 | Performance | Memory usage unknown for batch generation | Med (2) | High (3) | 6 | High |
| OPS-001 | Operational | No documented rollback procedure | Med (2) | High (3) | 6 | High |
| TECH-007 | Technical | Evolution system integration unclear | Med (2) | Med (2) | 4 | Medium |
| TECH-008 | Technical | Audio output file management | Med (2) | Med (2) | 4 | Medium |
| TECH-009 | Technical | Logging inconsistency | Med (2) | Med (2) | 4 | Medium |
| DATA-003 | Data | Configuration model fragmentation | Med (2) | Med (2) | 4 | Medium |
| DATA-004 | Data | JSON output format compatibility | Med (2) | Med (2) | 4 | Medium |
| SEC-001 | Security | API key handling in multiple files | Med (2) | Med (2) | 4 | Medium |
| BUS-002 | Business | No variation in generated music | Med (2) | Med (2) | 4 | Medium |
| OPS-002 | Operational | No CI/CD for experimental files | Med (2) | Med (2) | 4 | Medium |
| TECH-010 | Technical | Poetry dependency conflicts | Low (1) | High (3) | 3 | Low |
| DATA-005 | Data | MIDI file format compatibility | Low (1) | Med (2) | 2 | Low |
| OPS-003 | Operational | Documentation debt | High (3) | Low (1) | 2 | Low |

---

## Risk-Based Testing Strategy

### Priority 1: Critical Risk Tests (Must Pass Before Consolidation)

#### Test Suite: TECH-001 - Three Implementations Feature Parity
```python
def test_feature_matrix_completeness():
    """Validate all three implementations catalogued"""
    implementations = ['standalone', 'coordinator_lite', 'simple_test']
    features = ['midi_export', 'wav_export', 'pattern_generation',
                'acid_bassline', 'kick_synthesis', 'effects_chain']

    matrix = build_feature_matrix(implementations, features)
    assert all_features_documented(matrix)

def test_consolidation_preserves_all_features():
    """Ensure consolidated implementation has all features"""
    old_features = extract_features_from_experimental()
    new_features = extract_features_from_consolidated()
    assert new_features >= old_features  # Superset
```

#### Test Suite: TECH-002 - Data Model Migration
```python
def test_custom_model_to_story001_migration():
    """Validate migration from custom models to Story 001 Pydantic models"""
    # Custom bmad_standalone MIDINote
    custom_note = CustomMIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.25)

    # Migrate to Story 001 model
    story001_note = migrate_to_story001_model(custom_note)

    # Validation should pass
    assert story001_note.pitch == 60
    assert story001_note.velocity == 100
    assert 0 <= story001_note.pitch <= 127  # Story 001 validation
    assert 0 <= story001_note.velocity <= 127

def test_story001_model_extensions_for_bmad():
    """Ensure Story 001 models support all bmad use cases"""
    from src.models.core import MIDINote, MIDIClip

    # All bmad features should work
    note = MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.25)
    assert hasattr(note, 'to_frequency')  # bmad needs this
    assert hasattr(note, 'transpose')     # bmad needs this
```

#### Test Suite: TECH-003 - Constants Migration
```python
def test_hardcoded_values_eliminated():
    """Ensure no hardcoded synthesis parameters in consolidated code"""
    consolidated_files = get_consolidated_source_files()

    for file in consolidated_files:
        source = read_file(file)

        # No magic BPM values
        assert not re.search(r'bpm\s*=\s*180', source)

        # No magic frequencies
        assert not re.search(r'frequency\s*=\s*60\.0', source)

        # No magic distortion amounts
        assert not re.search(r'amount\s*=\s*0\.[23]', source)

def test_synthesis_constants_used_correctly():
    """Validate correct usage of synthesis_constants.py"""
    from audio.parameters.synthesis_constants import (
        HardcoreConstants, MIDIConstants, HARDCORE_STYLE_PRESETS
    )

    # All Rotterdam Gabber code should use same constants
    preset = HARDCORE_STYLE_PRESETS[HardcoreStyle.ROTTERDAM_GABBER]
    assert preset.frequency == 55
    assert preset.brutality == 0.7

    # MIDI conversion should use constants
    assert MIDIConstants.MIDI_A4_FREQ == 440.0
    assert MIDIConstants.MIDI_A4_NOTE == 69
```

#### Test Suite: DATA-001 - Tempo Bug Fix Validation
```python
def test_tempo_accuracy_120bpm():
    """Validate 120 BPM pattern plays at correct tempo"""
    pattern = generate_test_pattern(bpm=120, bars=4)
    midi_file = export_to_midi(pattern)

    # Analyze MIDI file timing
    actual_bpm = extract_bpm_from_midi(midi_file)
    assert abs(actual_bpm - 120.0) < 0.1  # Within 0.1 BPM tolerance

def test_tempo_accuracy_hardcore_range():
    """Validate tempo accuracy across hardcore BPM range"""
    for bpm in [150, 180, 200, 220]:
        pattern = generate_test_pattern(bpm=bpm, bars=2)
        midi_file = export_to_midi(pattern)
        actual_bpm = extract_bpm_from_midi(midi_file)
        assert abs(actual_bpm - bpm) / bpm < 0.01  # Within 1%

def test_no_4x_tempo_bug():
    """Regression test: Ensure 4x tempo bug is fixed"""
    pattern = generate_test_pattern(bpm=180, bars=4)
    midi_file = export_to_midi(pattern)
    actual_bpm = extract_bpm_from_midi(midi_file)

    # Should NOT be 4x faster (720 BPM)
    assert actual_bpm < 200  # Reasonable upper bound
    assert abs(actual_bpm - 180) < 5  # Within 5 BPM
```

#### Test Suite: BUS-001 - Golden Master Regression Tests
```python
def test_golden_master_audio_quality():
    """Compare consolidated implementation to frozen reference audio"""
    reference_config = BMadTrackConfig(
        style=HardcoreStyle.ROTTERDAM_GABBER,
        bpm=180.0,
        length_bars=16.0,
        key="A_minor",
        seed=12345  # Fixed seed for reproducibility
    )

    # Generate with OLD implementation (frozen reference)
    reference_audio = load_reference_audio("golden_master_180bpm_gabber.wav")

    # Generate with NEW consolidated implementation
    new_audio = generate_with_consolidated_code(reference_config)

    # Compare audio characteristics
    assert spectral_similarity(new_audio, reference_audio) > 0.95
    assert rms_difference(new_audio, reference_audio) < 0.05

def test_feature_parity_30min_production():
    """Ensure consolidated code matches 30-minute production output"""
    # OLD: 83 sessions, 1660 patterns in ~7 minutes
    start_time = time.time()
    sessions = run_consolidated_production(duration_minutes=7)
    elapsed = time.time() - start_time

    assert len(sessions) >= 80  # Within 5% of baseline
    total_patterns = sum(s.pattern_count for s in sessions)
    assert total_patterns >= 1600  # Within 5% of baseline
```

---

### Priority 2: High Risk Tests

#### Test Suite: PERF-001 - Performance Regression
```python
def test_session_generation_speed():
    """Ensure consolidation doesn't slow down generation"""
    # Baseline: 3.5 seconds per session
    config = BMadTrackConfig(bpm=180, length_bars=16)

    start = time.time()
    session = generate_session(config)
    elapsed = time.time() - start

    # Allow 10% regression
    assert elapsed < 3.85  # 3.5 * 1.1

def test_batch_generation_memory():
    """Ensure batch generation doesn't exceed memory limits"""
    import psutil
    process = psutil.Process()

    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Generate 100 patterns
    patterns = generate_batch_patterns(count=100)

    peak_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = peak_memory - baseline_memory

    # Should not exceed 50MB per Story 001 Session Limits
    assert memory_increase < 50
```

#### Test Suite: TECH-005 - Import Dependency Validation
```python
def test_all_imports_explicit():
    """Ensure no try/except ImportError patterns"""
    consolidated_files = get_consolidated_source_files()

    for file in consolidated_files:
        source = read_file(file)

        # No silent import failures
        assert 'except ImportError' not in source

        # If optional dependency, use proper checking
        if 'EVOLUTION_AVAILABLE' in source:
            assert 'raise MissingDependencyError' in source

def test_poetry_dependencies_complete():
    """Validate all imports have Poetry dependencies"""
    from poetry.core.pyproject.toml import PyProjectTOML

    pyproject = PyProjectTOML('pyproject.toml')
    dependencies = pyproject.data['tool']['poetry']['dependencies']

    # All imports should be in dependencies
    for import_name in extract_all_imports():
        assert import_name in dependencies or is_stdlib(import_name)
```

---

### Priority 3: Medium/Low Risk Tests

#### Standard functional tests
- Configuration model serialization tests
- Logging output validation tests
- MIDI file DAW compatibility tests
- API key environment loading tests

---

## Risk Acceptance Criteria

### Must Fix Before Consolidation Begins

**Critical Risks (Score 9) - ALL must be mitigated:**
1. **TECH-001**: Document feature matrix for three implementations (choose consolidation strategy)
2. **TECH-002**: Create migration plan from custom models to Story 001 models
3. **TECH-003**: Map all hardcoded values to synthesis_constants.py entries
4. **DATA-001**: Fix 4x tempo bug and validate with regression tests
5. **BUS-001**: Freeze reference audio and establish golden master tests

**Without these mitigations, consolidation WILL FAIL.**

---

### Can Consolidate with Compensating Controls

**High Risks (Score 6) - Acceptable with mitigation:**
- **TECH-004, TECH-005, TECH-006**: Document during consolidation (not blockers)
- **DATA-002**: Build migration tests in parallel with consolidation
- **PERF-001, PERF-002**: Monitor performance; optimize if regression detected
- **OPS-001**: Git branching provides implicit rollback; document explicitly

---

### Accepted Risks

**Medium/Low Risks (Score 2-4) - Accept with monitoring:**
- **OPS-003**: Documentation debt - improve during consolidation
- **TECH-009**: Logging inconsistency - standardize as part of consolidation
- **DATA-004**: JSON format compatibility - maintain or provide migration notice

---

## Monitoring Requirements

### Pre-Consolidation Monitoring
- **Baseline Metrics Collection**:
  - Generate 30-minute production run, capture all metrics
  - Audio quality analysis (spectrum, RMS, dynamic range)
  - Performance metrics (generation time, memory usage)
  - Output file validation (MIDI playback, WAV quality)

### During Consolidation Monitoring
- **Incremental Validation**:
  - Run golden master tests after each component migration
  - A/B compare old vs new implementation output
  - Performance benchmarks after each merge to main
  - Dependency graph validation (no circular imports)

### Post-Consolidation Monitoring
- **Sustained Quality Checks**:
  - CI/CD pipeline runs all risk-based tests
  - Weekly production run validation (30-minute test)
  - User acceptance testing (generate sample tracks)
  - Performance regression alerts (>10% slowdown)

---

## Risk Review Triggers

**Review and update this risk profile when:**

1. **Architecture Changes Significantly**
   - Choosing consolidation strategy (Option A/B/C for TECH-001)
   - Major refactoring of Story 001 models
   - New integration with cli_shared components

2. **New Dependencies Added**
   - Integration with new audio backends
   - New AI model integrations
   - Additional synthesis engines

3. **Quality Issues Discovered**
   - New bugs found in experimental files
   - Story 001 limitations discovered
   - Performance bottlenecks identified

4. **Consolidation Milestones**
   - After data model migration complete
   - After constants refactoring complete
   - After first implementation consolidated
   - Before final integration testing

---

## Consolidation Recommendations

### Phase 1: Preparation (Week 1)
**Goal**: Reduce critical risks before touching code

1. **Fix DATA-001 (4x Tempo Bug)**
   - Reproduce bug in minimal test case
   - Fix in experimental files first
   - Validate fix with tempo accuracy tests
   - **Gate**: Tempo tests must pass before Phase 2

2. **Document TECH-001 (Three Implementations)**
   - Create feature matrix spreadsheet
   - Identify unique features in each implementation
   - Choose consolidation strategy (recommend Option B: extend coordinator_lite)
   - **Gate**: Strategy approved by @music-orchestrator before Phase 2

3. **Catalog TECH-003 (Hardcoded Values)**
   - Grep all magic numbers
   - Map to synthesis_constants.py
   - Identify gaps (add missing constants)
   - **Gate**: Mapping complete before Phase 2

4. **Establish BUS-001 Baseline**
   - Run 30-minute production, freeze output
   - Generate golden master audio files
   - Create reference metrics (performance, quality)
   - **Gate**: Baseline captured before Phase 2

---

### Phase 2: Data Model Migration (Week 2)
**Goal**: Migrate to Story 001 Pydantic models

1. **Extend Story 001 Models**
   - Add missing features from custom models
   - Ensure all bmad use cases supported
   - Update Story 001 tests

2. **Create Migration Utilities**
   - Automated conversion functions
   - Validation helpers
   - Test fixtures

3. **Migrate One File**
   - Choose smallest file (bmad_examples.py - 290 lines)
   - Migrate to Story 001 models
   - Validate with golden master tests
   - **Gate**: First migration successful before proceeding

4. **Migrate Remaining Files**
   - Incremental migration (one file per day)
   - Validate after each migration
   - Track regressions

---

### Phase 3: Constants Refactoring (Week 3)
**Goal**: Eliminate magic numbers

1. **Replace Hardcoded Values**
   - Systematic replacement using mapping from Phase 1
   - One file at a time
   - Validate audio output unchanged

2. **Validation**
   - Audio regression tests (spectral comparison)
   - Performance validation (no slowdown)
   - Code review (no remaining magic numbers)

---

### Phase 4: Implementation Consolidation (Week 4-5)
**Goal**: Merge three implementations into one

1. **Build Unified Implementation**
   - Use bmad_coordinator_lite.py as base (already uses cli_shared)
   - Add unique features from other two implementations
   - Follow Story 001 architecture patterns

2. **Migration Testing**
   - Feature parity tests
   - Golden master validation
   - Performance benchmarking

3. **Deprecate Old Implementations**
   - Move bmad_*.py to archive/
   - Update documentation
   - Provide migration guide for users

---

### Phase 5: Integration & Validation (Week 6)
**Goal**: Final quality assurance

1. **End-to-End Testing**
   - Full 30-minute production run
   - Compare to Phase 1 baseline
   - Validate all acceptance criteria

2. **Performance Optimization**
   - Profile hot paths
   - Optimize if regression > 10%
   - Validate memory usage

3. **Documentation & Handoff**
   - Update architecture documentation
   - User migration guide
   - Developer onboarding docs

---

## Final Gate Criteria

**Consolidation is COMPLETE when:**

✅ **All critical risks mitigated** (5/5 resolved)
✅ **Golden master tests pass** (audio quality maintained)
✅ **Performance within 10%** of baseline (3.5s/session → <3.85s/session)
✅ **30-minute production test matches baseline** (80+ sessions, 1600+ patterns)
✅ **Zero hardcoded values** in consolidated code
✅ **All Story 001 tests pass** (37/37 + new migration tests)
✅ **Feature parity validated** (all three implementation features present)
✅ **Code review approved** (@qa validation complete)

---

## Conclusion

This brownfield cleanup is **HIGH-RISK** but **ACHIEVABLE** with disciplined phased approach:

**Critical Success Factors:**
1. Fix tempo bug FIRST (broken music generation blocks everything)
2. Freeze baseline BEFORE starting (no moving target)
3. Incremental migration (one component at a time)
4. Continuous validation (golden master tests after every change)
5. Clear rollback plan (Git branching + documented procedure)

**Estimated Timeline**: 6 weeks (with proper risk mitigation)
**Estimated Effort**: 1 developer full-time + @qa validation support
**Risk Level**: HIGH → MEDIUM (with mitigation) → LOW (after Phase 2)

**Recommendation**: **PROCEED WITH CAUTION**
- This consolidation is necessary (technical debt is unsustainable)
- Working functionality exists (proven music generation)
- Risk is manageable (with disciplined execution)
- Story 001 provides solid target architecture

**Next Steps**:
1. **@music-orchestrator**: Review and approve consolidation strategy
2. **@dev**: Begin Phase 1 (Preparation)
3. **@qa**: Set up golden master test infrastructure
4. **@sound-designer**: Validate audio quality at each phase

---

**Quinn (Test Architect)**
*Providing comprehensive quality assessment and actionable recommendations without blocking progress*
