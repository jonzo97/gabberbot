# BMAD Migration Plan
**Step-by-Step Consolidation Implementation Guide**

Date: 2025-10-02
Analyst: Morgan (Dev Agent)
Story: story-phase1-consolidation-002-strategy-doc
Based on: Consolidation Strategy (Option B - bmad_coordinator_lite.py)

---

## Executive Summary

This migration plan provides detailed, step-by-step instructions for consolidating 17 BMAD files into a unified system based on bmad_coordinator_lite.py. The plan follows a 4-phase approach over 1-2 weeks with rollback points at each phase.

**Base Implementation:** bmad_coordinator_lite.py (487 lines)
**Target:** Unified BMAD system with 95%+ feature preservation
**Timeline:** 1-2 weeks (10 working days)
**Risk Level:** LOW (incremental approach with rollback strategy)

---

## Prerequisites & Setup

### Before Starting

**✅ Required:**
1. [ ] User approval of consolidation strategy (Option B)
2. [ ] Story 001 completion (TempoSyncConfig model available)
3. [ ] Feature inventory reviewed (feature-inventory.md)
4. [ ] Feature comparison matrix reviewed (feature-comparison-matrix.md)
5. [ ] Consolidation strategy approved (consolidation-strategy.md)

**✅ Environment Setup:**
1. [ ] Create git branch: `consolidation-phase1`
2. [ ] Backup all 17 bmad_*.py files to `bmad_backup/` directory
3. [ ] Set up test environment
4. [ ] Establish performance baseline (83 sessions data)
5. [ ] Configure rollback procedures

**✅ Tools & Dependencies:**
1. [ ] cli_shared.generators (AcidBasslineGenerator, TunedKickGenerator)
2. [ ] cli_shared.models.midi_clips (MIDIClip)
3. [ ] Story 001 models (Pydantic validation)
4. [ ] Audio synthesis engines
5. [ ] Testing framework

---

## Phase 1: Core Enhancements (Week 1, Days 1-5)

**Goal:** Enhance bmad_coordinator_lite.py with essential features from bmad_simple_test.py

**Base File:** `bmad_coordinator_lite.py` (487 lines)
**Source Files:** `bmad_simple_test.py` (475 lines)
**Deliverable:** Enhanced coordinator with config, mixing, validation

### Step 1.1: Add BMadTrackConfig Dataclass (Day 1)

**Source:** bmad_simple_test.py lines 40-80

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add after imports)

from dataclasses import dataclass
from typing import Optional

@dataclass
class BMadTrackConfig:
    """Track configuration for BMAD generation"""
    style: 'HardcoreStyle'
    bpm: float
    length_bars: float
    key: str
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration"""
        if self.bpm < 150 or self.bpm > 250:
            raise ValueError(f"BPM {self.bpm} outside hardcore range (150-250)")
        if self.length_bars <= 0:
            raise ValueError(f"length_bars must be positive, got {self.length_bars}")
```

**Validation:**
- [ ] Run: `python -c "from bmad_coordinator_lite import BMadTrackConfig; print('✅ Import successful')"`
- [ ] Test valid config: `BMadTrackConfig(style=HardcoreStyle.ROTTERDAM_GABBER, bpm=180, length_bars=16, key='A_minor')`
- [ ] Test invalid BPM: Should raise ValueError
- [ ] Test invalid length: Should raise ValueError

**Rollback:** If fails, remove dataclass, use dict-based config

---

### Step 1.2: Add HardcoreStyle Enum (Day 1)

**Source:** bmad_simple_test.py lines 20-30

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add after imports)

from enum import Enum

class HardcoreStyle(Enum):
    """Hardcore music styles supported by BMAD"""
    ROTTERDAM_GABBER = "rotterdam_gabber"
    FRENCHCORE = "frenchcore"

    @property
    def default_bpm(self) -> float:
        """Get default BPM for this style"""
        return {
            self.ROTTERDAM_GABBER: 180.0,
            self.FRENCHCORE: 200.0
        }[self]

    @property
    def bpm_range(self) -> tuple[float, float]:
        """Get BPM range for this style"""
        return {
            self.ROTTERDAM_GABBER: (170.0, 190.0),
            self.FRENCHCORE: (190.0, 220.0)
        }[self]
```

**Validation:**
- [ ] Run: `python -c "from bmad_coordinator_lite import HardcoreStyle; print(HardcoreStyle.ROTTERDAM_GABBER.default_bpm)"`
- [ ] Verify ROTTERDAM_GABBER defaults: 180 BPM, (170, 190) range
- [ ] Verify FRENCHCORE defaults: 200 BPM, (190, 220) range
- [ ] Integration test with BMadTrackConfig

**Rollback:** If fails, remove enum, use string constants

---

### Step 1.3: Add BMadMixEngineer (Day 2-3)

**Source:** bmad_simple_test.py lines 200-350

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add new class)

class BMadMixEngineer:
    """Professional mixing engine for BMAD hardcore tracks"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.master_gain = 0.8  # Professional headroom

    def mix_tracks(self, tracks: dict[str, np.ndarray]) -> np.ndarray:
        """
        Mix multiple audio tracks with professional balance

        Args:
            tracks: Dict of track_name -> audio_array

        Returns:
            Mixed stereo audio array
        """
        # Level balancing (hardcore mixing principles)
        levels = {
            'kick': 1.0,      # Kick loudest (hardcore priority)
            'bassline': 0.7,  # Bassline supporting kick
            'lead': 0.5,      # Lead sits back
            'effects': 0.4    # Effects subtle
        }

        # Find longest track
        max_length = max(len(track) for track in tracks.values())

        # Initialize mix bus (stereo)
        mix = np.zeros((max_length, 2))

        # Mix each track with proper levels and panning
        for name, audio in tracks.items():
            # Get level for this track type
            level = levels.get(name, 0.5)

            # Determine panning (hardcore = mostly center)
            if name == 'kick':
                pan = 0.0  # Center (mono)
            elif name == 'bassline':
                pan = 0.0  # Center (mono for sub)
            elif name == 'lead':
                pan = 0.1  # Slightly right
            else:
                pan = 0.0  # Default center

            # Convert mono to stereo if needed
            if audio.ndim == 1:
                audio_stereo = np.column_stack([audio, audio])
            else:
                audio_stereo = audio

            # Pad to match length
            if len(audio_stereo) < max_length:
                padding = np.zeros((max_length - len(audio_stereo), 2))
                audio_stereo = np.vstack([audio_stereo, padding])

            # Apply level and panning
            left_gain = level * (1.0 - max(0, pan))
            right_gain = level * (1.0 + min(0, pan))

            mix[:, 0] += audio_stereo[:, 0] * left_gain
            mix[:, 1] += audio_stereo[:, 1] * right_gain

        # Apply master gain (leave headroom for mastering)
        mix *= self.master_gain

        # Soft clip to prevent harsh clipping
        mix = np.tanh(mix * 1.2) / 1.2

        return mix

    def apply_hardcore_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply hardcore-specific processing"""
        # Subtle saturation for warmth
        processed = np.tanh(audio * 1.1) / 1.1

        # Ensure stereo
        if processed.ndim == 1:
            processed = np.column_stack([processed, processed])

        return processed
```

**Integration into coordinator:**
```python
# In main generate_hardcore_track() function:
# Replace basic mixing with:

mix_engineer = BMadMixEngineer(sample_rate=44100)
mixed_audio = mix_engineer.mix_tracks({
    'kick': kick_audio,
    'bassline': bassline_audio,
    'lead': lead_audio
})
final_audio = mix_engineer.apply_hardcore_processing(mixed_audio)
```

**Validation:**
- [ ] Test mix_tracks() with sample audio
- [ ] Verify level balancing (kick loudest)
- [ ] Verify panning (kick/bassline center)
- [ ] Test soft clipping (no harsh distortion)
- [ ] Integration test: Generate full track, verify mixing quality
- [ ] Compare with bmad_simple_test.py output (should be equivalent)

**Rollback:** If fails, revert to coordinator_lite basic mixing

---

### Step 1.4: Add Story 001 Pydantic Validation (Day 4)

**Source:** Story 001 models (TempoSyncConfig)

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add Pydantic validation layer)

from pydantic import BaseModel, Field, validator

class BMadTrackConfigValidated(BaseModel):
    """Pydantic-validated track configuration"""
    style: HardcoreStyle
    bpm: float = Field(ge=150, le=250, description="BPM (150-250 for hardcore)")
    length_bars: float = Field(gt=0, description="Track length in bars")
    key: str = Field(regex=r'^[A-G][#b]?_(major|minor)$', description="Musical key")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    @validator('bpm')
    def validate_bpm_for_style(cls, v, values):
        """Validate BPM is appropriate for style"""
        if 'style' in values:
            style = values['style']
            min_bpm, max_bpm = style.bpm_range
            if not (min_bpm <= v <= max_bpm):
                raise ValueError(f"{style.name} BPM should be {min_bpm}-{max_bpm}, got {v}")
        return v

    class Config:
        use_enum_values = True

    def to_dataclass(self) -> BMadTrackConfig:
        """Convert to dataclass for internal use"""
        return BMadTrackConfig(**self.dict())
```

**Integration:**
```python
# Update main function to accept Pydantic model:
def generate_hardcore_track(config: Union[BMadTrackConfig, BMadTrackConfigValidated]) -> str:
    """Generate hardcore track with validated config"""
    # Convert Pydantic to dataclass if needed
    if isinstance(config, BMadTrackConfigValidated):
        config = config.to_dataclass()

    # Rest of generation logic...
```

**Validation:**
- [ ] Test Pydantic validation (valid config)
- [ ] Test invalid BPM (should raise ValidationError)
- [ ] Test invalid key format (should raise ValidationError)
- [ ] Test BPM/style mismatch (should raise ValidationError)
- [ ] Test conversion to dataclass
- [ ] Integration test: Full generation with Pydantic config

**Rollback:** If fails, use dataclass-only validation

---

### Step 1.5: Phase 1 Testing & Validation (Day 5)

**Comprehensive Tests:**
1. [ ] Unit tests for BMadTrackConfig
2. [ ] Unit tests for HardcoreStyle enum
3. [ ] Unit tests for BMadMixEngineer
4. [ ] Unit tests for Pydantic validation
5. [ ] Integration test: Full track generation
6. [ ] Regression test: Compare with original coordinator_lite
7. [ ] Performance test: Verify no slowdown

**Quality Gates:**
- [ ] All tests passing
- [ ] Code coverage > 80%
- [ ] Performance within 10% of baseline
- [ ] No regressions detected
- [ ] Documentation updated

**Phase 1 Deliverables:**
- ✅ Enhanced bmad_coordinator_lite.py with:
  - BMadTrackConfig dataclass
  - HardcoreStyle enum
  - BMadMixEngineer mixing
  - Pydantic validation layer
- ✅ Test suite covering new features
- ✅ Documentation for new components

**Phase 1 Rollback Point:**
If Phase 1 fails, revert to original bmad_coordinator_lite.py (487 lines backup)

---

## Phase 2: Supporting Systems Integration (Week 2, Days 6-7)

**Goal:** Integrate professional mastering, QA, performance monitoring, workflow templates

**Source Files:**
- bmad_mastering_chain.py (34,096 bytes)
- bmad_qa_suite.py (78,585 bytes)
- bmad_performance_monitor.py (40,079 bytes)
- bmad_workflow_templates.py (50,700 bytes)

### Step 2.1: Integrate Mastering Chain (Day 6 morning)

**Source:** bmad_mastering_chain.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_mastering_chain import BMADMasteringChain, MasteringTarget

# In generate_hardcore_track() function, after mixing:
def generate_hardcore_track(config: BMadTrackConfigValidated,
                            mastering_target: MasteringTarget = MasteringTarget.HARDCORE_CLUB) -> str:
    # ... existing generation logic ...

    # Add mastering
    mastering_chain = BMADMasteringChain()
    mastered_audio = mastering_chain.master_track(
        audio=final_audio,
        target=mastering_target,
        sample_rate=44100
    )

    # Save mastered audio
    mastered_path = output_dir / f"{session_id}_mastered.wav"
    scipy.io.wavfile.write(mastered_path, 44100, mastered_audio)

    return session_id
```

**Validation:**
- [ ] Test mastering with HARDCORE_CLUB target (-6 LUFS)
- [ ] Test mastering with WAREHOUSE_SYSTEM target (-5 LUFS)
- [ ] Verify LUFS targeting accuracy
- [ ] Test all 7 mastering targets
- [ ] Compare output quality with bmad_mastering_chain.py standalone

**Rollback:** If fails, remove mastering integration, keep basic output

---

### Step 2.2: Integrate QA Suite (Day 6 afternoon)

**Source:** bmad_qa_suite.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_qa_suite import BMADQualityAssurance, validate_output_directory

# After track generation:
def generate_hardcore_track(config: BMadTrackConfigValidated,
                            mastering_target: MasteringTarget = MasteringTarget.HARDCORE_CLUB,
                            run_qa: bool = True) -> tuple[str, dict]:
    # ... existing generation logic ...

    # Run QA if requested
    qa_results = {}
    if run_qa:
        qa_system = BMADQualityAssurance()
        qa_results = validate_output_directory(output_dir)

        # Log QA results
        print(f"QA Score: {qa_results.get('overall_score', 0)}/10")
        print(f"QA Status: {'PASS' if qa_results.get('pass', False) else 'FAIL'}")

    return session_id, qa_results
```

**Validation:**
- [ ] Test QA on generated tracks
- [ ] Verify quality scoring (0-10 scale)
- [ ] Test hardcore authenticity checks
- [ ] Test MIDI validation
- [ ] Test audio quality validation
- [ ] Compare with bmad_qa_suite.py standalone results

**Rollback:** If fails, make QA optional (run_qa=False default)

---

### Step 2.3: Integrate Performance Monitor (Day 7 morning)

**Source:** bmad_performance_monitor.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_performance_monitor import BMADPerformanceMonitor, monitor_generation

# Wrap generation with monitoring:
def generate_hardcore_track(config: BMadTrackConfigValidated,
                            mastering_target: MasteringTarget = MasteringTarget.HARDCORE_CLUB,
                            run_qa: bool = True,
                            monitor_performance: bool = True) -> tuple[str, dict, dict]:

    performance_metrics = {}

    if monitor_performance:
        with monitor_generation(session_id, config) as monitor:
            # ... existing generation logic ...
            performance_metrics = monitor.get_metrics()
    else:
        # ... existing generation logic ...

    return session_id, qa_results, performance_metrics
```

**Validation:**
- [ ] Test performance monitoring enabled
- [ ] Verify CPU/memory tracking
- [ ] Test generation time tracking
- [ ] Verify performance metrics accuracy
- [ ] Test monitoring overhead (< 5% slowdown)

**Rollback:** If fails, make monitoring optional (monitor_performance=False default)

---

### Step 2.4: Integrate Workflow Templates (Day 7 afternoon)

**Source:** bmad_workflow_templates.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_workflow_templates import BMADWorkflowTemplates, WorkflowTemplate, WorkflowConfig

# Add workflow execution function:
def execute_workflow(template: WorkflowTemplate,
                     name: str,
                     **kwargs) -> dict:
    """Execute production workflow template"""
    workflow_system = BMADWorkflowTemplates()

    workflow_config = WorkflowConfig(
        template=template,
        name=name,
        **kwargs
    )

    result = workflow_system.execute_workflow(workflow_config)

    return {
        'success': result.success,
        'tracks_generated': result.total_tracks_generated,
        'execution_time': result.execution_time_seconds,
        'output_directories': result.output_directories,
        'quality_score': result.quality_score
    }
```

**Validation:**
- [ ] Test SINGLE_RELEASE workflow
- [ ] Test EP_RELEASE workflow (4-6 tracks)
- [ ] Test ALBUM_RELEASE workflow (8-12 tracks)
- [ ] Test DJ_PERFORMANCE_SET workflow
- [ ] Verify all 12 workflow templates functional
- [ ] Test workflow result packaging

**Rollback:** If fails, remove workflow templates, keep core generation

---

### Step 2.5: Phase 2 Testing & Validation (Day 7 end)

**Comprehensive Tests:**
1. [ ] Test mastering integration (all 7 targets)
2. [ ] Test QA integration (quality scoring)
3. [ ] Test performance monitoring (metrics collection)
4. [ ] Test workflow templates (all 12 templates)
5. [ ] Integration test: Full track with all Phase 2 features
6. [ ] Regression test: Phase 1 features still working
7. [ ] Performance test: Acceptable overhead

**Quality Gates:**
- [ ] All tests passing
- [ ] No regressions from Phase 1
- [ ] Performance overhead < 10%
- [ ] QA scores match standalone bmad_qa_suite.py
- [ ] Mastering LUFS targets accurate

**Phase 2 Deliverables:**
- ✅ Mastering integration (7 targets)
- ✅ QA suite integration (comprehensive validation)
- ✅ Performance monitoring (real-time metrics)
- ✅ Workflow templates (12 production workflows)

**Phase 2 Rollback Point:**
If Phase 2 fails, revert to Phase 1 completion state

---

## Phase 3: Professional Features (Week 2, Days 8-9)

**Goal:** Integrate album production, DJ export, factory orchestration, system integration

**Source Files:**
- bmad_album_producer.py (36,412 bytes)
- bmad_dj_export.py (40,247 bytes)
- bmad_hardcore_factory.py (34,154 bytes)
- bmad_integration_bridge.py (37,775 bytes)

### Step 3.1: Integrate Album Producer (Day 8 morning)

**Source:** bmad_album_producer.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_album_producer import BMadAlbumProducer, AlbumConfig

# Add album generation function:
def generate_album(album_config: AlbumConfig) -> dict:
    """Generate full hardcore album"""
    album_producer = BMadAlbumProducer()

    result = album_producer.produce_album(album_config)

    return {
        'success': result.success,
        'album_name': result.album_name,
        'tracks': result.tracks,
        'total_duration': result.total_duration_minutes,
        'output_directory': result.output_directory,
        'continuous_mix': result.continuous_mix_path
    }
```

**Validation:**
- [ ] Test EP production (4-6 tracks)
- [ ] Test album production (8-12 tracks)
- [ ] Test track sequencing/ordering
- [ ] Test BPM progression
- [ ] Test energy curve
- [ ] Test continuous mix creation

**Rollback:** If fails, remove album features, keep single track generation

---

### Step 3.2: Integrate DJ Export (Day 8 afternoon)

**Source:** bmad_dj_export.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_dj_export import BMadDJExporter, DJExportFormat

# Add DJ export function:
def export_for_dj(session_id: str,
                  formats: list[DJExportFormat] = [DJExportFormat.WAV, DJExportFormat.MP3]) -> dict:
    """Export track in DJ-ready formats"""
    dj_exporter = BMadDJExporter()

    result = dj_exporter.export_track(
        session_id=session_id,
        formats=formats,
        include_metadata=True,
        generate_cues=True
    )

    return {
        'success': result.success,
        'exported_files': result.exported_files,
        'metadata': result.metadata,
        'cue_points': result.cue_points
    }
```

**Validation:**
- [ ] Test WAV export (44.1kHz/16-bit, 24-bit)
- [ ] Test MP3 export (320kbps)
- [ ] Test AIFF export
- [ ] Test Rekordbox metadata
- [ ] Test Serato markers
- [ ] Test cue point generation
- [ ] Test BPM/key detection

**Rollback:** If fails, remove DJ export, use basic WAV output

---

### Step 3.3: Integrate Factory Orchestration (Day 9 morning)

**Source:** bmad_hardcore_factory.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_hardcore_factory import BMADHardcoreFactory, BMADFactoryConfig, ProductionMode

# Add factory function:
def factory_generate(factory_config: BMADFactoryConfig) -> dict:
    """Generate using factory pattern"""
    factory = BMADHardcoreFactory(factory_config)

    result = factory.generate_hardcore_music(factory_config.session_name)

    return {
        'success': result.success,
        'session_id': result.session_id,
        'production_mode': factory_config.production_mode,
        'tracks_generated': result.tracks_generated,
        'quality_level': factory_config.quality_level,
        'output_directory': result.output_directory
    }
```

**Validation:**
- [ ] Test SINGLE_TRACK mode
- [ ] Test ALBUM_EP mode (4-6 tracks)
- [ ] Test FULL_ALBUM mode (8-12 tracks)
- [ ] Test DJ_SET mode (10-20 tracks)
- [ ] Test LIVE_PERFORMANCE mode
- [ ] Test all quality levels (DRAFT, STANDARD, PROFESSIONAL, MASTERED)

**Rollback:** If fails, remove factory orchestration, use direct generation

---

### Step 3.4: Integrate System Bridge (Day 9 afternoon)

**Source:** bmad_integration_bridge.py

**Implementation:**
```python
# File: bmad_coordinator_lite.py (add import)
from bmad_integration_bridge import BMADIntegrationBridge, UnifiedCommand, IntegrationMode

# Add bridge initialization:
def get_integration_bridge() -> BMADIntegrationBridge:
    """Get or create integration bridge"""
    return BMADIntegrationBridge(IntegrationConfig(
        integration_mode=IntegrationMode.FULL_INTEGRATION,
        enable_main_system=True,
        enable_tui=True
    ))

# Add unified command execution:
def execute_unified_command(command: UnifiedCommand) -> dict:
    """Execute command through integration bridge"""
    bridge = get_integration_bridge()
    result = bridge.execute_unified_command(command)
    return result
```

**Validation:**
- [ ] Test BMAD-only mode
- [ ] Test main system integration
- [ ] Test TUI integration
- [ ] Test hybrid workflows
- [ ] Test command routing (hardcore → BMAD, other → main)
- [ ] Test unified output directory

**Rollback:** If fails, remove integration bridge, use standalone mode

---

### Step 3.5: Phase 3 Testing & Validation (Day 9 end)

**Comprehensive Tests:**
1. [ ] Test album production (EP, full album)
2. [ ] Test DJ export (all formats)
3. [ ] Test factory orchestration (all modes)
4. [ ] Test system integration (TUI, main, hybrid)
5. [ ] Integration test: Full workflow using all Phase 3 features
6. [ ] Regression test: Phase 1+2 features still working
7. [ ] Performance test: Acceptable overhead

**Quality Gates:**
- [ ] All tests passing
- [ ] No regressions from Phase 1+2
- [ ] Album production quality matches bmad_album_producer.py
- [ ] DJ exports compatible with Rekordbox/Serato
- [ ] Factory modes all functional
- [ ] Integration bridge routing correctly

**Phase 3 Deliverables:**
- ✅ Album production (EP, full album, continuous mix)
- ✅ DJ export (WAV, MP3, AIFF, metadata, cues)
- ✅ Factory orchestration (6 production modes)
- ✅ System integration (main, TUI, hybrid)

**Phase 3 Rollback Point:**
If Phase 3 fails, revert to Phase 2 completion state

---

## Phase 4: Cleanup & Documentation (Week 2, Day 10)

**Goal:** Constants extraction, documentation, final testing, production readiness

### Step 4.1: Constants Extraction (Day 10 morning)

**Prepare for Story 3:**
```python
# Create: bmad_constants.py

# BPM Constants
HARDCORE_BPM_MIN = 150
HARDCORE_BPM_MAX = 250
ROTTERDAM_GABBER_BPM_DEFAULT = 180.0
ROTTERDAM_GABBER_BPM_RANGE = (170.0, 190.0)
FRENCHCORE_BPM_DEFAULT = 200.0
FRENCHCORE_BPM_RANGE = (190.0, 220.0)

# Audio Constants
SAMPLE_RATE_DEFAULT = 44100
MASTER_GAIN_DEFAULT = 0.8
HEADROOM_DB = -0.5

# Mixing Levels
KICK_LEVEL = 1.0
BASSLINE_LEVEL = 0.7
LEAD_LEVEL = 0.5
EFFECTS_LEVEL = 0.4

# Mastering Targets
MASTERING_TARGET_HARDCORE_CLUB = -6  # LUFS
MASTERING_TARGET_WAREHOUSE = -5      # LUFS
MASTERING_TARGET_FRENCHCORE = -4     # LUFS
MASTERING_TARGET_INDUSTRIAL = -8     # LUFS

# QA Thresholds
QA_PASS_THRESHOLD = 8.0  # Minimum score for pass
QA_HARDCORE_AUTHENTICITY_MIN = 7.0
```

**Update coordinator to use constants:**
```python
# Replace hardcoded values with constants
from bmad_constants import *

# Example:
self.master_gain = MASTER_GAIN_DEFAULT  # Instead of 0.8
```

**Validation:**
- [ ] All constants extracted
- [ ] No hardcoded values remain
- [ ] Constants documented
- [ ] Tests still passing with constants

---

### Step 4.2: Documentation (Day 10 morning)

**Create/Update Documentation:**

1. **API Documentation:**
```markdown
# File: docs/bmad_api.md

## Core Functions

### generate_hardcore_track()
Generate single hardcore track with professional quality.

**Parameters:**
- config (BMadTrackConfigValidated): Track configuration
- mastering_target (MasteringTarget): Mastering target (default: HARDCORE_CLUB)
- run_qa (bool): Run quality assurance (default: True)
- monitor_performance (bool): Monitor performance (default: True)

**Returns:**
- tuple[str, dict, dict]: (session_id, qa_results, performance_metrics)

**Example:**
```python
config = BMadTrackConfigValidated(
    style=HardcoreStyle.ROTTERDAM_GABBER,
    bpm=180,
    length_bars=16,
    key='A_minor'
)
session_id, qa, perf = generate_hardcore_track(config)
```

### execute_workflow()
Execute production workflow template.

**Parameters:**
- template (WorkflowTemplate): Workflow template to execute
- name (str): Workflow name
- **kwargs: Template-specific parameters

**Returns:**
- dict: Workflow execution results

**Example:**
```python
result = execute_workflow(
    WorkflowTemplate.EP_RELEASE,
    name='Warehouse_Destroyer_EP',
    track_count=6,
    bpm_range=(180, 200)
)
```
```

2. **Migration Guide:**
```markdown
# File: docs/bmad_migration_guide.md

## Migrating from Old BMAD Files

### From bmad_standalone.py
If you used bmad_standalone.py, migrate to:
- Use `generate_hardcore_track()` instead of standalone methods
- Replace `SimpleMIDIExporter` with `MIDIClip` integration
- Update imports: `from bmad_coordinator_lite import *`

### From bmad_simple_test.py
If you used bmad_simple_test.py, migrate to:
- `BMadSimpleCoordinator` → `generate_hardcore_track()`
- `BMadTrackConfig` → `BMadTrackConfigValidated` (Pydantic)
- `BMadMixEngineer` → Integrated mixing (automatic)

### From bmad_coordinator_lite.py
No migration needed if using basic functionality.
New features available: mastering, QA, monitoring, workflows.
```

3. **Usage Examples:**
```markdown
# File: docs/bmad_examples.md

## Quick Start Examples

### Example 1: Single Track
```python
from bmad_coordinator_lite import *

config = BMadTrackConfigValidated(
    style=HardcoreStyle.ROTTERDAM_GABBER,
    bpm=180,
    length_bars=16,
    key='A_minor'
)

session_id, qa, perf = generate_hardcore_track(config)
print(f"Generated: {session_id}")
print(f"Quality Score: {qa['overall_score']}/10")
```

### Example 2: EP Production
```python
result = execute_workflow(
    WorkflowTemplate.EP_RELEASE,
    name='My_Hardcore_EP',
    track_count=6,
    bpm_range=(180, 200),
    quality_level=QualityLevel.PROFESSIONAL
)
print(f"EP tracks: {result['tracks_generated']}")
```
```

**Validation:**
- [ ] API docs complete
- [ ] Migration guide complete
- [ ] Usage examples complete
- [ ] README updated
- [ ] All public functions documented

---

### Step 4.3: Final Testing (Day 10 afternoon)

**Comprehensive Test Suite:**

1. **Unit Tests:**
- [ ] BMadTrackConfig dataclass
- [ ] HardcoreStyle enum
- [ ] BMadMixEngineer
- [ ] Pydantic validation
- [ ] All new functions

2. **Integration Tests:**
- [ ] Phase 1 features (config, mixing, validation)
- [ ] Phase 2 features (mastering, QA, monitoring, workflows)
- [ ] Phase 3 features (album, DJ, factory, bridge)
- [ ] End-to-end workflow (single track → mastering → DJ export)

3. **Regression Tests:**
- [ ] Compare with 83 sessions baseline
- [ ] Verify quality scores match bmad_qa_suite.py standalone
- [ ] Verify mastering LUFS targets accurate
- [ ] Verify all original coordinator_lite functionality preserved

4. **Performance Tests:**
- [ ] Single track generation time
- [ ] Album generation time (8 tracks)
- [ ] Memory usage
- [ ] CPU usage
- [ ] Overhead from monitoring (< 5%)

**Quality Gates:**
- [ ] All tests passing (100%)
- [ ] Code coverage > 80%
- [ ] Performance within 10% of baseline
- [ ] No regressions detected
- [ ] Documentation complete

---

### Step 4.4: Production Readiness (Day 10 end)

**Final Checklist:**

**Code Quality:**
- [ ] No hardcoded values (all in constants)
- [ ] All functions documented
- [ ] Type hints complete
- [ ] Error handling robust
- [ ] Logging configured

**Testing:**
- [ ] Unit tests: 100% passing
- [ ] Integration tests: 100% passing
- [ ] Regression tests: 100% passing
- [ ] Performance tests: Meeting benchmarks

**Documentation:**
- [ ] API docs complete
- [ ] Migration guide complete
- [ ] Usage examples complete
- [ ] README updated
- [ ] Change log updated

**Integration:**
- [ ] cli_shared integration working
- [ ] Story 001 models integrated
- [ ] All supporting files integrated
- [ ] System bridge functional

**Deployment:**
- [ ] Git branch clean
- [ ] All changes committed
- [ ] Tests passing in CI/CD
- [ ] Ready for merge to main

---

## Rollback Procedures

### Rollback Triggers
Execute rollback if:
1. Build failures that can't be resolved in 1 day
2. Quality regression > 10% (below 83 sessions benchmark)
3. Performance degradation > 20%
4. Critical features lost
5. Epic timeline at risk (> 4 weeks elapsed)

### Rollback Levels

**Level 1: Rollback to Phase 3**
```bash
git checkout consolidation-phase1
git revert <phase-4-commits>
git push
```
**Result:** Phase 1+2+3 functional, Phase 4 removed

**Level 2: Rollback to Phase 2**
```bash
git checkout consolidation-phase1
git revert <phase-3-commits> <phase-4-commits>
git push
```
**Result:** Phase 1+2 functional, Phase 3+4 removed

**Level 3: Rollback to Phase 1**
```bash
git checkout consolidation-phase1
git revert <phase-2-commits> <phase-3-commits> <phase-4-commits>
git push
```
**Result:** Phase 1 functional, Phase 2+3+4 removed

**Level 4: Complete Rollback**
```bash
git checkout consolidation-phase1
git reset --hard <pre-consolidation-commit>
git push --force
```
**Result:** Original bmad_coordinator_lite.py restored

**Recovery Steps After Rollback:**
1. Analyze rollback cause
2. Document lessons learned
3. Create fix plan
4. Test fix in isolation
5. Re-attempt consolidation with fixes

---

## File Management

### Files to Create
1. ✅ `bmad_coordinator_lite.py` (enhanced version)
2. ✅ `bmad_constants.py` (constants extraction)
3. ✅ `docs/bmad_api.md` (API documentation)
4. ✅ `docs/bmad_migration_guide.md` (migration guide)
5. ✅ `docs/bmad_examples.md` (usage examples)

### Files to Modify
1. ✅ `bmad_coordinator_lite.py` (base file - enhanced)
2. ✅ `README.md` (update with new features)
3. ✅ `CHANGELOG.md` (document consolidation)

### Files to Deprecate (Move to archive/)
1. ❌ `bmad_standalone.py` → `archive/bmad_standalone.py`
2. ❌ `bmad_simple_test.py` → `archive/bmad_simple_test.py`
3. ❌ `bmad_phase3_demo.py` → `archive/bmad_phase3_demo.py`
4. ❌ `bmad_phase3_demo_final.py` → `archive/bmad_phase3_demo_final.py`
5. ❌ `bmad_phase3_demo_fixed.py` → `archive/bmad_phase3_demo_fixed.py`
6. ❌ `bmad_examples.py` → `archive/bmad_examples.py`

### Files to Keep (Integrated)
1. ✅ `bmad_coordinator_lite.py` (enhanced base)
2. ✅ `bmad_mastering_chain.py` (integrated)
3. ✅ `bmad_qa_suite.py` (integrated)
4. ✅ `bmad_performance_monitor.py` (integrated)
5. ✅ `bmad_workflow_templates.py` (integrated)
6. ✅ `bmad_album_producer.py` (integrated)
7. ✅ `bmad_dj_export.py` (integrated)
8. ✅ `bmad_hardcore_factory.py` (integrated)
9. ✅ `bmad_integration_bridge.py` (integrated)
10. ✅ `bmad_init.py` (integrated)
11. ✅ `bmad_init_simple.py` (integrated)

---

## Success Criteria

### Phase 1 Success
- [ ] BMadTrackConfig integrated
- [ ] HardcoreStyle enum working
- [ ] BMadMixEngineer mixing professionally
- [ ] Pydantic validation functional
- [ ] All tests passing
- [ ] Performance maintained

### Phase 2 Success
- [ ] Mastering producing professional output (LUFS targets accurate)
- [ ] QA validating tracks (scores match bmad_qa_suite.py)
- [ ] Performance monitoring tracking metrics
- [ ] Workflow templates executing (all 12 working)
- [ ] All Phase 1 features still working

### Phase 3 Success
- [ ] Album production operational (EP, full album)
- [ ] DJ export working (all formats, metadata, cues)
- [ ] Factory orchestration functional (all modes)
- [ ] Integration bridge routing correctly
- [ ] All Phase 1+2 features still working

### Phase 4 Success
- [ ] Constants extracted (no hardcoded values)
- [ ] Documentation complete (API, migration, examples)
- [ ] All tests passing (unit, integration, regression, performance)
- [ ] Production ready (deploy checklist complete)
- [ ] Ready for Stories 3-6

### Overall Success
- [ ] 95%+ features preserved from all 17 files
- [ ] Quality maintained (matches 83 sessions benchmark)
- [ ] Performance maintained (within 10% of baseline)
- [ ] Timeline met (1-2 weeks, < 6-week epic)
- [ ] Architecture aligned (Story 001 + Architecture Spec)
- [ ] User approval obtained

---

## Timeline Summary

**Week 1 (Days 1-5): Phase 1**
- Day 1: BMadTrackConfig + HardcoreStyle enum
- Day 2-3: BMadMixEngineer
- Day 4: Pydantic validation
- Day 5: Testing & validation

**Week 2 (Days 6-10): Phases 2-4**
- Day 6: Mastering + QA integration
- Day 7: Performance monitoring + Workflow templates
- Day 8: Album production + DJ export
- Day 9: Factory orchestration + Integration bridge
- Day 10: Cleanup + Documentation + Final testing

**Total: 10 working days (2 weeks)**

---

## Dependencies

**External:**
- cli_shared.generators (AcidBasslineGenerator, TunedKickGenerator)
- cli_shared.models.midi_clips (MIDIClip)
- Story 001 models (Pydantic validation)
- Audio synthesis engines

**Internal:**
- bmad_mastering_chain.py
- bmad_qa_suite.py
- bmad_performance_monitor.py
- bmad_workflow_templates.py
- bmad_album_producer.py
- bmad_dj_export.py
- bmad_hardcore_factory.py
- bmad_integration_bridge.py

**Prerequisites:**
- User approval of strategy
- Git branch created
- Test environment configured
- Performance baseline established

---

## Next Steps

### Upon Approval
1. ✅ User approval of migration plan
2. ✅ Create git branch: `consolidation-phase1`
3. ✅ Backup all 17 files to `bmad_backup/`
4. ✅ Begin Phase 1 implementation (Day 1)
5. ✅ Update story status to "In Progress"

### Daily Check-ins
- [ ] Day 1: Phase 1.1-1.2 complete (config + enum)
- [ ] Day 2: Phase 1.3 complete (mixing)
- [ ] Day 4: Phase 1.4 complete (validation)
- [ ] Day 5: Phase 1.5 complete (testing)
- [ ] Day 6: Phase 2.1-2.2 complete (mastering + QA)
- [ ] Day 7: Phase 2.3-2.5 complete (monitoring + workflows + testing)
- [ ] Day 8: Phase 3.1-3.2 complete (album + DJ export)
- [ ] Day 9: Phase 3.3-3.5 complete (factory + bridge + testing)
- [ ] Day 10: Phase 4 complete (cleanup + docs + final testing)

### Final Deliverables
1. ✅ Enhanced bmad_coordinator_lite.py (consolidated system)
2. ✅ bmad_constants.py (extracted constants)
3. ✅ Complete documentation (API, migration, examples)
4. ✅ Test suite (unit, integration, regression, performance)
5. ✅ Production-ready system (95%+ features preserved)

---

## Conclusion

This migration plan provides a detailed, step-by-step approach to consolidating 17 BMAD files into a unified system. The plan follows a 4-phase approach over 1-2 weeks with rollback points at each phase.

**Key Strengths:**
- Incremental approach (low risk)
- Phase-based rollback (can recover at any point)
- Comprehensive testing (quality assurance)
- Feature preservation (95%+ maintained)
- Timeline certainty (1-2 weeks)

**Migration Ready:** ✅

**Awaiting:** User approval to begin Phase 1

---

**End of Migration Plan**
