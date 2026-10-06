# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

**Music Assistant** - A conversational AI assistant for hardcore/industrial electronic music production. This is "ChatOps for music production" - like having a techno producer friend who codes.

### Core Mission
- **Primary Focus**: Hardcore, gabber, industrial techno, uptempo (150-250 BPM)
- **Key Principle**: CRUNCHY KICKDRUMS are non-negotiable - this is the soul of the project
- **Approach**: Jamming partner that generates initial ideas and refines through conversation
- **Aesthetic**: Aggressive, industrial, warehouse sound - never suggest "softening" unless explicitly asked

### Target User
Electronic music producers working in underground genres who value speed over perfection and appreciate aggressive aesthetics.

## BMAD Development Workflow

This project uses the **BMAD (Better Method for AI Development)** methodology for systematic software development through specialized agents.

### Development Agents (BMAD Commands)

BMAD agents are invoked using `/command` syntax:

**Core Development Team:**
- `/sm` - Scrum Master: Create stories from epics, manage sprint planning
- `/dev` - Developer: Implement stories following architecture patterns
- `/qa` - QA Engineer: Validate implementations, run tests, quality gates
- `/architect` - Architect: Architectural guidance and design decisions
- `/analyst` - Business Analyst: Requirements analysis and research

**Management & Coordination:**
- `/pm` - Project Manager: Project management and coordination
- `/po` - Product Owner: Product validation and epic management
- `/bmad-orchestrator` - Coordinates complex multi-agent workflows
- `/bmad-master` - High-level project oversight

**Example Usage:**
```
> /sm create next story from epic-01-prototyper
> /dev implement story-phase1-prototyper-001.yaml
> /qa review story-phase1-prototyper-001.yaml
```

**IMPORTANT:** For complex features, use BMAD agents systematically rather than implementing manually. The agents have full context of your planning docs and architecture.

### BMAD Workflow (Systematic Development)

```
1. Story Creation (sm agent)
   ↓
   Epic → Planning Docs → Story YAML file

2. Implementation (dev agent)
   ↓
   Story → Code + Tests → Ready for Review

3. Quality Validation (qa agent)
   ↓
   Review → Tests → Verdict (PASS/FAIL)

4. Iterate until complete
```

### Planning Documentation Structure

```
docs/
├── bmad-planning/           # Project planning docs
│   ├── 01-project-brief.md        # Business analysis
│   ├── 02-architecture-spec.md    # Technical architecture (AUTHORITATIVE)
│   ├── 03-prd.md                  # Product requirements
│   ├── 04-po-validation.md        # Phased roadmap
│   └── context-files/             # Current state, decisions, tech debt
├── bmad-development/        # Development artifacts
│   ├── epics/                     # Epic definitions
│   └── stories/                   # Story YAML files
└── bmad-agents/            # Original agent definitions (reference)
```

### Using Agents for Complex Work

**For multi-step features:**
```
> Use the sm agent to analyze the current codebase drift and create a story for fixing architecture violations

> Use the dev agent to implement the architecture fix story

> Use the qa agent to validate the implementation meets requirements
```

**Agent Benefits:**
- Systematic approach to complex tasks
- Complete context from planning docs
- Consistent quality through defined workflows
- Clear handoffs between phases
- Documentation built into process

## Technical Architecture

### Core Stack (BMAD-Enhanced)
```
Audio Engine:    SuperCollider (C++ real-time synthesis server)
Pattern Engine:  TidalCycles (Haskell-based pattern language)
Orchestration:   Python 3.11+ with asyncio + Supriya (SC bridge)
Frontend:        React + TypeScript + Monaco Editor
Communication:   OSC protocol for real-time audio control
Database:        PostgreSQL + Redis
AI:              Multi-model (Claude primary, GPT, Gemini)
BMAD:            Music production expansion pack with 9 specialized agents
```

### Current Implementation (MIDI-Based Architecture)
- **cli_shared/models/midi_clips.py**: Core MIDIClip and TriggerClip classes
- **Pattern generators**: Located in cli_shared/generators/ (built on MIDI foundation)
- **Multiple backends**: TidalCycles, SuperCollider, MIDI export all supported
- **AI integration**: Unified clip-based tools instead of scattered specific tools

### Existing Infrastructure to Leverage
- **cli_shared/interfaces/synthesizer.py**: AbstractSynthesizer (USE THIS)
- **cli_strudel/synthesis/fm_synthesizer.py**: Professional FM synthesis
- **cli_strudel/synthesis/sidechain_compressor.py**: Sidechain processing
- **cli_sc/core/supercollider_synthesizer.py**: SuperCollider backend

## Music Intelligence

### Genre-Specific Knowledge
- **Gabber (150-200 BPM)**: Extreme kick distortion, Rotterdam style "doorlussen" technique
- **Industrial Techno (130-150 BPM)**: Berlin rumble kicks, metallic reverb, minimal arrangements  
- **Hardcore (180-250 BPM)**: Heavy compression, hoover sounds, complex breakbeats

### Kick Drum Synthesis (Critical)
```yaml
gabber_kick:
  source: "TR-909 analog kick"
  processing: "Heavy mixer overdrive + serial distortion"
  characteristics: "Monolithic, tonal, aggressive"

industrial_kick:  
  architecture: "3-layer system (main + rumble + ghost)"
  rumble_chain: "Reverb → Overdrive → Low-pass filter"
  characteristics: "Separated transient + sub-bass tail"
```

### Hardcore Production Parameters
```python
# Authentic hardcore synthesis settings (user-validated)
HARDCORE_PARAMS = {
    'kick_sub_freqs': [41.2, 82.4, 123.6],  # E1, E2, E2+fifth Hz
    'detune_cents': [-19, -10, -5, 0, 5, 10, 19, 29],  # Reduced by 20%
    'distortion_db': 15,  # Reduced from 18 per user feedback
    'highpass_hz': 120,   # Clean low-end for kick space
    'compression_ratio': 8,
    'limiter_threshold': -0.5,
    'bitcrush_depth': 12  # Reduced for cleaner sound
}
```

## MANDATORY CODE QUALITY STANDARDS

### Core Principles (NON-NEGOTIABLE)

1. **Quality Over Speed**: Take time to build properly. Never rush or create spaghetti code.

2. **No Reinventing Wheels**: Always check existing codebase before writing new code:
   - `cli_shared/`: Interfaces, models, utilities (USE THIS FIRST)
   - `cli_strudel/`: Complete synthesis library (FM, sidechain, etc.)
   - `cli_sc/`: SuperCollider integration

3. **Use Existing Infrastructure**: All new components MUST use existing interfaces:
   - `cli_shared/interfaces/synthesizer.py` → AbstractSynthesizer interface is MANDATORY
   - `cli_shared/models/hardcore_models.py` → Use existing data models

4. **Modular Everything**: All functions must be importable and reusable

5. **Zero Magic Numbers**: All parameters must be in dedicated constants files

6. **Professional Architecture**: Follow DAW industry standards:
   - Track-based design: Control Source → Audio Source → FX Chain → Mixer
   - Use composition over inheritance
   - Interface-based design patterns

### Code Review Standards

- **Spaghetti Code is FORBIDDEN**: No arbitrary inheritance chains
- **DRY Principle**: Don't Repeat Yourself - one implementation per function
- **Type Hints**: All functions must have proper type annotations
- **Documentation**: Every function/class needs clear docstrings
- **Single Responsibility**: One function, one job

## Critical Constraints

### Non-Negotiables
1. **Crunchy kickdrums** - Must sound hard, not toy-like
2. **Native execution** - Not cloud-dependent for core functionality
3. **200+ BPM support** - No timing issues at hardcore speeds
4. **Industrial aesthetic** - Dark, aggressive, warehouse-focused
5. **CLI-friendly** - Terminal-style interface option

### Avoid These Pitfalls
- Over-quantization (kills groove)
- Weak synthesis (toy sounds)
- Academic music theory focus
- Jazz/ambient bias
- Cloud-only architecture

## Development Commands

### BMAD Music Production Workflow
```bash
# Use orchestrator for team coordination
@music-orchestrator # Coordinate music production teams

# Individual agent access
@music-producer     # Track composition and creative direction
@sound-designer     # Synthesis and sound design
@mix-engineer       # Professional mixing and mastering
@music-analyst      # Genre analysis and pattern recognition
@theory-engine      # Music theory and harmonic analysis
@innovation-lab     # Experimental techniques and fusion
```

### Quality-First Development Workflow
```bash
# 1. Before coding - check existing infrastructure
find cli_shared/ cli_strudel/ cli_sc/ -name "*.py" | grep -i [functionality]

# 2. Use existing synthesis components
python cli_strudel/synthesis/fm_synthesizer.py  # Professional FM synthesis
python cli_strudel/synthesis/sidechain_compressor.py  # Sidechain processing

# 3. SuperCollider backend (properly structured)
python cli_sc/core/supercollider_synthesizer.py

# 4. Play generated audio
paplay audio_tests/[generated_file].wav
```

## Project Structure

```
music_code_cli/
├── BMAD-AT-CLAUDE/expansion-packs/bmad-music-production/  # BMAD music agents & infrastructure
├── cli_shared/          # Professional interfaces, data models, production engines
├── cli_strudel/         # Complete synthesis library (FM, sidechain, arpeggiators)  
├── cli_sc/              # SuperCollider backend with AbstractSynthesizer implementation
├── design/              # Living documentation & diagrams
├── tests/               # Active test suites
├── audio_tests/         # Generated audio outputs
└── archive/             # Historical/obsolete files
```

**Remember**: This is not academic music software. It's for making hard, aggressive electronic music that destroys sound systems. Every decision should serve that goal. Use the BMAD music production agents for all music-related work.