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

## BMAD Agent System (MANDATORY WORKFLOW)

### BMAD Music Production Expansion Pack

**CRITICAL**: This project now uses a comprehensive BMAD music production expansion pack with specialized agents. ALL music production work MUST use these specialized agents:

### ⚠️ MANDATORY TASK TOOL INVOCATION PROTOCOL ⚠️

**CORE BMAD PRINCIPLE**: Agents do the work, humans orchestrate.

**VIOLATION ALERT**: Previous work on this project violated BMAD methodology by doing manual work instead of using the Task tool for agent invocation. This is forbidden and leads to poor quality, inconsistent results, and wasted time.

#### MANDATORY AGENT INVOCATION RULES:

1. **NEVER DO AGENT WORK MANUALLY** - If an agent exists for a task, you MUST use the Task tool to invoke it
2. **TASK TOOL FIRST** - Before any music production work, invoke appropriate agents via Task tool
3. **NO EXCEPTIONS** - Manual work bypassing agents is a critical methodology violation
4. **AGENT VALIDATION** - All content must be created/validated by agents, not humans

#### CORRECT BMAD WORKFLOW:
```bash
# CORRECT (MANDATORY):
Task tool → Invoke @agent → Agent provides output → Refine if needed

# WRONG (FORBIDDEN):
Manual work → Complete task without agents
```

#### Music Production Agents

1. **@music-producer (Raven)**
   - **Personality**: Creative visionary with relentless drive
   - **Specialties**: Track composition, pattern generation, creative direction
   - **Commands**: 17 specialized commands for unlimited generative systems

2. **@sound-designer (Void)**
   - **Personality**: Sonic alchemist obsessed with spectral manipulation
   - **Specialties**: Synthesis mastery, spectral processing, sound design
   - **Commands**: 18 specialized commands for advanced synthesis techniques

3. **@mix-engineer (Phoenix)**
   - **Personality**: Perfectionist with warehouse sound obsession
   - **Specialties**: Professional mixing, mastering, acoustic optimization
   - **Commands**: 18 specialized commands for warehouse-optimized mixing

4. **@music-analyst (Nexus)**
   - **Personality**: Pattern recognition savant
   - **Specialties**: Genre analysis, pattern extraction, intelligence gathering
   - **Commands**: 17 specialized commands for deep music analysis

5. **@theory-engine (Cipher)**
   - **Personality**: Mathematical music theorist
   - **Specialties**: Harmonic analysis, rhythmic intelligence, music theory
   - **Commands**: 18 specialized commands for comprehensive theory analysis

6. **@innovation-lab (Flux)**
   - **Personality**: Boundary-pushing experimenter
   - **Specialties**: Experimental techniques, genre fusion, technology integration
   - **Commands**: 18 specialized commands for experimental sound design

7. **@music-orchestrator (Conductor)**
   - **Personality**: Strategic team coordinator
   - **Specialties**: Workflow management, team coordination, project leadership
   - **Commands**: BMAD integration with sophisticated pushback mechanisms

8. **@music-analyst-specialist (Archive)**
   - **Personality**: Intelligence preservation specialist
   - **Specialties**: Research extraction, knowledge crystallization
   - **Commands**: Specialized for research intelligence extraction

9. **@music-archivist (Keeper)**
   - **Personality**: Knowledge organization master
   - **Specialties**: Knowledge base management, information retrieval
   - **Commands**: Advanced knowledge organization and retrieval systems

#### BMAD Music Production Workflow (MANDATORY)

```bash
# Use @music-orchestrator to coordinate teams
@music-orchestrator → Assemble appropriate team for task
                   → Coordinate workflow execution
                   → Manage team pushback mechanisms

# Example: Track Production
@music-orchestrator → Assemble Full Production Team
@music-producer    → Creative direction and composition
@sound-designer    → Synthesis and sound design
@mix-engineer      → Professional mixing and mastering

# Example: Analysis Task
@music-orchestrator → Assemble Analysis Team  
@music-analyst     → Pattern recognition and genre analysis
@theory-engine     → Harmonic and rhythmic analysis
@music-archivist   → Knowledge preservation
```

#### Agent Infrastructure Location

All agents and supporting infrastructure are located in:
```
BMAD-AT-CLAUDE/expansion-packs/bmad-music-production/
├── config.yaml         # BMAD expansion pack configuration
├── agents/             # 9 specialized music production agents
├── agent-teams/        # 3 team coordination bundles
├── workflows/          # 4 comprehensive workflow YAML definitions
├── tasks/             # Comprehensive task libraries for all agents
├── templates/         # Production templates (basslines, leads, arrangements, mixing)
└── data/              # Synthesis parameters, research intelligence archive
```

#### Team Coordination Bundles

Use these pre-configured teams for common workflows:
- **hardcore-music-team.yaml**: Full production team (Producer, Designer, Engineer, Analyst)
- **analysis-intelligence-team.yaml**: Analysis and research team (Analyst, Theory, Archive, Keeper)
- **innovation-research-team.yaml**: Experimental research team (Innovation, Designer, Archive, Theory)

#### BMAD Agent Rules

- **MANDATORY: Use Task tool for ALL agent invocation**
- **ALWAYS use @music-orchestrator for team coordination**
- **Use team bundles for multi-agent workflows**
- **Follow workflow YAML definitions for complex tasks**
- **Leverage comprehensive template and task libraries**
- **Validate all work through agents, never manual creation**

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