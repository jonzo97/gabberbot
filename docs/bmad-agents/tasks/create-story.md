# Create Story Task - Scrum Master Workflow

## Purpose
Transform epic requirements into implementable story files with complete BMAD context for the Hardcore Music Production System.

## Workflow Steps

### 1. Epic Analysis
- Load the target epic from `docs/bmad-development/epics/`
- Identify the next logical story to create based on:
  - Dependencies already completed
  - Phase requirements from `bmad-planning/04-po-validation.md`
  - Current system state

### 2. Context Gathering

#### From PRD (`bmad-planning/03-prd.md`):
- Extract relevant functional requirements
- Identify affected user personas (Devin/Raveena)
- Note performance requirements
- Capture user story elements

#### From Architecture Spec (`bmad-planning/02-architecture-spec.md`):
- Identify architectural patterns to follow
- Note component responsibilities
- Extract technical constraints
- Reference multi-process architecture requirements

#### From Validation Strategy (`development/validation-testing-strategy.md`):
- Determine testing requirements for the phase
- Set quality gates
- Define success criteria

#### From Current State:
- Review `TECHNICAL_DEBT.md` for gotchas
- Check `INTEGRATION_MAP.md` for connection points
- Identify existing code to leverage from `cli_shared/`

### 3. Story Sizing

Estimate story size based on:
- **XS (1-2h)**: Single file change, clear requirements
- **S (2-4h)**: Multiple files, straightforward implementation
- **M (4-8h)**: New component or significant integration
- **L (8-16h)**: Complex feature with multiple components
- **XL (16+h)**: Architectural change or major feature

Consider:
- Testing requirements add 30-50% to base estimate
- Integration work adds complexity
- Audio/music domain specifics may increase effort

### 4. Story Creation

Use the story template to create a new story file:
```
docs/bmad-development/stories/story-{phase}-{epic}-{sequence}.yaml
```

Fill all sections:
- **Metadata**: Story ID, epic link, size, priority
- **Context**: Business value, epic relationship, current state
- **Requirements**: Functional and non-functional from PRD
- **Architecture**: Patterns, components, constraints from Architecture Spec
- **Acceptance Criteria**: Measurable completion conditions
- **Testing**: Test cases from validation strategy
- **Dev Notes**: Implementation hints, existing code references
- **Tasks**: Breakdown into subtasks with checkboxes

### 5. Dependency Management

- List all required stories that must complete first
- Note external dependencies (libraries, tools)
- Update epic tracking with new story
- Verify no circular dependencies

### 6. Validation

Before finalizing:
- [ ] All template sections filled
- [ ] Requirements traceable to PRD
- [ ] Architecture guidance included
- [ ] Testing requirements specified
- [ ] Acceptance criteria measurable
- [ ] Dependencies identified
- [ ] Size estimate reasonable

### 7. Story Sequencing

Update the story sequence in:
- Epic file (mark story as created)
- Sprint backlog (if applicable)
- Dependency graph

## Music Production Specific Considerations

### Audio Generation Stories
- Include BPM and key requirements
- Specify genre characteristics (hardcore/gabber/industrial)
- Note synthesis parameters from CLAUDE.md
- Reference existing generators in `cli_shared/generators/`

### MIDI Processing Stories
- Always use `MIDIClip` data model
- Reference pattern generators
- Include timing and velocity requirements

### AI Integration Stories
- Consolidate into `intelligent_music_agent_v2.py`
- Include prompt engineering requirements
- Specify parameter extraction needs

### Performance Critical Stories
- Note <20ms latency requirement for real-time
- Specify concurrent processing needs
- Include memory constraints

## Output

The completed story file should enable a developer to:
1. Understand what to build without additional context
2. Know which patterns and components to use
3. Have clear acceptance criteria
4. Understand testing requirements
5. Leverage existing code effectively

## Common Pitfalls to Avoid

- Creating stories that are too large (>16h)
- Missing architectural guidance
- Vague acceptance criteria
- Ignoring existing components
- Not considering technical debt
- Forgetting music domain requirements

## Success Criteria

A well-created story:
- Can be implemented without clarification
- Includes all necessary context
- References appropriate documentation
- Has clear, measurable completion criteria
- Fits within a reasonable time estimate
- Advances the epic goals effectively