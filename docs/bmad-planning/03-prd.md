# Product Requirements Document: Hardcore Music Production System

**Version:** 1.2 **Status:** Aligned & Approved **Author:** John, Product Manager 📋 **Purpose:** This document provides the complete functional and non-functional requirements for the Hardcore Music Production System. It consolidates all existing analysis and architectural planning into a unified vision, serving as the definitive guide for development. This PRD is aligned with the project's strategic positioning as a "local-first" AI music tool, focusing on a self-contained, high-performance desktop experience.

## 1\. COMPLETE FEATURE INVENTORY

This section documents all features, existing and planned, as formal requirements. It establishes a priority framework to guide the implementation strategy.

### **Feature List & Formal Requirements**

#### **Core Music Generation**

-   **MIDI Clip Architecture:** The system **shall** provide a robust data model for MIDI clips, including notes, velocity, timing, and accents. (Existing: ✅)
    
-   **Genre-Specific Pattern Generators:** The system **shall** include dedicated generators for acid basslines and tuned hardcore kicks. (Existing: ✅)
    
-   **Advanced Pattern Generators:** The system **shall** implement generators for riffs, arpeggios, and chord progressions. (Planned)
    
-   **Euclidean Rhythms:** The system **shall** support the generation of Euclidean rhythms. (Planned)
    
-   **MIDI Export:** The system **shall** be able to export any MIDI clip to a standard `.mid` file. (Existing: ✅)
    
-   **Alternative Pattern Exports:** The system **shall** support exporting patterns to TidalCycles and OSC message formats. (Planned Enhancement)
    

#### **Audio Synthesis & Processing**

-   **Real-Time Audio Engine:** The system **shall** use SuperCollider as the primary low-latency audio synthesis engine. (Planned)
    
-   **VST3 Plugin Support:** The system **shall** be capable of hosting and routing audio through third-party VST3 plugins. (Planned)
    
-   **Built-in Effects:** The system **shall** provide native audio effects, including distortion, filters (LP, HP, BP, Notch), and dynamics (compressor, limiter). (Planned Enhancement)
    
-   **Spatial Effects:** The system **shall** provide native spatial effects, including reverb and delay, available on dedicated send busses. (Planned)
    

#### **AI & Natural Language**

-   **Natural Language Prompting:** The system **shall** accept natural language text prompts as the primary input for music generation. (Planned)
    
-   **Intent Classification & Parameter Extraction:** The system's AI **shall** parse user prompts to identify musical intent (e.g., "create kick pattern") and extract key parameters (e.g., BPM, key, style). (Planned)
    
-   **Audio Analysis Feedback Loop:** The system **shall** be able to analyze its own audio output to inform subsequent AI generation, creating a "hearing" mechanism. (Planned)
    
-   **User Profile & Learning:** The system **shall** maintain a user profile to learn and adapt to a user's creative preferences over time (e.g., preferred reverb decay time, common kick drum envelope). (Planned)
    
-   **Knowledge Core:** The AI **shall** leverage a dedicated knowledge base of hardcore/gabber/industrial music theory and production techniques to ensure genre-authentic output. (Planned Enhancement)
    

#### **User Interface & Experience**

-   **Textual User Interface (TUI):** The primary user interface **shall** be a terminal-based application built with the Textual framework. (Planned)
    
-   **Real-Time Transport & Mixing:** The TUI **shall** provide controls for real-time playback, stopping, and mixing of multiple audio tracks. (Planned)
    
-   **Session View (Scenes):** The system **shall** support an Ableton-style "Session View" where clips on multiple tracks can be launched together as scenes. (Planned)
    
-   **Arrangement View:** The system **shall** provide a linear timeline for arranging scenes into a full song structure. (Planned)
    
-   **Undo/Redo:** The system **shall** implement a multi-level undo/redo stack for all state-changing operations. (Planned)
    

#### **Data & Project Management**

-   **Project State Persistence:** The system **shall** save and load the entire project state to a dedicated file format (`.hcs`). (Planned)
    
-   **User Library:** The system **shall** allow users to save and recall presets for instruments, effects chains, and MIDI patterns. (Planned)
    
-   **Professional Export:** The system **shall** be able to export the final arrangement as a master audio file and as individual track stems. (Planned)
    

### **Feature Priority Matrix**

This matrix aligns features with the phased implementation plan, balancing immediate user value against technical effort.

Priority

Feature

User Value

Technical Complexity

Rationale

**P0 (MVP)**

Natural Language -> Audio Generation

High

Medium

Validates the core "ChatOps" value proposition with compelling sound.

**P1**

Real-Time Audio Engine (SuperCollider)

Critical

High

Transforms the tool from a generator to an instrument.

**P1**

TUI with Transport & Mixer

Critical

High

Creates the primary interactive user experience.

**P1**

Undo/Redo Stack

High

Medium

Essential for a non-destructive creative workflow.

**P2**

Session View (Scenes)

High

Medium

Enables song-building and live jamming.

**P2**

User Profile & Learning

High

Medium

Makes the AI feel like a personalized partner.

**P2**

MIDI Hardware I/O

Medium

Medium

Adds tactile control for producers with hardware.

**P3**

Arrangement View

High

High

Enables creation of complete song structures.

**P3**

VST3 Plugin Support

High

High

Unlocks professional sound design capabilities.

**P3**

Professional Stem & Master Export

Critical

Medium

Bridges the gap to use in professional workflows.

## 2\. COMPREHENSIVE USER PERSONAS & JOURNEYS

### **User Personas**

1.  **Devin the Developer (Current User)**
    
    -   **Role:** Python developer with an interest in electronic music.
        
    -   **Goal:** To programmatically generate genre-specific MIDI patterns for use in other DAWs or personal projects.
        
    -   **Frustrations:** The current codebase is fragmented and undocumented, making it hard to find the working parts. He has to manually integrate the output into a separate music application.
        
2.  **Raveena the Producer (Target User)**
    
    -   **Role:** Electronic music producer focused on hardcore, techno, and industrial genres.
        
    -   **Goal:** To capture creative ideas quickly and develop them into full tracks without the friction of a traditional DAW. She wants an intelligent tool that understands her genre and collaborates with her.
        
    -   **Frustrations:** Creative blocks, spending too much time on repetitive tasks (like programming drum patterns), and losing the initial spark of an idea by the time she has her DAW project set up.
        

### **User Journeys**

-   **As-Is Journey (Devin):**
    
    1.  Clones the project repository from Git.
        
    2.  Spends 30 minutes digging through folders to locate `cli_shared/generators/acid_bassline.py`.
        
    3.  Writes a separate Python script to import and execute the generator function.
        
    4.  Runs the script from his terminal.
        
    5.  Locates the output `acid.mid` file.
        
    6.  Opens Ableton Live, creates a new project, and imports the MIDI file.
        
    7.  Assigns a synth to the MIDI track to finally hear the result.
        
    
    -   **Gap:** The entire journey is manual, requires coding knowledge, and happens outside a dedicated music creation environment.
        
-   **To-Be Journey (Raveena - MVP to V1):**
    
    1.  Opens her terminal and types `hardcore-music-app`. The TUI loads instantly.
        
    2.  In the prompt, she types: `give me a brutal 4-bar gabber kick at 190 bpm with a distorted tail`.
        
    3.  The generation worker processes the request. A new track appears labeled "Kick" with a MIDI clip.
        
    4.  She hits the spacebar. The pattern plays immediately in a loop, synthesized by the real-time audio engine.
        
    5.  She types `add a dark acid line to follow the kick's rhythm`. A second track appears and plays in sync.
        
    6.  She navigates to the mixer view and adjusts the volume of the acid line.
        
    7.  She dislikes the change and hits `Ctrl+Z` to undo it.
        
    
    -   **Success Metric:** Raveena creates a multi-track loop she is happy with in under 5 minutes, staying within a single application.
        

## 3\. FUNCTIONAL & NON-FUNCTIONAL REQUIREMENTS

### **Functional Requirements**

-   **FR-1 (AI Generation):** The system **shall** translate natural language prompts into structured `MIDIClip` data objects as defined in the architecture.
    
-   **FR-2 (Real-Time Playback):** The system **shall** send OSC commands to the `scsynth` audio engine to trigger and manipulate audio in real-time.
    
-   **FR-3 (State Management):** The `Main Controller` component **shall** be the sole authority for modifying the `ProjectState` data model.
    
-   **FR-4 (Background Processing):** All AI generation and audio analysis tasks **shall** be executed in separate background processes to ensure the TUI remains responsive.
    
-   **FR-5 (Project Persistence):** The system **shall** serialize the entire `ProjectState` object to a `.hcs` file upon user command (Save) and deserialize it upon user command (Load).
    
-   **FR-6 (Undo/Redo):** The system **shall** maintain a history stack of `ProjectState` objects to enable multi-level undo and redo operations.
    

### **Non-Functional Requirements**

-   **NFR-1 (Performance):** The real-time audio pipeline (MIDI input -> OSC -> Audio Output) **shall** have a round-trip latency of less than 20ms under typical load.
    
-   **NFR-2 (Performance):** AI generation of a single 4-bar pattern **shall** complete in under 10 seconds.
    
-   **NFR-3 (Reliability):** A crash or error in a background generation worker **shall not** crash the main application process. The error must be reported gracefully to the user in the TUI.
    
-   **NFR-4 (Usability):** All core functions (playback, prompting, mixing) **shall** be accessible via keyboard shortcuts within the TUI.
    
-   **NFR-5 (Configuration):** All system settings (API keys, audio device configuration) **shall** be managed via Pydantic models from a central configuration file.
    
-   **NFR-6 (Data Integrity):** The project file format **shall** be versioned to allow for backward compatibility or graceful migration as data models evolve.
    
-   **NFR-7 (Testability):** The codebase **shall** adhere to a testing pyramid with a target of 80% unit test coverage for all business logic in the Controller and data models.
    
-   **NFR-8 (Security):** All secrets, including external API keys (e.g., OpenAI, Gemini), **shall not** be committed to the source code repository. They **must** be managed via environment variables or a configuration file included in `.gitignore`.
    

## 4\. EPIC & STORY BREAKDOWN

Features are organized into epics that align with the phased implementation strategy.

### **Epic 1: The Offline AI Sound Generator (MVP)**

-   _Description:_ Prove the core concept by creating a single-process tool that takes a text prompt and renders a genre-correct, high-quality audio file.
    
-   **User Story 1.1:** As a producer, I want to run a script from my command line with a text prompt so that I receive a compelling `.wav` file of a generated musical idea.
    
    -   **AC:** `python main.py "create a 16th-note acid bassline at 160 bpm in A minor"` produces an audible `.wav` file that is musically coherent and has a high-quality synth sound.
        
-   **Technical Story 1.2:** As a developer, I need to establish the foundational engineering pillars (Poetry for dependencies, Pydantic for models) to ensure the project is maintainable from the start.
    
    -   **AC:** `poetry install` creates a reproducible environment. `ProjectState` and its sub-models are defined as Pydantic classes.
        

### **Epic 2: The Interactive Instrument & Mixer**

-   _Description:_ Transform the offline generator into a real-time, multi-track musical instrument with a TUI and non-destructive editing.
    
-   **User Story 2.1:** As a producer, I want to see my tracks in a text-based interface and control playback so that I can interact with my creation in real time.
    
    -   **AC:** A Textual TUI launches. Hitting 'Play' triggers audio playback via SuperCollider. Mixer values are displayed and updated.
        
-   **User Story 2.2:** As a producer, I want to undo any change I make to a pattern or the mixer so that I can experiment without fear of making a mistake.
    
    -   **AC:** Pressing `Ctrl+Z` reverts the last action (e.g., clip generation, volume change). Pressing `Ctrl+Y` re-applies it.
        
-   **Technical Story 2.3:** As a developer, I need to refactor the AI generation logic into a background worker process managed by the Main Controller to prevent the UI from freezing.
    
    -   **AC:** Initiating a generation task displays a "Working..." status in the TUI, which remains fully responsive. The result is returned via an IPC queue.
        

### **Epic 3: The Smart Jam Session Tool**

-   _Description:_ Add musical intelligence and expressiveness to the instrument through scenes, groove, and hardware control.
    
-   **User Story 3.1:** As a producer, I want to group clips into scenes and trigger them together so that I can build and perform different sections of a song.
    
    -   **AC:** The user can define a scene. Triggering the scene starts playback of all associated clips simultaneously.
        

## 5\. IMPLEMENTATION STRATEGY

This strategy is a direct adoption of the architect's phased roadmap, focusing on delivering iterative value while building on a stable core.

### **Phase 1: The MVP - "Offline AI Sound Generator"**

-   **Goal:** Validate the quality, musicality, and viability of the core AI-to-audio generation loop.
    
-   **Scope:** Epic 1. A single-process Python script with no GUI. Input is a command-line argument. The script will generate a MIDI pattern and render it to a `.wav` file using a basic but high-quality, genre-appropriate Python synthesizer.
    
-   **Success Metrics:**
    
    -   **Qualitative:** In a blind listening test, 3 out of 5 target users (Raveena persona) rate the generated audio files as "musically interesting" and "high-quality sound."
        
    -   **Quantitative:** Successful audio generation for 95% of test prompts.
        

### **Phase 2: The Interactive Instrument & Mixer**

-   **Goal:** Build the core real-time application and user experience.
    
-   **Scope:** Epic 2. Implement the full multi-process architecture (TUI, Controller, Generation Worker, Audio Engine). Deliver the TUI with mixing, real-time playback, and undo/redo.
    
-   **Success Metrics:**
    
    -   **Engagement:** Average user session time exceeds 15 minutes.
        
    -   **Task Completion:** 80% of beta testers can successfully create and mix a 4-track loop without assistance.
        
    -   **Performance:** Audio latency remains below the 20ms target during typical use.
        

### **Phase 3: The Smart Jam Session Tool**

-   **Goal:** Evolve the instrument into an intelligent musical partner.
    
-   **Scope:** Epic 3. Implement scenes, groove templates, user profile learning, and basic MIDI hardware control.
    
-   **Success Metrics:**
    
    -   **Usability:** Users report the "Groove" feature significantly improves the feel of generated patterns.
        
    -   **Adoption:** 50% of beta testers successfully map and use a MIDI controller.
        

### **Phase 4: The AI Arranger & Professional Exporter**

-   **Goal:** Complete the vision of an end-to-end music production environment.
    
-   **Scope:** Implement the Arrangement timeline, teach the AI to structure full songs, integrate VST3 support, and build the stem/master export manager.
    
-   **Success Metrics:**
    
    -   **Completion:** A user can go from a single text prompt to a fully arranged, 2-minute track that can be exported as stems.
        
    -   **Integration:** Successful loading and audio routing of 5 popular commercial VST3 plugins.

