# Epic 3: The Partner

**Phase:** 3 **Goal:** Evolve the instrument into an intelligent musical partner that has a sense of rhythm, learns from the user, and can be controlled with hardware.

## Description

With the core architecture in place, this phase is about adding musicality and intelligence. We will implement features that make the application feel less like a static tool and more like a collaborative partner. This includes building the Ableton-style Session View (Scenes), adding groove and feel, and allowing the AI to learn from user choices.

### User Story 3.1: Build song sections with scenes

-   **As a:** Producer
    
-   **I want to:** Group clips from different tracks into scenes and trigger them together
    
-   **So that:** I can build and perform different sections of a song (e.g., intro, drop, breakdown).
    
-   **Acceptance Criteria:**
    
    1.  The TUI includes a "Scene" panel.
        
    2.  The user can create a new scene that captures the currently playing clips.
        
    3.  Triggering a scene (e.g., by pressing 'Enter' on it) launches all associated clips simultaneously and in sync.
        
    4.  The user can navigate and launch different scenes to create a live arrangement.
        

### User Story 3.2: Add human feel to my patterns

-   **As a:** Producer
    
-   **I want to:** Apply groove templates to my MIDI patterns
    
-   **So that:** My rhythms sound less robotic and have more human feel.
    
-   **Acceptance Criteria:**
    
    1.  The system includes a library of classic groove templates (e.g., "MPC Swing 16-65").
        
    2.  The user can apply a selected groove to any `MIDIClip`.
        
    3.  Applying a groove non-destructively adjusts the timing and velocity of notes during real-time playback.
        
    4.  The underlying `MIDIClip` data remains unchanged.
        

### User Story 3.3: Control the app with my MIDI controller

-   **As a:** Producer
    
-   **I want to:** Map my hardware MIDI controller's knobs and buttons to functions in the app
    
-   **So that:** I can have tactile, hands-on control over mixing and playback.
    
-   **Acceptance Criteria:**
    
    1.  The application detects and connects to available MIDI devices.
        
    2.  A `MIDI_IO` listener runs in the real-time process.
        
    3.  The user can enter a "MIDI Learn" mode.
        
    4.  Moving a hardware knob and then clicking a parameter in the TUI creates a mapping.
        
    5.  Mapped controls adjust their corresponding parameters in the `AudioEngine` with low latency.

