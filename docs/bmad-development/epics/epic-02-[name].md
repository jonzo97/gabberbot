# Epic 2: The Instrument

**Phase:** 2 **Goal:** Transform the offline generator into a real-time, multi-track musical instrument with an interactive TUI and non-destructive editing.

## Description

This phase is focused on building the architectural foundation for the complete application vision. We will implement the multi-process model defined in the architecture spec, introducing the real-time `AudioEngine` (SuperCollider), the background `Generation Worker`, and the `Main Controller`. This is the most technically intensive phase, transforming the prototype into a robust, responsive application.

### User Story 2.1: Interact with my music in real-time

-   **As a:** Producer (Raveena)
    
-   **I want to:** See my tracks in a text-based interface and control playback with my keyboard
    
-   **So that:** I can interact with my creation in real time and make immediate creative decisions.
    
-   **Acceptance Criteria:**
    
    1.  Running `hardcore-music-app` launches a **Textual TUI**.
        
    2.  The TUI displays a list of tracks and a master mixer.
        
    3.  Pressing the spacebar starts/stops looped playback of all clips.
        
    4.  Audio is generated in real-time by the **SuperCollider engine**.
        
    5.  The TUI remains responsive at all times, even during playback.
        

### User Story 2.2: Experiment without fear

-   **As a:** Producer
    
-   **I want to:** Undo any change I make to a pattern, the mixer, or an effect
    
-   **So that:** I can experiment freely without fear of losing my work or making a permanent mistake.
    
-   **Acceptance Criteria:**
    
    1.  The `Main Controller` implements a history stack for the `ProjectState`.
        
    2.  Pressing `Ctrl+Z` reverts the last action (e.g., clip generation, volume change, adding an effect).
        
    3.  The TUI updates instantly to reflect the reverted state.
        
    4.  Pressing `Ctrl+Y` (or `Ctrl+Shift+Z`) re-applies the undone action.
        
    5.  The undo stack can handle at least 20 previous states.
        

### Technical Story 2.3: Decouple generation from performance

-   **As a:** Developer
    
-   **I need to:** Refactor the AI generation logic into a background worker process
    
-   **So that:** The UI never freezes or becomes unresponsive while waiting for the LLM.
    
-   **Acceptance Criteria:**
    
    1.  The `Main Controller` manages a `multiprocessing.Queue` for jobs.
        
    2.  When a user prompts for generation, a job is sent to the `Generation Worker` process.
        
    3.  The TUI displays a non-blocking "Generating..." status indicator.
        
    4.  When the worker completes, the result is sent back to the `Main Controller` via a results queue and integrated into the `ProjectState`.
        
    5.  Errors in the worker process are caught, reported to the user in the TUI, and do not crash the main application.

