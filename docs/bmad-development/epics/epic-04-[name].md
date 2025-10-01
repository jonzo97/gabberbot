# Epic 4: The Studio

**Phase:** 4 **Goal:** Fulfill the complete vision of an end-to-end music production environment that can take an idea from prompt to a fully arranged and professionally exportable track.

## Description

This final phase adds the features necessary for creating complete songs and integrating with professional workflows. We will build the linear Arrangement View, enable the AI to assist with structuring entire tracks, and, most critically, add VST3 plugin support and high-quality stem export. This makes the application a legitimate tool for finishing music, not just starting it.

### User Story 4.1: Arrange my scenes into a full song

-   **As a:** Producer
    
-   **I want to:** Place my scenes on a linear timeline
    
-   **So that:** I can create a complete song structure from start to finish.
    
-   **Acceptance Criteria:**
    
    1.  The TUI includes a toggle to switch to an "Arrangement View".
        
    2.  The user can drag/place scenes from the library onto the timeline at specific bar numbers.
        
    3.  Playing from the Arrangement View sequences the scenes in order.
        
    4.  The user can edit and loop sections of the arrangement.
        

### User Story 4.2: Use my professional plugins

-   **As a:** Producer
    
-   **I want to:** Load my own VST3 instruments and effects plugins
    
-   **So that:** I can use my signature sounds and professional tools within the application.
    
-   **Acceptance Criteria:**
    
    1.  The application scans standard system paths for VST3 plugins on startup.
        
    2.  The user can replace a track's default synthesizer with a VST3 instrument.
        
    3.  The user can add VST3 effects to the insert slots on any track or bus.
        
    4.  The TUI provides a generic interface for tweaking VST3 parameters.
        
    5.  Audio is correctly routed through the VST3 plugins in the `AudioEngine`.
        

### User Story 4.3: Export my track for release

-   **As a:** Producer
    
-   **I want to:** Export my final arrangement as a high-quality master file and as individual track stems
    
-   **So that:** I can send it for mastering, upload it for distribution, or collaborate with others.
    
-   **Acceptance Criteria:**
    
    1.  An "Export" option is available from the main menu.
        
    2.  The user can choose to export "Master" or "Stems".
        
    3.  Exporting renders the full arrangement timeline to `.wav` files (24-bit, 48kHz).
        
    4.  "Master" export produces a single stereo file.
        
    5.  "Stems" export produces a separate, perfectly synced `.wav` file for each track.

