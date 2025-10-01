# Epic 1: The Prototyper (MVP)

**Phase:** 1 **Goal:** Prove the core value proposition by creating a single-process, command-line tool that takes a text prompt and renders a high-quality, genre-correct `.wav` file.

## Description

This phase focuses on leveraging the project's strongest existing assets—the MIDI data models and pattern generators—and connecting them to a single, consolidated AI engine. We will create an "offline" generator that provides immediate value to our current user persona (Devin the Developer) and serves as the functional prototype for the entire system. This is a validation phase for both the technology and the user experience.

### User Story 1.1: Generate a musical idea from a text prompt

-   **As a:** Producer (playing the role of Devin)
    
-   **I want to:** Run a single script from my command line with a text prompt
    
-   **So that:** I receive a compelling `.wav` file of a generated musical idea without needing a DAW.
    
-   **Acceptance Criteria:**
    
    1.  A command `python main.py "create a 16th-note acid bassline at 160 bpm in A minor"` executes without errors.
        
    2.  The command produces an audible `.wav` file in the output directory.
        
    3.  The generated audio is musically coherent, matches the prompt's parameters (BPM, key), and has a high-quality synth sound.
        
    4.  The script provides clear feedback to the console (e.g., "Generating pattern...", "Rendering audio...", "Done. File saved to output.wav").
        

### Technical Story 1.2: Establish foundational engineering pillars

-   **As a:** Developer
    
-   **I need to:** Establish robust dependency and configuration management from the start
    
-   **So that:** The project is maintainable, reproducible, and scalable.
    
-   **Acceptance Criteria:**
    
    1.  The project uses **Poetry** for dependency management; `poetry install` creates a complete, reproducible virtual environment.
        
    2.  All core data structures (`MIDIClip`, `Note`, etc.) are defined as **Pydantic** models in `common/models.py`.
        
    3.  Secrets (like API keys) are loaded from environment variables or a `.env` file, not hardcoded.
        
    4.  A basic unit test suite is established, and tests for the core MIDI generation logic pass.
        

### Technical Story 1.3: Consolidate AI and Synthesis Logic

-   **As a:** Developer
    
-   **I need to:** Create a single, linear text-to-audio pipeline
    
-   **So that:** The core generation logic is clear, testable, and free of architectural confusion.
    
-   **Acceptance Criteria:**
    
    1.  All AI logic is consolidated into a single `GenerationService` that takes text and returns a `MIDIClip` object.
        
    2.  All audio rendering logic is consolidated into a single `AudioService` that takes a `MIDIClip` object and returns a `.wav` file.
        
    3.  The `main.py` script orchestrates these two services.
        
    4.  Code for all unused AI agents and synthesis engines is deleted.

