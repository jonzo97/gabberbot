# Architecture Canvas: Hardcore Music Production System

**Version: 5.0 (TUI-Centric, Production-Ready)** **Author: Winston, Architect** **Status: The Definitive Blueprint**

This document specifies the final, complete architecture for the Hardcore Music Production System. It incorporates all feature requirements and the foundational engineering practices necessary for building a stable, resilient, and professional-grade creative application. This is our source of truth.

## 1.0 Guiding Principles

-   **Local-First:** A performant, self-contained desktop application.
    
-   **Pragmatism Over Complexity:** Start simple, add power where it matters.
    
-   **Decouple Generation from Performance:** A non-negotiable dual-pipeline for a responsive experience.
    
-   **Model-Driven:** Clean, explicit data models are the source of truth.
    
-   **Phased Evolution:** Build features incrementally on a stable core, always maintaining a working product.
    

## 2.0 System Architecture Diagram

The high-level, multi-process architecture is confirmed as the correct model for balancing responsiveness and powerful background processing.

    graph TD
        subgraph Main Process (The Conductor)
            TUI[Textual TUI]
            Controller[Main Controller & State Manager]
        end
    
        subgraph Background Process Pool
            GenWorker[Generation Worker]
            AnalysisWorker[Analysis Worker]
        end
        
        subgraph Real-Time Process (Performance Pipeline)
            AudioEngine[Audio Engine (SuperCollider `scsynth` + VST Host)]
            MIDI_IO[MIDI I/O Listener]
        end
    
        TUI -- User Commands --> Controller
        Controller -- UI Updates --> TUI
        Controller -- IPC (Jobs) --> GenWorker
        Controller -- IPC (Jobs) --> AnalysisWorker
        GenWorker -- IPC (Results) --> Controller
        AnalysisWorker -- IPC (Results) --> Controller
        Controller -- OSC Commands --> AudioEngine
        
        MIDIHardware[MIDI Hardware] --> MIDI_IO
        MIDI_IO -- Low-Latency OSC --> AudioEngine
        
        AudioEngine --> AudioOutput[User Soundcard]
    

## 3.0 Component Breakdown & Responsibilities

### 3.1 Main Process (The Conductor)

-   **`TUI (Textual TUI)`**
    
    -   **Responsibility:** The face of the application. It renders the `ProjectState` and captures all user input, delegating all logic to the `Controller`.
        
-   **`Main Controller & State Manager`**
    
    -   **Responsibility:** The brain of the application. It is the sole authority for managing the `ProjectState` and orchestrating all other components.
        
    -   **Core Tasks:**
        
        1.  **State Management & Undo/Redo:** Holds the current `ProjectState` in memory and manages the history stack for undo/redo functionality.
            
        2.  **Job & Process Management:** Manages a pool of background workers, assigns unique `job_id`s to tasks, and handles results or errors gracefully.
            
        3.  **Real-Time Orchestration:** Sends low-latency OSC commands to the `AudioEngine` for playback, mixing, and parameter changes.
            
        4.  **Groove, Profile, & Export Management:** Applies groove templates, updates the user profile with learned preferences, and manages the high-quality export of stems and masters.
            
        5.  **Persistence:** Manages saving and loading the `ProjectState` and `UserProfile` to and from the file system.
            

### 3.2 Background Process Pool

-   **`Generation Worker`**
    
    -   **Responsibility:** Executes all high-latency AI and music generation tasks without blocking the main application.
        
    -   **Core Tasks:** Assembles rich prompts for the LLM, communicates with external APIs, and translates the responses into structured musical data for the `ProjectState`.
        
-   **`Analysis Worker`**
    
    -   **Responsibility:** Provides the AI with a "hearing" mechanism by analyzing audio.
        
    -   **Core Tasks:** Receives audio buffers and returns structured `AudioAnalysis` objects containing characteristics like frequency content, envelope, and timbre.
        

### 3.3 Real-Time Process (Performance Pipeline)

-   **`Audio Engine (SuperCollider scsynth + VST Host)`**
    
    -   **Responsibility:** Make sound with the lowest possible latency. It is a "dumb" but powerful engine that executes commands from the `Controller`.
        
    -   **Core Tasks:** Loads synths and VST plugins, builds a virtual mixer, plays notes, and triggers samples.
        
-   **`MIDI I/O Listener`**
    
    -   **Responsibility:** Provides a direct, low-latency path from hardware to the `AudioEngine`.
        
    -   **Core Tasks:** Listens for MIDI messages and translates them directly into OSC commands for real-time parameter control.
        

## 4.0 The Foundational Pillars: Engineering Discipline

These are the non-negotiable engineering practices required for a stable and maintainable project.

1.  **Dependency Management:** We will use **Poetry** for reproducible environments.
    
2.  **Testing Strategy:** We will adhere to a **Unit -> Integration -> E2E** testing pyramid.
    
3.  **Configuration Management:** We will use **Pydantic** for type-safe, centralized settings.
    
4.  **Continuous Integration (CI):** We will use **GitHub Actions** for automated quality control.
    

## 5.0 The Connective Tissue: How It All Works Together

This section makes the implicit mechanisms of communication and data management explicit.

-   **Persistence Strategy:** The `ProjectState` is saved as an explicit `.hcs` file. The `UserLibrary` and `UserProfile` are saved to a dedicated application support directory.
    
-   **Inter-Process Communication (IPC):** We will use Python's `multiprocessing.Queue` to send jobs and receive results asynchronously between the `Controller` and the background workers, using unique `job_id`s to track every task.
    
-   **Error Handling & State Synchronization:** The `Controller` is the _only_ component allowed to modify the `ProjectState`. Workers operate on copies of data and return results, which the Controller safely integrates. This ensures stability and a single source of truth.
    

## 6.0 Data Models (Final & Complete)

This is the definitive data structure for the entire application, designed for serialization and state management.

    # --- Core Musical Data ---
    class AutomationPoint:
        time: float; value: float
    
    class AutomationClip:
        parameter_name: str; points: list[AutomationPoint]
    
    class MIDIClip:
        id: str; name: str; notes: list[...] # Represents notes, timing, velocity
    
    class AudioAnalysis:
        dominant_freq_hz: float; envelope_shape: str; spectral_centroid: float
    
    # --- Groove & Timing ---
    class GrooveTemplate:
        id: str; name: str
        timing_map: dict[float, float]
        velocity_map: dict[float, float]
    
    # --- Mixing & Effects ---
    class Effect:
        id: str; plugin_type: str # 'vst3' or 'native'
        plugin_name: str; parameters: dict
    
    class MixerSettings:
        volume_db: float; pan: float
        insert_effects: list[Effect]; send_levels: dict[str, float]
    
    # --- User Library & Profile ---
    class Preset:
        id: str; preset_type: str # 'instrument', 'fx_chain'
        name: str; data: dict
    
    class UserLibrary:
        instruments: list[Preset]; fx_chains: list[Preset]
    
    class UserProfile:
        learned_parameters: dict # e.g., {'mixer.reverb.send_level': -3.0}
    
    # --- Arrangement & Scenes (Ableton-style Session View) ---
    class Scene:
        id: str; name: str
        clip_slots: dict[str, str] # {track_id: midi_clip_id}
    
    class Arrangement:
        timeline: dict[float, str] # {beat_number: scene_id}
    
    # --- The Complete Project State ---
    class ProjectState:
        bpm: float; key: str
        tracks: list['Track']
        fx_busses: dict[str, list[Effect]] # e.g., {'ReverbBus': [SCReverb(...)]}
        scenes: list[Scene]
        arrangement: Arrangement | None
        
        # Foundational Pillars
        user_library: UserLibrary
        user_profile: UserProfile
        groove_templates: list[GrooveTemplate]
        active_groove_id: str | None
        
    class Track:
        id: str; name:str
        midi_clips: dict[str, MIDIClip]
        automation_clips: list[AutomationClip]
        audio_analysis: AudioAnalysis | None
        mixer_settings: MixerSettings
    

## 7.0 Phased Implementation Roadmap

This roadmap introduces complexity in logical stages, ensuring we always have a working product.

### Phase 1: The MVP - "Offline AI Pattern Generator"

-   **Goal:** Prove the core text-to-audio concept and validate the AI's musical output quality.
    
-   **Architecture:** Single-process, synchronous script. No background workers, no real-time engine.
    
-   **Core Features:**
    
    -   Simple TUI for text input.
        
    -   Call external LLM API.
        
    -   Utilize `Knowledge Core` to generate one `MIDIClip`.
        
    -   Render to a `.wav` file using a basic Python synthesizer.
        
    -   Implement foundational pillars: Poetry, Pydantic, and basic unit tests.
        
-   **Outcome:** A powerful tool for generating musical ideas that can be used in other DAWs.
    

### Phase 2: The Interactive Instrument & Mixer

-   **Goal:** Introduce real-time playback, a full mixing environment, and essential usability features.
    
-   **Architecture:** Implement the full multi-process model (Controller, AudioEngine, Generation Worker).
    
-   **Core Features:**
    
    -   Integrate SuperCollider (`scsynth`) as the `AudioEngine`.
        
    -   Refactor AI generation into the background `Generation Worker`.
        
    -   Implement the full `MixerSettings` and `Effect` models in the `AudioEngine`.
        
    -   **Implement the Undo/Redo stack in the `Main Controller`.**
        
    -   Build the `UserLibrary` for saving and loading presets.
        
-   **Outcome:** An interactive jam tool where users can create and mix multi-track loops in real time.
    

### Phase 3: The Smart Jam Session Tool

-   **Goal:** Make the tool feel alive, intelligent, and musical.
    
-   **Architecture:** Build upon the V2 architecture.
    
-   **Core Features:**
    
    -   Implement the `Scene` model for launching clips together.
        
    -   Implement the `GrooveEngine` to add human feel to patterns.
        
    -   Implement the `UserProfile` for implicit, learned preferences.
        
    -   Implement the `Analysis Worker` and the full audio feedback loop.
        
    -   Implement the `MIDI_IO Listener` for hardware control.
        
-   **Outcome:** An intelligent musical partner that learns from the user and has a sense of rhythm and musicality.
    

### Phase 4: The AI Arranger & Professional Exporter

-   **Goal:** Fulfill the complete vision of an AI-assisted production environment from start to finish.
    
-   **Architecture:** Build upon the V3 architecture.
    
-   **Core Features:**
    
    -   Implement the `Arrangement` model.
        
    -   Teach the `Generation Worker` how to create full song structures based on scenes.
        
    -   Implement the Stem & Master **Export Manager** for professional workflows.
        
    -   Integrate VST3 plugin support into the `AudioEngine`.
        
-   **Outcome:** A complete, end-to-end music creation tool that can take an idea from a simple text prompt to a fully arranged and exportable track.
    

## 8.0 Target Directory Structure

This structure provides a clean, scalable home for every component of our architecture.

    hardcore_music_app/
    ├── main.py                 # Application entry point
    ├── knowledge_base/         # AI's "producer handbook" (YAML, JSON files)
    │   ├── synthesis/
    │   └── patterns/
    ├── samples/                # Built-in sample library (.wav files)
    ├── analysis/               # Audio analysis logic (FFT, etc.)
    │   └── tools.py
    ├── tui/
    │   └── app.py              # Textual TUI definition
    ├── controller/
    │   └── controller.py       # Main Controller, Undo/Redo, Groove, etc.
    ├── generation/
    │   └── worker.py           # Generation Worker logic (LLM calls)
    ├── audio/
    │   ├── engine.py           # Wrapper for managing scsynth process
    │   └── midi.py             # MIDI I/O Listener logic
    ├── common/
    │   └── models.py           # All Python data models (ProjectState, etc.)
    ├── synthdefs/              # SuperCollider synth definitions (.scd files)
    ├── config/
    │   └── settings.py         # Logic for loading config.toml
    └── tests/
        └── ...

