#!/usr/bin/env python3
"""
Configuration Management for Hardcore Music Production CLI

Handles settings, preferences, and customization for both Strudel and SuperCollider backends.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

class ThemeType(Enum):
    """Available UI themes"""
    INDUSTRIAL_DARK = "industrial_dark"
    GABBER_ORANGE = "gabber_orange"
    WAREHOUSE_MINIMAL = "warehouse_minimal"
    ACID_GREEN = "acid_green"
    CUSTOM = "custom"

@dataclass
class AudioConfig:
    """Audio system configuration"""
    sample_rate: int = 44100
    buffer_size: int = 512
    backend: str = "supercollider"  # "strudel" or "supercollider"
    device: Optional[str] = None
    latency_ms: float = 10.0
    
    # Hardcore-specific audio settings
    kick_boost_db: float = 3.0
    master_limiter: bool = True
    crunch_processing: bool = True
    doorlussen_enabled: bool = True

@dataclass
class MIDIConfig:
    """MIDI controller configuration"""
    enabled: bool = True
    controller_type: str = "auto"  # "auto", "launchpad_mk1", etc.
    device_name: Optional[str] = None
    
    # Hardware mappings
    launchpad_mapping: Dict[str, Any] = None
    midi_fighter_mapping: Dict[str, Any] = None
    
    # MIDI settings
    velocity_curve: str = "linear"  # "linear", "exponential", "logarithmic"
    note_repeat: bool = False
    step_recording: bool = True

@dataclass
class UIConfig:
    """User interface configuration"""
    theme: ThemeType = ThemeType.INDUSTRIAL_DARK
    show_spectrum_analyzer: bool = True
    show_kick_analyzer: bool = True
    vim_navigation: bool = True
    
    # TUI specific settings
    refresh_rate_hz: int = 60
    smooth_animations: bool = True
    ascii_visualizers: bool = True
    
    # Panel layout
    panel_layout: str = "default"  # "default", "minimal", "producer", "performer"
    show_panel_borders: bool = True
    status_bar_position: str = "bottom"  # "top", "bottom", "hidden"

@dataclass
class HardcoreConfig:
    """Hardcore-specific production settings"""
    default_bpm: float = 175.0
    preferred_kick_type: str = "gabber_kick"
    auto_crunch_factor: float = 0.7
    sidechain_intensity: float = 0.8
    
    # Genre presets
    gabber_mode: bool = False
    industrial_mode: bool = False
    rawstyle_mode: bool = False
    
    # Production preferences
    quantize_strength: float = 1.0
    swing_amount: float = 0.0
    pattern_length: int = 16
    auto_arrange: bool = False

@dataclass
class KeybindConfig:
    """Keyboard shortcut configuration"""
    # Transport controls
    play_stop: str = "space"
    record: str = "r"
    loop: str = "l"
    metronome: str = "m"
    
    # Navigation
    step_left: str = "h"
    step_right: str = "l" 
    track_up: str = "k"
    track_down: str = "j"
    
    # Pattern editing
    clear_step: str = "x"
    copy_step: str = "c"
    paste_step: str = "v"
    
    # Synthesis
    trigger_kick: str = "1"
    trigger_bass: str = "2"
    trigger_synth: str = "3"
    trigger_stab: str = "4"
    
    # Quick functions
    increase_crunch: str = "+"
    decrease_crunch: str = "-"
    quick_save: str = "ctrl+s"
    quick_export: str = "ctrl+e"

@dataclass
class GabberbotConfig:
    """Complete Gabberbot configuration"""
    audio: AudioConfig = None
    midi: MIDIConfig = None
    ui: UIConfig = None
    hardcore: HardcoreConfig = None
    keybinds: KeybindConfig = None
    
    # Session settings
    auto_save: bool = True
    save_interval_minutes: int = 5
    backup_patterns: bool = True
    
    # Performance monitoring
    show_cpu_usage: bool = True
    show_latency_meter: bool = True
    performance_warnings: bool = True
    
    def __post_init__(self):
        if self.audio is None:
            self.audio = AudioConfig()
        if self.midi is None:
            self.midi = MIDIConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.hardcore is None:
            self.hardcore = HardcoreConfig()
        if self.keybinds is None:
            self.keybinds = KeybindConfig()

class ConfigManager:
    """Configuration manager for Gabberbot"""
    
    DEFAULT_CONFIG_PATH = Path.home() / ".config" / "gabberbot"
    CONFIG_FILE = "config.json"
    THEMES_DIR = "themes"
    PRESETS_DIR = "presets"
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or self.DEFAULT_CONFIG_PATH
        self.config_file = self.config_dir / self.CONFIG_FILE
        self.themes_dir = self.config_dir / self.THEMES_DIR
        self.presets_dir = self.config_dir / self.PRESETS_DIR
        
        self._ensure_config_directories()
        self.config = self._load_or_create_config()
    
    def _ensure_config_directories(self):
        """Create config directories if they don't exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.themes_dir.mkdir(exist_ok=True)
        self.presets_dir.mkdir(exist_ok=True)
    
    def _load_or_create_config(self) -> GabberbotConfig:
        """Load existing config or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return self._dict_to_config(data)
            except Exception as e:
                print(f"⚠️ Error loading config: {e}. Using defaults.")
        
        # Create default config
        config = GabberbotConfig()
        self.save_config(config)
        return config
    
    def _dict_to_config(self, data: Dict[str, Any]) -> GabberbotConfig:
        """Convert dictionary to config object"""
        config = GabberbotConfig()
        
        if "audio" in data:
            config.audio = AudioConfig(**data["audio"])
        if "midi" in data:
            config.midi = MIDIConfig(**data["midi"])
        if "ui" in data:
            ui_data = data["ui"].copy()
            if "theme" in ui_data:
                ui_data["theme"] = ThemeType(ui_data["theme"])
            config.ui = UIConfig(**ui_data)
        if "hardcore" in data:
            config.hardcore = HardcoreConfig(**data["hardcore"])
        if "keybinds" in data:
            config.keybinds = KeybindConfig(**data["keybinds"])
        
        # Copy other fields
        for key, value in data.items():
            if key not in ["audio", "midi", "ui", "hardcore", "keybinds"]:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def _config_to_dict(self, config: GabberbotConfig) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        data = asdict(config)
        
        # Convert enums to strings
        if "ui" in data and "theme" in data["ui"]:
            data["ui"]["theme"] = data["ui"]["theme"].value
        
        return data
    
    def save_config(self, config: Optional[GabberbotConfig] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            data = self._config_to_dict(config)
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get_config(self) -> GabberbotConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = GabberbotConfig()
        self.save_config()
    
    def get_theme_config(self, theme: ThemeType) -> Dict[str, Any]:
        """Get theme configuration"""
        theme_file = self.themes_dir / f"{theme.value}.json"
        
        if theme_file.exists():
            try:
                with open(theme_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading theme {theme.value}: {e}")
        
        # Return default theme config
        return self._get_default_theme_config(theme)
    
    def _get_default_theme_config(self, theme: ThemeType) -> Dict[str, Any]:
        """Get default theme configuration"""
        
        if theme == ThemeType.INDUSTRIAL_DARK:
            return {
                "name": "Industrial Dark",
                "colors": {
                    "background": "#1a1a1a",
                    "foreground": "#e0e0e0", 
                    "accent": "#ff6b00",      # Industrial orange
                    "kick_color": "#ff0040",   # Bright red for kicks
                    "bass_color": "#00ff40",   # Green for bass
                    "synth_color": "#40a0ff",  # Blue for synths
                    "warning": "#ffff00",
                    "error": "#ff0000",
                    "success": "#00ff00"
                },
                "ascii_chars": {
                    "kick": "▮",
                    "bass": "▬",
                    "synth": "▲",
                    "empty": "░"
                }
            }
        
        elif theme == ThemeType.GABBER_ORANGE:
            return {
                "name": "Gabber Orange",
                "colors": {
                    "background": "#2a1a0a",
                    "foreground": "#ffcc80",
                    "accent": "#ff8f00",
                    "kick_color": "#ff0000",
                    "bass_color": "#ff4500",
                    "synth_color": "#ffa500",
                    "warning": "#ffff00",
                    "error": "#ff0000", 
                    "success": "#90ff90"
                },
                "ascii_chars": {
                    "kick": "█",
                    "bass": "▓",
                    "synth": "▒",
                    "empty": "░"
                }
            }
        
        elif theme == ThemeType.WAREHOUSE_MINIMAL:
            return {
                "name": "Warehouse Minimal",
                "colors": {
                    "background": "#0f0f0f",
                    "foreground": "#c0c0c0",
                    "accent": "#606060",
                    "kick_color": "#ffffff",
                    "bass_color": "#a0a0a0",
                    "synth_color": "#808080",
                    "warning": "#ffff80",
                    "error": "#ff8080",
                    "success": "#80ff80"
                },
                "ascii_chars": {
                    "kick": "■",
                    "bass": "▪",
                    "synth": "∙",
                    "empty": " "
                }
            }
        
        elif theme == ThemeType.ACID_GREEN:
            return {
                "name": "Acid Green",
                "colors": {
                    "background": "#001a00",
                    "foreground": "#80ff80",
                    "accent": "#00ff00", 
                    "kick_color": "#ff0080",
                    "bass_color": "#00ff80",
                    "synth_color": "#80ff00",
                    "warning": "#ffff00",
                    "error": "#ff0040",
                    "success": "#00ff00"
                },
                "ascii_chars": {
                    "kick": "●",
                    "bass": "◆",
                    "synth": "▼",
                    "empty": "·"
                }
            }
        
        else:
            return self._get_default_theme_config(ThemeType.INDUSTRIAL_DARK)
    
    def save_theme(self, theme: ThemeType, config: Dict[str, Any]):
        """Save custom theme configuration"""
        theme_file = self.themes_dir / f"{theme.value}.json"
        try:
            with open(theme_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Theme {theme.value} saved")
        except Exception as e:
            print(f"❌ Error saving theme: {e}")
    
    def list_available_presets(self) -> List[str]:
        """List available hardcore presets"""
        presets = []
        if self.presets_dir.exists():
            for file in self.presets_dir.glob("*.json"):
                presets.append(file.stem)
        return presets
    
    def load_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Load a hardcore preset"""
        preset_file = self.presets_dir / f"{preset_name}.json"
        if preset_file.exists():
            try:
                with open(preset_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading preset {preset_name}: {e}")
        return None
    
    def save_preset(self, preset_name: str, preset_data: Dict[str, Any]):
        """Save a hardcore preset"""
        preset_file = self.presets_dir / f"{preset_name}.json"
        try:
            with open(preset_file, 'w') as f:
                json.dump(preset_data, f, indent=2)
            print(f"✅ Preset {preset_name} saved")
        except Exception as e:
            print(f"❌ Error saving preset: {e}")
    
    def create_default_presets(self):
        """Create default hardcore presets"""
        presets = {
            "classic_gabber": {
                "bpm": 180,
                "kick_type": "gabber_kick",
                "crunch_factor": 0.8,
                "sidechain_intensity": 0.9,
                "pattern_style": "4_4_hardcore"
            },
            "industrial_rumble": {
                "bpm": 140,
                "kick_type": "industrial_kick",
                "crunch_factor": 0.6,
                "sidechain_intensity": 0.7,
                "pattern_style": "industrial_sparse"
            },
            "rawstyle_power": {
                "bpm": 150,
                "kick_type": "rawstyle_kick",
                "crunch_factor": 0.9,
                "sidechain_intensity": 0.85,
                "pattern_style": "rawstyle_reverse"
            },
            "uptempo_madness": {
                "bpm": 220,
                "kick_type": "piep_kick",
                "crunch_factor": 0.95,
                "sidechain_intensity": 0.8,
                "pattern_style": "uptempo_rush"
            }
        }
        
        for name, preset in presets.items():
            self.save_preset(name, preset)
    
    def get_keybind(self, action: str) -> Optional[str]:
        """Get keybinding for an action"""
        return getattr(self.config.keybinds, action, None)
    
    def set_keybind(self, action: str, key: str):
        """Set keybinding for an action"""
        if hasattr(self.config.keybinds, action):
            setattr(self.config.keybinds, action, key)
            self.save_config()
    
    def export_config(self, export_path: Path):
        """Export configuration to external file"""
        try:
            data = self._config_to_dict(self.config)
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Configuration exported to {export_path}")
        except Exception as e:
            print(f"❌ Error exporting config: {e}")
    
    def import_config(self, import_path: Path):
        """Import configuration from external file"""
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            self.config = self._dict_to_config(data)
            self.save_config()
            print(f"✅ Configuration imported from {import_path}")
        except Exception as e:
            print(f"❌ Error importing config: {e}")

def test_config_manager():
    """Test configuration manager"""
    print("⚙️ Testing Configuration Manager")
    print("=" * 40)
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Get current config
    config = config_manager.get_config()
    print(f"📋 Current backend: {config.audio.backend}")
    print(f"📋 Default BPM: {config.hardcore.default_bpm}")
    print(f"📋 Theme: {config.ui.theme.value}")
    
    # Test theme loading
    theme_config = config_manager.get_theme_config(config.ui.theme)
    print(f"🎨 Theme colors: {list(theme_config.get('colors', {}).keys())}")
    
    # Create default presets
    config_manager.create_default_presets()
    presets = config_manager.list_available_presets()
    print(f"🎵 Available presets: {presets}")
    
    # Test preset loading
    if presets:
        preset = config_manager.load_preset(presets[0])
        print(f"🎵 Loaded preset '{presets[0]}': {preset}")
    
    print("✅ Configuration manager test completed")

if __name__ == "__main__":
    test_config_manager()