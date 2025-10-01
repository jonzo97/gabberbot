"""
Configuration models for the Hardcore Music Production System.

Pydantic models for type-safe configuration management including
API keys, audio settings, and application preferences.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict, SecretStr


class AIConfig(BaseModel):
    """Configuration for AI/LLM providers."""
    
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key for GPT models"
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key for Claude models"
    )
    google_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Google API key for Gemini models"
    )
    default_provider: str = Field(
        default="openai",
        description="Default AI provider to use"
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4000,
        description="Maximum tokens for AI responses"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="AI temperature for creativity control"
    )
    
    @field_validator('default_provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate AI provider name."""
        valid_providers = ['openai', 'anthropic', 'google']
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of: {', '.join(valid_providers)}")
        return v


class AudioConfig(BaseModel):
    """Configuration for audio processing and synthesis."""
    
    sample_rate: int = Field(
        default=44100,
        description="Audio sample rate in Hz"
    )
    buffer_size: int = Field(
        default=512,
        description="Audio buffer size in samples"
    )
    bit_depth: int = Field(
        default=16,
        description="Audio bit depth"
    )
    output_directory: Path = Field(
        default=Path("output"),
        description="Directory for generated audio files"
    )
    temp_directory: Path = Field(
        default=Path("temp"),
        description="Directory for temporary audio files"
    )
    max_polyphony: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Maximum number of simultaneous voices"
    )
    
    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate audio sample rate."""
        valid_rates = [22050, 44100, 48000, 96000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of: {valid_rates}")
        return v
    
    @field_validator('bit_depth')
    @classmethod
    def validate_bit_depth(cls, v: int) -> int:
        """Validate audio bit depth."""
        valid_depths = [16, 24, 32]
        if v not in valid_depths:
            raise ValueError(f"Bit depth must be one of: {valid_depths}")
        return v
    
    @field_validator('output_directory', 'temp_directory')
    @classmethod
    def ensure_directory_exists(cls, v: str) -> str:
        """Ensure directories exist, create if necessary."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class HardcoreConfig(BaseModel):
    """Configuration specific to hardcore music generation."""
    
    default_bpm_range: tuple[int, int] = Field(
        default=(150, 250),
        description="Default BPM range for hardcore music"
    )
    kick_sub_frequencies: List[float] = Field(
        default=[41.2, 82.4, 123.6],
        description="Sub-bass frequencies for kick drums (Hz)"
    )
    detune_cents: List[int] = Field(
        default=[-19, -10, -5, 0, 5, 10, 19, 29],
        description="Detune values in cents for hardcore synthesis"
    )
    distortion_db: float = Field(
        default=15.0,
        ge=0.0,
        le=30.0,
        description="Default distortion level in dB"
    )
    compression_ratio: float = Field(
        default=8.0,
        ge=1.0,
        le=20.0,
        description="Default compression ratio"
    )
    preferred_keys: List[str] = Field(
        default=["Am", "Dm", "Em", "Gm", "Cm"],
        description="Preferred keys for hardcore music"
    )
    
    @field_validator('default_bpm_range')
    @classmethod
    def validate_bpm_range(cls, v: tuple) -> tuple:
        """Validate BPM range."""
        if len(v) != 2:
            raise ValueError("BPM range must be a tuple of (min, max)")
        if v[0] >= v[1]:
            raise ValueError("BPM minimum must be less than maximum")
        if v[0] < 60 or v[1] > 300:
            raise ValueError("BPM values must be between 60 and 300")
        return v


class DevelopmentConfig(BaseModel):
    """Configuration for development and debugging."""
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching for AI responses"
    )
    cache_directory: Path = Field(
        default=Path(".cache"),
        description="Directory for cache files"
    )
    profile_performance: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.upper()


class Settings(BaseModel):
    """
    Main configuration model combining all settings categories.
    
    This is the primary settings class that loads and validates
    all configuration from environment variables and .env files.
    """
    
    # Core settings
    project_name: str = Field(
        default="Hardcore Music Prototyper",
        description="Name of the application"
    )
    version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    environment: str = Field(
        default="development",
        description="Runtime environment"
    )
    
    # Component configurations
    ai: AIConfig = Field(default_factory=AIConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    hardcore: HardcoreConfig = Field(default_factory=HardcoreConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    
    # Additional settings
    data_directory: Path = Field(
        default=Path("data"),
        description="Base directory for all data files"
    )
    max_clip_length_bars: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Maximum length for MIDI clips in bars"
    )
    auto_save_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Auto-save interval in seconds"
    )
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_envs = ['development', 'testing', 'production']
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {', '.join(valid_envs)}")
        return v
    
    @field_validator('data_directory')
    @classmethod
    def ensure_data_directory(cls, v: str) -> str:
        """Ensure data directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        str_strip_whitespace=True
    )
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider."""
        if provider == "openai" and self.ai.openai_api_key:
            return self.ai.openai_api_key.get_secret_value()
        elif provider == "anthropic" and self.ai.anthropic_api_key:
            return self.ai.anthropic_api_key.get_secret_value()
        elif provider == "google" and self.ai.google_api_key:
            return self.ai.google_api_key.get_secret_value()
        return None
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for provider."""
        return self.get_api_key(provider) is not None
    
    def get_available_providers(self) -> List[str]:
        """Get list of AI providers with valid API keys."""
        providers = []
        for provider in ['openai', 'anthropic', 'google']:
            if self.has_api_key(provider):
                providers.append(provider)
        return providers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding secrets)."""
        data = self.model_dump()
        # Remove secret values for logging/debugging
        if 'ai' in data:
            for key in data['ai']:
                if 'api_key' in key and data['ai'][key] is not None:
                    data['ai'][key] = "***REDACTED***"
        return data