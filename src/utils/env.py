"""
Environment configuration loading utilities.

Provides functions to load and validate configuration from environment
variables and .env files using Pydantic models.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv
from pydantic import ValidationError

from ..models.config import Settings


def load_settings(env_file: Optional[str] = None) -> Settings:
    """
    Load application settings from environment variables and .env file.
    
    Args:
        env_file: Optional path to .env file. If None, looks for .env in current directory.
        
    Returns:
        Settings: Validated settings object with all configuration.
        
    Raises:
        ValidationError: If configuration validation fails.
        FileNotFoundError: If specified .env file doesn't exist.
    """
    # Determine .env file path
    if env_file is None:
        env_file = ".env"
    
    env_path = Path(env_file)
    
    # Load .env file if it exists
    if env_path.exists():
        load_dotenv(env_path)
    elif env_file != ".env":
        # Only raise error if user explicitly specified a file
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    
    try:
        # Create settings instance (will automatically load from environment)
        settings = Settings()
        return settings
    except ValidationError as e:
        # Enhance error message with helpful context
        error_msg = f"Configuration validation failed:\n{e}"
        if not env_path.exists():
            error_msg += f"\n\nNote: No .env file found at {env_path.absolute()}"
            error_msg += "\nConsider creating a .env file with your configuration."
        raise ValidationError(error_msg) from e


def create_example_env_file(path: str = ".env.example") -> None:
    """
    Create an example .env file with all available configuration options.
    
    Args:
        path: Path where to create the example file.
    """
    example_content = """# Hardcore Music Prototyper - Environment Configuration
# Copy this file to .env and fill in your values

# AI/LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Default AI provider (openai, anthropic, or google)
AI_PROVIDER=openai

# Audio Settings
AUDIO_SAMPLE_RATE=44100
AUDIO_BUFFER_SIZE=512

# Development Settings
DEBUG=false
LOG_LEVEL=INFO

# Application Settings
ENVIRONMENT=development
PROJECT_NAME=Hardcore Music Prototyper
"""
    
    with open(path, 'w') as f:
        f.write(example_content)


def validate_environment() -> Dict[str, Any]:
    """
    Validate the current environment configuration.
    
    Returns:
        dict: Validation results with status and any issues found.
    """
    results: Dict[str, Any] = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'available_providers': [],
    }
    
    try:
        settings = load_settings()
        
        # Check for available AI providers
        available_providers = settings.get_available_providers()
        results['available_providers'] = available_providers
        
        if not available_providers:
            results['warnings'].append(
                "No AI provider API keys found. "
                "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY."
            )
        
        # Check if default provider is available
        if settings.ai.default_provider not in available_providers:
            if available_providers:
                results['warnings'].append(
                    f"Default AI provider '{settings.ai.default_provider}' "
                    f"not available. Using {available_providers[0]} instead."
                )
            else:
                results['errors'].append(
                    f"Default AI provider '{settings.ai.default_provider}' "
                    "not available and no alternatives found."
                )
                results['valid'] = False
        
        # Check directories
        try:
            settings.audio.output_directory.mkdir(parents=True, exist_ok=True)
            settings.audio.temp_directory.mkdir(parents=True, exist_ok=True)
            settings.data_directory.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            results['errors'].append(f"Cannot create required directories: {e}")
            results['valid'] = False
        
    except ValidationError as e:
        results['valid'] = False
        results['errors'].append(f"Configuration validation failed: {e}")
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Unexpected error loading configuration: {e}")
    
    return results


def get_config_summary(settings: Optional[Settings] = None) -> str:
    """
    Get a human-readable summary of the current configuration.
    
    Args:
        settings: Optional settings object. If None, loads from environment.
        
    Returns:
        str: Formatted configuration summary.
    """
    if settings is None:
        try:
            settings = load_settings()
        except Exception as e:
            return f"Error loading configuration: {e}"
    
    available_providers = settings.get_available_providers()
    
    summary = f"""Hardcore Music Prototyper Configuration
{'=' * 45}

Project: {settings.project_name} v{settings.version}
Environment: {settings.environment}

AI Configuration:
  Default Provider: {settings.ai.default_provider}
  Available Providers: {', '.join(available_providers) if available_providers else 'None'}
  Max Tokens: {settings.ai.max_tokens}
  Temperature: {settings.ai.temperature}

Audio Configuration:
  Sample Rate: {settings.audio.sample_rate} Hz
  Buffer Size: {settings.audio.buffer_size} samples
  Bit Depth: {settings.audio.bit_depth} bits
  Output Directory: {settings.audio.output_directory}

Hardcore Music Settings:
  BPM Range: {settings.hardcore.default_bpm_range[0]}-{settings.hardcore.default_bpm_range[1]}
  Preferred Keys: {', '.join(settings.hardcore.preferred_keys)}
  Distortion Level: {settings.hardcore.distortion_db} dB

Development:
  Debug Mode: {settings.development.debug}
  Log Level: {settings.development.log_level}
  Cache Enabled: {settings.development.cache_enabled}
"""
    
    return summary


if __name__ == "__main__":
    """Command-line interface for configuration utilities."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            results = validate_environment()
            print("Environment Validation Results")
            print("=" * 30)
            print(f"Valid: {results['valid']}")
            
            if results['warnings']:
                print("\nWarnings:")
                for warning in results['warnings']:
                    print(f"  - {warning}")
            
            if results['errors']:
                print("\nErrors:")
                for error in results['errors']:
                    print(f"  - {error}")
            
            if results['available_providers']:
                print(f"\nAvailable AI Providers: {', '.join(results['available_providers'])}")
            
            sys.exit(0 if results['valid'] else 1)
        
        elif command == "summary":
            try:
                print(get_config_summary())
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        
        elif command == "create-example":
            output_file = sys.argv[2] if len(sys.argv) > 2 else ".env.example"
            create_example_env_file(output_file)
            print(f"Created example environment file: {output_file}")
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    else:
        print("Usage: python -m src.utils.env [validate|summary|create-example]")
        sys.exit(1)