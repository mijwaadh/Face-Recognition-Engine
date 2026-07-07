import os
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings configuration loaded from environment variables and dotenv file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Settings
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Directory Paths
    data_dir: str = Field(default="data", alias="DATA_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    # Biometric Thresholds
    match_threshold: float = Field(default=0.75, alias="MATCH_THRESHOLD")
    liveness_threshold: float = Field(default=0.85, alias="LIVENESS_THRESHOLD")

    # Camera Settings
    camera_index: int = Field(default=0, alias="CAMERA_INDEX")
    capture_width: int = Field(default=640, alias="CAPTURE_WIDTH")
    capture_height: int = Field(default=480, alias="CAPTURE_HEIGHT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("match_threshold", "liveness_threshold")
    @classmethod
    def validate_thresholds(cls, val: float) -> float:
        """Ensures biometric thresholds fall within range [0.0, 1.0]."""
        if not (0.0 <= val <= 1.0):
            raise ValueError("Thresholds must be strictly between 0.0 and 1.0 inclusive.")
        return val

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    try:
        return Settings()
    except Exception as e:
        # Fallback print if logging system is not yet initialized
        print(f"CRITICAL: Failed to load application settings. Error: {e}")
        raise e
