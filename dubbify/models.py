from typing import Optional

from pydantic import BaseModel, Field, SecretStr
import os


class DubbifyConfig(BaseModel):
    voice: str = Field(default="alloy", description="Default ElevenLabs voice name or ID")
    language: Optional[str] = Field(default=None, description="ISO 639-1 target/output language for transcript and TTS (auto-detect input)")
    model: str = Field(default="eleven_v3", description="ElevenLabs TTS model name (v3)")

    api_key_elevenlabs: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("ELEVENLABS_API_KEY", "")),
        description="ElevenLabs API key (from environment)",
    )

    def require_elevenlabs_key(self) -> str:
        key = self.api_key_elevenlabs.get_secret_value()
        if not key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs features")
        return key

