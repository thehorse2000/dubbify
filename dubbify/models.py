from typing import Optional

from pydantic import BaseModel, Field, SecretStr
import os


class DubbifyConfig(BaseModel):
    voice: str = Field(default="alloy", description="Default ElevenLabs voice name or ID")
    voice_id: Optional[str] = Field(default=None, description="Explicit ElevenLabs voice_id for Convert endpoint")
    language: Optional[str] = Field(default=None, description="ISO 639-1 target/output language for transcript and TTS (auto-detect input)")
    model: str = Field(default="eleven_multilingual_v2", description="ElevenLabs TTS model name (v3)")
    use_convert_endpoint: bool = Field(default=True, description="Use ElevenLabs text_to_speech.convert (no timestamps)")
    openai_text_model: str = Field(
        default="gpt-5-mini",
        description="OpenAI text model used for subtitle refinement and translation",
    )

    api_key_elevenlabs: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("ELEVENLABS_API_KEY", "")),
        description="ElevenLabs API key (from environment)",
    )

    api_key_openai: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("OPENAI_API_KEY", "")),
        description="OpenAI API key (from environment)",
    )

    def require_elevenlabs_key(self) -> str:
        key = self.api_key_elevenlabs.get_secret_value()
        if not key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs features")
        return key

    def require_openai_key_for_text(self) -> str:
        key = self.api_key_openai.get_secret_value()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI transcription features")
        return key

