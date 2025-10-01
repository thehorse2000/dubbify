from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr
import os


class DubbifyConfig(BaseModel):
    voice: str = Field(default="alloy", description="Default ElevenLabs voice name or ID")
    language: Optional[str] = Field(default=None, description="ISO 639-1 target/output language for transcript and TTS (auto-detect input)")
    transcriber: Literal["openai"] = Field(default="openai", description="Transcription provider")
    tts_provider: Literal["elevenlabs", "openai"] = Field(default="elevenlabs", description="Text-to-Speech provider")
    model: str = Field(default="eleven_multilingual_v2", description="ElevenLabs TTS model name")
    openai_tts_model: str = Field(default="tts-1-hd", description="OpenAI TTS model name")
    openai_text_model: str = Field(default="gpt-4o-mini", description="OpenAI text model for translation")

    api_key_openai: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("OPENAI_API_KEY", "")),
        description="OpenAI API key (from environment)",
    )
    api_key_elevenlabs: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("ELEVENLABS_API_KEY", "")),
        description="ElevenLabs API key (from environment)",
    )

    def require_elevenlabs_key(self) -> str:
        key = self.api_key_elevenlabs.get_secret_value()
        if not key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs features")
        return key

    def require_openai_key(self) -> str:
        key = self.api_key_openai.get_secret_value()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI features")
        return key

    def require_openai_key_for_tts(self) -> str:
        key = self.api_key_openai.get_secret_value()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI TTS features")
        return key

    def require_openai_key_for_text(self) -> str:
        key = self.api_key_openai.get_secret_value()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI translation features")
        return key

