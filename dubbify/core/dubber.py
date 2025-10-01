from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List

import srt as srt_lib
from pydub import AudioSegment

from ..models import DubbifyConfig


class Dubber:
    def __init__(self, config: DubbifyConfig):
        self.config = config
        # Delay import for optional dependency until used
        self._elevenlabs_client = None
        # Prefer using top-level `elevenlabs.generate` if available for broader compatibility
        self._elevenlabs_generate = None

    def _ensure_elevenlabs(self):
        # Only initialize if ElevenLabs is the active TTS provider
        if self.config.tts_provider != "elevenlabs":
            return
        # No-op if already initialized
        if self._elevenlabs_generate is not None or self._elevenlabs_client is not None:
            return
        api_key = self.config.require_elevenlabs_key()
        # Try the top-level API first (older SDKs expose `generate` at module level)
        try:
            from elevenlabs import set_api_key as el_set_api_key  # type: ignore
            from elevenlabs import generate as el_generate  # type: ignore

            el_set_api_key(api_key)
            self._elevenlabs_generate = el_generate
            return
        except Exception:
            pass

        # Fallback to the client-based API (newer SDKs)
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore

            self._elevenlabs_client = ElevenLabs(api_key=api_key)
        except Exception as exc:
            raise RuntimeError("Failed to initialize ElevenLabs client. Please verify the 'elevenlabs' package installation.") from exc

    def generate_dub_track(self, srt_content: str) -> AudioSegment:
        subtitles: List[srt_lib.Subtitle] = list(srt_lib.parse(srt_content))
        if not subtitles:
            return AudioSegment.silent(duration=0)

        temp_dir = Path("tmp_dubbify_segments")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize provider-specific clients if needed
        if self.config.tts_provider == "elevenlabs":
            self._ensure_elevenlabs()

        segment_paths: List[Path] = []
        for idx, sub in enumerate(subtitles):
            text = (sub.content or "").strip()
            if not text:
                # Create a short silent placeholder for blank lines
                silent_path = temp_dir / f"segment_{idx:05d}.mp3"
                AudioSegment.silent(duration=100).export(str(silent_path), format="mp3")
                segment_paths.append(silent_path)
                continue

            audio_bytes = self._synthesize_text(text)
            seg_path = temp_dir / f"segment_{idx:05d}.mp3"
            with open(seg_path, "wb") as f:
                f.write(audio_bytes)
            segment_paths.append(seg_path)

        total_duration_ms = int(subtitles[-1].end.total_seconds() * 1000)
        final_track = AudioSegment.silent(duration=total_duration_ms)

        for idx, sub in enumerate(subtitles):
            start_ms = int(sub.start.total_seconds() * 1000)
            clip = AudioSegment.from_file(segment_paths[idx])
            final_track = final_track.overlay(clip, position=start_ms)

        # Cleanup best-effort
        for p in segment_paths:
            try:
                p.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass

        return final_track

    def _synthesize_text(self, text: str) -> bytes:
        if self.config.tts_provider == "openai":
            return self._synthesize_text_openai(text)
        # Default to ElevenLabs
        voice = self.config.voice
        model = self.config.model
        # Prefer top-level `generate` if available
        if self._elevenlabs_generate is not None:
            audio = self._elevenlabs_generate(
                text=text,
                voice=voice,
                model=model,
                output_format="mp3_44100_128",
            )
        else:
            assert self._elevenlabs_client is not None
            # Older code path used `client.generate`; if not present, raise a clear error
            if hasattr(self._elevenlabs_client, "generate"):
                audio = self._elevenlabs_client.generate(
                    text=text,
                    voice=voice,
                    model=model,
                    output_format="mp3_44100_128",
                )
            else:
                # Newer SDKs often require voice_id and use client.text_to_speech.convert
                # This codebase currently accepts a voice name; mapping to voice_id is not implemented.
                raise RuntimeError(
                    "The installed 'elevenlabs' SDK does not support client.generate(). "
                    "Either install a version that exposes the top-level 'generate' API, "
                    "or update configuration to provide a valid voice_id and adapt to the "
                    "client.text_to_speech.convert API."
                )
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        # Some versions stream chunks
        chunks: List[bytes] = []
        for chunk in audio:  # type: ignore[assignment]
            chunks.append(chunk if isinstance(chunk, bytes) else bytes(chunk))
        return b"".join(chunks)

    def _synthesize_text_openai(self, text: str) -> bytes:
        # Ensure API key is present
        api_key = self.config.require_openai_key_for_tts()
        # Lazy import to avoid hard dependency unless used
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is not installed. Please install 'openai' package.") from exc

        client = OpenAI(api_key=api_key)
        model = self.config.openai_tts_model
        voice = (self.config.voice or "alloy").strip().lower()
        voice = self._validate_openai_voice(voice)
        # Attempt non-streaming API first; if it fails, try streaming
        try:
            # Newer SDKs may support direct bytes via .audio.speech.create
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
            )
            # Try common access patterns for binary content
            if hasattr(response, "content") and isinstance(response.content, (bytes, bytearray)):
                return bytes(response.content)
            # Some SDKs return a generator-like object with iter_bytes()
            if hasattr(response, "iter_bytes"):
                return b"".join(response.iter_bytes())  # type: ignore[attr-defined]
        except Exception:
            # Fallback to streaming response API pattern
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=text,
                ) as stream:
                    # Collect chunks into memory
                    chunks: List[bytes] = []
                    for chunk in stream.iter_bytes():  # type: ignore[attr-defined]
                        chunks.append(chunk if isinstance(chunk, bytes) else bytes(chunk))
                    return b"".join(chunks)
            except Exception as exc:
                raise RuntimeError("Failed to synthesize speech via OpenAI TTS. Check model name and SDK version.") from exc

        # If none of the above paths returned, raise a helpful error
        raise RuntimeError("Unsupported OpenAI SDK response format for TTS.")

    def _validate_openai_voice(self, desired_voice: str) -> str:
        """Validate the provided voice against the OpenAI-supported list.

        Returns the voice if valid; otherwise raises a ValueError.
        """
        allowed = {
            "nova",
            "shimmer",
            "echo",
            "onyx",
            "fable",
            "alloy",
            "ash",
            "sage",
            "coral",
        }
        if desired_voice in allowed:
            return desired_voice
        raise ValueError(
            "Invalid OpenAI TTS voice. Use one of: nova, shimmer, echo, onyx, fable, alloy, ash, sage, coral"
        )

