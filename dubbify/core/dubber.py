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

        # Initialize ElevenLabs client if needed
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
        # ElevenLabs-only
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


