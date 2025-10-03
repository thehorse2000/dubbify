from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List, Tuple
import io

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

    def _ensure_elevenlabs_client(self):
        # Ensure client-based API is available (used for convert endpoint)
        if self._elevenlabs_client is not None:
            return
        api_key = self.config.require_elevenlabs_key()
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

        # Initialize ElevenLabs API(s) if needed
        # We may use either top-level generate/client.generate or client.text_to_speech.convert
        self._ensure_elevenlabs()

        segment_paths: List[Path] = []
        speech_durations_ms: List[int] = []
        for idx, sub in enumerate(subtitles):
            text = (sub.content or "").strip()
            if not text:
                # Create a short silent placeholder for blank lines
                silent_path = temp_dir / f"segment_{idx:05d}.mp3"
                AudioSegment.silent(duration=100).export(str(silent_path), format="mp3")
                segment_paths.append(silent_path)
                speech_durations_ms.append(100)
                continue

            audio_bytes: bytes
            # Prefer convert endpoint if configured; use voice_id if provided, else fall back to voice as id
            if getattr(self.config, "use_convert_endpoint", False):
                try:
                    audio_bytes = self._synthesize_text_convert(text)
                except Exception:
                    # Fallback to legacy generation on failure
                    audio_bytes = self._synthesize_text(text)
            else:
                audio_bytes = self._synthesize_text(text)

            seg_path = temp_dir / f"segment_{idx:05d}.mp3"
            with open(seg_path, "wb") as f:
                f.write(audio_bytes)
            segment_paths.append(seg_path)
            # Measure duration
            try:
                clip = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                speech_durations_ms.append(len(clip))
            except Exception:
                # Fallback by reading from file if in-memory decode fails
                clip = AudioSegment.from_file(seg_path)
                speech_durations_ms.append(len(clip))

        original_total_end_ms = int(subtitles[-1].end.total_seconds() * 1000)
        final_track = AudioSegment.silent(duration=original_total_end_ms)

        # Compute adjusted timings using forward push + backward reconciliation
        original_starts_ms: List[int] = [int(s.start.total_seconds() * 1000) for s in subtitles]
        original_ends_ms: List[int] = [int(s.end.total_seconds() * 1000) for s in subtitles]

        new_times: List[Tuple[int, int]] = []
        offset_ms = 0
        for i, sub in enumerate(subtitles):
            speech_ms = speech_durations_ms[i]
            new_start = original_starts_ms[i] + offset_ms
            new_end = new_start + speech_ms
            # If speech overflows original slot, push offset forward
            overflow = new_end - (original_ends_ms[i] + offset_ms)
            if overflow > 0:
                offset_ms += overflow
            new_times.append((new_start, new_end))

        # Backward reconciliation to preserve final end time, if needed
        scheduled_end = new_times[-1][1]
        remaining_pull = scheduled_end - original_ends_ms[-1]
        if remaining_pull > 0:
            i = len(new_times) - 1
            while remaining_pull > 0 and i > 0:
                prev_end = new_times[i - 1][1]
                cur_start = new_times[i][0]
                available_gap = max(0, cur_start - prev_end)
                take = min(available_gap, remaining_pull)
                if take > 0:
                    s, e = new_times[i]
                    new_times[i] = (s - take, e - take)
                    remaining_pull -= take
                i -= 1
            if remaining_pull > 0:
                # Not enough slack to reconcile
                raise RuntimeError(
                    "Unable to reconcile timings: total synthesized speech exceeds available inter-cue gaps."
                )

        for idx, _sub in enumerate(subtitles):
            start_ms = new_times[idx][0]
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
                # Try modern convert endpoint using voice_id or voice fallback
                try:
                    return self._synthesize_text_convert(text)
                except Exception as exc:
                    raise RuntimeError(
                        "The installed 'elevenlabs' SDK does not support client.generate(), and convert failed. "
                        "Provide a valid voice_id or install a compatible SDK."
                    ) from exc
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        # Some versions stream chunks
        chunks: List[bytes] = []
        for chunk in audio:  # type: ignore[assignment]
            chunks.append(chunk if isinstance(chunk, bytes) else bytes(chunk))
        return b"".join(chunks)

    def _synthesize_text_convert(self, text: str) -> bytes:
        # Preferred modern API: client.text_to_speech.convert (no timestamps)
        # Use explicit voice_id if provided, otherwise treat voice as id (common usage)
        voice_id = getattr(self.config, "voice_id", None) or getattr(self.config, "voice", None)
        if not voice_id:
            raise RuntimeError("voice_id or voice is required for the convert endpoint")
        model = self.config.model
        self._ensure_elevenlabs_client()
        assert self._elevenlabs_client is not None
        client = self._elevenlabs_client
        # Access nested API if present
        tts_api = getattr(client, "text_to_speech", None)
        if tts_api is None or not hasattr(tts_api, "convert"):
            # Older SDKs may expose convert directly, try that
            if hasattr(client, "convert"):
                audio = client.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=model,
                    output_format="mp3_44100_128",
                )
            else:
                # Fallback to legacy generate path
                return self._synthesize_text(text)
        else:
            audio = tts_api.convert(
                voice_id=voice_id,
                text=text,
                model_id=model,
                output_format="mp3_44100_128",
            )
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        chunks: List[bytes] = []
        for chunk in audio:  # type: ignore[assignment]
            chunks.append(chunk if isinstance(chunk, bytes) else bytes(chunk))
        return b"".join(chunks)


