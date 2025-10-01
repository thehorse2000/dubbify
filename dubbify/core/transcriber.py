from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import srt as srt_lib
from datetime import timedelta

from . import media_utils
from pydub import AudioSegment
from ..models import DubbifyConfig


class Transcriber:
    def __init__(self, config: DubbifyConfig):
        self.config = config

    def transcribe(self, media_path: str) -> str:
        input_path = Path(media_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input media not found: {media_path}")

        media_for_transcription = str(input_path)
        if input_path.suffix.lower() in {".mp4", ".mov", ".mkv"}:
            media_for_transcription = media_utils.extract_audio(str(input_path))

        srt_text = self._transcribe_elevenlabs(media_for_transcription)

        return srt_text

    def _transcribe_elevenlabs(self, audio_path: str) -> str:
        # Ensure API key present before any audio processing (keeps tests deterministic)
        api_key = self.config.require_elevenlabs_key()
        target_lang = (self.config.language or "").strip().lower() or None

        # Attempt to use ElevenLabs SDK if available; otherwise fall back to HTTP
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore
            client = ElevenLabs(api_key=api_key)
            # Prefer SDK speech-to-text if exposed
            if hasattr(client, "speech_to_text") and hasattr(client.speech_to_text, "convert"):
                with open(audio_path, "rb") as f:
                    result = client.speech_to_text.convert(file=f)  # type: ignore[arg-type]
                data = result if isinstance(result, dict) else getattr(result, "to_dict", lambda: {} )()
            else:
                raise ImportError
        except Exception:
            # Fallback to HTTP API
            import requests
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            headers = {"xi-api-key": api_key}
            files = {"file": (Path(audio_path).name, open(audio_path, "rb"), "application/octet-stream")}
            data = {}
            if target_lang:
                data["language"] = target_lang
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)
            resp.raise_for_status()
            data = resp.json()

        # Try to build subtitles from common STT response shapes
        segments = data.get("segments") or data.get("words") or []
        if not segments and data.get("text"):
            return srt_lib.compose([
                srt_lib.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=0), content=str(data.get("text") or ""))
            ])

        subtitles: List[srt_lib.Subtitle] = []
        for idx, seg in enumerate(segments, start=1):
            # Support multiple key shapes
            start_seconds = float(seg.get("start", seg.get("start_time", seg.get("timestamp", 0.0))))
            end_seconds = float(seg.get("end", seg.get("end_time", start_seconds)))
            content = (seg.get("text") or seg.get("word") or "").strip()
            if not content:
                continue
            subtitles.append(
                srt_lib.Subtitle(
                    index=idx,
                    start=timedelta(seconds=max(0.0, start_seconds)),
                    end=timedelta(seconds=max(start_seconds, end_seconds)),
                    content=content,
                )
            )

        return srt_lib.compose(subtitles)

    # OpenAI-based transcription and translation removed; ElevenLabs only

    # Chunked transcription helpers related to OpenAI removed

    def _split_audio_into_chunks(self, audio_path: str, target_max_bytes: int) -> List[Tuple[Path, float]]:
        """Split an audio file into chunks whose sizes are approximately under target_max_bytes.

        Returns a list of (chunk_path, offset_seconds).
        """
        src_path = Path(audio_path)
        seg = AudioSegment.from_file(str(src_path))
        duration_ms = len(seg)
        file_size = src_path.stat().st_size
        if duration_ms <= 0 or file_size <= 0:
            # Degenerate case; return as a single chunk
            return [(src_path, 0.0)]

        bytes_per_ms = file_size / duration_ms
        max_ms = int(target_max_bytes / max(bytes_per_ms, 1e-6))
        # Cap chunk window to avoid extremely long chunks; but never exceed size-based max
        max_window_ms_cap = 180 * 1000
        chunk_window_ms = min(max_ms, max_window_ms_cap)
        # Ensure at least 1s to make progress
        if chunk_window_ms < 1000:
            chunk_window_ms = 1000

        chunks: List[Tuple[Path, float]] = []
        chunk_dir = Path("tmp_dubbify_transcribe")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        ext = src_path.suffix.lstrip(".").lower() or "wav"
        start_ms = 0
        chunk_index = 0
        while start_ms < duration_ms:
            end_ms = min(start_ms + chunk_window_ms, duration_ms)
            piece = seg[start_ms:end_ms]
            offset_seconds = start_ms / 1000.0
            out_path = chunk_dir / f"chunk_{chunk_index:05d}.{ext}"
            piece.export(str(out_path), format=ext)
            # If exported chunk still exceeds target size (due to codec), split further by halving window
            if out_path.stat().st_size > target_max_bytes and (end_ms - start_ms) > 1000:
                # Retry with smaller window next iteration
                chunk_window_ms = max(1000, (end_ms - start_ms) // 2)
                continue
            chunks.append((out_path, offset_seconds))
            start_ms = end_ms
            chunk_index += 1

        return chunks

    # Translation via OpenAI removed; ElevenLabs-only pipeline

