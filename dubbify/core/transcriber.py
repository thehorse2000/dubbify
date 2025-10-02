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

        srt_text = self._transcribe_openai(media_for_transcription)

        return srt_text

    def _transcribe_openai(self, audio_path: str) -> str:
        # Ensure API key present before any audio processing (keeps tests deterministic)
        _ = self.config.require_openai_key_for_text()
        # Split audio into chunks under provider size limit and merge into one SRT
        subtitles: List[srt_lib.Subtitle] = []
        chunk_dir = Path("tmp_dubbify_transcribe")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        try:
            chunks = self._split_audio_into_chunks(audio_path, target_max_bytes=24 * 1024 * 1024)
            running_index = 1
            for chunk_path, offset_seconds in chunks:
                chunk_subs = self._transcribe_openai_cloud_chunk(
                    audio_path=str(chunk_path),
                    offset_seconds=offset_seconds,
                )
                for sub in chunk_subs:
                    sub.index = running_index
                    running_index += 1
                subtitles.extend(chunk_subs)
        finally:
            # Cleanup chunk files best-effort
            try:
                for p in chunk_dir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                chunk_dir.rmdir()
            except Exception:
                pass

        return srt_lib.compose(subtitles)

    def _transcribe_openai_cloud_chunk(self, audio_path: str, offset_seconds: float) -> List[srt_lib.Subtitle]:
        from typing import Any, Dict
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self.config.require_openai_key_for_text())
        model_name = "whisper-1"

        def to_dict(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()  # type: ignore[no-any-return]
            if isinstance(obj, dict):
                return obj
            return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}

        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=model_name,
                file=f,
                response_format="verbose_json",
            )

        data = to_dict(result)
        segments = data.get("segments") or data.get("text_segments") or []
        if not segments and data.get("text"):
            return [
                srt_lib.Subtitle(
                    index=1,
                    start=timedelta(seconds=offset_seconds),
                    end=timedelta(seconds=offset_seconds),
                    content=str(data.get("text") or ""),
                )
            ]

        subtitles: List[srt_lib.Subtitle] = []
        for idx, seg in enumerate(segments, start=1):
            start_seconds = float(seg.get("start", 0.0))
            end_seconds = float(seg.get("end", start_seconds))
            content = (seg.get("text") or "").strip()
            if not content:
                continue
            subtitles.append(
                srt_lib.Subtitle(
                    index=idx,
                    start=timedelta(seconds=max(0.0, start_seconds + offset_seconds)),
                    end=timedelta(seconds=max(start_seconds, end_seconds) + offset_seconds),
                    content=content,
                )
            )

        return subtitles

    # Chunk splitting retained for provider upload limits

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

    # Translation not implemented; output SRT is source language

