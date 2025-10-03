from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import logging
import srt as srt_lib
from datetime import timedelta

from . import media_utils
from pydub import AudioSegment
from ..models import DubbifyConfig

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, config: DubbifyConfig):
        self.config = config
        self.last_enhancement_applied: bool = False

    def transcribe(self, media_path: str) -> str:
        input_path = Path(media_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input media not found: {media_path}")

        media_for_transcription = str(input_path)
        if input_path.suffix.lower() in {".mp4", ".mov", ".mkv"}:
            media_for_transcription = media_utils.extract_audio(str(input_path))

        srt_text = self._transcribe_openai(media_for_transcription)
        self.last_enhancement_applied = False
        try:
            srt_text = self._enhance_subtitles_with_openai(original_srt=srt_text)
        except Exception as exc:
            logger.exception("Subtitle enhancement failed; using raw transcription.")

        return srt_text

    def _transcribe_openai(self, audio_path: str) -> str:
        # Use OpenAI cloud transcription/translation only; no local Whisper fallback
        # Ensure API key present before any audio processing (keeps tests deterministic)
        _ = self.config.require_openai_key_for_text()
        target_lang = (self.config.language or "").strip().lower() or None

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
                    target_lang=target_lang,
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

    def _enhance_subtitles_with_openai(self, original_srt: str) -> str:
        """Call OpenAI to revise SRT timings for dubbing; return ONLY revised SRT text."""
        from openai import OpenAI  # type: ignore

        system_prompt = (
            "## 1. Role and Goal\n\n"
            "You are an AI assistant specializing in the technical revision of SubRip Subtitle (`.srt`) files.\n\n"
            "Your **PRIMARY GOAL** is to **accurately revise the English SRT file timings** to ensure a natural, comfortable reading pace for dubbing, while **STRICTLY preserving the original total duration** of the video content.\n\n"
            "It is imperative that you follow these instructions to the fullest.\n\n"
            "---\n\n"
            "## 2. Core Directives\n\n"
            "Follow these directives meticulously to ensure high-quality output.\n\n"
            "### Positive Imperatives (DO):\n\n"
            "* DO **Analyze and Adjust Timings**: Examine the English text for each subtitle line. You must **increase the duration** for any sentence where the complexity, length, or likely pronunciation speed suggests the audio reading will exceed the current time setting.\n"
            "* DO **Maintain Overall Integrity**: **CRITICALLY IMPORTANT**: Do not increase the total runtime of the entire SRT file. If you extend one sentence's duration, you **MUST** reduce the duration of an adjacent, less complex/shorter line (either immediately preceding or following) by an equal amount to compensate. The goal is to **redistribute the existing time for better pacing.**\n\n"
            "### Negative Imperatives (DO NOT):\n\n"
            "* DO NOT alter the actual **dialogue text**. Only the timestamps (start and end times) should be changed.\n"
            "* DO NOT add, remove, or change the order of any subtitle blocks.\n"
            "* DO NOT introduce any expressive elements, such as audio tags (e.g., `[laughing]`), capitalization for emphasis, or ellipses. **Your focus is strictly technical timing.**\n"
            "* DO NOT increase the total overall duration of the final subtitle file.\n\n"
            "---\n\n"
            "## 3. Input and Output Format\n\n"
            "1.  **Input Data**: The complete original SRT file content.\n"
            "2.  **Output Format**: Reply **ONLY** with the full, revised SRT file text, ready for use.\n\n"
            "---\n\n"
        )

        client = OpenAI(api_key=self.config.require_openai_key_for_text())
        model = self.config.openai_text_model

        user_prompt = (
            "Here is the original SRT. Return only the revised SRT with adjusted timings:\n\n"
            f"{original_srt}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return original_srt

        # Validate it's parseable as SRT; fallback if not
        try:
            list(srt_lib.parse(content))
            if content != original_srt:
                self.last_enhancement_applied = True
            return content
        except Exception:
            return original_srt

    def _transcribe_openai_cloud(self, audio_path: str, target_lang: Optional[str]) -> str:
        from typing import Any, Dict
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self.config.require_openai_key_for_text())
        model_name = "whisper-1"

        def to_dict(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()  # type: ignore[no-any-return]
            if isinstance(obj, dict):
                return obj
            # Fallback: map attributes
            return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}

        # Use translation endpoint when target is English to improve quality
        if target_lang in {"en", "eng", "english"}:
            with open(audio_path, "rb") as f:
                result = client.audio.translations.create(  # type: ignore[attr-defined]
                    model=model_name,
                    file=f,
                    response_format="verbose_json",
                )
        else:
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=model_name,
                    file=f,
                    response_format="verbose_json",
                )

        data = to_dict(result)
        segments = data.get("segments") or data.get("text_segments") or []
        if not segments and data.get("text"):
            # No timestamps available; return a single-block SRT
            return srt_lib.compose([
                srt_lib.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=0), content=str(data.get("text") or ""))
            ])

        subtitles = []
        for idx, seg in enumerate(segments, start=1):
            start_seconds = float(seg.get("start", 0.0))
            end_seconds = float(seg.get("end", start_seconds))
            content = (seg.get("text") or "").strip()
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

        # If a non-English target was requested, translate segment-wise using OpenAI text model
        if target_lang and target_lang not in {"en", "eng", "english"}:
            subtitles = self._translate_subtitles_openai(subtitles, target_lang)

        return srt_lib.compose(subtitles)

    def _transcribe_openai_cloud_chunk(self, audio_path: str, target_lang: Optional[str], offset_seconds: float) -> List[srt_lib.Subtitle]:
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

        if target_lang in {"en", "eng", "english"}:
            with open(audio_path, "rb") as f:
                result = client.audio.translations.create(  # type: ignore[attr-defined]
                    model=model_name,
                    file=f,
                    response_format="verbose_json",
                )
        else:
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

        if target_lang and target_lang not in {"en", "eng", "english"}:
            subtitles = self._translate_subtitles_openai(subtitles, target_lang)

        return subtitles

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

    def _translate_subtitles_openai(self, subtitles: list[srt_lib.Subtitle], target_lang: str) -> list[srt_lib.Subtitle]:
        # Batch translate while preserving timing and indices; simple per-line translation for now
        api_key = self.config.require_openai_key_for_text()
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is not installed. Please install 'openai' package for translation.") from exc

        client = OpenAI(api_key=api_key)
        model = self.config.openai_text_model

        def translate_text(text: str) -> str:
            prompt = (
                "Translate the following subtitle text to {lang}. Keep it concise and natural. "
                "Return only the translated text without quotes.\n\n{text}"
            ).format(lang=target_lang, text=text)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a high-quality translator."},
                        {"role": "user", "content": prompt},
                    ]
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                raise RuntimeError("Failed to translate subtitles with OpenAI.") from exc

        translated: list[srt_lib.Subtitle] = []
        for sub in subtitles:
            translated.append(
                srt_lib.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=translate_text(sub.content),
                )
            )
        return translated

    # ElevenLabs transcriber removed