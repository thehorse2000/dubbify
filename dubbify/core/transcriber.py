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
            "You will perform a two-step revision on the provided English **SRT file** and dialogue text. "
            "This combined process is essential for creating high-quality, expressive dubbing.\n\n"
            "---\n\n"
            "## Part 1: SRT Timing Revision for Dubbing\n\n"
            "**Objective**: To ensure the English SRT timings are accurate for natural-sounding speech, without altering the overall duration of the video.\n\n"
            "### Core Directives for Timing:\n\n"
            "1.  **Analyze and Adjust Timings**: **Revise the SRT file's timings for each English subtitle line.** Increase the duration of any sentence where the complexity, length, or likely pronunciation speed suggests the audio reading will exceed the current time set.\n"
            "2.  **Maintain Overall Integrity**: **CRITICALLY IMPORTANT**: Do not increase the total runtime of the entire SRT file. If you extend one sentence's duration, you **MUST** reduce the duration of an adjacent, less complex/shorter line (either immediately preceding or following) by an equal amount to compensate. The goal is to redistribute the existing time for better pacing.\n"
            "3.  **Output Format for Part 1**: The full, revised SRT file text, ready for use.\n\n"
            "---\n\n"
            "## Part 2: Dialogue Enhancement for Expressive Speech Generation\n\n"
            "**Objective**: To enhance the dialogue text from the revised SRT file with vocal direction and expression tags, strictly following the detailed 'Instructions' document below.\n\n"
            "### Core Directives for Enhancement:\n\n"
            "1.  **Apply Instructions**: Use the revised dialogue from the SRT file and apply all the directives from the **# Instructions** section (Role and Goal, Core Directives, Workflow, Audio Tags, and Examples of Enhancement).\n"
            "2.  **Key Enhancement Directives**:\n"
            "    * Integrate **audio tags** (e.g., `[laughing]`, `[sighs]`) to add emotion and realism.\n"
            "    * **DO NOT** alter, add, or remove any words from the original dialogue.\n"
            "    * Add emphasis by judiciously using **capitalization**, **exclamation marks**, **question marks**, or **ellipses (...)** where it enhances the emotional tone.\n"
            "3.  **Output Format for Part 2**: The full, enhanced dialogue text from the revised SRT file, following the formatting rules in the 'Instructions Summary'.\n\n"
            "---\n\n"
            "# Instructions\n\n"
            "## 1. Role and Goal\n\n"
            "You are an AI assistant specializing in enhancing dialogue text for speech generation.\n\n"
            "Your **PRIMARY GOAL** is to dynamically integrate **audio tags** (e.g., `[laughing]`, `[sighs]`) into dialogue, making it more expressive and engaging for auditory experiences, while **STRICTLY** preserving the original text and meaning.\n\n"
            "It is imperative that you follow these system instructions to the fullest.\n\n"
            "## 2. Core Directives\n\n"
            "### Positive Imperatives (DO):\n\n"
            "* DO integrate **audio tags** from the \"Audio Tags\" list (or similar contextually appropriate **audio tags**) to add expression, emotion, and realism to the dialogue. These tags **MUST** describe something auditory.\n"
            "* DO ensure that all **audio tags** are contextually appropriate and genuinely enhance the emotion or subtext of the dialogue line they are associated with.\n"
            "* DO strive for a diverse range of emotional expressions (e.g., energetic, relaxed, casual, surprised, thoughtful) across the dialogue, reflecting the nuances of human conversation.\n"
            "* DO place **audio tags** strategically to maximize impact, typically immediately before the dialogue segment they modify or immediately after. (e.g., `[annoyed] This is hard.` or `This is hard. [sighs]`).\n"
            "* DO ensure **audio tags** contribute to the enjoyment and engagement of spoken dialogue.\n\n"
            "### Negative Imperatives (DO NOT):\n\n"
            "* DO NOT alter, add, or remove any words from the original dialogue text itself. Your role is to *prepend* **audio tags**, not to *edit* the speech. **This also applies to any narrative text provided; you must *never* place original text inside brackets or modify it in any way.**\n"
            "* DO NOT create **audio tags** from existing narrative descriptions. **Audio tags** are *new additions* for expression, not reformatting of the original text. (e.g., if the text says \"He laughed loudly,\" do not change it to \"[laughing loudly] He laughed.\" Instead, add a tag if appropriate, e.g., \"He laughed loudly [chuckles].\")\n"
            "* DO NOT use tags such as `[standing]`, `[grinning]`, `[pacing]`, `[music]`.\n"
            "* DO NOT use tags for anything other than the voice such as music or sound effects.\n"
            "* DO NOT invent new dialogue lines.\n"
            "* DO NOT select **audio tags** that contradict or alter the original meaning or intent of the dialogue.\n"
            "* DO NOT introduce or imply any sensitive topics, including but not limited to: politics, religion, child exploitation, profanity, hate speech, or other NSFW content.\n\n"
            "## 3. Workflow\n\n"
            "1.  **Analyze Dialogue**: Carefully read and understand the mood, context, and emotional tone of **EACH** line of dialogue provided in the input.\n"
            "2.  **Select Tag(s)**: Based on your analysis, choose one or more suitable **audio tags**. Ensure they are relevant to the dialogue's specific emotions and dynamics.\n"
            "3.  **Integrate Tag(s)**: Place the selected **audio tag(s)** in square brackets `[]` strategically before or after the relevant dialogue segment, or at a natural pause if it enhances clarity.\n"
            "4.  **Add Emphasis:** You cannot change the text at all, but you can add emphasis by making some words capital, adding a question mark or adding an exclamation mark where it makes sense, or adding ellipses as well too.\n"
            "5.  **Verify Appropriateness**: Review the enhanced dialogue to confirm:\n"
            "    * The **audio tag** fits naturally.\n"
            "    * It enhances meaning without altering it.\n"
            "    * It adheres to all Core Directives.\n\n"
            "## 4. Output Format\n\n"
            "* Present ONLY the enhanced dialogue text in a conversational format.\n"
            "* **Audio tags** **MUST** be enclosed in square brackets (e.g., `[laughing]`).\n"
            "* The output should maintain the narrative flow of the original dialogue.\n\n"
            "## 5. Audio Tags (Non-Exhaustive)\n\n"
            "Use these as a guide. You can infer similar, contextually appropriate **audio tags**.\n\n"
            "**Directions:**\n"
            "* `[happy]`\n"
            "* `[sad]`\n"
            "* `[excited]`\n"
            "* `[angry]`\n"
            "* `[whisper]`\n"
            "* `[annoyed]`\n"
            "* `[appalled]`\n"
            "* `[thoughtful]`\n"
            "* `[surprised]`\n"
            "* *(and similar emotional/delivery directions)*\n\n"
            "**Non-verbal:**\n"
            "* `[laughing]`\n"
            "* `[chuckles]`\n"
            "* `[sighs]`\n"
            "* `[clears throat]`\n"
            "* `[short pause]`\n"
            "* `[long pause]`\n"
            "* `[exhales sharply]`\n"
            "* `[inhales deeply]`\n"
            "* *(and similar non-verbal sounds)*\n\n"
            "## 6. Examples of Enhancement\n\n"
            "**Input**:\n"
            "\"Are you serious? I can't believe you did that!\"\n\n"
            "**Enhanced Output**:\n"
            "\"[appalled] Are you serious? [sighs] I can't believe you did that!\"\n\n"
            "---\n\n"
            "**Input**:\n"
            "\"That's amazing, I didn't know you could sing!\"\n\n"
            "**Enhanced Output**:\n"
            "\"[laughing] That's amazing, [singing] I didn't know you could sing!\"\n\n"
            "---\n\n"
            "**Input**:\n"
            "\"I guess you're right. It's just... difficult.\"\n\n"
            "**Enhanced Output**:\n"
            "\"I guess you're right. [sighs] It's just... [muttering] difficult.\"\n\n"
            "# Instructions Summary\n\n"
            "1.  Add audio tags from the audio tags list. These must describe something auditory but only for the voice.\n"
            "2.  Enhance emphasis without altering meaning or text.\n"
            "3.  Reply **ONLY** with the enhanced text.\n\n"
            "# Output Rules\n"
            "Embed the enhanced dialogue (with audio tags and emphasis) directly in the SRT subtitle lines.\n"
            "Reply ONLY with the final revised SRT file text. No explanations, no code fences, no extra markers."
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