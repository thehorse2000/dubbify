from .models import DubbifyConfig  # noqa: F401
from .core.transcriber import Transcriber  # noqa: F401
from .core.dubber import Dubber  # noqa: F401
from . import core  # noqa: F401

from pathlib import Path
from typing import Optional

from pydub import AudioSegment

from .core import media_utils


class Dubbify:
    def __init__(self, config: DubbifyConfig):
        self.config = config
        self.transcriber = Transcriber(config=config)
        self.dubber = Dubber(config=config)

    def transcribe(self, input_path: str, output_srt_path: str) -> None:
        srt_content = self.transcriber.transcribe(media_path=input_path)
        Path(output_srt_path).write_text(srt_content, encoding="utf-8")

    def dub(self, srt_path: str, output_audio_path: str) -> None:
        srt_content = Path(srt_path).read_text(encoding="utf-8")
        final_track: AudioSegment = self.dubber.generate_dub_track(srt_content=srt_content)
        final_track.export(output_audio_path, format=Path(output_audio_path).suffix.lstrip("."))

    def run(self, input_path: str, output_path: str) -> None:
        srt_content = self.transcriber.transcribe(media_path=input_path)
        final_track: AudioSegment = self.dubber.generate_dub_track(srt_content=srt_content)

        output_suffix = Path(output_path).suffix.lower()
        if output_suffix == ".mp3":
            final_track.export(output_path, format="mp3")
            return

        if output_suffix == ".mp4":
            tmp_audio_path = str(Path(output_path).with_suffix(".tmp.mp3"))
            final_track.export(tmp_audio_path, format="mp3")
            try:
                media_utils.replace_audio_in_video(video_path=input_path, new_audio_path=tmp_audio_path, output_path=output_path)
            finally:
                try:
                    Path(tmp_audio_path).unlink(missing_ok=True)
                except Exception:
                    pass
            return

        raise ValueError(f"Unsupported output extension: {output_suffix}. Use .mp3 or .mp4")

