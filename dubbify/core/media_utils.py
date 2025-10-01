from pathlib import Path
from typing import Optional

import ffmpeg


def extract_audio(video_path: str) -> str:
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input media not found: {video_path}")

    output_path = input_path.with_suffix(".audio.wav")

    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), acodec="pcm_s16le", ac=2, ar="44100")
        .overwrite_output()
        .run(quiet=True)
    )

    return str(output_path)


def replace_audio_in_video(video_path: str, new_audio_path: str, output_path: str) -> None:
    input_video = Path(video_path)
    input_audio = Path(new_audio_path)
    output_file = Path(output_path)

    if not input_video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not input_audio.exists():
        raise FileNotFoundError(f"Audio file not found: {new_audio_path}")

    # Merge original video stream with new audio stream
    video_stream = ffmpeg.input(str(input_video))
    audio_stream = ffmpeg.input(str(input_audio))

    (
        ffmpeg
        .output(
            video_stream.video,
            audio_stream.audio,
            str(output_file),
            vcodec="copy",
            acodec="aac",
        )
        .overwrite_output()
        .run(quiet=True)
    )

