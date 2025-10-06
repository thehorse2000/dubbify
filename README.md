# Dubbify

Create AI-generated dubbed audio tracks for media via CLI and SDK.

## Example
#### EN
https://github.com/thehorse2000/dubbify/blob/main/examples/input.mp4

#### ES (Dubbed Version)
https://github.com/thehorse2000/dubbify/blob/main/examples/output.mp4

## Installation

```bash
pip install dubbify
```

## Dependencies

- Python 3.9
- Python packages (installed automatically with `pip install dubbify`):
  - typer0.12.3
  - rich13.7.1
  - pydantic2.7.1
  - srt3.5.3
  - pydub0.25.1
  - ffmpeg-python0.2.0
  - elevenlabs1.3.1
  - openai1.40.0
- System dependency: FFmpeg (required for audio processing and muxing)

Install FFmpeg:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

## Configuration

Set API keys and default model as environment variables:

```bash
export OPENAI_API_KEY="sk-..."          # used for transcription only
export ELEVENLABS_API_KEY="your_elevenlabs_key"  # used for TTS only
export DUBBIFY_TTS_MODEL="eleven_multilingual_v2"  # optional, defaults to eleven_multilingual_v2
```

## CLI Usage

- End-to-end:

```bash
dubbify run \
  --input path/to/video.mp4 \
  --output path/to/dubbed_video.mp4 \
  --voice asDeXBMC8hUkhqqL7agO \
  --language en
```

- Transcribe only:

```bash
dubbify transcribe \
  --input path/to/video.mp4 \
  --output path/to/transcript.srt \
  --language en
```

- Dub from SRT:

```bash
dubbify dub \
  --input path/to/transcript.srt \
  --output path/to/dubbed_audio.mp3 \
  --voice asDeXBMC8hUkhqqL7agO
```

Notes:
- `--voice-id` can be provided to use the ElevenLabs Convert endpoint explicitly; if omitted, `--voice` is treated as the ID.
- `--output` accepts `.mp3` (audio only) or `.mp4` (video with replaced audio).
- Run `dubbify --help` to list all options.

## SDK Usage

```python
from dubbify import Dubbify, DubbifyConfig

config = DubbifyConfig(voice="Bella", language="en")
project = Dubbify(config=config)
project.run(input_path="path/to/video.mp4", output_path="path/to/dubbed_video.mp4")
```

More examples:

```python
# Transcribe to an SRT file
project.transcribe(input_path="path/to/video.mp4", output_srt_path="out.srt")

# Generate dubbed audio from an existing SRT
project.dub(srt_path="out.srt", output_audio_path="dubbed.mp3")
```

## How it Works

- Transcription pipeline:
  - Extracts audio from video with FFmpeg when needed.
  - Splits long audio into size-limited chunks for reliable cloud transcription.
  - Uses OpenAI Whisper (translations for English target, transcriptions otherwise) to produce time-stamped segments.
  - Optionally translates subtitles when a non-English `language` is set.
  - Optionally refines SRT timings via an OpenAI text model to improve dubbing pace while preserving total duration.

- Dubbing pipeline:
  - For each subtitle line, synthesizes speech with ElevenLabs.
    - Prefers the modern Convert endpoint when available (`voice_id` or `voice` as ID); otherwise falls back to `generate`.
  - Measures each synthesized clip’s duration and schedules overlays using a forward-push plus backward reconciliation algorithm to keep the final end time consistent.
  - Assembles a single audio track and, if output is `.mp4`, swaps the original audio with the dubbed track using FFmpeg while copying the video stream.

Environment requirements:
- `OPENAI_API_KEY` is required for transcription/translation and timing refinement.
- `ELEVENLABS_API_KEY` is required for TTS.
- FFmpeg must be installed and available on PATH.

## Known Limitations
1. Dubbify doesn't handle Music and sound Effects correctly -- Needs some work.
2. Dubbify doesn't recognize multiple speakers if there are any (it generates same voice for all speakers)

## License

MIT
