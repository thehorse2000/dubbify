# Dubbify

Create AI-generated dubbed audio tracks for media via CLI and SDK.

**Author:** Amr Osama  
**Repository:** https://github.com/thehorse2000/dubbify

## Installation

```bash
pip install dubbify
```

## Configuration

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ELEVENLABS_API_KEY="your_elevenlabs_key"
```

## CLI Usage

- End-to-end:

```bash
dubbify run \
  --input path/to/video.mp4 \
  --output path/to/dubbed_video.mp4 \
  --voice alloy \
  --language en \
  --transcriber openai
```

- Transcribe only:

```bash
dubbify transcribe \
  --input path/to/video.mp4 \
  --output path/to/transcript.srt \
  --language en \
  --transcriber openai
```

- Dub from SRT:

```bash
dubbify dub \
  --input path/to/transcript.srt \
  --output path/to/dubbed_audio.mp3 \
  --voice alloy
```

## SDK Usage

```python
from dubbify import Dubbify, DubbifyConfig

config = DubbifyConfig(voice="Bella", language="en", transcriber="openai")
project = Dubbify(config=config)
project.run(input_path="path/to/video.mp4", output_path="path/to/dubbed_video.mp4")
```

## License

MIT