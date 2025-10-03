## ElevenLabs Convert Integration (No Timestamps)

### Overview
This document explains how Dubbify integrates with ElevenLabs Text-to-Speech using the Create speech (Convert) endpoint to synthesize per-cue audio from an SRT file, measure actual audio durations, and reconcile subtitle timings so that the dubbed audio stays in sync while preserving the original global start and end times.

- API reference: [Create speech](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)

### What’s implemented
- Per-cue synthesis using ElevenLabs Convert (no timestamps/alignment).
- Actual duration is measured from the returned audio.
- Forward/backward timing reconciliation ensures:
  - Cues remain ordered and non-overlapping.
  - The very first start and final end of the subtitle track remain unchanged.
- Fallbacks:
  - If Convert isn’t available in the installed SDK, Dubbify tries legacy generation paths.
  - If `voice_id` isn’t provided, `voice` is treated as the `voice_id` (commonly works when a voice ID is provided as the voice name).

### Requirements
- Environment
  - Set `ELEVENLABS_API_KEY` in your environment.
  - Ensure `ffmpeg` is installed and on PATH (required by `pydub`).
- Python dependencies (installed via your project’s environment):
  - `elevenlabs` (SDK)
  - `pydub`
  - `srt`

### Configuration
The `DubbifyConfig` supports the following relevant fields (see `dubbify/models.py`):

- `voice: str` — Voice name or ID (used as a fallback `voice_id` when `voice_id` is not explicitly provided)
- `voice_id: Optional[str]` — Explicit ElevenLabs `voice_id` for the Convert endpoint
- `model: str` — TTS model (e.g., `eleven_multilingual_v2`)
- `use_convert_endpoint: bool` — If true, use the Convert endpoint (default true)

The API key is read from `ELEVENLABS_API_KEY`.

### CLI usage

Generate dubbed audio directly from an input media file (end-to-end transcribe → dub → mux):

```bash
python -m dubbify.main run \
  --input data/input/input.mp4 \
  --output data/output/output.mp4 \
  --voice-id JBFqnCBsd6RMkjVDRZzb
```

Generate dubbed audio from an existing SRT file:

```bash
python -m dubbify.main dub \
  --input data/output/transcript_en_openai.srt \
  --output data/output/output.mp3 \
  --voice-id JBFqnCBsd6RMkjVDRZzb
```

Notes:
- If `--voice-id` is omitted, `--voice` is used as a fallback `voice_id`.
- The Convert endpoint is preferred when `use_convert_endpoint` is true; otherwise the legacy path is attempted.

### Timing reconciliation details

1) For each SRT cue, Dubbify calls Convert and decodes audio bytes. The actual duration `speech_duration_i` is measured using `pydub`.

2) Forward pass (push):
- Maintain `offset_ms = 0`.
- For cue i:
  - `new_start_i = original_start_i + offset_ms`
  - `new_end_i = new_start_i + speech_duration_i`
  - If `new_end_i` exceeds `(original_end_i + offset_ms)`, increase `offset_ms` by the overflow.

3) Backward pass (pull):
- If the final scheduled end is later than the original track end, walk backward and pull cues earlier within the available gaps between consecutive cues until the final end matches the original final end.
- If there isn’t enough slack to reconcile (extreme mismatch), an error is raised with a clear message.

This preserves the original global start and end while accommodating speech that is shorter or longer than individual cue windows.

### Fallback behavior
- If the SDK exposes `client.text_to_speech.convert`, it is used.
- If it only exposes `client.convert`, that path is also handled.
- If Convert fails or is unavailable, Dubbify tries legacy generation (top-level `generate` or `client.generate`).
- If both modern and legacy paths are unavailable, a clear error is emitted describing the required configuration.

### Troubleshooting
- "SDK does not support client.generate": Provide `--voice-id` (or use `--voice` with a valid voice ID) so the Convert path is used.
- Missing dependencies warnings: Ensure `pydub`, `srt`, and the `elevenlabs` SDK are installed in your environment.
- Audio decoding errors: Install `ffmpeg` and verify it’s available in your shell.
- Reconciliation failed: Total synthesized speech is significantly longer than the sum of cue windows. Consider editing subtitles, regenerating text, or splitting cues.

### References
- ElevenLabs Create speech: [Create speech](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)


