from pathlib import Path

import srt

from dubbify.models import DubbifyConfig
from dubbify.core.transcriber import Transcriber


def test_transcriber_raises_without_api_key(tmp_path: Path, monkeypatch):
    # Create a fake audio file path
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    # Ensure no ELEVENLABS_API_KEY is present
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    cfg = DubbifyConfig()
    t = Transcriber(cfg)
    try:
        t.transcribe(str(audio_path))
        assert False, "Expected an error due to missing ELEVENLABS_API_KEY"
    except Exception as exc:
        assert "ELEVENLABS_API_KEY" in str(exc)

