from pathlib import Path
from datetime import timedelta

import srt
from pydub import AudioSegment

from dubbify.models import DubbifyConfig
from dubbify.core.dubber import Dubber


def test_dubber_assembles_track(monkeypatch, tmp_path: Path):
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="world"),
    ]
    srt_text = srt.compose(subs)

    cfg = DubbifyConfig(voice="TestVoice")
    d = Dubber(cfg)

    class DummyClient:
        def generate(self, text, voice, model, output_format):
            # Return 100ms of silence as mp3 bytes per segment
            return AudioSegment.silent(duration=100).export(format="mp3").read()

    def ensure_client():
        d._elevenlabs_client = DummyClient()

    monkeypatch.setattr(d, "_ensure_elevenlabs", ensure_client)

    track = d.generate_dub_track(srt_text)
    assert isinstance(track, AudioSegment)
    # Total duration should be at least 2 seconds (from last subtitle end)
    assert track.duration_seconds >= 1.9

