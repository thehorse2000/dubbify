from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress

from .models import DubbifyConfig
from . import Dubbify


app = typer.Typer(help="Dubbify - AI-generated dubbing for media files")
console = Console()


@app.command()
def run(
    input: str = typer.Option(..., "--input", help="Path to input media (.mp4/.mov/.mp3/.wav)"),
    output: str = typer.Option(..., "--output", help="Path to output (.mp3 or .mp4)"),
    voice: str = typer.Option("asDeXBMC8hUkhqqL7agO", "--voice", help="ElevenLabs voice name or ID"),
    voice_id: Optional[str] = typer.Option(None, "--voice-id", help="Explicit ElevenLabs voice_id for Convert endpoint"),
    language: Optional[str] = typer.Option(None, "--language", help="ISO 639-1 target/output language (transcript & TTS). Input language auto-detected."),
):
    """Run end-to-end dubbing: transcribe and generate dubbed output."""
    cfg = DubbifyConfig(voice=voice, voice_id=voice_id, language=language)
    dubbify = Dubbify(config=cfg)
    with Progress() as progress:
        t1 = progress.add_task("Transcribing", total=None)
        dubbify.transcriber  # ensure init
        srt_content = dubbify.transcriber.transcribe(input)
        progress.update(t1, completed=1)

        t2 = progress.add_task("Synthesizing & assembling audio", total=None)
        final_track = dubbify.dubber.generate_dub_track(srt_content)
        progress.update(t2, completed=1)

    output_path = Path(output)
    if output_path.suffix.lower() == ".mp3":
        final_track.export(str(output_path), format="mp3")
        console.print("[green]Wrote dubbed audio:[/green]" , str(output_path))
        raise typer.Exit(code=0)
    elif output_path.suffix.lower() == ".mp4":
        tmp_audio = str(output_path.with_suffix(".tmp.mp3"))
        final_track.export(tmp_audio, format="mp3")
        from .core.media_utils import replace_audio_in_video

        replace_audio_in_video(video_path=input, new_audio_path=tmp_audio, output_path=str(output_path))
        Path(tmp_audio).unlink(missing_ok=True)
        console.print("[green]Wrote dubbed video:[/green]", str(output_path))
        raise typer.Exit(code=0)
    else:
        console.print("[red]Unsupported output extension. Use .mp3 or .mp4[/red]")
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    input: str = typer.Option(..., "--input", help="Path to input media"),
    output: str = typer.Option(..., "--output", help="Path to save .srt file"),
    language: Optional[str] = typer.Option(None, "--language", help="ISO 639-1 target/output language (transcript). Input language auto-detected."),
):
    """Transcribe media to SRT subtitle file."""
    cfg = DubbifyConfig(language=language)
    dubbify = Dubbify(config=cfg)
    srt_content = dubbify.transcriber.transcribe(input)
    Path(output).write_text(srt_content, encoding="utf-8")
    console.print("[green]Wrote transcript:[/green]", output)


@app.command()
def dub(
    input: str = typer.Option(..., "--input", help="Path to input .srt file"),
    output: str = typer.Option(..., "--output", help="Path to dubbed audio .mp3"),
    voice: str = typer.Option("asDeXBMC8hUkhqqL7agO", "--voice", help="ElevenLabs voice name or ID"),
    voice_id: Optional[str] = typer.Option(None, "--voice-id", help="Explicit ElevenLabs voice_id for Convert endpoint"),
):
    """Generate a dubbed audio track from an SRT file."""
    cfg = DubbifyConfig(voice=voice, voice_id=voice_id)
    dubbify = Dubbify(config=cfg)
    srt_content = Path(input).read_text(encoding="utf-8")
    final_track = dubbify.dubber.generate_dub_track(srt_content)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    final_track.export(output, format="mp3")
    console.print("[green]Wrote dubbed audio:[/green]", output)

