"""Download YouTube audio and convert it to 16 kHz mono WAV for speaker models.

Standalone usage:
    python youtube_audio.py URL [--start 30] [--end 1:30] [--out-dir audio_cache]

Requires ffmpeg on PATH and the `yt-dlp` Python package.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import soundfile as sf
import torch
import yt_dlp

SAMPLE_RATE = 16000
DEFAULT_CACHE = Path(__file__).resolve().parent / "audio_cache"


def parse_time(t):
    """Parse seconds ('90', '90.5') or 'MM:SS' / 'HH:MM:SS' into float seconds."""
    if t is None or t == "":
        return None
    if isinstance(t, (int, float)):
        return float(t)
    secs = 0.0
    for part in str(t).split(":"):
        secs = secs * 60 + float(part)
    return secs


def _fmt_range(start, end):
    if start is None and end is None:
        return "full"
    s = f"{start:g}" if start is not None else "0"
    e = f"{end:g}" if end is not None else "end"
    return f"{s}-{e}"


def download_audio(url, out_dir=DEFAULT_CACHE, start=None, end=None):
    """Download the best audio stream of `url` and return a 16 kHz mono WAV path.

    `start` / `end` (seconds or 'HH:MM:SS') trim the output so you can pick a
    section where only the target speaker is talking. Downloads and conversions
    are cached by video ID in `out_dir`.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (e.g. `conda install -c conda-forge ffmpeg`)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = parse_time(start), parse_time(end)

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    vid = info["id"]

    wav_path = out_dir / f"{vid}_{_fmt_range(start, end)}_16k.wav"
    if wav_path.exists():
        return wav_path

    raw = next(out_dir.glob(f"{vid}.raw.*"), None)
    if raw is None:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / f"{vid}.raw.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        raw = next(out_dir.glob(f"{vid}.raw.*"))

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(raw), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-sample_fmt", "s16", str(wav_path)]
    subprocess.run(cmd, check=True)
    return wav_path


def load_wav(path):
    """Load a WAV as a float32 tensor of shape (1, T) at 16 kHz."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz, expected {SAMPLE_RATE} Hz")
    return torch.from_numpy(data.mean(axis=1)).unsqueeze(0)


def chunk(wav, win_s=6.0, hop_s=3.0, silence_db=30.0):
    """Split a (1, T) waveform into overlapping windows, shape (N, win).

    Windows whose RMS is more than `silence_db` below the loudest window are
    dropped as silence. Audio shorter than one window is returned whole.
    """
    win, hop = int(win_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
    x = wav.squeeze(0)
    if x.numel() <= win:
        return x.unsqueeze(0)
    chunks = x.unfold(0, win, hop)
    rms_db = 20 * torch.log10(chunks.pow(2).mean(dim=1).sqrt() + 1e-8)
    keep = rms_db > rms_db.max() - silence_db
    return chunks[keep]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("--start", help="start time (seconds or HH:MM:SS)")
    p.add_argument("--end", help="end time (seconds or HH:MM:SS)")
    p.add_argument("--out-dir", default=DEFAULT_CACHE)
    args = p.parse_args()

    path = download_audio(args.url, args.out_dir, args.start, args.end)
    info = sf.info(str(path))
    print(f"{path}  ({info.samplerate} Hz, {info.channels} ch, {info.duration:.1f} s)")


if __name__ == "__main__":
    main()
