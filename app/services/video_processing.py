import os
import subprocess
import shutil
from typing import Optional

import requests


def _ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH. Please install ffmpeg and retry.")


def download_video_file(video_url: str, destination_path: str, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with requests.get(video_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(destination_path, "wb") as dest_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    dest_file.write(chunk)


def extract_audio_with_ffmpeg(input_video_path: str, output_audio_path: str) -> None:
    _ensure_ffmpeg_exists()
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        output_audio_path,
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {completed.stderr.decode(errors='ignore')}")


def extract_frames_with_ffmpeg(
    input_video_path: str,
    frames_dir: str,
    fps: int = 1,
    filename_pattern: str = "frame_%04d.png",
) -> None:
    _ensure_ffmpeg_exists()
    os.makedirs(frames_dir, exist_ok=True)
    output_pattern = os.path.join(frames_dir, filename_pattern)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"fps={fps}",
        output_pattern,
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {completed.stderr.decode(errors='ignore')}")


