import argparse
import json
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict
import numpy as np

from app.services.video_processing import (
    extract_audio_with_ffmpeg,
    extract_frames_with_ffmpeg,
)
from app.services.analysis import (
    analyze_frames_with_deepface,
    analyze_audio_with_vesper,
    merge_timelines,
    detect_micro_spikes,
)
from app.services.congruence import (
    attach_text_bins_to_timeline,
    estimate_text_bins_emotion,
    compute_congruence_metrics,
    extract_congruence_events,
)
from app.services.transcription import transcribe_audio_with_faster_whisper
from app.utils.paths import get_workspace_root, create_session_directories


logger = logging.getLogger("local_test")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(_handler)


def _write_json(path, data):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(data), f, indent=2, ensure_ascii=False)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run_local_pipeline(
    input_video_path: str,
    patient_id: str,
    spike_threshold: float = 0.2,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    """
    Local runner that mirrors the FastAPI /process_session pipeline:
      1) Copy local video into session media dir as input.mp4
      2) Extract audio (wav) with ffmpeg
      3) Extract frames (1 FPS) with ffmpeg
      4) DeepFace analysis on frames
      5) Optional audio emotion analysis (Vesper)
      6) Merge timelines
      7) Detect micro-spikes
      8) Save outputs to outputs/ directory
    """
    start_time = time.time()
    workspace_root = get_workspace_root()
    session_ts = int(time.time())
    session_dir, media_dir, frames_dir, outputs_dir = create_session_directories(
        workspace_root=workspace_root,
        patient_id=patient_id,
        session_ts=session_ts,
    )

    # Copy source video into session as input.mp4
    dest_video_path = os.path.join(media_dir, "input.mp4")
    logger.info("Copying video -> %s", dest_video_path)
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    shutil.copy2(input_video_path, dest_video_path)

    # Extract audio
    audio_path = os.path.join(media_dir, "audio.wav")
    logger.info("Extracting audio -> %s", audio_path)
    extract_audio_with_ffmpeg(
        input_video_path=dest_video_path, output_audio_path=audio_path
    )

    # Extract frames (1 FPS)
    logger.info("Extracting frames -> %s", frames_dir)
    extract_frames_with_ffmpeg(
        input_video_path=dest_video_path,
        frames_dir=frames_dir,
        fps=1,
        filename_pattern="frame_%04d.png",
    )

    # Transcription (best-effort)
    logger.info("Transcribing audio (best-effort, fast_mode=%s)", fast_mode)
    transcript_text, transcript_segments = transcribe_audio_with_faster_whisper(
        audio_path=audio_path,
        fast_mode=fast_mode,
    )
    logger.info(
        "Transcription completed chars=%d segments=%d",
        len(transcript_text or ""),
        len(transcript_segments or []),
    )
    if transcript_text:
        logger.info("Transcript text:\n%s", transcript_text)

    # DeepFace facial emotion analysis
    logger.info("Analyzing frames with DeepFace (fast_mode=%s)", fast_mode)
    # For 2-hour videos (~7200 frames), limit to 1800 frames for 4x speedup
    max_frames = 1800 if fast_mode else None
    facial_timeline = analyze_frames_with_deepface(
        frames_dir=frames_dir,
        max_frames=max_frames,
        silent=True,
    )

    # Audio emotion analysis (Vesper required)
    logger.info("Analyzing audio emotions (Vesper)")
    audio_timeline = analyze_audio_with_vesper(audio_path=audio_path)

    # Merge timelines
    logger.info("Merging timelines")
    merged_timeline = merge_timelines(
        facial_timeline=facial_timeline, audio_timeline=audio_timeline
    )

    # Detect micro-spikes
    logger.info("Detecting micro-spikes (threshold=%.3f)", spike_threshold)
    spikes = detect_micro_spikes(merged_timeline, threshold=spike_threshold)

    # Congruence analysis (text ↔ audio ↔ face)
    enriched_timeline = attach_text_bins_to_timeline(
        merged_timeline=merged_timeline, transcript_segments=transcript_segments
    )
    # Use LLM if configured; otherwise, a heuristic fallback will be used.
    enriched_timeline = estimate_text_bins_emotion(
        enriched_timeline=enriched_timeline, use_llm=True
    )
    congruence_timeline = compute_congruence_metrics(
        enriched_timeline_with_text=enriched_timeline, spikes=spikes
    )
    congruence_events = extract_congruence_events(
        congruence_timeline=congruence_timeline, score_threshold=0.4, max_events=32
    )

    # Build 10Hz congruence signal and session summary
    from app.services.congruence_engine import (
        build_congruence_timeline,
        build_session_summary,
    )
    congruence_timeline_10hz = build_congruence_timeline(
        merged_timeline=merged_timeline,
        transcript_segments=transcript_segments,
        spikes=spikes,
        target_hz=10.0,
    )
    session_summary = build_session_summary(
        congruence_timeline=congruence_timeline_10hz,
        patient_id=patient_id,
        session_id=session_ts,
        transcript_segments=transcript_segments,
    )

    # Save outputs
    timeline_json_path = os.path.join(outputs_dir, "timeline.json")
    timeline_1hz_path = os.path.join(outputs_dir, "timeline_1hz.json")
    spikes_json_path = os.path.join(outputs_dir, "spikes.json")
    transcript_txt_path = os.path.join(outputs_dir, "transcript.txt")
    transcript_segments_path = os.path.join(outputs_dir, "transcript_segments.json")
    congruence_timeline_path = os.path.join(outputs_dir, "congruence_timeline.json")
    congruence_events_path = os.path.join(outputs_dir, "congruence_events.json")
    session_summary_path = os.path.join(outputs_dir, "session_summary.json")

    logger.info("Writing outputs to %s", outputs_dir)
    # Enriched 10Hz timeline for UI consumption
    _write_json(timeline_json_path, congruence_timeline_10hz)
    # Preserve the previous 1Hz merged timeline for debugging
    _write_json(timeline_1hz_path, merged_timeline)
    _write_json(spikes_json_path, spikes)
    _write_json(congruence_timeline_path, congruence_timeline)
    _write_json(congruence_events_path, congruence_events)
    _write_json(session_summary_path, session_summary)
    if transcript_text:
        _write_text(transcript_txt_path, transcript_text)
    if transcript_segments:
        _write_json(transcript_segments_path, transcript_segments)

    duration = time.time() - start_time
    logger.info(
        "Completed local pipeline patient_id=%s session_ts=%d duration_s=%.2f",
        patient_id,
        session_ts,
        duration,
    )

    return {
        "patient_id": patient_id,
        "session_timestamp": session_ts,
        "paths": {
            "session_dir": session_dir,
            "media_dir": media_dir,
            "frames_dir": frames_dir,
            "outputs_dir": outputs_dir,
            "video_path": dest_video_path,
            "audio_path": audio_path,
        },
        "timeline_count": len(merged_timeline),
        "timeline_10hz_count": len(congruence_timeline_10hz),
        "spikes_count": len(spikes),
        "transcript_chars": len(transcript_text or ""),
        "transcript_segments_count": len(transcript_segments or []),
        "duration_s": duration,
        "session_summary": session_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local emotion analysis pipeline on a video file."
    )
    default_video = os.path.join(
        get_workspace_root(),
        "data",
        "sessions",
        "4e3c1260-9e27-4cc8-9720-114e068d03f1",
        "1763914951",
        "media",
        "input.mp4",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=default_video,
        help="Path to local MP4 to analyze",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="local_test",
        help="Patient/subject identifier for the session directory",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=0.2,
        help="Delta threshold for spike detection (0.0 - 1.0)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=True,
        help="Enable fast mode for 2-hour video scale (parallel LLM, faster models, frame sampling)",
    )
    parser.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="Disable fast mode (slower but higher quality)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        result = run_local_pipeline(
            input_video_path=args.video,
            patient_id=args.patient_id,
            spike_threshold=args.spike_threshold,
            fast_mode=args.fast_mode,
        )
        # Print a concise summary
        print(json.dumps(result, indent=2))
        # Also print the session_summary explicitly for convenience
        ss = result.get("session_summary")
        if ss:
            print("\nSESSION SUMMARY:")
            print(json.dumps(ss, indent=2))
    except Exception:
        logger.exception("Local pipeline failed")
        sys.exit(1)
