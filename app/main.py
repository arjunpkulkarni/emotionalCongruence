import os
import time
import shutil
from typing import Dict
import logging
import contextlib
import glob

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import ProcessSessionRequest, ProcessSessionResponse
from app.services.video_processing import (
    download_video_file,
    extract_audio_with_ffmpeg,
    extract_frames_with_ffmpeg,
)
from app.services.analysis import (
    analyze_frames_with_deepface,
    analyze_audio_with_vesper,
    merge_timelines,
    detect_micro_spikes,
)
from app.services.transcription import transcribe_audio_with_faster_whisper
from app.services.congruence_engine import (
    build_congruence_timeline,
    build_session_summary,
)
from app.utils.paths import (
    get_workspace_root,
    create_session_directories,
)

import json

logger = logging.getLogger("emotion_api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(_handler)


app = FastAPI(title="Emotion Analysis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/process_session", response_model=ProcessSessionResponse)
def process_session(payload: ProcessSessionRequest) -> ProcessSessionResponse:
    """
    1) Download video
    2) Extract audio via ffmpeg
    3) Extract frames (1 FPS) via ffmpeg
    4) DeepFace on frames -> facial emotions timeline
    5) Vesper audio emotions (if available)
    6) Merge timelines
    7) Detect micro-spikes
    """
    # Prepare session directories under workspace
    start_time = time.time()
    logger.info(
        "process_session START patient_id=%s video_url=%s",
        payload.patient_id,
        payload.video_url,
    )
    workspace_root = get_workspace_root()
    session_ts = int(time.time())
    session_dir, media_dir, frames_dir, outputs_dir = create_session_directories(
        workspace_root=workspace_root,
        patient_id=payload.patient_id,
        session_ts=session_ts,
    )

    video_path = os.path.join(media_dir, "input.mp4")
    audio_path = os.path.join(media_dir, "audio.wav")

    # 1) Download video
    try:
        download_video_file(video_url=payload.video_url, destination_path=video_path)
    except Exception as exc:
        # Best-effort cleanup for partially downloaded files
        with contextlib.suppress(Exception):
            if os.path.isdir(session_dir):
                shutil.rmtree(session_dir)
        logger.exception("Video download failed")
        raise HTTPException(status_code=400, detail=f"Video download failed: {exc}") from exc
    logger.info("Video downloaded to %s", video_path)

    # 2) Extract audio
    try:
        extract_audio_with_ffmpeg(input_video_path=video_path, output_audio_path=audio_path)
    except Exception as exc:
        logger.exception("Audio extraction failed")
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {exc}") from exc
    logger.info("Audio extracted to %s", audio_path)

    # 3) Extract frames (1 FPS)
    try:
        extract_frames_with_ffmpeg(
            input_video_path=video_path,
            frames_dir=frames_dir,
            fps=1,
            filename_pattern="frame_%04d.png",
        )
    except Exception as exc:
        logger.exception("Frame extraction failed")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {exc}") from exc
    frame_count = len(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    logger.info("Frames extracted to %s count=%d", frames_dir, frame_count)

    # 3.5) Speech-to-text transcription from extracted audio (best-effort)
    transcript_text = None
    transcript_segments = None
    try:
        t_text, t_segments = transcribe_audio_with_faster_whisper(audio_path=audio_path)
        if t_text:
            transcript_text = t_text
            transcript_segments = t_segments
        logger.info(
            "Transcription completed chars=%d segments=%d",
            len(transcript_text or ""),
            len(transcript_segments or []),
        )
        if transcript_text:
            logger.info("Transcript text:\n%s", transcript_text)
        if transcript_segments:
            logger.debug("Transcript segments: %s", transcript_segments)
    except Exception:
        # Do not fail on transcription
        transcript_text = None
        transcript_segments = None
        logger.info("Transcription skipped or failed (best-effort)")

    # 4) DeepFace facial emotion analysis per frame
    try:
        facial_timeline = analyze_frames_with_deepface(frames_dir=frames_dir)
    except Exception as exc:
        logger.exception("DeepFace analysis failed")
        raise HTTPException(status_code=500, detail=f"DeepFace analysis failed: {exc}") from exc
    logger.info("DeepFace analysis completed entries=%d", len(facial_timeline))

    # 5) Audio emotion analysis using Vesper (required)
    try:
        audio_timeline = analyze_audio_with_vesper(audio_path=audio_path)
    except Exception as exc:
        logger.exception("Audio emotion analysis failed (Vesper required)")
        raise HTTPException(status_code=500, detail=f"Audio emotion analysis failed: {exc}") from exc
    logger.info("Audio emotion timeline entries=%d", len(audio_timeline))

    # 6) Merge into a unified timeline
    merged_timeline = merge_timelines(facial_timeline=facial_timeline, audio_timeline=audio_timeline)
    logger.info("Merged timeline entries=%d", len(merged_timeline))

    # 7) Detect micro-spikes on merged timeline
    spikes = detect_micro_spikes(merged_timeline, threshold=payload.spike_threshold)
    logger.info("Detected spikes=%d", len(spikes))

    # 8) Build 10Hz congruence signal and session summary and write outputs for UI
    try:
        def _write_json(path: str, obj: object) -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        congruence_timeline_10hz = build_congruence_timeline(
            merged_timeline=merged_timeline,
            transcript_segments=transcript_segments,
            spikes=spikes,
            target_hz=10.0,
        )
        session_summary = build_session_summary(
            congruence_timeline=congruence_timeline_10hz,
            patient_id=payload.patient_id,
            session_id=session_ts,
            transcript_segments=transcript_segments,
        )
        timeline_json_path = os.path.join(outputs_dir, "timeline.json")
        timeline_1hz_path = os.path.join(outputs_dir, "timeline_1hz.json")
        spikes_json_path = os.path.join(outputs_dir, "spikes.json")
        session_summary_path = os.path.join(outputs_dir, "session_summary.json")
        _write_json(timeline_json_path, congruence_timeline_10hz)
        _write_json(timeline_1hz_path, merged_timeline)
        _write_json(spikes_json_path, spikes)
        _write_json(session_summary_path, session_summary)
        if transcript_text:
            transcript_txt_path = os.path.join(outputs_dir, "transcript.txt")
            with open(transcript_txt_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
        if transcript_segments:
            transcript_segments_path = os.path.join(outputs_dir, "transcript_segments.json")
            _write_json(transcript_segments_path, transcript_segments)
        logger.info("Wrote enriched timeline and session summary to outputs/")
    except Exception as exc:
        logger.exception("Failed to write enriched outputs: %s", exc)

    resp = ProcessSessionResponse(
        patient_id=payload.patient_id,
        session_timestamp=session_ts,
        paths={
            "session_dir": session_dir,
            "media_dir": media_dir,
            "frames_dir": frames_dir,
            "audio_path": audio_path,
            "video_path": video_path,
        },
        timeline_json=merged_timeline,
        spikes_json=spikes,
        timeline_10hz=locals().get("congruence_timeline_10hz"),
        session_summary=locals().get("session_summary"),
        notes="Audio emotion analysis performed with Vesper. Enriched timeline/session_summary written to outputs/.",
        transcript_text=transcript_text,
        transcript_segments=transcript_segments,
    )
    duration = time.time() - start_time
    logger.info(
        "process_session END patient_id=%s session_ts=%d duration_s=%.2f",
        payload.patient_id,
        session_ts,
        duration,
    )
    return resp


