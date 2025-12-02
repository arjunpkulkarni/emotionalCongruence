import glob
import os
from typing import Any, Dict, List, Optional


CANONICAL_EMOTIONS: List[str] = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "neutral",
]


def _normalize_emotion_dict(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Map backend-specific keys into our canonical 7-emotion space and
    soft-normalize to sum to 1. Missing emotions -> 0.
    """
    mapped: Dict[str, float] = {e: 0.0 for e in CANONICAL_EMOTIONS}
    # Example mappings – adjust to actual DeepFace/Vesper labels if needed
    mapping_table: Dict[str, str] = {
        "happy": "joy",
        "sad": "sadness",
        "angry": "anger",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
        "neutral": "neutral",
        # Also accept canonical keys directly
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
    }
    for k, v in (raw or {}).items():
        canonical = mapping_table.get(str(k).lower())
        if canonical is not None:
            mapped[canonical] += float(v)
    s = sum(mapped.values()) or 1.0
    for k in mapped:
        mapped[k] /= s
    return mapped


def _sorted_frame_paths(frames_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    return paths


def analyze_frames_with_deepface(
    frames_dir: str, 
    max_frames: Optional[int] = None,
    silent: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each frame, run DeepFace.analyze(..., actions=['emotion']) and return
    a list of entries: {"t": second_index, "emotions": {emotion: score, ...}}
    
    Args:
        frames_dir: Directory containing frame images
        max_frames: Optional limit on number of frames to process (for faster processing)
        silent: Suppress DeepFace logging
    
    Returns:
        List of timeline entries with emotion analysis
    """
    try:
        from deepface import DeepFace  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "DeepFace is not installed or failed to import. Please install 'deepface'."
        ) from exc

    timeline: List[Dict[str, Any]] = []
    frame_paths = _sorted_frame_paths(frames_dir)
    
    # For very long videos, optionally sample frames to reduce processing time
    if max_frames and len(frame_paths) > max_frames:
        # Sample frames evenly throughout the video
        step = len(frame_paths) / max_frames
        sampled_indices = [int(i * step) for i in range(max_frames)]
        sampled_paths = [(frame_paths[i], i) for i in sampled_indices]
    else:
        sampled_paths = [(path, idx) for idx, path in enumerate(frame_paths)]

    for frame_path, original_idx in sampled_paths:
        # Guard against frames without detectable faces by disabling strict enforcement
        analysis_obj = DeepFace.analyze(
            img_path=frame_path,
            actions=["emotion"],
            enforce_detection=False,
            silent=silent,
        )
        # DeepFace may return dict or list depending on detection results
        if isinstance(analysis_obj, list) and analysis_obj:
            first = analysis_obj[0]
            emotions = first.get("emotion", {}) if isinstance(first, dict) else {}
        elif isinstance(analysis_obj, dict):
            emotions = analysis_obj.get("emotion", {})
        else:
            emotions = {}

        norm = _normalize_emotion_dict(emotions)
        entry = {
            "t": original_idx,  # seconds, aligned to 1 FPS frames
            "emotions": norm,
            "frame_path": frame_path,
        }
        timeline.append(entry)
    
    # Sort by time to ensure proper ordering
    timeline.sort(key=lambda x: x["t"])
    return timeline


def analyze_audio_with_vesper(audio_path: str) -> List[Dict[str, Any]]:
    """
    Audio emotion analysis using lightweight Vesper inference shim.
    Returns a per-second timeline like [{"t": second, "emotions": {...}}].
    If the backend is unavailable, returns a neutral-leaning timeline using audio duration.
    """
    # Determine audio duration in seconds
    duration_s = 0.0
    try:
        import soundfile as sf  # type: ignore
        _data, sr = sf.read(audio_path, always_2d=False)
        frames = getattr(sf.SoundFile(audio_path), "frames", None)
        if frames is not None:
            duration_s = float(frames) / float(sr or 1)
        else:
            # Fallback if frames attribute is unavailable
            import numpy as np  # type: ignore
            n = float(np.asarray(_data).shape[0])
            duration_s = n / float(sr or 1)
    except Exception:
        try:
            import librosa  # type: ignore
            duration_s = float(librosa.get_duration(path=audio_path))
        except Exception:
            duration_s = 0.0

    # Use vesper.inference if available
    emotions_dist: Dict[str, float] = {}
    try:
        from vesper.inference import predict_emotion  # type: ignore
        emotions_dist = predict_emotion(audio_path) or {}
    except Exception:
        emotions_dist = {}

    # Always return a timeline; if duration unknown, default to a single point
    total_seconds = max(1, int(round(duration_s)))
    emotions_dist = {k: float(v) for k, v in (emotions_dist or {}).items()}
    norm = _normalize_emotion_dict(emotions_dist)
    timeline: List[Dict[str, Any]] = []
    for t in range(total_seconds):
        timeline.append({"t": t, "emotions": norm})
    return timeline


def _merge_emotions(face: Dict[str, float], audio: Dict[str, float]) -> Dict[str, float]:
    """
    Combine face and audio emotions by averaging when both present, otherwise pass-through.
    """
    keys = set(face.keys()) | set(audio.keys())
    merged: Dict[str, float] = {}
    for k in keys:
        fv = face.get(k)
        av = audio.get(k)
        if fv is not None and av is not None:
            merged[k] = 0.5 * (float(fv) + float(av))
        elif fv is not None:
            merged[k] = float(fv)
        elif av is not None:
            merged[k] = float(av)
    return merged


def merge_timelines(
    facial_timeline: List[Dict[str, Any]],
    audio_timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Returns entries of form:
      {
        "t": second,
        "face": {...},
        "audio": {...},
        "combined": {...},
        "micro_spike": False
      }
    """
    face_by_t = {e["t"]: e.get("emotions", {}) for e in facial_timeline}
    audio_by_t = {e["t"]: e.get("emotions", {}) for e in audio_timeline}
    all_ts = sorted(set(face_by_t.keys()) | set(audio_by_t.keys()))
    merged: List[Dict[str, Any]] = []
    for t in all_ts:
        face_em = face_by_t.get(t, {})
        aud_em = audio_by_t.get(t, {})
        combined = _merge_emotions(face_em, aud_em)
        merged.append(
            {
                "t": t,
                "face": face_em,
                "audio": aud_em,
                "combined": combined,
                "micro_spike": False,
            }
        )
    return merged


def detect_micro_spikes(
    merged_timeline: List[Dict[str, Any]],
    threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Mark visual micro-spikes only.
    For each second, compare face distribution vs previous second on the canonical 7D space.
    entry['micro_spike'] = True if any dimension jump exceeds threshold, else False.
    Returns the annotated merged timeline (same list, mutated).
    """
    prev_face: Optional[Dict[str, float]] = None
    for entry in sorted(merged_timeline, key=lambda e: e["t"]):
        face = entry.get("face") or {}
        if prev_face is not None and face:
            deltas = [
                abs(float(face.get(e, 0.0)) - float(prev_face.get(e, 0.0)))
                for e in CANONICAL_EMOTIONS
            ]
            entry["micro_spike"] = any(d > threshold for d in deltas)
        else:
            entry["micro_spike"] = False
        prev_face = face
    return merged_timeline


