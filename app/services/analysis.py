import glob
import os
from typing import Any, Dict, List, Optional, Tuple


def _sorted_frame_paths(frames_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    return paths


def analyze_frames_with_deepface(frames_dir: str) -> List[Dict[str, Any]]:
    """
    For each frame, run DeepFace.analyze(..., actions=['emotion']) and return
    a list of entries: {"t": second_index, "emotions": {emotion: score, ...}}
    """
    try:
        from deepface import DeepFace  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "DeepFace is not installed or failed to import. Please install 'deepface'."
        ) from exc

    timeline: List[Dict[str, Any]] = []
    frame_paths = _sorted_frame_paths(frames_dir)

    for idx, frame_path in enumerate(frame_paths):
        # Guard against frames without detectable faces by disabling strict enforcement
        analysis_obj = DeepFace.analyze(
            img_path=frame_path,
            actions=["emotion"],
            enforce_detection=False,
        )
        # DeepFace may return dict or list depending on detection results
        if isinstance(analysis_obj, list) and analysis_obj:
            first = analysis_obj[0]
            emotions = first.get("emotion", {}) if isinstance(first, dict) else {}
        elif isinstance(analysis_obj, dict):
            emotions = analysis_obj.get("emotion", {})
        else:
            emotions = {}

        entry = {
            "t": idx,  # seconds, aligned to 1 FPS frames
            "emotions": emotions,
            "frame_path": frame_path,
        }
        timeline.append(entry)
    return timeline


def analyze_audio_with_vesper(audio_path: str) -> List[Dict[str, Any]]:
    """
    Optional audio emotion analysis using Vesper-style API:
      from vesper import extract_features, predict

    Returns a list like [{"t": second, "emotions": {...}}].
    If Vesper is unavailable, returns an empty list.
    """
    try:
        # Placeholder import path based on the prompt; may vary depending on the actual package
        from vesper import extract_features, predict  # type: ignore
    except Exception:
        # Graceful fallback if Vesper is not installed
        return []

    try:
        features, times = extract_features(audio_path)  # expected to return features and per-frame times
        preds = predict(features)  # expected to return per-time emotion distributions
    except Exception:
        return []

    timeline: List[Dict[str, Any]] = []
    for i, t in enumerate(times):
        emotions = preds[i] if i < len(preds) and isinstance(preds[i], dict) else {}
        timeline.append({"t": int(round(t)), "emotions": emotions})
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
        "combined": {...}
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
            }
        )
    return merged


def detect_micro_spikes(
    merged_timeline: List[Dict[str, Any]],
    threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    For each emotion dimension, compare value vs previous frame.
    If abs(delta) > threshold -> mark as spike.
    Returns:
      [{"t": second, "emotion": name, "delta": value, "from": prev, "to": curr}]
    """
    spikes: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, float]] = None
    for entry in sorted(merged_timeline, key=lambda e: e["t"]):
        current = entry.get("combined") or entry.get("face") or {}
        if prev is not None:
            keys = set(prev.keys()) | set(current.keys())
            for k in keys:
                v_prev = float(prev.get(k, 0.0))
                v_curr = float(current.get(k, 0.0))
                delta = v_curr - v_prev
                if abs(delta) > threshold:
                    spikes.append(
                        {
                            "t": entry["t"],
                            "emotion": k,
                            "delta": delta,
                            "from": v_prev,
                            "to": v_curr,
                        }
                    )
        prev = current
    return spikes


