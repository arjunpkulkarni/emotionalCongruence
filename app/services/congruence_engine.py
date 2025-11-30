from typing import Any, Dict, List, Optional, Tuple
import math

from .llm import analyze_text_emotion_with_llm, generate_incongruence_reason


# Supported emotions and canonical order (canonical 7D for TECS)
EMOTION_ORDER: List[str] = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
EMOTION_KEYS: List[str] = EMOTION_ORDER[:]

# Canonical 7-emotion basis for TECS/MSW-TECS
CANONICAL_EMOTIONS: List[str] = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "neutral",
]

# Fixed valence/arousal lookup as specified
VALENCE_TABLE: Dict[str, float] = {
    "happy": 1.0,
    "neutral": 0.0,
    "sad": -0.8,
    "angry": -0.9,
    "fear": -0.9,
    "disgust": -0.9,
    "surprise": 0.3,
}
AROUSAL_TABLE: Dict[str, float] = {
    "happy": 0.7,
    "neutral": 0.2,
    "sad": 0.3,
    "angry": 0.9,
    "fear": 0.9,
    "disgust": 0.6,
    "surprise": 0.9,
}

# Thresholds
VALENCE_THRESHOLD_BASE: float = 0.4
VALENCE_THRESHOLD_TEXT_RELAXED: float = 0.6
AROUSAL_THRESHOLD_BASE: float = 0.3

def _normalize_distribution(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    s = sum(float(v) for v in d.values())
    if s <= 0:
        return {k: 0.0 for k in d.keys()}
    return {k: float(v) / s for k, v in d.items()}


def _ensure_full_probs(d: Dict[str, float], keys: List[str] = EMOTION_KEYS) -> Dict[str, float]:
    if not d:
        out = {k: 0.0 for k in keys}
        out["neutral"] = 1.0
        return out
    out = {k: float(d.get(k, 0.0)) for k in keys}
    return _normalize_distribution(out)


def _interpolate_distributions(
    d0: Dict[str, float],
    d1: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    keys = set(d0.keys()) | set(d1.keys()) | set(EMOTION_KEYS)
    out: Dict[str, float] = {}
    for k in keys:
        v0 = float(d0.get(k, 0.0))
        v1 = float(d1.get(k, 0.0))
        out[k] = (1.0 - alpha) * v0 + alpha * v1
    return _normalize_distribution(out)


def _build_lookup_by_second(timeline: List[Dict[str, Any]], field: str) -> Dict[int, Dict[str, float]]:
    by_t: Dict[int, Dict[str, float]] = {}
    for e in timeline:
        try:
            t = int(e.get("t", 0))
        except Exception:
            t = 0
        by_t[t] = _ensure_full_probs(dict(e.get(field, {}) or {}))
    return by_t


def _sample_emotions_at_time(
    face_by_t: Dict[int, Dict[str, float]],
    audio_by_t: Dict[int, Dict[str, float]],
    t: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    t0 = int(math.floor(t))
    t1 = t0 + 1
    alpha = float(t - t0)
    f0 = face_by_t.get(t0, {})
    f1 = face_by_t.get(t1, {})
    a0 = audio_by_t.get(t0, {})
    a1 = audio_by_t.get(t1, {})
    face = _interpolate_distributions(f0, f1, alpha) if alpha > 0.0 else _ensure_full_probs(f0)
    audio = _interpolate_distributions(a0, a1, alpha) if alpha > 0.0 else _ensure_full_probs(a0)
    return face, audio


def _active_segment_at_t(segments: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    for seg in segments:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        if s <= t < e:
            return seg
    return None


def _count_spikes_near_t_1hz(merged_timeline: List[Dict[str, Any]], t: float) -> int:
    """
    Count visual micro-spikes around second t using the annotated 1Hz merged timeline.
    """
    sec = int(math.floor(t))
    for e in merged_timeline:
        if int(e.get("t", -1)) == sec:
            return 1 if bool(e.get("micro_spike", False)) else 0
    return 0


def _analyze_transcript_segments(segments: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not segments:
        return []
    analyzed: List[Dict[str, Any]] = []
    for seg in segments:
        txt = str(seg.get("text", "")).strip()
        analysis = analyze_text_emotion_with_llm(txt) if txt else None
        if not analysis or "emotion_distribution" not in analysis:
            dist = {"neutral": 1.0}
            val = 0.0
            aro = 0.0
            style = "uncertain"
        else:
            dist = analysis.get("emotion_distribution", {}) or {}
            val = float(analysis.get("valence", 0.0) or 0.0)
            aro = float(analysis.get("arousal", 0.0) or 0.0)
            style = str(analysis.get("style", "uncertain") or "uncertain")
        analyzed.append(
            {
                **seg,
                "emotion_distribution": dist,
                "valence": val,
                "arousal": aro,
                "style": style,
            }
        )
    return analyzed


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    num = 0.0
    na = 0.0
    nb = 0.0
    for e in CANONICAL_EMOTIONS:
        va = float(a.get(e, 0.0))
        vb = float(b.get(e, 0.0))
        num += va * vb
        na += va * va
        nb += vb * vb
    denom = math.sqrt(na) * math.sqrt(nb) + 1e-8
    return num / denom


def _valence(vec: Dict[str, float], alpha: float = 0.5) -> float:
    joy = float(vec.get("joy", 0.0))
    surprise = float(vec.get("surprise", 0.0))
    sadness = float(vec.get("sadness", 0.0))
    anger = float(vec.get("anger", 0.0))
    fear = float(vec.get("fear", 0.0))
    disgust = float(vec.get("disgust", 0.0))
    positive = joy + alpha * surprise
    negative = sadness + anger + fear + disgust
    return positive - negative


def _intensity(vec: Dict[str, float]) -> float:
    return 1.0 - float(vec.get("neutral", 0.0))


def _interp_face_audio_to_10hz(
    merged_timeline: List[Dict[str, Any]],
    target_hz: float,
) -> List[Dict[str, Any]]:
    face_by_t = _build_lookup_by_second(merged_timeline, "face")
    audio_by_t = _build_lookup_by_second(merged_timeline, "audio")
    if not merged_timeline:
        return []
    max_t = max(int(e.get("t", 0)) for e in merged_timeline)
    duration = float(max_t)
    step = 1.0 / max(target_hz, 1.0)
    out: List[Dict[str, Any]] = []
    t = 0.0
    while t <= duration + 1e-9:
        tt = round(t, 2)
        face_probs, audio_probs = _sample_emotions_at_time(face_by_t, audio_by_t, t)
        out.append({"t": tt, "face": face_probs, "audio": audio_probs})
        t += step
    return out


def _attach_text_to_timeline(
    timeline_10hz: List[Dict[str, Any]],
    analyzed_segments: List[Dict[str, Any]],
) -> None:
    for step in timeline_10hz:
        seg = _active_segment_at_t(analyzed_segments, float(step.get("t", 0.0)))
        if seg:
            step["text"] = {
                "emotion_distribution": dict(seg.get("emotion_distribution", {})),
                "style": seg.get("style", "uncertain"),
            }
            step["client_speaking"] = True
        else:
            step["text"] = None
            step["client_speaking"] = False


def _attach_spikes_to_10hz(
    timeline_10hz: List[Dict[str, Any]],
    merged_timeline_1hz: List[Dict[str, Any]],
) -> None:
    for step in timeline_10hz:
        step["micro_spike"] = bool(_count_spikes_near_t_1hz(merged_timeline_1hz, float(step.get("t", 0.0))))


def _majority_smooth_bool(seq: List[Optional[bool]], window_radius: int = 3) -> List[Optional[bool]]:
    n = len(seq)
    out: List[Optional[bool]] = [None] * n
    for i in range(n):
        if seq[i] is None:
            out[i] = None
            continue
        L = max(0, i - window_radius)
        R = min(n - 1, i + window_radius)
        window_vals = [seq[j] for j in range(L, R + 1) if seq[j] is not None]
        if not window_vals:
            out[i] = seq[i]
            continue
        true_count = sum(1 for v in window_vals if v)
        false_count = sum(1 for v in window_vals if v is False)
        out[i] = True if true_count >= false_count else False
    return out


def build_congruence_timeline(
    merged_timeline: List[Dict[str, Any]],
    transcript_segments: Optional[List[Dict[str, Any]]],
    spikes: Optional[List[Dict[str, Any]]],
    target_hz: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Construct a 10Hz timeline with Tri-Modal Emotional Congruence Score (TECS)
    and flags for incongruent moments.
    """
    if not merged_timeline:
        return []
    # 1) Interpolate face/audio to 10Hz
    timeline_10hz = _interp_face_audio_to_10hz(merged_timeline, target_hz)
    # 2) Attach text distribution and speaking flag
    analyzed_segments = _analyze_transcript_segments(transcript_segments or [])
    _attach_text_to_timeline(timeline_10hz, analyzed_segments)
    # 3) Propagate micro_spike to 10Hz
    _attach_spikes_to_10hz(timeline_10hz, merged_timeline)

    # 4) Compute TECS, valence, intensity, incongruence
    for step in timeline_10hz:
        face = (step.get("face") or {"neutral": 1.0})
        audio = (step.get("audio") or {"neutral": 1.0})
        text_field = step.get("text") if isinstance(step.get("text"), dict) else None
        text_dist = text_field.get("emotion_distribution") if text_field else None
        text = (text_dist or {"neutral": 1.0})

        s_ta = _cosine(text, audio)
        s_tv = _cosine(text, face)
        s_av = _cosine(audio, face)
        tecs = (s_ta + s_tv + s_av) / 3.0
        step["tecs"] = float(tecs)

        v_text = _valence(text)
        v_audio = _valence(audio)
        v_face = _valence(face)
        step["valence"] = {"text": v_text, "audio": v_audio, "face": v_face}

        I_text = _intensity(text)
        I_audio = _intensity(audio)
        I_face = _intensity(face)
        I_base = max(I_text, I_audio, I_face)
        step["intensity"] = float(I_base)

        client_speaking = bool(step.get("client_speaking", False))
        # Incongruence rule
        def _sign(x: float) -> int:
            return 1 if x > 1e-6 else (-1 if x < -1e-6 else 0)

        valence_disagree = (
            (_sign(v_text) != _sign(v_audio))
            or (_sign(v_text) != _sign(v_face))
            or (_sign(v_audio) != _sign(v_face))
        )
        low_sim = tecs < 0.6
        high_intensity = I_base > 0.2
        step["is_incongruent"] = bool((valence_disagree or low_sim) and high_intensity and client_speaking)

    # Optional smoothing over 'is_incongruent'
    raw_flags: List[Optional[bool]] = [bool(e.get("is_incongruent")) for e in timeline_10hz]
    smoothed = _majority_smooth_bool(raw_flags, window_radius=3)
    for e, sm in zip(timeline_10hz, smoothed):
        e["is_incongruent_smooth"] = sm
    return timeline_10hz


def build_session_summary(
    congruence_timeline: List[Dict[str, Any]],
    patient_id: str,
    session_id: int,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute MSW-TECS overall_congruence, list of incongruent intervals, and aggregate emotion distributions.
    """
    if not congruence_timeline:
        return {
            "patient_id": patient_id,
            "session_id": session_id,
            "duration": 0.0,
            "overall_congruence": 0.0,
            "incongruent_moments": [],
            "emotion_distribution": {"face": {}, "audio": {}, "text": {}},
            "metrics": {"avg_tecs": 0.0},
        }

    dt = 1.0 / 10.0

    # 1) MSW-TECS
    num = 0.0
    den = 0.0
    for step in congruence_timeline:
        tecs = float(step.get("tecs", 0.0))
        intensity = float(step.get("intensity", 0.0))
        micro_spike = 1.0 if step.get("micro_spike") else 0.0
        client_speaking = 1.0 if step.get("client_speaking") else 0.0
        w = intensity * (1.0 + 0.3 * micro_spike) * client_speaking
        num += w * tecs
        den += w
    overall_congruence = float(num / (den + 1e-8))

    # 2) Incongruent intervals
    moments: List[Dict[str, Any]] = []
    in_run = False
    start_idx = 0
    for i, e in enumerate(congruence_timeline):
        flag = bool(e.get("is_incongruent_smooth", e.get("is_incongruent", False)))
        if flag and not in_run:
            in_run = True
            start_idx = i
        elif not flag and in_run:
            in_run = False
            end_idx = i - 1
            if end_idx >= start_idx:
                start_t = float(congruence_timeline[start_idx].get("t", 0.0))
                end_t = float(congruence_timeline[end_idx].get("t", 0.0))
                if (end_t - start_t) >= 0.3:
                    seg_entries = congruence_timeline[start_idx : end_idx + 1]
                    face_vals = [float((x.get("valence") or {}).get("face", 0.0)) for x in seg_entries]
                    audio_vals = [float((x.get("valence") or {}).get("audio", 0.0)) for x in seg_entries]
                    text_vals = [float((x.get("valence") or {}).get("text", 0.0)) for x in seg_entries]
                    mean_face_val = sum(face_vals) / max(1, len(face_vals))
                    mean_audio_val = sum(audio_vals) / max(1, len(audio_vals))
                    mean_text_val = sum(text_vals) / max(1, len(text_vals))
                    mean_nontext_val = 0.5 * (mean_face_val + mean_audio_val)
                    # Build text snippet for reason
                    snippet = ""
                    if transcript_segments:
                        parts: List[str] = []
                        for seg in transcript_segments:
                            try:
                                s = float(seg.get("start", 0.0))
                                e2 = float(seg.get("end", 0.0))
                                txt = str(seg.get("text", "")).strip()
                            except Exception:
                                continue
                            if txt and not (e2 <= start_t or s >= end_t):
                                parts.append(txt)
                        snippet = " ".join(parts)[:400].strip()
                    reason_llm = generate_incongruence_reason(
                        text_snippet=snippet,
                        metrics={
                            "mean_text_valence": mean_text_val,
                            "mean_nontext_valence": mean_nontext_val,
                            "mean_face_valence": mean_face_val,
                            "mean_audio_valence": mean_audio_val,
                            "mean_text_arousal": 0.0,
                            "mean_nontext_arousal": 0.0,
                        },
                    )
                    reason = reason_llm or "valence/arousal mismatch"
                    moments.append({"start": round(start_t, 2), "end": round(end_t, 2), "reason": reason})
    if in_run and congruence_timeline:
        start_t = float(congruence_timeline[start_idx].get("t", 0.0))
        end_t = float(congruence_timeline[-1].get("t", 0.0))
        if (end_t - start_t) >= 0.3:
            seg_entries = congruence_timeline[start_idx:]
            face_vals = [float((x.get("valence") or {}).get("face", 0.0)) for x in seg_entries]
            audio_vals = [float((x.get("valence") or {}).get("audio", 0.0)) for x in seg_entries]
            text_vals = [float((x.get("valence") or {}).get("text", 0.0)) for x in seg_entries]
            mean_face_val = sum(face_vals) / max(1, len(face_vals))
            mean_audio_val = sum(audio_vals) / max(1, len(audio_vals))
            mean_text_val = sum(text_vals) / max(1, len(text_vals))
            mean_nontext_val = 0.5 * (mean_face_val + mean_audio_val)
            snippet = ""
            if transcript_segments:
                parts = []
                for seg in transcript_segments:
                    try:
                        s = float(seg.get("start", 0.0))
                        e2 = float(seg.get("end", 0.0))
                        txt = str(seg.get("text", "")).strip()
                    except Exception:
                        continue
                    if txt and not (e2 <= start_t or s >= end_t):
                        parts.append(txt)
                snippet = " ".join(parts)[:400].strip()
            reason_llm = generate_incongruence_reason(
                text_snippet=snippet,
                metrics={
                    "mean_text_valence": mean_text_val,
                    "mean_nontext_valence": mean_nontext_val,
                    "mean_face_valence": mean_face_val,
                    "mean_audio_valence": mean_audio_val,
                    "mean_text_arousal": 0.0,
                    "mean_nontext_arousal": 0.0,
                },
            )
            reason = reason_llm or "valence/arousal mismatch"
            moments.append({"start": round(start_t, 2), "end": round(end_t, 2), "reason": reason})

    # 3) Aggregate distributions
    face_acc: Dict[str, float] = {}
    audio_acc: Dict[str, float] = {}
    text_acc: Dict[str, float] = {}
    for e in congruence_timeline:
        for k, v in (e.get("face", {}) or {}).items():
            face_acc[k] = face_acc.get(k, 0.0) + float(v)
        for k, v in (e.get("audio", {}) or {}).items():
            audio_acc[k] = audio_acc.get(k, 0.0) + float(v)
        tfield = e.get("text") or {}
        tdist = (tfield.get("emotion_distribution") if isinstance(tfield, dict) else None) or None
        if tdist:
            for k, v in tdist.items():
                text_acc[k] = text_acc.get(k, 0.0) + float(v)
    face_dist = _normalize_distribution(face_acc)
    audio_dist = _normalize_distribution(audio_acc)
    text_dist = _normalize_distribution(text_acc)

    duration = len(congruence_timeline) * dt
    # Legacy duration-based congruence for back-compat
    total_incong = 0.0
    for m in moments:
        total_incong += max(0.0, float(m.get("end", 0.0)) - float(m.get("start", 0.0)))
    legacy_congruence = float(max(0.0, min(1.0, 1.0 - (total_incong / (duration + 1e-8))))) if duration > 0 else 1.0

    return {
        "patient_id": patient_id,
        "session_id": session_id,
        "duration": round(duration, 2),
        "overall_congruence": round(overall_congruence, 4),
        "legacy_congruence": round(legacy_congruence, 4),
        "incongruent_moments": moments,
        "emotion_distribution": {
            "face": face_dist,
            "audio": audio_dist,
            "text": text_dist,
        },
        "metrics": {
            "avg_tecs": round(overall_congruence, 4),
            "num_incongruent_segments": len(moments),
        },
    }

def compute_emotional_congruence(
    merged_timeline: List[Dict[str, Any]],
    transcript_segments: Optional[List[Dict[str, Any]]],
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Convenience helper:
      - Builds a 10Hz tri-modal congruence timeline (without spikes consideration)
      - Computes session summary
      - Returns (overall_congruence, incongruent_moments, emotion_distribution)
    """
    timeline_10hz = build_congruence_timeline(
        merged_timeline=merged_timeline,
        transcript_segments=transcript_segments,
        spikes=None,
        target_hz=10.0,
    )
    summary = build_session_summary(
        congruence_timeline=timeline_10hz,
        patient_id="",
        session_id=0,
    )
    overall = float(summary.get("overall_congruence", 0.0))
    moments = summary.get("incongruent_moments", []) or []
    distributions = summary.get("emotion_distribution", {}) or {}
    return overall, moments, distributions
