from typing import Any, Dict, List, Optional, Tuple
import math

from .llm import analyze_text_emotion_with_llm, generate_incongruence_reason


# Supported emotions and canonical order
EMOTION_ORDER: List[str] = ["happy", "neutral", "sad", "angry", "fear", "disgust", "surprise"]
EMOTION_KEYS: List[str] = EMOTION_ORDER[:]

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

# Instruction appended to the LLM system prompt to enforce a strict 7-emotion distribution
LLM_TEXT_EMOTION_INSTRUCTION: str = (
    "Return an emotion_distribution over EXACTLY these 7 emotions with probabilities that sum to 1.0: "
    "['happy','neutral','sad','angry','fear','disgust','surprise']. "
    "Do not include any other labels. Base your judgment ONLY on the semantic content of the transcript text "
    "(ignore acoustic or facial cues). Ensure emotion_distribution values are numbers in [0,1] and total to 1.0."
)


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


def _count_spikes_near_t(spikes: List[Dict[str, Any]], t: float, window: float = 0.2) -> int:
    count = 0
    for s in spikes:
        try:
            ts = float(s.get("t", 0.0))
        except Exception:
            continue
        if abs(ts - t) <= window:
            count += 1
    return count


def _analyze_transcript_segments(segments: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not segments:
        return []
    analyzed: List[Dict[str, Any]] = []
    for seg in segments:
        txt = str(seg.get("text", "")).strip()
        analysis = analyze_text_emotion_with_llm(txt, instruction=LLM_TEXT_EMOTION_INSTRUCTION) if txt else {
            "valence": 0.0,
            "arousal": 0.0,
            "emotion_distribution": {"neutral": 1.0},
            "style": "uncertain",
        }
        new_seg = dict(seg)
        new_seg["analysis"] = analysis
        analyzed.append(new_seg)
    return analyzed


def _compute_valence_arousal_from_probs(probs: Dict[str, float]) -> Tuple[float, float]:
    v = sum(VALENCE_TABLE.get(k, 0.0) * float(p) for k, p in probs.items())
    a = sum(AROUSAL_TABLE.get(k, 0.0) * float(p) for k, p in probs.items())
    return float(v), float(a)


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
    Construct a 10Hz timeline with tri-modal congruence:
    - face/audio: emotion distributions per 10Hz via interpolation
    - text: segment-level emotion distribution/valence/arousal/style attached to active time
    - per-step pairwise distances and congruence decision (with style-aware relaxation)
    """
    if not merged_timeline:
        return []
    face_by_t = _build_lookup_by_second(merged_timeline, "face")
    audio_by_t = _build_lookup_by_second(merged_timeline, "audio")
    max_t = max(int(e.get("t", 0)) for e in merged_timeline)
    duration = float(max_t)
    step = 1.0 / max(target_hz, 1.0)
    analyzed_segments = _analyze_transcript_segments(transcript_segments or [])
    spikes_list = spikes or []

    # Build timeline
    out: List[Dict[str, Any]] = []
    t = 0.0
    while t <= duration + 1e-9:
        tt = round(t, 2)
        face_probs, audio_probs = _sample_emotions_at_time(face_by_t, audio_by_t, t)
        face_probs = _ensure_full_probs(face_probs)
        audio_probs = _ensure_full_probs(audio_probs)
        face_val, face_aro = _compute_valence_arousal_from_probs(face_probs)
        audio_val, audio_aro = _compute_valence_arousal_from_probs(audio_probs)

        seg = _active_segment_at_t(analyzed_segments, t)
        text_probs: Optional[Dict[str, float]] = None
        text_val: Optional[float] = None
        text_aro: Optional[float] = None
        text_style: Optional[str] = None
        if seg and isinstance(seg.get("analysis"), dict):
            analysis = seg["analysis"]
            ed = analysis.get("emotion_distribution") or analysis.get("emotions") or {}
            text_probs = _ensure_full_probs({k: float(v) for k, v in (ed or {}).items()})
            # For consistency, recompute valence/arousal from distribution
            text_val, text_aro = _compute_valence_arousal_from_probs(text_probs)
            text_style = str(analysis.get("style", "") or "uncertain").lower()
            if text_style not in {"serious", "joking", "sarcastic", "uncertain"}:
                text_style = "uncertain"

        valid = text_probs is not None

        # Pairwise distances and congruence decision
        dv_ft = abs(face_val - text_val) if valid else None
        dv_at = abs(audio_val - text_val) if valid else None
        dv_fa = abs(face_val - audio_val)
        da_ft = abs(face_aro - text_aro) if valid else None
        da_at = abs(audio_aro - text_aro) if valid else None
        da_fa = abs(face_aro - audio_aro)

        m_ft = 0.7 * dv_ft + 0.3 * da_ft if valid and dv_ft is not None and da_ft is not None else None
        m_at = 0.7 * dv_at + 0.3 * da_at if valid and dv_at is not None and da_at is not None else None
        m_fa = 0.7 * dv_fa + 0.3 * da_fa
        mismatch_score = (
            (m_ft if m_ft is not None else 0.0)
            + (m_at if m_at is not None else 0.0)
            + (m_fa if m_fa is not None else 0.0)
        )
        denom = 3 if valid else 1  # only face-audio if text missing
        mismatch_score = float(mismatch_score / max(1, denom))

        # Threshold logic
        if valid:
            vt_face_text = VALENCE_THRESHOLD_TEXT_RELAXED if text_style in {"joking", "sarcastic"} else VALENCE_THRESHOLD_BASE
            pair_ft = (dv_ft is not None and dv_ft <= vt_face_text) and (da_ft is not None and da_ft <= AROUSAL_THRESHOLD_BASE)
            pair_at = (dv_at is not None and dv_at <= vt_face_text) and (da_at is not None and da_at <= AROUSAL_THRESHOLD_BASE)
            pair_fa = (dv_fa <= VALENCE_THRESHOLD_BASE) and (da_fa <= AROUSAL_THRESHOLD_BASE)
            num_congruent = sum(1 for p in [pair_ft, pair_at, pair_fa] if p)
            congruent = True if num_congruent >= 2 else False
        else:
            # If text is missing at this step, the sample is not valid for tri-modal congruence
            pair_ft = None
            pair_at = None
            pair_fa = (dv_fa <= VALENCE_THRESHOLD_BASE) and (da_fa <= AROUSAL_THRESHOLD_BASE)
            congruent = None

        entry: Dict[str, Any] = {
                "t": tt,
            "face": face_probs,
            "audio": audio_probs,
            "face_valence": round(face_val, 4),
            "face_arousal": round(face_aro, 4),
            "audio_valence": round(audio_val, 4),
            "audio_arousal": round(audio_aro, 4),
            "text": {
                "emotion_distribution": text_probs if text_probs is not None else None,
                "style": text_style,
            }
            if valid or text_style is not None
            else None,
            "text_valence": round(text_val, 4) if text_val is not None else None,
            "text_arousal": round(text_aro, 4) if text_aro is not None else None,
            "metrics": {
                "dv_face_text": dv_ft,
                "dv_audio_text": dv_at,
                "dv_face_audio": dv_fa,
                "da_face_text": da_ft,
                "da_audio_text": da_at,
                "da_face_audio": da_fa,
                "mismatch_score": round(mismatch_score, 4),
                "pair_congruent": {
                    "face_text": pair_ft,
                    "audio_text": pair_at,
                    "face_audio": pair_fa,
                },
            },
            "spikes": _count_spikes_near_t(spikes_list, tt, window=0.2),
            "congruent": congruent,
            "valid": bool(valid),
        }
        out.append(entry)
        t += step

    # Majority smoothing on congruent over ±3 frames for valid steps
    raw_flags: List[Optional[bool]] = [e["congruent"] if e.get("valid") else None for e in out]
    smoothed = _majority_smooth_bool(raw_flags, window_radius=3)
    for e, sm in zip(out, smoothed):
        e["congruent_smooth"] = sm
    return out


def _assign_reason(
    mean_text_val: float,
    mean_nontext_val: float,
    mean_face_val: float,
    mean_audio_val: float,
    mean_text_aro: float,
    mean_nontext_aro: float,
) -> str:
    # Heuristics per spec
    strongly_pos = 0.5
    strongly_neg = -0.5
    near_neutral = 0.2
    high_arousal = 0.7
    # text positive vs nontext negative
    if mean_text_val >= strongly_pos and mean_nontext_val <= strongly_neg:
        return "negative tone vs verbal content"
    # text negative vs nontext positive
    if mean_text_val <= strongly_neg and mean_nontext_val >= strongly_pos:
        return "positive tone vs verbal content"
    # text near-neutral but nontext high arousal
    if abs(mean_text_val) <= near_neutral and mean_nontext_aro >= high_arousal:
        return "high arousal vs neutral content"
    # face vs audio disagree strongly
    if abs(mean_face_val - mean_audio_val) >= 0.6:
        return "face vs audio mismatch"
    # default
    return "valence/arousal mismatch"


def build_session_summary(
    congruence_timeline: List[Dict[str, Any]],
    patient_id: str,
    session_id: int,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not congruence_timeline:
        return {
            "patient_id": patient_id,
            "session_id": session_id,
            "duration": 0.0,
            "overall_congruence": 0.0,
            "incongruent_moments": [],
            "emotion_distribution": {"face": {}, "audio": {}, "text": {}},
        }

    duration = float(congruence_timeline[-1].get("t", 0.0))
    dt = 0.1

    # Accumulate distributions and identify incongruent intervals over valid steps
    face_acc: Dict[str, float] = {}
    audio_acc: Dict[str, float] = {}
    text_acc: Dict[str, float] = {}
    valid_entries: List[Dict[str, Any]] = [e for e in congruence_timeline if e.get("valid")]

    # Interval extraction over smoothed flags
    moments: List[Dict[str, Any]] = []
    in_run = False
    start_idx = 0
    for i, e in enumerate(valid_entries):
        sm = e.get("congruent_smooth")
        if sm is False and not in_run:
            in_run = True
            start_idx = i
        elif (sm is True or sm is None) and in_run:
            in_run = False
            end_idx = i - 1
            if end_idx >= start_idx:
                start_t = float(valid_entries[start_idx]["t"])
                end_t = float(valid_entries[end_idx]["t"])
                if (end_t - start_t) >= 0.3:
                    # Compute means for reason assignment
                    seg_entries = valid_entries[start_idx : end_idx + 1]
                    face_vals = [float(x.get("face_valence", 0.0) or 0.0) for x in seg_entries]
                    audio_vals = [float(x.get("audio_valence", 0.0) or 0.0) for x in seg_entries]
                    text_vals = [float(x.get("text_valence", 0.0) or 0.0) for x in seg_entries]
                    face_aros = [float(x.get("face_arousal", 0.0) or 0.0) for x in seg_entries]
                    audio_aros = [float(x.get("audio_arousal", 0.0) or 0.0) for x in seg_entries]
                    text_aros = [float(x.get("text_arousal", 0.0) or 0.0) for x in seg_entries]
                    mean_face_val = sum(face_vals) / max(1, len(face_vals))
                    mean_audio_val = sum(audio_vals) / max(1, len(audio_vals))
                    mean_text_val = sum(text_vals) / max(1, len(text_vals))
                    mean_nontext_val = 0.5 * (mean_face_val + mean_audio_val)
                    mean_text_aro = sum(text_aros) / max(1, len(text_aros))
                    mean_nontext_aro = 0.5 * (
                        (sum(face_aros) / max(1, len(face_aros)))
                        + (sum(audio_aros) / max(1, len(audio_aros)))
                    )
                    # Build text snippet from overlapping transcript segments, if available
                    snippet = ""
                    if transcript_segments:
                        parts: List[str] = []
                        for seg in transcript_segments:
                            try:
                                s = float(seg.get("start", 0.0))
                                e = float(seg.get("end", 0.0))
                                txt = str(seg.get("text", "")).strip()
                            except Exception:
                                continue
                            if not txt:
                                continue
                            if not (e <= start_t or s >= end_t):
                                parts.append(txt)
                        snippet = " ".join(parts)[:400].strip()
                    # LLM-crafted reason with heuristic fallback
                    reason_llm = generate_incongruence_reason(
                        text_snippet=snippet,
                        metrics={
                            "mean_text_valence": mean_text_val,
                            "mean_nontext_valence": mean_nontext_val,
                            "mean_face_valence": mean_face_val,
                            "mean_audio_valence": mean_audio_val,
                            "mean_text_arousal": mean_text_aro,
                            "mean_nontext_arousal": mean_nontext_aro,
                        },
                    )
                    reason = reason_llm or _assign_reason(
                        mean_text_val=mean_text_val,
                        mean_nontext_val=mean_nontext_val,
                        mean_face_val=mean_face_val,
                        mean_audio_val=mean_audio_val,
                        mean_text_aro=mean_text_aro,
                        mean_nontext_aro=mean_nontext_aro,
                    )
                    moments.append({"start": round(start_t, 2), "end": round(end_t, 2), "reason": reason})
    # If run goes till end
    if in_run and valid_entries:
        start_t = float(valid_entries[start_idx]["t"])
        end_t = float(valid_entries[-1]["t"])
        if (end_t - start_t) >= 0.3:
            seg_entries = valid_entries[start_idx:]
            face_vals = [float(x.get("face_valence", 0.0) or 0.0) for x in seg_entries]
            audio_vals = [float(x.get("audio_valence", 0.0) or 0.0) for x in seg_entries]
            text_vals = [float(x.get("text_valence", 0.0) or 0.0) for x in seg_entries]
            face_aros = [float(x.get("face_arousal", 0.0) or 0.0) for x in seg_entries]
            audio_aros = [float(x.get("audio_arousal", 0.0) or 0.0) for x in seg_entries]
            text_aros = [float(x.get("text_arousal", 0.0) or 0.0) for x in seg_entries]
            mean_face_val = sum(face_vals) / max(1, len(face_vals))
            mean_audio_val = sum(audio_vals) / max(1, len(audio_vals))
            mean_text_val = sum(text_vals) / max(1, len(text_vals))
            mean_nontext_val = 0.5 * (mean_face_val + mean_audio_val)
            mean_text_aro = sum(text_aros) / max(1, len(text_aros))
            mean_nontext_aro = 0.5 * (
                (sum(face_aros) / max(1, len(face_aros))) + (sum(audio_aros) / max(1, len(audio_aros)))
            )
            snippet = ""
            if transcript_segments:
                parts = []
                for seg in transcript_segments:
                    try:
                        s = float(seg.get("start", 0.0))
                        e = float(seg.get("end", 0.0))
                        txt = str(seg.get("text", "")).strip()
                    except Exception:
                        continue
                    if not txt:
                        continue
                    if not (e <= start_t or s >= end_t):
                        parts.append(txt)
                snippet = " ".join(parts)[:400].strip()
            reason_llm = generate_incongruence_reason(
                text_snippet=snippet,
                metrics={
                    "mean_text_valence": mean_text_val,
                    "mean_nontext_valence": mean_nontext_val,
                    "mean_face_valence": mean_face_val,
                    "mean_audio_valence": mean_audio_val,
                    "mean_text_arousal": mean_text_aro,
                    "mean_nontext_arousal": mean_nontext_aro,
                },
            )
            reason = reason_llm or _assign_reason(
                mean_text_val=mean_text_val,
                mean_nontext_val=mean_nontext_val,
                mean_face_val=mean_face_val,
                mean_audio_val=mean_audio_val,
                mean_text_aro=mean_text_aro,
                mean_nontext_aro=mean_nontext_aro,
            )
            moments.append({"start": round(start_t, 2), "end": round(end_t, 2), "reason": reason})

    # Distributions and totals over valid steps
    for e in valid_entries:
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

    T_total = len(valid_entries) * dt
    T_incongruent = sum(max(0.0, float(m.get("end", 0.0)) - float(m.get("start", 0.0))) for m in moments)
    if T_total <= 0.0:
        overall = 1.0
    else:
        overall = max(0.0, min(1.0, 1.0 - (T_incongruent / T_total)))

    return {
        "patient_id": patient_id,
        "session_id": session_id,
        "duration": round(duration, 2),
        "overall_congruence": round(float(overall), 4),
        "incongruent_moments": moments,
        "emotion_distribution": {
            "face": face_dist,
            "audio": audio_dist,
            "text": text_dist,
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
