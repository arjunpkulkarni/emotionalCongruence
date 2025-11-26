import os
from typing import Any, Dict, Optional


_DEFAULT_OPENAI_KEY = "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"
_DEFAULT_MODEL = "gpt-4o-mini"


def _get_openai_client():
    """
    Lazily import and initialize the OpenAI client if available and key is present.
    Returns (client, model) or (None, None) when unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY") or _DEFAULT_OPENAI_KEY
    if not api_key:
        return None, None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None, None
    client = OpenAI(api_key=api_key)
    return client, _DEFAULT_MODEL


def analyze_text_emotion_with_llm(
    text: str,
    model: Optional[str] = None,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask an LLM to estimate valence [-1, 1], arousal [0, 1], basic emotions, a style label, and a short rationale.
    Returns a structured dict. Falls back to a lightweight heuristic if the LLM is unavailable.
    """
    client, default_model = _get_openai_client()
    if model is None:
        model = default_model

    if client is None or model is None or not text.strip():
        return _heuristic_text_emotion(text)

    sys_msg = (
        "You are an affect analysis assistant. Given a short text excerpt, estimate:\n"
        "1) valence in [-1,1] negative to positive,\n"
        "2) arousal in [0,1] calm to excited,\n"
        "3) probabilities for basic emotions: neutral, happy, sad, angry, fear, disgust, surprise,\n"
        "4) conversational style: one of 'serious' | 'joking' | 'sarcastic' | 'uncertain'.\n"
        "Return a strict JSON object with keys:\n"
        "- valence (number)\n"
        "- arousal (number)\n"
        "- emotion_distribution (object mapping emotion -> probability)\n"
        "- style (string)\n"
        "- rationale (string)"
    )
    if instruction:
        sys_msg += f" Additional instruction: {instruction}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        import json

        parsed = json.loads(content)
        # Basic validation
        valence = float(parsed.get("valence", 0.0))
        arousal = float(parsed.get("arousal", 0.0))
        # Accept either 'emotion_distribution' or legacy 'emotions'
        emotions = parsed.get("emotion_distribution", {}) or parsed.get("emotions", {}) or {}
        style = str(parsed.get("style", "") or "").strip().lower()
        if style not in {"serious", "joking", "sarcastic", "uncertain"}:
            style = "uncertain"
        rationale = parsed.get("rationale", "")
        return {
            "valence": max(-1.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            # Keep both keys for compatibility
            "emotion_distribution": {str(k): float(v) for k, v in emotions.items()},
            "emotions": {str(k): float(v) for k, v in emotions.items()},
            "style": style,
            "rationale": rationale,
            "source": "llm",
        }
    except Exception:
        return _heuristic_text_emotion(text)


_POSITIVE_WORDS = {
    "good",
    "great",
    "happy",
    "glad",
    "love",
    "excited",
    "relieved",
    "calm",
    "fine",
    "okay",
    "ok",
}
_NEGATIVE_WORDS = {
    "bad",
    "sad",
    "angry",
    "mad",
    "upset",
    "anxious",
    "nervous",
    "scared",
    "afraid",
    "disgusted",
    "worried",
}


def _heuristic_text_emotion(text: str) -> Dict[str, Any]:
    """
    Lightweight fallback if LLM isn't available: bag-of-words valence, basic arousal,
    neutral-leaning emotion distribution, and simple style inference.
    """
    words = {w.strip(".,!?;:").lower() for w in text.split()} if text else set()
    pos_hits = len(words & _POSITIVE_WORDS)
    neg_hits = len(words & _NEGATIVE_WORDS)
    total = pos_hits + neg_hits
    if total == 0:
        valence = 0.0
    else:
        valence = (pos_hits - neg_hits) / total
        valence = max(-1.0, min(1.0, valence))
    emotions = {
        "neutral": 0.7 if total == 0 else max(0.0, 0.7 - 0.1 * total),
        "happy": max(0.0, 0.2 * pos_hits),
        "sad": max(0.0, 0.2 * neg_hits),
        "angry": 0.0,
        "fear": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
    }
    # Normalize
    s = sum(emotions.values()) or 1.0
    emotions = {k: float(v) / s for k, v in emotions.items()}
    # Naive style inference
    lower_text = (text or "").lower()
    style = "serious"
    if any(tok in lower_text for tok in ["lol", "haha", "joke", "joking", "kidding", "jk"]):
        style = "joking"
    elif any(tok in lower_text for tok in ["sarcasm", "sarcastic", "yeah right"]):
        style = "sarcastic"
    elif len(words) == 0:
        style = "uncertain"
    return {
        "valence": float(valence),
        "arousal": 0.3 if total == 0 else min(1.0, 0.3 + 0.1 * total),
        # Keep both keys for compatibility
        "emotion_distribution": emotions,
        "emotions": emotions,
        "style": style,
        "rationale": "Heuristic estimate based on word list.",
        "source": "heuristic",
    }


def generate_incongruence_reason(
    text_snippet: str,
    metrics: Dict[str, float],
    model: Optional[str] = None,
) -> str:
    """
    Use an LLM to craft a concise, clinically meaningful reason for an incongruent interval.
    Inputs:
      - text_snippet: concatenated transcript text overlapping the interval (may be empty)
      - metrics: dictionary with keys such as:
          mean_text_valence, mean_nontext_valence, mean_face_valence, mean_audio_valence,
          mean_text_arousal, mean_nontext_arousal
    Returns a short natural-language reason string.
    Falls back to a heuristic phrasing if LLM is unavailable.
    """
    client, default_model = _get_openai_client()
    if model is None:
        model = default_model

    def _fallback_reason() -> str:
        mtv = float(metrics.get("mean_text_valence", 0.0))
        mnv = float(metrics.get("mean_nontext_valence", 0.0))
        mfv = float(metrics.get("mean_face_valence", 0.0))
        mav = float(metrics.get("mean_audio_valence", 0.0))
        mna = float(metrics.get("mean_nontext_arousal", 0.0))
        if mtv >= 0.5 and mnv <= -0.5:
            return "Positive verbal content with negative affective tone."
        if mtv <= -0.5 and mnv >= 0.5:
            return "Negative verbal content with positive affective tone."
        if abs(mtv) <= 0.2 and mna >= 0.7:
            return "Neutral verbal content with heightened affective arousal."
        if abs(mfv - mav) >= 0.6:
            return "Facial and vocal affect diverge strongly."
        return "Valence/arousal misalignment across modalities."

    if client is None or model is None:
        return _fallback_reason()

    try:
        import json
        sys_msg = (
            "You are assisting with psychotherapy session analysis.\n"
            "Given a transcript snippet and affect metrics, produce one concise reason (<= 24 words)\n"
            "describing why the moment appears incongruent across face, voice, and text. Focus on clinically\n"
            "useful phrasing (e.g., 'smiles while describing distress', 'flat expression with excited tone',\n"
            "'joking tone dampening distress'). Return strict JSON: {\"reason\": string}."
        )
        user_payload = {
            "text_snippet": text_snippet[:800],
            "metrics": metrics,
        }
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        reason = str(parsed.get("reason", "")).strip()
        return reason or _fallback_reason()
    except Exception:
        return _fallback_reason()


