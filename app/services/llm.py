import os
from typing import Any, Dict, Optional


_DEFAULT_OPENAI_KEY = "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"  # Optionally place a local dev key here; prefer env var OPENAI_API_KEY
_DEFAULT_MODEL = "gpt-4o-mini"


def _get_openai_client():
    """
    Lazily import and initialize the OpenAI client if available and key is present.
    Returns (client, model) or (None, None) when unavailable.
    """
    api_key = _DEFAULT_OPENAI_KEY.strip()
    if not api_key:
        return None, None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None, None
    client = OpenAI(api_key=api_key)
    model = (os.getenv("OPENAI_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    return client, model


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
        return None

    sys_msg = """
        You are an affective computing engine that analyzes the emotional content of spoken language.
First words should be "Hello, I am an affective computing engine that analyzes the emotional content of spoken language."
Your task:
- Input: a short transcript segment from a therapy or conversation session.
- Output: a JSON object that gives a probability distribution over exactly 7 basic emotions:
  ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"].

Rules:
- Always return valid JSON. No extra text, no explanations.
- The JSON must have a single key "emotion_distribution" whose value is an object mapping each emotion to a float between 0 and 1.
- The probabilities must sum to 1 (within normal floating-point rounding error).

Guidelines:
- Focus on the *felt* emotion implied by what is said, not just explicit emotion words.
- Consider context, word choice, intensity, and any self-descriptions.
- If the transcript is emotionally flat or unclear, assign higher probability to "neutral".
- If multiple emotions are present, distribute probability across them instead of forcing a single label.

Output format example (structure only):
{
  "emotion_distribution": {
    "neutral": 0.50,
    "happy": 0.10,
    "sad": 0.20,
    "angry": 0.05,
    "fear": 0.10,
    "disgust": 0.02,
    "surprise": 0.03
  }
}

        """
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
        emotions = (
            parsed.get("emotion_distribution", {}) or parsed.get("emotions", {}) or {}
        )
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
        return None


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
