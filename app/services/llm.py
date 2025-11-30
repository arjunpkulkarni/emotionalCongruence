import os
import json
from typing import Any, Dict, Optional


_DEFAULT_MODEL = "gpt-4o-mini"

CANONICAL_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "neutral",
]


def _normalize_emotion_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a possibly partial distribution to the canonical 7-emotion basis.
    Unknown/missing emotions -> 0, soft-normalized to sum to 1.
    Accepts common variants from other libs/LLMs.
    """
    norm: Dict[str, float] = {e: 0.0 for e in CANONICAL_EMOTIONS}
    mapping = {
        "joy": "joy",
        "happy": "joy",
        "happiness": "joy",
        "sadness": "sadness",
        "sad": "sadness",
        "anger": "anger",
        "angry": "anger",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
        "neutral": "neutral",
    }
    for k, v in (dist or {}).items():
        k_low = str(k).lower()
        mapped = mapping.get(k_low)
        if mapped in norm:
            norm[mapped] += float(v)
    s = sum(norm.values()) or 1.0
    for k in norm:
        norm[k] /= s
    return norm


def _get_openai_client():
    """
    Lazily import and initialize the OpenAI client if available and key is present.
    Returns (client, model) or (None, None) when unavailable.
    """
    api_key = "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"
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
) -> Optional[Dict[str, Any]]:
    """
    Ask an LLM to estimate:
      - emotion_distribution over 7 emotions
      - valence in [-1, 1]
      - arousal in [0, 1]
      - style label

    Returns:
      {
        "emotion_distribution": {emotion: prob, ...},  # canonical 7D
        "valence": float,
        "arousal": float,
        "style": str,
      }
    or None if unavailable.
    """
    client, default_model = _get_openai_client()
    if model is None:
        model = default_model

    if client is None or model is None or not text.strip():
        return None

    sys_msg = """
You are an affective computing engine that analyzes the emotional content of spoken language.
Your task:
- Input: a short transcript segment from a therapy or conversation session.
- Output: a JSON object that gives a probability distribution over exactly 7 basic emotions:
  ["joy","sadness","anger","fear","disgust","surprise","neutral"].

Rules:
- Always return valid JSON. No extra text, no explanations.
- The JSON must have a single key "emotion_distribution" whose value is an object mapping each of the 7 emotions to a float in [0,1].
- The probabilities must sum to 1 (within normal floating-point rounding error).
- Optionally include keys "valence" (in [-1,1]), "arousal" (in [0,1]), and "style" (one of: serious, joking, sarcastic, uncertain).

Guidelines:
- Focus on the felt emotion implied by what is said, not only explicit emotion words.
- If the transcript is emotionally flat or unclear, assign higher probability to "neutral".
- If multiple emotions are present, distribute probability across them instead of forcing a single label.

Output format example (structure only):
{
  "emotion_distribution": {
    "joy": 0.10,
    "sadness": 0.20,
    "anger": 0.05,
    "fear": 0.10,
    "disgust": 0.02,
    "surprise": 0.03,
    "neutral": 0.50
  },
  "valence": 0.0,
  "arousal": 0.3,
  "style": "serious"
}
    """.strip()

    if instruction:
        sys_msg += f"\nAdditional instruction: {instruction}"

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
        parsed = json.loads(content)

        # Extract + clamp valence/arousal
        valence = float(parsed.get("valence", 0.0))
        arousal = float(parsed.get("arousal", 0.0))

        raw_emotions = parsed.get("emotion_distribution", {}) or parsed.get("emotions", {}) or {}
        emotions = _normalize_emotion_distribution(raw_emotions) if raw_emotions else {"neutral": 1.0}

        style = str(parsed.get("style", "") or "").strip().lower()
        if style not in {"serious", "joking", "sarcastic", "uncertain"}:
            style = "uncertain"

        return {
            "valence": max(-1.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "emotion_distribution": {str(k): float(v) for k, v in emotions.items()},
            "style": style,
        }
    except Exception:
        return None


def generate_incongruence_reason(
    text_snippet: str,
    metrics: Dict[str, Any],
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Ask an LLM for a concise, plain-English reason describing possible
    incongruence between verbal content (text) and non-verbal signals (face/audio),
    using the provided metrics as context.
    Returns a short string (1–2 sentences), or None if unavailable.
    """
    client, default_model = _get_openai_client()
    if model is None:
        model = default_model
    if client is None or model is None:
        return None

    # Keep payloads bounded
    snippet_use = (text_snippet or "").strip()[:600]
    try:
        metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
    except Exception:
        metrics_json = "{}"

    system_msg = (
        "You are a clinical communication analyst. Given a transcript snippet and "
        "numeric affect metrics, provide a brief explanation of why the speaker's "
        "verbal content and non-verbal signals may be incongruent.\n"
        "- Be precise and neutral in tone.\n"
        "- Prefer concrete cues over speculation.\n"
        "- 1–2 sentences only."
    )

    user_msg = (
        f"Transcript snippet:\n\"\"\"\n{snippet_use}\n\"\"\"\n\n"
        f"Metrics (JSON): {metrics_json}\n\n"
        "Write a single concise reason (1–2 sentences). If insufficient information, say: "
        "\"Insufficient information.\""
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or None
    except Exception:
        return None
