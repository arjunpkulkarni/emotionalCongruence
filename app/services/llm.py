import os
import json
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed


_DEFAULT_MODEL = "gpt-4o-mini"
_FAST_MODEL = "gpt-3.5-turbo"  # Faster, cheaper alternative for batch processing

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
    ensemble_size: int = 1,
    few_shot: bool = False,
    temperature: Optional[float] = None,
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
Your job is to map short transcript segments to numerical affect metrics and structured diarization.

PRIMARY TASK:
- Input: a short transcript segment from a therapy or conversation session. The snippet may contain one or more speakers (e.g., therapist and client).
- Output: a **valid JSON object** with the following structure:

1. A top-level key "emotion_distribution" containing a probability distribution across exactly these 7 emotions:
   ["joy","sadness","anger","fear","disgust","surprise","neutral"].
   - Each value must be a float in [0,1].
   - The probabilities must sum to 1 (± floating point rounding).

2. If the snippet contains **multiple speakers** (explicit labels or inferred), include:
   "speakers": [
      {
        "speaker": "<label>",
        "text": "<portion attributed to this speaker>",
        "emotion_distribution": { seven emotions, summing to 1 },
        Optional: "valence": float in [-1,1],
                  "arousal": float in [0,1],
                  "style": "serious" | "joking" | "sarcastic" | "uncertain"
      },
      ...
   ]
   - Use explicit speaker labels if present (e.g., "Therapist:", "Client:").
   - If unlabeled, assign canonical labels: "SPEAKER_00", "SPEAKER_01", etc.

3. If only **one speaker** is present:
   - You may include "speaker": "<label>" at the top level.
   - Do NOT include a "speakers" array unless multiple speakers exist.

DIARIZATION RULES:
- Perform text-based diarization.
- Split turns correctly.
- Trim leading/trailing whitespace.
- Keep each speaker's text exactly as it appears (minus label prefixes).

EMOTION ESTIMATION:
- Focus on the felt emotional content, not just explicit emotion words.
- If the snippet is emotionally flat, assign higher "neutral".
- If mixed emotions appear, distribute probability instead of forcing one label.
- The 7 emotions **must always appear** with probability values.

OPTIONAL METRICS:
You may include these at the top level or per speaker:
- "valence": float ∈ [-1,1]
- "arousal": float ∈ [0,1]
- "style": one of "serious", "joking", "sarcastic", "uncertain"

STRICT OUTPUT RULES:
- **Always return valid JSON.**
- **No extra text or explanations.**
- **Do not include any keys not described above.**
- **Do not include reasoning or chain-of-thought.**
- The final output must be a pure JSON object.

CONSTRAINTS FOR CONSISTENCY:
- Use stable, conservative probability assignments.
- Avoid speculation; rely only on linguistic cues.
- Ensure diarization is consistent across similar inputs.

Output format examples (structure only):
Single speaker:
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
  "style": "serious",
  "speaker": "SPEAKER_00"
}

Multiple speakers:
{
  "emotion_distribution": {
    "joy": 0.12,
    "sadness": 0.18,
    "anger": 0.06,
    "fear": 0.09,
    "disgust": 0.03,
    "surprise": 0.04,
    "neutral": 0.48
  },
  "speakers": [
    {
      "speaker": "Therapist",
      "text": "How have you been sleeping?",
      "emotion_distribution": {
        "joy": 0.08,
        "sadness": 0.10,
        "anger": 0.02,
        "fear": 0.05,
        "disgust": 0.01,
        "surprise": 0.02,
        "neutral": 0.72
      },
      "valence": 0.0,
      "arousal": 0.2,
      "style": "serious"
    },
    {
      "speaker": "Client",
      "text": "Not great; I wake up a lot at night.",
      "emotion_distribution": {
        "joy": 0.03,
        "sadness": 0.25,
        "anger": 0.05,
        "fear": 0.20,
        "disgust": 0.02,
        "surprise": 0.03,
        "neutral": 0.42
      },
      "valence": -0.25,
      "arousal": 0.4,
      "style": "serious"
    }
  ]
}
    """.strip()

    if instruction:
        sys_msg += f"\nAdditional instruction: {instruction}"

    messages = [{"role": "system", "content": sys_msg}]
    
    try:
        messages_call = list(messages) + [{"role": "user", "content": text[:4000]}]
        use_ensemble = max(1, int(ensemble_size or 1))
        temp = (
            float(temperature)
            if temperature is not None
            else (0.2 if use_ensemble == 1 else 0.7)
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages_call,
            n=use_ensemble,
            temperature=temp,
            response_format={"type": "json_object"},
        )

        # Aggregate over choices (self-consistency)
        agg_emotions: Dict[str, float] = {}
        valence_sum = 0.0
        arousal_sum = 0.0
        styles: Dict[str, int] = {}
        valid = 0
        first_reason: Optional[str] = None
        first_speaker: Optional[str] = None
        first_speakers_block: Optional[Any] = None
        for choice in (resp.choices or []):
            content = getattr(choice.message, "content", None)
            if not content:
                continue
            try:
                parsed = json.loads(content)
            except Exception:
                continue
            valence = float(parsed.get("valence", 0.0))
            arousal = float(parsed.get("arousal", 0.0))
            raw_emotions = parsed.get("emotion_distribution", {}) or parsed.get("emotions", {}) or {}
            emotions = _normalize_emotion_distribution(raw_emotions) if raw_emotions else {"neutral": 1.0}
            for k, v in emotions.items():
                agg_emotions[k] = agg_emotions.get(k, 0.0) + float(v)
            style_val = str(parsed.get("style", "") or "").strip().lower()
            if style_val in {"serious", "joking", "sarcastic", "uncertain"}:
                styles[style_val] = styles.get(style_val, 0) + 1
            valence_sum += valence
            arousal_sum += arousal
            valid += 1
            if first_reason is None:
                r = parsed.get("reason") or parsed.get("rationale")
                if isinstance(r, str) and r.strip():
                    first_reason = r.strip()
            if first_speaker is None:
                s = parsed.get("speaker")
                if isinstance(s, str) and s.strip():
                    first_speaker = s.strip()
            if first_speakers_block is None and isinstance(parsed.get("speakers"), list):
                first_speakers_block = parsed.get("speakers")

        if valid == 0:
            return None

        # Average and normalize
        for k in list(agg_emotions.keys() or []):
            agg_emotions[k] /= float(valid)
        norm_emotions = _normalize_emotion_distribution(agg_emotions)
        avg_valence = max(-1.0, min(1.0, valence_sum / float(valid)))
        avg_arousal = max(0.0, min(1.0, arousal_sum / float(valid)))
        style = "uncertain"
        if styles:
            style = max(styles.items(), key=lambda kv: kv[1])[0]

        result: Dict[str, Any] = {
            "valence": avg_valence,
            "arousal": avg_arousal,
            "emotion_distribution": {str(k): float(v) for k, v in norm_emotions.items()},
            "style": style,
        }
        if first_reason:
            result["reason"] = first_reason
        if first_speaker:
            result["speaker"] = first_speaker
        if first_speakers_block is not None:
            result["speakers"] = first_speakers_block
        return result
    except Exception:
        return None


def batch_analyze_text_emotions(
    texts: List[str],
    model: Optional[str] = None,
    max_workers: int = 10,
    temperature: Optional[float] = 0.2,
    use_fast_model: bool = True,
) -> List[Optional[Dict[str, Any]]]:
    """
    Analyze multiple text segments in parallel using thread pool.
    Returns list of analysis results in same order as input texts.
    
    Args:
        texts: List of text strings to analyze
        model: OpenAI model to use (if None, uses fast model if use_fast_model=True)
        max_workers: Maximum number of parallel threads
        temperature: LLM temperature
        use_fast_model: Use faster/cheaper model for batch processing (default: True)
    
    Returns:
        List of analysis dicts (same format as analyze_text_emotion_with_llm)
    """
    # Use fast model by default for batch processing unless specific model requested
    if model is None and use_fast_model:
        model = _FAST_MODEL
    if not texts:
        return []
    
    # Filter out empty texts but preserve indices
    text_indices = [(i, text) for i, text in enumerate(texts) if text.strip()]
    
    if not text_indices:
        return [None] * len(texts)
    
    results = [None] * len(texts)
    
    def analyze_single(idx_text_tuple):
        idx, text = idx_text_tuple
        return idx, analyze_text_emotion_with_llm(
            text=text,
            model=model,
            ensemble_size=1,  # Disable ensemble for speed
            temperature=temperature,
        )
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single, it): it for it in text_indices}
        
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception:
                # Keep None for failed analyses
                pass
    
    return results


def batch_generate_incongruence_reasons(
    snippets_and_metrics: List[tuple[str, Dict[str, Any]]],
    model: Optional[str] = None,
    max_workers: int = 10,
    use_fast_model: bool = True,
) -> List[Optional[str]]:
    """
    Generate incongruence reasons for multiple segments in parallel.
    
    Args:
        snippets_and_metrics: List of (text_snippet, metrics_dict) tuples
        model: OpenAI model to use (if None, uses fast model if use_fast_model=True)
        max_workers: Maximum number of parallel threads
        use_fast_model: Use faster/cheaper model for batch processing (default: True)
    
    Returns:
        List of reason strings in same order as input
    """
    # Use fast model by default for batch processing unless specific model requested
    if model is None and use_fast_model:
        model = _FAST_MODEL
    if not snippets_and_metrics:
        return []
    
    results = [None] * len(snippets_and_metrics)
    
    def generate_single(idx_data):
        idx, (snippet, metrics) = idx_data
        return idx, generate_incongruence_reason(
            text_snippet=snippet,
            metrics=metrics,
            model=model,
        )
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        indexed_data = list(enumerate(snippets_and_metrics))
        futures = {executor.submit(generate_single, item): item for item in indexed_data}
        
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception:
                # Keep None for failed generations
                pass
    
    return results


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
        "You are a clinical communication analyst. Given a transcript snippet and numeric affect metrics, "
        "explain WHY the speaker's verbal content (text) and non-verbal signals (face/audio) are incongruent.\n"
        "\n"
        "Write like a careful detective:\n"
        "- Explicitly compare modalities: state which ones disagree (text vs face, text vs audio, face vs audio).\n"
        "- Use the provided mean valences and cite them with signs and 3 decimals (e.g., text_v: +0.245, face_v: -0.512).\n"
        "- If a transcript is available, quote a short fragment (≤8 words) that most reflects the verbal tone.\n"
        "- Prefer concrete observations over speculation; do not invent cues.\n"
        "- Keep a neutral, precise tone. Avoid generic phrasing.\n"
        "- 1–2 sentences only.\n"
        "- End with a compact bracket summarizing metrics, and include the time range if provided in metrics as 'start'/'end': "
        "[t: <start>–<end> s; text_v: <x.xxx>; face_v: <y.yyy>; audio_v: <z.zzz>]."
    )

    user_msg = (
        f"Transcript snippet:\n\"\"\"\n{snippet_use}\n\"\"\"\n\n"
        f"Metrics (JSON): {metrics_json}\n\n"
        "Task: Provide a single concise detective-style reason (1–2 sentences) comparing modalities explicitly, "
        "citing numeric mean valences with 3 decimals and ending with the bracket summary. "
        "If insufficient information, respond exactly with: \"Insufficient information.\""
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
