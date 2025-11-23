from typing import Any, Dict, List, Optional, Tuple


def transcribe_audio_with_faster_whisper(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (full_text, segments). Each segment has:
      { "start": float, "end": float, "text": str }
    If faster-whisper is not installed or fails, returns ("", []).
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return "", []

    try:
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        segments_iter, _info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
        )
        segments_list: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        for seg in segments_iter:
            segments_list.append(
                {"start": float(seg.start), "end": float(seg.end), "text": seg.text}
            )
            if seg.text:
                full_text_parts.append(seg.text.strip())
        full_text = " ".join(full_text_parts).strip()
        return full_text, segments_list
    except Exception:
        return "", []


