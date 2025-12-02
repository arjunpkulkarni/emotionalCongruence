from typing import Any, Dict, List, Optional, Tuple


def transcribe_audio_with_faster_whisper(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = "en",
    fast_mode: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (full_text, segments). Each segment has:
      { "start": float, "end": float, "text": str }
    If faster-whisper is not installed or fails, returns ("", []).
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code
        fast_mode: Use faster settings (lower beam_size, enable VAD) for 2-3x speedup
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return "", []

    try:
        # Use base model in fast mode for better speed/accuracy tradeoff
        if fast_mode and model_size == "small":
            model_size = "base"
        
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        
        # Fast mode: lower beam size and enable VAD for faster transcription
        beam_size = 1 if fast_mode else 5
        vad_filter = fast_mode  # VAD reduces processing for silent portions
        
        segments_iter, _info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
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

