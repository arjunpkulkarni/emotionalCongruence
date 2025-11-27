import argparse
import json
import os
from typing import Any, Dict

from app.services.llm import (
    analyze_text_emotion_with_llm,
    generate_incongruence_reason,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM-only functions.")
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        default="I feel a bit anxious about tomorrow, but I'm trying to stay positive.",
        help="Transcript text to analyze.",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        required=False,
        default=json.dumps(
            {
                "mean_text_valence": 0.2,
                "mean_nontext_valence": -0.4,
                "mean_face_valence": -0.2,
                "mean_audio_valence": -0.6,
                "mean_text_arousal": 0.5,
                "mean_nontext_arousal": 0.7,
            }
        ),
        help="JSON string with metrics for generate_incongruence_reason.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Override model name for LLM calls.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=False,
        default=None,
        help="Optional extra instruction for text emotion analysis.",
    )
    args = parser.parse_args()

    # Inform user about API key presence
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))

    # 1) Analyze text emotion
    analysis = analyze_text_emotion_with_llm(
        text=args.text,
        model=args.model,
        instruction=args.instruction,
    )

    # 2) Generate incongruence reason
    try:
        metrics: Dict[str, Any] = json.loads(args.metrics_json)
    except Exception:
        metrics = {}
    reason = generate_incongruence_reason(
        text_snippet=args.text,
        metrics=metrics,
        model=args.model,
    )

    result = {
        "used_openai_api_key_env": api_key_set,
        "input_text": args.text,
        "analysis": analysis,
        "incongruence_reason": reason,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


