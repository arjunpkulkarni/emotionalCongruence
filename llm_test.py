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
        default="""THERAPIST: I appreciate you coming in today. How have you been since our last session?
CLIENT: Uh, busy mostly. Work’s been… a lot. I’m kind of excited, actually. We’re launching this new initiative, and everyone’s looking at me to lead it.
THERAPIST: You say you’re excited. I’m also noticing you wringing your hands and your breathing is a little shallow. What’s happening as you talk about the launch?
CLIENT: Oh, that? It’s nothing. I’m fine. I’m totally fine. I mean, I’m thrilled—this is what I wanted, right? It’s a big opportunity. I’m ready.
THERAPIST: Ready—and also perhaps a bit tense?
CLIENT: Maybe. I mean, I’ve been waking up at 3 AM, heart racing. But that’s just adrenaline. My team keeps asking for clarity, and I keep saying, “We’ll figure it out as we go,” you know? It’s fine.
THERAPIST: What do you notice in your body when you say, “It’s fine”?
CLIENT: Hah, well, my jaw is tight, and my shoulders feel… stuck. But honestly, it’s nothing to worry about. I just need to push through. That’s what I do. I push.
THERAPIST: Pushing through has worked for you before. Is there any part of you that isn’t convinced this time?
CLIENT: Maybe a small part. I keep having this thought like… “What if they realize I’m not as capable as they think?” But that’s ridiculous. I’m capable. I’ve done harder things. I’m smiling because it’s funny how dramatic that sounds, right?
THERAPIST: I hear the words “I’m capable,” and I also hear a fear of being found out. As you smile, what are you feeling?
CLIENT: Embarrassed, I guess. And… scared. I don’t want to mess it up. My boss keeps saying, “We believe in you,” and I nod and say, “Absolutely,” but inside I’m thinking, “Please don’t look too closely.”
THERAPIST: That sounds like a lot to hold. When the fear shows up, how do you tend to respond?
CLIENT: I get louder. I get cheerful. I send confident emails with lots of exclamation points. I joke in meetings so nobody notices my hands are shaking under the table. I tell myself I’m excited. It’s like I try to drown out the doubt with positivity.
THERAPIST: Almost like a volume knob—turning up excitement to drown out fear.
CLIENT: Exactly. And at home, I’m exhausted. My partner asked if I’m okay, and I laughed and said, “Totally!” Then I went to the bathroom, locked the door, and just stared at the mirror for five minutes trying to breathe.
THERAPIST: What did you see in the mirror?
CLIENT: Someone who’s pretending to be fearless. Someone who’s clenching their teeth so hard their temples hurt. But also… someone who really wants to do well and doesn’t want to disappoint anyone.
THERAPIST: There’s a lot of care there—wanting to do well, not disappoint. What would it be like to let the fear be present without having to cover it with excitement?
CLIENT: Honestly? It feels risky. Like if I let it in, it will swallow me. If I keep it upbeat, people keep trusting me. If I let the fear show, I’m afraid they’ll lose faith. So I put on the bright voice and the big smile. “We’ve got this!” You know?
THERAPIST: I’m hearing a belief that showing fear means losing others’ trust. What evidence have you seen for and against that belief?
CLIENT: For… well, I don’t know. My old manager once told me I needed to project confidence even when I wasn’t sure. Against… my current team actually appreciates honesty. Last week I admitted I needed more time to think through a dependency, and they were… kind. They asked how they could help.
THERAPIST: How did your body respond in that moment?
CLIENT: Softer. My chest loosened a bit. I still felt anxious, but it wasn’t crushing. I didn’t have to hold my breath to get through the meeting.
THERAPIST: If we bring that memory into the room—the kindness, the softening—what happens right now?
CLIENT: My shoulders drop a little. I feel my feet. I notice I’m not smiling as hard.
THERAPIST: And as you sit with that, what would you want to say to the part of you that feels it must be “on,” cheerful, certain?
CLIENT: I guess I’d say… “You don’t have to do this alone. You don’t have to be so loud. We can tell the truth and still be okay.”
THERAPIST: What’s it like to hear yourself say that?
CLIENT: Calming. Scary, but calming. Part of me still wants to say, “It’s fine, everything’s fine!”—like a reflex. But another part’s like, “Maybe we can be real.”
THERAPIST: If being real were safe enough, how might your voice and face change at work?
CLIENT: I think I’d speak a little slower. I’d breathe. I’d say, “I’m unsure about X, here’s what I do know, and here’s what I need from you.” And I wouldn’t paste on that grin that hurts my cheeks.
THERAPIST: As you imagine that, what’s happening in your body right now?
CLIENT: Less buzzing. My hands aren’t shaking. I still feel a knot in my stomach, but it’s smaller.
THERAPIST: Thank you for staying with this. Before we pause today, is there one small, concrete way you could practice letting the fear be present—without having to hide it behind excitement?
CLIENT: Yeah. Tomorrow I’ll start the stand-up by saying, “I’m feeling a bit stretched and could use help clarifying timelines.” That’s honest. And I’ll try to notice my breath instead of forcing a laugh.
THERAPIST: That sounds like a grounded experiment. What support do you need to make it doable?
CLIENT: Maybe a reminder on my phone: “Breathe; tell the truth.” And I’ll ask a teammate to check in with me afterward. I think I can do that.
THERAPIST: We’ll review how it went next time. As we finish, what do you want to carry with you from today?
CLIENT: That I don’t have to be noisy to be strong. I can be scared and still lead. And maybe… it’s okay if my outside matches my inside a little more.""",
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
