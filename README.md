## Emotion Analysis API (FastAPI)

### Overview
This service exposes a single endpoint:
- `POST /process_session` with body:
  ```json
  {
    "video_url": "https://example.com/video.mp4",
    "patient_id": "abc123",
    "spike_threshold": 0.2
  }
  ```
It will:
1. Download the video.
2. Extract audio (`audio.wav`) via FFmpeg.
3. Extract frames at 1 FPS via FFmpeg.
4. Run DeepFace on frames for facial emotions.
5. Optionally run audio emotion analysis with Vesper (if installed).
6. Merge into a unified `timeline_json`.
7. Detect micro-spikes and return `spikes_json`.
8. Transcribe audio (best-effort) with Faster-Whisper and return `transcript_text` and `transcript_segments`.

Artifacts are saved under `data/sessions/{patient_id}/{session_ts}/`.

### Requirements
- Python 3.10+
- FFmpeg installed and available on PATH.
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
  - Windows: https://ffmpeg.org/download.html
- Python deps:
  ```bash
  pip install -r requirements.txt
  ```
  DeepFace is installed via PyPI; it may download model weights on first run.
  Faster-Whisper will download a model on first run; model size can be changed in code.

Optional:
- Vesper audio emotion library as per your environment. If not installed, audio emotions are skipped gracefully.

### Run
```bash
uvicorn app.main:app --reload
```

### Example Request
```bash
curl -X POST http://localhost:8000/process_session \
  -H "Content-Type: application/json" \
  -d '{
        "video_url": "https://example.com/sample.mp4",
        "patient_id": "test_patient",
        "spike_threshold": 0.2
      }'
```

### Notes
- The endpoint runs synchronously and may take several minutes depending on video length and model loading.
- If Vesper is unavailable, the response will still include `timeline_json` based on facial emotions and a note about audio analysis fallback.
- Transcription is best-effort; if Faster-Whisper is not installed or fails, transcript fields will be omitted.


# emotionalCongruence
