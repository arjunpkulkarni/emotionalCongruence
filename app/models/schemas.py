from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class ProcessSessionRequest(BaseModel):
    video_url: HttpUrl = Field(..., description="Publicly accessible URL of the video file (mp4)")
    patient_id: str = Field(..., description="Patient or subject identifier")
    spike_threshold: float = Field(0.2, ge=0.0, le=1.0, description="Delta threshold for spike detection")


class ProcessSessionResponse(BaseModel):
    patient_id: str
    session_timestamp: int
    paths: Dict[str, str]
    timeline_json: List[Dict[str, Any]]
    spikes_json: List[Dict[str, Any]]
    notes: Optional[str] = None
    transcript_text: Optional[str] = None
    transcript_segments: Optional[List[Dict[str, Any]]] = None


