import os
from typing import Tuple


def get_workspace_root() -> str:
    # Default to current working directory if not running inside the given workspace path
    return os.getcwd()


def create_session_directories(
    workspace_root: str,
    patient_id: str,
    session_ts: int,
) -> Tuple[str, str, str, str]:
    """
    Creates:
      {workspace_root}/data/sessions/{patient_id}/{session_ts}/
        - media/
        - frames/
        - outputs/
    """
    session_dir = os.path.join(
        workspace_root, "data", "sessions", patient_id, str(session_ts)
    )
    media_dir = os.path.join(session_dir, "media")
    frames_dir = os.path.join(session_dir, "frames")
    outputs_dir = os.path.join(session_dir, "outputs")
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    return session_dir, media_dir, frames_dir, outputs_dir


