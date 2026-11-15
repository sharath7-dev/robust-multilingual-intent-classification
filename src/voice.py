"""Optional, lazy microphone transcription for the Colab demonstration."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch


_PIPELINE = None
_LOCK = Lock()


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                from transformers import pipeline

                _PIPELINE = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny",
                    device=0 if torch.cuda.is_available() else -1,
                )
    return _PIPELINE


def transcribe_audio(audio_path: str | None) -> str:
    if not audio_path:
        raise ValueError("Record a voice request first.")
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError("The recorded audio file is unavailable.")
    result = _get_pipeline()(str(path))
    text = str(result.get("text", "")).strip()
    if not text:
        raise ValueError("No speech was detected. Please record again in a quieter place.")
    return text
