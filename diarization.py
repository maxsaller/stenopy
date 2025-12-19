import os
from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline


def get_diarization_pipeline() -> Pipeline:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable not set.\n"
            "To set up:\n"
            "1. Create account at https://huggingface.co\n"
            "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Create token at https://huggingface.co/settings/tokens\n"
            "4. Run: export HF_TOKEN=hf_xxxxx"
        )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    return pipeline


def load_audio(audio_path: Path) -> dict:
    """Load audio file as a dict for pyannote (workaround for torchcodec issues)."""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    return {"waveform": waveform, "sample_rate": sample_rate}


def run_diarization(
    audio_path: Path, num_speakers: int, pipeline: Pipeline
) -> list[tuple[float, float, int]]:
    """
    Run speaker diarization on audio file.

    Returns list of (start_time, end_time, speaker_id) tuples.
    """
    # Load audio in-memory to avoid torchcodec/ffmpeg issues
    audio = load_audio(audio_path)
    output = pipeline(audio, num_speakers=num_speakers)

    segments = []
    speaker_map: dict[str, int] = {}

    # pyannote 4.x returns DiarizeOutput with speaker_diarization attribute
    diarization = getattr(output, "speaker_diarization", output)

    # Handle both old Annotation.itertracks() and new format
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)
            segments.append((turn.start, turn.end, speaker_map[speaker]))
    else:
        # New format: iterate directly
        for turn, speaker in diarization:
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)
            segments.append((turn.start, turn.end, speaker_map[speaker]))

    return segments
