import os
from pathlib import Path

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


def run_diarization(
    audio_path: Path, num_speakers: int, pipeline: Pipeline
) -> list[tuple[float, float, int]]:
    """
    Run speaker diarization on audio file.

    Returns list of (start_time, end_time, speaker_id) tuples.
    """
    diarization = pipeline(str(audio_path), num_speakers=num_speakers)

    segments = []
    speaker_map: dict[str, int] = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = len(speaker_map)
        segments.append((turn.start, turn.end, speaker_map[speaker]))

    return segments
