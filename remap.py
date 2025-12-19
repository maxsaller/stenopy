import re
from pathlib import Path


def remap_speakers(transcript_path: Path, names: list[str]) -> None:
    """
    Replace 'Speaker N' with actual names in transcript file.

    Names should be provided in order: names[0] = Speaker 0, names[1] = Speaker 1, etc.
    """
    content = transcript_path.read_text()

    # Find all unique speaker numbers
    speaker_nums = set(re.findall(r"Speaker (\d+)", content))
    max_speaker = max(int(n) for n in speaker_nums) if speaker_nums else -1

    if len(names) <= max_speaker:
        raise ValueError(
            f"Not enough names provided. Transcript has speakers 0-{max_speaker} "
            f"but only {len(names)} names given."
        )

    # Replace speaker labels (use regex to match exact speaker numbers)
    for i, name in enumerate(names):
        content = re.sub(rf"\bSpeaker {i}:", f"{name}:", content)

    # Write back
    transcript_path.write_text(content)
    print(f"Remapped speakers in {transcript_path}")
