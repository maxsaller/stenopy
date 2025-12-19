# D&D Session Transcription CLI - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that transcribes long D&D session audio files with speaker diarization and checkpoint/resume support.

**Architecture:** CLI with subcommands (transcribe, remap). Core processing uses faster-whisper for transcription and pyannote for diarization. Audio is processed in 30-minute chunks with checkpoint files enabling resume after interruption.

**Tech Stack:** Python 3.13, faster-whisper, pyannote.audio, tqdm, argparse

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pyannote.audio and tqdm to dependencies**

```toml
[project]
name = "transcription"
version = "0.1.0"
description = "D&D session transcription with speaker diarization"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "faster-whisper>=1.2.1",
    "pyannote-audio>=3.1",
    "tqdm>=4.66",
]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add pyannote-audio and tqdm dependencies"
```

---

## Task 2: Create Checkpoint Module

**Files:**
- Create: `checkpoint.py`

**Step 1: Create checkpoint.py with save/load/validate functions**

```python
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    speaker: int
    text: str


@dataclass
class Checkpoint:
    audio_file: str
    audio_hash: str
    num_speakers: int
    completed_chunks: list[int] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "audio_file": self.audio_file,
            "audio_hash": self.audio_hash,
            "num_speakers": self.num_speakers,
            "completed_chunks": self.completed_chunks,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        segments = [Segment(**s) for s in data.get("segments", [])]
        return cls(
            audio_file=data["audio_file"],
            audio_hash=data["audio_hash"],
            num_speakers=data["num_speakers"],
            completed_chunks=data.get("completed_chunks", []),
            segments=segments,
        )


def get_checkpoint_path(audio_path: Path) -> Path:
    return audio_path.parent / f".{audio_path.stem}_checkpoint.json"


def compute_audio_hash(audio_path: Path) -> str:
    hasher = hashlib.md5()
    with open(audio_path, "rb") as f:
        # Read first 10MB for hash (enough to detect changes, fast for large files)
        hasher.update(f.read(10 * 1024 * 1024))
    return hasher.hexdigest()


def save_checkpoint(checkpoint: Checkpoint, audio_path: Path) -> None:
    checkpoint_path = get_checkpoint_path(audio_path)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)


def load_checkpoint(audio_path: Path) -> Checkpoint | None:
    checkpoint_path = get_checkpoint_path(audio_path)
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path) as f:
        return Checkpoint.from_dict(json.load(f))


def delete_checkpoint(audio_path: Path) -> None:
    checkpoint_path = get_checkpoint_path(audio_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def validate_checkpoint(
    checkpoint: Checkpoint, audio_path: Path, num_speakers: int
) -> tuple[bool, str]:
    current_hash = compute_audio_hash(audio_path)
    if checkpoint.audio_hash != current_hash:
        return False, "Audio file has changed since checkpoint was created"
    if checkpoint.num_speakers != num_speakers:
        return False, f"Speaker count mismatch: checkpoint has {checkpoint.num_speakers}, requested {num_speakers}"
    return True, ""
```

**Step 2: Commit**

```bash
git add checkpoint.py
git commit -m "feat: add checkpoint module for save/load/resume support"
```

---

## Task 3: Create Diarization Module

**Files:**
- Create: `diarization.py`

**Step 1: Create diarization.py with pyannote integration**

```python
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
        use_auth_token=hf_token,
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
```

**Step 2: Commit**

```bash
git add diarization.py
git commit -m "feat: add diarization module with pyannote integration"
```

---

## Task 4: Create Transcriber Module

**Files:**
- Create: `transcriber.py`

**Step 1: Create transcriber.py with chunked processing**

```python
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

from checkpoint import (
    Checkpoint,
    Segment,
    compute_audio_hash,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from diarization import get_diarization_pipeline, run_diarization


CHUNK_DURATION = 30 * 60  # 30 minutes in seconds
MODEL_SIZE = "large-v3"


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using faster-whisper's internal decoder."""
    from faster_whisper.audio import decode_audio
    audio = decode_audio(str(audio_path))
    return len(audio) / 16000  # 16kHz sample rate


def assign_speaker_to_segment(
    seg_start: float,
    seg_end: float,
    diarization_segments: list[tuple[float, float, int]],
) -> int:
    """Find the speaker with most overlap for a given transcription segment."""
    best_speaker = 0
    best_overlap = 0.0

    for d_start, d_end, speaker in diarization_segments:
        overlap_start = max(seg_start, d_start)
        overlap_end = min(seg_end, d_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


def transcribe_audio(
    audio_path: Path,
    num_speakers: int,
    output_path: Path | None = None,
) -> Path:
    """
    Transcribe audio file with speaker diarization.

    Returns path to output transcript file.
    """
    if output_path is None:
        output_path = audio_path.parent / f"{audio_path.stem}_transcript.txt"

    # Check for existing checkpoint
    checkpoint = load_checkpoint(audio_path)
    if checkpoint is not None:
        valid, msg = validate_checkpoint(checkpoint, audio_path, num_speakers)
        if not valid:
            print(f"Checkpoint invalid: {msg}")
            print("Starting fresh...")
            checkpoint = None

    # Initialize checkpoint if needed
    if checkpoint is None:
        checkpoint = Checkpoint(
            audio_file=str(audio_path),
            audio_hash=compute_audio_hash(audio_path),
            num_speakers=num_speakers,
        )
    else:
        print(f"Resuming from checkpoint (chunks completed: {len(checkpoint.completed_chunks)})")

    # Get audio duration and calculate chunks
    duration = get_audio_duration(audio_path)
    num_chunks = int(duration // CHUNK_DURATION) + (1 if duration % CHUNK_DURATION > 0 else 0)

    # Load models
    print("Loading Whisper model...")
    whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    print("Loading diarization pipeline...")
    diarization_pipeline = get_diarization_pipeline()

    # Run diarization on full audio (pyannote handles this efficiently)
    print("Running speaker diarization...")
    diarization_segments = run_diarization(audio_path, num_speakers, diarization_pipeline)

    # Process chunks
    print(f"\nTranscribing {num_chunks} chunks...")
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        if chunk_idx in checkpoint.completed_chunks:
            continue

        chunk_start = chunk_idx * CHUNK_DURATION
        chunk_end = min((chunk_idx + 1) * CHUNK_DURATION, duration)

        # Transcribe chunk
        segments, _ = whisper_model.transcribe(
            str(audio_path),
            language="en",
            clip_timestamps=[chunk_start, chunk_end] if chunk_idx > 0 else None,
        )

        # Process segments and assign speakers
        for seg in segments:
            if chunk_idx > 0 and seg.start < chunk_start:
                continue
            if seg.end > chunk_end:
                break

            speaker = assign_speaker_to_segment(
                seg.start, seg.end, diarization_segments
            )
            checkpoint.segments.append(
                Segment(start=seg.start, end=seg.end, speaker=speaker, text=seg.text.strip())
            )

        checkpoint.completed_chunks.append(chunk_idx)
        save_checkpoint(checkpoint, audio_path)

    # Write final output
    write_transcript(checkpoint.segments, output_path)
    delete_checkpoint(audio_path)

    print(f"\nTranscription complete: {output_path}")
    return output_path


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_transcript(segments: list[Segment], output_path: Path) -> None:
    """Write segments to transcript file."""
    with open(output_path, "w") as f:
        for seg in segments:
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            f.write(f"[{start} - {end}] Speaker {seg.speaker}: {seg.text}\n")
```

**Step 2: Commit**

```bash
git add transcriber.py
git commit -m "feat: add transcriber module with chunked processing"
```

---

## Task 5: Create Remap Module

**Files:**
- Create: `remap.py`

**Step 1: Create remap.py for speaker name remapping**

```python
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

    # Replace speaker labels
    for i, name in enumerate(names):
        content = content.replace(f"Speaker {i}:", f"{name}:")

    # Write back
    transcript_path.write_text(content)
    print(f"Remapped speakers in {transcript_path}")
```

**Step 2: Commit**

```bash
git add remap.py
git commit -m "feat: add remap module for speaker name replacement"
```

---

## Task 6: Create CLI Entry Point

**Files:**
- Modify: `main.py`

**Step 1: Replace main.py with CLI using argparse**

```python
import argparse
import sys
from pathlib import Path

from remap import remap_speakers
from transcriber import transcribe_audio


def cmd_transcribe(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None

    try:
        transcribe_audio(audio_path, args.speakers, output_path)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_remap(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript_file)
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}", file=sys.stderr)
        return 1

    names = [n.strip() for n in args.names.split(",")]

    try:
        remap_speakers(transcript_path, names)
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe an audio file"
    )
    transcribe_parser.add_argument("audio_file", help="Path to audio file")
    transcribe_parser.add_argument(
        "--speakers", "-s", type=int, required=True,
        help="Number of speakers in the recording"
    )
    transcribe_parser.add_argument(
        "--output", "-o",
        help="Output transcript path (default: <audio>_transcript.txt)"
    )
    transcribe_parser.set_defaults(func=cmd_transcribe)

    # Remap command
    remap_parser = subparsers.add_parser(
        "remap", help="Remap speaker labels to names"
    )
    remap_parser.add_argument("transcript_file", help="Path to transcript file")
    remap_parser.add_argument(
        "--names", "-n", required=True,
        help="Comma-separated speaker names (e.g., 'DM,Alice,Bob')"
    )
    remap_parser.set_defaults(func=cmd_remap)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add CLI with transcribe and remap commands"
```

---

## Task 7: Test with Sample Audio

**Step 1: Create a short test recording or use existing audio**

Use any short audio file (1-2 minutes) with 2+ speakers to verify everything works.

**Step 2: Run transcription**

Run: `python main.py transcribe test.mp3 --speakers 2`
Expected: Progress bars, transcript file created

**Step 3: Verify output format**

Check that output file contains lines like:
```
[00:00:05 - 00:00:12] Speaker 0: Hello, welcome to the session.
[00:00:13 - 00:00:18] Speaker 1: Thanks for having me.
```

**Step 4: Test remap**

Run: `python main.py remap test_transcript.txt --names "Host,Guest"`
Expected: File updated with names instead of "Speaker N"

**Step 5: Commit any fixes**

If fixes were needed, commit them with descriptive message.

---

## Task 8: Update README

**Files:**
- Modify: `README.md`

**Step 1: Write README with usage instructions**

```markdown
# D&D Session Transcription

Transcribe long audio recordings with speaker diarization. Designed for D&D sessions but works with any multi-speaker audio.

## Features

- Speaker diarization (who said what)
- Checkpoint/resume for long recordings
- Speaker name remapping

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up HuggingFace token for pyannote:
   - Create account at https://huggingface.co
   - Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Create token at https://huggingface.co/settings/tokens
   - Export token:
     ```bash
     export HF_TOKEN=hf_xxxxx
     ```

## Usage

### Transcribe

```bash
python main.py transcribe session.mp3 --speakers 5
```

Options:
- `--speakers, -s`: Number of speakers (required)
- `--output, -o`: Output file path (default: `<audio>_transcript.txt`)

### Remap Speaker Names

After reviewing the transcript to identify who is Speaker 0, Speaker 1, etc:

```bash
python main.py remap session_transcript.txt --names "DM,Alice,Bob,Carol,Dave"
```

## Output Format

```
[00:00:15 - 00:00:23] Speaker 0: Alright everyone, welcome back.
[00:00:24 - 00:00:28] Speaker 2: I want to talk to the bartender.
```

After remapping:
```
[00:00:15 - 00:00:23] DM: Alright everyone, welcome back.
[00:00:24 - 00:00:28] Alice: I want to talk to the bartender.
```

## Resume After Interruption

If transcription is interrupted, simply run the same command again. Progress is saved in checkpoint files and will resume automatically.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add dependencies (pyannote-audio, tqdm) |
| 2 | Create checkpoint module |
| 3 | Create diarization module |
| 4 | Create transcriber module |
| 5 | Create remap module |
| 6 | Create CLI entry point |
| 7 | Test with sample audio |
| 8 | Update README |
