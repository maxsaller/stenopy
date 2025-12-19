# Pause-Based Transcription Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify transcription by removing speaker diarization and outputting clean paragraphs based on natural pauses.

**Architecture:** Remove pyannote dependency entirely. Transcribe with faster-whisper, merge segments by pause gaps, output plain text with optional timestamps. Checkpoint system remains for resume support.

**Tech Stack:** Python 3.13, faster-whisper, tqdm, python-dotenv

---

## Task 1: Remove Pyannote Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml to remove pyannote-audio**

```toml
[project]
name = "transcription"
version = "0.2.0"
description = "D&D session transcription with pause-based paragraph detection"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "faster-whisper>=1.2.1",
    "python-dotenv>=1.2.1",
    "tqdm>=4.66",
]
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Lighter dependency tree without pyannote/torch

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: remove pyannote-audio dependency"
```

---

## Task 2: Delete Unused Files

**Files:**
- Delete: `diarization.py`
- Delete: `remap.py`

**Step 1: Remove diarization.py and remap.py**

```bash
rm diarization.py remap.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "chore: remove diarization and remap modules"
```

---

## Task 3: Simplify Checkpoint Module

**Files:**
- Modify: `checkpoint.py`

**Step 1: Rewrite checkpoint.py without speaker tracking**

```python
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Checkpoint:
    audio_file: str
    audio_hash: str
    pause_threshold: float
    completed_chunks: list[int] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "audio_file": self.audio_file,
            "audio_hash": self.audio_hash,
            "pause_threshold": self.pause_threshold,
            "completed_chunks": self.completed_chunks,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        segments = [Segment(**s) for s in data.get("segments", [])]
        return cls(
            audio_file=data["audio_file"],
            audio_hash=data["audio_hash"],
            pause_threshold=data["pause_threshold"],
            completed_chunks=data.get("completed_chunks", []),
            segments=segments,
        )


def get_checkpoint_path(audio_path: Path) -> Path:
    return audio_path.parent / f".{audio_path.stem}_checkpoint.json"


def compute_audio_hash(audio_path: Path) -> str:
    hasher = hashlib.md5()
    with open(audio_path, "rb") as f:
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
    checkpoint: Checkpoint, audio_path: Path, pause_threshold: float
) -> tuple[bool, str]:
    current_hash = compute_audio_hash(audio_path)
    if checkpoint.audio_hash != current_hash:
        return False, "Audio file has changed since checkpoint was created"
    if checkpoint.pause_threshold != pause_threshold:
        return False, f"Pause threshold mismatch: checkpoint has {checkpoint.pause_threshold}, requested {pause_threshold}"
    return True, ""
```

**Step 2: Commit**

```bash
git add checkpoint.py
git commit -m "refactor: simplify checkpoint module (remove speaker tracking)"
```

---

## Task 4: Rewrite Transcriber Module

**Files:**
- Modify: `transcriber.py`

**Step 1: Rewrite transcriber.py with pause-based merging**

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


CHUNK_DURATION = 30 * 60  # 30 minutes in seconds
MODEL_SIZE = "large-v3"
DEFAULT_PAUSE_THRESHOLD = 2.0


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using faster-whisper's internal decoder."""
    from faster_whisper.audio import decode_audio
    audio = decode_audio(str(audio_path))
    return len(audio) / 16000  # 16kHz sample rate


def merge_segments_by_pause(
    segments: list[Segment], pause_threshold: float
) -> list[list[Segment]]:
    """
    Merge segments into paragraphs based on pause gaps.

    Returns list of paragraphs, where each paragraph is a list of segments.
    """
    if not segments:
        return []

    paragraphs = []
    current: list[Segment] = []

    for seg in segments:
        if current and (seg.start - current[-1].end) > pause_threshold:
            paragraphs.append(current)
            current = []
        current.append(seg)

    if current:
        paragraphs.append(current)

    return paragraphs


def transcribe_audio(
    audio_path: Path,
    output_path: Path | None = None,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
    include_timestamps: bool = False,
) -> Path:
    """
    Transcribe audio file with pause-based paragraph detection.

    Returns path to output transcript file.
    """
    if output_path is None:
        output_path = audio_path.parent / f"{audio_path.stem}_transcript.txt"

    # Check for existing checkpoint
    checkpoint = load_checkpoint(audio_path)
    if checkpoint is not None:
        valid, msg = validate_checkpoint(checkpoint, audio_path, pause_threshold)
        if not valid:
            print(f"Checkpoint invalid: {msg}")
            print("Starting fresh...")
            checkpoint = None

    # Initialize checkpoint if needed
    if checkpoint is None:
        checkpoint = Checkpoint(
            audio_file=str(audio_path),
            audio_hash=compute_audio_hash(audio_path),
            pause_threshold=pause_threshold,
        )
    else:
        print(f"Resuming from checkpoint (chunks completed: {len(checkpoint.completed_chunks)})")

    # Get audio duration and calculate chunks
    duration = get_audio_duration(audio_path)
    num_chunks = int(duration // CHUNK_DURATION) + (1 if duration % CHUNK_DURATION > 0 else 0)

    # Load model
    print("Loading Whisper model...")
    try:
        whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"CUDA loading failed ({e}), falling back to CPU...")
        whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

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
        )

        # Collect segments for this chunk
        for seg in segments:
            if chunk_idx > 0 and seg.start < chunk_start:
                continue
            if seg.end > chunk_end:
                break

            checkpoint.segments.append(
                Segment(start=seg.start, end=seg.end, text=seg.text.strip())
            )

        checkpoint.completed_chunks.append(chunk_idx)
        save_checkpoint(checkpoint, audio_path)

    # Merge segments into paragraphs and write output
    paragraphs = merge_segments_by_pause(checkpoint.segments, pause_threshold)
    write_transcript(paragraphs, output_path, include_timestamps)
    delete_checkpoint(audio_path)

    print(f"\nTranscription complete: {output_path}")
    return output_path


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_transcript(
    paragraphs: list[list[Segment]],
    output_path: Path,
    include_timestamps: bool = False,
) -> None:
    """Write paragraphs to transcript file."""
    with open(output_path, "w") as f:
        for para in paragraphs:
            if not para:
                continue

            # Combine segment texts into paragraph
            text = " ".join(seg.text for seg in para)

            if include_timestamps:
                timestamp = format_timestamp(para[0].start)
                f.write(f"[{timestamp}] {text}\n\n")
            else:
                f.write(f"{text}\n\n")
```

**Step 2: Commit**

```bash
git add transcriber.py
git commit -m "refactor: replace diarization with pause-based paragraph merging"
```

---

## Task 5: Simplify CLI

**Files:**
- Modify: `main.py`

**Step 1: Rewrite main.py with simplified CLI**

```python
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from transcriber import transcribe_audio, DEFAULT_PAUSE_THRESHOLD

load_dotenv()


def cmd_transcribe(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None

    try:
        transcribe_audio(
            audio_path,
            output_path=output_path,
            pause_threshold=args.pause_threshold,
            include_timestamps=args.timestamps,
        )
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with pause-based paragraph detection"
    )

    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--output", "-o",
        help="Output transcript path (default: <audio>_transcript.txt)"
    )
    parser.add_argument(
        "--timestamps", "-t",
        action="store_true",
        help="Include timestamps at the start of each paragraph"
    )
    parser.add_argument(
        "--pause-threshold", "-p",
        type=float,
        default=DEFAULT_PAUSE_THRESHOLD,
        help=f"Pause duration (seconds) that triggers a paragraph break (default: {DEFAULT_PAUSE_THRESHOLD})"
    )

    args = parser.parse_args()
    return cmd_transcribe(args)


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Commit**

```bash
git add main.py
git commit -m "refactor: simplify CLI (remove subcommands, add pause/timestamp options)"
```

---

## Task 6: Update README

**Files:**
- Modify: `README.md`

**Step 1: Rewrite README with new usage**

```markdown
# D&D Session Transcription

Transcribe long audio recordings into clean paragraphs. Designed for D&D sessions - output is optimized for downstream LLM processing to infer speakers from context.

## Features

- Pause-based paragraph detection
- Checkpoint/resume for long recordings
- Optional timestamps
- Configurable pause threshold

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

### Basic Transcription

```bash
uv run python main.py session.wav
```

Output: Clean paragraphs separated by blank lines.

### With Timestamps

```bash
uv run python main.py session.wav --timestamps
```

Output:
```
[00:00:15] Alright everyone, welcome back. Last session you were in the tavern.

[00:00:45] I want to talk to the bartender about the rumors.
```

### Custom Pause Threshold

Default is 2 seconds. Increase for fewer, longer paragraphs:

```bash
uv run python main.py session.wav --pause-threshold 3.0
```

### All Options

```bash
uv run python main.py session.wav --timestamps --pause-threshold 1.5 --output notes.txt
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file path (default: `<audio>_transcript.txt`) |
| `--timestamps` | `-t` | Include timestamps at paragraph start |
| `--pause-threshold` | `-p` | Seconds of silence for paragraph break (default: 2.0) |

## Resume After Interruption

If transcription is interrupted, run the same command again. Progress is saved in checkpoint files.

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA (recommended)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for pause-based transcription"
```

---

## Task 7: Test with Sample Audio

**Step 1: Run transcription on sample**

```bash
uv run python main.py samples/sample.wav
```

Expected: Creates `samples/sample_transcript.txt` with paragraphs.

**Step 2: Test with timestamps**

```bash
uv run python main.py samples/sample.wav --timestamps --output samples/sample_with_timestamps.txt
```

Expected: Paragraphs with `[HH:MM:SS]` prefixes.

**Step 3: Test with custom threshold**

```bash
uv run python main.py samples/sample.wav --pause-threshold 1.0 --output samples/sample_short_pauses.txt
```

Expected: More, shorter paragraphs.

**Step 4: Verify output quality**

Check that:
- Text flows naturally as paragraphs
- Pauses correctly break speaker turns
- Timestamps are accurate (if enabled)

**Step 5: Commit any fixes if needed**

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Remove pyannote-audio dependency |
| 2 | Delete diarization.py and remap.py |
| 3 | Simplify checkpoint module |
| 4 | Rewrite transcriber with pause-based merging |
| 5 | Simplify CLI |
| 6 | Update README |
| 7 | Test with sample audio |
