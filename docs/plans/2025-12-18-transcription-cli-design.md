# D&D Session Transcription CLI - Design

## Overview

A CLI tool for transcribing long audio files (4+ hours) with speaker diarization, designed for D&D session recordings. Uses faster-whisper for transcription and pyannote for speaker detection.

## Core Functionality

### Basic Usage

```bash
# Transcribe with 5 speakers
python main.py transcribe session.mp3 --speakers 5

# With custom output path
python main.py transcribe session.mp3 --speakers 5 --output ./transcripts/session1.txt

# Remap speaker names after reviewing
python main.py remap session_transcript.txt --names "DM,Alice,Bob,Carol,Dave"
```

### Output Format

Initial transcription:
```
[00:00:15 - 00:00:23] Speaker 0: Alright everyone, welcome back. Last session you were in the tavern.
[00:00:24 - 00:00:28] Speaker 2: I want to talk to the bartender about the rumors.
[00:00:29 - 00:00:35] Speaker 1: While they do that, I'm keeping an eye on the door.
```

After remapping:
```
[00:00:15 - 00:00:23] DM: Alright everyone, welcome back. Last session you were in the tavern.
[00:00:24 - 00:00:28] Alice: I want to talk to the bartender about the rumors.
[00:00:29 - 00:00:35] Bob: While they do that, I'm keeping an eye on the door.
```

## Chunked Processing & Checkpoints

- Audio is processed in 30-minute chunks
- After each chunk completes, progress is saved to a checkpoint file (`.transcription_checkpoint.json` next to the audio)
- If interrupted, re-running the same command detects the checkpoint and resumes

### Checkpoint File

```json
{
  "audio_file": "session.mp3",
  "audio_hash": "abc123...",
  "num_speakers": 5,
  "completed_chunks": [0, 1, 2],
  "segments": [
    {"start": 0.5, "end": 3.2, "speaker": 0, "text": "..."}
  ]
}
```

### Behavior

- On completion, checkpoint is deleted and final transcript is written
- If audio file changes (different hash), checkpoint is invalidated and starts fresh
- Progress bar shows overall progress: `Processing chunk 3/8 [████░░░░] 37%`

### Resume Example

```bash
$ python main.py transcribe session.mp3 --speakers 5
Processing chunk 1/8...
Processing chunk 2/8...
^C  # interrupted

$ python main.py transcribe session.mp3 --speakers 5
Resuming from chunk 3/8...
Processing chunk 3/8...
```

## Architecture

### File Structure

```
transcription/
├── main.py           # CLI entry point (argparse)
├── transcriber.py    # Core transcription logic
├── diarization.py    # Speaker diarization with pyannote
├── checkpoint.py     # Checkpoint save/load/resume
├── remap.py          # Speaker name remapping
└── pyproject.toml
```

### Dependencies

- `faster-whisper>=1.2.1` (already installed)
- `pyannote.audio` - Speaker diarization
- `tqdm` - Progress bars

### Processing Flow

1. Load audio file
2. Run faster-whisper transcription (chunks with timestamps)
3. Run pyannote diarization (speaker segments)
4. Merge: align whisper segments with speaker labels
5. Save checkpoint after each chunk
6. On completion, write final transcript

## Setup Requirements

### One-time Setup

1. Create a HuggingFace account
2. Accept the pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Set `HF_TOKEN` environment variable

### Environment

```bash
export HF_TOKEN=hf_xxxxx
```

## CLI Interface

### Commands

```bash
# Transcribe audio
python main.py transcribe <audio_file> --speakers <N> [--output <path>]

# Remap speaker names in existing transcript
python main.py remap <transcript_file> --names "DM,Alice,Bob,Carol"
```

### Error Handling

- Missing `HF_TOKEN` → clear error message with setup instructions
- Audio file not found → error with path
- Checkpoint mismatch (different speaker count) → prompt to restart or abort
- Interrupted → message about resuming with same command

## Technical Details

- Model: large-v3 (hardcoded)
- Platform: WSL Arch Linux with NVIDIA RTX 4090
- Chunk size: 30 minutes
