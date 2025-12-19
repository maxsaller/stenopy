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
