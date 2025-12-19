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
   - Accept terms at:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
     - https://huggingface.co/pyannote/speaker-diarization-community-1
   - Create token at https://huggingface.co/settings/tokens
   - Create `.env` file:
     ```bash
     echo "HF_TOKEN=hf_xxxxx" > .env
     ```

## Usage

### Transcribe

```bash
uv run python main.py transcribe session.mp3 --speakers 5
```

Options:
- `--speakers, -s`: Number of speakers (required)
- `--output, -o`: Output file path (default: `<audio>_transcript.txt`)

### Remap Speaker Names

After reviewing the transcript to identify who is Speaker 0, Speaker 1, etc:

```bash
uv run python main.py remap session_transcript.txt --names "DM,Alice,Bob,Carol,Dave"
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

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA (recommended for faster processing)
- HuggingFace account with access to pyannote models
