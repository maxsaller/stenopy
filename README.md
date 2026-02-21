# D&D Session Transcription

Transcribe long audio recordings into clean paragraphs. Designed for D&D sessions — output is optimized for downstream LLM processing to infer speakers from context.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend) with the `large-v3` model. Audio is processed in 30-minute chunks with automatic checkpointing, so interrupted transcriptions can be resumed.

## Features

- Pause-based paragraph detection (configurable threshold)
- Checkpoint/resume for long recordings
- Optional `[HH:MM:SS]` timestamps per paragraph
- Automatic GPU acceleration (CUDA float16) with CPU fallback (int8)
- Supports any audio format FFmpeg can decode (WAV, MP3, FLAC, OGG, M4A, etc.)

## Prerequisites

### 1. Install uv (Python package manager)

[uv](https://docs.astral.sh/uv/) manages both Python versions and project dependencies. Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell or run `source $HOME/.local/bin/env` to add `uv` to your PATH.

> **Note:** uv will automatically download and manage Python 3.13 when you run `uv sync` — you do **not** need to install Python separately.

### 2. Install FFmpeg

FFmpeg is required at runtime for audio decoding.

<details>
<summary><strong>Arch Linux / Manjaro</strong></summary>

```bash
sudo pacman -S ffmpeg
```

</details>

<details>
<summary><strong>RHEL 10 / CentOS Stream / AlmaLinux / Rocky Linux</strong></summary>

FFmpeg is not in the default RHEL repos. You need [RPM Fusion](https://rpmfusion.org/):

```bash
# Enable EPEL (required by RPM Fusion)
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm

# Enable RPM Fusion Free
sudo dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm

# Install FFmpeg
sudo dnf install -y ffmpeg
```

</details>

<details>
<summary><strong>Fedora</strong></summary>

```bash
# Enable RPM Fusion Free
sudo dnf install -y https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

# Install FFmpeg
sudo dnf install -y ffmpeg
```

</details>

<details>
<summary><strong>Ubuntu / Debian</strong></summary>

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

</details>

### 3. Install NVIDIA GPU drivers (optional, recommended)

GPU acceleration makes transcription **5-10x faster**. Without a GPU, the tool falls back to CPU (int8 quantization) automatically.

cuDNN and cuBLAS are bundled as Python dependencies (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`) and installed automatically by `uv sync` on Linux — **no manual CUDA toolkit or cuDNN installation required**. You only need the NVIDIA GPU drivers.

<details>
<summary><strong>Arch Linux</strong></summary>

```bash
sudo pacman -S nvidia nvidia-utils
```

For WSL2: Install the [NVIDIA WSL driver](https://developer.nvidia.com/cuda/wsl) on the **Windows host** side. WSL2 automatically exposes the GPU — do not install Linux NVIDIA drivers inside WSL2.

</details>

<details>
<summary><strong>RHEL 10 / CentOS Stream</strong></summary>

```bash
# Add NVIDIA CUDA repo
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel$(rpm -E %rhel)/x86_64/cuda-rhel$(rpm -E %rhel).repo

# Install drivers
sudo dnf install -y cuda-drivers
```

</details>

<details>
<summary><strong>Ubuntu / Debian</strong></summary>

```bash
sudo apt-get install -y nvidia-driver-560
```

Or use the [CUDA toolkit installer](https://developer.nvidia.com/cuda-downloads) from NVIDIA.

</details>

Verify GPU access:

```bash
nvidia-smi
```

## Setup

```bash
git clone <repo-url>
cd transcription
uv sync
```

`uv sync` will:
- Download Python 3.13 if not present (managed by uv)
- Create a virtual environment in `.venv/`
- Install all pinned dependencies from `uv.lock`

### First run — model download

The first transcription will download the Whisper `large-v3` model (~3 GB) from Hugging Face to `~/.cache/huggingface/hub/`. Subsequent runs use the cached model.

## Usage

### Basic transcription

```bash
uv run python main.py recording.wav
```

Output is written to `recording_transcript.txt` (same directory as input). Paragraphs are separated by blank lines:

```
Alright everyone, welcome back. Last session you were in the tavern.

I want to talk to the bartender about the rumors we heard.

While they do that, I'm keeping an eye on the door.
```

### With timestamps

```bash
uv run python main.py recording.wav --timestamps
```

```
[00:00:15] Alright everyone, welcome back. Last session you were in the tavern.

[00:00:45] I want to talk to the bartender about the rumors we heard.

[00:01:12] While they do that, I'm keeping an eye on the door.
```

### Custom pause threshold

The default pause threshold is **2.0 seconds**. Silence longer than this between speech segments triggers a paragraph break.

```bash
# Fewer, longer paragraphs (less sensitive to pauses)
uv run python main.py recording.wav --pause-threshold 3.0

# More, shorter paragraphs (more sensitive to pauses)
uv run python main.py recording.wav --pause-threshold 1.0
```

### Custom output path

```bash
uv run python main.py recording.wav --output /path/to/transcript.txt
```

### All options combined

```bash
uv run python main.py recording.wav --timestamps --pause-threshold 1.5 --output session_notes.txt
```

## Options Reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `audio_file` | | *(required)* | Path to input audio file |
| `--output` | `-o` | `<audio>_transcript.txt` | Output file path |
| `--timestamps` | `-t` | off | Prefix each paragraph with `[HH:MM:SS]` |
| `--pause-threshold` | `-p` | `2.0` | Seconds of silence that triggers a paragraph break |

## Resuming interrupted transcriptions

Audio is processed in **30-minute chunks**. After each chunk completes, progress is saved to a hidden checkpoint file (`.{audio_stem}_checkpoint.json`) in the same directory as the audio file.

If a transcription is interrupted (Ctrl+C, crash, etc.), just re-run the same command:

```bash
# Interrupted halfway through
uv run python main.py long_session.wav
# ^C

# Resume from where it left off
uv run python main.py long_session.wav
# "Resuming from checkpoint (chunks completed: 4)"
```

**Checkpoint details:**
- Stored as JSON alongside the audio file
- Validated against an MD5 hash of the audio file (first 10 MB)
- Invalidated if the audio file changes or the `--pause-threshold` differs
- Automatically deleted on successful completion

## How it works

1. **Audio duration** is calculated using faster-whisper's decoder (16 kHz sample rate)
2. **Chunking** — audio is split into 30-minute chunks for memory efficiency
3. **Transcription** — each chunk is transcribed with Whisper `large-v3` (English)
4. **Checkpointing** — segments are saved after each chunk completes
5. **Paragraph merging** — segments are grouped by pause gaps exceeding the threshold
6. **Output** — merged paragraphs are written to the transcript file

### GPU vs CPU

The model loader automatically detects CUDA capability:

| Mode | Device | Compute Type | Speed (approx.) |
|------|--------|-------------|-----------------|
| GPU | `cuda` | `float16` | ~10 min per hour of audio |
| CPU | `cpu` | `int8` | ~60-120 min per hour of audio |

Detection checks for `float16` support via `ctranslate2.get_supported_compute_types("cuda")`. cuDNN and cuBLAS libraries are preloaded from pip-installed `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` packages at runtime, so no system-level CUDA toolkit or cuDNN installation is needed. If CUDA is unavailable or fails, it silently falls back to CPU.

## Troubleshooting

### "CUDA not available, using CPU..."

This means no GPU was detected. Verify with `nvidia-smi`. On WSL2, ensure you have the NVIDIA WSL driver installed on the Windows host.

### FFmpeg not found / audio decoding errors

Ensure FFmpeg is installed and accessible:

```bash
ffmpeg -version
```

### Model download fails

The Whisper model is downloaded from Hugging Face. If you're behind a proxy or firewall, set:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
```

Or pre-download the model manually — it caches to `~/.cache/huggingface/hub/`.

### Checkpoint mismatch on resume

If you change `--pause-threshold` between runs, the existing checkpoint is invalidated and transcription restarts from scratch. Use the same options to resume.

## Project structure

```
transcription/
├── main.py            # CLI entry point (argparse)
├── transcriber.py     # Whisper transcription + pause-based paragraph merging
├── checkpoint.py      # Checkpoint save/load/resume/validate
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Pinned dependency versions
└── docs/plans/        # Design and implementation notes
```

## Requirements

- **Python**: 3.13+ (automatically managed by uv)
- **FFmpeg**: Required for audio decoding
- **Disk**: ~4 GB for the Whisper model cache, ~1.2 GB for cuDNN/cuBLAS packages
- **RAM**: 4 GB minimum, 8 GB+ recommended for GPU mode
- **GPU** (optional): NVIDIA with drivers installed — cuDNN and cuBLAS are bundled as pip dependencies (Linux only)
