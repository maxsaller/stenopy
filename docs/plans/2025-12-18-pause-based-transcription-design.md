# Pause-Based Transcription - Design

## Overview

Simplify the transcription tool by removing speaker diarization (which was unreliable) and instead outputting clean paragraphs based on natural pauses. The output is optimized for downstream LLM processing (e.g., Gemini) to infer speakers from context.

## Core Changes

**What changes:**
- Remove speaker diarization entirely (delete diarization.py, remove pyannote dependency)
- Merge whisper segments into paragraphs based on pause detection
- Output clean plain text optimized for LLM post-processing

## CLI Interface

```bash
# Basic transcription (plain text, 2s pause threshold)
python main.py transcribe session.wav

# With timestamps
python main.py transcribe session.wav --timestamps

# Custom pause threshold
python main.py transcribe session.wav --pause-threshold 3.0

# Both
python main.py transcribe session.wav --timestamps --pause-threshold 1.5
```

## Output Format

**Default (plain text):**
```
Alright everyone, welcome back. Last session you were in the tavern discussing the rumors about the missing merchant.

I want to talk to the bartender about what he saw that night.

While they do that, I'm keeping an eye on the door. You never know who might walk in.
```

**With --timestamps:**
```
[00:00:15] Alright everyone, welcome back. Last session you were in the tavern discussing the rumors about the missing merchant.

[00:00:45] I want to talk to the bartender about what he saw that night.

[00:01:02] While they do that, I'm keeping an eye on the door. You never know who might walk in.
```

## Architecture

**Files to remove:**
- `diarization.py` - No longer needed
- `remap.py` - No speakers to remap

**Files to modify:**
- `transcriber.py` - Remove diarization, add pause-based merging
- `main.py` - Simplify CLI (remove remap command, update transcribe flags)
- `checkpoint.py` - Simplify segment structure (no speaker field)
- `pyproject.toml` - Remove pyannote-audio dependency

**New file structure:**
```
transcription/
├── main.py           # CLI (transcribe command only)
├── transcriber.py    # Transcription + pause-based merging
├── checkpoint.py     # Simplified checkpoint (no speakers)
└── pyproject.toml    # Lighter dependencies
```

**Dependencies after cleanup:**
```toml
dependencies = [
    "faster-whisper>=1.2.1",
    "python-dotenv>=1.2.1",
    "tqdm>=4.66",
]
```

## Pause-Based Merging Logic

**How it works:**

1. Whisper returns segments with timestamps: `[(0.0, 2.5, "Hello"), (2.5, 4.0, "everyone"), (4.1, 6.0, "welcome back"), (8.5, 10.0, "I want to...")]`

2. Merge segments where gap < pause threshold:
   - Gap between "welcome back" (ends 6.0) and "I want to" (starts 8.5) = 2.5 seconds
   - If threshold is 2.0s, this creates a paragraph break

3. Output merged paragraphs with optional timestamps (start time of first segment in paragraph)

**Algorithm:**
```python
def merge_segments(segments, pause_threshold=2.0):
    paragraphs = []
    current = []

    for seg in segments:
        if current and (seg.start - current[-1].end) > pause_threshold:
            # Gap detected - flush current paragraph
            paragraphs.append(current)
            current = []
        current.append(seg)

    if current:
        paragraphs.append(current)

    return paragraphs
```

**Edge cases:**
- Very long continuous speech (no pauses) → stays as one paragraph
- Lots of short pauses → many small paragraphs (user can increase threshold)

## Benefits

1. **Simpler** - No pyannote, torch, torchaudio dependencies
2. **Faster install** - Much lighter dependency footprint
3. **More reliable** - No unreliable speaker detection
4. **LLM-ready** - Clean output for downstream processing
5. **Flexible** - Configurable pause threshold and optional timestamps
