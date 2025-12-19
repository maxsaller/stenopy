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


def _load_whisper_model() -> WhisperModel:
    """Load Whisper model, preferring CUDA if available."""
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if "float16" in cuda_types:
            return WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception:
        pass
    print("CUDA not available, using CPU...")
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


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
    whisper_model = _load_whisper_model()

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
