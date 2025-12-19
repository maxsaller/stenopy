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
    try:
        whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"CUDA loading failed ({e}), falling back to CPU...")
        whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

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
        # For single chunk or first chunk, transcribe the whole file
        # For subsequent chunks, we'd need to handle seeking (not yet implemented)
        segments, _ = whisper_model.transcribe(
            str(audio_path),
            language="en",
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
