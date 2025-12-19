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
