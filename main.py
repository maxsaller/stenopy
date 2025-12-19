import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from remap import remap_speakers
from transcriber import transcribe_audio

load_dotenv()


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
