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
