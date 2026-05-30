from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHRASES_PATH = ROOT / "data" / "phrases.txt"
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_RATE = 16000


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    phrases = [line.strip().lower() for line in PHRASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, str]] = []

    for phrase_index, phrase in enumerate(phrases):
        for variation in range(8):
            audio = synthesize_phrase(phrase, phrase_index, variation)
            filename = f"{slugify(phrase)}_{variation + 1:02d}.wav"
            path = RAW_DIR / filename
            write_wav(path, audio)
            rows.append({"audio_path": str(path.relative_to(ROOT)), "transcript": phrase})

        sample_name = f"sample_{slugify(phrase)}.wav"
        write_wav(RAW_DIR / sample_name, synthesize_phrase(phrase, phrase_index, 99))

    manifest = RAW_DIR / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio_path", "transcript"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"created {len(rows)} training WAV files")
    print(f"created manifest: {manifest}")
    print(f"sample audio: {RAW_DIR / 'sample_hello_world.wav'}")


def synthesize_phrase(phrase: str, phrase_index: int, variation: int) -> np.ndarray:
    rng = np.random.default_rng(phrase_index * 1000 + variation)
    audio_parts = []
    base_pitch = 180 + phrase_index * 55 + rng.uniform(-5, 5)

    for word_index, word in enumerate(phrase.split()):
        duration = 0.22 + 0.035 * len(word) + rng.uniform(-0.02, 0.02)
        t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        word_hash = sum(ord(char) for char in word)
        f1 = base_pitch + (word_hash % 90) + word_index * 22
        f2 = f1 * (1.45 + 0.04 * math.sin(word_hash))
        envelope = np.sin(np.linspace(0.0, math.pi, t.size)) ** 0.7
        tone = 0.5 * np.sin(2 * math.pi * f1 * t) + 0.25 * np.sin(2 * math.pi * f2 * t)
        noise = rng.normal(0, 0.01, t.size)
        audio_parts.append((tone + noise) * envelope)
        audio_parts.append(np.zeros(int(SAMPLE_RATE * (0.05 + rng.uniform(0.0, 0.02)))))

    audio = np.concatenate(audio_parts).astype(np.float32)
    audio /= max(0.01, float(np.max(np.abs(audio))))
    return 0.85 * audio


def write_wav(path: Path, audio: np.ndarray) -> None:
    samples = np.clip(audio, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def slugify(text: str) -> str:
    return "_".join(text.lower().split())


if __name__ == "__main__":
    main()
