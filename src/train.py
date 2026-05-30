from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from audio_features import extract_log_spectrogram
from rnn_model import SimpleRNNClassifier


def load_manifest(path: Path) -> tuple[list[Path], list[str]]:
    audio_paths: list[Path] = []
    transcripts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = path.parent.parent.parent / audio_path
            audio_paths.append(audio_path)
            transcripts.append(row["transcript"].strip().lower())
    if not audio_paths:
        raise ValueError(f"No training rows found in {path}")
    return audio_paths, transcripts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RNN speech-to-text phrase recognizer.")
    parser.add_argument("--manifest", default="data/raw/manifest.csv", help="CSV with audio_path,transcript columns.")
    parser.add_argument("--model", default="models/rnn_speech_model.npz", help="Output model path.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    audio_paths, transcripts = load_manifest(manifest)
    labels = sorted(set(transcripts))
    label_to_index = {label: index for index, label in enumerate(labels)}

    print(f"loading {len(audio_paths)} audio files...")
    features = np.stack([extract_log_spectrogram(path) for path in audio_paths])
    targets = np.array([label_to_index[text] for text in transcripts], dtype=np.int64)

    model = SimpleRNNClassifier(
        input_size=features.shape[-1],
        hidden_size=args.hidden_size,
        output_size=len(labels),
        learning_rate=args.learning_rate,
    )
    model.fit(features, targets, epochs=args.epochs)

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), labels)
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
