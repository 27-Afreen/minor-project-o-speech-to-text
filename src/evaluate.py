from __future__ import annotations

import argparse
from pathlib import Path

from audio_features import extract_log_spectrogram
from rnn_model import SimpleRNNClassifier
from train import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained RNN speech recognizer.")
    parser.add_argument("--manifest", default="data/raw/manifest.csv")
    parser.add_argument("--model", default="models/rnn_speech_model.npz")
    args = parser.parse_args()

    model, labels = SimpleRNNClassifier.load(args.model)
    audio_paths, transcripts = load_manifest(Path(args.manifest))

    correct = 0
    for audio_path, expected in zip(audio_paths, transcripts):
        features = extract_log_spectrogram(audio_path)
        predicted = labels[model.predict(features)]
        correct += int(predicted == expected)

    total = len(audio_paths)
    print(f"Accuracy: {correct / total:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
