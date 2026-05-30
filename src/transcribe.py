from __future__ import annotations

import argparse
from pathlib import Path

from audio_features import extract_log_spectrogram
from rnn_model import SimpleRNNClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a WAV file with the trained RNN model.")
    parser.add_argument("--audio", required=True, help="Path to a WAV file.")
    parser.add_argument("--model", default="models/rnn_speech_model.npz", help="Saved model path.")
    parser.add_argument("--top-k", type=int, default=3, help="How many candidate phrases to show.")
    args = parser.parse_args()

    model, labels = SimpleRNNClassifier.load(args.model)
    features = extract_log_spectrogram(Path(args.audio))
    probabilities = model.predict_proba(features)
    ranked = probabilities.argsort()[::-1][: args.top_k]

    print("Transcription:", labels[int(ranked[0])])
    print("Confidence:", f"{probabilities[int(ranked[0])] * 100:.2f}%")
    print("Top candidates:")
    for index in ranked:
        print(f"  {labels[int(index)]:24s} {probabilities[int(index)] * 100:6.2f}%")


if __name__ == "__main__":
    main()
