from __future__ import annotations

import argparse
from pathlib import Path

from audio_features import extract_log_spectrogram
from rnn_model import SimpleRNNClassifier


def transcribe_with_google(audio_path: Path) -> str:
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise SystemExit(
            "SpeechRecognition is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    recognizer = sr.Recognizer()
    with sr.AudioFile(str(audio_path)) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError as exc:
        raise SystemExit("Could not understand the audio clearly enough to transcribe it.") from exc
    except sr.RequestError as exc:
        raise SystemExit(f"Speech recognition service error: {exc}") from exc


def transcribe_with_rnn(audio_path: Path, model_path: Path) -> str:
    model, labels = SimpleRNNClassifier.load(str(model_path))
    features = extract_log_spectrogram(audio_path)
    return labels[model.predict(features)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an uploaded audio file.")
    parser.add_argument("--audio", required=True, help="Path to a WAV, AIFF, AIFF-C, or FLAC audio file.")
    parser.add_argument(
        "--engine",
        choices=["google", "rnn"],
        default="google",
        help="Use google for general speech, or rnn for this project's trained demo phrases.",
    )
    parser.add_argument("--model", default="models/rnn_speech_model.npz", help="RNN model path.")
    parser.add_argument("--verbose", action="store_true", help="Print engine details with the transcript.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    if args.engine == "google":
        transcript = transcribe_with_google(audio_path)
    else:
        transcript = transcribe_with_rnn(audio_path, Path(args.model))

    if args.verbose:
        print(f"Engine: {args.engine}")
        print(f"Audio: {audio_path}")
        print("Transcript:")
    print(transcript)


if __name__ == "__main__":
    main()
