# Speech-to-Text System using RNN

This project demonstrates a speech-to-text workflow using Python, audio preprocessing, sequence features, and a recurrent neural network classifier.

The starter version is intentionally lightweight so it can run on a normal laptop without downloading a large speech corpus. It recognizes a small set of spoken phrases after training on WAV files. You can expand it later with more phrases, more speakers, or a deep learning framework such as PyTorch.

## Project Structure

```text
Speech-to-Text-RNN/
  .vscode/              VS Code run/debug setup
  data/
    phrases.txt         Training phrases
    raw/                WAV files and manifest.csv
  models/               Saved RNN model files
  scripts/              Dataset/setup helper scripts
  src/                  Python source code
```

## Quick Start

Use these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\make_tone_dataset.py
python src\train.py --manifest data\raw\manifest.csv --model models\rnn_speech_model.npz
python src\evaluate.py --manifest data\raw\manifest.csv --model models\rnn_speech_model.npz
python src\transcribe.py --audio data\raw\sample_hello_world.wav --model models\rnn_speech_model.npz
```

If `python` is not recognized in your terminal, install Python from <https://www.python.org/downloads/> and check **Add python.exe to PATH** during installation.

## Using Real Voice Samples

Record short mono WAV files for each phrase and create a CSV file like this:

```csv
audio_path,transcript
data/raw/hello_world_01.wav,hello world
data/raw/good_morning_01.wav,good morning
```

Then train:

```powershell
python src\train.py --manifest data\raw\manifest.csv --model models\rnn_speech_model.npz
```

And transcribe:

```powershell
python src\transcribe.py --audio path\to\your_audio.wav --model models\rnn_speech_model.npz
```

## Transcribe Any Uploaded Audio File

For normal spoken audio, use:

```powershell
python src\transcribe_uploaded_audio.py --audio "path\to\your_audio.wav"
```

This prints only the transcript by default. Supported input formats are WAV, AIFF, AIFF-C, and FLAC. The default `google` engine works for general speech and requires an internet connection. The `rnn` engine uses this project's trained demo model and only recognizes phrases it has learned:

```powershell
python src\transcribe_uploaded_audio.py --engine rnn --audio data\raw\sample_hello_world.wav --model models\rnn_speech_model.npz
```

## Notes

- `scripts/make_tone_dataset.py` creates a deterministic demo dataset so the full pipeline can execute immediately.
- `scripts/create_sapi_dataset.ps1` can create spoken WAV files on Windows using the built-in Speech API if it is available on your system.
- This starter model is phrase-level speech recognition. For open-vocabulary transcription, train a larger RNN/CTC model on a speech dataset such as LibriSpeech.
