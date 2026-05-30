# Speech-to-Text System using Recurrent Neural Networks (RNN)

## Overview

This project demonstrates a speech recognition workflow that converts audio input into text output. It uses Python-based audio preprocessing, sequence feature extraction, and a lightweight Recurrent Neural Network (RNN) classifier.

The project is designed to run easily in VS Code. It includes scripts for creating a demo audio dataset, training an RNN model, evaluating model accuracy, and transcribing audio files.

## Technologies Used

- Python
- NumPy
- SpeechRecognition
- Natural Language Processing
- Deep Learning
- Recurrent Neural Networks
- Audio Processing
- Sequence Modeling

## Input and Output

### Input

The system accepts audio files such as:

- `.wav`
- `.flac`
- `.aiff`
- `.aifc`

Example input:

```text
data/raw/sample_hello_world.wav
```

### Output

The system returns a text transcript.

Example output:

```text
hello world
```

## Project Structure

```text
minor-project-o-speech-to-text/
  .vscode/                         VS Code run and task configuration
  data/
    phrases.txt                    Demo training phrases
    raw/                           Generated or uploaded audio files
  models/                          Saved trained model files
  scripts/
    make_tone_dataset.py           Creates demo WAV files and manifest
    create_sapi_dataset.ps1        Creates Windows text-to-speech WAV files
  src/
    audio_features.py              Audio loading and log-spectrogram features
    rnn_model.py                   Simple RNN classifier implementation
    train.py                       Trains the RNN model
    evaluate.py                    Evaluates model accuracy
    transcribe.py                  Transcribes using trained RNN model
    transcribe_uploaded_audio.py   Transcribes uploaded audio files
  requirements.txt
  README.md
```

## Setup

Open PowerShell or the VS Code terminal inside the project folder:

```powershell
cd "C:\Users\missa\OneDrive\Documents\Github Projects\minor-project-o-speech-to-text"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

Create the demo audio dataset:

```powershell
python scripts\make_tone_dataset.py
```

Train the RNN model:

```powershell
python src\train.py --manifest data\raw\manifest.csv --model models\rnn_speech_model.npz --epochs 300 --hidden-size 64 --learning-rate 0.01
```

Evaluate the model:

```powershell
python src\evaluate.py --manifest data\raw\manifest.csv --model models\rnn_speech_model.npz
```

Transcribe a demo audio file:

```powershell
python src\transcribe.py --audio data\raw\sample_hello_world.wav --model models\rnn_speech_model.npz
```

## Transcribe Your Own Audio

For general speech transcription, run:

```powershell
python src\transcribe_uploaded_audio.py --audio "C:\path\to\your_audio.wav"
```

For the trained RNN demo model, run:

```powershell
python src\transcribe_uploaded_audio.py --engine rnn --audio data\raw\sample_hello_world.wav --model models\rnn_speech_model.npz
```

## Example Result

```text
Transcription: hello world
Confidence: 47.15%
```

## Important Notes

- The RNN model in this project is phrase-level and recognizes phrases it was trained on.
- The general uploaded-audio transcription command uses the `google` engine from the `SpeechRecognition` package and requires an internet connection.
- Generated audio files and trained model files are not uploaded to GitHub. They are created locally when you run the project.
- For a production-level open-vocabulary speech recognition system, a larger dataset such as LibriSpeech and a deep learning framework such as PyTorch or TensorFlow should be used.

## Skills Demonstrated

- Python programming
- Audio preprocessing
- Log-spectrogram feature extraction
- Sequence modeling
- Recurrent Neural Networks
- Model training and evaluation
- Speech recognition workflow
- Command-line project execution
