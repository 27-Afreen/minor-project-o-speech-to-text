from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: str | Path, target_rate: int = 16000) -> tuple[int, np.ndarray]:
    """Read a PCM WAV file, mix to mono, normalize, and resample if needed."""
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_rate:
        audio = _resample_linear(audio, sample_rate, target_rate)
        sample_rate = target_rate

    return sample_rate, audio.astype(np.float32)


def extract_log_spectrogram(
    path: str | Path,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    fft_size: int = 512,
    max_frames: int = 180,
) -> np.ndarray:
    """Convert audio into normalized log-spectrogram frames for RNN input."""
    rate, audio = read_wav_mono(path, sample_rate)
    if audio.size == 0:
        raise ValueError(f"No audio samples found in {path}")

    audio = _trim_silence(audio)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    frame_length = int(rate * frame_ms / 1000)
    hop_length = int(rate * hop_ms / 1000)
    frames = _frame_signal(audio, frame_length, hop_length)
    window = np.hamming(frame_length).astype(np.float32)
    spectra = np.fft.rfft(frames * window, n=fft_size)
    power = (np.abs(spectra) ** 2).astype(np.float32)
    features = np.log1p(power)

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std

    return _pad_or_trim(features, max_frames).astype(np.float32)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    duration = audio.size / source_rate
    source_times = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    target_size = max(1, int(duration * target_rate))
    target_times = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_times, source_times, audio).astype(np.float32)


def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    indices = np.where(mask)[0]
    start = max(0, int(indices[0]) - 800)
    end = min(audio.size, int(indices[-1]) + 800)
    return audio[start:end]


def _frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))

    frame_count = 1 + int(np.ceil((audio.size - frame_length) / hop_length))
    padded_length = frame_length + (frame_count - 1) * hop_length
    if padded_length > audio.size:
        audio = np.pad(audio, (0, padded_length - audio.size))

    frames = np.empty((frame_count, frame_length), dtype=np.float32)
    for i in range(frame_count):
        start = i * hop_length
        frames[i] = audio[start : start + frame_length]
    return frames


def _pad_or_trim(features: np.ndarray, max_frames: int) -> np.ndarray:
    if features.shape[0] > max_frames:
        return features[:max_frames]
    if features.shape[0] < max_frames:
        pad = np.zeros((max_frames - features.shape[0], features.shape[1]), dtype=features.dtype)
        return np.vstack([features, pad])
    return features
