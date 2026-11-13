"""Cached OpenAI speech synthesis and measured word timestamps."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np


def prepare_mono_audio(waveform: Any, sample_rate: int, *, target_sample_rate: int = 24_000) -> np.ndarray:
    """Finite mono float32 at target_sample_rate (channels-last 2-D input is averaged; integer PCM scaled to [-1, 1])."""
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    audio = np.asarray(waveform)
    if audio.ndim == 0:
        raise ValueError("waveform must contain at least one sample")
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / float(max(abs(info.min), abs(info.max)))
    else:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        if 0 in audio.shape:
            raise ValueError("waveform cannot have an empty channel or time axis")
        audio = audio.mean(axis=1 if audio.shape[0] > audio.shape[1] else 0)
    elif audio.ndim != 1:
        raise ValueError("waveform must be mono or a two-dimensional channel array")
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("waveform is empty or contains NaN/inf samples")
    if sample_rate == target_sample_rate:
        return np.ascontiguousarray(audio)
    from scipy.signal import resample_poly
    d = gcd(sample_rate, target_sample_rate)
    return np.ascontiguousarray(resample_poly(audio, target_sample_rate // d, sample_rate // d), dtype=np.float32)

class SpeechAlignmentError(ValueError):
    """Raised when measured word timings do not pass alignment quality checks."""

    def __init__(self, message: str, *, role: str | None = None, partial_speech=None):
        super().__init__(message)
        self.role = role
        self.partial_speech = partial_speech


@dataclass(frozen=True)
class Word:
    word: str
    start: float
    end: float


@dataclass
class Speech:
    waveform: np.ndarray
    sample_rate: int
    words: list[Word]
    source_text: str
    cache_file: str
    alignment_match: float = 1.0
    cut_mid_word: bool = False

    @property
    def duration(self) -> float:
        return len(self.waveform) / self.sample_rate

    @property
    def spoken_text(self) -> str:
        return " ".join(word.word.strip() for word in self.words).strip()


def transcript_match(expected: str, observed: str) -> float:
    def normalize(text):
        return re.findall(r"\w+", text.casefold())

    return SequenceMatcher(
        None, normalize(expected), normalize(observed), autojunk=False
    ).ratio()


def stop_after_barge_in(
    speech: Speech, onset: float, stop_ms: int, *, fade_ms: int = 180
) -> Speech:
    """Stop at the reaction deadline with a smooth fade to silence.

    The deadline is relative to the user's interruption onset, not the beginning
    of the assistant utterance. The final fade_ms (or remaining post-onset time)
    fades to silence smoothly, and a partially spoken final word is marked.
    """
    if not 0 <= onset < speech.duration or stop_ms < 0 or fade_ms < 0:
        raise ValueError("invalid interruption onset or stop budget")
    deadline = min(speech.duration, onset + stop_ms / 1000)
    if deadline >= speech.duration:
        return speech
    stop = deadline
    samples = min(int(np.floor(stop * speech.sample_rate + 1e-8)), len(speech.waveform))
    stop = samples / speech.sample_rate
    words = [word for word in speech.words if word.end <= stop + 1e-8]
    cut_mid_word = any(word.start < stop < word.end - 1e-8 for word in speech.words)
    waveform = speech.waveform[:samples].copy()
    onset_sample = min(round(onset * speech.sample_rate), samples)
    fade_samples = min(
        round(speech.sample_rate * fade_ms / 1000),
        max(0, samples - onset_sample),
    )
    if fade_samples:
        fade = 0.5 * (
            1.0
            + np.cos(np.linspace(0.0, np.pi, fade_samples, endpoint=True))
        )
        if fade_samples == 1:
            fade[0] = 0.0
        waveform[-fade_samples:] *= fade.astype(waveform.dtype, copy=False)
    return Speech(
        waveform,
        speech.sample_rate,
        words,
        speech.source_text,
        speech.cache_file,
        speech.alignment_match,
        cut_mid_word,
    )


class OpenAISpeech:
    """Injected OpenAI-compatible client; no key is stored in output metadata.

    Word timings are Whisper transcriptions, not forced alignment of supplied
    text. A transcript match check rejects mismatched utterances; all measured
    words and the original text remain separate in the annotations.
    """

    def __init__(
        self,
        client: Any,
        cache_dir: str | Path,
        *,
        model="gpt-4o-mini-tts",
        user_voice="alloy",
        assistant_voice="coral",
        alignment_model="whisper-1",
        min_alignment_match=0.9,
        sample_rate=24000,
    ):
        if alignment_model != "whisper-1":
            raise ValueError("word timestamp extraction requires whisper-1")
        if not 0 <= min_alignment_match <= 1:
            raise ValueError("alignment match threshold must be in [0,1]")
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.voices = {"user": user_voice, "assistant": assistant_voice}
        self.alignment_model = alignment_model
        self.min_alignment_match = min_alignment_match
        self.sample_rate = sample_rate

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "openai-compatible",
            "tts_model": self.model,
            "voices": self.voices,
            "word_timestamp_model": self.alignment_model,
            "min_alignment_match": self.min_alignment_match,
            "sample_rate": self.sample_rate,
            "alignment_type": "transcription_word_timestamps",
        }

    def cache_identity(self, text: str, role: str) -> dict[str, Any]:
        return {
            "provider": "openai-compatible",
            "text": text,
            "voice": self.voices[role],
            "model": self.model,
            "format": "wav",
            "alignment_model": self.alignment_model,
        }

    def synthesize_to_file(self, text: str, role: str, path: Path) -> None:
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model, voice=self.voices[role], input=text, response_format="wav"
        ) as response:
            response.stream_to_file(path)

    def synthesize(self, text: str, role: str) -> Speech:
        import soundfile as sf

        if role not in self.voices or not text.strip() or len(text) > 4096:
            raise ValueError("speech needs a valid role and 1..4096 characters")
        identity = self.cache_identity(text, role)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        audio_path = self.cache_dir / f"{key}.wav"
        if not audio_path.exists():
            temporary = audio_path.with_suffix(".partial.wav")
            self.synthesize_to_file(text, role, temporary)
            sf.info(temporary)
            temporary.replace(audio_path)
        waveform, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = prepare_mono_audio(
            waveform, rate, target_sample_rate=self.sample_rate
        )
        duration = len(waveform) / self.sample_rate
        if duration <= 0 or np.max(np.abs(waveform)) > 1.001:
            raise ValueError("empty or clipped speech waveform")
        alignment_path = self.cache_dir / f"{key}.words.json"
        if alignment_path.exists():
            aligned = json.loads(alignment_path.read_text())
        else:
            with audio_path.open("rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.alignment_model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    prompt=text,
                )
            aligned = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            if not isinstance(aligned, dict):
                raise ValueError("word timestamp API must return verbose JSON")
            temporary = alignment_path.with_suffix(".partial.json")
            temporary.write_text(
                json.dumps(aligned, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(alignment_path)
        words = []
        last_start = 0.0
        for row in aligned.get("words", []):
            start, end = float(row["start"]), float(row["end"])
            if not row["word"].strip() or not (
                last_start <= start <= end <= duration + 0.03
            ):
                raise ValueError("invalid or nonmonotonic word timestamps")
            words.append(Word(row["word"], start, min(end, duration)))
            last_start = start
        if not words:
            raise ValueError("speech alignment returned no measured words")
        match = transcript_match(text, " ".join(word.word for word in words))
        if match < self.min_alignment_match:
            raise ValueError(
                f"alignment transcript differs from TTS input (match={match:.3f})"
            )
        return Speech(
            waveform, self.sample_rate, words, text, str(audio_path.resolve()), match
        )
