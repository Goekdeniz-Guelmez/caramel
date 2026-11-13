"""Generic TTS adapter for models supported by ``mlx-audio``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from synthetic.audio import Speech, SpeechAlignmentError, Word
from synthetic.qwen_speech import (
    DEFAULT_QWEN_TTS_MODEL,
    Qwen3ClonedSpeech,
)

CHUNK_PAUSE_MS = 120


def _split_text(text: str, limit: int) -> list[str]:
    remaining = text.strip()
    chunks = []
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit + 1)
        for punctuation in (". ", "? ", "! ", "; ", ", "):
            boundary = remaining.rfind(punctuation, 0, limit + 1)
            if boundary >= 0:
                cut = max(cut, boundary + 1)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


class MLXAudioSpeech(Qwen3ClonedSpeech):
    """Use ``mlx-audio`` for TTS and the configured local aligner for timing.

    ``generation_kwargs`` and ``role_kwargs`` are passed to the selected model's
    ``generate`` method. Set ``use_voice_references=False`` for preset-voice
    models such as Kokoro, then provide role-specific ``voice`` values.
    """

    def __init__(
        self,
        *args,
        generation_kwargs: dict[str, Any] | None = None,
        role_kwargs: dict[str, dict[str, Any]] | None = None,
        reference_kwargs: dict[str, str] | None = None,
        use_voice_references: bool = True,
        max_text_chars: int = 4096,
        **kwargs,
    ):
        kwargs.setdefault("model", DEFAULT_QWEN_TTS_MODEL)
        if max_text_chars < 1 or max_text_chars > 4096:
            raise ValueError("max_text_chars must be between 1 and 4096")
        if type(use_voice_references) is not bool:
            raise ValueError("use_voice_references must be a boolean")
        super().__init__(*args, **kwargs)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.role_kwargs = {
            role: dict(options) for role, options in (role_kwargs or {}).items()
        }
        self.reference_kwargs = (
            dict(reference_kwargs)
            if reference_kwargs is not None
            else {"audio": "ref_audio", "text": "ref_text"}
        )
        if set(self.reference_kwargs) != {"audio", "text"} or any(
            not isinstance(name, str) or not name
            for name in self.reference_kwargs.values()
        ):
            raise ValueError("reference_kwargs must map audio and text to argument names")
        unknown_roles = set(self.role_kwargs) - set(self.references)
        if unknown_roles:
            raise ValueError(f"unknown TTS roles: {sorted(unknown_roles)}")
        self.use_voice_references = use_voice_references
        self.max_text_chars = max_text_chars

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "provider": "mlx-audio-generic-tts-and-forced-aligner",
                "generation_kwargs": self.generation_kwargs,
                "role_kwargs": self.role_kwargs,
                "reference_kwargs": self.reference_kwargs,
                "use_voice_references": self.use_voice_references,
                "max_text_chars": self.max_text_chars,
            }
        )
        if not self.use_voice_references:
            metadata["references"] = {"user": None, "assistant": None}
        return metadata

    def cache_identity(self, text: str, role: str) -> dict[str, Any]:
        identity = super().cache_identity(text, role)
        identity.update(
            {
                "provider": "mlx-audio-generic-tts-and-forced-aligner",
                "reference": (
                    self.reference_ids[role] if self.use_voice_references else None
                ),
                "generation_kwargs": self.generation_kwargs,
                "role_kwargs": self.role_kwargs.get(role, {}),
                "reference_kwargs": self.reference_kwargs,
                "use_voice_references": self.use_voice_references,
                "max_text_chars": self.max_text_chars,
            }
        )
        return identity

    def synthesize(self, text: str, role: str) -> Speech:
        """Align each provider-sized segment, then join its audio and timings."""
        if not isinstance(text, str) or len(text) <= self.max_text_chars:
            return super().synthesize(text, role)
        chunks = _split_text(text, self.max_text_chars)
        if len(chunks) <= 1:
            return super().synthesize(text, role)

        parts = []
        for index, chunk in enumerate(chunks):
            try:
                parts.append(super().synthesize(chunk, role))
            except SpeechAlignmentError as exc:
                raise SpeechAlignmentError(
                    f"{role} speech segment {index + 1}/{len(chunks)} failed: {exc}",
                    role=role,
                ) from exc
        rate = parts[0].sample_rate
        if any(part.sample_rate != rate for part in parts):
            raise ValueError("mlx-audio returned inconsistent sample rates")
        pause = np.zeros(round(rate * CHUNK_PAUSE_MS / 1000), dtype=np.float32)
        audio = []
        words = []
        offset = 0
        for index, part in enumerate(parts):
            if index:
                audio.append(pause)
                offset += len(pause)
            words.extend(
                Word(word.word, word.start + offset / rate, word.end + offset / rate)
                for word in part.words
            )
            audio.append(part.waveform)
            offset += len(part.waveform)
        return Speech(
            np.concatenate(audio),
            rate,
            words,
            text,
            ";".join(part.cache_file for part in parts),
            min(part.alignment_match for part in parts),
        )

    def synthesize_to_file(self, text: str, role: str, path: Path) -> None:
        import mlx.core as mx
        import soundfile as sf

        if role not in self.references:
            raise ValueError("speech role must be user or assistant")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("speech text must be nonempty text")
        if len(text) > self.max_text_chars:
            raise ValueError(
                f"speech text has {len(text)} characters; selected model limit is "
                f"{self.max_text_chars}"
            )
        reference = self.references[role]
        if self.use_voice_references:
            if reference is None:
                raise ValueError(f"a voice reference is required for {role}")
            if reference.identity() != self.reference_ids[role]:
                raise ValueError("voice reference changed; create a new provider")

        if self._tts_model is None:
            from mlx_audio.tts.utils import load_model

            self._tts_model = load_model(self.model)
        kwargs = dict(self.generation_kwargs)
        kwargs.update(self.role_kwargs.get(role, {}))
        if self.use_voice_references:
            kwargs.setdefault(self.reference_kwargs["audio"], str(reference.wav))
            kwargs.setdefault(
                self.reference_kwargs["text"], reference.text.strip()
            )

        identity = self.cache_identity(text, role)
        seed = int(
            hashlib.sha256(
                json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:8],
            16,
        )
        mx.random.seed(seed)
        results = self._tts_model.generate(text=text, **kwargs)
        if hasattr(results, "audio"):
            results = [results]
        audio_chunks = []
        sample_rate = None
        for result in results:
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            result_rate = int(result.sample_rate)
            if sample_rate is not None and result_rate != sample_rate:
                raise ValueError("mlx-audio returned inconsistent sample rates")
            sample_rate = result_rate
            if audio.size:
                audio_chunks.append(audio)
        if not audio_chunks or sample_rate is None:
            raise ValueError("mlx-audio produced no audio")
        waveform = np.concatenate(audio_chunks)
        if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) > 1.001:
            raise ValueError("mlx-audio produced nonfinite or clipped audio")
        sf.write(path, waveform, sample_rate, subtype="FLOAT")
