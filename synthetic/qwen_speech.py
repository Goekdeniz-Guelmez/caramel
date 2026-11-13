"""Local MLX Audio Qwen3-TTS voice cloning for synthetic speaker tracks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synthetic.audio import prepare_mono_audio
from synthetic.audio import (
    Speech,
    SpeechAlignmentError,
    Word,
    transcript_match,
)

DEFAULT_QWEN_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"


@dataclass(frozen=True)
class VoiceReference:
    wav: Path
    text: str

    def __post_init__(self):
        import soundfile as sf

        path = Path(self.wav).expanduser().resolve()
        object.__setattr__(self, "wav", path)
        if not self.text.strip() or "REPLACE_WITH" in self.text:
            raise ValueError(
                "Replace the reference transcript placeholder with the exact words"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Replace the reference WAV placeholder with your recording: {path}"
            )
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"voice reference is empty: {path}")

    def identity(self) -> dict[str, str]:
        return {
            "audio_sha256": hashlib.sha256(self.wav.read_bytes()).hexdigest(),
            "transcript_sha256": hashlib.sha256(self.text.strip().encode()).hexdigest(),
        }


DEFAULT_QWEN_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


class Qwen3ClonedSpeech:
    """Synthesize and word-align cloned speech locally with MLX Audio.

    Loads one Qwen3-TTS model and one Qwen3 ForcedAligner lazily, then reuses
    each across both speaker roles. The aligner receives the known TTS text and
    generated WAV; no remote transcription or alignment service is used.
    """

    def __init__(
        self,
        cache_dir_or_legacy_client: Any,
        cache_dir: str | Path | None = None,
        *,
        user_reference: VoiceReference | None = None,
        assistant_reference: VoiceReference | None = None,
        model=DEFAULT_QWEN_TTS_MODEL,
        alignment_model=DEFAULT_QWEN_ALIGNER_MODEL,
        language="english",
        alignment_language=None,
        temperature=0.7,
        max_tokens=2048,
        loaded_model=None,
        loaded_aligner_model=None,
        min_alignment_match=0.9,
        sample_rate=24000,
    ):
        if temperature < 0 or max_tokens < 1:
            raise ValueError(
                "TTS temperature must be nonnegative and max_tokens positive"
            )
        if not 0 <= min_alignment_match <= 1:
            raise ValueError("alignment match threshold must be in [0,1]")
        if sample_rate != 24000:
            raise ValueError("synthetic speech must use a 24 kHz sample rate")
        # A leading alignment client was accepted by the old API. Ignore it so
        # existing callers can migrate without sending audio to that client.
        self.cache_dir = Path(
            cache_dir if cache_dir is not None else cache_dir_or_legacy_client
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.alignment_model = alignment_model
        self.references = {"user": user_reference, "assistant": assistant_reference}
        self.reference_ids = {
            role: ref.identity() if ref is not None else None
            for role, ref in self.references.items()
        }
        self.use_voice_references = True
        self.language = language
        self.alignment_language = alignment_language or language.title()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_alignment_match = min_alignment_match
        self.sample_rate = sample_rate
        self._tts_model = loaded_model
        self._aligner_model = loaded_aligner_model

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "mlx-audio-qwen3-tts-and-forced-aligner",
            "tts_model": self.model,
            "alignment_model": self.alignment_model,
            "alignment_type": "local_forced_alignment",
            "language": self.language,
            "alignment_language": self.alignment_language,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "min_alignment_match": self.min_alignment_match,
            "sample_rate": self.sample_rate,
            "references": {
                role: (
                    {"wav": str(ref.wav), **self.reference_ids[role]}
                    if ref is not None
                    else None
                )
                for role, ref in self.references.items()
            },
        }

    def cache_identity(self, text: str, role: str) -> dict[str, Any]:
        if role not in self.references:
            raise ValueError("speech role must be user or assistant")
        return {
            "provider": "mlx-audio-qwen3-tts-and-forced-aligner",
            "text": text,
            "role": role,
            "tts_model": self.model,
            "alignment_model": self.alignment_model,
            "reference": self.reference_ids[role],
            "language": self.language,
            "alignment_language": self.alignment_language,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "sample_rate": self.sample_rate,
            "format": "wav",
        }

    def synthesize_to_file(self, text: str, role: str, path: Path) -> None:
        import mlx.core as mx
        import soundfile as sf

        if role not in self.references:
            raise ValueError("speech role must be user or assistant")
        reference = self.references[role]
        if reference is None and self.use_voice_references:
            raise ValueError(f"Qwen TTS requires a voice reference for {role}")
        if reference is not None and reference.identity() != self.reference_ids[role]:
            raise ValueError(
                "voice reference changed during this run; create a new provider"
            )
        if self._tts_model is None:
            from mlx_audio.tts.utils import load_model

            self._tts_model = load_model(self.model)
        identity = self.cache_identity(text, role)
        seed = int(
            hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
                :8
            ],
            16,
        )
        mx.random.seed(seed)
        chunks = []
        rate = None
        for result in self._tts_model.generate(
            text=text,
            ref_audio=str(reference.wav),
            ref_text=reference.text.strip(),
            lang_code=self.language,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            verbose=False,
        ):
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            chunk_rate = int(result.sample_rate)
            if rate is not None and chunk_rate != rate:
                raise ValueError("Qwen TTS returned inconsistent audio sample rates")
            rate = chunk_rate
            if audio.size:
                chunks.append(audio)
        if not chunks:
            raise ValueError("Qwen TTS produced no audio")
        waveform = np.concatenate(chunks)
        if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) > 1.001:
            raise ValueError("Qwen TTS produced nonfinite or clipped audio")
        sf.write(path, waveform, rate, subtype="FLOAT")

    def _align(self, audio: Any, text: str) -> list[dict[str, Any]]:
        if self._aligner_model is None:
            from mlx_audio.stt import load

            self._aligner_model = load(self.alignment_model)
        try:
            result = self._aligner_model.generate(
                audio=audio, text=text, language=self.alignment_language
            )
        except Exception as exc:
            raise SpeechAlignmentError(
                f"MLX Audio forced aligner failed: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(result, dict):
            rows = result.get("items", [])
        else:
            rows = getattr(result, "items", result)
            if callable(rows):
                rows = result
        aligned = []
        last_start = 0.0
        for row in rows:
            if isinstance(row, dict):
                word = row.get("text", row.get("word", ""))
                start = row.get("start_time", row.get("start"))
                end = row.get("end_time", row.get("end"))
            else:
                word = getattr(row, "text", getattr(row, "word", ""))
                start = getattr(row, "start_time", getattr(row, "start", None))
                end = getattr(row, "end_time", getattr(row, "end", None))
            word = str(word).strip()
            if not word:
                continue
            try:
                start, end = float(start), float(end)
            except (TypeError, ValueError) as exc:
                raise SpeechAlignmentError(
                    "MLX Audio returned a word without usable timestamps"
                ) from exc
            if (
                not np.isfinite([start, end]).all()
                or not last_start <= start <= end
            ):
                raise SpeechAlignmentError(
                    "MLX Audio returned invalid or unordered word times"
                )
            aligned.append({"word": word, "start": start, "end": end})
            last_start = start
        if not aligned:
            raise SpeechAlignmentError(
                "MLX Audio forced aligner returned no word timings"
            )
        return aligned

    @staticmethod
    def _active_audio_bounds(waveform: np.ndarray, sample_rate: int) -> tuple[int, int]:
        """Estimate speech bounds from short-time energy, retaining a safety pad."""
        frame_samples = max(1, round(sample_rate * 0.02))
        frame_count = (len(waveform) + frame_samples - 1) // frame_samples
        padded = np.pad(waveform, (0, frame_count * frame_samples - len(waveform)))
        frames = padded.reshape(frame_count, frame_samples)
        energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        peak = float(energy.max())
        if peak <= 1e-6:
            return 0, len(waveform)
        active = np.flatnonzero(energy > max(1e-4, peak * 0.025))
        if active.size == 0:
            return 0, len(waveform)
        pad = round(sample_rate * 0.15)
        start = max(0, int(active[0]) * frame_samples - pad)
        end = min(len(waveform), (int(active[-1]) + 1) * frame_samples + pad)
        return start, end

    def _validate_alignment(
        self,
        aligned: list[dict[str, Any]],
        text: str,
        duration: float,
        active_start: float,
        active_end: float,
    ) -> tuple[list[Word], float]:
        words = []
        last_start = 0.0
        for row in aligned:
            try:
                word = str(row["word"]).strip()
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SpeechAlignmentError(
                    "local forced-alignment cache contains an invalid word row"
                ) from exc
            if (
                not word
                or not np.isfinite([start, end]).all()
                or not last_start <= start <= end <= duration + 0.03
            ):
                raise SpeechAlignmentError(
                    "local forced alignment returned invalid or nonmonotonic timestamps"
                )
            words.append(Word(word, start, min(end, duration)))
            last_start = start
        if not words:
            raise SpeechAlignmentError(
                "local forced alignment returned no measured words"
            )
        match = transcript_match(text, " ".join(word.word for word in words))
        if match < self.min_alignment_match:
            raise SpeechAlignmentError(
                f"aligned text differs from TTS input (coverage={match:.3f})"
            )
        # Forced aligners can return ordered, in-range timestamps that still land
        # well inside the utterance. Reject that silent offset failure as well.
        if (
            abs(words[0].start - active_start) > 1.25
            or abs(words[-1].end - active_end) > 1.25
        ):
            raise SpeechAlignmentError(
                "word timings do not cover the audible start and end of the utterance"
            )
        return words, match

    def _align_with_recovery(
        self, waveform: np.ndarray, text: str, duration: float
    ) -> tuple[list[dict[str, Any]], list[Word], float]:
        aligner_rate = 16_000
        start, end = self._active_audio_bounds(waveform, self.sample_rate)
        active_start, active_end = start / self.sample_rate, end / self.sample_rate
        full_audio = prepare_mono_audio(
            waveform, self.sample_rate, target_sample_rate=aligner_rate
        )
        candidates = [(full_audio, 0.0)]
        if start > 0 or end < len(waveform):
            trimmed = prepare_mono_audio(
                waveform[start:end], self.sample_rate, target_sample_rate=aligner_rate
            )
            if len(trimmed) >= aligner_rate // 4 and (
                start > self.sample_rate // 10
                or len(waveform) - end > self.sample_rate // 10
            ):
                candidates.append((trimmed, active_start))

        errors = []
        for audio, offset in candidates:
            try:
                rows = self._align(audio, text)
                if offset:
                    rows = [
                        {
                            **row,
                            "start": row["start"] + offset,
                            "end": row["end"] + offset,
                        }
                        for row in rows
                    ]
                words, match = self._validate_alignment(
                    rows, text, duration, active_start, active_end
                )
                return rows, words, match
            except (SpeechAlignmentError, ValueError, TypeError, RuntimeError) as exc:
                errors.append(str(exc))
        reason = "; retry also failed: ".join(errors)
        raise SpeechAlignmentError(
            f"forced alignment failed quality checks: {reason}"
        )

    def synthesize(self, text: str, role: str) -> Speech:
        import soundfile as sf

        if role not in self.references:
            raise ValueError("speech role must be user or assistant")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("speech text must be nonempty text")
        if len(text) > 4096:
            raise ValueError(
                f"speech text has {len(text)} characters; limit is 4096"
            )
        reference = self.references[role]
        if reference is None and self.use_voice_references:
            raise ValueError(f"Qwen TTS requires a voice reference for {role}")
        if reference is not None and reference.identity() != self.reference_ids[role]:
            raise ValueError("voice reference changed; create a new speech provider")
        identity = self.cache_identity(text, role)
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        audio_path = self.cache_dir / f"{key}.wav"
        if not audio_path.exists():
            temporary_audio = audio_path.with_suffix(".partial.wav")
            self.synthesize_to_file(text, role, temporary_audio)
            sf.info(temporary_audio)
            temporary_audio.replace(audio_path)
        waveform, source_rate = sf.read(
            audio_path, dtype="float32", always_2d=True
        )
        waveform = prepare_mono_audio(
            waveform, source_rate, target_sample_rate=self.sample_rate
        )
        duration = len(waveform) / self.sample_rate
        if duration <= 0 or np.max(np.abs(waveform)) > 1.001:
            raise ValueError("empty or clipped speech waveform")

        alignment_path = self.cache_dir / f"{key}.words.json"
        active_start_sample, active_end_sample = self._active_audio_bounds(
            waveform, self.sample_rate
        )
        active_start = active_start_sample / self.sample_rate
        active_end = active_end_sample / self.sample_rate
        aligned_rows = None
        words = None
        match = None
        if alignment_path.exists():
            try:
                cached = json.loads(alignment_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    cached_rows = cached.get("words")
                else:
                    cached_rows = cached  # Older caches stored a bare list.
                if not isinstance(cached_rows, list):
                    raise SpeechAlignmentError("cached alignment has an invalid format")
                words, match = self._validate_alignment(
                    cached_rows, text, duration, active_start, active_end
                )
                aligned_rows = cached_rows
            except (json.JSONDecodeError, UnicodeDecodeError, SpeechAlignmentError):
                # Bad cached timestamps must not poison every retry of this utterance.
                alignment_path.unlink(missing_ok=True)
        if aligned_rows is None:
            try:
                aligned_rows, words, match = self._align_with_recovery(
                    waveform, text, duration
                )
            except SpeechAlignmentError as exc:
                unaligned_speech = Speech(
                    waveform,
                    self.sample_rate,
                    [],
                    text,
                    str(audio_path.resolve()),
                    0.0,
                )
                raise SpeechAlignmentError(
                    str(exc), role=role, partial_speech=unaligned_speech
                ) from exc
            temporary_alignment = alignment_path.with_suffix(".partial.json")
            temporary_alignment.write_text(
                json.dumps({"words": aligned_rows}, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_alignment.replace(alignment_path)
        return Speech(
            waveform, self.sample_rate, words, text, str(audio_path.resolve()), match
        )
