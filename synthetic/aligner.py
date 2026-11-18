"""Cached TTS + word timestamps from a local forced aligner (Qwen3-ForcedAligner via MLX Audio).

Subclasses provide `cache_identity(text, role)` and `synthesize_to_file(text, role, path)`; this base caches each
utterance's WAV and word timings on disk, validates the alignment against the known text and retries on trimmed
audio before giving up.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from synthetic.audio import Speech, SpeechAlignmentError, Word, prepare_mono_audio, transcript_match

DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


# MLX models (TTS, aligner) are not safe to call from several threads at once; parallel episodes share this lock,
# so their teacher requests overlap with each other's GPU work instead.
GPU_LOCK = threading.RLock()


class AlignedSpeech:
    """Base for speech providers: disk cache + forced alignment. Needs cache_dir, sample_rate, references,
    alignment_model, alignment_language, min_alignment_match and _aligner_model set by the subclass."""

    cache_dir: Path
    sample_rate: int
    references: dict[str, Any]

    def cache_identity(self, text: str, role: str) -> dict[str, Any]:
        raise NotImplementedError

    def synthesize_to_file(self, text: str, role: str, path: Path) -> None:
        raise NotImplementedError

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
        identity = self.cache_identity(text, role)
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        audio_path = self.cache_dir / f"{key}.wav"
        if not audio_path.exists():
            temporary_audio = audio_path.with_suffix(".partial.wav")
            with GPU_LOCK:
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
                with GPU_LOCK:
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
