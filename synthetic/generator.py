"""Render multi-turn speech episodes onto a synchronized dual-speaker timeline."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synthetic.audio import (
    Speech,
    SpeechAlignmentError,
    Word,
    stop_after_barge_in,
)

from synthetic.teacher import TeacherTimeoutError

MAX_SPEECH_TEXT_CHARS = 4096
SPEECH_CHUNK_PAUSE_MS = 120


def _split_speech_text(
    text: str, max_chars: int = MAX_SPEECH_TEXT_CHARS
) -> list[str]:
    """Split long TTS input without dropping words or punctuation."""
    if max_chars < 1 or max_chars > MAX_SPEECH_TEXT_CHARS:
        raise ValueError("speech chunk size must be between 1 and 4096 characters")
    remaining = text.strip()
    chunks = []
    while len(remaining) > max_chars:
        limit = max_chars
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


def _barge_in_anchor_end(
    speech: Speech, anchor: str | None, target_onset: float | None = None
) -> float | None:
    """Return the end time of an exact aligned word or phrase from the answer."""
    if not isinstance(anchor, str) or not anchor.strip():
        return None
    wanted = re.findall(r"\w+", anchor.casefold())
    if not wanted:
        return None
    aligned_tokens = [
        (token, word)
        for word in speech.words
        for token in re.findall(r"\w+", word.word.casefold())
    ]
    matches = []
    for start in range(len(aligned_tokens) - len(wanted) + 1):
        candidate = [
            token for token, _ in aligned_tokens[start : start + len(wanted)]
        ]
        if candidate == wanted:
            end = aligned_tokens[start + len(wanted) - 1][1].end
            if 0 < end < speech.duration:
                matches.append(end)
    if not matches:
        return None
    if target_onset is None:
        return matches[0]
    return min(matches, key=lambda end: abs(end - target_onset))


@dataclass(frozen=True)
class SyntheticConfig:
    max_turns: int = 8
    barge_in_probability: float = 0.2
    assistant_stop_ms: int = 1000
    assistant_fade_ms: int = 180
    system_prompt: str = (
        "You are JOSIE created by Gökdeniz Gülmez and you are talking to the "
        "assistant so no markdown, code or math"
    )
    sample_rate: int = 24000
    seed: int = 42
    followup_mode: str = "model"
    response_gap_ms: int = 200
    turn_gap_ms: int = 250
    leading_silence_ms: int = 160
    trailing_silence_ms: int = 240
    validation_fraction: float = 0.1
    wait_probability: float = 0.15  # "can you wait a minute?" hold scene
    listen_probability: float = 0.1  # "listen until I finish my presentation" scene
    max_wait_seconds: float = 60.0
    check_in_probability: float = 0.5  # assistant asks "time is up, can I continue?"
    backchannel_probability: float = 0.5  # per long user turn: "mhm", "right", ...
    backchannel_min_seconds: float = 4.0

    def __post_init__(self):
        if type(self.max_turns) is not int or self.max_turns < 2:
            raise ValueError("max_turns must be at least 2")
        if not 0 <= self.barge_in_probability <= 1:
            raise ValueError("barge-in probability must lie in [0,1]")
        if (
            min(
                self.assistant_stop_ms,
                self.assistant_fade_ms,
                self.response_gap_ms,
                self.turn_gap_ms,
                self.leading_silence_ms,
                self.trailing_silence_ms,
            )
            < 0
        ):
            raise ValueError("timing parameters must be nonnegative")
        if self.sample_rate != 24000 or not 0 <= self.validation_fraction < 1:
            raise ValueError("use 24 kHz audio and validation fraction in [0,1)")
        for p in (self.wait_probability, self.listen_probability, self.check_in_probability,
                  self.backchannel_probability):
            if not 0 <= p <= 1:
                raise ValueError("scene probabilities must lie in [0,1]")
        if self.wait_probability + self.listen_probability > 1 or self.max_wait_seconds < 10:
            raise ValueError("wait+listen probability must be <= 1 and max_wait_seconds >= 10")
        if self.followup_mode not in {"model", "repeat_seed"}:
            raise ValueError("followup_mode must be model or repeat_seed")


@dataclass
class Utterance:
    role: str
    turn: int
    start_sample: int
    speech: Speech
    full_speech: Speech

    @property
    def end_sample(self):
        return self.start_sample + len(self.speech.waveform)


class SyntheticConversationGenerator:
    def __init__(
        self,
        teacher: Any,
        speech: Any,
        output_dir: str | Path,
        config: SyntheticConfig,
        *,
        source_metadata: dict[str, Any],
    ):
        self.teacher = teacher
        self.speech = speech
        self.output = Path(output_dir)
        self.config = config
        self.metadata = {
            "schema": "josie-synthetic-duplex-v1",
            "config": asdict(config),
            "source": source_metadata,
            "teacher": teacher.metadata(),
            "speech": speech.metadata(),
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(self.metadata, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        self.output.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output / "generation_config.json"
        if metadata_path.exists():
            if (
                json.loads(metadata_path.read_text()).get("fingerprint")
                != self.fingerprint
            ):
                raise ValueError(
                    "generation settings differ; choose a new output directory"
                )
        else:
            if (self.output / "episodes").exists():
                raise ValueError(
                    "existing episodes have no matching generation configuration"
                )
            metadata_path.write_text(
                json.dumps(
                    {**self.metadata, "fingerprint": self.fingerprint},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (self.output / "episodes").mkdir(exist_ok=True)

    def generate_episode(
        self, seed_prompt: str, row_index: int, *, progress=None
    ) -> dict[str, Any]:
        if not isinstance(seed_prompt, str) or not seed_prompt.strip():
            raise ValueError("seed prompt must be nonempty text")
        final = self.output / "episodes" / f"{row_index:06d}"
        existing = final / "annotations.json"
        if existing.exists():
            record = json.loads(existing.read_text())
            if (
                record["fingerprint"] != self.fingerprint
                or record["seed_prompt"] != seed_prompt
            ):
                raise ValueError(
                    "existing episode does not match its prompt/configuration"
                )
            if not (final / "dialogue.wav").is_file():
                raise ValueError("completed episode has no stereo audio")
            return record
        if final.exists():
            raise ValueError(f"unrecognized incomplete episode directory: {final}")
        episode_seed = int.from_bytes(
            hashlib.sha256(
                f"{self.config.seed}:{row_index}:{seed_prompt}".encode()
            ).digest()[:8],
            "big",
        )
        rng = random.Random(episode_seed)
        if hasattr(self.speech, "begin_episode"):  # per-conversation user voice (Phonon pool)
            self.speech.begin_episode(random.Random(episode_seed ^ 0x5EED_FACE))
        turns = rng.randint(2, self.config.max_turns)
        history = [{"role": "system", "content": self.config.system_prompt}]
        utterances = []
        barge_ins = []
        scenes, backchannels = [], []
        rate = self.config.sample_rate
        from synthetic.scenes import SceneBuilder

        builder = SceneBuilder(self, rng, history, utterances, barge_ins, next_turn=100)
        draw = rng.random()
        scene = ("wait" if draw < self.config.wait_probability else
                 "listen" if draw < self.config.wait_probability + self.config.listen_probability
                 else None)
        scene_turn = rng.randint(1, turns - 1) if scene else -1

        def save_partial(*, failed_role: str, error: str):
            if not utterances or utterances[-1].role != "assistant":
                raise SpeechAlignmentError(
                    f"alignment failed for {failed_role} before an assistant output was available: {error}",
                    role=failed_role,
                )
            with tempfile.TemporaryDirectory(
                prefix=".building-", dir=self.output
            ) as temporary:
                staging = Path(temporary) / "episode"
                staging.mkdir()
                record = self._write_episode(
                    staging,
                    final,
                    utterances,
                    barge_ins,
                    seed_prompt,
                    row_index,
                    episode_seed,
                    complete=False,
                    failed_role=failed_role,
                    alignment_error=error,
                    scenes=scenes,
                    backchannels=backchannels,
                )
                staging.rename(final)
            return record

        for turn in range(turns):
            if turn == scene_turn:
                if progress:
                    progress(turn + 1, turns, f"{scene} scene")
                if scene == "wait":
                    scenes.append(builder.wait(seed_prompt))
                else:
                    start = utterances[-1].end_sample + round(self.config.turn_gap_ms * rate / 1000)
                    scenes.append(builder.listen(seed_prompt, start))
                continue
            user_start = round(self.config.leading_silence_ms * rate / 1000)
            heard_history = [dict(message) for message in history]
            previous = utterances[-1] if turn else None
            will_barge_in = bool(
                previous is not None
                and rng.random() < self.config.barge_in_probability
            )
            provisional_onset = None
            if will_barge_in:
                provisional_onset = previous.speech.duration * rng.uniform(0.25, 0.65)
                provisional_samples = max(
                    0,
                    min(
                        round(provisional_onset * rate),
                        len(previous.speech.waveform) - 1,
                    ),
                )
                provisional_onset = provisional_samples / rate
                heard_history[-1]["content"] = " ".join(
                    word.word
                    for word in previous.speech.words
                    if word.end <= provisional_onset
                )

            barge_in_word = None
            if turn == 0 or self.config.followup_mode == "repeat_seed":
                prompt = seed_prompt
            else:
                followup_method = self.teacher.followup
                try:
                    parameters = inspect.signature(followup_method).parameters
                    accepts_extra = any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                except (TypeError, ValueError):
                    parameters, accepts_extra = {}, True
                followup_kwargs = {"seed": rng.randrange(2**31)}
                if accepts_extra or "is_barge_in" in parameters:
                    followup_kwargs["is_barge_in"] = will_barge_in
                if accepts_extra or "assistant_response" in parameters:
                    followup_kwargs["assistant_response"] = (
                        previous.full_speech.source_text
                    )
                result = followup_method(
                    seed_prompt,
                    heard_history if will_barge_in else history,
                    **followup_kwargs,
                )
                if isinstance(result, dict):
                    prompt = result.get("prompt")
                    barge_in_word = result.get("barge_in_word")
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError("follow-up result needs a nonempty prompt")
                    prompt = prompt.strip()
                else:
                    prompt = result
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("follow-up teacher returned an empty prompt")

            if turn:
                if will_barge_in:
                    onset = _barge_in_anchor_end(
                        previous.speech, barge_in_word, provisional_onset
                    )
                    if onset is None:
                        onset = provisional_onset
                    onset_samples = round(onset * rate)
                    # Only words heard before interruption can inform the next assistant.
                    heard_history[-1]["content"] = " ".join(
                        word.word for word in previous.speech.words if word.end <= onset
                    )
                    previous.speech = stop_after_barge_in(
                        previous.speech,
                        onset,
                        self.config.assistant_stop_ms,
                        fade_ms=self.config.assistant_fade_ms,
                    )
                    history[-1]["content"] = previous.speech.spoken_text
                    user_start = previous.start_sample + onset_samples
                    barge_ins.append(
                        {
                            "user_turn": turn,
                            "assistant_turn": turn - 1,
                            "onset_sample": user_start,
                            "assistant_stop_sample": previous.end_sample,
                            "barge_in_word": barge_in_word,
                            "reaction_ms": (previous.end_sample - user_start)
                            * 1000
                            / rate,
                            "cut_mid_word": previous.speech.cut_mid_word,
                        }
                    )
                else:
                    user_start = previous.end_sample + round(
                        self.config.turn_gap_ms * rate / 1000
                    )
            if progress:
                progress(turn + 1, turns, "user speech")
            try:
                user_speech = self._synthesize(prompt, "user")
            except SpeechAlignmentError as exc:
                if barge_ins and barge_ins[-1]["user_turn"] == turn:
                    barge_ins.pop()
                    previous = utterances[-1]
                    previous.speech = previous.full_speech
                if not utterances:
                    raise
                return save_partial(failed_role="user", error=str(exc))
            self._validate_speech(user_speech)
            user = Utterance("user", turn, user_start, user_speech, user_speech)
            utterances.append(user)
            if (user_speech.duration >= self.config.backchannel_min_seconds
                    and rng.random() < self.config.backchannel_probability):
                last_ai = max((u.end_sample for u in utterances if u.role == "assistant"), default=0)
                backchannels += builder.backchannel(user, last_ai + round(0.3 * rate))
                utterances.remove(user)
                utterances.append(user)  # keep the user turn last for the answer logic
            history.append({"role": "user", "content": prompt})
            if progress:
                progress(turn + 1, turns, "Gemma response")
            answer = self.teacher.generate(
                [dict(message) for message in history], seed=rng.randrange(2**31)
            )
            if progress:
                progress(turn + 1, turns, "assistant speech")
            assistant_start = user.end_sample + round(
                self.config.response_gap_ms * rate / 1000
            )
            if turn:
                # Keep consecutive assistant utterances from overlapping.
                assistant_start = max(assistant_start, max(
                    u.end_sample for u in utterances if u.role == "assistant"))
            try:
                assistant_speech = self._synthesize(answer, "assistant")
            except SpeechAlignmentError as exc:
                assistant_speech = exc.partial_speech
                if assistant_speech is not None:
                    self._validate_audio(assistant_speech)
                    utterances.append(
                        Utterance(
                            "assistant",
                            turn,
                            assistant_start,
                            assistant_speech,
                            assistant_speech,
                        )
                    )
                    history.append({"role": "assistant", "content": answer})
                    return save_partial(failed_role="assistant", error=str(exc))
                utterances.pop()  # Drop the current user turn; it has no answer.
                if barge_ins and barge_ins[-1]["user_turn"] == turn:
                    barge_ins.pop()
                    previous = utterances[-1]
                    previous.speech = previous.full_speech
                if not utterances:
                    raise
                return save_partial(failed_role="assistant", error=str(exc))
            self._validate_speech(assistant_speech)
            utterances.append(
                Utterance(
                    "assistant",
                    turn,
                    assistant_start,
                    assistant_speech,
                    assistant_speech,
                )
            )
            history.append({"role": "assistant", "content": answer})
        with tempfile.TemporaryDirectory(
            prefix=".building-", dir=self.output
        ) as temporary:
            staging = Path(temporary) / "episode"
            staging.mkdir()
            record = self._write_episode(
                staging,
                final,
                utterances,
                barge_ins,
                seed_prompt,
                row_index,
                episode_seed,
                scenes=scenes,
                backchannels=backchannels,
            )
            staging.rename(final)
        return record

    def _validate_speech(self, speech):
        self._validate_audio(speech)
        if not speech.words or any(
            not (0 <= w.start <= w.end <= speech.duration + 1e-8) for w in speech.words
        ):
            raise ValueError("speech provider returned invalid/missing word alignment")

    def _validate_audio(self, speech):
        if (
            speech.sample_rate != self.config.sample_rate
            or speech.waveform.ndim != 1
            or speech.duration <= 0
            or not np.isfinite(speech.waveform).all()
            or np.max(np.abs(speech.waveform)) > 1.001
        ):
            raise ValueError("speech provider returned incompatible audio")

    def _synthesize(self, text: str, role: str) -> Speech:
        """Synthesize long utterances in provider-sized aligned chunks."""
        max_chars = min(
            MAX_SPEECH_TEXT_CHARS,
            getattr(self.speech, "max_text_chars", MAX_SPEECH_TEXT_CHARS),
        )
        chunks = _split_speech_text(text, max_chars)
        if len(chunks) <= 1:
            return self.speech.synthesize(text, role)

        waveforms = []
        words = []
        cache_files = []
        offset_samples = 0
        pause = np.zeros(
            round(self.config.sample_rate * SPEECH_CHUNK_PAUSE_MS / 1000),
            dtype=np.float32,
        )
        alignment_match = 1.0
        for index, chunk in enumerate(chunks):
            try:
                part = self.speech.synthesize(chunk, role)
            except SpeechAlignmentError as exc:
                # Partial speech from the provider covers only this chunk, not
                # the complete multi-chunk utterance.
                raise SpeechAlignmentError(
                    f"{role} speech chunk {index + 1}/{len(chunks)} failed: {exc}",
                    role=role,
                ) from exc
            self._validate_speech(part)
            if index:
                waveforms.append(pause)
                offset_samples += len(pause)
            words.extend(
                Word(
                    word.word,
                    word.start + offset_samples / self.config.sample_rate,
                    word.end + offset_samples / self.config.sample_rate,
                )
                for word in part.words
            )
            waveforms.append(part.waveform)
            offset_samples += len(part.waveform)
            cache_files.append(part.cache_file)
            alignment_match = min(alignment_match, part.alignment_match)
        return Speech(
            np.concatenate(waveforms),
            self.config.sample_rate,
            words,
            text,
            ";".join(cache_files),
            alignment_match,
        )

    def _write_episode(
        self,
        staging,
        final,
        utterances,
        barge_ins,
        seed_prompt,
        row_index,
        episode_seed,
        *,
        complete=True,
        failed_role=None,
        alignment_error=None,
        scenes=None,
        backchannels=None,
    ):
        import soundfile as sf

        rate = self.config.sample_rate
        frame_samples = rate * 80 // 1000
        samples = max(u.end_sample for u in utterances) + round(
            self.config.trailing_silence_ms * rate / 1000
        )
        samples = ((samples + frame_samples - 1) // frame_samples) * frame_samples
        stereo = np.zeros((samples, 2), dtype=np.float32)
        last_end = {"assistant": 0, "user": 0}
        messages = []
        for utterance in utterances:
            role = utterance.role
            if utterance.start_sample < last_end[role]:
                raise ValueError(f"self-overlapping {role} utterances")
            last_end[role] = utterance.end_sample
            channel = 0 if role == "assistant" else 1
            stereo[utterance.start_sample : utterance.end_sample, channel] = (
                utterance.speech.waveform
            )
            filename = f"{role}_{utterance.turn:02d}.wav"
            sf.write(
                staging / filename, utterance.speech.waveform, rate, subtype="FLOAT"
            )
            full_filename = f"{role}_{utterance.turn:02d}_full.wav"
            sf.write(
                staging / full_filename,
                utterance.full_speech.waveform,
                rate,
                subtype="FLOAT",
            )
            start = utterance.start_sample / rate
            words = [
                {
                    "word": w.word,
                    "start": start + w.start,
                    "end": start + w.end,
                    "relative_start": w.start,
                    "relative_end": w.end,
                }
                for w in utterance.speech.words
            ]
            message = {
                "role": role,
                "turn": utterance.turn,
                "text": utterance.full_speech.source_text,
                "spoken_text": utterance.speech.spoken_text,
                "audio": filename,
                "full_audio": full_filename,
                "start": start,
                "end": utterance.end_sample / rate,
                "words": words,
                "alignment_status": "aligned" if words else "failed",
                "cut_mid_word": utterance.speech.cut_mid_word,
                "alignment_match": utterance.full_speech.alignment_match,
                "source_audio_cache": utterance.full_speech.cache_file,
            }
            (staging / filename.replace(".wav", ".words.json")).write_text(
                json.dumps(message, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            messages.append(message)
        sf.write(staging / "dialogue.wav", stereo, rate, subtype="FLOAT")
        sf.write(staging / "assistant.wav", stereo[:, 0], rate, subtype="FLOAT")
        sf.write(staging / "user.wav", stereo[:, 1], rate, subtype="FLOAT")
        split_value = (
            int(
                hashlib.sha256(
                    f"{self.config.seed}:{seed_prompt}".encode()
                ).hexdigest()[:8],
                16,
            )
            / 2**32
        )
        record = {
            "fingerprint": self.fingerprint,
            "row_index": row_index,
            "seed_prompt": seed_prompt,
            "episode_seed": episode_seed,
            "split": "validation"
            if split_value < self.config.validation_fraction
            else "train",
            "audio": str((final / "dialogue.wav").relative_to(self.output)),
            "annotation": str((final / "annotations.json").relative_to(self.output)),
            "sample_rate": rate,
            "duration": samples / rate,
            "channels": {"assistant": 0, "user": 1},
            "voices": self.speech.episode_voices() if hasattr(self.speech, "episode_voices") else None,
            "turns": len(utterances) // 2,
            "complete": complete,
            "failed_role": failed_role,
            "alignment_error": alignment_error,
            "messages": messages,
            "barge_ins": barge_ins,
            "scenes": scenes or [],
            "backchannels": backchannels or [],
        }
        (staging / "annotations.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return record

    def generate(
        self, dataset: Any, *, max_samples: int | None = None,
        retry_failed: bool = False,
    ) -> list[dict[str, Any]]:
        from tqdm.auto import tqdm

        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive or None")
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        records = []
        indexed_records = {}
        teacher_failures_path = self.output / "teacher_failures.jsonl"
        teacher_failures = {}
        if teacher_failures_path.exists():
            teacher_failures = {
                entry["row_index"]: entry
                for line in teacher_failures_path.read_text().splitlines()
                if line.strip()
                for entry in [json.loads(line)]
            }

        def save_teacher_failures():
            temporary = teacher_failures_path.with_suffix(".partial.jsonl")
            temporary.write_text(
                "".join(json.dumps(entry, ensure_ascii=False) + "\n"
                        for _, entry in sorted(teacher_failures.items())),
                encoding="utf-8",
            )
            temporary.replace(teacher_failures_path)
        failures_path = self.output / "alignment_failures.jsonl"
        failures = {}
        if failures_path.exists():
            for line in failures_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    failure = json.loads(line)
                    failures[failure["row_index"]] = failure

        def save_failures():
            if not failures:
                failures_path.unlink(missing_ok=True)
                return
            temporary = failures_path.with_suffix(".partial.jsonl")
            temporary.write_text(
                "".join(
                    json.dumps(failures[index], ensure_ascii=False) + "\n"
                    for index in sorted(failures)
                ),
                encoding="utf-8",
            )
            temporary.replace(failures_path)

        self.alignment_failures = failures
        for annotation in (self.output / "episodes").glob("*/annotations.json"):
            record = json.loads(annotation.read_text())
            if record["fingerprint"] != self.fingerprint:
                raise ValueError("stored episode belongs to a different generation run")
            indexed_records[record["row_index"]] = record
        completed = sum(index < total for index in indexed_records)
        tqdm.write(f"Resuming {self.output}: {completed} saved episodes reused.")
        with tqdm(total=total, desc="Synthetic conversations", unit="episode") as bar:
            for index in range(total):

                def progress(turn, turns, phase, episode_index=index):
                    bar.set_postfix(
                        episode=episode_index, turn=f"{turn}/{turns}", phase=phase
                    )

                seed_prompt = dataset[index]["prompt"]
                if (index in teacher_failures and index not in indexed_records
                        and not retry_failed):
                    if teacher_failures[index]["seed_prompt"] != seed_prompt:
                        raise ValueError("failed episode does not match its prompt")
                    bar.set_postfix(episode=index, phase="previous timeout skipped")
                    bar.update(1)
                    continue
                try:
                    record = self.generate_episode(
                        seed_prompt, index, progress=progress
                    )
                except TeacherTimeoutError as exc:
                    teacher_failures[index] = {
                        "row_index": index, "seed_prompt": seed_prompt,
                        "error": str(exc),
                    }
                    save_teacher_failures()
                    tqdm.write(f"Episode {index} skipped: {exc}")
                    bar.set_postfix(episode=index, phase="teacher timeout skipped")
                    bar.update(1)
                    continue
                except SpeechAlignmentError as exc:
                    failures[index] = {
                        "row_index": index,
                        "seed_prompt": seed_prompt,
                        "error": str(exc),
                        "failed_role": exc.role,
                        "saved_partial": False,
                    }
                    self.alignment_failures = failures
                    save_failures()
                    bar.set_postfix(episode=index, phase="alignment skipped")
                    bar.update(1)
                    continue
                if index in teacher_failures:
                    teacher_failures.pop(index)
                    save_teacher_failures()
                if record.get("complete", True):
                    failures.pop(index, None)
                else:
                    failures[index] = {
                        "row_index": index,
                        "seed_prompt": seed_prompt,
                        "error": record.get("alignment_error"),
                        "failed_role": record.get("failed_role"),
                        "saved_partial": True,
                    }
                self.alignment_failures = failures
                save_failures()
                records.append(record)
                indexed_records[index] = record
                bar.set_postfix(
                    turns=record["turns"],
                    barge_ins=len(record["barge_ins"]),
                    phase="partial saved" if not record.get("complete", True) else "saved",
                )
                bar.update(1)
                # Rebuild indices without duplicating resumed episodes.
                for split in ("train", "validation", "all"):
                    selected = [
                        r
                        for _, r in sorted(indexed_records.items())
                        if split == "all" or r["split"] == split
                    ]
                    destination = self.output / f"{split}.jsonl"
                    temporary = destination.with_suffix(".partial.jsonl")
                    temporary.write_text(
                        "".join(
                            json.dumps(
                                {
                                    k: r.get(k, True if k == "complete" else None)
                                    for k in (
                                        "audio",
                                        "annotation",
                                        "row_index",
                                        "split",
                                        "turns",
                                        "duration",
                                        "complete",
                                        "failed_role",
                                    )
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                            for r in selected
                        ),
                        encoding="utf-8",
                    )
                    temporary.replace(destination)
        return records
