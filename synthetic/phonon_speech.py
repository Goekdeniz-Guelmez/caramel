"""Gradium Phonon TTS (on-device, `pip install ptts`) as a synthetic speech provider.

Phonon gives no word timestamps, so words are measured with a local forced
aligner (`synthetic.aligner.AlignedSpeech`, which also handles the disk cache).

Each role's voice is one of
* a catalog or checkpoint voice name ("Zoey", "Freya", ... - `python -m ptts --list-voices`),
* a voice file (`voice.safetensors` downloaded or cloned with Gradium's scripts),
* a reference WAV (~10 s) cloned locally, if the checkpoint supports cloning.

`user_voice` may also be a list of voices (e.g. `phonon_voices(gender="female")`):
the generator then picks one at random per conversation, so the model hears many
different users while the assistant keeps one consistent voice.

Models: "hf://gradium/phonon-40m-multilingual" (en/fr/de/es/pt, CC BY-NC 4.0, gated:
accept the terms on Hugging Face and log in) or "hf://gradium/phonon-31m-en" (Apache 2.0).
"""

from __future__ import annotations

import hashlib
import json
import threading
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synthetic.aligner import DEFAULT_ALIGNER_MODEL, AlignedSpeech

DEFAULT_PHONON_MODEL = "hf://gradium/phonon-40m-multilingual"

# Phonon catalog voices: name -> (voice id, accent, gender)
PHONON_VOICES: dict[str, tuple[str, str, str]] = {
    "Zoey": ("NbpkqMVS3CJeq2j8", "us", "female"),
    "Juniper": ("NIWIOv5vTi5eRMj8", "us", "female"),
    "Everly": ("vjFyp2BcCyTlUmYb", "us", "female"),
    "Harper": ("4SZHfMpw-p46Ywgs", "us", "female"),
    "Hadley": ("bXwrkHhURL1VFJdG", "us", "female"),
    "Delia": ("nGzMHMPAWGaJFHOo", "us", "female"),
    "Easton": ("LMDIYE0Oa377Qh6Z", "us", "male"),
    "Micah": ("40QrMVgAxFiUnP9f", "us", "male"),
    "Trevor": ("0zLApsZfVLLNaEGl", "us", "male"),
    "Russell": ("_6Aslh2DxfmnRLmP", "us", "male"),
    "Preston": ("qIxlZ7Ei8KvWlyQH", "us", "male"),
    "Miles": ("d9XCyhjo4JhkHF1d", "us", "male"),
    "Maeve": ("6PWnV0Nq4wu7RVBT", "uk", "female"),
    "Freya": ("GgfEkEJtxZR7gnpy", "uk", "female"),
    "Pippa": ("uem82D50GRv2Dwma", "uk", "female"),
    "Elodie-Rose": ("y9bQqIDnyxwqK01t", "uk", "female"),
    "Imogen": ("gDT1nz7Ie36ZhL-C", "uk", "female"),
    "Lottie": ("m9MxpY6IUGXtWp76", "uk", "female"),
    "Bryony": ("gfP44RlHwr2kEwk7", "uk", "female"),
    "Alfie": ("X6NceqF285Gr04gb", "uk", "male"),
    "Hugh": ("YyKcxPG7iIgS3jsK", "uk", "male"),
    "Edwin": ("zUCxzrOrugTSis03", "uk", "male"),
    "Spencer": ("JOW0vchyswkQUMDS", "uk", "male"),
    "Toby": ("dME3IWyZBvmh1n1q", "uk", "male"),
    "Teddy": ("Zn7g6NxLqXlJb7Qi", "uk", "male"),
}


def phonon_voices(*names: str, accent: str | None = None, gender: str | None = None,
                  exclude: Sequence[str] = ()) -> list[PhononVoice]:
    """Catalog voices by name and/or filter: phonon_voices(accent="uk", gender="male")."""
    unknown = [n for n in names if n not in PHONON_VOICES]
    if unknown:
        raise ValueError(f"unknown Phonon voices: {unknown}")
    picked = [n for n, (_, a, g) in PHONON_VOICES.items()
              if (not names or n in names) and accent in (None, a) and gender in (None, g)
              and n not in exclude]
    if not picked:
        raise ValueError("no Phonon voice matches the selection")
    return [PhononVoice(name=n) for n in picked]


@dataclass(frozen=True)
class PhononVoice:
    name: str | None = None  # catalog / checkpoint voice
    file: str | Path | None = None  # voice.safetensors
    wav: str | Path | None = None  # reference recording to clone

    def __post_init__(self):
        if sum(v is not None for v in (self.name, self.file, self.wav)) != 1:
            raise ValueError("give exactly one of name, file or wav")
        for path in (self.file, self.wav):
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(path)

    def identity(self) -> dict[str, str]:
        if self.name is not None:
            return {"name": self.name}
        path = Path(self.file or self.wav)
        kind = "file" if self.file is not None else "wav"
        return {kind: hashlib.sha256(path.read_bytes()).hexdigest()}


class PhononSpeech(AlignedSpeech):
    """Phonon synthesis + forced word alignment, cached on disk."""

    def __init__(self, cache_dir: str | Path, *, user_voice: PhononVoice | Sequence[PhononVoice],
                 assistant_voice: PhononVoice | Sequence[PhononVoice],
                 model: str = DEFAULT_PHONON_MODEL, language: str = "en", temperature: float | None = None,
                 alignment_model: str = DEFAULT_ALIGNER_MODEL, alignment_language: str = "English",
                 min_alignment_match: float = 0.9, loaded_aligner_model: Any = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.language = language
        self.temperature = temperature
        self.alignment_model = alignment_model
        self.alignment_language = alignment_language
        self.min_alignment_match = min_alignment_match
        self.sample_rate = 24000
        self.max_tokens = 0
        pool = [user_voice] if isinstance(user_voice, PhononVoice) else list(user_voice)
        if not pool or not all(isinstance(v, PhononVoice) for v in pool):
            raise ValueError("user_voice must be a PhononVoice or a nonempty list of them")
        self.user_pool = pool
        apool = [assistant_voice] if isinstance(assistant_voice, PhononVoice) else list(assistant_voice)
        if not apool or not all(isinstance(v, PhononVoice) for v in apool):
            raise ValueError("assistant_voice must be a PhononVoice or a nonempty list of them")
        if len(apool) > 1 and len({json.dumps(v.identity(), sort_keys=True) for v in apool + pool}) < 2:
            raise ValueError("pretraining needs at least two distinct voices")
        self.assistant_pool = apool   # one voice: fixed assistant (JOSIE); several: random per conversation (pretraining)
        self.assistant_voice = apool[0]
        self._episode = threading.local()   # each worker thread has its own conversation's user voice
        self._aligner_model = loaded_aligner_model
        self._tts = None
        self._voice_names: dict[str, str | None] = {}  # identity json -> Phonon voice name

    def metadata(self) -> dict[str, Any]:
        return {"provider": "gradium-phonon-and-qwen3-forced-aligner", "tts_model": self.model,
                "language": self.language, "temperature": self.temperature,
                "alignment_model": self.alignment_model, "alignment_language": self.alignment_language,
                "min_alignment_match": self.min_alignment_match, "sample_rate": self.sample_rate,
                "voices": {"user": [v.identity() for v in self.user_pool],
                           "assistant": ([v.identity() for v in self.assistant_pool] if len(self.assistant_pool) > 1
                                         else self.reference_ids["assistant"])}}

    @property
    def references(self) -> dict[str, PhononVoice]:
        return {"user": getattr(self._episode, "user", self.user_pool[0]),
                "assistant": getattr(self._episode, "assistant", self.assistant_voice)}

    @property
    def reference_ids(self) -> dict[str, Any]:
        return {role: v.identity() for role, v in self.references.items()}

    def begin_episode(self, rng: random.Random) -> None:
        """Pick this conversation's user voice from the pool (per thread, so parallel episodes don't mix voices)."""
        self._episode.user = self.user_pool[rng.randrange(len(self.user_pool))]
        if len(self.assistant_pool) > 1:   # pretraining: speaker A gets a random voice too, never the same as B
            key = lambda v: json.dumps(v.identity(), sort_keys=True)
            choices = [v for v in self.assistant_pool if key(v) != key(self._episode.user)]
            self._episode.assistant = choices[rng.randrange(len(choices))]

    def episode_voices(self) -> dict[str, Any]:
        return dict(self.reference_ids)

    def cache_identity(self, text: str, role: str) -> dict[str, Any]:
        if role not in self.references:
            raise ValueError("speech role must be user or assistant")
        return {"provider": "gradium-phonon", "text": text, "role": role, "tts_model": self.model,
                "voice": self.reference_ids[role], "language": self.language,
                "temperature": self.temperature, "alignment_model": self.alignment_model,
                "alignment_language": self.alignment_language, "format": "wav"}

    def _load(self):
        if self._tts is None:
            import ptts

            self._tts = ptts.TTS(self.model, lang=self.language)
            voices = [("user", v) for v in self.user_pool] + [("assistant", v) for v in self.assistant_pool]
            for i, (role, voice) in enumerate(voices):
                key = json.dumps(voice.identity(), sort_keys=True)
                if key not in self._voice_names:
                    self._voice_names[key] = self._register(f"{role}-{i}", voice)
        return self._tts

    def _register(self, role: str, voice: PhononVoice) -> str | None:
        tts = self._tts
        if voice.name is not None:
            return voice.name
        name = f"josie-{role}"
        if voice.file is not None:
            tts.add_voice(name, str(voice.file))
            return name
        if not tts.supports_voice_cloning:
            raise ValueError(f"{self.model} cannot clone voices; use a voice name or voice file")
        from synthetic.audio import prepare_mono_audio
        import soundfile as sf

        pcm, rate = sf.read(voice.wav, dtype="float32", always_2d=True)
        pcm = prepare_mono_audio(pcm, rate, target_sample_rate=tts.voice_prompt_sample_rate)
        tts.clone_voice(name, np.ascontiguousarray(pcm[: 15 * tts.voice_prompt_sample_rate]))
        return name

    def synthesize_to_file(self, text: str, role: str, path: Path) -> None:
        import soundfile as sf

        tts = self._load()
        identity = self.cache_identity(text, role)
        seed = int(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8], 16)
        voice = self._voice_names[json.dumps(self.reference_ids[role], sort_keys=True)]
        kwargs = {"voice": voice, "seed": seed}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        waveform = np.asarray(tts.synth(text, **kwargs), dtype=np.float32).reshape(-1)
        if not waveform.size or not np.isfinite(waveform).all():
            raise ValueError("Phonon produced no or nonfinite audio")
        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak
        sf.write(path, waveform, int(tts.sample_rate), subtype="FLOAT")
