"""Text teachers for synthetic conversations.

* ChatTeacher: any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, ...).
* MLXTeacher: a local MLX-VLM model on Apple Silicon (no server needed).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

DEFAULT_FOLLOWUP_PROMPT = """### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next in this conversation as a **user**, based on the chat history.

### Guidelines:
- Write all follow-up questions from the user's point of view, directed to the assistant.
- Make questions concise, clear, spoken-style, and directly related to the discussed topic(s).
- Do not repeat what was already covered.
- Response must be a JSON object with a "follow_ups" array of strings, no extra text.

### Output:
JSON format: {{ "follow_ups": ["Prompt 1?", "Prompt 2?", "Prompt 3?"] }}

### Chat History:
<chat_history>
{chat_history}
</chat_history>"""


class TeacherTimeoutError(TimeoutError):
    """The text teacher did not respond within the request timeout."""


class ChatTeacher:
    """Assistant replies and user follow-ups from an OpenAI-compatible chat API."""

    def __init__(self, client: Any, *, assistant_model: str, followup_model: str | None = None,
                 followup_prompt: str = DEFAULT_FOLLOWUP_PROMPT):
        self.client = client
        self.assistant_model = assistant_model
        self.followup_model = followup_model or assistant_model
        self.followup_prompt = followup_prompt

    def metadata(self) -> dict[str, Any]:
        return {
            "assistant_model": self.assistant_model,
            "followup_model": self.followup_model,
            "followup_prompt_sha256": hashlib.sha256(self.followup_prompt.encode()).hexdigest(),
        }

    def _complete(self, model: str, messages: list[dict[str, str]], seed: int, json_mode: bool) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        from openai import APITimeoutError

        # Bound retries here too: notebook kernels can retain older clients. Keep the client's own timeout.
        timeout = getattr(self.client, "timeout", None)
        timeout = timeout if isinstance(timeout, (int, float)) else 120.0
        client = self.client.with_options(timeout=timeout, max_retries=0)
        from openai import APIConnectionError, InternalServerError, RateLimitError
        import time

        # Cloud endpoints rate-limit (429) and have transient 5xx / connection errors: back off 30 s, 60 s, ... up to
        # 5 min; after 8 attempts this conversation is skipped (logged), the run goes on.
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, seed=seed, **extra
                )
                break
            except (RateLimitError, InternalServerError, APIConnectionError) as exc:
                if isinstance(exc, APITimeoutError):   # APITimeoutError subclasses APIConnectionError
                    raise TeacherTimeoutError(f"Teacher {model!r} timed out after {timeout:g} seconds") from exc
                if attempt == 7:
                    raise TeacherTimeoutError(f"Teacher {model!r} failed 8 times: {type(exc).__name__}: {exc}") from exc
                time.sleep(min(30 * 2 ** attempt, 300))
            except APITimeoutError as exc:
                raise TeacherTimeoutError(
                    f"Teacher {model!r} timed out after {timeout:g} seconds"
                ) from exc
        return response.choices[0].message.content or ""

    def _complete_once(self, model: str, messages: list[dict[str, str]], seed: int, json_mode: bool) -> str:
        """One request, no retries (FallbackTeacher switches teacher instead of waiting); timeouts still skip."""
        from openai import APITimeoutError
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            response = self.client.chat.completions.create(model=model, messages=messages, seed=seed, **extra)
        except APITimeoutError as exc:
            raise TeacherTimeoutError(f"Teacher {model!r} timed out") from exc
        return response.choices[0].message.content or ""

    def generate(self, messages: list[dict[str, str]], *, seed: int) -> str:
        for attempt in range(2):   # a bad reply skips this conversation (logged), it must not stop the whole run
            content = self._complete(self.assistant_model, messages, seed + attempt, False).strip()
            if content:
                return content
        raise TeacherTimeoutError(f"teacher {self.assistant_model!r} returned an empty assistant response twice")

    def followup(self, seed_prompt: str, history: list[dict[str, str]], *, seed: int) -> str:
        chat = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history if m.get("role") in {"user", "assistant"}
        )
        problem = ""
        for attempt in range(2):   # malformed JSON or no options: retry once, then skip this conversation (logged)
            raw = self._complete(self.followup_model,
                                 [{"role": "user", "content": self.followup_prompt.format(chat_history=chat)}],
                                 seed + attempt, True)
            match = re.search(r"[\[{].*[\]}]", raw, re.S)
            try:
                parsed = json.loads(match.group(0) if match else "{}")
            except json.JSONDecodeError as exc:
                problem = f"malformed follow-up JSON ({exc})"
                continue
            options = parsed if isinstance(parsed, list) else parsed.get("follow_ups") if isinstance(parsed, dict) else None
            options = [o.strip() for o in options or [] if isinstance(o, str) and o.strip()]
            if options:
                return random.Random(seed).choice(options)
            problem = "no follow-up prompts"
        raise TeacherTimeoutError(f"teacher {self.followup_model!r} gave {problem} twice")


class FallbackTeacher(ChatTeacher):
    """A primary teacher (e.g. OpenAI) that hands over to a fallback (e.g. Ollama Cloud) when the primary has no key,
    no quota, no access to the model, or keeps failing. The primary is retried every `retry_after` seconds, and its key
    is re-read from the environment / `key_file` each time, so adding a key while a run is going switches it over."""

    def __init__(self, fallback: ChatTeacher, *, primary_model: str, key_file: str | None = None,
                 base_url: str | None = None, timeout: float = 180.0, retry_after: float = 900.0):
        super().__init__(None, assistant_model=primary_model)
        self.fallback, self.primary_model, self.key_file = fallback, primary_model, key_file
        self.base_url, self.timeout, self.retry_after = base_url, timeout, retry_after
        self._primary, self._down_until, self._lock = None, 0.0, __import__("threading").Lock()
        self.followup_prompt = fallback.followup_prompt

    def metadata(self) -> dict[str, Any]:   # fixed, so switching teacher mid-run keeps the resume fingerprint
        return {"type": "fallback", "primary": self.primary_model, "fallback": self.fallback.metadata()}

    def _key(self) -> str | None:
        import os
        key = os.environ.get("OPENAI_API_KEY")
        if not key and self.key_file and os.path.isfile(os.path.expanduser(self.key_file)):
            for line in open(os.path.expanduser(self.key_file)):   # a .env file: OPENAI_API_KEY = sk-...
                name, _, value = line.partition("=")
                if name.strip() == "OPENAI_API_KEY" and value.strip():
                    key = value.strip().strip("'\"")
        return key

    def _get_primary(self) -> ChatTeacher | None:
        import time
        with self._lock:
            if time.time() < self._down_until:
                return None
            if self._primary is None:
                key = self._key()
                if not key:
                    self._mark_down("no OPENAI_API_KEY")
                    return None
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
                self._primary = ChatTeacher(client, assistant_model=self.primary_model)
                self._primary.followup_prompt = self.followup_prompt
                print(f"[teacher] using {self.primary_model}", flush=True)
            return self._primary

    def _mark_down(self, why: str) -> None:
        import time
        if time.time() >= self._down_until:
            print(f"[teacher] {self.primary_model} unavailable ({why}); using fallback for {self.retry_after / 60:.0f} min", flush=True)
        self._down_until, self._primary = time.time() + self.retry_after, None

    def _complete(self, model: str, messages: list[dict[str, str]], seed: int, json_mode: bool) -> str:
        primary = self._get_primary()
        if primary is not None:
            import openai
            try:
                return primary._complete_once(self.primary_model, messages, seed, json_mode)
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError,
                    openai.BadRequestError, openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as exc:
                with self._lock:
                    self._mark_down(f"{type(exc).__name__}: {str(exc)[:120]}")
        fb_model = self.fallback.followup_model if json_mode else self.fallback.assistant_model
        return self.fallback._complete(fb_model, messages, seed, json_mode)


class MLXTeacher(ChatTeacher):
    """Local MLX-VLM text teacher, e.g. "mlx-community/gemma-4-e4b-it-4bit".

    The follow-up model defaults to the same loaded model (one model in memory).
    """

    def __init__(self, model: str = "mlx-community/gemma-4-e4b-it-4bit", *, followup_model: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.7,
                 followup_prompt: str = DEFAULT_FOLLOWUP_PROMPT):
        super().__init__(None, assistant_model=model, followup_model=followup_model,
                         followup_prompt=followup_prompt)
        self.max_tokens, self.temperature = max_tokens, temperature
        self._models: dict[str, Any] = {}

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "backend": "mlx-vlm", "max_tokens": self.max_tokens,
                "temperature": self.temperature}

    def _complete(self, model: str, messages: list[dict[str, str]], seed: int, json_mode: bool) -> str:
        import mlx.core as mx
        from mlx_vlm import generate, load

        if model not in self._models:
            self._models[model] = load(model)
        mdl, processor = self._models[model]
        tokenizer = getattr(processor, "tokenizer", processor)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        mx.random.seed(seed % 2**32)
        result = generate(mdl, processor, prompt=prompt, max_tokens=self.max_tokens,
                          temperature=self.temperature, verbose=False)
        return getattr(result, "text", result)


def load_prompt_corpus(dataset_id="Goekdeniz-Guelmez/JOSIE-Prompts-Only-Corpus-300", *,
                       revision=None, split="train"):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    pinned = HfApi().dataset_info(dataset_id, revision=revision).sha
    dataset = load_dataset(dataset_id, split=split, revision=pinned)
    if "prompt" not in dataset.column_names:
        raise ValueError("prompt corpus must have a prompt column")
    return dataset, {"dataset_id": dataset_id, "revision": pinned, "split": split, "rows": len(dataset)}
