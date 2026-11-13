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

        # Bound retries here too: notebook kernels can retain older clients.
        client = self.client.with_options(timeout=120.0, max_retries=0)
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, seed=seed, **extra
            )
        except APITimeoutError as exc:
            raise TeacherTimeoutError(
                f"Teacher {model!r} timed out after 120 seconds"
            ) from exc
        return response.choices[0].message.content or ""

    def generate(self, messages: list[dict[str, str]], *, seed: int) -> str:
        content = self._complete(self.assistant_model, messages, seed, False).strip()
        if not content:
            raise ValueError("teacher returned an empty assistant response")
        return content

    def followup(self, seed_prompt: str, history: list[dict[str, str]], *, seed: int) -> str:
        chat = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history if m.get("role") in {"user", "assistant"}
        )
        raw = self._complete(self.followup_model,
                             [{"role": "user", "content": self.followup_prompt.format(chat_history=chat)}],
                             seed, True)
        match = re.search(r"[\[{].*[\]}]", raw, re.S)
        parsed = json.loads(match.group(0) if match else "{}")
        options = parsed if isinstance(parsed, list) else parsed.get("follow_ups")
        options = [o.strip() for o in options or [] if isinstance(o, str) and o.strip()]
        if not options:
            raise ValueError("teacher returned no follow-up prompts")
        return random.Random(seed).choice(options)


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
