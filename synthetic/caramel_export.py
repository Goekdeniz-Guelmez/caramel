"""Turn generator episodes into caramel manifests for `prepare.py --manifest`.

caramel never sees text: it only needs two time-aligned speaker tracks. Each episode already has `assistant.wav`
(speaker A) and `user.wav` (speaker B) on one shared timeline (barge-ins, holds and backchannels included), so a
conversation is two segments starting at 0. train.py trains every conversation from both sides, so A/B is only a label.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_caramel_manifests(output_dir: str | Path, min_seconds: float = 10.0, include_partial: bool = False) -> dict[str, int]:
    """Write `caramel_{train,validation}.jsonl` next to the generator's `{split}.jsonl`; returns rows per split."""
    out, counts = Path(output_dir), {}
    for split in ("train", "validation"):
        index, rows = out / f"{split}.jsonl", []
        if index.is_file():
            for line in index.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["duration"] < min_seconds or (not r.get("complete", True) and not include_partial):
                    continue
                episode = Path(r["audio"]).parent
                if not ((out / episode / "assistant.wav").is_file() and (out / episode / "user.wav").is_file()):
                    continue
                rows.append({"id": f"synthetic_{episode.name}", "segments": [
                    {"speaker": "A", "audio": str(episode / "assistant.wav"), "start": 0.0},
                    {"speaker": "B", "audio": str(episode / "user.wav"), "start": 0.0}]})
        (out / f"caramel_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        counts[split] = len(rows)
    return counts
