"""Generate in chunks, push each chunk's Mimi tokens to a Hugging Face dataset repo, then free the local disk.

Per chunk of `chunk_size` prompts: generate episodes -> caramel manifests -> Mimi tokens -> one parquet shard per split
(`data/{split}-{chunk:05d}.parquet`, one row per conversation: id, width, frames, tokens = uint16 [frames, width] bytes)
-> upload -> delete the episode audio (annotations.json stays, so resumed runs skip finished episodes) and the speech
cache. `meta.json` in the repo records finished chunks and frames per split; `train.py --data hf://<repo>` streams it.
`meta.json["skipped"]` maps each conversation that produced no row to the reason (teacher timeout, alignment failure,
partial episode, too short), since the local failure logs are not kept.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path


def _skip_reasons(generator, indices, uploaded: set[str], min_seconds: float) -> dict[str, str]:
    """Why each row index in `indices` has no uploaded row, from the generator's own logs and annotations."""
    out = generator.output
    teacher = {}
    path = out / "teacher_failures.jsonl"
    if path.is_file():
        teacher = {e["row_index"]: e for line in path.read_text().splitlines() if line.strip() for e in [json.loads(line)]}
    alignment = getattr(generator, "alignment_failures", {}) or {}
    reasons = {}
    for index in indices:
        if f"synthetic_{index:06d}" in uploaded:
            continue
        annotation = out / "episodes" / f"{index:06d}" / "annotations.json"
        record = json.loads(annotation.read_text()) if annotation.is_file() else None
        if index in teacher:
            reason = f"teacher timeout: {teacher[index].get('error')}"
        elif record is None:
            failure = alignment.get(index)
            reason = (f"alignment failed ({failure.get('failed_role')}) before any assistant turn: {failure.get('error')}"
                      if failure else "no episode written (unknown)")
        elif not record.get("complete", True):
            reason = (f"partial episode excluded ({record.get('turns')} turns, {record.get('duration', 0):.0f}s): "
                      f"alignment failed for {record.get('failed_role')}: {record.get('alignment_error')}")
        elif record.get("duration", 0) < min_seconds:
            reason = f"too short ({record.get('duration', 0):.1f}s < {min_seconds}s)"
        else:
            reason = "episode complete but not uploaded (missing audio?)"
        reasons[str(index)] = reason[:500]
    return reasons


def generate_to_hub(generator, dataset, repo_id: str, *, chunk_size: int = 100, max_samples: int | None = None,
                    min_seconds: float = 10.0, speech_cache: str | Path | None = None, private: bool = True,
                    workers: int = 1, chunks_at_once: int = 1, include_partial: bool = False):
    """`workers` conversations are built at once (see SyntheticConversationGenerator.generate). `chunks_at_once`
    consecutive pending chunks are generated together (so workers can exceed chunk_size without changing the chunk
    numbering of an existing repo), then tokenized and uploaded one chunk at a time. `include_partial` keeps the
    finished turns of a conversation that failed later (useful for long conversations)."""
    import pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from prepare import SR, encode_long, load_mimi, render
    from synthetic.caramel_export import write_caramel_manifests

    api, out = HfApi(), generator.output
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    try:
        meta = json.loads(Path(api.hf_hub_download(repo_id, "meta.json", repo_type="dataset")).read_text())
    except Exception:
        meta = {"width": 16, "frames": {"train": 0, "validation": 0}, "chunks": []}
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    mimi = load_mimi()
    pending = [(c, s) for c, s in enumerate(range(0, total, chunk_size)) if c not in meta["chunks"]]
    groups, i = [], 0
    while i < len(pending):   # runs of consecutive chunks, at most chunks_at_once long
        g = [pending[i]]
        while len(g) < chunks_at_once and i + len(g) < len(pending) and pending[i + len(g)][0] == g[-1][0] + 1:
            g.append(pending[i + len(g)])
        groups.append(g); i += len(g)
    for group in groups:
      g_start, g_end = group[0][1], min(group[-1][1] + chunk_size, total)
      generator.generate(dataset, start=g_start, max_samples=g_end, workers=workers)
      write_caramel_manifests(out, min_seconds=min_seconds, include_partial=include_partial)   # episodes still on disk
      for chunk, start in group:
        end = min(start + chunk_size, total)
        uploaded = set()
        for split in ("train", "validation"):
            rows = []
            for name, x in render(out / f"caramel_{split}.jsonl"):
                if not start <= int(name.rsplit("_", 1)[-1]) < end:   # this chunk's conversations only
                    continue
                if x.shape[1] >= min_seconds * SR:
                    tok = encode_long(mimi, x)
                    rows.append({"id": name, "width": tok.shape[1], "frames": len(tok), "tokens": tok.tobytes()})
            if rows:
                buf = io.BytesIO()
                pq.write_table(pa.Table.from_pylist(rows), buf)
                api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=f"data/{split}-{chunk:05d}.parquet",
                                repo_id=repo_id, repo_type="dataset", commit_message=f"{split} chunk {chunk}")
                meta["frames"][split] += sum(r["frames"] for r in rows)
                uploaded.update(r["id"] for r in rows)
        skipped = _skip_reasons(generator, range(start, min(start + chunk_size, total)), uploaded, min_seconds)
        meta.setdefault("skipped", {}).update(skipped)
        for index, reason in skipped.items():
            print(f"chunk {chunk}: conversation {index} skipped: {reason}", flush=True)
        meta["chunks"].append(chunk)
        api.upload_file(path_or_fileobj=json.dumps(meta, indent=2).encode(), path_in_repo="meta.json",
                        repo_id=repo_id, repo_type="dataset", commit_message=f"chunk {chunk} done")
        print(f"chunk {chunk} uploaded to {repo_id}: {meta['frames']['train'] / 12.5 / 3600:.2f} h train", flush=True)
      for episode in (out / "episodes").iterdir():   # whole group uploaded: keep annotations.json as the resume marker
          for f in episode.iterdir():
              if f.name != "annotations.json":
                  f.unlink() if f.is_file() else shutil.rmtree(f)
      if speech_cache:
          shutil.rmtree(speech_cache, ignore_errors=True); Path(speech_cache).mkdir(parents=True, exist_ok=True)
    return meta
