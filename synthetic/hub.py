"""Generate in chunks, push each chunk's Mimi tokens to a Hugging Face dataset repo, then free the local disk.

Per chunk of `chunk_size` prompts: generate episodes -> caramel manifests -> Mimi tokens -> one parquet shard per split
(`data/{split}-{chunk:05d}.parquet`, one row per conversation: id, width, frames, tokens = uint16 [frames, width] bytes)
-> upload -> delete the episode audio (annotations.json stays, so resumed runs skip finished episodes) and the speech
cache. `meta.json` in the repo records finished chunks and frames per split; `train.py --data hf://<repo>` streams it.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path


def generate_to_hub(generator, dataset, repo_id: str, *, chunk_size: int = 100, max_samples: int | None = None,
                    min_seconds: float = 10.0, speech_cache: str | Path | None = None, private: bool = True):
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
    for chunk, start in enumerate(range(0, total, chunk_size)):
        if chunk in meta["chunks"]:
            continue
        generator.generate(dataset, start=start, max_samples=min(start + chunk_size, total))
        write_caramel_manifests(out, min_seconds=min_seconds)   # only episodes whose audio is still on disk
        for split in ("train", "validation"):
            rows = []
            for name, x in render(out / f"caramel_{split}.jsonl"):
                if x.shape[1] >= min_seconds * SR:
                    tok = encode_long(mimi, x)
                    rows.append({"id": name, "width": tok.shape[1], "frames": len(tok), "tokens": tok.tobytes()})
            if rows:
                buf = io.BytesIO()
                pq.write_table(pa.Table.from_pylist(rows), buf)
                api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=f"data/{split}-{chunk:05d}.parquet",
                                repo_id=repo_id, repo_type="dataset", commit_message=f"{split} chunk {chunk}")
                meta["frames"][split] += sum(r["frames"] for r in rows)
        meta["chunks"].append(chunk)
        api.upload_file(path_or_fileobj=json.dumps(meta, indent=2).encode(), path_in_repo="meta.json",
                        repo_id=repo_id, repo_type="dataset", commit_message=f"chunk {chunk} done")
        for episode in (out / "episodes").iterdir():   # keep annotations.json as the resume marker
            for f in episode.iterdir():
                if f.name != "annotations.json":
                    f.unlink() if f.is_file() else shutil.rmtree(f)
        if speech_cache:
            shutil.rmtree(speech_cache, ignore_errors=True); Path(speech_cache).mkdir(parents=True, exist_ok=True)
        print(f"chunk {chunk} uploaded to {repo_id}: {meta['frames']['train'] / 12.5 / 3600:.2f} h train", flush=True)
    return meta
