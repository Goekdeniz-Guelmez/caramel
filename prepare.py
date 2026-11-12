import argparse, glob, json, os, re, time
from collections import defaultdict
import numpy as np, mlx.core as mx
from generate import SR, NQ, load_mimi, read_wav

FRAME = SR * 2 // 25   # 1920 samples = one 80 ms frame
CHUNK = FRAME * 250    # 20 s of audio per streaming encode step: constant memory for any file length


# x [C, samples] float32 at 24 kHz -> [frames, C*8] uint16 (speaker 1 cb0-7 | speaker 2 cb0-7). Streaming encode with a rotating
# KV cache for the encoder transformer's 250-step window: identical codes to one-shot encoding, linear time in the length.
def encode_long(mimi, x):
  mimi.reset_state(); mimi.downsample.reset_state(); mimi.encoder_cache = mimi.encoder_transformer.make_rot_cache()
  n, codes = x.shape[1] // FRAME * FRAME, []
  for s in range(0, n, CHUNK):
    c = mimi.encode_step(mx.array(x[:, None, s:min(s + CHUNK, n)]))[:, :NQ]
    mx.eval(c); codes.append(np.array(c))
  c = np.concatenate(codes, -1)   # [C, 8, T]
  return c.reshape(-1, c.shape[-1]).T.astype(np.uint16)


def mono(path): return read_wav(path).mean(0)


# A manifest (JSONL, one conversation per line) places separate utterances on two time-aligned tracks, the natural output of a
# synthetic (TTS) pipeline: {"id": "c1", "segments": [{"speaker": "A", "audio": "a1.wav", "start": 0.0}, {"speaker": "B", ...}]}
# speaker is "A" or "B", start is in seconds from the start of the conversation, relative audio paths are relative to the
# manifest. Segments may overlap; a track is silent wherever its speaker has no segment.
def render(path):
  root = os.path.dirname(os.path.abspath(path))
  for line in open(path):
    if not line.strip(): continue
    c, segs = json.loads(line), []
    for s in c["segments"]:
      segs.append(("AB".index(s["speaker"]), round(s["start"] * SR), mono(os.path.join(root, s["audio"]))))
    x = np.zeros((2, max(o + len(w) for _, o, w in segs)), np.float32)
    for k, o, w in segs: x[k, o:o + len(w)] += w
    yield c["id"], x


# one clip per conversation (two time-aligned speaker tracks) or per monologue
def clips(a):
  if a.manifest: yield from (c for m in a.inputs for c in render(m)); return
  files = sorted({f for p in a.inputs for f in glob.glob(p, recursive=True)})
  if a.pair_regex:   # one file per speaker; files whose names share the regex's first group are the two sides of one conversation
    groups = defaultdict(list)
    for f in files:
      if m := re.search(a.pair_regex, os.path.basename(f)): groups[m.group(1)].append(f)
    for name, fs in sorted(groups.items()):
      if len(fs) != 2: print(f"skip {name}: {len(fs)} tracks"); continue
      x, y = mono(fs[0]), mono(fs[1])
      n = min(len(x), len(y)); yield name, np.stack([x[:n], y[:n]])
  else:
    for f in files:
      x = read_wav(f)
      if a.stereo:
        if x.shape[0] != 2: print(f"skip {f}: {x.shape[0]} channels"); continue
        yield f, x
      else: yield f, x.mean(0, keepdims=True)


def main():
  ap = argparse.ArgumentParser(description="audio files -> token store (tokens.bin, index.npy, meta.json) for train.py")
  ap.add_argument("inputs", nargs="+", help="audio files or glob patterns (quote them; ** recurses)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--stereo", action="store_true", help="two-channel files, one speaker per channel")
  ap.add_argument("--pair_regex", default="", help="one file per speaker, paired by this regex's first group, e.g. '(V\\d+_S\\d+_I\\d+)_P\\d+'")
  ap.add_argument("--manifest", action="store_true", help="inputs are JSONL manifests of utterances to place on two tracks (see render)")
  ap.add_argument("--min_seconds", type=float, default=10.0)
  a = ap.parse_args()
  W = 16 if a.stereo or a.pair_regex or a.manifest else 8
  os.makedirs(a.out, exist_ok=True)
  # an existing store is appended to, so a big dataset can be downloaded, tokenized and deleted one shard at a time
  meta, index, total = os.path.join(a.out, "meta.json"), [], 0
  if os.path.exists(meta):
    m = json.load(open(meta))
    assert m["width"] == W, f"{a.out} holds width {m['width']}, these clips are width {W}"
    index, total = np.load(os.path.join(a.out, "index.npy")).tolist(), m["frames"]
  mimi, start, t0 = load_mimi(), total, time.time()
  with open(os.path.join(a.out, "tokens.bin"), "ab") as out:
    out.truncate(total * W * 2)   # drop anything an interrupted run wrote past the last saved index
    for name, x in clips(a):
      if x.shape[1] < a.min_seconds * SR: continue
      tok = encode_long(mimi, x)
      tok.tofile(out); index.append((total, len(tok))); total += len(tok)
      print(f"{name}: {len(tok) / 12.5:.0f} s | {len(index)} clips, {total / 12.5 / 3600:.1f} h in store, "
            f"{(total - start) / 12.5 / (time.time() - t0):.0f}x real time", flush=True)
  np.save(os.path.join(a.out, "index.npy"), np.asarray(index, np.int64).reshape(-1, 2))
  json.dump({"width": W, "frames": total}, open(os.path.join(a.out, "meta.json"), "w"))


if __name__ == "__main__": main()
