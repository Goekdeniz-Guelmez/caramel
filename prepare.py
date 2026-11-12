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


# one clip per conversation (two time-aligned speaker tracks) or per monologue
def clips(a):
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
  ap.add_argument("--min_seconds", type=float, default=10.0)
  a = ap.parse_args()
  W = 16 if a.stereo or a.pair_regex else 8
  os.makedirs(a.out, exist_ok=True)
  mimi, index, total, t0 = load_mimi(), [], 0, time.time()
  with open(os.path.join(a.out, "tokens.bin"), "wb") as out:
    for name, x in clips(a):
      if x.shape[1] < a.min_seconds * SR: continue
      tok = encode_long(mimi, x)
      tok.tofile(out); index.append((total, len(tok))); total += len(tok)
      h = total / 12.5 / 3600
      print(f"{name}: {len(tok) / 12.5:.0f} s | {len(index)} clips, {h:.1f} h, {h * 3600 / (time.time() - t0):.0f}x real time", flush=True)
  np.save(os.path.join(a.out, "index.npy"), np.asarray(index, np.int64).reshape(-1, 2))
  json.dump({"width": W, "frames": total}, open(os.path.join(a.out, "meta.json"), "w"))


if __name__ == "__main__": main()
