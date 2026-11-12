import argparse, math, os, time
import numpy as np, soundfile as sf, mlx.core as mx
from scipy.signal import resample_poly
from mlx_audio.codec.models.mimi import Mimi
from model import load, join_parts

SR, FPS, NQ = 24000, 12.5, 8   # Mimi: 24 kHz audio, one frame of 8 codebook tokens every 80 ms
MIMI_REPO = "kyutai/moshiko-pytorch-bf16"   # tokenizer-e351c8d8-checkpoint125.safetensors, the weights behind kyutai/mimi


def load_mimi():
  mimi = Mimi.from_pretrained(MIMI_REPO)
  mimi.quantizer.rvq_rest.vq.layers = mimi.quantizer.rvq_rest.vq.layers[:NQ - 1]   # encode 8 codebooks, not 32
  return mimi


# audio [C, samples] float32 at 24 kHz -> codes [C, 8, T] int32 (each channel encoded on its own)
def encode(mimi, audio): return mimi.encode(mx.array(audio)[:, None])[:, :NQ].astype(mx.int32)


def read_wav(path):
  x, sr = sf.read(path, dtype="float32", always_2d=True)
  x = np.ascontiguousarray(x.T)
  if sr != SR: g = math.gcd(SR, sr); x = resample_poly(x, SR // g, sr // g, axis=1).astype(np.float32)
  return x


# codes [8, T] -> delayed [8, total]: codebook k shifted right by its delay, start token in the gap, -2 (= sample it) after the codes
def delay(codes, total, delays, card):
  D = np.full((8, total), -2, dtype=np.int32)
  for k, d in enumerate(delays):
    n = min(codes.shape[1], total - d)
    D[k, :d], D[k, d:d + n] = card, codes[k, :n]
  return D


# Two instances talking: speakers A and B are the two rows of one batch. Row 0 sees [A | B] (A is "me", B "the other"),
# row 1 sees [B | A], the same both-orders view used in training. Each frame, each row samples its own speaker's 8 codebooks,
# and the next frame feeds them back crosswise, so each speaker hears the other with no lag beyond the frame itself.
# The tokens never leave the GPU: frame t+1 is built from frame t's lazy output while frame t is still running (async_eval).
# prompt [2, 8, T0] opening codes (T0 may be 0) -> codes [2, 8, T0 + n_frames], plus ms per frame when timed
def selfplay(model, prompt, n_frames, temp=1.1, top_k=50, seed=0, timed=False):
  c = model.cfg
  delays, T = c.delays[:c.dep_q], prompt.shape[2] + n_frames
  S = T + 1   # one extra frame to emit the acoustic codebooks of the last frame, which run one frame behind
  assert S <= c.context, f"opening + generation must fit in {c.context} frames ({c.context / FPS:.0f} s)"
  # during the opening, forced holds the real tokens and the model just reads them; afterwards it holds -2 and the model samples
  forced = mx.array(np.stack([delay(p, S, delays, c.card) for p in prompt]))   # [2, 8, S]
  st, key = model.init_state(2, S, temp, top_k), mx.random.key(seed)
  a = b = mx.full((c.dep_q,), c.card, mx.int32)
  outs, ms = [], []
  for t in range(S):
    key, sk = mx.random.split(key)
    t0 = time.perf_counter()
    out = model.step(st, mx.stack([mx.concatenate([a, b]), mx.concatenate([b, a])]), forced[:, :, t], sk)
    if timed: mx.eval(out); ms.append((time.perf_counter() - t0) * 1e3)
    else: mx.async_eval(out)
    a, b = out[0], out[1]
    outs.append(out)
  X = mx.stack(outs, -1)   # [2, 8, S] delayed
  return mx.stack([X[:, k, d:d + T] for k, d in enumerate(delays)], 1), ms


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="./caramel-1")
  ap.add_argument("--prompt", default="", help="opening wav to continue (stereo: one speaker per channel; mono: speaker A)")
  ap.add_argument("--seconds", type=float, default=30.0, help="seconds to generate after the opening")
  # the settings used for the released samples: temperature 1.1 and top-k 50 on all 8 codebooks
  ap.add_argument("--temp", type=float, default=1.1); ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--bits", type=int, default=0, choices=[0, 4, 8], help="weight-only quantization (0 = bf16)")
  ap.add_argument("--out", default="conversation.wav")
  ap.add_argument("--realtime", action="store_true", help="report ms per frame (real time needs < 80 ms)")
  a = ap.parse_args()
  join_parts(a.model)
  model, mimi = load(a.model, bits=a.bits), load_mimi()
  prompt = np.zeros((2, 8, 0), np.int32)
  if a.prompt:
    x = read_wav(a.prompt)[:2]
    if x.shape[0] == 1: x = np.concatenate([x, np.zeros_like(x)])   # mono: speaker A talks, B is silent
    prompt = np.array(encode(mimi, x))
  codes, ms = selfplay(model, prompt, round(a.seconds * FPS), a.temp, a.top_k, a.seed, a.realtime)
  if a.realtime:
    m = np.asarray(ms[3:] or ms)
    print(f"{np.median(m):.2f} ms/frame median, {np.percentile(m, 95):.2f} p95 over {len(m)} frames, both speakers "
          f"(MLX {mx.default_device()}{f', {a.bits}-bit' if a.bits else ''}); real time is 80 ms/frame, {80 / np.median(m):.1f}x")
  audio = np.array(mimi.decode(codes)[:, 0].astype(mx.float32))   # [2, samples]
  stem = a.out[:-4] if a.out.endswith(".wav") else a.out
  os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
  sf.write(stem + ".wav", audio.T, SR)   # stereo: left = A, right = B
  for name, w in (("_a", audio[0]), ("_b", audio[1]), ("_mix", np.clip(audio.sum(0), -1, 1))): sf.write(stem + name + ".wav", w, SR)
  print(f"wrote {stem}.wav (stereo) and {stem}_a.wav, {stem}_b.wav, {stem}_mix.wav: {audio.shape[1] / SR:.1f} s")


if __name__ == "__main__": main()
