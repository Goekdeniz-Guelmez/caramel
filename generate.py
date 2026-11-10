import argparse, os, time
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from transformers import MimiModel
from model import load

SR, FPS = 24000, 12.5   # Mimi: 24 kHz audio, one frame of 8 codebook tokens every 80 ms


# Two instances talking: speakers A and B are the two rows of one batch. Row 0 sees [A | B] (A is "me", B "the other"),
# row 1 sees [B | A], the same both-orders view used in training. Each frame, each row samples its own speaker's 8 codebooks,
# and the next frame feeds them back crosswise, so each speaker hears the other with no lag beyond the frame itself.
class SelfPlay:
  def __init__(self, model, Tmax, temp, top_k):
    self.m, self.temp, self.top_k, c, dev = model, temp, top_k, model.cfg, model.emb.weight.device
    self.st = model.init_state(2, Tmax)
    self.inp = torch.full((2, c.n_q), c.card, dtype=torch.long, device=dev)
    self.forced = torch.full((2, c.dep_q), -2, dtype=torch.long, device=dev)
    self.graph = None
    if dev.type == "cuda":
      # CUDA graph: one frame is hundreds of tiny kernels, so launching them costs more than running them. Record the frame
      # once and replay it with a single launch. Warm up on a side stream first so lazy initialisation stays out of the recording.
      s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(s):
        for _ in range(3): model.step(self.st, self.inp, self.forced, temp, top_k)
      torch.cuda.current_stream().wait_stream(s)
      self.graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self.graph): self.out = model.step(self.st, self.inp, self.forced, temp, top_k)
    self.st["pos"].zero_()   # warm-up wrote caches at positions 0..3; they are overwritten before they are ever read

  def step(self, inp, forced):
    self.inp.copy_(inp); self.forced.copy_(forced)
    if self.graph is None: return self.m.step(self.st, self.inp, self.forced, self.temp, self.top_k)
    self.graph.replay(); return self.out.clone()


# codes [8, T] -> delayed [8, total]: codebook k shifted right by its delay, start token in the gap, -2 (= sample it) after the codes
def delay(codes, total, delays, card):
  D = torch.full((8, total), -2, dtype=torch.long, device=codes.device)
  for k, d in enumerate(delays):
    n = min(codes.shape[1], total - d)
    D[k, :d], D[k, d:d + n] = card, codes[k, :n]
  return D


def sync(dev):
  if dev.type == "cuda": torch.cuda.synchronize()
  elif dev.type == "mps": torch.mps.synchronize()


# prompt [2, 8, T0] opening codes (T0 may be 0) -> [2, 8, T0 + n_frames], plus ms per frame when timed
@torch.no_grad()
def selfplay(model, prompt, n_frames, temp=1.1, top_k=50, timed=False):
  c, dev = model.cfg, model.emb.weight.device
  delays, T = c.delays[:c.dep_q], prompt.shape[2] + n_frames
  S = T + 1   # one extra frame to emit the acoustic codebooks of the last frame, which run one frame behind
  assert S <= c.context, f"opening + generation must fit in {c.context} frames ({c.context / FPS:.0f} s)"
  g, A, B = SelfPlay(model, S, temp, top_k), delay(prompt[0], S, delays, c.card), delay(prompt[1], S, delays, c.card)
  start, ms = torch.full((8,), c.card, dtype=torch.long, device=dev), []
  for t in range(S):
    a, b = (A[:, t - 1], B[:, t - 1]) if t else (start, start)
    if timed: sync(dev)
    t0 = time.perf_counter()
    # during the opening, forced holds the real tokens and the model just reads them; afterwards it holds -2 and the model samples
    out = g.step(torch.stack([torch.cat([a, b]), torch.cat([b, a])]), torch.stack([A[:, t], B[:, t]]))
    A[:, t], B[:, t] = out[0], out[1]
    if timed: sync(dev)
    ms.append((time.perf_counter() - t0) * 1e3)
  return torch.stack([torch.stack([X[k, d:d + T] for k, d in enumerate(delays)]) for X in (A, B)]), ms


# wav -> Mimi codes [2, 8, T]: stereo is one speaker per channel, mono is speaker A with B silent
def encode(mimi, path, dev):
  x, sr = sf.read(path, dtype="float32", always_2d=True)
  x = torch.from_numpy(np.ascontiguousarray(x.T))
  if sr != SR: x = F.interpolate(x[None, None], size=(x.shape[0], round(x.shape[1] * SR / sr)), mode="bicubic", antialias=True)[0, 0]
  if x.shape[0] == 1: x = torch.cat([x, torch.zeros_like(x)])
  return mimi.encode(x[:2, None].to(dev), num_quantizers=8).audio_codes


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="./caramel-1")
  ap.add_argument("--prompt", default="", help="opening wav to continue (stereo: one speaker per channel; mono: speaker A)")
  ap.add_argument("--seconds", type=float, default=30.0, help="seconds to generate after the opening")
  # the settings used for the released samples: temperature 1.1 and top-k 50 on all 8 codebooks
  ap.add_argument("--temp", type=float, default=1.1); ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default="conversation.wav")
  ap.add_argument("--realtime", action="store_true", help="report ms per frame (real time needs < 80 ms)")
  a = ap.parse_args()
  torch.manual_seed(a.seed)
  # NVIDIA GPU, else the Apple GPU (Metal) on a Mac, else CPU; an M3 Pro MacBook runs two copies about 4.8x faster than real time
  dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  model, mimi = load(a.model, dev), MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
  with torch.no_grad(): prompt = encode(mimi, a.prompt, dev) if a.prompt else torch.zeros(2, 8, 0, dtype=torch.long, device=dev)
  codes, ms = selfplay(model, prompt, round(a.seconds * FPS), a.temp, a.top_k, a.realtime)
  if a.realtime:
    m = np.asarray(ms[3:] or ms)
    print(f"{np.median(m):.2f} ms/frame median, {np.percentile(m, 95):.2f} p95 over {len(m)} frames, both speakers "
          f"({'CUDA graph' if dev == 'cuda' else 'eager ' + dev}); real time is 80 ms/frame, {80 / np.median(m):.1f}x")
  with torch.no_grad(): audio = mimi.decode(codes).audio_values[:, 0].float().cpu().numpy()   # [2, samples]
  stem = a.out[:-4] if a.out.endswith(".wav") else a.out
  os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
  sf.write(stem + ".wav", audio.T, SR)   # stereo: left = A, right = B
  for name, w in (("_a", audio[0]), ("_b", audio[1]), ("_mix", np.clip(audio.sum(0), -1, 1))): sf.write(stem + name + ".wav", w, SR)
  print(f"wrote {stem}.wav (stereo) and {stem}_a.wav, {stem}_b.wav, {stem}_mix.wav: {audio.shape[1] / SR:.1f} s")


if __name__ == "__main__": main()
