"""Talk to caramel live: your microphone is the "other" speaker, the model answers through the speakers.

Every 80 ms one 1920-sample block of mic audio is encoded with Mimi, fed to the model as the other speaker's stream,
and the model samples its own next frame, which is decoded and played. Use headphones: the model must not hear itself.

  python live.py --model runs/poc/best                       # live, mic + speakers
  python live.py --model runs/poc/best --mic_file me.wav     # offline test: the wav plays the role of the mic
"""
import argparse, queue, time
import numpy as np, soundfile as sf, mlx.core as mx
from generate import SR, FPS, load_mimi, read_wav
from model import load, join_parts

BLOCK = int(SR / FPS)   # 1920 samples = one frame


class Talker:
  def __init__(self, model, mimi, temp, top_k, seed):
    self.m, self.mimi, c = model, mimi, model.cfg
    self.delays, self.card, self.K = c.delays[:c.dep_q], c.card, c.dep_q
    self.max_frames = c.context - 1
    self.st, self.key = model.init_state(1, c.context, temp, top_k), mx.random.key(seed)
    self.forced = mx.full((1, self.K), -2, mx.int32)
    self.me = mx.full((self.K,), self.card, mx.int32)   # model's delayed tokens of the previous frame
    self.user_hist, self.prev_out, self.t = [], None, 0
    mimi.reset_state()

  # the user's delayed tokens for frame t-1: codebook k with delay d holds the user's frame t-1-d
  def user_delayed(self):
    t = self.t
    return mx.array([self.user_hist[t - 1 - d][k] if t - 1 - d >= 0 else self.card for k, d in enumerate(self.delays)], mx.int32)

  # one mic block [BLOCK] float32 -> one block of model audio [BLOCK] (the model's frame t-1: its acoustic codebooks lag one frame)
  def step(self, block):
    codes = self.mimi.encode_step(mx.array(block, mx.float32)[None, None])[0, :self.K, 0]   # [8]
    inp = mx.concatenate([self.me, self.user_delayed()])[None]   # row sees [me | other]
    self.key, sk = mx.random.split(self.key)
    out = self.m.step(self.st, inp, self.forced, sk)[0]   # [8] delayed: cb0 of frame t, cb1-7 of frame t-1
    mx.eval(codes, out)
    self.user_hist.append(np.array(codes).astype(int).tolist())
    pcm = np.zeros(BLOCK, np.float32)
    if self.prev_out is not None:
      frame = mx.concatenate([self.prev_out[:1], out[1:]])[None, :, None]   # [1, 8, 1] undelayed frame t-1
      pcm = np.array(self.mimi.decode_step(frame)[0, 0].astype(mx.float32))
    self.prev_out, self.me, self.t = out, out, self.t + 1
    return pcm

  @property
  def full(self): return self.t >= self.max_frames


def run_file(talker, mic, out):
  mic = mic[: (len(mic) // BLOCK) * BLOCK].reshape(-1, BLOCK)[: talker.max_frames]
  ms, said = [], []
  for b in mic:
    t0 = time.perf_counter(); said.append(talker.step(b)); ms.append((time.perf_counter() - t0) * 1e3)
  return np.concatenate(said), mic.reshape(-1), ms


def run_live(talker, in_dev, out_dev):
  import sounddevice as sd
  q_in, q_out = queue.Queue(), queue.Queue()
  def cb(indata, outdata, frames, tinfo, status):
    q_in.put(indata[:, 0].copy())
    try: outdata[:, 0] = q_out.get_nowait()
    except queue.Empty: outdata.fill(0)
  mic_rec, said, ms, late = [], [], [], 0
  print(f"live: talk now (headphones!). Ctrl+C to stop. Context limit {talker.max_frames / FPS:.0f} s.")
  with sd.Stream(samplerate=SR, blocksize=BLOCK, channels=1, dtype="float32", callback=cb, device=(in_dev, out_dev)):
    try:
      while not talker.full:
        b = q_in.get()
        t0 = time.perf_counter(); pcm = talker.step(b); ms.append((time.perf_counter() - t0) * 1e3)
        if ms[-1] > 80: late += 1
        q_out.put(pcm); mic_rec.append(b); said.append(pcm)
        while q_in.qsize() > 2: mic_rec.append(q_in.get())   # fell behind: drop stale mic blocks rather than drift
    except KeyboardInterrupt: pass
  print(f"{late} of {len(ms)} frames over 80 ms")
  return np.concatenate(said), np.concatenate(mic_rec), ms


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="./caramel-1")
  ap.add_argument("--temp", type=float, default=1.1); ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--bits", type=int, default=0, choices=[0, 4, 8], help="weight-only quantization (0 = bf16)")
  ap.add_argument("--mic_file", default="", help="use this wav as the mic instead of live audio (offline test)")
  ap.add_argument("--in_dev", default=None, help="sounddevice input device (name or index); see python -m sounddevice")
  ap.add_argument("--out_dev", default=None, help="sounddevice output device")
  ap.add_argument("--out", default="live.wav", help="recording: stereo, left = model, right = you")
  a = ap.parse_args()
  join_parts(a.model)
  model, mimi = load(a.model, bits=a.bits), load_mimi()
  mimi.warmup()
  talker = Talker(model, mimi, a.temp, a.top_k, a.seed)
  dev = lambda d: int(d) if d is not None and d.isdigit() else d
  said, heard, ms = run_file(talker, read_wav(a.mic_file).mean(0), a.out) if a.mic_file else run_live(talker, dev(a.in_dev), dev(a.out_dev))
  m = np.asarray(ms[3:] or ms)
  print(f"{np.median(m):.1f} ms/frame median, {np.percentile(m, 95):.1f} p95 (real time needs < 80)")
  n = min(len(said), len(heard))
  sf.write(a.out, np.stack([said[:n], heard[:n]], 1), SR)
  print(f"wrote {a.out}: {n / SR:.1f} s, left = model, right = you")


if __name__ == "__main__": main()
