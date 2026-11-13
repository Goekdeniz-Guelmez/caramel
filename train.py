import argparse, json, math, os, queue, threading, time
from functools import partial
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from model import Caramel, Config, join_parts, save

CARD, DELAYS = Config.card, Config().delays
BUCKETS = [750, 1000, 1250, 1500]   # window lengths in frames: 60, 80, 100, 120 s


# ── data ──
# A store is one folder: tokens.bin uint16 [frames, W] frame-major (W = 16 stereo: speaker 1 cb0-7 | speaker 2 cb0-7,
# or W = 8 mono), index.npy int64 [clips, 2] (offset, length) in frames, and meta.json {"width": W, "frames": total}.
class TokenStore:
  def __init__(self, d, hours=0.0):
    meta = json.load(open(os.path.join(d, "meta.json")))
    self.W, self.index = meta["width"], np.load(os.path.join(d, "index.npy"))
    if hours:   # keep only the first clips, up to `hours` of audio (at least one clip): small proof-of-concept runs
      self.index = self.index[:max(1, int(np.searchsorted(np.cumsum(self.index[:, 1]), hours * 3600 * 12.5, side="right")))]
    self.tok = np.memmap(os.path.join(d, "tokens.bin"), dtype=np.uint16, mode="r", shape=(meta["frames"], self.W))
    self.cum = np.cumsum(self.index[:, 1])   # clips are drawn proportionally to their length

  def clip(self, rng): return int(np.searchsorted(self.cum, rng.random() * self.cum[-1], side="right"))
  def window(self, i, start, L): return self.tok[self.index[i, 0] + start:self.index[i, 0] + start + L]


# raw [L, W] -> seq [L+1, 16] int16, ready for Caramel.__call__: start token first, then each codebook shifted by its delay
# with the start token as filler. A mono clip gets -1 (no input, no loss) for the second speaker. swap puts speaker 2 first,
# so the model learns both sides of every conversation.
def delay_window(raw, L, swap):
  x = raw.astype(np.int16)
  mono = x.shape[1] == 8
  if mono: x = np.concatenate([x, np.full_like(x, -1)], 1)
  elif swap: x = np.concatenate([x[:, 8:], x[:, :8]], 1)
  if x.shape[0] < L: x = np.concatenate([x, np.full((L - x.shape[0], 16), -1, np.int16)], 0)   # short clip: masked padding
  seq = np.empty((L + 1, 16), np.int16)
  seq[0] = CARD
  for k, d in enumerate(DELAYS): seq[1:1 + d, k], seq[1 + d:, k] = CARD, x[:L - d, k]
  if mono: seq[:, 8:] = -1
  return seq


# Each batch draws one window length L from BUCKETS and holds frames_per_batch // L windows, so the compiled graphs only ever
# see 4 shapes. Stores are picked by weight, clips by length, and the window start uniformly. One background thread keeps
# a few batches ready; a single producer keeps the stream deterministic, so a resumed run sees the same data.
def batch_stream(dirs, fpb, weights, seed, rank, hours=0.0):
  rng, stores = np.random.default_rng([seed, rank]), [TokenStore(d, hours) for d in dirs]
  sw = np.asarray(weights or [s.cum[-1] for s in stores], dtype=np.float64)
  sw /= sw.sum()
  q = queue.Queue(maxsize=8)

  def produce():
    while True:
      L, seqs = int(rng.choice(BUCKETS)), []
      for _ in range(max(1, fpb // L)):
        s = stores[rng.choice(len(stores), p=sw)]
        i = s.clip(rng)
        start = int(rng.integers(0, max(1, s.index[i, 1] - L + 1)))
        seqs.append(delay_window(np.asarray(s.window(i, start, L)), L, bool(rng.integers(2))))
      q.put(np.stack(seqs))

  threading.Thread(target=produce, daemon=True).start()
  while True: yield q.get()


# fixed held-out windows (fixed seed), each scored in both speaker orders; same shape as a training bucket so graphs are reused
def eval_batches(d, n=64, L=1500, bs=16, seed=12345):
  s, rng, seqs = TokenStore(d), np.random.default_rng(seed), []
  for _ in range(n):
    i = s.clip(rng)
    raw = np.asarray(s.window(i, int(rng.integers(0, max(1, s.index[i, 1] - L + 1))), L))
    seqs += [delay_window(raw, L, False), delay_window(raw, L, True)]
  return [np.stack(seqs[k:k + bs]) for k in range(0, len(seqs), bs)]


# ── schedule ──
# linear warmup to the peak, then cosine down to 0.1x peak at --steps. --sched_start restarts the schedule at a resumed step.
def lr_at(step, a):
  step -= a.sched_start
  if step < a.warmup: return a.lr * (step + 1) / a.warmup
  p = (step - a.warmup) / max(1, a.steps - a.sched_start - a.warmup)
  return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


# ── checkpoints: a folder with the float32 weights, the optimizer state and the step ──
def save_ckpt(path, model, opt, step):
  os.makedirs(path, exist_ok=True)
  mx.save_safetensors(os.path.join(path, "weights.safetensors"), dict(tree_flatten(model.parameters())))
  mx.save_safetensors(os.path.join(path, "opt.safetensors"), dict(tree_flatten(opt.state)))
  json.dump({"step": step, "cfg": model.cfg.__dict__}, open(os.path.join(path, "meta.json"), "w"))


def load_ckpt(path, model, opt):
  model.load_weights(os.path.join(path, "weights.safetensors"), strict=True)
  opt.state = tree_unflatten(list(mx.load(os.path.join(path, "opt.safetensors")).items()))
  return json.load(open(os.path.join(path, "meta.json")))["step"]


# ── train ──
def main(a):
  group = mx.distributed.init()   # multi-Mac data parallel when started with mlx.launch, a no-op group of 1 otherwise
  rank, world = group.rank(), group.size()
  if a.epochs:   # passes over the training data -> steps (with --weights the stores are seen in proportion, not exactly N times)
    frames = 0
    for d in a.data:
      f = json.load(open(os.path.join(d, "meta.json")))["frames"]
      frames += min(f, int(a.train_hours * 3600 * 12.5)) if a.train_hours else f
    a.steps = math.ceil(a.epochs * frames / (a.fpb * a.accum * world))
    if rank == 0: print(f"--epochs {a.epochs:g}: {frames} frames / {a.fpb * a.accum * world} per step -> {a.steps} steps", flush=True)
  elif not a.steps: raise SystemExit("give --steps or --epochs")
  mx.random.seed(a.seed)
  os.makedirs(a.out, exist_ok=True)
  cfg = Config.from_dict(json.load(open(os.path.join(a.init, "config.json")))) if a.init else Config(**json.loads(a.cfg))
  model = Caramel(cfg)   # float32 master weights, bf16 matmuls
  if a.init:   # fine-tune a model folder (e.g. the caramel-1 release): its weights, a fresh optimizer
    join_parts(a.init)
    model.load_weights(os.path.join(a.init, "model.safetensors"), strict=True); model.set_dtype(mx.float32)
  # AdamW (bias-corrected, like PyTorch); weight decay only on matrices, not on norms
  adam = partial(optim.AdamW, learning_rate=a.lr, betas=[0.9, 0.95], eps=1e-8, bias_correction=True)
  opt = optim.MultiOptimizer([adam(weight_decay=a.wd), adam(weight_decay=0.0)], [lambda path, p: p.ndim >= 2])
  opt.init(model.trainable_parameters())
  start = load_ckpt(a.resume, model, opt) if a.resume else 0   # the next stage keeps the weights and the optimizer state
  mx.eval(model.parameters(), opt.state)
  if rank == 0: print(f"params {sum(v.size for _, v in tree_flatten(model.parameters())) / 1e6:.1f}M  world {world}  cfg {json.dumps(model.cfg.__dict__)}", flush=True)

  # compiled graphs: one per window length (4 shapes). Forward + backward of a micro-batch, then the averaged, clipped update.
  state = [model.state, opt.state]
  loss_grad = nn.value_and_grad(model, model.__call__)

  @partial(mx.compile, inputs=model.state, outputs=model.state)   # value_and_grad writes the params back into the model
  def micro(seq): return loss_grad(seq)

  @partial(mx.compile, inputs=state, outputs=state)
  def update(grads):
    if world > 1: grads = nn.average_gradients(grads)
    grads, gn = optim.clip_grad_norm(grads, a.clip)
    opt.update(model, grads)
    return gn

  evaluate = mx.compile(model.__call__, inputs=model.state)

  data = batch_stream(a.data, a.fpb, a.weights, a.seed + 7919 * start, rank, a.train_hours)
  end = a.stop_at or a.steps
  eval_at = (set(range(0, a.steps + 1, a.eval_every)) if a.eval_every else set()) | ({a.stop_at} if a.stop_at else set())
  dev_batches = eval_batches(a.dev, bs=max(1, a.fpb // 1500)) if a.dev else []
  log, evals, best = open(os.path.join(a.out, f"log_rank{rank}.jsonl"), "a"), [], []

  # every rank holds the same weights, so every rank computes the same dev loss and makes the same stop decision
  def do_eval(s):
    model.eval()
    res = [evaluate(mx.array(b, mx.int32)) for b in dev_batches]
    mx.eval(res)
    model.train()
    e = {"step": s, "dev_loss": float(np.mean([l.item() for l, _ in res])), "dev_cb": np.mean([np.array(c) for _, c in res], 0).tolist()}
    evals.append(e); log.write(json.dumps({"eval": e}) + "\n"); log.flush()
    if rank == 0: print(f"EVAL step {s} dev_loss {e['dev_loss']:.4f}", flush=True)
    if a.save_best and (not best or e["dev_loss"] < best[0]):
      best[:] = [e["dev_loss"], s]
      if rank == 0: save(model, os.path.join(a.out, "best"))   # bf16 release folder: config.json + model.safetensors
    d = [x["dev_loss"] for x in evals[-(a.patience + 1):]]
    stop = a.patience and len(d) == a.patience + 1 and all(d[i + 1] > d[i] for i in range(a.patience))
    if stop and rank == 0: print(f"early stop at step {s}", flush=True)
    return stop

  t_last = time.perf_counter()
  for step in range(start, end):
    if step in eval_at and dev_batches:
      if do_eval(step): end = step; break
      t_last = time.perf_counter()
    opt.learning_rate = lr_at(step, a)
    grads, loss, per_cb, frames = None, 0.0, 0.0, 0
    for _ in range(a.accum):
      seq = next(data)
      (l, cb), g = micro(mx.array(seq, mx.int32))
      grads = g if grads is None else tree_map(mx.add, grads, g)
      loss, per_cb, frames = loss + l, per_cb + cb, frames + seq.shape[0] * (seq.shape[1] - 1)
      mx.eval(grads, loss, per_cb)   # bound memory to one micro-batch of activations
    if a.accum > 1: grads = tree_map(lambda g: g / a.accum, grads)
    gn = update(grads)
    mx.eval(model.parameters(), opt.state, gn)
    t1 = time.perf_counter()
    dt, t_last = t1 - t_last, t1
    lv, cbs = loss.item() / a.accum, [round(x / a.accum, 4) for x in per_cb.tolist()]
    rec = {"step": step + 1, "loss": lv, "cb": cbs, "gn": gn.item(), "lr": lr_at(step, a), "dt": dt, "frames_s": frames * world / dt}
    log.write(json.dumps(rec) + "\n")
    if rank == 0 and (step % 10 == 0 or step + 1 == end):
      log.flush()
      print(f"step {step + 1} loss {lv:.4f} cb0 {cbs[0]:.3f} lr {rec['lr']:.2e} dt {dt * 1e3:.0f}ms frames/s {rec['frames_s']:.0f}", flush=True)
    if step + 1 == end and rank == 0: save_ckpt(os.path.join(a.out, "ckpt"), model, opt, step + 1)
  if end in eval_at and dev_batches and not (evals and evals[-1]["step"] == end): do_eval(end)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", nargs="+", required=True, help="token store folders")
  ap.add_argument("--weights", type=float, nargs="*", default=None, help="sampling weight per store (default: by frames)")
  ap.add_argument("--train_hours", type=float, default=0, help="use only the first N hours of each --data store (0 = all); --dev is never cut")
  ap.add_argument("--out", required=True)
  ap.add_argument("--steps", type=int, default=0, help="end of the LR schedule")
  ap.add_argument("--epochs", type=float, default=0, help="set --steps from passes over --data (overrides --steps)")
  ap.add_argument("--stop_at", type=int, default=0, help="stop (and save ckpt/) here; the schedule still spans --steps")
  ap.add_argument("--resume", default="", help="ckpt/ folder of the previous stage")
  ap.add_argument("--init", default="", help="model folder to fine-tune (config.json + model.safetensors); --cfg is ignored")
  ap.add_argument("--sched_start", type=int, default=0, help="restart warmup + cosine at this step")
  ap.add_argument("--fpb", type=int, default=6000, help="frames per micro-batch per process")
  ap.add_argument("--accum", type=int, default=8, help="micro-batches per step; fpb x accum x processes = frames per step (original recipe: 48000)")
  ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--clip", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--dev", default="", help="held-out store for dev loss")
  ap.add_argument("--eval_every", type=int, default=0); ap.add_argument("--patience", type=int, default=0); ap.add_argument("--save_best", action="store_true")
  ap.add_argument("--cfg", default="{}", help="json overrides for Config")
  main(ap.parse_args())
