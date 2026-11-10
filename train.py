import argparse, json, math, os, threading, time
import numpy as np, torch, torch.distributed as dist
from model import Caramel, Config

CARD, DELAYS = Config.card, Config().delays
BUCKETS = [750, 1000, 1250, 1500]   # window lengths in frames: 60, 80, 100, 120 s
PEAK_BF16 = 989.4e12               # H100 SXM dense bf16, for MFU


# ── data ──
# A store is one folder: tokens.bin uint16 [frames, W] frame-major (W = 16 stereo: speaker 1 cb0-7 | speaker 2 cb0-7,
# or W = 8 mono), index.npy int64 [clips, 2] (offset, length) in frames, and meta.json {"width": W, "frames": total}.
class TokenStore:
  def __init__(self, d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    self.W, self.index = meta["width"], np.load(os.path.join(d, "index.npy"))
    self.tok = np.memmap(os.path.join(d, "tokens.bin"), dtype=np.uint16, mode="r", shape=(meta["frames"], self.W))

  def window(self, i, start, L): return self.tok[self.index[i, 0] + start:self.index[i, 0] + start + L]


# raw [L, W] -> seq [L+1, 16] int16, ready for Caramel.forward: start token first, then each codebook shifted by its delay
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
# see 4 shapes. Stores are picked by weight, clips by length, and the window start uniformly.
class BatchStream(torch.utils.data.IterableDataset):
  def __init__(self, dirs, fpb, weights=None, seed=0, rank=0): self.dirs, self.fpb, self.weights, self.seed, self.rank = dirs, fpb, weights, seed, rank

  def __iter__(self):
    wi = torch.utils.data.get_worker_info()
    rng, stores = np.random.default_rng([self.seed, self.rank, wi.id if wi else 0]), [TokenStore(d) for d in self.dirs]
    sw = np.asarray(self.weights or [s.index[:, 1].sum() for s in stores], dtype=np.float64)
    sw /= sw.sum()
    while True:
      L, seqs = int(rng.choice(BUCKETS)), []
      for _ in range(max(1, self.fpb // L)):
        s = stores[rng.choice(len(stores), p=sw)]
        lens = s.index[:, 1]
        i = int(rng.choice(len(lens), p=lens / lens.sum()))
        start = int(rng.integers(0, max(1, lens[i] - L + 1)))
        seqs.append(delay_window(np.asarray(s.window(i, start, L)), L, bool(rng.integers(2))))
      yield torch.from_numpy(np.stack(seqs))


# fixed held-out windows (fixed seed), each scored in both speaker orders; same shape as a training bucket so graphs are reused
def eval_batches(d, n=64, L=1500, bs=16, seed=12345):
  s, rng, seqs = TokenStore(d), np.random.default_rng(seed), []
  lens = s.index[:, 1]
  for _ in range(n):
    i = int(rng.choice(len(lens), p=lens / lens.sum()))
    raw = np.asarray(s.window(i, int(rng.integers(0, max(1, lens[i] - L + 1))), L))
    seqs += [delay_window(raw, L, False), delay_window(raw, L, True)]
  return [torch.from_numpy(np.stack(seqs[k:k + bs])) for k in range(0, len(seqs), bs)]


# ── schedule ──
# linear warmup to the peak, then cosine down to 0.1x peak at --steps. --sched_start restarts the schedule at a resumed step.
def lr_at(step, a):
  step -= a.sched_start
  if step < a.warmup: return a.lr * (step + 1) / a.warmup
  p = (step - a.warmup) / max(1, a.steps - a.sched_start - a.warmup)
  return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

# training FLOPs per frame: 6 x matmul params (forward + backward) + causal attention
def flops_per_frame(m, T):
  c = m.cfg
  tparams = sum(p.numel() for n, p in m.layers.named_parameters() if "norm" not in n)
  dparams = sum(p.numel() for n, p in m.dep_layers.named_parameters() if "norm" not in n) // c.dep_q
  attn = c.num_layers * 2 * T * c.dim + c.dep_layers * 2 * c.dep_q * c.dep_q * c.dep_dim / 2
  return 6 * (tparams + c.dep_q * (dparams + c.dim * c.dep_dim + c.dep_dim * c.card)) + 3 * attn


# ── train ──
def main(a):
  ddp = "RANK" in os.environ
  if ddp: dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
  rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
  cuda = torch.cuda.is_available()
  dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
  torch.manual_seed(a.seed)
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
  os.makedirs(a.out, exist_ok=True)
  cfg = Config(**json.loads(a.cfg))
  model = Caramel(cfg).to(dev)
  if cuda:
    # compile each block on its own; static shapes (one graph per length bucket) keep recompiles to 4
    torch._dynamo.config.automatic_dynamic_shapes, torch._dynamo.config.cache_size_limit = False, 16
    for blk in list(model.layers) + list(model.dep_layers): blk.compile()
    model.embed, model.head_ce = torch.compile(model.embed), torch.compile(model.head_ce)
  # AdamW; weight decay only on matrices, not on norms
  params = [p for p in model.parameters() if p.dim() >= 2], [p for p in model.parameters() if p.dim() < 2]
  opt = torch.optim.AdamW([{"params": params[0], "weight_decay": a.wd}, {"params": params[1], "weight_decay": 0.0}],
                          lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=cuda)
  start = 0
  if a.resume:   # the next stage keeps the weights, the optimizer state and the RNG of the previous one
    ck = torch.load(a.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    torch.set_rng_state(ck["rng_cpu"].cpu())
    if cuda and ck.get("rng_cuda") is not None: torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
  # DDP: gradients are all-reduced in 64 MB buckets while the backward pass is still running
  dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], bucket_cap_mb=64, gradient_as_bucket_view=True, static_graph=True) if ddp else model
  if rank == 0: print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  cfg {json.dumps(cfg.__dict__)}", flush=True)

  loader = iter(torch.utils.data.DataLoader(BatchStream(a.data, a.fpb, a.weights, a.seed + 7919 * start, rank), batch_size=None,
                num_workers=a.workers, pin_memory=cuda, prefetch_factor=4 if a.workers else None, persistent_workers=a.workers > 0))
  end = a.stop_at or a.steps
  eval_at = (set(range(0, a.steps + 1, a.eval_every)) if a.eval_every else set()) | ({a.stop_at} if a.stop_at else set())
  dev_batches = eval_batches(a.dev, bs=max(1, a.fpb // 1500)) if a.dev else []
  amp = lambda: torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda)
  log, evals, best, savers, stop = open(os.path.join(a.out, f"log_rank{rank}.jsonl"), "a"), [], [], [], torch.zeros(1, device=dev)

  def do_eval(s):
    model.eval()
    with torch.no_grad(), amp(): res = [model(b.to(dev).long()) for b in dev_batches]
    model.train()
    e = {"step": s, "dev_loss": float(np.mean([l.item() for l, _ in res])), "dev_cb": torch.stack([c.float().cpu() for _, c in res]).mean(0).tolist()}
    evals.append(e); log.write(json.dumps({"eval": e}) + "\n")
    if rank == 0:
      print(f"EVAL step {s} dev_loss {e['dev_loss']:.4f}", flush=True)
      if a.save_best and (not best or e["dev_loss"] < best[0]):   # best.pt (weights only) is written in the background
        best[:] = [e["dev_loss"], s]
        sd = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if savers: savers[-1].join()
        savers.append(threading.Thread(target=torch.save, args=({"model": sd, "step": s, "dev_loss": e["dev_loss"], "cfg": cfg.__dict__}, os.path.join(a.out, "best.pt"))))
        savers[-1].start()
      d = [x["dev_loss"] for x in evals[-(a.patience + 1):]]
      if a.patience and len(d) == a.patience + 1 and all(d[i + 1] > d[i] for i in range(a.patience)): stop.fill_(1); print(f"early stop at step {s}", flush=True)
    if ddp: dist.broadcast(stop, 0)

  t_last = time.perf_counter()
  for step in range(start, end):
    if step in eval_at and dev_batches:
      do_eval(step); t_last = time.perf_counter()
      if stop.item(): end = step; break
    seq = next(loader).to(dev, non_blocking=True).long()
    for g in opt.param_groups: g["lr"] = lr_at(step, a)
    with amp(): loss, per_cb = dmodel(seq)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
    opt.step(); opt.zero_grad(set_to_none=True)
    lv, t1 = loss.item(), time.perf_counter()
    dt, t_last = t1 - t_last, t1
    frames = seq.shape[0] * (seq.shape[1] - 1)
    rec = {"step": step + 1, "loss": lv, "cb": [round(x, 4) for x in per_cb.tolist()], "gn": gn.item(), "lr": lr_at(step, a), "dt": dt,
           "frames_s": frames / dt, "mfu": flops_per_frame(model, seq.shape[1] - 1) * frames / dt / PEAK_BF16}
    log.write(json.dumps(rec) + "\n")
    if rank == 0 and (step % 10 == 0 or step + 1 == end):
      print(f"step {step + 1} loss {lv:.4f} cb0 {rec['cb'][0]:.3f} lr {rec['lr']:.2e} dt {dt * 1e3:.0f}ms frames/s {rec['frames_s']:.0f} mfu {rec['mfu'] * 100:.1f}%", flush=True)
    if step + 1 == end and rank == 0:
      torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "cfg": cfg.__dict__, "rng_cpu": torch.get_rng_state(),
                  "rng_cuda": torch.cuda.get_rng_state() if cuda else None}, os.path.join(a.out, "ckpt.pt"))
  if end in eval_at and dev_batches and not stop.item(): do_eval(end)
  for t in savers: t.join()
  if ddp: dist.destroy_process_group()


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", nargs="+", required=True, help="token store folders")
  ap.add_argument("--weights", type=float, nargs="*", default=None, help="sampling weight per store (default: by frames)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--steps", type=int, required=True, help="end of the LR schedule")
  ap.add_argument("--stop_at", type=int, default=0, help="stop (and save ckpt.pt) here; the schedule still spans --steps")
  ap.add_argument("--resume", default="", help="ckpt.pt of the previous stage")
  ap.add_argument("--sched_start", type=int, default=0, help="restart warmup + cosine at this step")
  ap.add_argument("--fpb", type=int, default=24000, help="frames per batch per GPU")
  ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--clip", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--dev", default="", help="held-out store for dev loss")
  ap.add_argument("--eval_every", type=int, default=0); ap.add_argument("--patience", type=int, default=0); ap.add_argument("--save_best", action="store_true")
  ap.add_argument("--cfg", default="{}", help="json overrides for Config")
  main(ap.parse_args())
