"""Reinforcement learning for caramel's live behaviour (turn-taking and voice), after the supervised fine-tune.

Each step draws real conversation windows. For every window, `--group` rollouts are sampled: the user side (codebooks
8-15) is the real recording, forced frame by frame exactly as live.py feeds the microphone, the first `--prefix` frames
of speaker A are the real assistant (context), and after that the model speaks for itself. Every rollout and every real
track is decoded with Mimi; 80 ms frame energy gives who-is-talking, a log-mel profile gives whose voice it is. Rewards
are events on that timeline:

  yield      the user starts talking (>= --barge_min frames) while A talks: +1 if A falls silent within --yield_s, else -1
  talkover   A starts talking while the user talks and overlaps them for more than --backchannel_s: -1
  respond    the user ends a turn and A starts within --respond_s while the user stays quiet: +1
  dead_air   both silent for more than --max_silence_s (and again for every further --max_silence_s): -1
  voice      per chunk of A's speech: cos(chunk, assistant profile) - cos(chunk, this user's profile)

Rewards are summed per chunk of --chunk_s, normalized across the group of the same window (GRPO: no value model), and
a frame's advantage is the discounted sum of the normalized chunk rewards from its own chunk on (process rewards, as in
Hibiki-Zero; local credit, as in DuplexPO). The policy loss is the PPO-clipped ratio on speaker A's sampled tokens
(all 8 codebooks), plus an optional KL to the starting model and an optional supervised term on the real windows.
`--algo dpo` instead turns each group into one (best, worst) pair and trains length-normalized DPO against the reference.

  python rl.py --init runs/josie/best --data hf://<user>/josie-live-synthetic --steps 200 --out runs/josie-rl
  python rl.py --init runs/josie/best --data hf://<user>/josie-live-synthetic --steps 0 --batches 4 --out runs/score   # score only
"""
import argparse, json, math, os, time
from functools import partial
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from model import BF16, Caramel, Config, join_parts, save
from train import StreamStore, TokenStore, delay_window

SR, FPS, BLOCK = 24000, 12.5, 1920
EVENTS = ("yield", "talkover", "respond", "dead_air")


# ── data: full-length stereo windows [L, 16] (A = columns 0-7, the side being trained) ──
def windows(paths, L, seed, pool, hours):
  rng = np.random.default_rng(seed)
  stores = [StreamStore(p, seed, 0, 1, pool) if p.startswith("hf://") else TokenStore(p, hours) for p in paths]
  w = np.array([s.frames for s in stores], np.float64); w /= w.sum()
  misses = 0
  while True:
    raw = stores[rng.choice(len(stores), p=w)].sample(rng, L)
    if raw.shape == (L, 16): misses = 0; yield raw
    else:
      misses += 1
      if misses > 1000: raise SystemExit(f"no stereo clip of {L} frames ({L / FPS:.0f} s) in --data; lower --seconds")


# ── per-token log-probabilities of speaker A's 8 codebooks (the same forward as Caramel.__call__) ──
# seq [B, T+1, 16] delayed -> logp [B, T, 8] of seq[:, 1:, :8] under the model
def token_logp(model, seq):
  c = model.cfg
  inp, tgt = seq[:, :-1], seq[:, 1:, :c.dep_q]
  B, T, _ = inp.shape
  h = model.temporal(model.embed(inp)).reshape(B * T, c.dim)
  t = tgt.reshape(B * T, c.dep_q)
  logits = model.heads(model.depformer(h, t[:, :-1])).astype(mx.float32)   # [8, N, card]
  return -nn.losses.cross_entropy(logits, mx.clip(t.T, 0, c.card - 1)).T.reshape(B, T, c.dep_q)


# ── rollouts: the user stream is forced from data, A is forced for raw frames < P and sampled after ──
# seqs [B, L+1, 16] delayed real windows -> the same with A's tokens sampled where gen [L+1, 8] is True
def rollout(model, seqs, gen, temp, top_k, key):
  B, S, _ = seqs.shape
  K = model.cfg.dep_q
  st = model.init_state(B, S, temp, top_k)
  st["kc"] = [x.astype(BF16) for x in st["kc"]]; st["vc"] = [x.astype(BF16) for x in st["vc"]]
  forced = mx.array(np.where(gen[None], -2, seqs[:, :, :K]).astype(np.int32))
  user = mx.array(seqs[:, :, K:].astype(np.int32))
  a, outs = mx.array(seqs[:, 0, :K].astype(np.int32)), []
  for t in range(S - 1):
    key, sk = mx.random.split(key)
    a = model.step(st, mx.concatenate([a, user[:, t]], 1), forced[:, t + 1], sk)
    mx.async_eval(a)
    outs.append(a)
  A = np.array(mx.stack(outs, 1)).astype(np.int16)
  out = seqs.copy(); out[:, 1:, :K] = A
  return out


# delayed [B, L+1, 16] -> raw frames [B, 8, L-1] of side 0 (cols 0-7) or 1 (cols 8-15); the last frame lacks its acoustics
def undelay(seqs, side, delays):
  L = seqs.shape[1] - 1
  return np.stack([seqs[:, 1 + d:L + d, 8 * side + k] for k, d in enumerate(delays)], 1)


# ── audio features: decode each row with Mimi -> frame energy in dBFS [T] and log-mel [T, 40] ──
def mel_fb(n_mels=40, n_fft=2048, lo=60.0, hi=8000.0):
  hz = lambda m: 700 * (10 ** (m / 2595) - 1)
  mel = lambda f: 2595 * np.log10(1 + f / 700)
  pts = hz(np.linspace(mel(lo), mel(hi), n_mels + 2))
  f = np.fft.rfftfreq(n_fft, 1 / SR)
  fb = np.zeros((len(f), n_mels), np.float32)
  for i in range(n_mels):
    l, c, r = pts[i:i + 3]
    fb[:, i] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
  return fb


FB, WIN = mel_fb(), np.hanning(BLOCK).astype(np.float32)


def features(mimi, raw):   # raw [8, T] tokens of one track
  pcm = np.array(mimi.decode(mx.array(raw[None].astype(np.int32)))[0, 0].astype(mx.float32))
  T = raw.shape[1]
  x = np.pad(pcm, (0, max(0, T * BLOCK - len(pcm))))[:T * BLOCK].reshape(T, BLOCK)
  db = 10 * np.log10((x ** 2).mean(1) + 1e-10)
  mel = np.log(np.abs(np.fft.rfft(x * WIN, n=2048)) ** 2 @ FB + 1e-8)
  return db, mel, pcm


# runs of equal values in a 1-D bool array -> [(start, end, value)]
def runs(x):
  if not len(x): return []
  cut = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
  s, e = np.r_[0, cut], np.r_[cut, len(x)]
  return list(zip(s.tolist(), e.tolist(), x[s].tolist()))


# energy VAD: frames above thr, gaps shorter than min_gap closed (pauses inside a turn), spurts shorter than min_len dropped
def vad(db, thr, min_gap, min_len):
  v = db > thr
  for s, e, val in runs(v):
    if not val and s > 0 and e < len(v) and e - s < min_gap: v[s:e] = True
  for s, e, val in runs(v):
    if val and e - s < min_len: v[s:e] = False
  return v


# ── rewards ──
# a, u [T] bool talk activity of A and of the user; events are only scored from frame P on. -> [(frame, kind, value)]
def turn_events(a, u, P, f):
  T, ev = len(a), []
  for s, e, val in runs(u):
    if not val or s < P: continue
    if e - s >= f["barge_min"] and a[s - 1] and a[s] and s + f["yield"] + f["hold"] <= T:   # the user barges in on A
      quiet = [t for t in range(s, s + f["yield"] + 1) if not a[t:t + f["hold"]].any()]
      ev.append((quiet[0], "yield", 1.0) if quiet else (s + f["yield"], "yield", -1.0))
  ru = runs(u)
  for (ps, pe, pv), (s, e, val) in zip(ru, ru[1:]):   # the user finished a turn (pe == s): did A answer while the user waits?
    if not pv or s < P or pe - ps < f["turn_min"]: continue
    on = [t for t in range(s, min(s + f["respond"], e)) if a[t] and not a[t - 1]]
    if on: ev.append((on[0], "respond", 1.0))
  for s, e, val in runs(a):   # A starts talking over the user
    if not val or s < P or s == 0 or not u[s]: continue
    both = int(np.argmin(np.r_[a[s:e] & u[s:e], False]))
    if both > f["backchannel"]: ev.append((s + f["backchannel"], "talkover", -1.0))
  for s, e, val in runs(~(a | u)):   # dead air
    if not val: continue
    for t in range(max(s, P - 1) + f["max_silence"], e, f["max_silence"]): ev.append((t, "dead_air", -1.0))
  return ev


def cos(x, y): return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))


class Profiles:   # running speech-frame log-mel means: the assistant (side A of real data) and a centre over both sides
  def __init__(self): self.a = self.c = None
  def update(self, mel_a, mel_u, mom=0.9):
    a, c = mel_a.mean(0), np.concatenate([mel_a, mel_u]).mean(0)
    self.a = a if self.a is None else mom * self.a + (1 - mom) * a
    self.c = c if self.c is None else mom * self.c + (1 - mom) * c


# per-chunk voice score of A's speech frames: closer to the assistant than to this window's user is positive
def voice_scores(mel, a, user_prof, prof, P, C, n_chunks, min_frames=5):
  out = np.zeros(n_chunks)
  for j in range(n_chunks):
    s = P + j * C
    m = a[s:s + C]
    if m.sum() >= min_frames:
      v = mel[s:s + C][m].mean(0) - prof.c
      out[j] = cos(v, prof.a - prof.c) - cos(v, user_prof - prof.c)
  return out


# ── training ──
def main(a):
  mx.random.seed(a.seed)
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
  if a.init:
    join_parts(a.init)
    cfg = Config.from_dict(json.load(open(os.path.join(a.init, "config.json"))))
  else: cfg = Config(**json.loads(a.cfg))   # random weights: smoke tests only
  model = Caramel(cfg)
  if a.init: model.load_weights(os.path.join(a.init, "model.safetensors"), strict=True)
  model.set_dtype(mx.float32 if a.steps else BF16)   # float32 master weights, bf16 matmuls; bf16 when only scoring
  ref = None
  if a.steps and (a.algo == "dpo" or a.kl > 0):   # frozen starting model
    ref = Caramel(cfg)
    if a.init: ref.load_weights(os.path.join(a.init, "model.safetensors"), strict=True)
    else: ref.update(model.parameters())
    ref.set_dtype(BF16); ref.freeze(); mx.eval(ref.parameters())
  from generate import load_mimi
  mimi = load_mimi()

  K, delays = cfg.dep_q, cfg.delays[:cfg.dep_q]
  L, P, C = round(a.seconds * FPS), round(a.prefix_s * FPS), round(a.chunk_s * FPS)
  assert 0 < P < L - C and L + 1 <= cfg.context, "need 0 < --prefix_s < --seconds - --chunk_s, within the model's context"
  Lr = L - 1   # decoded frames
  n_chunks = math.ceil((Lr - P) / C)
  fr = lambda s: max(1, round(s * FPS))
  f = {"yield": fr(a.yield_s), "hold": fr(a.hold_s), "barge_min": fr(a.barge_min_s), "backchannel": fr(a.backchannel_s),
       "respond": fr(a.respond_s), "turn_min": fr(a.turn_min_s), "max_silence": fr(a.max_silence_s)}
  w = {"yield": a.w_yield, "talkover": a.w_talkover, "respond": a.w_respond, "dead_air": a.w_dead_air}
  frame = np.arange(L + 1)[:, None] - 1 - np.array(delays)[None]   # raw frame of each (delayed row, codebook)
  gen = frame >= P                                                  # tokens the model samples (and is trained on)
  tmask = (gen & (frame < L))[1:].astype(np.float32)               # [L, 8] target mask (row 0 is the start token)
  vad_args = (a.vad_db, fr(a.vad_gap_s), fr(a.vad_min_s))

  adam = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0, bias_correction=True)
  if a.steps: adam.init(model.trainable_parameters())

  # losses; `den` is the total token count (or pair count) of the whole batch, so micro-batch gradients just add up
  use_old, use_ref = a.inner > 1, ref is not None   # unused lp_old / lp_ref are passed as placeholders

  def grpo(seq, mask, adv, lp_old, lp_ref, den):
    lp = token_logp(model, seq)
    r = mx.exp(lp - (lp_old if use_old else mx.stop_gradient(lp)))
    loss = -mx.minimum(r * adv, mx.clip(r, 1 - a.clip_eps, 1 + a.clip_eps) * adv)
    if use_ref:   # k3 estimator of KL(policy || ref), per token
      d = lp_ref - lp; loss = loss + a.kl * (mx.exp(d) - d - 1)
    return (loss * mask).sum() / den

  def dpo(seq, mask, lp_ref, den):   # rows [chosen, rejected, chosen, rejected, ...]; length-normalized (DPO-LN)
    s = ((token_logp(model, seq) - lp_ref) * mask).sum((1, 2)) / mx.maximum(mask.sum((1, 2)), 1)
    return -nn.log_sigmoid(a.beta * (s[0::2] - s[1::2])).sum() / den

  def sft(seq, den): return model(seq)[0] / den
  grads_of = {k: partial(mx.compile, inputs=model.state, outputs=model.state)(nn.value_and_grad(model, fn))
              for k, fn in (("grpo", grpo), ("dpo", dpo), ("sft", sft))}
  logp_now = mx.compile(partial(token_logp, model), inputs=model.state)
  logp_ref = mx.compile(partial(token_logp, ref)) if ref is not None else None

  def in_mbs(n, fn, step):   # run fn(slice) over rows in micro-batches
    return [fn(slice(i, min(n, i + step))) for i in range(0, n, step)]

  data, prof = windows(a.data, L, a.seed, a.pool, a.train_hours), Profiles()
  log = open(os.path.join(a.out, "log.jsonl"), "a")
  steps = a.steps if a.steps else a.batches
  for step in range(steps):
    t0 = time.perf_counter()
    raws = [next(data) for _ in range(a.windows)]
    real = np.stack([delay_window(r, L, False) for r in raws])        # [W, L+1, 16]
    seqs = np.repeat(real, a.group, 0)                                  # [W*G, L+1, 16]
    model.eval()
    roll = rollout(model, seqs, gen, a.temp, a.top_k, mx.random.key(a.seed * 100003 + step))
    model.train()
    t_roll = time.perf_counter() - t0

    # who talks when, and in whose voice
    feats_real = [(features(mimi, x[0]), features(mimi, x[1])) for x in zip(undelay(real, 0, delays), undelay(real, 1, delays))]
    vr = [(vad(fa[0], *vad_args), vad(fu[0], *vad_args)) for fa, fu in feats_real]
    for (fa, fu), (va, vu) in zip(feats_real, vr):
      if va.sum() > 5 and vu.sum() > 5: prof.update(fa[1][va], fu[1][vu])
    feats_roll = [features(mimi, x) for x in undelay(roll, 0, delays)]
    R = np.zeros((len(roll), n_chunks)); stats = {k: [] for k in EVENTS}; voice, talk, pcms = [], [], []
    for i, (db, mel, pcm) in enumerate(feats_roll):
      wi = i // a.group
      (_, mel_u, _), vu = feats_real[wi][1], vr[wi][1]
      va = vad(db, *vad_args)
      for t, kind, val in turn_events(va, vu, P, f):
        R[i, min(n_chunks - 1, (t - P) // C)] += w[kind] * val; stats[kind].append(val)
      if prof.a is not None and vu.sum() > 5:
        vs = voice_scores(mel, va, mel_u[vu].mean(0), prof, P, C, n_chunks)
        R[i] += a.w_voice * vs; voice += vs[vs != 0].tolist()
      talk.append(va[P:].mean()); pcms.append(pcm)
    # the real assistant on the same windows, scored the same way: what the rewards look like for the target behaviour
    ref_stats = {k: [] for k in EVENTS}
    for va, vu in vr:
      for _, kind, val in turn_events(va, vu, P, f): ref_stats[kind].append(val)

    # GRPO advantages: normalize each chunk's reward across the group of one window, then discounted sum from a frame's chunk on
    Rg = R.reshape(a.windows, a.group, n_chunks)
    Rn = ((Rg - Rg.mean(1, keepdims=True)) / (Rg.std(1, keepdims=True) + 1e-4)).reshape(-1, n_chunks)
    adv_c = np.zeros_like(Rn)
    for j in reversed(range(n_chunks)): adv_c[:, j] = Rn[:, j] + (a.gamma * adv_c[:, j + 1] if j + 1 < n_chunks else 0)
    ch = np.clip((frame[1:] - P) // C, 0, n_chunks - 1)                 # [L, 8] chunk of each target token
    adv = adv_c[:, ch].astype(np.float32)                               # [B, L, 8]

    total = R.sum(1)
    rec = {"step": step + 1, "reward": float(total.mean()), "reward_std_in_group": float(Rg.sum(2).std(1).mean()),
           "talk_frac": float(np.mean(talk)), "voice": float(np.mean(voice)) if voice else None, "t_rollout": t_roll}
    min_gen = (Lr - P) * a.windows * a.group / FPS / 60
    for k in EVENTS:
      rec[k] = {"n_per_min": len(stats[k]) / min_gen, "mean": float(np.mean(stats[k])) if stats[k] else None,
                "data_mean": float(np.mean(ref_stats[k])) if ref_stats[k] else None}

    if a.steps:   # update
      opt_lr = a.lr * min(1.0, (step + 1) / max(1, a.warmup)); adam.learning_rate = opt_lr
      if a.algo == "grpo":
        rows, mask = mx.array(roll.astype(np.int32)), mx.array(np.broadcast_to(tmask, (len(roll), L, K)).copy())
        advm, den = mx.array(adv), mx.array(float(tmask.sum() * len(roll)))
        lp_ref = (mx.concatenate(in_mbs(len(roll), lambda s: logp_ref(rows[s]), a.mb)) if ref is not None else None)
        lp_old = mx.concatenate(in_mbs(len(roll), lambda s: logp_now(rows[s]), a.mb)) if a.inner > 1 else None
        if lp_ref is not None: mx.eval(lp_ref)
        if lp_old is not None: mx.eval(lp_old)
        fn = lambda s: grads_of["grpo"](rows[s], mask[s], advm[s], (lp_old if use_old else mask)[s], (lp_ref if use_ref else mask)[s], den)
        n_rows, mb = len(roll), a.mb
      else:   # DPO: best vs worst rollout of each window
        best, worst = Rg.sum(2).argmax(1), Rg.sum(2).argmin(1)
        keep = [wi for wi in range(a.windows) if Rg[wi].sum(1)[best[wi]] - Rg[wi].sum(1)[worst[wi]] >= a.margin]
        idx = [wi * a.group + j for wi in keep for j in (best[wi], worst[wi])]
        rec["pairs"] = len(keep)
        if not idx: fn = None
        else:
          rows = mx.array(roll[idx].astype(np.int32)); mask = mx.array(np.broadcast_to(tmask, (len(idx), L, K)).copy())
          lp_ref = mx.concatenate(in_mbs(len(idx), lambda s: logp_ref(rows[s]), a.mb)); mx.eval(lp_ref)
          den = mx.array(float(len(keep)))
          fn = lambda s: grads_of["dpo"](rows[s], mask[s], lp_ref[s], den)
          n_rows, mb = len(idx), max(2, a.mb - a.mb % 2)
      if fn is not None:
        for _ in range(a.inner if a.algo == "grpo" else 1):
          grads, loss = None, 0.0
          for s in range(0, n_rows, mb):
            l, g = fn(slice(s, min(n_rows, s + mb)))
            grads = g if grads is None else tree_map(mx.add, grads, g); loss = loss + l
            mx.eval(grads, loss)
          if a.sft > 0:   # supervised anchor on the real windows (assistant side), as in RLHF's pretraining mix
            rs, den_s = mx.array(real.astype(np.int32)), mx.array(float(math.ceil(len(real) / a.mb)))
            for s in range(0, len(real), a.mb):
              l, g = grads_of["sft"](rs[s:s + a.mb], den_s)
              grads = tree_map(lambda x, y: x + a.sft * y, grads, g); rec["sft_loss"] = rec.get("sft_loss", 0.0) + l.item()
              mx.eval(grads)
          grads, gn = optim.clip_grad_norm(grads, a.grad_clip)
          adam.update(model, grads)
          mx.eval(model.parameters(), adam.state)
        rec.update(loss=float(loss.item()), gn=float(gn.item()), lr=opt_lr)
    rec["dt"] = time.perf_counter() - t0
    log.write(json.dumps(rec) + "\n"); log.flush()
    ev = "  ".join(f"{k} {rec[k]['n_per_min']:.1f}/min" + (f" ({rec[k]['mean']:+.2f}, data {rec[k]['data_mean'] if rec[k]['data_mean'] is None else round(rec[k]['data_mean'], 2)})" if rec[k]["mean"] is not None else "") for k in EVENTS)
    print(f"step {step + 1} reward {rec['reward']:+.2f} talk {rec['talk_frac']:.2f} voice {rec['voice'] if rec['voice'] is None else round(rec['voice'], 3)}"
          + (f" loss {rec['loss']:+.4f} gn {rec['gn']:.3f}" if "loss" in rec else "") + f" | {ev} | {rec['dt']:.0f}s (rollout {t_roll:.0f}s)", flush=True)
    if a.dump and (step % a.dump_every == 0 or step + 1 == steps):   # listen: left = model (rollout 0), right = the user
      import soundfile as sf
      d = os.path.join(a.out, "samples"); os.makedirs(d, exist_ok=True)
      user = np.array(mimi.decode(mx.array(undelay(real[:1], 1, delays).astype(np.int32)))[0, 0].astype(mx.float32))
      n = min(len(pcms[0]), len(user))
      sf.write(os.path.join(d, f"step{step + 1:05d}.wav"), np.stack([pcms[0][:n], user[:n]], 1), SR)
    if a.steps and ((step + 1) % a.save_every == 0 or step + 1 == steps): save(model, os.path.join(a.out, f"step{step + 1:05d}"))   # keep them all: pick by score, RL can over-optimize


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", nargs="+", required=True, help="stereo token stores or hf://<user>/<dataset> (speaker A = the side to train)")
  ap.add_argument("--init", default="", help="model folder to start from (config.json + model.safetensors), e.g. runs/josie/best")
  ap.add_argument("--cfg", default="{}", help="json Config for random weights when there is no --init (smoke tests only)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--algo", default="grpo", choices=["grpo", "dpo"])
  ap.add_argument("--steps", type=int, default=100, help="updates; 0 = only roll out and score --batches batches")
  ap.add_argument("--batches", type=int, default=4, help="batches scored with --steps 0")
  ap.add_argument("--windows", type=int, default=2, help="data windows per step")
  ap.add_argument("--group", type=int, default=4, help="rollouts per window")
  ap.add_argument("--seconds", type=float, default=60.0, help="window length")
  ap.add_argument("--prefix_s", type=float, default=15.0, help="seconds of real assistant before the model takes over")
  ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--top_k", type=int, default=250)
  ap.add_argument("--lr", type=float, default=5e-7); ap.add_argument("--warmup", type=int, default=10)
  ap.add_argument("--grad_clip", type=float, default=1.0)
  ap.add_argument("--clip_eps", type=float, default=0.2, help="PPO ratio clip")
  ap.add_argument("--inner", type=int, default=1, help="updates per rollout batch (> 1 makes the ratio and clip matter)")
  ap.add_argument("--kl", type=float, default=0.02, help="KL-to-start coefficient (0 = no reference model)")
  ap.add_argument("--beta", type=float, default=0.3, help="DPO beta"); ap.add_argument("--margin", type=float, default=1.0, help="DPO: min reward gap of a pair")
  ap.add_argument("--sft", type=float, default=0.0, help="weight of the supervised loss on the real windows")
  ap.add_argument("--gamma", type=float, default=0.5, help="discount per chunk when summing later chunk rewards into an advantage")
  ap.add_argument("--chunk_s", type=float, default=2.0)
  ap.add_argument("--mb", type=int, default=4, help="rows per micro-batch in the loss")
  # rewards
  ap.add_argument("--w_yield", type=float, default=1.0); ap.add_argument("--w_talkover", type=float, default=1.0)
  ap.add_argument("--w_respond", type=float, default=0.5); ap.add_argument("--w_dead_air", type=float, default=1.0)
  ap.add_argument("--w_voice", type=float, default=1.0)
  ap.add_argument("--yield_s", type=float, default=1.0, help="A must fall silent this soon after a barge-in")
  ap.add_argument("--hold_s", type=float, default=0.32, help="... and stay silent this long")
  ap.add_argument("--barge_min_s", type=float, default=0.64, help="shorter user spurts are backchannels, not barge-ins")
  ap.add_argument("--backchannel_s", type=float, default=1.0, help="A may overlap the user this long (a backchannel)")
  ap.add_argument("--respond_s", type=float, default=1.5); ap.add_argument("--turn_min_s", type=float, default=1.0)
  ap.add_argument("--max_silence_s", type=float, default=3.0)
  ap.add_argument("--vad_db", type=float, default=-45.0, help="frame energy (dBFS) above which a track counts as talking")
  ap.add_argument("--vad_gap_s", type=float, default=0.4); ap.add_argument("--vad_min_s", type=float, default=0.16)
  # data / io
  ap.add_argument("--pool", type=int, default=16, help="clips held from a streamed hf:// dataset")
  ap.add_argument("--train_hours", type=float, default=0)
  ap.add_argument("--save_every", type=int, default=25)
  ap.add_argument("--dump", action="store_true", help="write samples/stepN.wav (left = rollout 0, right = user)")
  ap.add_argument("--dump_every", type=int, default=10)
  ap.add_argument("--seed", type=int, default=0)
  main(ap.parse_args())
