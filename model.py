import glob, hashlib, json, math, os, re, shutil, sys
from dataclasses import dataclass, field, asdict
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file, save_file
from kernels import dep_attn, rms_norm, add_rms_norm, rms_norm_ref


@dataclass
class Config:
  card: int = 2048; n_q: int = 16; dep_q: int = 8; context: int = 3000; max_period: float = 10000.0
  dim: int = 1024; num_layers: int = 16; num_heads: int = 16; hidden_scale: float = 4.125
  dep_dim: int = 512; dep_layers: int = 6; dep_heads: int = 8; dep_hidden_scale: float = 4.125
  # Mimi's codebook 0 carries what is said, 1..7 how it sounds. Acoustic codebooks run one frame behind codebook 0, so the
  # model settles the words of a frame before its acoustics. Layout: 8 codebooks of the speaker being modelled, then 8 of the other.
  delays: list = field(default_factory=lambda: [0] + [1] * 7 + [0] + [1] * 7)

  @classmethod
  def from_dict(cls, d): return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _hidden(dim, scale): return 2 * int(scale * dim) // 3   # SiLU-gated FFN: two thirds of a plain FFN's width, same params
def _trunc_normal(w, fan_in): std = 1 / math.sqrt(fan_in); nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)


class RMSNorm(nn.Module):
  def __init__(self, dim): super().__init__(); self.weight = nn.Parameter(torch.ones(dim))
  def forward(self, x): return rms_norm_ref(x, self.weight)


def rope_tables(T, hd, max_period, device):
  ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * torch.exp(torch.arange(0, hd, 2, device=device, dtype=torch.float32) * (-math.log(max_period) / hd))
  return ang.cos(), ang.sin()   # [T, hd/2]

# rotary position embedding on interleaved pairs (2i, 2i+1): each pair is rotated by an angle that grows with the frame index
def apply_rope(x, cos, sin):
  xf = x.float().view(*x.shape[:-1], -1, 2)
  xr, xi, c, s = xf[..., 0], xf[..., 1], cos[:, None, :], sin[:, None, :]
  return torch.stack([xr * c - xi * s, xr * s + xi * c], dim=-1).view_as(x).to(x.dtype)


# ── temporal transformer: one position per 80 ms frame, causal over time ──
class TemporalBlock(nn.Module):
  def __init__(self, c: Config):
    super().__init__()
    d, self.h, hid = c.dim, c.num_heads, _hidden(c.dim, c.hidden_scale)
    self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
    self.in_proj, self.out_proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)   # in_proj rows [q | k | v]
    self.lin_in, self.lin_out = nn.Linear(d, 2 * hid, bias=False), nn.Linear(hid, d, bias=False)

  def ffn(self, x): a, b = self.lin_in(x).chunk(2, dim=-1); return self.lin_out(F.silu(a) * b)
  def qkv(self, x): B, T, _ = x.shape; return self.in_proj(x).view(B, T, 3, self.h, -1).unbind(2)

  def forward(self, x, cos, sin):
    B, T, D = x.shape
    q, k, v = self.qkv(rms_norm(x, self.norm1.weight))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    x, h = add_rms_norm(x, self.out_proj(o.transpose(1, 2).reshape(B, T, D)), self.norm2.weight)
    return x + self.ffn(h)

  # one frame: x [B, 1, D]; kc/vc [B, H, Tmax, hd] preallocated caches written at pos, so every shape stays fixed (CUDA graph safe)
  def step(self, x, cos, sin, kc, vc, pos, mask):
    B, T, D = x.shape
    q, k, v = self.qkv(self.norm1(x))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    kc.index_copy_(2, pos.view(1), k.transpose(1, 2)); vc.index_copy_(2, pos.view(1), v.transpose(1, 2))
    o = F.scaled_dot_product_attention(q.transpose(1, 2), kc, vc, attn_mask=mask)
    x = x + self.out_proj(o.transpose(1, 2).reshape(B, T, D))
    return x + self.ffn(self.norm2(x))


# K independent linears, one per codebook, applied to x [K, N, din] -> [K, N, dout] as a single batched matmul
class MultiLinear(nn.Module):
  def __init__(self, K, din, dout): super().__init__(); self.weight = nn.Parameter(torch.empty(K, din, dout)); _trunc_normal(self.weight, din)
  def forward(self, x): return torch.bmm(x, self.weight if torch.is_autocast_enabled() else self.weight.to(x.dtype))
  def one(self, x, k): return x @ (self.weight[k] if torch.is_autocast_enabled() else self.weight[k].to(x.dtype))

  # same input for all K: one [N, din] @ [din, K*dout] matmul -> [K, N, dout]
  def fan_out(self, x):
    K, din, dout = self.weight.shape
    w = self.weight.permute(1, 0, 2).reshape(din, K * dout)
    return (x @ (w if torch.is_autocast_enabled() else w.to(x.dtype))).view(x.shape[0], K, dout).transpose(0, 1)


# ── depth transformer: within one frame, predicts codebook k from the frame's hidden state and codebooks 0..k-1 ──
# It is causal over the 8 codebooks and has separate weights for each codebook position (MultiLinear).
class DepBlock(nn.Module):
  def __init__(self, c: Config):
    super().__init__()
    K, d, self.h, hid = c.dep_q, c.dep_dim, c.dep_heads, _hidden(c.dep_dim, c.dep_hidden_scale)
    self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
    self.in_proj, self.out_proj = MultiLinear(K, d, 3 * d), MultiLinear(K, d, d)
    self.lin_in, self.lin_out = MultiLinear(K, d, 2 * hid), MultiLinear(K, hid, d)

  # x [Kq, N, d]: all frames of the batch at once, the 8 codebooks as the sequence
  def forward(self, x):
    o = dep_attn(self.in_proj(rms_norm(x, self.norm1.weight)), self.h)
    x, h = add_rms_norm(x, self.out_proj(o), self.norm2.weight)
    a, b = self.lin_in(h).chunk(2, dim=-1)
    return x + self.lin_out(F.silu(a) * b)

  # x [B, d] for codebook k; kc/vc [B, H, Kq, hd] hold codebooks 0..k-1 of this frame
  def step(self, x, k, kc, vc):
    B, d = x.shape
    q, k_, v = self.in_proj.one(self.norm1(x), k).view(B, 3, self.h, -1).unbind(1)
    kc[:, :, k], vc[:, :, k] = k_, v
    o = F.scaled_dot_product_attention(q[:, :, None], kc[:, :, :k + 1], vc[:, :, :k + 1])
    x = x + self.out_proj.one(o.reshape(B, d), k)
    a, b = self.lin_in.one(self.norm2(x), k).chunk(2, dim=-1)
    return x + self.lin_out.one(F.silu(a) * b, k)


class Caramel(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg, C, Kq = cfg, cfg.card + 1, cfg.dep_q   # token id == card is the "start / delay filler" token
    # all 16 input codebooks share one flat embedding table (row = k * C + token): one gather, then a sum over codebooks
    self.emb = nn.Embedding(cfg.n_q * C, cfg.dim)
    self.register_buffer("emb_offsets", torch.arange(cfg.n_q) * C, persistent=False)
    self.layers = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.num_layers)])
    self.out_norm = RMSNorm(cfg.dim)
    self.dep_in = MultiLinear(Kq, cfg.dim, cfg.dep_dim)
    # embeddings of codebooks 0..6, fed into depth steps 1..7
    self.dep_emb = nn.Embedding((Kq - 1) * C, cfg.dep_dim)
    self.register_buffer("dep_offsets", torch.arange(Kq - 1) * C, persistent=False)
    self.dep_layers = nn.ModuleList([DepBlock(cfg) for _ in range(cfg.dep_layers)])
    self.heads = MultiLinear(Kq, cfg.dep_dim, cfg.card)
    for m in self.modules():
      if isinstance(m, nn.Linear): _trunc_normal(m.weight, m.in_features)
      elif isinstance(m, nn.Embedding): _trunc_normal(m.weight, m.embedding_dim)
    # small output heads so a randomly initialised model starts at loss ~ln(2048) instead of ~10
    with torch.no_grad(): self.heads.weight.mul_(0.1)

  # tokens [B, T, n_q], -1 = no input (masked stream): contributes a zero vector
  def embed(self, tokens):
    e = self.emb(tokens.clamp_min(0) + self.emb_offsets) * (tokens >= 0)[..., None].to(self.emb.weight.dtype)
    return e.sum(2)

  def temporal(self, x):
    cos, sin = rope_tables(x.shape[1], self.cfg.dim // self.cfg.num_heads, self.cfg.max_period, x.device)
    for layer in self.layers: x = layer(x, cos, sin)
    return self.out_norm(x)

  # h [N, dim]; prev [N, dep_q-1] true tokens of codebooks 0..6 (teacher forcing). Returns [Kq, N, dep_dim]
  def depformer(self, h, prev):
    x = self.dep_in.fan_out(h)
    te = self.dep_emb(prev.clamp_min(0) + self.dep_offsets) * (prev >= 0)[..., None].to(self.dep_emb.weight.dtype)
    x = x + F.pad(te.transpose(0, 1), (0, 0, 0, 0, 1, 0)).to(x.dtype)   # step 0 gets no token
    for layer in self.dep_layers: x = layer(x)
    return x

  def head_ce(self, z, tgt): return F.cross_entropy(self.heads(z).float().flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)

  # seq [B, T+1, n_q] delayed codes with seq[:, 0] = start token. Returns (loss, per-codebook loss)
  def forward(self, seq):
    c = self.cfg
    inp, tgt = seq[:, :-1], seq[:, 1:]
    B, T, _ = inp.shape
    h = self.temporal(self.embed(inp)).reshape(B * T, c.dim)
    tgt = tgt[..., :c.dep_q].reshape(B * T, c.dep_q)
    z = self.depformer(h, tgt[:, :-1])
    tgt = tgt.t()                                   # [Kq, N]
    # each codebook's loss is averaged over its own real targets (masked -1 and delay filler excluded), then the 8 are averaged equally
    m = ((tgt >= 0) & (tgt < c.card)).float()
    per_cb = (self.head_ce(z, tgt.clamp(0, c.card - 1)) * m).sum(1) / m.sum(1).clamp_min(1)
    return per_cb.mean(), per_cb.detach()

  # ── streaming inference ──
  def init_state(self, B, Tmax=None, dtype=torch.bfloat16):
    c, dev = self.cfg, self.emb.weight.device
    Tmax, hd, dhd = Tmax or c.context, c.dim // c.num_heads, c.dep_dim // c.dep_heads
    z = lambda *s: torch.zeros(*s, device=dev, dtype=dtype)
    cos, sin = rope_tables(Tmax, hd, c.max_period, dev)
    return {"kc": [z(B, c.num_heads, Tmax, hd) for _ in self.layers], "vc": [z(B, c.num_heads, Tmax, hd) for _ in self.layers],
            "dkc": [z(B, c.dep_heads, c.dep_q, dhd) for _ in self.dep_layers], "dvc": [z(B, c.dep_heads, c.dep_q, dhd) for _ in self.dep_layers],
            "pos": torch.zeros((), dtype=torch.long, device=dev), "cos": cos, "sin": sin, "ar": torch.arange(Tmax, device=dev)}

  # One 80 ms frame. inp [B, n_q] delayed tokens of the previous frame (card = start, -1 = none); forced [B, dep_q] (>= 0 forces
  # that token, -2 samples). Returns the frame's delayed tokens [B, dep_q]. Static shapes only, so it can be captured in a CUDA graph.
  @torch.no_grad()
  def step(self, st, inp, forced, temp=1.1, top_k=50):
    c, pos = self.cfg, st["pos"]
    x = self.embed(inp[:, None]).to(torch.bfloat16)
    cos, sin = st["cos"].index_select(0, pos.view(1)), st["sin"].index_select(0, pos.view(1))
    mask = (st["ar"] <= pos)[None, None, None, :]
    for i, layer in enumerate(self.layers): x = layer.step(x, cos, sin, st["kc"][i], st["vc"][i], pos, mask)
    h, out, prev = self.out_norm(x)[:, 0], [], None
    for k in range(c.dep_q):
      z = self.dep_in.one(h, k)
      if prev is not None: z = z + self.dep_emb(prev + (k - 1) * (c.card + 1)).to(z.dtype)
      for j, layer in enumerate(self.dep_layers): z = layer.step(z, k, st["dkc"][j], st["dvc"][j])
      tok = sample_topk(self.heads.one(z, k).float(), temp, top_k)
      prev = torch.where(forced[:, k] >= 0, forced[:, k], tok)
      out.append(prev)
    pos.add_(1)
    return torch.stack(out, 1)


# keep the top_k logits, divide by temp, sample. Gumbel-max via an exponential race: argmax(p / Exp(1)) is a draw from p,
# with no data-dependent shapes or host syncs, so it runs inside a CUDA graph. temp <= 0 is greedy.
def sample_topk(logits, temp, top_k):
  if temp <= 0: return logits.argmax(-1)
  v, i = logits.topk(top_k, dim=-1)
  p = torch.softmax(v / temp, dim=-1)
  return i.gather(-1, (p / torch.empty_like(p).exponential_()).argmax(-1, keepdim=True))[:, 0]


# ── weights: a folder with config.json and model.safetensors (or model.safetensors.partNNN pieces, joined on first load) ──
def load(path, device="cuda", dtype=torch.bfloat16):
  cfg, f = Config.from_dict(json.load(open(os.path.join(path, "config.json")))), os.path.join(path, "model.safetensors")
  if not os.path.exists(f):
    parts = sorted(glob.glob(f + ".part*"), key=lambda p: int(re.sub(r"\D", "", p.rsplit(".part", 1)[1]) or 0))
    assert parts, f"no model.safetensors or model.safetensors.part* in {path}"
    with open(f + ".tmp", "wb") as o:
      for p in parts:
        with open(p, "rb") as i: shutil.copyfileobj(i, o, 1 << 24)
    sums = os.path.join(path, "SHA256SUMS")
    want = dict(l.split()[::-1] for l in open(sums) if l.strip()).get("model.safetensors") if os.path.exists(sums) else None
    if want: assert hashlib.file_digest(open(f + ".tmp", "rb"), "sha256").hexdigest() == want, "joined model.safetensors fails SHA256SUMS"
    os.replace(f + ".tmp", f)
  m = Caramel(cfg)
  m.load_state_dict(load_file(f), strict=True)
  return m.to(device, dtype).eval()

# training checkpoint (.pt from train.py: {"model", "cfg", ...}) -> config.json + model.safetensors (bf16, as generation runs)
def convert(ckpt, out, dtype=torch.bfloat16):
  ck = torch.load(ckpt, map_location="cpu", weights_only=False)
  cfg = Config.from_dict(ck["cfg"])
  Caramel(cfg).load_state_dict(ck["model"], strict=True)   # fails loudly if the checkpoint is not this architecture
  os.makedirs(out, exist_ok=True)
  json.dump(asdict(cfg), open(os.path.join(out, "config.json"), "w"), indent=1)
  save_file({k: v.to(dtype).contiguous() for k, v in ck["model"].items()}, os.path.join(out, "model.safetensors"))


if __name__ == "__main__": convert(sys.argv[1], sys.argv[2])   # python model.py runs/s3/best.pt ./caramel-1
