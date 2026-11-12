import json, math, os, sys
from dataclasses import dataclass, field, asdict
from functools import partial
import mlx.core as mx, mlx.nn as nn

BF16 = mx.bfloat16   # matmul / attention dtype; training keeps float32 master weights and a float32 temporal residual stream


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
def _trunc_normal(shape, fan_in): return mx.random.truncated_normal(-3.0, 3.0, shape) / math.sqrt(fan_in)

# RMSNorm in float32 (fused Metal kernel), output in the matmul dtype
def norm(x, w, eps=1e-5): return mx.fast.rms_norm(x, w.astype(x.dtype), eps).astype(BF16)
def swiglu(x): a, b = mx.split(x, 2, axis=-1); return nn.silu(a) * b


class RMSNorm(nn.Module):
  def __init__(self, dim): super().__init__(); self.weight = mx.ones((dim,))


# bias-free linear whose weight is cast to the input dtype, so float32 master weights run bf16 matmuls.
# Subclasses nn.Linear so nn.quantize turns it into a QuantizedLinear for generation.
class Linear(nn.Linear):
  def __init__(self, din, dout): super().__init__(din, dout, bias=False); self.weight = _trunc_normal((dout, din), din)
  def __call__(self, x): return x @ self.weight.astype(x.dtype).T


# K independent linears, one per codebook: x [K, N, din] @ weight [K, din, dout] is one batched matmul
class MultiLinear(nn.Module):
  def __init__(self, K, din, dout): super().__init__(); self.weight = _trunc_normal((K, din, dout), din)
  def __call__(self, x): return x @ self.weight.astype(x.dtype)
  def one(self, x, k): return x @ self.weight[k].astype(x.dtype)

  # same input for all K: one [N, din] @ [din, K*dout] matmul -> [K, N, dout]
  def fan_out(self, x):
    K, din, dout = self.weight.shape
    w = self.weight.transpose(1, 0, 2).reshape(din, K * dout).astype(x.dtype)
    return (x @ w).reshape(x.shape[0], K, dout).transpose(1, 0, 2)

  def to_quantized(self, group_size=32, bits=8, mode="affine"): return QuantizedMultiLinear(self.weight, group_size, bits)


# generation only: per-codebook weights stored transposed and quantized, used one codebook at a time
class QuantizedMultiLinear(nn.Module):
  def __init__(self, w, group_size, bits):
    super().__init__()
    self.group_size, self.bits = group_size, bits
    self.weight, self.scales, self.biases = mx.quantize(w.swapaxes(1, 2), group_size, bits)
    self.freeze()

  def one(self, x, k): return mx.quantized_matmul(x, self.weight[k], self.scales[k], self.biases[k], transpose=True, group_size=self.group_size, bits=self.bits)


# ── temporal transformer: one position per 80 ms frame, causal over time ──
class TemporalBlock(nn.Module):
  def __init__(self, c: Config):
    super().__init__()
    d, hid = c.dim, _hidden(c.dim, c.hidden_scale)
    self.h, self.hd, self.base = c.num_heads, d // c.num_heads, c.max_period
    self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
    self.in_proj, self.out_proj = Linear(d, 3 * d), Linear(d, d)   # in_proj rows [q | k | v]
    self.lin_in, self.lin_out = Linear(d, 2 * hid), Linear(hid, d)

  # -> q, k, v [B, H, T, hd] with rotary embedding on interleaved pairs (2i, 2i+1) starting at frame `offset`
  def qkv(self, x, offset):
    B, T, _ = x.shape
    q, k, v = self.in_proj(norm(x, self.norm1.weight)).reshape(B, T, 3, self.h, self.hd).transpose(2, 0, 3, 1, 4)
    rope = partial(mx.fast.rope, dims=self.hd, traditional=True, base=self.base, scale=1.0, offset=offset)
    return rope(q), rope(k), v

  def finish(self, x, o):
    B, H, T, hd = o.shape
    x = x + self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, T, H * hd))
    return x + self.lin_out(swiglu(self.lin_in(norm(x, self.norm2.weight))))

  def __call__(self, x):
    q, k, v = self.qkv(x, 0)
    return self.finish(x, mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask="causal"))

  # one frame at position pos: x [B, 1, D]; kc/vc [B, H, Tmax, hd] preallocated, written in place, read up to pos
  def step(self, x, kc, vc, pos):
    q, k, v = self.qkv(x, pos)
    kc[:, :, pos:pos + 1], vc[:, :, pos:pos + 1] = k, v
    return self.finish(x, mx.fast.scaled_dot_product_attention(q, kc[:, :, :pos + 1], vc[:, :, :pos + 1], scale=self.hd ** -0.5))


# ── depth transformer: within one frame, predicts codebook k from the frame's hidden state and codebooks 0..k-1 ──
# It is causal over the 8 codebooks and has separate weights for each codebook position (MultiLinear).
class DepBlock(nn.Module):
  def __init__(self, c: Config):
    super().__init__()
    K, d, hid = c.dep_q, c.dep_dim, _hidden(c.dep_dim, c.dep_hidden_scale)
    self.h, self.hd = c.dep_heads, d // c.dep_heads
    self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
    self.in_proj, self.out_proj = MultiLinear(K, d, 3 * d), MultiLinear(K, d, d)
    self.lin_in, self.lin_out = MultiLinear(K, d, 2 * hid), MultiLinear(K, hid, d)

  # x [Kq, N, d]: all frames of the batch at once, the 8 codebooks as the sequence
  def __call__(self, x):
    K, N, d = x.shape
    q, k, v = self.in_proj(norm(x, self.norm1.weight)).reshape(K, N, 3, self.h, self.hd).transpose(2, 1, 3, 0, 4)   # [N, H, Kq, hd]
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask="causal")
    x = x + self.out_proj(o.transpose(2, 0, 1, 3).reshape(K, N, d))
    return x + self.lin_out(swiglu(self.lin_in(norm(x, self.norm2.weight))))

  # x [B, d] for codebook k (a Python int); ks/vs hold the [B, H, 1, hd] keys/values of codebooks 0..k-1 and are appended to
  def step(self, x, k, ks, vs):
    B, d = x.shape
    q, k_, v = self.in_proj.one(norm(x, self.norm1.weight), k).reshape(B, 3, self.h, 1, self.hd).transpose(1, 0, 2, 3, 4)
    ks.append(k_); vs.append(v)
    o = mx.fast.scaled_dot_product_attention(q, mx.concatenate(ks, 2), mx.concatenate(vs, 2), scale=self.hd ** -0.5)
    x = x + self.out_proj.one(o.reshape(B, d), k)
    return x + self.lin_out.one(swiglu(self.lin_in.one(norm(x, self.norm2.weight), k)), k)


class Caramel(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg, C, Kq = cfg, cfg.card + 1, cfg.dep_q   # token id == card is the "start / delay filler" token
    # all 16 input codebooks share one flat embedding table (row = k * C + token): one gather, then a sum over codebooks
    self.emb = nn.Embedding(cfg.n_q * C, cfg.dim); self.emb.weight = _trunc_normal(self.emb.weight.shape, cfg.dim)
    self.layers = [TemporalBlock(cfg) for _ in range(cfg.num_layers)]
    self.out_norm = RMSNorm(cfg.dim)
    self.dep_in = MultiLinear(Kq, cfg.dim, cfg.dep_dim)
    # embeddings of codebooks 0..6, fed into depth steps 1..7
    self.dep_emb = nn.Embedding((Kq - 1) * C, cfg.dep_dim); self.dep_emb.weight = _trunc_normal(self.dep_emb.weight.shape, cfg.dep_dim)
    self.dep_layers = [DepBlock(cfg) for _ in range(cfg.dep_layers)]
    self.heads = MultiLinear(Kq, cfg.dep_dim, cfg.card)
    self.heads.weight = self.heads.weight * 0.1   # small output heads: a random model starts at loss ~ln(2048) instead of ~10

  # tokens [..., K] with codebook k looked up in table rows k*C + token; -1 = no input (masked stream) contributes zero
  @staticmethod
  def _lookup(table, tokens):
    off = mx.arange(tokens.shape[-1]) * (table.weight.shape[0] // tokens.shape[-1])
    return table(mx.maximum(tokens, 0) + off) * (tokens >= 0)[..., None]

  def embed(self, tokens): return self._lookup(self.emb, tokens).sum(-2)   # [B, T, n_q] -> [B, T, dim]

  def temporal(self, x):
    for layer in self.layers: x = layer(x)
    return norm(x, self.out_norm.weight)

  # h [N, dim]; prev [N, dep_q-1] true tokens of codebooks 0..6 (teacher forcing). Returns [Kq, N, dep_dim]
  def depformer(self, h, prev):
    x = self.dep_in.fan_out(h)
    te = self._lookup(self.dep_emb, prev).transpose(1, 0, 2).astype(x.dtype)
    x = x + mx.pad(te, [(1, 0), (0, 0), (0, 0)])   # step 0 gets no token
    for layer in self.dep_layers: x = layer(x)
    return x

  # seq [B, T+1, n_q] delayed codes with seq[:, 0] = start token. Returns (loss, per-codebook loss)
  def __call__(self, seq):
    c = self.cfg
    inp, tgt = seq[:, :-1], seq[:, 1:]
    B, T, _ = inp.shape
    h = self.temporal(self.embed(inp)).reshape(B * T, c.dim)
    tgt = tgt[..., :c.dep_q].reshape(B * T, c.dep_q)
    z = self.depformer(h, tgt[:, :-1])
    tgt = tgt.T                                         # [Kq, N]
    # each codebook's loss is averaged over its own real targets (masked -1 and delay filler excluded), then the 8 are averaged equally
    m = (tgt >= 0) & (tgt < c.card)
    ce = nn.losses.cross_entropy(self.heads(z).astype(mx.float32), mx.clip(tgt, 0, c.card - 1))
    per_cb = (ce * m).sum(1) / mx.maximum(m.sum(1), 1)
    return per_cb.mean(), per_cb

  # ── streaming inference ──
  # temporal KV caches are preallocated for Tmax frames; the 8-codebook depth pass of a frame is one compiled graph
  def init_state(self, B, Tmax, temp=1.1, top_k=50):
    c, dt = self.cfg, self.out_norm.weight.dtype
    z = lambda: mx.zeros((B, c.num_heads, Tmax, c.dim // c.num_heads), dt)
    return {"kc": [z() for _ in self.layers], "vc": [z() for _ in self.layers], "pos": 0,
            "dep": mx.compile(partial(self.dep_frame, temp=temp, top_k=top_k))}

  # h [B, dim] -> the frame's delayed tokens [B, dep_q]; forced [B, dep_q] (>= 0 forces that token, -2 samples)
  def dep_frame(self, h, forced, key, temp, top_k):
    c, out, prev = self.cfg, [], None
    keys, ks, vs = mx.random.split(key, c.dep_q), [[] for _ in self.dep_layers], [[] for _ in self.dep_layers]
    for k in range(c.dep_q):
      z = self.dep_in.one(h, k)
      if prev is not None: z = z + self.dep_emb(prev + (k - 1) * (c.card + 1)).astype(z.dtype)
      for j, layer in enumerate(self.dep_layers): z = layer.step(z, k, ks[j], vs[j])
      tok = sample_topk(self.heads.one(z, k).astype(mx.float32), temp, top_k, keys[k])
      prev = mx.where(forced[:, k] >= 0, forced[:, k], tok)
      out.append(prev)
    return mx.stack(out, 1)

  # One 80 ms frame. inp [B, n_q] delayed tokens of the previous frame (card = start, -1 = none). Lazy: nothing is evaluated here.
  def step(self, st, inp, forced, key):
    x, pos = self.embed(inp[:, None]).astype(BF16), st["pos"]
    for i, layer in enumerate(self.layers): x = layer.step(x, st["kc"][i], st["vc"][i], pos)
    st["pos"] = pos + 1
    return st["dep"](norm(x, self.out_norm.weight)[:, 0], forced, key)


# keep the top_k logits, divide by temp, sample; temp <= 0 is greedy
def sample_topk(logits, temp, top_k, key):
  if temp <= 0: return logits.argmax(-1).astype(mx.int32)
  idx = mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]
  pick = mx.random.categorical(mx.take_along_axis(logits, idx, -1) / temp, key=key)
  return mx.take_along_axis(idx, pick[..., None], -1)[..., 0].astype(mx.int32)


# ── weights: a folder with config.json and model.safetensors (same names and layout as the original PyTorch release) ──
def load(path, dtype=BF16, bits=0):
  cfg = Config.from_dict(json.load(open(os.path.join(path, "config.json"))))
  m = Caramel(cfg)
  m.load_weights(os.path.join(path, "model.safetensors"), strict=True)
  m.set_dtype(dtype)
  # weight-only quantization of every matmul except the embedding tables; generation reads all weights every frame
  if bits: nn.quantize(m, group_size=32, bits=bits, class_predicate=lambda p, mod: isinstance(mod, (Linear, MultiLinear)) and mod.weight.shape[1] % 32 == 0)
  mx.eval(m.parameters())
  m.eval()
  return m


def save(model, path, dtype=BF16):
  os.makedirs(path, exist_ok=True)
  json.dump(asdict(model.cfg), open(os.path.join(path, "config.json"), "w"), indent=1)
  mx.save_safetensors(os.path.join(path, "model.safetensors"), {k: v.astype(dtype) for k, v in nn.utils.tree_flatten(model.parameters())})


# a release of the original PyTorch model ships model.safetensors as .partNNN pieces: join them once
def join_parts(path):
  import glob, hashlib, re, shutil
  f = os.path.join(path, "model.safetensors")
  if os.path.exists(f): return
  parts = sorted(glob.glob(f + ".part*"), key=lambda p: int(re.sub(r"\D", "", p.rsplit(".part", 1)[1]) or 0))
  assert parts, f"no model.safetensors or model.safetensors.part* in {path}"
  with open(f + ".tmp", "wb") as o:
    for p in parts:
      with open(p, "rb") as i: shutil.copyfileobj(i, o, 1 << 24)
  sums = os.path.join(path, "SHA256SUMS")
  want = dict(l.split()[::-1] for l in open(sums) if l.strip()).get("model.safetensors") if os.path.exists(sums) else None
  if want: assert hashlib.file_digest(open(f + ".tmp", "rb"), "sha256").hexdigest() == want, "joined model.safetensors fails SHA256SUMS"
  os.replace(f + ".tmp", f)


if __name__ == "__main__": join_parts(sys.argv[1])   # python model.py ./caramel-1
