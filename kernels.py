import math
import torch, torch.nn.functional as F
try:
  import triton, triton.language as tl
  TRITON = torch.cuda.is_available()
except ImportError: TRITON = False

# ── pure-PyTorch reference: used on CPU / Mac / without Triton, and the math every kernel below must match ──
def rms_norm_ref(x, w, eps=1e-5): return F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps).to(x.dtype)

def dep_attn_ref(qkv, heads):
  Kq, N, D3 = qkv.shape
  q, k, v = qkv.view(Kq, N, 3, heads, -1).permute(2, 1, 3, 0, 4).float().unbind(0)   # [N, H, Kq, hd]
  s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
  s = s.masked_fill(~torch.ones(Kq, Kq, dtype=torch.bool, device=qkv.device).tril(), float("-inf"))
  return (s.softmax(-1) @ v).to(qkv.dtype).permute(2, 0, 1, 3).reshape(Kq, N, D3 // 3)

if TRITON:
  # ── depth-transformer attention: causal attention over the 8 codebooks of each frame ──
  # The sequence is only 8 long, so flash-attention kernels waste most of their tiles. Here 2 frames of one head are
  # stacked into 16 rows (the smallest tensor-core tile) and a block-diagonal causal mask keeps the frames apart.
  # q, k, v are read straight from the packed [Kq, N, 3*d] projection, so no permute or copy is needed. 6.70 -> 1.17 ms/layer.
  @triton.jit
  def _dep_attn_fwd(QKV, O, s_qk, s_qn, s_ok, s_on, N, H, scale, D: tl.constexpr, HD: tl.constexpr, KQ: tl.constexpr, FPB: tl.constexpr):
    R: tl.constexpr = FPB * KQ
    pid = tl.program_id(0).to(tl.int64)
    nb, h, r, di = pid // H, pid % H, tl.arange(0, R), tl.arange(0, HD)
    n, kk = nb * FPB + r // KQ, r % KQ
    rm = (n < N)[:, None]
    offs = (kk * s_qk + n * s_qn)[:, None] + h * HD + di[None, :]
    q, k, v = tl.load(QKV + offs, mask=rm, other=0.0), tl.load(QKV + D + offs, mask=rm, other=0.0), tl.load(QKV + 2 * D + offs, mask=rm, other=0.0)
    s = tl.dot(q, tl.trans(k)) * scale
    s = tl.where(((r[:, None] // KQ) == (r[None, :] // KQ)) & (kk[:, None] >= kk[None, :]), s, float("-inf"))
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    o = tl.dot(p, v.to(tl.float32), input_precision="ieee")
    tl.store(O + (kk * s_ok + n * s_on)[:, None] + h * HD + di[None, :], o.to(O.dtype.element_ty), mask=rm)

  # backward recomputes the 8x8 probabilities from q and k (nothing saved but the input) and writes dq|dk|dv packed like the input
  @triton.jit
  def _dep_attn_bwd(QKV, DO, DQKV, s_qk, s_qn, s_ok, s_on, N, H, scale, D: tl.constexpr, HD: tl.constexpr, KQ: tl.constexpr, FPB: tl.constexpr):
    R: tl.constexpr = FPB * KQ
    pid = tl.program_id(0).to(tl.int64)
    nb, h, r, di = pid // H, pid % H, tl.arange(0, R), tl.arange(0, HD)
    n, kk = nb * FPB + r // KQ, r % KQ
    rm = (n < N)[:, None]
    offs = (kk * s_qk + n * s_qn)[:, None] + h * HD + di[None, :]
    q, k, v = tl.load(QKV + offs, mask=rm, other=0.0), tl.load(QKV + D + offs, mask=rm, other=0.0), tl.load(QKV + 2 * D + offs, mask=rm, other=0.0)
    do = tl.load(DO + (kk * s_ok + n * s_on)[:, None] + h * HD + di[None, :], mask=rm, other=0.0)
    s = tl.dot(q, tl.trans(k)) * scale
    s = tl.where(((r[:, None] // KQ) == (r[None, :] // KQ)) & (kk[:, None] >= kk[None, :]), s, float("-inf"))
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    dv = tl.dot(tl.trans(p), do.to(tl.float32), input_precision="ieee")
    dp = tl.dot(do, tl.trans(v))   # bf16 x bf16 -> f32, exact products
    ds = p * (dp - tl.sum(p * dp, axis=1)[:, None]) * scale
    dq, dk = tl.dot(ds, k.to(tl.float32), input_precision="ieee"), tl.dot(tl.trans(ds), q.to(tl.float32), input_precision="ieee")
    ty = DQKV.dtype.element_ty
    tl.store(DQKV + offs, dq.to(ty), mask=rm); tl.store(DQKV + D + offs, dk.to(ty), mask=rm); tl.store(DQKV + 2 * D + offs, dv.to(ty), mask=rm)

  def _dep_launch(kernel, qkv, o, heads, *extra):
    Kq, N, D3 = qkv.shape
    hd = D3 // 3 // heads
    kernel[(triton.cdiv(N, 2) * heads,)](qkv, o, *extra, qkv.stride(0), qkv.stride(1), o.stride(0), o.stride(1), N, heads, 1.0 / math.sqrt(hd),
                                         D=D3 // 3, HD=hd, KQ=Kq, FPB=2, num_warps=1)

  # wrapped as torch custom ops so torch.compile treats each kernel as one opaque node
  @torch.library.custom_op("caramel::dep_attn", mutates_args=())
  def _dep_attn(qkv: torch.Tensor, heads: int) -> torch.Tensor:
    qkv = qkv.contiguous(); o = qkv.new_empty(qkv.shape[0], qkv.shape[1], qkv.shape[2] // 3)
    _dep_launch(_dep_attn_fwd, qkv, o, heads); return o
  _dep_attn.register_fake(lambda qkv, heads: qkv.new_empty(qkv.shape[0], qkv.shape[1], qkv.shape[2] // 3))

  @torch.library.custom_op("caramel::dep_attn_bwd", mutates_args=())
  def _dep_attn_bwd_op(qkv: torch.Tensor, do: torch.Tensor, heads: int) -> torch.Tensor:
    qkv, do = qkv.contiguous(), do.contiguous(); g = torch.empty_like(qkv)
    _dep_launch(_dep_attn_bwd, qkv, do, heads, g); return g
  _dep_attn_bwd_op.register_fake(lambda qkv, do, heads: qkv.new_empty(qkv.shape))

  def _dep_setup(ctx, inputs, output): ctx.save_for_backward(inputs[0]); ctx.heads = inputs[1]
  _dep_attn.register_autograd(lambda ctx, go: (_dep_attn_bwd_op(ctx.saved_tensors[0], go, ctx.heads), None), setup_context=_dep_setup)

  # ── RMSNorm, optionally fused with the residual add before it (one read of x and r instead of three passes) ──
  # statistics in float32, the residual stream itself is stored back at its own dtype
  @triton.jit
  def _rms_fwd(X, R, W, Y, S, RSTD, D, eps, HAS_R: tl.constexpr, BLOCK: tl.constexpr):
    row, c = tl.program_id(0).to(tl.int64), tl.arange(0, BLOCK)
    m = c < D
    x = tl.load(X + row * D + c, mask=m, other=0.0).to(tl.float32)
    if HAS_R:
      x += tl.load(R + row * D + c, mask=m, other=0.0).to(tl.float32)
      x = x.to(S.dtype.element_ty)
      tl.store(S + row * D + c, x, mask=m)
      x = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    tl.store(RSTD + row, rstd)
    w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * D + c, (x * rstd * w).to(Y.dtype.element_ty), mask=m)

  # dx = rstd * (dy*w - xhat * mean(dy*w*xhat)) (+ ds, the gradient into the residual output);
  # each program owns a run of rows and writes one float32 partial row of dw, summed afterwards
  @triton.jit
  def _rms_bwd(DY, DS, X, W, RSTD, DX, DWP, M, D, rows_per_prog, HAS_DS: tl.constexpr, BLOCK: tl.constexpr):
    pid, c = tl.program_id(0), tl.arange(0, BLOCK)
    m = c < D
    w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK], dtype=tl.float32)
    r0 = pid * rows_per_prog
    for r in range(r0, tl.minimum(r0 + rows_per_prog, M)):
      row = r.to(tl.int64)
      x, dy = tl.load(X + row * D + c, mask=m, other=0.0).to(tl.float32), tl.load(DY + row * D + c, mask=m, other=0.0).to(tl.float32)
      rstd = tl.load(RSTD + row)
      xh, g = x * rstd, dy * w
      dw += dy * xh
      dx = rstd * (g - xh * (tl.sum(g * xh, axis=0) / D))
      if HAS_DS: dx += tl.load(DS + row * D + c, mask=m, other=0.0).to(tl.float32)
      tl.store(DX + row * D + c, dx.to(DX.dtype.element_ty), mask=m)
    tl.store(DWP + pid * D + c, dw, mask=m)

  def _rms_launch_fwd(x, r, w, eps, y_bf16):
    D = x.shape[-1]; x2 = x.reshape(-1, D); M = x2.shape[0]
    y = torch.empty_like(x2, dtype=torch.bfloat16 if y_bf16 else x2.dtype)
    s = torch.empty_like(x2, dtype=torch.promote_types(x.dtype, r.dtype)) if r is not None else x2
    rstd, BLOCK = torch.empty(M, device=x.device, dtype=torch.float32), triton.next_power_of_2(D)
    _rms_fwd[(M,)](x2, r.reshape(-1, D) if r is not None else x2, w, y, s, rstd, D, eps, HAS_R=r is not None, BLOCK=BLOCK, num_warps=4 if BLOCK <= 1024 else 8)
    return y.view(x.shape), s.view(x.shape), rstd

  def _rms_launch_bwd(dy, ds, s, w, rstd):
    D = s.shape[-1]; s2 = s.reshape(-1, D); M = s2.shape[0]; BLOCK = triton.next_power_of_2(D)
    rpp = triton.cdiv(M, min(M, 132 * 8))   # ~8 programs per H100 SM
    nprog = triton.cdiv(M, rpp)
    dx, dwp = torch.empty_like(s2), torch.empty(nprog, D, device=s.device, dtype=torch.float32)
    _rms_bwd[(nprog,)](dy.reshape(-1, D), ds.reshape(-1, D) if ds is not None else dy, s2, w, rstd, dx, dwp, M, D, rpp,
                       HAS_DS=ds is not None, BLOCK=BLOCK, num_warps=4 if BLOCK <= 1024 else 8)
    return dx.view_as(s), dwp.sum(0).to(w.dtype)

  @torch.library.custom_op("caramel::add_rms_norm", mutates_args=())
  def _add_rms_norm(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float, y_bf16: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y, s, rstd = _rms_launch_fwd(x.contiguous(), r.contiguous(), w, eps, y_bf16); return s, y, rstd
  _add_rms_norm.register_fake(lambda x, r, w, eps, y_bf16: (x.new_empty(x.shape, dtype=torch.promote_types(x.dtype, r.dtype)),
    x.new_empty(x.shape, dtype=torch.bfloat16 if y_bf16 else x.dtype), x.new_empty(x.numel() // x.shape[-1], dtype=torch.float32)))

  @torch.library.custom_op("caramel::rms_norm_fwd", mutates_args=())
  def _rms_norm_fwd(x: torch.Tensor, w: torch.Tensor, eps: float, y_bf16: bool) -> tuple[torch.Tensor, torch.Tensor]:
    y, _, rstd = _rms_launch_fwd(x.contiguous(), None, w, eps, y_bf16); return y, rstd
  _rms_norm_fwd.register_fake(lambda x, w, eps, y_bf16: (x.new_empty(x.shape, dtype=torch.bfloat16 if y_bf16 else x.dtype),
    x.new_empty(x.numel() // x.shape[-1], dtype=torch.float32)))

  @torch.library.custom_op("caramel::rms_norm_bwd", mutates_args=())
  def _rms_norm_bwd(dy: torch.Tensor, ds: torch.Tensor, s: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor, has_ds: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _rms_launch_bwd(dy.contiguous(), ds.contiguous() if has_ds else None, s.contiguous(), w, rstd)
  _rms_norm_bwd.register_fake(lambda dy, ds, s, w, rstd, has_ds: (s.new_empty(s.shape), w.new_empty(w.shape)))

  def _arn_setup(ctx, inputs, output): ctx.save_for_backward(output[0], inputs[2], output[2])
  def _arn_backward(ctx, gs, gy, grstd):
    s, w, rstd = ctx.saved_tensors
    dx, dw = _rms_norm_bwd(gy, gs if gs is not None else gy, s, w, rstd, gs is not None)
    return dx, dx, dw, None, None
  _add_rms_norm.register_autograd(_arn_backward, setup_context=_arn_setup)

  def _rn_setup(ctx, inputs, output): ctx.save_for_backward(inputs[0], inputs[1], output[1])
  def _rn_backward(ctx, gy, grstd):
    x, w, rstd = ctx.saved_tensors
    dx, dw = _rms_norm_bwd(gy, gy, x, w, rstd, False); return dx, dw, None, None
  _rms_norm_fwd.register_autograd(_rn_backward, setup_context=_rn_setup)

# ── public API: Triton on CUDA, the reference everywhere else ──
def _gpu(x): return TRITON and x.is_cuda

def dep_attn(qkv, heads): return _dep_attn(qkv, heads) if _gpu(qkv) and qkv.dtype != torch.float32 else dep_attn_ref(qkv, heads)

# the normed output feeds only a matmul, so under autocast it is written as bf16 directly
def rms_norm(x, w, eps=1e-5): return _rms_norm_fwd(x, w, eps, torch.is_autocast_enabled())[0] if _gpu(x) else rms_norm_ref(x, w, eps)

def add_rms_norm(x, r, w, eps=1e-5):
  if _gpu(x): return _add_rms_norm(x, r, w, eps, torch.is_autocast_enabled())[:2]
  s = x + r
  return s, rms_norm_ref(s, w, eps)
