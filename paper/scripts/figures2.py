"""Figures for the full technical report (lifecycle, shapes, delays, funnel, streaming, RL)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excali import Diagram
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")


def lifecycle():
  d = Diagram("fig10_lifecycle")
  d.text(20, 10, "caramel lifecycle: from scraped audio to a duplex assistant", 20, bold=True)
  top = [("scrape\n66,824 feeds\n3,216 CC channels", "gray"), ("closeness filter\n13 statistics\ntarget bands", "yellow"),
         ("speaker separation\none track\nper speaker", "orange"), ("Mimi tokens\nprepare.py\n11,555 h, 6.2B", "teal"),
         ("stage 1\nlearn to speak\n(mono)", "blue")]
  bot = [("stage 2\nlearn to converse\n5,743 h", "blue"), ("stage 3\nkeep going\nto 5.65 passes", "blue"),
         ("caramel-1\n413.2M params\nreleased", "green"), ("synthetic pretrain\nrandom voices\n(not trained)", "purple"),
         ("JOSIE SFT\n--no_swap\n(running)", "purple"), ("RL: GRPO / DPO\nrl.py\n(not trained)", "red")]
  w, h = 175, 95
  for i, (s, c) in enumerate(top):
    x = 20 + i * 215; d.box(x, 60, w, h, s, c, 14)
    if i: d.arrow([(x - 40, 107), (x, 107)])
  d.arrow([(20 + 4 * 215 + 87, 155), (20 + 4 * 215 + 87, 185), (107, 185), (107, 215)])
  for i, (s, c) in enumerate(bot):
    x = 20 + i * 215 * 5 / 6 * 1.0; x = 20 + i * 200; d.box(x, 215, 170, h, s, c, 14)
    if i: d.arrow([(x - 30, 262), (x, 262)])
  d.text(20, 330, "grey/yellow/orange/teal: data; blue: pre-training (README); purple/red: post-training (this report)", 14)
  d.save(OUT)


def shapes():
  d = Diagram("fig11_shapes")
  d.text(20, 10, "One training forward pass with tensor shapes (B windows, T frames, N = B*T)", 20, bold=True)
  rows = [("seq  [B, T+1, 16]  int, delayed, start token S = 2048", "yellow"),
          ("inp = seq[:, :-1]  ->  flat embedding gather E[k*2049 + tok] (zero where tok = -1)  ->  sum over 16 codebooks  [B, T, 1024]", "yellow"),
          ("temporal transformer x16: RMSNorm -> QKV [B, 16, T, 64] + RoPE -> causal SDPA -> SwiGLU (2816)  [B, T, 1024]", "blue"),
          ("out RMSNorm, reshape  h  [N, 1024]", "blue"),
          ("dep_in fan-out (8 per-codebook linears)  [8, N, 512]  +  dep_emb(tgt cb 0..6) shifted by one  [8, N, 512]", "purple"),
          ("depformer x6: per-codebook weights, causal 8x8 attention, 8 heads of 64, SwiGLU (1408)  [8, N, 512]", "purple"),
          ("heads: 8 linears 512 -> 2048  logits [8, N, 2048]  ->  masked CE per codebook  ->  mean of 8", "green")]
  for i, (s, c) in enumerate(rows):
    y = 55 + i * 75; d.box(20, y, 1150, 55, s, c, 15)
    if i: d.arrow([(595, y - 20), (595, y)])
  d.save(OUT)


def delays():
  d = Diagram("fig12_delays")
  d.text(20, 10, "Delay pattern: what is predicted at step t (speaker A shown; speaker B identical as input)", 20, bold=True)
  x0, y0, cw, ch = 190, 70, 95, 44
  for t in range(6):
    d.text(x0 + t * cw + 47, 45, f"step {t}", 15, "middle")
  for k, lab in enumerate(["c0 (delay 0)", "c1 (delay 1)", "c2..c7 (delay 1)"]):
    d.text(x0 - 12, y0 + k * ch + 12, lab, 15, "end")
    for t in range(6):
      src = t - (0 if k == 0 else 1)
      txt = "S" if (t == 0 or src < 1) else f"x{src}"
      col = "gray" if txt == "S" else ("blue" if k == 0 else "purple")
      d.box(x0 + t * cw, y0 + k * ch, cw - 6, ch - 6, txt, col, 15)
  d.text(x0, y0 + 3 * ch + 15, "Column t of seq is the input of step t and the target of step t-1. At step t the depformer first writes", 15)
  d.text(x0, y0 + 3 * ch + 40, "c0 of frame t, then c1..c7 of frame t-1: words of a frame are settled before its acoustics.", 15)
  d.text(x0, y0 + 3 * ch + 65, "Generation runs T+1 steps to emit the acoustics of the last frame (generate.py: S = T + 1).", 15)
  d.save(OUT)


def funnel():
  d = Diagram("fig13_funnel")
  d.text(20, 10, "Pre-training data funnel (README numbers)", 20, bold=True)
  st = [("candidate sources: 66,824 podcast feeds + 3,216 Creative Commons YouTube channels", 1100, "gray"),
        ("closeness-ranked; only close sources downloaded at scale (~5,400 h of audio per hour of downloading)", 960, "yellow"),
        ("speaker separation of mixed recordings (validated on already-separated tracks)", 820, "orange"),
        ("~6,500 h two-person conversation + ~5,800 h one-person speech", 680, "teal"),
        ("tokenised: 5,743 h conv. (4.1B tokens) + 5,812 h mono (2.1B) = 11,555 h, 6.2B", 540, "blue"),
        ("trained: ~33B tokens seen (multiple passes)", 400, "green")]
  for i, (s, w, c) in enumerate(st):
    y = 55 + i * 72; d.box(20 + (1100 - w) / 2, y, w, 56, s, c, 14)
  d.save(OUT)


def streaming():
  d = Diagram("fig14_streaming")
  d.text(20, 10, "Streaming inference: one 80 ms step with KV caches (live.py / generate.py)", 20, bold=True)
  d.box(20, 60, 200, 70, "mic block\n1920 samples", "orange", 15)
  d.box(260, 60, 200, 70, "Mimi encode_step\n-> 8 codes (user)", "teal", 15)
  d.box(500, 60, 230, 70, "input [me | other]\n16 delayed tokens", "yellow", 15)
  d.box(770, 60, 280, 70, "temporal step at pos\nwrite K,V into cache[pos]", "blue", 15)
  d.arrow([(220, 95), (260, 95)]); d.arrow([(460, 95), (500, 95)]); d.arrow([(730, 95), (770, 95)])
  d.box(770, 180, 280, 70, "compiled depth pass\n8 codebooks x 6 layers\n+ top-k sampling", "purple", 14)
  d.arrow([(910, 130), (910, 180)])
  d.box(500, 180, 230, 70, "Mimi decode_step\nframe t-1 -> 1920 samples", "teal", 14)
  d.arrow([(770, 215), (730, 215)])
  d.box(260, 180, 200, 70, "speaker\n(80 ms later)", "green", 15)
  d.arrow([(500, 215), (460, 215)])
  d.arrow([(1050, 215), (1100, 215), (1100, 300), (480, 300), (480, 120), (500, 120)], "sampled tokens become 'me' input of the next step (stay on GPU)", dashed=True, lab_off=(0, 24))
  d.box(20, 360, 1080, 90, "KV cache: 16 layers x [B, 16 heads, Tmax, 64] preallocated, Tmax <= 3000 frames (240 s); attention reads 0..pos.\n"
        "Budget: < 80 ms per step. Stale mic blocks are dropped rather than drifting. mx.async_eval overlaps frame t+1 with t.\n"
        "Weight-only 8-bit / 4-bit quantisation for bandwidth-bound decoding.", "gray", 14)
  d.save(OUT)


def rl():
  d = Diagram("fig15_rl")
  d.text(20, 10, "RL stage (rl.py, implemented, not yet trained)", 20, bold=True)
  d.box(20, 60, 230, 80, "real window (60 s)\nuser side forced", "orange", 15)
  d.box(300, 60, 250, 80, "group of G = 4 rollouts\nA real for 15 s, then sampled", "blue", 15)
  d.box(600, 60, 230, 80, "Mimi decode\nenergy VAD + log-mel", "teal", 15)
  d.box(880, 60, 260, 80, "events per 2 s chunk:\nyield, talkover,\nrespond, dead_air, voice", "yellow", 14)
  d.arrow([(250, 100), (300, 100)]); d.arrow([(550, 100), (600, 100)]); d.arrow([(830, 100), (880, 100)])
  d.box(880, 200, 260, 80, "group-normalised chunk\nreward, discounted\nsum (gamma 0.5)", "purple", 14)
  d.arrow([(1010, 140), (1010, 200)])
  d.box(560, 200, 270, 80, "GRPO: PPO-clip ratio (0.2)\n+ k3 KL to start (0.02)\n[or DPO-LN, beta 0.3]", "red", 14)
  d.arrow([(880, 240), (830, 240)])
  d.box(300, 200, 210, 80, "policy update\nAdamW, lr 5e-7", "green", 15)
  d.arrow([(560, 240), (510, 240)])
  d.arrow([(405, 200), (405, 140)], dashed=True)
  d.save(OUT)


for f in (lifecycle, shapes, delays, funnel, streaming, rl): f()
print("ok")
