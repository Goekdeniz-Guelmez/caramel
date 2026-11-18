"""Builds the nine Excalidraw figures of the paper (paper/figures/*.excalidraw, .svg, .pdf, .png)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excali import Diagram

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def overview():
  d = Diagram("fig1_overview")
  d.text(20, 10, "caramel: audio in, audio out, every 80 ms", 22, bold=True)
  d.box(20, 70, 170, 60, "user audio\n(24 kHz)", "orange")
  d.box(20, 200, 170, 60, "own audio\n(fed back)", "green")
  d.box(250, 130, 150, 70, "Mimi encoder\n8 codebooks", "teal")
  d.arrow([(190, 100), (250, 150)]); d.arrow([(190, 230), (250, 180)])
  d.box(460, 120, 170, 90, "16 tokens / frame\nA: own (0-7)\nB: other (8-15)", "yellow", 15)
  d.arrow([(400, 165), (460, 165)])
  d.box(690, 60, 220, 80, "temporal transformer\n16 layers, d = 1024", "blue")
  d.box(690, 190, 220, 80, "depformer\n6 layers, d = 512", "purple")
  d.arrow([(630, 150), (690, 100)]); d.arrow([(800, 140), (800, 190)], "h_t", lab_off=(22, 4))
  d.box(970, 190, 150, 80, "8 new tokens\nfor speaker A", "yellow")
  d.arrow([(910, 230), (970, 230)])
  d.box(970, 60, 150, 80, "Mimi decoder", "teal")
  d.arrow([(1045, 190), (1045, 140)])
  d.box(1170, 60, 150, 80, "speech out", "green")
  d.arrow([(1120, 100), (1170, 100)])
  d.arrow([(1045, 270), (1045, 310), (105, 310), (105, 260)], "sampled tokens become next frame's input (no VAD, no turn rule)", dashed=True, lab_off=(0, 26))
  d.save(OUT)


def tokens():
  d = Diagram("fig2_tokens")
  d.text(20, 10, "Mimi token layout of one window (12.5 frames/s, 2048 entries per codebook)", 20, bold=True)
  x0, y0, cw, ch = 150, 90, 44, 34
  d.text(x0 + 4 * cw, 55, "speaker A (modelled): columns 0-7", 16, "middle")
  d.text(x0 + 12 * cw + 20, 55, "speaker B (other): columns 8-15", 16, "middle")
  T = 6
  for t in range(T):
    d.text(x0 - 15, y0 + t * ch + 8, f"frame t={t}" if t else "t=0 (start)", 14, "end")
    for k in range(16):
      gap = 20 if k >= 8 else 0
      cb = k % 8; delay = 0 if cb == 0 else 1
      src = t - delay
      if t == 0: lab, col = "S", "gray"
      elif src < 1: lab, col = "S", "gray"
      else: lab, col = f"{src}", ("blue" if k < 8 else "orange") if cb == 0 else ("purple" if k < 8 else "yellow")
      d.box(x0 + k * cw + gap, y0 + t * ch, cw - 4, ch - 4, lab, col, 14)
  for k in range(16):
    gap = 20 if k >= 8 else 0
    d.text(x0 + k * cw + gap + 20, y0 + T * ch + 4, f"c{k%8}", 14, "middle")
  yb = y0 + T * ch + 40
  d.box(150, yb, 200, 50, "c0: semantic, delay 0", "blue", 15)
  d.box(370, yb, 230, 50, "c1-c7: acoustic, delay 1", "purple", 15)
  d.box(620, yb, 190, 50, "S: start / filler", "gray", 15)
  d.text(150, yb + 66, "Cell value = index of the source frame. Loss is taken on columns 0-7 only; columns 8-15 are input.", 15)
  d.text(150, yb + 90, "1 h of one speaker = 12.5 x 8 x 3600 = 360,000 tokens.", 15)
  d.save(OUT)


def arch():
  d = Diagram("fig3_arch")
  d.text(20, 10, "Two-level factorisation: time (temporal transformer) x depth (depformer)", 20, bold=True)
  for t in range(4):
    x = 40 + t * 150
    d.box(x, 400, 130, 50, f"embed sum\n16 tokens, t={t}", "yellow", 14)
    d.arrow([(x + 65, 400), (x + 65, 360)])
  d.box(30, 230, 610, 130, "temporal transformer (causal over frames)\n16 layers, 16 heads, d = 1024, RoPE, SwiGLU, RMSNorm\ncontext 3000 frames = 240 s", "blue", 16)
  d.arrow([(565, 230), (565, 190), (720, 190)], "h_t", lab_off=(0, -6))
  d.text(30, 470, "flat embedding table: row = k(C)+token, C = 2049 (2048 + start)", 14)
  d.box(720, 60, 560, 340, "", "none", dashed=True)
  d.text(735, 70, "depformer (one frame, causal over 8 codebooks)", 16, bold=True)
  for k in range(4):
    x = 740 + k * 135
    lab = f"c{k}" if k < 3 else "... c7"
    d.box(x, 300, 115, 50, f"in_proj_k(h_t)\n+ emb(c{k-1})" if k else "in_proj_0(h_t)", "purple", 13)
    d.box(x, 200, 115, 60, "6 layers\nper-k weights", "purple", 14)
    d.box(x, 110, 115, 50, f"head_k -> {lab}", "green", 14)
    d.arrow([(x + 57, 300), (x + 57, 260)]); d.arrow([(x + 57, 200), (x + 57, 160)])
    if k < 3: d.arrow([(x + 100, 160), (x + 145, 300)], dashed=True)
  d.text(740, 360, "6 layers, 8 heads, d = 512; teacher forcing in training", 14)
  d.text(720, 420, "413.2 M parameters in total; 8 tokens of speaker A per 80 ms frame", 15)
  d.save(OUT)


def selfplay():
  d = Diagram("fig4_selfplay")
  d.text(40, 490, "Live inference and self-play: same weights, two rows of a batch", 20, bold=True)
  d.box(40, 70, 250, 80, "row 0: model as A\ninput [A0 | B0]", "blue")
  d.box(40, 230, 250, 80, "row 1: model as B\ninput [B1 | A1] (swapped)", "orange")
  d.box(420, 140, 220, 90, "caramel step\n(one 80 ms frame)", "purple")
  d.arrow([(290, 110), (420, 170)]); d.arrow([(290, 270), (420, 200)])
  d.box(760, 70, 200, 80, "8 tokens for A", "blue"); d.box(760, 230, 200, 80, "8 tokens for B", "orange")
  d.arrow([(640, 170), (760, 110)]); d.arrow([(640, 200), (760, 270)])
  d.arrow([(960, 110), (1010, 110), (1010, 340), (40, 340), (40, 310)], "A's tokens are row 1's 'other' input", dashed=True, lab_off=(0, 20))
  d.arrow([(960, 270), (1040, 270), (1040, 30), (165, 30), (165, 70)], "B's tokens are row 0's 'other' input", dashed=True, lab_off=(0, 0))
  d.box(40, 400, 920, 70, "live mode: row 1 is replaced by the microphone (Mimi-encoded each 80 ms); forced[...] >= 0 pins tokens\nreal time = 80 ms/frame; ~11x real time on GPU, ~4x for both copies on an M3 Pro (README)", "yellow", 15)
  d.save(OUT)


def pipeline():
  d = Diagram("fig5_pipeline")
  d.text(20, 10, "Synthetic full-duplex data pipeline (one chunk of conversations at a time)", 20, bold=True)
  w, h, y = 170, 90, 70
  steps = [("seed prompt\nUltraFeedback", "gray"), ("teacher LLM\ndeepseek-v4.1-flash\n(bonsai-2:27b)", "orange"),
           ("Phonon TTS\nJOSIE = Zoey\nuser: 1 of 25", "green"), ("forced aligner\nword times\nmatch >= 0.8", "red"),
           ("scene builder\nbarge-in, holds,\nbackchannels", "purple"), ("Mimi encoder\n2 x 8 tokens\n12.5 fps", "teal"),
           ("HF upload\nparquet shard\n(audio deleted)", "blue")]
  for i, (s, c) in enumerate(steps):
    x = 20 + i * (w + 30)
    d.box(x, y, w, h, s, c, 15)
    if i: d.arrow([(x - 30, y + h / 2), (x, y + h / 2)])
  d.arrow([(20 + 3 * 200 + 85, 160), (20 + 3 * 200 + 85, 230)], "match < 0.8", dashed=True, lab_off=(55, 20))
  d.box(560, 230, 260, 60, "dropped (~25% of conversations;\nreason in meta.json[skipped])", "red", 14)
  d.box(20, 230, 500, 90, "turn loop: JOSIE replies (system prompt: spoken, no markdown);\nfollow-up model writes the next user turn;\nmax turns 32 / 4 / 8 by prompt range", "yellow", 14)
  d.box(860, 230, 520, 90, "train.py --data hf://... streams shards;\n8 workers: ~9-10 h of audio per wall-clock hour\nJOSIE set: 1,431 conv., 137.1 h train + 14.2 h val.", "blue", 14)
  d.save(OUT)


def scenes():
  d = Diagram("fig6_scenes")
  d.text(20, 10, "Scene timelines (top track: user B, bottom track: JOSIE A); x = time", 20, bold=True)
  rows = [("barge-in  p = 0.25", [(0, 120, "B", "question"), (140, 420, "A", "answer ..."), (330, 470, "B", "interrupts"),
                                   (490, 640, "A", "yields, replies")], "A stops <= 750 ms after onset"),
          ("wait  p = 0.10", [(0, 200, "B", "can you wait a minute?"), (220, 300, "A", "sure"), (560, 700, "A", "can I continue?")],
           "silence up to 60 s (both quiet)"),
          ("listen  p = 0.10", [(0, 190, "B", "listen until I finish"), (200, 260, "A", "ok"), (270, 560, "B", "long talk ..."),
                                 (580, 720, "A", "feedback")], "A holds silent while B talks"),
          ("backchannel  p = 0.2", [(0, 520, "B", "long user turn ..."), (150, 200, "A", "mhm"), (330, 390, "A", "right"),
                                     (540, 720, "A", "reply")], "in pauses of long user turns")]
  x0 = 200
  for r, (name, segs, note) in enumerate(rows):
    y = 60 + r * 130
    d.text(20, y + 30, name, 16, bold=True)
    d.line([(x0, y + 25), (x0 + 760, y + 25)], ink="#868e96", width=1); d.line([(x0, y + 75), (x0 + 760, y + 75)], ink="#868e96", width=1)
    for s, e, who, lab in segs:
      yy = y + 5 if who == "B" else y + 55
      d.box(x0 + s, yy, e - s, 40, lab, "orange" if who == "B" else "blue", 14)
    d.text(x0 + 770, y + 40, note, 14)
  d.text(x0, 590, "B = user (orange), A = JOSIE (blue). wait, listen: once per conversation; barge-in: per user turn; backchannel: per long user turn.", 14)
  d.save(OUT)


def objective():
  d = Diagram("fig7_objective")
  d.text(20, 10, "Training objective: speaker swap (default) vs --no_swap", 20, bold=True)
  d.box(20, 60, 560, 330, "", "none", dashed=True); d.text(35, 70, "default: random order per window, p = 1/2", 16, bold=True)
  d.box(40, 110, 240, 60, "[ A | B ]", "blue"); d.box(320, 110, 240, 60, "[ B | A ]", "orange")
  d.box(40, 200, 240, 60, "loss on A (JOSIE)", "green", 15); d.box(320, 200, 240, 60, "loss on B (user)", "red", 15)
  d.arrow([(160, 170), (160, 200)]); d.arrow([(440, 170), (440, 200)])
  d.box(40, 290, 520, 80, "model learns to produce both sides:\nuser voices (25, many male) and user barge-ins", "red", 15)
  d.box(620, 60, 560, 330, "", "none", dashed=True); d.text(635, 70, "--no_swap: JOSIE always speaker A", 16, bold=True)
  d.box(640, 110, 240, 60, "[ A | B ]", "blue"); d.box(920, 110, 240, 60, "B: input only", "gray")
  d.box(640, 200, 240, 60, "loss on A (JOSIE)", "green", 15); d.arrow([(760, 170), (760, 200)])
  d.box(640, 290, 520, 80, "one voice, one role; user side is context\n(dev loss scored on A only: not comparable)", "green", 15)
  d.save(OUT)


def recipe():
  d = Diagram("fig8_recipe")
  d.text(20, 10, "Training recipe: caramel-1 (released) -> synthetic pretrain -> JOSIE fine-tune", 20, bold=True)
  d.box(20, 70, 250, 140, "caramel-1 (README)\n1. one-person speech, 27 min\n2. 5,743 h conv., 46 min\n3. keep going, 61 min\n11,555 h total", "gray", 14)
  d.box(330, 70, 300, 140, "stage A: synthetic pretrain\nrandom voice on both sides\n(B never A's voice)\nboth speaker orders\nmix 0.7 real : 0.3 synthetic", "purple", 14)
  d.box(690, 70, 300, 140, "stage B: assistant fine-tune\nJOSIE = Zoey as A\nrandom user voice as B\n--no_swap: loss on A only\nlr 1e-4, 3 epochs", "blue", 14)
  d.box(1050, 95, 170, 90, "JOSIE\nfull-duplex\nassistant", "green")
  d.arrow([(270, 140), (330, 140)], "--init"); d.arrow([(630, 140), (690, 140)], "--init"); d.arrow([(990, 140), (1050, 140)])
  d.box(330, 250, 660, 60, "status: stage A data notebook (synthetic_pretrain.ipynb) exists; stage A not yet trained.\nRuns in this paper: caramel-1 -> stage B directly (swap and --no_swap).", "yellow", 14)
  d.save(OUT)


def failure():
  d = Diagram("fig9_failure")
  d.text(20, 10, "Failure mode of the both-sides fine-tune (josie-synth-overnight2)", 20, bold=True)
  d.box(20, 60, 330, 120, "training windows\nhalf as user (B first):\n25 user voices, barge-ins live\non the user side", "orange", 15)
  d.box(420, 60, 330, 120, "learned: 'whoever I am,\ninterrupt and speak in\nsome catalog voice'", "red", 15)
  d.arrow([(350, 120), (420, 120)])
  d.box(820, 40, 380, 70, "speaker-identity leakage:\nJOSIE sometimes in a (male) user voice", "red", 15)
  d.box(820, 130, 380, 70, "barge-in imitation:\ntalks over the user", "red", 15)
  d.arrow([(750, 100), (820, 75)]); d.arrow([(750, 140), (820, 165)])
  d.box(20, 240, 1180, 70, "self-play, 5 seeds x 30 s from a real opening: JOSIE side active 71%, other side 4% (per seed 0, 0, 14, 5, 0),\nboth 1%, silent 26%; caramel-1 reference for the other side: 32%. Fix: --no_swap (Fig. 7).", "yellow", 15)
  d.save(OUT)


for f in (overview, tokens, arch, selfplay, pipeline, scenes, objective, recipe, failure): f()
print("ok")
