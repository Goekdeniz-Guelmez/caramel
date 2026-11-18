"""Dev-loss and train-loss plots from runs/*/train.log (plain SVG -> PDF via rsvg-convert, no plotting library)."""
import os, re, subprocess
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")


def parse(run):
  ev, tr = {}, {}
  for l in open(os.path.join(ROOT, "runs", run, "train.log"), errors="ignore"):
    if m := re.match(r"EVAL step (\d+) dev_loss ([\d.]+)", l): ev[int(m[1])] = float(m[2])
    if m := re.match(r"step (\d+) loss ([\d.]+) cb0 ([\d.]+) lr ([\d.e+-]+)", l): tr[int(m[1])] = (float(m[2]), float(m[3]), float(m[4]))
  return sorted(ev.items()), sorted(tr.items())


def chart(name, series, xmax, ymin, ymax, ylabel, title, yticks):
  W, H, L, R, T, B = 640, 380, 70, 20, 40, 55
  sx = lambda x: L + (W - L - R) * x / xmax
  sy = lambda y: T + (H - T - B) * (1 - (y - ymin) / (ymax - ymin))
  s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="13">',
       '<rect width="100%" height="100%" fill="#fff"/>', f'<text x="{L}" y="24" font-size="15" font-weight="bold">{title}</text>']
  for y in yticks:
    s.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#e5e5e5"/>')
    s.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" text-anchor="end" fill="#555">{y:.2f}</text>')
  for x in range(0, xmax + 1, 50):
    s.append(f'<text x="{sx(x):.1f}" y="{H-B+18}" text-anchor="middle" fill="#555">{x}</text>')
  s.append(f'<line x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}" stroke="#333"/><line x1="{L}" x2="{L}" y1="{T}" y2="{H-B}" stroke="#333"/>')
  s.append(f'<text x="{(L+W-R)/2}" y="{H-12}" text-anchor="middle">optimizer step (48k frames = 64 min of conversation per step)</text>')
  s.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
  for k, (lab, pts, col, dash) in enumerate(series):
    d = " ".join(f"{'M' if i == 0 else 'L'} {sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
    s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2"{" stroke-dasharray=\"5 4\"" if dash else ""}/>')
    if not dash:
      for x, y in pts: s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{col}"/>')
    ly = T + 18 + 20 * k
    s.append(f'<line x1="{W-R-250}" x2="{W-R-225}" y1="{ly}" y2="{ly}" stroke="{col}" stroke-width="2.5"/><text x="{W-R-218}" y="{ly+4}">{lab}</text>')
  s.append("</svg>")
  open(f"{OUT}/{name}.svg", "w").write("\n".join(s))
  subprocess.run(["rsvg-convert", "-f", "pdf", "-o", f"{OUT}/{name}.pdf", f"{OUT}/{name}.svg"], check=True)
  subprocess.run(["rsvg-convert", "-z", "2", "-o", f"{OUT}/{name}.png", f"{OUT}/{name}.svg"], check=True)


e50, t50 = parse("josie-synth-50h"); e137, t137 = parse("josie-synth-overnight2"); eA, tA = parse("josie-synth-overnight-aonly")
chart("plot_dev_loss", [("50 h, both sides (141 steps)", e50, "#1971c2", False), ("137 h, both sides (386 steps)", e137, "#e8590c", False)],
      400, 1.30, 1.66, "dev loss (both speaker orders)", "Dev loss of caramel-1 fine-tuned on JOSIE synthetic data", [1.3, 1.4, 1.5, 1.6])
chart("plot_train_loss", [("137 h train loss", [(x, v[0]) for x, v in t137], "#e8590c", True),
                          ("137 h codebook-0 loss", [(x, v[1]) for x, v in t137], "#2f9e44", True),
                          ("137 h dev loss", e137, "#1e1e1e", False)],
      400, 0.8, 1.9, "loss (nats)", "Training vs dev loss, josie-synth-overnight2", [0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
print("dev 50h", e50, "\ndev 137h", e137, "\naonly", eA, tA[-1] if tA else None)
