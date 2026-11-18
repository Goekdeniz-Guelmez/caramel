"""Tiny diagram DSL that writes both an Excalidraw file and a matching SVG (rendered to PDF with rsvg-convert)."""
import json, math, random, subprocess
from xml.sax.saxutils import escape

COL = dict(blue="#a5d8ff", green="#b2f2bb", orange="#ffd8a8", purple="#d0bfff", red="#ffc9c9",
           yellow="#fff3bf", teal="#c3fae8", gray="#e9ecef", white="#ffffff", none="transparent")
INK = "#1e1e1e"


class Diagram:
  def __init__(self, name):
    self.name, self.el, self.svg, self.n = name, [], [], 0
    self.rng = random.Random(name)
    self.maxx = self.maxy = 0

  def _id(self, p): self.n += 1; return f"{p}{self.n}"

  def _ext(self, x, y): self.maxx, self.maxy = max(self.maxx, x), max(self.maxy, y)

  def _wobble(self, x, y, w, h):   # slightly hand-drawn rectangle path, Excalidraw-like
    j = lambda: self.rng.uniform(-1.2, 1.2)
    p = [(x + j(), y + j()), (x + w + j(), y + j()), (x + w + j(), y + h + j()), (x + j(), y + h + j())]
    return "M " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in p) + " Z"

  def box(self, x, y, w, h, text="", color="blue", size=16, shape="rectangle", dashed=False, bold=False, ink=INK):
    i = self._id("s"); fill = COL.get(color, color)
    e = dict(type=shape, id=i, x=x, y=y, width=w, height=h, strokeColor=ink, backgroundColor=fill,
             fillStyle="solid", strokeWidth=2, roughness=1, opacity=100, strokeStyle="dashed" if dashed else "solid")
    if shape == "rectangle": e["roundness"] = {"type": 3}
    self.el.append(e)
    da = ' stroke-dasharray="6 4"' if dashed else ""
    if shape == "ellipse":
      self.svg.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{fill}" stroke="{ink}" stroke-width="2"{da}/>')
    elif shape == "diamond":
      self.svg.append(f'<path d="M {x+w/2},{y} L {x+w},{y+h/2} L {x+w/2},{y+h} L {x},{y+h/2} Z" fill="{fill}" stroke="{ink}" stroke-width="2"{da}/>')
    else:
      self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="none"/>')
      self.svg.append(f'<path d="{self._wobble(x, y, w, h)}" fill="none" stroke="{ink}" stroke-width="2" stroke-linejoin="round"{da}/>')
    if text:
      t = self._id("t"); e["boundElements"] = [{"id": t, "type": "text"}]
      self.el.append(dict(type="text", id=t, x=x + 5, y=y + 5, width=w - 10, height=h - 10, text=text, originalText=text,
                          fontSize=size, fontFamily=1, strokeColor=ink, textAlign="center", verticalAlign="middle",
                          containerId=i, autoResize=True))
      self._svgtext(x + w / 2, y + h / 2, text, size, "middle", bold, ink, center_v=True)
    self._ext(x + w, y + h)
    return (x, y, w, h)

  def _svgtext(self, x, y, text, size, anchor, bold, ink, center_v=False):
    lines = text.split("\n"); lh = size * 1.22
    y0 = y - (len(lines) - 1) * lh / 2 + size * 0.35 if center_v else y + size * 0.85
    fw = ' font-weight="bold"' if bold else ""
    for k, ln in enumerate(lines):
      self.svg.append(f'<text x="{x:.1f}" y="{y0 + k*lh:.1f}" font-size="{size}" text-anchor="{anchor}"{fw} fill="{ink}">{escape(ln)}</text>')

  def text(self, x, y, s, size=16, anchor="start", bold=False, ink=INK):
    i = self._id("t")
    wid = max(len(l) for l in s.split("\n")) * size * 0.5
    ex = x - wid / 2 if anchor == "middle" else (x - wid if anchor == "end" else x)
    self.el.append(dict(type="text", id=i, x=ex, y=y, width=wid, height=size * 1.25 * len(s.split("\n")), text=s,
                        originalText=s, fontSize=size, fontFamily=1, strokeColor=ink, autoResize=True))
    self._svgtext(x, y, s, size, anchor, bold, ink)
    self._ext(ex + wid, y + size * 1.3 * len(s.split("\n")))

  def arrow(self, pts, label="", dashed=False, size=14, ink=INK, head=True, both=False, lab_off=(0, -10), width=2):
    i = self._id("a"); x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    e = dict(type="arrow", id=i, x=x0, y=y0, width=max(abs(p[0]) for p in rel), height=max(abs(p[1]) for p in rel),
             points=rel, endArrowhead="arrow" if head else None, startArrowhead="arrow" if both else None,
             strokeColor=ink, strokeWidth=width, roughness=1, strokeStyle="dashed" if dashed else "solid")
    self.el.append(e)
    da = ' stroke-dasharray="7 5"' if dashed else ""
    d = "M " + " L ".join(f"{px},{py}" for px, py in pts)
    self.svg.append(f'<path d="{d}" fill="none" stroke="{ink}" stroke-width="{width}"{da} stroke-linejoin="round"/>')
    def headat(a, b):
      ang = math.atan2(b[1] - a[1], b[0] - a[0]); L = 11
      p1 = (b[0] - L * math.cos(ang - 0.42), b[1] - L * math.sin(ang - 0.42))
      p2 = (b[0] - L * math.cos(ang + 0.42), b[1] - L * math.sin(ang + 0.42))
      self.svg.append(f'<path d="M {p1[0]:.1f},{p1[1]:.1f} L {b[0]},{b[1]} L {p2[0]:.1f},{p2[1]:.1f}" fill="none" stroke="{ink}" stroke-width="{width}" stroke-linejoin="round"/>')
    if head: headat(pts[-2], pts[-1])
    if both: headat(pts[1], pts[0])
    if label:
      k = (len(pts) - 1) // 2; mx_ = (pts[k][0] + pts[k + 1][0]) / 2 + lab_off[0]; my_ = (pts[k][1] + pts[k + 1][1]) / 2 + lab_off[1]
      self.text(mx_, my_ - size, label, size, "middle")
    for p in pts: self._ext(*p)

  def line(self, pts, dashed=False, ink=INK, width=2): self.arrow(pts, dashed=dashed, ink=ink, head=False, width=width)

  def save(self, outdir, pad=20):
    W, H = self.maxx + pad, self.maxy + pad
    json.dump(dict(type="excalidraw", version=2, source="caramel-paper", elements=self.el,
                   appState={"viewBackgroundColor": "#ffffff"}, files={}),
              open(f"{outdir}/{self.name}.excalidraw", "w"), indent=1)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>' + "\n".join(self.svg) + "</svg>")
    open(f"{outdir}/{self.name}.svg", "w").write(svg)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", f"{outdir}/{self.name}.pdf", f"{outdir}/{self.name}.svg"], check=True)
    subprocess.run(["rsvg-convert", "-z", "2", "-o", f"{outdir}/{self.name}.png", f"{outdir}/{self.name}.svg"], check=True)
