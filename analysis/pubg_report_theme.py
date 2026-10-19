"""Shared visual theme for PUBG new-model HTML pages (premium gaming DAC tone)."""

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {
  --bg0: #0a0c10;
  --bg1: #11151c;
  --bg2: #181d27;
  --line: rgba(212, 168, 95, 0.28);
  --line2: rgba(255, 255, 255, 0.08);
  --text: #e8e4dc;
  --muted: #9a9488;
  --copper: #d4a85f;
  --copper2: #f0c987;
  --signal: #5ec8c0;
  --raw: #c8c2b6;
  --old: #8a8478;
  --new: #5ec8c0;
  --danger: #c97b63;
  --radius: 2px;
  --font-display: "Orbitron", "Noto Sans SC", sans-serif;
  --font-body: "Noto Sans SC", "Microsoft YaHei", sans-serif;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }

body {
  margin: 0;
  min-height: 100vh;
  color: var(--text);
  font: 15px/1.65 var(--font-body);
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(212, 168, 95, 0.12), transparent 55%),
    radial-gradient(900px 500px at 90% 0%, rgba(94, 200, 192, 0.08), transparent 50%),
    linear-gradient(180deg, var(--bg0), var(--bg1) 40%, var(--bg0));
  background-attachment: fixed;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.035;
  background-image:
    linear-gradient(var(--line2) 1px, transparent 1px),
    linear-gradient(90deg, var(--line2) 1px, transparent 1px);
  background-size: 48px 48px;
  z-index: 0;
}

.wrap {
  position: relative;
  z-index: 1;
  max-width: 980px;
  margin: 0 auto;
  padding: 0 22px 64px;
}

/* —— hero —— */
.hero {
  position: relative;
  padding: 48px 0 28px;
  margin-bottom: 8px;
  overflow: hidden;
  animation: rise 0.7s ease-out both;
}
.hero::after {
  content: "";
  position: absolute;
  left: 0; right: 0; bottom: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--copper), var(--signal), transparent);
  opacity: 0.7;
}
.brand {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: clamp(1.85rem, 4.5vw, 2.55rem);
  letter-spacing: 0.08em;
  line-height: 1.15;
  color: #fff;
  margin: 0 0 10px;
  text-transform: uppercase;
}
.brand span {
  color: var(--copper2);
  font-weight: 500;
}
.tagline {
  margin: 0 0 18px;
  max-width: 36em;
  color: var(--muted);
  font-size: 0.95rem;
}
.eyebrow {
  display: inline-block;
  font-family: var(--font-display);
  font-size: 0.68rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--copper);
  margin-bottom: 14px;
}

.cta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 8px;
}
.cta a {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(212,168,95,0.12), rgba(212,168,95,0.02));
  color: var(--copper2);
  text-decoration: none;
  font-size: 0.86rem;
  letter-spacing: 0.04em;
  transition: border-color .2s, background .2s, transform .2s;
}
.cta a:hover {
  border-color: var(--copper2);
  background: rgba(212,168,95,0.18);
  transform: translateY(-1px);
}
.cta a.ghost {
  background: transparent;
  color: var(--text);
  border-color: var(--line2);
}
.cta a.ghost:hover {
  border-color: var(--signal);
  color: var(--signal);
}

/* —— sections —— */
section.block {
  margin: 36px 0 0;
  animation: rise 0.65s ease-out both;
}
section.block:nth-of-type(2) { animation-delay: 0.08s; }
section.block:nth-of-type(3) { animation-delay: 0.14s; }
section.block:nth-of-type(4) { animation-delay: 0.2s; }

h2 {
  font-family: var(--font-display);
  font-size: 0.95rem;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--copper2);
  margin: 0 0 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--line2);
  display: flex;
  align-items: baseline;
  gap: 10px;
}
h2 .n {
  color: var(--signal);
  font-size: 0.75rem;
  letter-spacing: 0.08em;
}

.verdict {
  position: relative;
  padding: 4px 0 4px 14px;
  background: none;
  border: none;
  border-left: 2px solid var(--copper);
  margin: 18px 0 8px;
}
.verdict b { color: var(--copper2); }
.verdict .sub { color: var(--muted); margin-top: 8px; font-size: 0.9rem; }

.readouts {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--line2);
  border: 1px solid var(--line2);
  margin: 18px 0 8px;
}
.ro {
  background: var(--bg2);
  padding: 16px 12px;
  text-align: center;
}
.ro b {
  display: block;
  font-family: var(--font-display);
  font-size: 1.05rem;
  color: var(--signal);
  letter-spacing: 0.02em;
  margin-bottom: 6px;
}
.ro span {
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.06em;
}

.note, .meta { color: var(--muted); font-size: 0.88rem; }
.meta { margin: 0 0 12px; }
a { color: var(--signal); }
a:hover { color: var(--copper2); }
code {
  font-size: 0.84em;
  color: var(--copper2);
  background: rgba(255,255,255,0.04);
  padding: 1px 6px;
  border: 1px solid var(--line2);
}

/* tables */
.tw { overflow-x: auto; margin: 12px 0 16px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  background: var(--bg2);
}
th, td {
  border: 1px solid var(--line2);
  padding: 9px 11px;
  text-align: center;
}
th {
  background: rgba(212,168,95,0.08);
  color: var(--copper2);
  font-weight: 500;
  letter-spacing: 0.04em;
  font-size: 0.8rem;
}
td:first-child, th:first-child { text-align: left; }
tbody tr:hover td { background: rgba(94,200,192,0.04); }

figure.fig {
  margin: 14px 0 8px;
  padding: 0;
  background: transparent;
  border: 1px solid var(--line2);
  overflow: hidden;
}
figure.fig img { display: block; width: 100%; height: auto; vertical-align: middle; }
figcaption {
  margin: 0;
  padding: 10px 12px;
  font-size: 0.8rem;
  color: var(--muted);
  text-align: center;
  background: var(--bg2);
  border-top: 1px solid var(--line2);
}

/* listen panels — interaction containers */
section.listen {
  margin: 0 0 16px;
  padding: 16px 18px;
  background: var(--bg2);
  border: 1px solid var(--line2);
  border-top: 2px solid var(--copper);
  transition: border-color .25s, box-shadow .25s;
}
section.listen:hover {
  border-color: rgba(94,200,192,0.35);
  box-shadow: 0 0 0 1px rgba(94,200,192,0.12);
}
section.listen h3 {
  font-family: var(--font-display);
  font-size: 0.92rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  margin: 0 0 8px;
  color: #fff;
}
.badge {
  display: inline-block;
  margin-left: 8px;
  padding: 2px 9px;
  font-family: var(--font-body);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.04em;
  color: var(--bg0);
  background: linear-gradient(90deg, var(--copper), var(--copper2));
  border: none;
  vertical-align: middle;
}
audio {
  width: min(210px, 100%);
  height: 34px;
  vertical-align: middle;
  accent-color: var(--signal);
}
section.listen table th:nth-child(2) { color: var(--raw); }
section.listen table th:nth-child(3) { color: var(--old); }
section.listen table th:nth-child(4) { color: var(--new); }

.files ul { padding-left: 1.1em; color: var(--muted); }
.files li { margin: 6px 0; }

.foot {
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid var(--line2);
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  font-family: var(--font-display);
  text-align: center;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 720px) {
  .readouts { grid-template-columns: 1fr 1fr; }
  .hero { padding-top: 32px; }
  .brand { letter-spacing: 0.04em; }
}
"""


def page_shell(title, hero_html, body_html, foot="C3 FPS AI · COMPETITIVE AUDIO VALIDATION"):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    {hero_html}
  </header>
  {body_html}
  <footer class="foot">{foot}</footer>
</div>
</body>
</html>
"""
