"""Listening page / sections for PUBG background false-boost cases (test4 / test14).

Cuts match fig_nm_bg_examples: full labeled gap + quiet magnifier window.
Tracks: raw / old FPS AI / newmodel.
Output: docs/PUBG 背景误增强试听.html
"""
import os
import html as html_mod
import pandas as pd
from listen_boosted import mp3_uri
from compare_pubg_newmodel import load_pubg_quad, RES, SR
from report_pubg_newmodel import analyze_bg_frames, _frame_db, _pick_zoom, FER_THR
from report_ad import gain_db as gain_db_local, HERE

OUT = os.path.join(os.path.dirname(HERE), "docs", "PUBG 背景误增强试听.html")
TRACKS = [("raw", "原始"), ("old", "旧模型"), ("new", "新模型")]


def player(src):
    return f'<audio controls preload="none" src="{src}"></audio>'


def cut_uris(a, t0, t1):
    s, e = int(t0 * SR), int(t1 * SR)
    return {k: mp3_uri(a[k][s:e]) for k, _ in TRACKS}


def pick_cases(bf):
    cases = bf[bf.G_old > FER_THR].sort_values("G_old", ascending=False)
    if len(cases) < 2:
        cases = bf.sort_values("G_old", ascending=False).head(2)
    fixed = cases[(cases.G_old > FER_THR) & (cases.G_new <= FER_THR)]
    other = cases[~cases.index.isin(fixed.index)]
    return pd.concat([fixed, other]).head(2).reset_index(drop=True)


def build_listen_sections(bf):
    """Return HTML blocks with embedded players for the two false-boost examples."""
    cases = pick_cases(bf)
    blocks = []
    for i, r in cases.iterrows():
        clip = int(r["clip"])
        start, dur = float(r["start"]), float(r["dur"])
        a, _ = load_pubg_quad(clip)
        s0, e0 = int(start * SR), int((start + dur) * SR)
        raw = a["raw"][s0:e0]
        fr = _frame_db(raw)
        z0, z1 = _pick_zoom(fr, len(raw))
        zt0, zt1 = start + z0 / SR, start + z1 / SR
        t1 = start + dur

        g_old = gain_db_local(a["old"][s0:e0], raw)
        g_new = gain_db_local(a["new"][s0:e0], a["raw"][s0:e0])
        g_zo = gain_db_local(a["old"][int(zt0 * SR):int(zt1 * SR)],
                             a["raw"][int(zt0 * SR):int(zt1 * SR)])
        g_zn = gain_db_local(a["new"][int(zt0 * SR):int(zt1 * SR)],
                             a["raw"][int(zt0 * SR):int(zt1 * SR)])

        full = cut_uris(a, start, t1)
        zoom = cut_uris(a, zt0, zt1)
        label = "A" if i == 0 else "B"
        if r.G_old > FER_THR and r.G_new <= FER_THR:
            badge = "整段误增强已消除"
        elif bool(r.contaminated):
            badge = "含突发声 · 听放大镜底噪"
        else:
            badge = "底噪对比"

        row_full = "".join(f"<td>{player(full[k])}</td>" for k, _ in TRACKS)
        row_zoom = "".join(f"<td>{player(zoom[k])}</td>" for k, _ in TRACKS)
        blocks.append(f"""
<section class="listen" id="listen-{label}">
  <h3>样例 {label} · PUBG test{clip}
      <span class="badge">{html_mod.escape(badge)}</span></h3>
  <p class="meta">整段 {start:.2f}–{t1:.2f} s　增益 旧 {g_old:+.1f} / 新 {g_new:+.1f} dB
    　·　放大镜 {zt0:.2f}–{zt1:.2f} s　增益 旧 {g_zo:+.1f} / 新 {g_zn:+.1f} dB</p>
  <table>
    <tr><th>窗口</th><th>原始</th><th>旧模型</th><th>新模型</th></tr>
    <tr><td>整段</td>{row_full}</tr>
    <tr><td>安静段放大镜</td>{row_zoom}</tr>
  </table>
</section>""")
    return "".join(blocks), cases


def write_listen_html(bf=None):
    if bf is None:
        bg = pd.read_csv(os.path.join(RES, "pubg_newmodel_background.csv"), encoding="utf-8-sig")
        bf = analyze_bg_frames(bg)
    from pubg_report_theme import page_shell

    sections, _ = build_listen_sections(bf)
    hero = """
<div class="eyebrow">A/B Listening Desk</div>
<h1 class="brand">LISTEN <span>/ BG FALSE BOOST</span></h1>
<p class="tagline">与报告波形图同一裁切。每例含整段与安静段放大镜；轨为原始 / 旧模型 / 新模型。
建议先听放大镜三轨，再对照整段。</p>
<nav class="cta">
  <a href="PUBG%20新旧模型对比.html">返回对比报告</a>
  <a class="ghost" href="PUBG%20FPS%20AI%20新旧模型对比报告.pdf">PDF 报告</a>
</nav>"""
    body = f"""
<section class="block">
  <h2><span class="n">AB</span> 对比试听</h2>
  {sections}
</section>
"""
    page = page_shell("C3 FPS AI · 背景误增强试听", hero, body)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    return OUT


def main():
    out = write_listen_html()
    print("->", out)


if __name__ == "__main__":
    main()
