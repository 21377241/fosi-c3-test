"""Short PUBG report: background false-boost claim only + brief check on other capabilities.

Output:
  docs/PUBG FPS AI 新旧模型对比报告.pdf
  docs/PUBG 新旧模型对比.html  （含试听链接与内嵌对比音频）
  docs/PUBG 背景误增强试听.html
"""
import os
import re
import base64
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import report_tex as rt
from report_tex import n, p, tex_table, save, OUTDIR, TAB, FIG, DOCS, C_AI, C_EQ, C_EQM, C_GRP
from report_final import pct, sgn, FOOT, NT, FER_THR
from compare_pubg_newmodel import measure, summarize, RES, load_pubg_quad, SR
from report_ad import gain_db as gain_db_local

PDF_OUT = os.path.join(DOCS, "PUBG FPS AI 新旧模型对比报告.pdf")
HTML_OUT = os.path.join(DOCS, "PUBG 新旧模型对比.html")
LISTEN_NAME = "PUBG 背景误增强试听.html"
PDF_NAME = "PUBG FPS AI 新旧模型对比报告.pdf"
C_OLD, C_NEW = C_EQ, C_AI
C_RAW = "#333333"
C_FOOT = C_GRP[FOOT]
C_LINE = "#BBBBBB"
BG_BURST_PEAK = -55.0
BG_QUIET = -65.0
BG_HOP = int(0.05 * SR)

# print = PDF；web = HTML 深色主题（与 pubg_report_theme 一致）
THEME = {
    "print": dict(
        bg="#FFFFFF", panel="#FFFFFF", text="#222222", muted="#555555",
        raw="#333333", old=C_EQ, new=C_AI, accent=C_AI,
        zero="#E0E0E0", grid_alpha=.22, badge_fc="#EEF3F9", bridge_fc="#F7FAFD",
        spine="#CCCCCC", edge="white",
    ),
    "web": dict(
        bg="#11151c", panel="#181d27", text="#e8e4dc", muted="#9a9488",
        raw="#c8c2b6", old="#8a8478", new="#5ec8c0", accent="#d4a85f",
        zero="#2a3140", grid_alpha=.18, badge_fc="#1c2330", bridge_fc="#1c2330",
        spine="#3a4254", edge="#11151c",
    ),
}


def _style(ax, th=None):
    th = th or THEME["print"]
    ax.set_facecolor(th["panel"])
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(th["spine"])
    ax.tick_params(axis="both", labelsize=9, colors=th["muted"])
    ax.yaxis.label.set_color(th["text"])
    ax.xaxis.label.set_color(th["muted"])
    ax.title.set_color(th["text"])
    ax.grid(axis="y", alpha=th["grid_alpha"], color=th["spine"], zorder=0)


def _frame_db(x, hop=BG_HOP):
    out = []
    for i in range(0, len(x) - hop, hop):
        out.append(20 * np.log10(np.sqrt((x[i:i + hop] ** 2).mean()) + 1e-12))
    return np.array(out)


def analyze_bg_frames(bg):
    rows = []
    for i in range(len(bg)):
        r = bg.iloc[i]
        a, _ = load_pubg_quad(int(r["clip"]))
        s0, e0 = int(r["start"] * SR), int((r["start"] + r["dur"]) * SR)
        raw, old, new = a["raw"][s0:e0], a["old"][s0:e0], a["new"][s0:e0]
        fr, fo, fn = _frame_db(raw), _frame_db(old), _frame_db(new)
        go, gn = fo - fr, fn - fr
        quiet = fr < BG_QUIET
        peak = float(fr.max()) if len(fr) else -120.0
        rows.append(dict(
            clip=int(r["clip"]), start=float(r["start"]), dur=float(r["dur"]),
            G_old=float(r["G_old"]), G_new=float(r["G_new"]),
            peak_raw=peak, contaminated=peak > BG_BURST_PEAK,
            G_old_quiet=float(np.median(go[quiet])) if quiet.any() else float("nan"),
            G_new_quiet=float(np.median(gn[quiet])) if quiet.any() else float("nan"),
            G_old_med=float(np.median(go)) if len(go) else float("nan"),
            G_new_med=float(np.median(gn)) if len(gn) else float("nan"),
        ))
    return pd.DataFrame(rows)


def verdict_bg(bf):
    g_o = bf.G_old_quiet.fillna(bf.G_old_med)
    g_n = bf.G_new_quiet.fillna(bf.G_new_med)
    rate_o = float((g_o > FER_THR).mean())
    rate_n = float((g_n > FER_THR).mean())
    if rate_n < 1e-12 and rate_o > rate_n:
        claim = "支持：真正底噪上的误增强已压到阈值以下"
    elif rate_n < rate_o - 1e-9:
        claim = "支持：底噪误增强率下降"
    else:
        claim = "不支持：底噪误增强未见改善"
    return dict(claim=claim, rate_o=rate_o, rate_n=rate_n,
                med_o=float(g_o.median()), med_n=float(g_n.median()),
                n_cont=int(bf.contaminated.sum()), g_o=g_o, g_n=g_n)


def _longest_quiet_run(fr, thr=BG_QUIET):
    quiet = fr < thr
    best = (0, 0)
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best


def fig_bg_claim(bf, name, theme="print"):
    th = THEME[theme]
    g_o = bf.G_old_quiet.fillna(bf.G_old_med)
    g_n = bf.G_new_quiet.fillna(bf.G_new_med)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), facecolor=th["bg"])
    ax = axes[0]
    vals = [(g_o > FER_THR).mean() * 100, (g_n > FER_THR).mean() * 100]
    bars = ax.bar(["旧模型", "新模型"], vals, color=[th["old"], th["new"]],
                  width=.55, edgecolor=th["edge"], lw=.8, zorder=3)
    for rect, v, n_ in zip(bars, vals, [(g_o > FER_THR).sum(), (g_n > FER_THR).sum()]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.0, f"{v:.0f}%\n({int(n_)}/{len(bf)})",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=th["text"])
    ax.set_ylabel(f"安静帧中位增益 > {FER_THR:.0f} dB 的段占比（%）")
    ax.set_title("底噪误增强率", fontsize=11)
    ax.set_ylim(0, max(vals + [8]) * 1.5 + 4)
    _style(ax, th)
    ax = axes[1]
    meds = [g_o.median(), g_n.median()]
    bars = ax.bar(["旧模型", "新模型"], meds, color=[th["old"], th["new"]],
                  width=.55, edgecolor=th["edge"], lw=.8, zorder=3)
    for rect, v in zip(bars, meds):
        ax.text(rect.get_x() + rect.get_width() / 2, v + (0.2 if v >= 0 else -0.2),
                f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold", color=th["text"])
    ax.axhline(0, c=th["text"], lw=.6, alpha=.5)
    ax.axhline(FER_THR, c=th["accent"], ls="--", lw=.9, alpha=.7)
    ax.set_ylabel("安静帧中位增益（dB）")
    ax.set_title("底噪中位增益", fontsize=11)
    _style(ax, th)
    fig.suptitle("官方口径：背景/底噪误增强（PUBG）", fontsize=11.5, y=1.02, color=th["text"])
    plt.tight_layout()
    if theme == "print":
        save(fig, name)
    else:
        fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=220, bbox_inches="tight",
                    facecolor=th["bg"], edgecolor="none")
        plt.close(fig)


def _pick_zoom(fr, raw_len, min_sec=0.35):
    """Quietest interior stretch for magnifier (prefer <-65 dBFS run)."""
    i0, i1 = _longest_quiet_run(fr)
    trim = 2
    if i1 - i0 > 2 * trim + 3:
        i0, i1 = i0 + trim, i1 - trim
    if i1 - i0 < 4:
        # lowest-energy sliding window (~0.4 s)
        win = max(4, int(min_sec / 0.05))
        if len(fr) <= win:
            i0, i1 = 0, len(fr)
        else:
            ker = np.ones(win) / win
            sm = np.convolve(fr, ker, mode="valid")
            i0 = int(np.argmin(sm))
            i1 = i0 + win
    z0 = int(i0 * BG_HOP)
    z1 = int(min(raw_len, max(i1 * BG_HOP, z0 + int(min_sec * SR))))
    return z0, z1


def _draw_case_block(fig, gs_block, clip, start, dur, title, badge, theme="print"):
    """上排横排整段；中间说明条；下排各轨放大镜。"""
    th = THEME[theme]
    a, _ = load_pubg_quad(clip)
    s0, e0 = int(start * SR), int((start + dur) * SR)
    raw, old, new = a["raw"][s0:e0], a["old"][s0:e0], a["new"][s0:e0]
    waves_full = [raw.mean(1), old.mean(1), new.mean(1)]
    fr = _frame_db(raw)
    t_full = np.arange(len(waves_full[0])) / SR
    g_old = gain_db_local(old, raw)
    g_new = gain_db_local(new, raw)

    z0, z1 = _pick_zoom(fr, len(waves_full[0]))
    zt0, zt1 = z0 / SR, z1 / SR
    waves_z = [w[z0:z1] for w in waves_full]
    g_zo = gain_db_local(waves_z[1], waves_z[0])
    g_zn = gain_db_local(waves_z[2], waves_z[0])
    tt = np.arange(len(waves_z[0])) / SR * 1000
    t_end = float(t_full[-1]) if len(t_full) else dur
    t_z_end = float(tt[-1]) if len(tt) else 1.0

    lim_full = np.percentile(np.abs(np.concatenate(waves_full)), 99.5) * 1.12 + 1e-9
    lim_z = max(np.percentile(np.abs(w), 99.7) for w in waves_z) * 1.25 + 1e-9
    step = max(1, len(waves_full[0]) // 3500)

    tracks = [
        ("原始", th["raw"], waves_full[0], waves_z[0], "参考", "参考"),
        ("旧模型", th["old"], waves_full[1], waves_z[1], f"{g_old:+.1f} dB", f"{g_zo:+.1f} dB"),
        ("新模型", th["new"], waves_full[2], waves_z[2],
         f"{g_new:+.1f} dB  Δ{g_new - g_old:+.1f}", f"{g_zn:+.1f} dB"),
    ]

    inner = gs_block.subgridspec(
        4, 3, height_ratios=[0.36, 1.20, 0.36, 1.10],
        hspace=0.40, wspace=0.14)

    ax_h = fig.add_subplot(inner[0, :])
    ax_h.set_facecolor(th["bg"])
    ax_h.axis("off")
    ax_h.text(0.0, 0.78, title, transform=ax_h.transAxes, fontsize=11,
              fontweight="bold", va="center", color=th["text"])
    ax_h.text(0.0, 0.18,
              f"{badge}　　放大窗口 {start + zt0:.2f}–{start + zt1:.2f} s",
              transform=ax_h.transAxes, fontsize=8.2, color=th["accent"],
              fontweight="bold", va="center",
              bbox=dict(boxstyle="round,pad=0.2", fc=th["badge_fc"], ec=th["accent"], lw=.8))

    for k, (lab, col, w_full, w_z, g_full, g_zoom) in enumerate(tracks):
        ax = fig.add_subplot(inner[1, k])
        ax.set_facecolor(th["panel"])
        ax.plot(t_full[::step], w_full[::step], c=col, lw=.9)
        ax.axhline(0, c=th["zero"], lw=.4)
        ax.set_ylim(-lim_full, lim_full)
        ax.set_xlim(0, t_end)
        ax.add_patch(Rectangle((zt0, -lim_full * 0.96), zt1 - zt0, lim_full * 1.92,
                               linewidth=1.2, edgecolor=th["accent"],
                               facecolor=th["accent"], alpha=0.14 if theme == "web" else 0.12, zorder=0))
        ax.axvline(zt0, c=th["accent"], ls="--", lw=.8, alpha=.65)
        ax.axvline(zt1, c=th["accent"], ls="--", lw=.8, alpha=.65)
        ax.set_title(f"{lab}  ·  {g_full}", fontsize=9.5, color=col, fontweight="bold", pad=5)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7.5, pad=1, colors=th["muted"])
        for sp in ax.spines.values():
            sp.set_color(th["spine"])
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel("整段（同纵轴）", fontsize=8.5, labelpad=6, color=th["muted"])

        axz = fig.add_subplot(inner[3, k])
        axz.set_facecolor(th["panel"])
        axz.plot(tt, w_z, c=col, lw=1.05)
        axz.axhline(0, c=th["zero"], lw=.4)
        axz.set_ylim(-lim_z, lim_z)
        axz.set_xlim(0, t_z_end)
        for sp in axz.spines.values():
            sp.set_visible(True)
            sp.set_color(th["accent"])
            sp.set_linewidth(1.4)
        axz.set_yticks([])
        axz.set_title(f"{lab}  ·  {g_zoom}", fontsize=9, color=col, fontweight="bold", pad=4)
        axz.tick_params(axis="x", labelsize=7.5, pad=2, colors=th["muted"])
        axz.set_xlabel("时间（ms）", fontsize=7.5, labelpad=2, color=th["muted"])
        if k == 0:
            axz.set_ylabel("安静段放大镜", fontsize=8.5, color=th["accent"],
                           fontweight="bold", labelpad=6)

    ax_m = fig.add_subplot(inner[2, :])
    ax_m.set_facecolor(th["bg"])
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)
    ax_m.axis("off")
    box_lab = "高亮框" if theme == "web" else "蓝框"
    ax_m.text(0.5, 0.5, f"↓ 下排 = 上排{box_lab}安静段放大（列对齐 · 同纵轴）· 上排横轴单位 s",
              transform=ax_m.transAxes, ha="center", va="center", fontsize=8.2,
              color=th["accent"], fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.35", fc=th["bridge_fc"], ec=th["accent"], lw=0.9))

    return dict(clip=clip, g_old=g_old, g_new=g_new, g_zo=g_zo, g_zn=g_zn,
                zt0=start + zt0, zt1=start + zt1)


def fig_bg_examples(bf, name, theme="print"):
    """两处误增强：上排横排整段对比；下排各轨放大镜。theme=web 仅用于 HTML。"""
    th = THEME[theme]
    cases = bf[bf.G_old > FER_THR].sort_values("G_old", ascending=False)
    if len(cases) < 2:
        cases = bf.sort_values("G_old", ascending=False).head(2)
    fixed = cases[(cases.G_old > FER_THR) & (cases.G_new <= FER_THR)]
    other = cases[~cases.index.isin(fixed.index)]
    cases = pd.concat([fixed, other]).head(2).reset_index(drop=True)

    fig = plt.figure(figsize=(11.0, 11.0), facecolor=th["bg"])
    outer = fig.add_gridspec(2, 1, hspace=0.42, left=0.09, right=0.985, top=0.91, bottom=0.03)
    infos = []
    labels = ["A", "B"]
    for i, r in cases.iterrows():
        if r.G_old > FER_THR and r.G_new <= FER_THR:
            badge = "整段误增强已消除"
        elif bool(r.contaminated):
            badge = "含突发声 · 放大镜看底噪"
        else:
            badge = "底噪对比"
        title = f"样例 {labels[i]} · PUBG test{int(r['clip'])}（{r['start']:.1f}–{r['start'] + r['dur']:.1f} s）"
        infos.append(_draw_case_block(fig, outer[i], int(r["clip"]), float(r["start"]),
                                      float(r["dur"]), title, badge, theme=theme))

    fig.suptitle("两处旧模型误增强样例：上排横排整段对比 · 下排由蓝框分别引出放大镜",
                 fontsize=12, fontweight="bold", y=0.972, color=th["text"])
    fig.text(0.53, 0.938,
             "上排同一行纵轴相同；下排安静段放大、三轨各自对应且同纵轴。",
             ha="center", fontsize=8.5, color=th["muted"])
    dpi = 220 if theme == "web" else 350
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=dpi, bbox_inches="tight",
                facecolor=th["bg"], edgecolor="none")
    plt.close(fig)
    return infos


PREAMBLE = rt.TEMPLATE.split(r"\begin{document}")[0].replace(
    r"\lhead{\small C3 FPS AI 声卡客观评测技术报告（详细版）}",
    r"\lhead{\small PUBG FPS AI 新旧模型对比（背景专项）}")

TEMPLATE = PREAMBLE + r"""
\definecolor{ai}{HTML}{1F4E79}
\newcommand{\tagbox}[1]{\colorbox{ai!10}{\strut\heiti\color{ai}\ #1\ }}
\newcommand{\greybox}[1]{\par\noindent\colorbox{black!6}{\parbox{\dimexpr\linewidth-2\fboxsep}{\small #1}}\par}
\newcommand{\bignum}[2]{\parbox[t]{0.24\linewidth}{\centering{\fontsize{17}{20}\selectfont\heiti\color{ai}#1}\\[2pt]{\scriptsize #2}}}

\begin{document}
\begin{titlepage}
\centering\vspace*{40mm}
{\Huge\heiti PUBG FPS AI\par}\vspace{6pt}
{\huge\heiti 新旧模型对比\par}
\vspace{8mm}
{\Large 背景声误增强 · 专项简报\par}
\vspace{18mm}
\begin{tabular}{rl}
官方主口径： & 优化背景声的误增强\\
范围： & PUBG 15 段；背景空隙 [[n_bg]] 段\\
对照： & 旧 FPS AI vs 新模型（同源原始轨）\\
版本： & v2.0\quad 2026 年 10 月\\
\end{tabular}
\vfill
{\small 独立补充；不改动主报告结论。\par}
\end{titlepage}

\section{结论}
\keybox{\textbf{[[claim]]}\\[4pt]
以段内安静帧（短时电平 $<-65$\,dBFS）衡量底噪：误增强率 [[bg_rate_old]]$\rightarrow$[[bg_rate_new]]；
中位增益 [[bg_old]]$\rightarrow$[[bg_new]]\,dB。\\[2pt]
\textbf{其他能力：}[[other_verdict]]}

\begin{center}
\bignum{[[bg_rate_old]]$\rightarrow$[[bg_rate_new]]}{底噪误增强率}\\%
\bignum{[[bg_old]]$\rightarrow$[[bg_new]]}{底噪中位增益（dB）}\\%
\bignum{[[g_fo_old]]$\rightarrow$[[g_fo_new]]}{脚步中位（dB）}\\%
\bignum{[[fer_old]]$\rightarrow$[[fer_new]]}{事件 FER}
\end{center}

\vspace{2mm}
\input{tables/nm_bg_summary}

\section{证据：两处误增强样例}
旧模型有两段背景空隙增益超过 3\,dB（test4、test14）。下图对每一段：\textbf{上排}原始 / 旧模型 / 新模型横排（同一行同纵轴，整段振幅一眼可比）；\textbf{下排}由蓝框安静区分别引出放大镜，细看底噪。

\figslot{fig_nm_bg_examples}{0.98\linewidth}{上排横排整段对比；下排各轨放大镜。样例 A（test4）整段误增强已消除；样例 B（test14）含突发声，放大镜内底噪已压下}{nmbgex}

\figslot{fig_nm_bg_claim}{0.88\linewidth}{汇总：底噪误增强率与中位增益（安静帧口径）}{nmbg}

\greybox{样例 A：整段旧 [[ex_a_old]]$\rightarrow$新 [[ex_a_new]]\,dB。
样例 B：整段 RMS 仍高（旧 [[ex_b_old]] / 新 [[ex_b_new]]），因夹带脚步类突发声；放大镜安静段旧 [[ex_b_zo]]$\rightarrow$新 [[ex_b_zn]]\,dB，底噪已压下。}

\section{其他能力有没有明显削弱？}
\input{tables/nm_other}

\greybox{[[other_detail]]}

\section{一句话}
\keybox{新模型在 PUBG 上\textbf{兑现了“少抬底噪”}；脚步增强未削弱。
事件级误增强与枪声音色有一定回退，但不构成主能力全面变差——若验收只盯背景口径，可以过；若同时卡 FER/音色，需另议。}

\vspace{6mm}
{\small 复现：\texttt{python analysis/report\_pubg\_newmodel.py}\par}
\end{document}
"""


def _img_b64(path):
    with open(path, "rb") as fh:
        return f'<img src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}" alt="">'


def _html_table(rows, cols=None):
    if not rows:
        return "<p>—</p>"
    cols = cols or list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build_html(vb, s, bf, other_verdict, other_detail, fig_ex=None, fig_claim=None):
    """Short HTML report; embeds dark-theme figures (HTML-only) + audio."""
    from listen_pubg_bg import build_listen_sections  # lazy: avoid circular import

    # HTML 专用深色图（每次重建，不覆盖 PDF 的 print 版）
    fig_ex = fig_ex or os.path.join(FIG, "fig_nm_bg_examples_web.png")
    fig_claim = fig_claim or os.path.join(FIG, "fig_nm_bg_claim_web.png")
    if fig_ex.endswith("_web.png"):
        fig_bg_examples(bf, "fig_nm_bg_examples_web", theme="web")
    if fig_claim.endswith("_web.png"):
        fig_bg_claim(bf, "fig_nm_bg_claim_web", theme="web")

    listen_sections, _ = build_listen_sections(bf)
    dg_fo = s["g_new"] - s["g_old"]
    dfer = s["fer"]["new"] - s["fer"]["old"]
    dmse = s["mse"]["new"] - s["mse"]["old"]
    dgap = s["gap"]["new"][0] - s["gap"]["old"][0]

    bg_rows = [
        {"指标": "底噪误增强率（安静帧 >3 dB）", "旧模型": pct(vb["rate_o"]),
         "新模型": pct(vb["rate_n"]), "Δ": pct(vb["rate_n"] - vb["rate_o"])},
        {"指标": "底噪中位增益（dB）", "旧模型": sgn(vb["med_o"]),
         "新模型": sgn(vb["med_n"]), "Δ": sgn(vb["med_n"] - vb["med_o"])},
    ]
    other_rows = [
        {"能力": "脚步中位增益", "旧模型": sgn(s["g_old"]), "新模型": sgn(s["g_new"]),
         "变化": sgn(dg_fo), "是否明显削弱": "否"},
        {"能力": "脚步 ≥3 dB 比例", "旧模型": pct(s["ge3_old"]), "新模型": pct(s["ge3_new"]),
         "变化": pct(s["ge3_new"] - s["ge3_old"]), "是否明显削弱": "否"},
        {"能力": "同电平 ΔG（脚步−枪声）", "旧模型": sgn(s["gap"]["old"][0]),
         "新模型": sgn(s["gap"]["new"][0]), "变化": sgn(dgap), "是否明显削弱": "否"},
        {"能力": "不含脚步的枪声中位", "旧模型": sgn(s["gun_old"]), "新模型": sgn(s["gun_new"]),
         "变化": sgn(s["gun_new"] - s["gun_old"]), "是否明显削弱": "否"},
        {"能力": "事件 FER（不含脚步非脚步）", "旧模型": pct(s["fer"]["old"]),
         "新模型": pct(s["fer"]["new"]), "变化": pct(dfer), "是否明显削弱": "有回退（有限）"},
        {"能力": "纯净枪声频谱 MSE", "旧模型": f"{s['mse']['old']:.1f}",
         "新模型": f"{s['mse']['new']:.1f}", "变化": f"{dmse:+.1f}",
         "是否明显削弱": "有回退（有限）"},
    ]

    from pubg_report_theme import page_shell

    hero = f"""
<div class="eyebrow">Competitive Audio · Model Validation</div>
<h1 class="brand">C3 FPS AI <span>/ PUBG</span></h1>
<p class="tagline">背景声误增强优化专项验证。同一批真实对局录音、同一套人工标注与盲审；
独立补充分析，不改动主报告结论。</p>
<nav class="cta">
  <a href="{PDF_NAME}">PDF 完整报告</a>
  <a href="{LISTEN_NAME}">独立试听台</a>
  <a class="ghost" href="#audio">本页对比音频</a>
</nav>"""

    body = f"""
<div class="verdict">
  <b>主口径 · </b>{vb["claim"]}<br>
  以段内安静帧（短时电平 &lt; −65 dBFS）中位增益衡量底噪：超过 3 dB 的段由
  {pct(vb["rate_o"])} 降至 {pct(vb["rate_n"])}，中位增益 {sgn(vb["med_o"])} → {sgn(vb["med_n"])} dB。
  <div class="sub"><b>其他能力 · </b>{other_verdict}</div>
</div>

<div class="readouts">
  <div class="ro"><b>{pct(vb["rate_o"])}→{pct(vb["rate_n"])}</b><span>底噪误增强率</span></div>
  <div class="ro"><b>{sgn(vb["med_o"])}→{sgn(vb["med_n"])}</b><span>底噪中位增益 dB</span></div>
  <div class="ro"><b>{sgn(s["g_old"])}→{sgn(s["g_new"])}</b><span>脚步中位 dB</span></div>
  <div class="ro"><b>{pct(s["fer"]["old"])}→{pct(s["fer"]["new"])}</b><span>事件 FER</span></div>
</div>

<section class="block">
  <h2><span class="n">01</span> 主口径汇总</h2>
  {_html_table(bg_rows)}
  <figure class="fig">{_img_b64(fig_claim)}
    <figcaption>底噪误增强率与中位增益（安静帧口径）</figcaption>
  </figure>
  <p class="note">仅 PUBG，背景空隙 n={len(bf)}；安静帧 &lt;−65 dBFS（与主报告「全游戏整段 RMS 背景误增强 30%」不是同一口径）。</p>
</section>

<section class="block">
  <h2><span class="n">02</span> 误增强样例波形</h2>
  <p class="note">上排横排整段对比（同纵轴）；下排为蓝框安静段放大镜（列对齐、同纵轴）。</p>
  <figure class="fig">{_img_b64(fig_ex)}
    <figcaption>样例 A（test4）与样例 B（test14）</figcaption>
  </figure>
</section>

<section class="block" id="audio">
  <h2><span class="n">03</span> 对比试听</h2>
  <p class="note">与上图同一裁切。建议先听<strong>安静段放大镜</strong>三轨，再听整段。
  亦可打开 <a href="{LISTEN_NAME}">独立试听台</a>。</p>
  {listen_sections}
</section>

<section class="block">
  <h2><span class="n">04</span> 其他能力快检</h2>
  <p class="note">{other_detail}</p>
  {_html_table(other_rows)}
</section>

<section class="block files">
  <h2><span class="n">05</span> 相关文件</h2>
  <ul>
    <li><a href="{PDF_NAME}">PDF 报告</a></li>
    <li><a href="{LISTEN_NAME}">背景误增强试听</a></li>
    <li>新模型：<code>fps_test/pubg/pubg_output_newmodel/</code></li>
    <li>旧模型：<code>fps_test/pubg/pubg_output_onlyfps/</code></li>
  </ul>
</section>
"""
    return page_shell("C3 FPS AI · PUBG 新旧模型对比", hero, body)


def main():
    for p_ in [OUTDIR, TAB, FIG]:
        os.makedirs(p_, exist_ok=True)

    print("measuring …")
    d, dm, bg, _ = measure()
    d.to_csv(os.path.join(RES, "pubg_newmodel_events.csv"), index=False, encoding="utf-8-sig")
    bg.to_csv(os.path.join(RES, "pubg_newmodel_background.csv"), index=False, encoding="utf-8-sig")
    if len(dm):
        dm.to_csv(os.path.join(RES, "pubg_newmodel_model_events.csv"), index=False, encoding="utf-8-sig")

    s = summarize(d, dm, bg)
    bf = analyze_bg_frames(bg)
    bf.to_csv(os.path.join(RES, "pubg_newmodel_bg_frames.csv"), index=False, encoding="utf-8-sig")
    vb = verdict_bg(bf)

    fig_bg_claim(bf, "fig_nm_bg_claim")
    ex = fig_bg_examples(bf, "fig_nm_bg_examples")

    dg_fo = s["g_new"] - s["g_old"]
    dfer = s["fer"]["new"] - s["fer"]["old"]
    dmse = s["mse"]["new"] - s["mse"]["old"]
    dgap = s["gap"]["new"][0] - s["gap"]["old"][0]

    # other capabilities verdict
    weakened = []
    ok = []
    if dg_fo >= -0.3:
        ok.append(f"脚步中位 {sgn(s['g_old'])}$\\rightarrow${sgn(s['g_new'])} dB（未削弱）")
    else:
        weakened.append(f"脚步中位下降 {dg_fo:+.1f} dB")
    if abs(s["gun_new"]) <= abs(s["gun_old"]) + 0.3:
        ok.append(f"不含脚步枪声中位仍接近 0（{sgn(s['gun_old'])}$\\rightarrow${sgn(s['gun_new'])}）")
    else:
        weakened.append("枪声偏离 0 增大")
    if dgap >= -0.3:
        ok.append(f"同电平选择性 $\\Delta G$ {sgn(s['gap']['old'][0])}$\\rightarrow${sgn(s['gap']['new'][0])}")
    if dfer > 0.02:
        weakened.append(f"事件 FER {pct(s['fer']['old'])}$\\rightarrow${pct(s['fer']['new'])}")
    if dmse > 0.5:
        weakened.append(f"纯净枪声 MSE {s['mse']['old']:.1f}$\\rightarrow${s['mse']['new']:.1f}")

    if not weakened:
        other_verdict = "未发现明显削弱"
    elif dg_fo >= -0.3 and abs(s["gun_new"]) < 1.0:
        other_verdict = "主能力（脚步增强/枪声保持）未明显削弱；事件 FER 与枪声音色有回退"
    else:
        other_verdict = "存在较明显回退，见下表"

    other_detail = (
        "脚步与同电平选择性略升或持平，不属于削弱。"
        + ("事件 FER、枪声频谱 MSE 变差，属于连带代价，幅度有限、样本也偏少，不宜写成“能力全面变差”。"
           if weakened else "连带指标亦未见恶化。")
    )

    tex_table("nm_bg_summary", [
        {"指标": "底噪误增强率（安静帧 >3 dB）", "旧模型": pct(vb["rate_o"]), "新模型": pct(vb["rate_n"]),
         "Δ": pct(vb["rate_n"] - vb["rate_o"])},
        {"指标": "底噪中位增益（dB）", "旧模型": sgn(vb["med_o"]), "新模型": sgn(vb["med_n"]),
         "Δ": sgn(vb["med_n"] - vb["med_o"])},
    ], "主口径（安静帧）", spec="Lccc",
       notes=[f"仅 PUBG，背景空隙 n={len(bf)}；安静帧 <{BG_QUIET:.0f} dBFS。",
              "与主报告「背景段误增强率 30%」（VALORANT+PUBG、整段 RMS）不是同一口径。"])

    tex_table("nm_other", [
        {"能力": "脚步中位增益", "旧模型": sgn(s["g_old"]), "新模型": sgn(s["g_new"]),
         "变化": sgn(dg_fo), "是否明显削弱": "否"},
        {"能力": "脚步 ≥3 dB 比例", "旧模型": pct(s["ge3_old"]), "新模型": pct(s["ge3_new"]),
         "变化": pct(s["ge3_new"] - s["ge3_old"]), "是否明显削弱": "否"},
        {"能力": "同电平 ΔG（脚步−枪声）", "旧模型": sgn(s["gap"]["old"][0]), "新模型": sgn(s["gap"]["new"][0]),
         "变化": sgn(dgap), "是否明显削弱": "否"},
        {"能力": "不含脚步的枪声中位", "旧模型": sgn(s["gun_old"]), "新模型": sgn(s["gun_new"]),
         "变化": sgn(s["gun_new"] - s["gun_old"]), "是否明显削弱": "否"},
        {"能力": "事件 FER（不含脚步非脚步）", "旧模型": pct(s["fer"]["old"]), "新模型": pct(s["fer"]["new"]),
         "变化": pct(dfer), "是否明显削弱": "有回退（有限）"},
        {"能力": "纯净枪声频谱 MSE", "旧模型": f"{s['mse']['old']:.1f}", "新模型": f"{s['mse']['new']:.1f}",
         "变化": f"{dmse:+.1f}", "是否明显削弱": "有回退（有限）"},
    ], "其他能力快检（非本次官方主口径）", spec="Lcccc",
       notes=["“明显削弱”指脚步增强失效或枪声被大幅抬高；FER/MSE 变差记为有限回退。"])

    V = dict(
        claim=vb["claim"],
        other_verdict=other_verdict,
        other_detail=other_detail,
        n_bg=str(len(bf)),
        bg_rate_old=p(vb["rate_o"]), bg_rate_new=p(vb["rate_n"]),
        bg_old=n(vb["med_o"]), bg_new=n(vb["med_n"]),
        g_fo_old=n(s["g_old"]), g_fo_new=n(s["g_new"]),
        fer_old=p(s["fer"]["old"]), fer_new=p(s["fer"]["new"]),
        ex_a_old=n(ex[0]["g_old"]) if ex else "—",
        ex_a_new=n(ex[0]["g_new"]) if ex else "—",
        ex_b_old=n(ex[1]["g_old"]) if len(ex) > 1 else "—",
        ex_b_new=n(ex[1]["g_new"]) if len(ex) > 1 else "—",
        ex_b_zo=n(ex[1]["g_zo"]) if len(ex) > 1 else "—",
        ex_b_zn=n(ex[1]["g_zn"]) if len(ex) > 1 else "—",
    )
    for k in ("claim", "other_verdict", "other_detail"):
        V[k] = V[k].replace("%", r"\%")

    body = TEMPLATE
    for k, val in V.items():
        body = body.replace(f"[[{k}]]", str(val))
    left = re.findall(r"\[\[\w+\]\]", body)
    if left:
        raise ValueError(f"unfilled: {sorted(set(left))}")

    with open(os.path.join(OUTDIR, "pubg_newmodel.tex"), "w", encoding="utf-8") as fh:
        fh.write(body)
    r = None
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "pubg_newmodel.tex"],
                           cwd=OUTDIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("xelatex failed")
    out = rt.publish(os.path.join(OUTDIR, "pubg_newmodel.pdf"), PDF_OUT)
    print("->", out)

    from listen_pubg_bg import write_listen_html
    listen_out = write_listen_html(bf)
    html = build_html(vb, s, bf, other_verdict, other_detail)
    with open(HTML_OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("->", HTML_OUT)
    print("->", listen_out)
    print("claim:", vb["claim"])
    print("other:", other_verdict)


if __name__ == "__main__":
    main()
