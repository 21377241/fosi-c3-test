"""LaTeX experiment report (ctex + booktabs), compiled with xelatex.

All tables are generated from the analysis results; every figure has a slot that picks up
figures/<name>.pdf (final, made from the figure spec) or figures/<name>.png (draft) or shows a placeholder.
Output: docs/report_tex/ (main.tex, tables/, figures/, figures/data/) and docs/C3 FPS AI 声卡客观评测 实验报告（详细版）.pdf
The short main report is report_brief.py.
"""
import os
import re
import html
import shutil
import subprocess
import numpy as np
import pandas as pd
import report_final as rf
from report_final import pct, sgn, RES, HERE, FOOT, NT, CLAIM, FER_THR, P_HI, EQ_Q, SHORT
from report_ad import MATCH_DB, MATCH_MIN
import paper
import matplotlib.pyplot as plt

DOCS = os.path.join(os.path.dirname(HERE), "docs")
OUTDIR = os.path.join(DOCS, "report_tex")
TAB, FIG = os.path.join(OUTDIR, "tables"), os.path.join(OUTDIR, "figures")
DATA = os.path.join(FIG, "data")
PDF_OUT = os.path.join(DOCS, "C3 FPS AI 声卡客观评测 技术报告（详细版）.pdf")
rng = np.random.default_rng(2026)

SYM = [("≥", r"$\geq$"), ("≤", r"$\leq$"), ("±", r"$\pm$"), ("×", r"$\times$"), ("→", r"$\rightarrow$"),
       ("Δ", r"$\Delta$"), ("κ", r"$\kappa$"), ("²", r"$^{2}$"), ("−", r"$-$"), ("·", r"$\cdot$"),
       ("…", r"\dots{}"), ("~", r"$\sim$"), ("–", "--"), ("⁷", r"$^{7}$")]


def symfix(s):
    """Only replace unicode symbols (for captions / notes that already contain LaTeX)."""
    for a, b in SYM:
        s = s.replace(a, b)
    return re.sub(r"(?<!\\)%", r"\\%", s)


def esc(s):
    """HTML-ish cell text -> LaTeX."""
    s = html.unescape(re.sub(r"</?b>", "", str(s)))
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    s = re.sub(r"(?<![\w.])-(?=\d)", "\x00", s)
    s = re.sub(r"(?<![\w.])\+(?=\d)", "\x01", s)
    s = s.replace("<", "\x02").replace(">", "\x03")
    for a, b in SYM:
        s = s.replace(a, b)
    return s.replace("\x00", "$-$").replace("\x01", "$+$").replace("\x02", "$<$").replace("\x03", "$>$")


def n(x, nd=1, sign=True):
    """Number for running text, with a proper minus sign."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return "$" + s.replace("-", "-") + "$"


def p(x):
    return pct(x).replace("%", r"\%")


class Raw(str):
    """Table cell that is already LaTeX."""


def stack(s, sep):
    """Break a cell into two lines at the first `sep` (kept on the second line)."""
    a, b = (s.split(sep, 1) + [None])[:2]
    return Raw(esc(a)) if b is None else Raw(r"\makecell{" + esc(a) + r"\\" + esc(sep + b) + "}")


def tex_table(name, rows, caption, notes=None, spec=None, size=r"\small", heads=None):
    cols = list(rows[0])
    if spec is None:
        spec = "l" + "".join("L" if max(len(str(r[c])) for r in rows) > 22 else "c" for c in cols[1:])
    body = [" & ".join(heads or [esc(c) for c in cols]) + r" \\", r"\midrule"]
    body += [" & ".join(r[c] if isinstance(r[c], Raw) else esc(r[c]) for c in cols) + r" \\" for r in rows]
    env = "tabularx" if ("L" in spec or "X" in spec) else "tabular"
    width = r"{\linewidth}" if env == "tabularx" else ""
    caption = symfix(caption)
    note = ""
    if notes:
        note = r"\begin{tablenotes}\footnotesize" + "".join(rf"\item {symfix(t)}" for t in notes) + r"\end{tablenotes}"
    tex = (r"\begin{table}[htbp]\centering" "\n" rf"\caption{{{caption}}}\label{{tab:{name}}}" "\n"
           rf"{{{size}\begin{{threeparttable}}" "\n" rf"\begin{{{env}}}{width}{{{spec}}}\toprule" "\n"
           + "\n".join(body) + "\n" rf"\bottomrule\end{{{env}}}" "\n" + note + r"\end{threeparttable}}" "\n" r"\end{table}" "\n")
    with open(os.path.join(TAB, f"{name}.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)


# ---------------------------------------------------------------- statistics not in report_final
def matched_gap_np(fl, fg, nl, ng):
    """Vectorised level-matched gap: median over non-target events of (median footstep gain within ±MATCH_DB) - own gain."""
    diffs = []
    for l, g in zip(nl, ng):
        near = fg[np.abs(fl - l) <= MATCH_DB]
        if len(near) >= MATCH_MIN:
            diffs.append(np.median(near) - g)
    return (np.median(diffs) if diffs else np.nan), len(diffs)


def gap_ci(x, reps=1000):
    """Level-matched gunshot gap with clip-cluster bootstrap CI."""
    f, g = x[x.group == FOOT], x[x.group == NT[0]]
    est = matched_gap_np(f.raw_level.values, f.G_AI.values, g.raw_level.values, g.G_AI.values)
    clips = x.clip_key.unique()
    by = {k: x[x.clip_key == k] for k in clips}
    vals = []
    for _ in range(reps):
        s = pd.concat([by[k] for k in rng.choice(clips, len(clips))])
        fs, gs = s[s.group == FOOT], s[s.group == NT[0]]
        v = matched_gap_np(fs.raw_level.values, fs.G_AI.values, gs.raw_level.values, gs.G_AI.values)[0]
        if not np.isnan(v):
            vals.append(v)
    return est[0], est[1], np.percentile(vals, [2.5, 97.5])


def fer_curve(d, bg, thr):
    out = []
    for t in thr:
        r = {"threshold_db": t}
        for pfx, x in [("nonfoot", d[d.group.isin(NT)]), ("background", bg)]:
            for k in ["AI", "EQ", "EQm"]:
                r[f"{pfx}_{k}"] = float((x[f"G_{k}"] > t).mean())
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures (drafts + data for the designer)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.unicode_minus": True})
C_AI, C_EQ, C_EQM = "#1F4E79", "#8C8C8C", "#E69F00"
C_GRP = {FOOT: "#4C9BE8", NT[0]: "#D62728", NT[1]: "#FF7F0E", NT[2]: "#2CA02C"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def draft_forest(rows, name):
    """First row is the main analysis (highlighted); hollow markers where the 95% CI crosses 0."""
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 0.9))
    ys = np.arange(len(rows))[::-1]
    lo_x = min(0, min(r["lo"] for r in rows)) - 1
    hi_x = max(r["hi"] for r in rows) + 1
    for k, (y, r) in enumerate(zip(ys, rows)):
        if k % 2 == 0:
            ax.axhspan(y - .5, y + .5, color="#f2f2f2", zorder=0)
        main = k == 0
        col = C_AI if main else "#4a6f94"
        cross = r["lo"] <= 0
        ax.errorbar(r["est"], y, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="D" if main else "o",
                    c=col, mfc="w" if cross else col, mec=col, ms=7 if main else 5.5, capsize=3,
                    lw=2 if main else 1.3, zorder=3)
        txt = f"{r['est']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]".replace("-", "−")
        ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=8.5,
                fontweight="bold" if main else "normal")
        ax.text(1.30, y, f"n={int(r['n_matched'])}", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color="#555")
    top = ys[0] + .5
    ax.text(1.02, top + .15, "ΔG [95% CI]", transform=ax.get_yaxis_transform(), fontsize=8.5, color="#555", va="bottom")
    ax.text(1.30, top + .15, "配对数", transform=ax.get_yaxis_transform(), fontsize=8.5, color="#555", va="bottom")
    ax.axvline(0, c="k", lw=1.3, zorder=1)
    ax.text(-.2, top + .15, "0 dB（H4 方向参考）", fontsize=8.5, ha="right", va="bottom")
    ax.axvline(3, c="#999999", ls="--", lw=.9, zorder=1)
    ax.text(3.2, top + .15, "3 dB（H2 效果参考）", color="#777777", fontsize=8, ha="left", va="bottom")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(ys[-1] - .5, top)
    ax.set_yticks(ys, [r["name"] for r in rows])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("同电平增益差 ΔG（枪声，dB）；空心点表示 95% CI 跨过 0")
    ax.grid(axis="x", alpha=.3, zorder=0)
    save(fig, name)


def fig_fer_curve(fc, name):
    """fc: fer_curve() output. Two panels (non-footstep events / background), legend outside on top."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    series = [("AI", "FPS AI", C_AI, "-"), ("EQ", "实际 EQ", C_EQ, "-"), ("EQm", "匹配静态 EQ（辅助对照）", C_EQM, "--")]
    for ax, pfx, title in [(axes[0], "nonfoot", "非脚步事件（枪声 / 换弹 / 道具）"), (axes[1], "background", "背景段（无事件的持续背景）")]:
        for k, lab, col, ls in series:
            ax.plot(fc.threshold_db, fc[f"{pfx}_{k}"] * 100, c=col, ls=ls, lw=1.8, marker="o", ms=2.5, label=lab)
        ax.axvline(FER_THR, c="k", ls=":", lw=.9)
        ax.text(FER_THR + .2, 60, f"本报告阈值 {FER_THR:.0f} dB", fontsize=8, va="center")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("误增强判定阈值（dB）")
        ax.set_xlim(fc.threshold_db.min(), fc.threshold_db.max())
        ax.set_ylim(0, 100)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("误增强率 FER（%）")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    save(fig, name)


_EQM_FIR = {}


def eqm_fir(sr):
    """Matched static EQ as the same 2049-tap linear-phase FIR, rebuilt from ad_response.csv (offset included)."""
    if sr not in _EQM_FIR:
        from scipy.signal import firwin2
        r = pd.read_csv(os.path.join(RES, "ad_response.csv"))
        f = np.linspace(0, sr / 2, 4097)
        _EQM_FIR[sr] = firwin2(2049, f / (sr / 2), 10 ** (np.interp(f, r.freq_hz, r.gain_db) / 20))
    return _EQM_FIR[sr]


def load_track(cls, eid, t):
    """Mono event clip for track t in raw / fps / eq / eqm (eqm = raw through the matched static EQ)."""
    from scipy.io import wavfile
    from scipy.signal import fftconvolve
    sr, x = wavfile.read(os.path.join(HERE, "clean_events", cls, f"e{eid:04d}_{'raw' if t == 'eqm' else t}.wav"))
    x = x.astype(np.float64)
    x = x.mean(1) if x.ndim > 1 else x
    if t == "eqm":
        x = fftconvolve(x, eqm_fir(sr), mode="same")
    return x, sr


def draft_spectrograms(examples, name, gains=None):
    """Raw spectrogram plus difference spectrograms (processed - raw) for FPS AI / actual EQ / matched EQ.
    gains: optional {event_id: (G_AI, G_EQ, G_EQm)} printed in the difference panels."""
    from scipy.signal import spectrogram
    from scipy.ndimage import uniform_filter
    from matplotlib.colors import ListedColormap
    cols = [("fps", "FPS AI − 原始", C_AI), ("eq", "实际 EQ − 原始", C_EQ), ("eqm", "匹配静态 EQ − 原始", C_EQM)]
    fig, axes = plt.subplots(len(examples), 4, figsize=(9.6, 2.4 * len(examples)), sharey=True,
                             constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    for i, (title, cls, eid) in enumerate(examples):
        S = {}
        for t in ["raw"] + [c[0] for c in cols]:
            x, sr = load_track(cls, eid, t)
            f, tt, Z = spectrogram(x, sr, nperseg=1024, noverlap=1024 - 240)
            S[t] = 10 * np.log10(Z + 1e-14)
        vmax = S["raw"].max()
        low = uniform_filter(S["raw"], size=3, mode="nearest") < vmax - 60
        im0 = axes[i, 0].pcolormesh(tt, f / 1000, S["raw"] - vmax, vmin=-80, vmax=0, cmap="magma", shading="auto")
        for j, (t, lab, col) in enumerate(cols, start=1):
            ax = axes[i, j]
            dd = np.ma.masked_array(uniform_filter(S[t] - S["raw"], size=3, mode="nearest"), mask=low)
            im1 = ax.pcolormesh(tt, f / 1000, dd, vmin=-15, vmax=15, cmap=cmap, shading="auto")
            if gains:
                ax.text(0.97, 0.93, f"{gains[eid][j - 1]:+.1f} dB".replace("-", "−"), transform=ax.transAxes,
                        ha="right", va="top", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=.85))
            if i == 0:
                ax.set_title(lab, fontsize=10, color=col, fontweight="bold")
        for j in range(4):
            ax = axes[i, j]
            ax.set_xlim(tt[0], tt[-1])
            ax.set_ylim(0, 12)
            if i == len(examples) - 1:
                ax.set_xlabel("时间 (s)")
        axes[i, 0].set_ylabel(f"{title}\n频率 (kHz)")
        if i == 0:
            axes[i, 0].set_title("原始（声谱）", fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[:, 0], location="bottom", shrink=.9, aspect=18, label="相对峰值（dB）")
    cb = fig.colorbar(im1, ax=axes[:, 1:], location="bottom", shrink=.5, aspect=30, label="相对原始（dB）")
    cb.set_ticks([-15, -10, -5, 0, 5, 10, 15])
    cb.set_ticklabels(["−15", "−10", "−5", "0", "+5", "+10", "+15"])
    save(fig, name)


def copy_draft(src, name):
    shutil.copy(src, os.path.join(FIG, f"{name}.png"))


def fig_waveforms(examples, gains, name, with_eq=True, with_eqm=False):
    """Rows = examples, columns = raw / FPS AI / EQ (/ matched EQ); same y scale within a row so level changes are visible.
    gains[eid] = (G_AI, G_EQ[, G_EQm])."""
    tracks = [("raw", "原始", "#333333"), ("fps", "FPS AI", C_AI), ("eq", "实际 EQ", C_EQ)][:3 if with_eq else 2]
    if with_eqm:
        tracks.append(("eqm", "匹配静态 EQ（辅助对照）", C_EQM))
    cw = 2.87 if len(tracks) <= 3 else 2.45
    fig, axes = plt.subplots(len(examples), len(tracks), figsize=(cw * len(tracks), 2.0 * len(examples)), sharey="row")
    for i, (title, cls, eid) in enumerate(examples):
        W = {t: load_track(cls, eid, t) for t, _, _ in tracks}
        lim = max(np.abs(w).max() for w, _ in W.values()) * 1.1
        for j, (t, lab, col) in enumerate(tracks):
            ax = axes[i, j]
            w, sr = W[t]
            ax.plot(np.arange(len(w)) / sr * 1000, w, c=col, lw=.5)
            ax.set_ylim(-lim, lim)
            ax.axhline(0, c="#ccc", lw=.4)
            if i == 0:
                ax.set_title(lab, fontsize=10, color=col, fontweight="bold")
            if j == 0:
                ax.set_ylabel(title, fontsize=10)
            ax.set_yticks([])
            if i == len(examples) - 1:
                ax.set_xlabel("时间 (ms)")
            if j > 0:
                g = gains[eid][j - 1]
                ax.text(0.97, 0.92, f"{g:+.1f} dB（幅值 ×{10 ** (g / 20):.1f}）".replace("-", "−"), transform=ax.transAxes, ha="right", va="top",
                        fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=.85))
    plt.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- labelling accuracy before / after
def zone(y, lab):
    """y: 1 = footstep (human); lab: 'f' / 'n' / 'a' (ambiguous) decisions of a method."""
    dec = lab != "a"
    ok = ((lab == "f") & (y == 1)) | ((lab == "n") & (y == 0))
    return dict(decided=dec.mean(), correct=ok[dec].mean(), wrong=(dec & ~ok).mean(),
                rec_f=(lab[y == 1] == "f").mean(), rec_n=(lab[y == 0] == "n").mean(),
                f_ok=int(((lab == "f") & (y == 1)).sum()), f_a=int(((lab == "a") & (y == 1)).sum()), f_bad=int(((lab == "n") & (y == 1)).sum()),
                n_ok=int(((lab == "n") & (y == 0)).sum()), n_a=int(((lab == "a") & (y == 0)).sum()), n_bad=int(((lab == "f") & (y == 0)).sum()))


def three(pv):
    return np.where(pv >= P_HI, "f", np.where(pv <= 1 - P_HI, "n", "a"))


def label_accuracy():
    """Stages: zero-shot baselines, then the labeller after rounds 1..4, all scored on the round-1 random sample."""
    from sklearn.metrics import roc_auc_score
    path = os.path.join(RES, "label_curve_probs.csv")
    if not os.path.exists(path):
        import label_curve
        label_curve.main()
    pr = pd.read_csv(path, encoding="utf-8-sig")
    curve = pd.read_csv(os.path.join(RES, "label_curve.csv"), encoding="utf-8-sig")
    y = pr.y.values
    v1 = pd.read_csv(os.path.join(RES, "events_labeled.csv"))
    v1["event_id"] = np.arange(len(v1))
    v1 = v1.set_index("event_id").loc[pr.event_id]
    lab_v1 = v1.label.map({"footstep": "f", "non-footstep": "n"}).fillna("a").values
    stages = [("现成模型 + 规则", "标注前", lab_v1, v1.p_foot_broad.values, 0),
              ("CLAP 文本零样本", "标注前", three(pr["CLAP 文本提示（零样本）"].values), pr["CLAP 文本提示（零样本）"].values, 0)]
    for k in range(1, 5):
        stages.append((f"第 1–{k} 轮后" if k > 1 else "第 1 轮后", "标注后", three(pr[f"round{k}"].values), pr[f"round{k}"].values,
                       int(curve.n_train.iloc[1 + k])))
    rows = [dict(stage=nm, phase=ph, n_train=ntr, auc=roc_auc_score(y, sc), **zone(y, lab)) for nm, ph, lab, sc, ntr in stages]
    return pd.DataFrame(rows), len(y), int(y.sum())


def label_composition(ev, v1):
    """Counts over all events. before: automatic labels used before annotation; after: human labels first, final model elsewhere."""
    h = ev.human_label != ""
    cnt = lambda lab, m: int(((ev.label == lab) & m).sum())
    before = {"f_m": int((v1.label == "footstep").sum()), "n_m": int((v1.label == "non-footstep").sum()),
              "a_m": int((v1.label == "ambiguous").sum()), "f_h": 0, "n_h": 0, "a_h": 0}
    after = {"f_h": cnt("footstep", h), "n_h": cnt("non-footstep", h), "a_h": cnt("ambiguous", h),
             "f_m": cnt("footstep", ~h), "n_m": cnt("non-footstep", ~h), "a_m": cnt("ambiguous", ~h)}
    return before, after


def fig_label_accuracy(comp, S, n_eval, name):
    """(a) labels over all events before / after annotation; (b) reliability of the decisions on the round-1 random sample."""
    from matplotlib.patches import Patch
    before, after = comp
    tot = sum(before.values())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1.7, 1]})
    segs = [("f_h", "#1F5FA8"), ("f_m", "#4C9BE8"), ("n_h", "#A31D1D"), ("n_m", "#E06666"), ("a_h", "#8C8C8C"), ("a_m", "#C8C8C8")]
    for yy, comp_ in [(1, before), (0, after)]:
        left = 0
        for k, col in segs:
            w = comp_[k] / tot * 100
            if w == 0:
                continue
            a.barh(yy, w, left=left, color=col, height=0.55, edgecolor="w")
            if w >= 5:
                a.text(left + w / 2, yy, f"{w:.0f}%", ha="center", va="center", fontsize=8.5, color="w" if k[0] != "a" else "#333")
            left += w
    amb = (after["a_h"] + after["a_m"]) / tot * 100
    a.annotate(f"判不出 {after['a_h'] + after['a_m']} 个（{amb:.1f}%）", xy=(100 - amb / 2, -0.3), xytext=(100 - amb / 2, -0.62),
               ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", lw=.6))
    a.set_yticks([1, 0], ["标注前\n（现成模型 + 规则）", "标注后\n（人工标注 + 最终模型）"], fontsize=8.5)
    a.set_ylim(-0.8, 1.45)
    a.set_xlim(0, 100)
    a.set_xlabel(f"占全部 {tot} 个事件的比例（%）")
    a.set_title(f"(a) 全部 {tot} 个事件的判别结果", fontsize=9.5, pad=24)
    a.legend(handles=[Patch(color="#1F5FA8", label="脚步（人工）"), Patch(color="#4C9BE8", label="脚步（模型）"),
                      Patch(color="#A31D1D", label="非脚步（人工）"), Patch(color="#E06666", label="非脚步（模型）"),
                      Patch(color="#C8C8C8", label="判不出")],
             ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7.5, handlelength=1.2, columnspacing=0.8)
    b0, a4 = S.iloc[0], S.iloc[-1]
    x = np.arange(2)
    for k, (lab, col, vals) in enumerate([("标注前", "#BBBBBB", [b0.correct * 100, b0.auc * 100]),
                                          ("标注后", C_AI, [a4.correct * 100, a4.auc * 100])]):
        bb = b.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, label=lab)
        for rect, v in zip(bb, vals):
            b.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
    b.set_xticks(x, ["给出判断时的\n正确率（%）", "区分能力\nAUC（×100）"], fontsize=8.5)
    b.set_ylim(0, 112)
    b.set_title(f"(b) 判定可靠性（第 1 轮随机样本 {n_eval} 条）", fontsize=9.5, pad=24)
    b.legend(fontsize=8, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    b.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- main
def main():
    for p_ in [OUTDIR, TAB, FIG, DATA]:
        os.makedirs(p_, exist_ok=True)
    c = rf.main()
    d, bg, ev, meta, dm = c["d"], c["bg"], c["ev"], c["meta"], c["dm"]
    fo_h, fo_all, nt_h = c["fo_h"], c["fo_all"], c["nt_h"]
    med, gap, sei, mse = c["med"], c["gap"], c["sei"], c["mse"]
    fer, fer_g, fer_bg = c["fer"], c["fer_g"], c["fer_bg"]
    rs, acc, rob, sp, prom = c["rs"], c["acc"], c["rob"], c["sp"], c["prom"]
    r2, r3, r4, n_r1 = c["r2"], c["r3"], c["r4"], c["n_r1"]
    ci_h, ci_all, eq, bs, clips = c["ci_h"], c["ci_all"], c["eq"], c["bs"], c["clips"]
    fa_g, gm = c["fa_g"], c["gm"]
    lvl_lo, lvl_mid, lvl_hi, hi_clip = c["lvl_lo"], c["lvl_mid"], c["lvl_hi"], c["hi_clip"]
    n_vl, n_pg, dur_min = c["n_vl"], c["n_pg"], c["dur_min"]
    gun = d[d.group == NT[0]]
    n_lab = n_r1 + len(r2) + len(r3) + len(r4)
    auc, n_auc_r1, n_auc_all, dims = paper.labeler_stats()
    mix = paper.round_mix()
    n_par = dims["AST"][1] + dims["CLAP"][1] + 2
    g_eqm = gap["EQm"][NT[0]][0]

    # ------------------------------------------------ tables
    tex_table("eq", [{"游戏": r["游戏"], **{k: v for k, v in r.items() if k != "游戏"}} for r in c["eq_rows"]],
              f"EQ 条件的各频段增益（dB），10 段 RBJ 峰值滤波器，$Q={EQ_Q}$", spec="l" + "c" * 10, size=r"\footnotesize")

    gl = ev.groupby("game")
    data_rows = []
    for g, name in [("valorant", "VALORANT"), ("pubg", "PUBG")]:
        e = ev[ev.game == g]
        data_rows.append({"游戏": name,
                          "片段数": int((clips.game == g).sum()), "总时长 (min)": f"{clips[clips.game == g].dur.sum() / 60:.1f}",
                          "候选事件": len(e), "人工标注": int((e.human_label != "").sum()),
                          "脚步": int((e.label == "footstep").sum()), "非脚步": int((e.label == "non-footstep").sum()),
                          "不确定": int((e.label == "ambiguous").sum())})
    data_rows.append({"游戏": "合计", "片段数": len(clips), "总时长 (min)": f"{dur_min:.1f}", "候选事件": len(ev),
                      "人工标注": int((ev.human_label != "").sum()), "脚步": int((ev.label == "footstep").sum()),
                      "非脚步": int((ev.label == "non-footstep").sum()), "不确定": int((ev.label == "ambiguous").sum())})
    tex_table("data", data_rows, "数据概况（VALORANT 与 PUBG 均为本阶段计划评测游戏）", spec="lccccccc",
              notes=["数据由品牌方提供，31 段全部使用，未按处理效果筛选。",
                     "脚步 / 非脚步 / 不确定为人工标注与标注模型合并后的最终标签；人工标注一律优先。"])

    plan = [("第 1 轮", n_r1, "按片段 × 电平分层随机", "初始训练集；无偏精度核对"),
            ("第 2 轮", len(r2), "两分支意见相反或集成概率居中（每段至多 6 个）", "主动学习，补充决策边界"),
            ("第 3 轮", len(r3), f"不确定事件按峰值分层 {len(c['r3_amb'])} 个 + 响亮高置信度脚步 {len(c['r3_loud'])} 个", "补齐响脚步 / 弱枪声"),
            ("第 4 轮", len(r4), "剩余不确定事件按峰值分两层随机", "补充剩余难例")]
    tex_table("rounds", [{"轮次": a, "条数": k, "抽样策略": s, "目的": u, "f": m["f"], "g": m["g"], "o": m["o"], "x": m["x"]}
                         for (a, k, s, u), m in zip(plan, mix)] +
              [{"轮次": "合计", "条数": n_lab, "抽样策略": "", "目的": "", **{k: sum(m[k] for m in mix) for k in "fgox"}}],
              "四轮人工标注的抽样策略与类别构成", spec="lcLLcccc",
              heads=["轮次", "条数", "抽样策略", "目的", "脚步", "枪声", "其他", "不确定"],
              notes=[f"表中为各轮首次标注结果；盲审复核后有 {rs['changed']} 条改判，最终人工标签统计见表 \\ref{{tab:label_ba}}。"])

    tex_table("quality", [
        {"指标": "盲审复核前后一致率", "数值": pct(rs["agree"]), "样本": f"第 1 轮 {rs['n']} 条", "说明": "隐藏原标注后重新盲听，两次判断相同的比例"},
        {"指标": "四类 Cohen κ", "数值": f"{rs['k5']:.2f}", "样本": f"第 1 轮 {rs['n']} 条", "说明": "扣除随机一致后的一致程度（脚步 / 枪声 / 其他 / 不确定）；> 0.8 为几乎完全一致"},
        {"指标": "脚步 vs 非脚步 κ", "数值": f"{rs['k2']:.2f}", "样本": "两次均为明确类别", "说明": "本报告关心的二分类上几乎完全一致"},
        {"指标": "盲审后改判", "数值": str(rs["changed"]), "样本": f"第 1 轮 {rs['n']} 条", "说明": "不一致条目复听裁定后修改的数量"},
        {"指标": "标注模型 AUC（留片段交叉验证）", "数值": f"{auc['cv_ens'][0]:.3f}", "样本": f"第 1 轮随机样本 {n_auc_r1} 条", "说明": "模型区分脚步与非脚步的能力；只用随机样本，无偏"},
        {"指标": "判为脚步的准确率", "数值": pct(acc["prec_f"]), "样本": f"{acc['n_f']} 个", "说明": f"模型高置信判脚步（p ≥ {P_HI}）中判对的比例"},
        {"指标": "判为非脚步的准确率", "数值": pct(acc["prec_n"]), "样本": f"{acc['n_n']} 个", "说明": f"模型高置信判非脚步（p ≤ {1 - P_HI:.1f}）中判对的比例"},
        {"指标": "最终排除的不确定事件", "数值": str(int((ev.label == "ambiguous").sum())), "样本": "全部 1620 个",
         "说明": f"占 {(ev.label == 'ambiguous').mean() * 100:.1f}%：模型拿不准 {c['n_amb_now']} + 人工不确定 {int((ev.human_label == 'x').sum())}"}],
        "标注质量指标", spec="lccL")

    tex_table("hyp", [
        {"假设": "H1 脚步增强有效", "对应宣称": "智能强化关键脚步声；脚步声细节增强 3–9 dB", "主要指标": "脚步增益 G 及其分布",
         "判定标准": f"人工确认脚步 G 中位数及 95% CI 下限均 > {CLAIM[0]:.0f} dB；另报告 G 落在 3–9 dB、> 9 dB、< 3 dB 的比例（描述性，不设通过线）"},
        {"假设": "H2 事件选择性（核心）", "对应宣称": "实时识别；不同于传统 EQ，只强化必要部分", "主要指标": "同电平 ΔG、匹配静态 EQ",
         "判定标准": "同电平枪声 ΔG ≥ 3 dB，且匹配静态 EQ 的 |ΔG| < 1 dB"},
        {"假设": "H3 非目标声音保持性", "对应宣称": "不干扰环境音或枪声，不改变其他声音特征", "主要指标": "频谱 MSE、FER",
         "判定标准": f"枪声频谱 MSE 低于实际 EQ，且非脚步 FER（> {FER_THR:.0f} dB）≤ 10%（本报告设定的工程容忍阈值，非行业标准）"},
        {"假设": "H4 稳定性", "对应宣称": "在目标游戏的复杂战场中表现一致", "主要指标": "分组重复 H2",
         "判定标准": "两款目标游戏、纯净度分档、峰值口径、20% 标签噪声下，选择性效应方向不变"}],
        "评测假设与判定标准", spec="lLlL")

    tex_table("threat", [
        {"潜在混杂或误差": "枪声比脚步响，处理可能只是“对响声增强少”", "控制手段": f"同电平比较（±{MATCH_DB:.0f} dB，至少 {MATCH_MIN} 个参照脚步）"},
        {"潜在混杂或误差": "选择性来自固定频响（脚步与枪声频谱不同）", "控制手段": "匹配静态 EQ：按 FPS AI 的脚步平均频响设计固定滤波器"},
        {"潜在混杂或误差": "标注错误或标注模型误判", "控制手段": "人工优先；盲审复核；标注噪声模拟；两套样本口径"},
        {"潜在混杂或误差": "标注模型利用电平，与同电平分析循环", "控制手段": "模型输入先做音量归一化"},
        {"潜在混杂或误差": "重叠声音污染增益测量", "控制手段": "纯净度三档分级，主分析排除“有重叠”"},
        {"潜在混杂或误差": "同片段事件相关，置信区间偏窄", "控制手段": "按片段聚类的自助法"},
        {"潜在混杂或误差": "增益窗口的选择", "控制手段": "区间 RMS 与起点后 80 ms 峰值两种口径对照"},
        {"潜在混杂或误差": "三轨未对齐或 EQ 轨含额外处理", "控制手段": "互相关对齐（0 样本偏移）；实测与理论 EQ 频响比对"}],
        "潜在混杂、误差来源与控制手段", spec="LL")

    tex_table("sample", c["cnt_rows"], "进入主分析的人工确认事件（按纯净度）", spec="lccc")

    t_a = [{"样本": "人工确认", "n": len(fo_h), "FPS AI 中位 [95% CI]": f"{sgn(med(fo_h, 'AI'))} [{sgn(ci_h[0])}, {sgn(ci_h[1])}]",
            "3–9 dB": pct(fo_h.G_AI.between(*CLAIM).mean()), "> 9 dB": pct((fo_h.G_AI > 9).mean()), "< 3 dB": pct((fo_h.G_AI < 3).mean()),
            "实际 EQ 中位": sgn(med(fo_h, "EQ"))},
           {"样本": "人工 + 模型高置信度", "n": len(fo_all), "FPS AI 中位 [95% CI]": f"{sgn(med(fo_all, 'AI'))} [{sgn(ci_all[0])}, {sgn(ci_all[1])}]",
            "3–9 dB": pct(fo_all.G_AI.between(*CLAIM).mean()), "> 9 dB": pct((fo_all.G_AI > 9).mean()), "< 3 dB": pct((fo_all.G_AI < 3).mean()),
            "实际 EQ 中位": sgn(med(fo_all, "EQ"))}]
    t_a += [{"样本": f"其中 {lab}", "n": len(x), "FPS AI 中位 [95% CI]": sgn(x.G_AI.median()), "3–9 dB": pct(x.G_AI.between(*CLAIM).mean()),
             "> 9 dB": pct((x.G_AI > 9).mean()), "< 3 dB": pct((x.G_AI < 3).mean()), "实际 EQ 中位": sgn(x.G_EQ.median())}
            for lab, x in [("< -40 dBFS", lvl_lo), ("-40 ~ -35 dBFS", lvl_mid), ("> -35 dBFS", lvl_hi)]]
    tex_table("gain_a", t_a, "脚步增益（dB）：总体与按原始电平分档", spec="lcccccc",
              notes=["电平分档基于“人工 + 模型高置信度”样本。", f"> $-35$ dBFS 的 {len(lvl_hi)} 个脚步来自 {len(hi_clip)} 个片段，增益主要由片段决定，不作稳定结论。"])

    tex_table("gain_cls", [{"类别": r["类别"], "n": r["n"], "原始电平中位": r["原始电平中位"], "FPS AI": r["FPS AI"],
                            "实际 EQ": r["EQ（实际）"], "匹配静态 EQ": r["匹配静态 EQ（模拟）"]} for r in c["gain_rows"]],
              "各类事件在三种处理下的中位增益（dB）", spec="lccccc", notes=["* 背景段距任何事件 ≥ 0.3 s、长 ≥ 1 s，未经人工逐段核实。"])

    tex_table("gap", [{"非脚步类别": r["非脚步类别"], "可匹配事件": r["可匹配事件数"], "FPS AI": r["FPS AI"], "实际 EQ": r["EQ（实际）"],
                       "匹配静态 EQ": r["匹配静态 EQ"]} for r in c["gap_rows"]],
              "同电平增益差 ΔG（dB；正值表示比同电平脚步少增强）", spec="lcccc",
              notes=[f"参照脚步为人工确认脚步（{len(fo_h)} 个），电平相差 ±{MATCH_DB:.0f} dB 以内且至少 {MATCH_MIN} 个；末行被检验事件为模型高置信度非脚步，属扩展分析。",
                     f"敏感性：参照脚步纳入模型高置信度脚步（共 {len(c['fo_all'])} 个）时，枪声可匹配 {c['gap_all']['AI'][1]} 个，"
                     f"FPS AI 为 {sgn(c['gap_all']['AI'][0])} dB，匹配静态 EQ 为 {sgn(c['gap_all']['EQm'][0])} dB，结论不变。".replace("-", "−")])

    tex_table("mse", [{"类别": SHORT[g], "n": int((d.group == g).sum()),
                       "FPS AI 总": f"{mse(g, 'MSE_AI'):.1f}", "FPS AI 形状": f"{mse(g, 'MSEshape_AI'):.1f}",
                       "实际 EQ 总": f"{mse(g, 'MSE_EQ'):.1f}", "实际 EQ 形状": f"{mse(g, 'MSEshape_EQ'):.1f}",
                       "匹配 EQ 总": f"{mse(g, 'MSE_EQm'):.0f}", "匹配 EQ 形状": f"{mse(g, 'MSEshape_EQm'):.1f}"} for g in NT + [FOOT]],
              "对数频谱均方误差（dB²，中位数）", spec="lccccccc",
              heads=["类别", "n", r"\makecell{FPS AI\\总}", r"\makecell{FPS AI\\形状}", r"\makecell{实际 EQ\\总}", r"\makecell{实际 EQ\\形状}",
                     r"\makecell{匹配 EQ\\总}", r"\makecell{匹配 EQ\\形状}"],
              notes=["“总”包含整体音量变化；“形状”先减去两者平均差，只反映音色改变。"])

    tex_table("fer", [{"类别": name, "n": len(x), "FPS AI": pct((x.G_AI > FER_THR).mean()), "实际 EQ": pct((x.G_EQ > FER_THR).mean()),
                       "匹配静态 EQ": pct((x.G_EQm > FER_THR).mean())}
                      for name, x in [("枪声", gun), ("换弹", d[d.group == NT[1]]), ("烟雾弹/吃药/爆能器", d[d.group == NT[2]]),
                                      ("全部非脚步", nt_h), ("背景段*", bg)]],
              f"误增强率（增益 > {FER_THR:.0f} dB 的比例）", spec="lcccc", notes=["* 背景段未经人工核实。"])

    pk = d.merge(ev[["event_id", "fps_gain", "raw_peak"]], on="event_id").assign(G_AI=lambda x: x.fps_gain, raw_level=lambda x: x.raw_peak)
    conds = [("主分析（纯净 + 有持续背景）", d), ("仅纯净", d[d.purity == "p"]), ("仅有持续背景", d[d.purity == "b"]),
             ("VALORANT", d[d.game == "valorant"]), ("PUBG", d[d.game == "pubg"]), ("峰值口径（起点后 80 ms）", pk)]
    rob_rows, forest = [], []
    for name, x in conds:
        est, k, ci = gap_ci(x)
        f_, g_ = x[x.group == FOOT], x[x.group == NT[0]]
        rob_rows.append({"检验条件": name, "脚步 n": len(f_), "脚步 G 中位": sgn(f_.G_AI.median()), "枪声 n": len(g_),
                         "枪声 G 中位": sgn(g_.G_AI.median()), "同电平 ΔG [95% CI]": f"{sgn(est)} [{sgn(ci[0])}, {sgn(ci[1])}]", "可匹配": k})
        forest.append(dict(name=name, est=est, lo=ci[0], hi=ci[1], n_matched=k))
    tex_table("robust", rob_rows, "稳定性检验：FPS AI 在不同条件下的脚步与枪声增益（dB）", spec="lcccccc",
              heads=["检验条件", r"\makecell{脚步\\n}", r"\makecell{脚步\\G 中位}", r"\makecell{枪声\\n}", r"\makecell{枪声\\G 中位}",
                     r"\makecell{同电平 $\Delta$G\\{[}95\% CI{]}}", r"\makecell{可匹配\\枪声数}"],
              notes=["各条件的参照脚步均为同一子集的人工确认脚步，首行即表 \\ref{tab:gap} 的主分析结果；CI 为按片段聚类自助法（1000 次）。"])
    rob_ok = all(r["est"] > 0 for r in forest)
    rob_wide = [r for r in forest if r["lo"] <= 0]
    rob_wide_txt = "、".join(f"{r['name']}（可匹配枪声 {r['n_matched']} 个）" for r in rob_wide)
    rob_rng = f"{sgn(min(r['est'] for r in forest))}～{sgn(max(r['est'] for r in forest))}"

    gcols = ["脚步（人工）", "脚步（人工+模型）", "枪声（人工）", "非脚步（模型）"]
    tex_table("game", [{"游戏": r["游戏"], "处理": r["处理"], **{k: stack(r[k], "（") for k in gcols}, "背景段": r["背景段"]}
                       for r in c["game_rows"]],
              "分游戏的中位增益（dB；方括号为 95% CI，圆括号为事件数）", spec="llccccc", size=r"\footnotesize",
              heads=["游戏", "处理", "脚步（人工）", r"\makecell{脚步\\（人工 + 模型）}", "枪声（人工）", r"\makecell{非脚步\\（模型）}", "背景段"])

    L = paper.loudness_stats()
    lp = L.pivot_table(index=["game", "clip_id"], columns="track", values=["rms", "l50", "l99", "lmax"])
    dl = lambda m, t: lp[(m, t)] - lp[(m, "raw")]
    is_loud = lp[("lmax", "raw")] >= paper.LOUD_SPLIT
    dmax_loud, dmax_quiet = dl("lmax", "fps")[is_loud], dl("lmax", "fps")[~is_loud]
    d50_loud = dl("l50", "fps")[is_loud].median()
    fps_max_quiet, raw_max_all = lp[("lmax", "fps")][~is_loud].max(), lp[("lmax", "raw")].max()
    tex_table("loud", [{"片段类型": nm, "片段数": int(m.sum()),
                        **{f"{lab} AI": sgn(dl(k, "fps")[m].median()) for k, lab in [("rms", "RMS"), ("l50", "L50"), ("l99", "L99"), ("lmax", "Lmax")]},
                        **{f"{lab} EQ": sgn(dl(k, "eq")[m].median()) for k, lab in [("rms", "RMS"), ("l50", "L50"), ("l99", "L99"), ("lmax", "Lmax")]}}
                       for nm, m in [("含响亮声音", is_loud), ("安静、以脚步为主", ~is_loud), ("全部", is_loud | ~is_loud)]],
              "整段响度变化（dB，各片段中位数）", spec="lcccccccccc".replace("cccccccccc", "ccccccccc"), size=r"\footnotesize",
              heads=["片段类型", "片段数", r"\makecell{RMS\\FPS AI}", r"\makecell{L50\\FPS AI}", r"\makecell{L99\\FPS AI}", r"\makecell{Lmax\\FPS AI}",
                     r"\makecell{RMS\\实际 EQ}", r"\makecell{L50\\实际 EQ}", r"\makecell{L99\\实际 EQ}", r"\makecell{Lmax\\实际 EQ}"],
              notes=[f"按原始最大短时电平分类，分界 {paper.LOUD_SPLIT:.0f} dBFS（两类片段之间的自然间隙）。L50 / L99 / Lmax 为 50 ms 短时电平的中位数、99\\% 分位数与最大值。".replace("-", "$-$", 1)])

    prod = [("脚步中位增益（dB）", sgn(med(fo_h, "EQ")), sgn(med(fo_h, "AI"))),
            ("脚步增益 ≥ 3 dB 的比例", pct((fo_h.G_EQ >= 3).mean()), pct((fo_h.G_AI >= 3).mean())),
            ("枪声中位增益（dB）", sgn(med(gun, "EQ")), sgn(med(gun, "AI"))),
            ("背景段中位增益（dB）", sgn(med(bg, "EQ")), sgn(med(bg, "AI"))),
            ("非脚步误增强率 FER", pct(fer["EQ"]), pct(fer["AI"])),
            ("枪声频谱总变化 MSE（dB²）", f"{mse(NT[0], 'MSE_EQ'):.1f}", f"{mse(NT[0], 'MSE_AI'):.1f}"),
            ("枪声频谱形状变化（dB²）", f"{mse(NT[0], 'MSEshape_EQ'):.1f}", f"{mse(NT[0], 'MSEshape_AI'):.1f}"),
            ("含响亮声音片段的最大电平变化（dB）", sgn(dl("lmax", "eq")[is_loud].median()), sgn(dmax_loud.median()))]
    tex_table("product", [{"指标": a_, "C3 实际 EQ（产品预设）": e_, "C3 FPS AI": f_} for a_, e_, f_ in prod],
              "产品主对比：C3 两种实际可用模式相对原始轨的表现", spec="Lcc",
              heads=["指标", r"\makecell{C3 实际 EQ\\（产品预设）}", r"\textbf{C3 FPS AI}"],
              notes=[f"人工确认事件：脚步 {len(fo_h)} 个、枪声 {len(gun)} 个、非脚步 {len(nt_h)} 个；背景段 {len(bg)} 段未经人工核实。增益与频谱变化均相对原始轨计算（定义见表 \\ref{{tab:metrics}}）。"])

    h1 = ci_h[0] > CLAIM[0]
    h2 = gap["AI"][NT[0]][0] >= 3 and abs(g_eqm) < 1
    h3a, h3b = mse(NT[0], "MSE_AI") < mse(NT[0], "MSE_EQ"), fer["AI"] <= 0.10
    v = lambda ok: "支持" if ok else "不支持"
    in_claim, gt9_, lt3_ = (pct(x) for x in (fo_h.G_AI.between(*CLAIM).mean(), (fo_h.G_AI > CLAIM[1]).mean(), (fo_h.G_AI < CLAIM[0]).mean()))
    gh = lambda g, grp=FOOT: d[(d.game == g) & (d.group == grp)].G_AI.median()
    tex_table("verdict", [
        {"假设": "H1 脚步增强有效", "结果": f"脚步 G 中位 {sgn(med(fo_h, 'AI'))} dB，95% CI [{sgn(ci_h[0])}, {sgn(ci_h[1])}]；{pct((fo_h.G_AI >= 3).mean())} ≥ 3 dB；3–9 dB {in_claim}，> 9 dB {gt9_}", "判定": v(h1)},
        {"假设": "H2 事件选择性（核心）", "结果": f"同电平枪声 ΔG：FPS AI {sgn(gap['AI'][NT[0]][0])} dB，匹配静态 EQ {sgn(g_eqm)} dB", "判定": v(h2)},
        {"假设": "H3 非目标声音保持性", "结果": f"枪声频谱 MSE {mse(NT[0], 'MSE_AI'):.1f} vs 实际 EQ {mse(NT[0], 'MSE_EQ'):.1f} dB²；非脚步 FER {pct(fer['AI'])}（> 10% 工程阈值）",
         "判定": "支持" if h3a and h3b else ("部分支持" if h3a else "不支持")},
        {"假设": "H4 稳定性", "结果": f"6 种条件下同电平 ΔG 均为正（{rob_rng} dB），{len(rob_wide)} 个小子集 CI 跨 0；20% 标签噪声下效应 {sgn(rob[0.2][0])} dB", "判定": v(rob_ok and rob[0.2][0] > 0)}],
        "假设判定汇总", spec="lLc")

    tex_table("claims", [
        {"宣传册宣称": "智能强化关键脚步声；增强 3–9 dB",
         "本报告结果": f"中位 {sgn(med(fo_h, 'AI'))} dB，{pct((fo_h.G_AI >= 3).mean())} ≥ 3 dB；{in_claim} 落在 3–9 dB，{gt9_} 超过 9 dB",
         "结论": "“显著增强”支持；3–9 dB 区间部分支持（中位位于区间上沿，较多样本超过 9 dB）"},
        {"宣传册宣称": "踩草地、跳窗等细微脚步声", "本报告结果": f"< -40 dBFS 脚步 {sgn(lvl_lo.G_AI.median())} dB，低于 3 dB 仅 {pct((lvl_lo.G_AI < 3).mean())}",
         "结论": "部分支持：安静脚步增强明显；未区分材质与动作"},
        {"宣传册宣称": "实时识别；不同于传统 EQ，只强化必要部分",
         "本报告结果": f"产品对比：C3 实际 EQ 脚步 {sgn(med(fo_h, 'EQ'))}、枪声 {sgn(med(gun, 'EQ'))} dB，FPS AI 脚步 {sgn(med(fo_h, 'AI'))}、枪声 {sgn(med(gun, 'AI'))} dB；"
                      f"机制对照：同电平枪声少增强 {gap['AI'][NT[0]][0]:.1f} dB，匹配静态 EQ {sgn(g_eqm)} dB",
         "结论": "支持：与产品实际 EQ 作用逻辑不同，且存在区别于固定静态 EQ 的事件选择性处理；本实验不直接推断设备内部识别机制"},
        {"宣传册宣称": "不改变枪声",
         "本报告结果": f"枪声中位 {sgn(med(gun, 'AI'))} dB；频谱 MSE {mse(NT[0], 'MSE_AI'):.1f} vs 实际 EQ {mse(NT[0], 'MSE_EQ'):.1f} dB²；{pct(fer_g['AI'])} 的枪声（远处安静枪声）> 3 dB",
         "结论": "部分支持：响亮枪声基本保持"},
        {"宣传册宣称": "不改变爆炸声", "本报告结果": "样本中无人工确认的爆炸声", "结论": "未覆盖"},
        {"宣传册宣称": "不干扰环境音；不改变其他声音特征",
         "本报告结果": f"背景段 {sgn(med(bg, 'AI'))} dB，脚步附近背景 {sgn(c['pre_bg'])} dB；非脚步 FER {pct(fer['AI'])}", "结论": "部分支持"},
        {"宣传册宣称": "游戏整体音量未做改变",
         "本报告结果": f"含响亮声音的片段 Lmax {sgn(dmax_loud.median())} dB；安静片段 {sgn(dmax_quiet.median())} dB（处理后至多 {fps_max_quiet:.0f} dBFS）",
         "结论": "部分支持：含响亮声音的场景成立，安静场景随脚步上升"},
        {"宣传册宣称": "传统 EQ 所有声音一并受影响",
         "本报告结果": f"脚步增益相同的匹配静态 EQ：枪声 {sgn(med(gun, 'EQm'))} dB、背景 {sgn(med(bg, 'EQm'))} dB", "结论": "支持（就固定静态 EQ 而言）"},
        {"宣传册宣称": "不同目标游戏上的一致性",
         "本报告结果": f"脚步：VALORANT {sgn(gh('valorant'))} dB、PUBG {sgn(gh('pubg'))} dB；枪声 {sgn(gh('valorant', NT[0]))} / {sgn(gh('pubg', NT[0]))} dB",
         "结论": "两款目标游戏中趋势一致"},
        {"宣传册宣称": "全链路处理延迟仅 40 ms", "本报告结果": "录音离线对齐，无法测量", "结论": "未覆盖"}],
        "宣传册宣称逐条核验（支持 / 部分支持 / 未覆盖）", spec="LLL", size=r"\footnotesize",
        notes=["“支持”指在本次有限实验数据范围内结果与宣称一致，不等同于对宣称的完全证明。"])

    tex_table("boosted", [{k: r[k] for k in ["类别", "备注", "片段", "时间 (s)", "原始电平", "FPS AI 增益", "EQ 增益"]} for r in c["boosted_rows"]],
              f"被 FPS AI 增强超过 {FER_THR:.0f} dB 的人工确认非脚步事件", spec="lllcccc", size=r"\footnotesize")

    det = c["det"]
    tr_, tf, te = (det["match"][("top1", t)] for t in ("raw", "fps", "eq"))
    cr, cf, ce = (det["match"][("closed", t)] for t in ("raw", "fps", "eq"))
    zr, zf, ze = c["zr"], c["zf"], c["ze"]
    ft = pd.read_csv(os.path.join(RES, "panns_ft_summary.csv"), encoding="utf-8-sig").set_index(["split", "track"])
    pp, ppci = c["pp"], c["ppci"]
    drow = lambda name, a, b, e, au: {"检测器": name, "原始 AUC": au, "原始检出率": pct(a["recall"]),
                                      "FPS AI − 原始": f"{pp(a['recall'], b['recall'])} {ppci(b['d_recall_ci'])}",
                                      "EQ − 原始": f"{pp(a['recall'], e['recall'])} {ppci(e['d_recall_ci'])}"}
    tex_table("detect", [drow("PANNs 零样本，仅 Walk, footsteps", zr, zf, ze, f"{zr['auc']:.2f}"),
                         drow("PANNs 零样本，六类中脚步占比", cr, cf, ce, f"{cr['auc']:.2f}"),
                         drow("PANNs 零样本，六类中判类别", tr_, tf, te, "--"),
                         drow("PANNs 微调，留出 8 个片段", ft.loc[("holdout", "raw")], ft.loc[("holdout", "fps")], ft.loc[("holdout", "eq")], f"{ft.loc[('holdout', 'raw')]['auc']:.2f}"),
                         drow("PANNs 微调，5 折交叉验证", ft.loc[("cv5", "raw")], ft.loc[("cv5", "fps")], ft.loc[("cv5", "eq")], f"{ft.loc[('cv5', 'raw')]['auc']:.2f}")],
              "探索性分析：不同检测器下的脚步检出率变化（百分点，方括号为 95% CI）", spec="lcccc", size=r"\footnotesize")

    S, n_eval, n_ef = label_accuracy()
    n_en = n_eval - n_ef
    b0, cz, a4 = S.iloc[0], S.iloc[1], S.iloc[-1]
    v1 = pd.read_csv(os.path.join(RES, "events_labeled.csv"))
    cover_before, cover_after = (v1.label != "ambiguous").mean(), (ev.label != "ambiguous").mean()
    cb_, ca_ = label_composition(ev, v1)
    n_all = sum(cb_.values())
    tex_table("model_eff", [
        {"方案": "标注前：AudioSet 分类器 + 规则", "人工标注": "不用", "监督参数": "0", "AUC": f"{b0.auc:.3f}"},
        {"方案": "CLAP 文本提示（零样本）", "人工标注": "不用", "监督参数": "0", "AUC": f"{cz.auc:.3f}"},
        {"方案": "AST 线性探针", "人工标注": "用", "监督参数": str(dims["AST"][1] + 1), "AUC": f"{auc['cv_AST'][0]:.3f}"},
        {"方案": "CLAP 线性探针", "人工标注": "用", "监督参数": str(dims["CLAP"][1] + 1), "AUC": f"{auc['cv_CLAP'][0]:.3f}"},
        {"方案": Raw(r"\textbf{AST + CLAP 集成（最终模型）}"), "人工标注": "用", "监督参数": str(n_par), "AUC": Raw(rf"\textbf{{{auc['cv_ens'][0]:.3f}}}")}],
        f"标注模型效果：人工标注与双分支集成带来的提升（第 1 轮随机样本 {n_eval} 条）", spec="lccc",
        notes=[f"“用”人工标注的方案以 {int(ev.human_label.isin(['f', 'g', 'o']).sum())} 条明确人工标注训练，按片段分组 5 折交叉验证，被评估事件所在片段不参与训练。",
               "“监督参数”指逻辑回归的权重与偏置；AST/CLAP 主干冻结，StandardScaler 与 PCA 在每折训练数据上无监督拟合（不使用标签），不计入。"])
    share = lambda k_: f"{k_}（{pct(k_ / n_all)}）"
    tex_table("label_ba", [
        {"项目": "判为脚步", "a": share(cb_["f_m"]), "b": f"{share(ca_['f_h'] + ca_['f_m'])}，其中人工 {ca_['f_h']}"},
        {"项目": "判为非脚步", "a": share(cb_["n_m"]), "b": f"{share(ca_['n_h'] + ca_['n_m'])}，其中人工 {ca_['n_h']}"},
        {"项目": "判不出（排除）", "a": f"{cb_['a_m']}（{cb_['a_m'] / n_all * 100:.1f}%）",
         "b": f"{ca_['a_h'] + ca_['a_m']}（{(ca_['a_h'] + ca_['a_m']) / n_all * 100:.1f}%）：模型拿不准 {ca_['a_m']}（{ca_['a_m'] / n_all * 100:.1f}%）+ 人工不确定 {ca_['a_h']}"},
        {"项目": "由人工直接确定", "a": "0", "b": str(ca_["f_h"] + ca_["n_h"] + ca_["a_h"])},
        {"项目": "给出判断时的正确率*", "a": pct(b0.correct), "b": pct(a4.correct)}],
        f"标注前后的事件判别情况（全部 {n_all} 个事件）", spec="lLL",
        heads=["项目", r"\makecell{标注前\\现成模型 + 规则}", r"\makecell{标注后\\人工标注 + 最终模型}"],
        notes=["“现成模型 + 规则”为 AudioSet 预训练声音分类器的脚步 / 枪声概率加阈值规则，即标注前使用的自动标签。",
               f"标注后人工标注一律优先，其余事件由最终模型按三区间判定：p ≥ {P_HI} 判脚步、p ≤ {1 - P_HI:.1f} 判非脚步，其余为“判不出”。",
               f"* 以第 1 轮随机样本 {n_eval} 条的人工判断为准估计；标注后为最终模型的留片段交叉验证结果（人工标注事件以人工为准，无需估计）。"])
    fig_label_accuracy((cb_, ca_), S, n_eval, "figb_label_accuracy")
    S.to_csv(os.path.join(DATA, "figb_label_accuracy.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(phase="标注前", **cb_), dict(phase="标注后", **ca_)]) \
        .to_csv(os.path.join(DATA, "figb_label_composition.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ figure drafts and data
    copy_draft(c["eq_fig"], "fig05_eq_response")
    copy_draft(c["fig_path"], "fig06_gain_vs_level")
    copy_draft(os.path.join(RES, "ad_gain_bars.png"), "fig07_gain_by_class")
    copy_draft(os.path.join(RES, "ad_response.png"), "fig08_fps_response")
    fc = fer_curve(d, bg, np.arange(0, 12.5, 0.5))
    fig_fer_curve(fc, "fig09_fer_curve")
    draft_forest(forest, "fig10_robustness_forest")
    loud_png = os.path.join(RES, "paper_loudness.png")
    paper.fig_loudness(lp, loud_png)
    copy_draft(loud_png, "fig11_loudness")
    pure = d[d.purity == "p"]
    fs = pure[pure.group == FOOT]
    ex_f = int(fs.iloc[(fs.G_AI - fs.G_AI.median()).abs().argsort().iloc[0]].event_id)
    ex_g = int(pure[pure.group == NT[0]].sort_values("raw_level").iloc[-1].event_id)
    ex_gains = {e: tuple(d.set_index("event_id").loc[e, ["G_AI", "G_EQ", "G_EQm"]]) for e in (ex_f, ex_g)}
    draft_spectrograms([("脚步", "footstep", ex_f), ("枪声", "gunshot", ex_g)], "fig12_spectrograms", ex_gains)
    fig_waveforms([("脚步", "footstep", ex_f), ("枪声", "gunshot", ex_g)], ex_gains, "figb_waveforms_full", with_eqm=True)

    for g in ["valorant", "pubg"]:
        e = eq[g]
        pd.DataFrame({"freq_hz": e["f"], "measured_db": e["meas"], "theory_db": e["theo"]}).to_csv(os.path.join(DATA, f"fig05_eq_{g}.csv"), index=False)
    pd.concat([d.assign(source="human"), dm.assign(source="model")])[["event_id", "source", "group", "game", "clip_key", "raw_level", "G_AI", "G_EQ", "G_EQm"]] \
        .to_csv(os.path.join(DATA, "fig06_fig07_events.csv"), index=False, encoding="utf-8-sig")
    bg[["game", "clip_key", "start", "dur", "raw_level", "G_AI", "G_EQ", "G_EQm"]].to_csv(os.path.join(DATA, "fig07_background.csv"), index=False)
    shutil.copy(os.path.join(RES, "ad_response.csv"), os.path.join(DATA, "fig08_fps_response.csv"))
    fc.to_csv(os.path.join(DATA, "fig09_fer_curve.csv"), index=False)
    pd.DataFrame(forest).to_csv(os.path.join(DATA, "fig10_robustness.csv"), index=False, encoding="utf-8-sig")
    lp.columns = [f"{a}_{b}" for a, b in lp.columns]
    lp.reset_index().to_csv(os.path.join(DATA, "fig11_loudness.csv"), index=False)
    pd.DataFrame([{"panel": "脚步", "event_id": ex_f, "folder": "clean_events/footstep"},
                  {"panel": "枪声", "event_id": ex_g, "folder": "clean_events/gunshot"}]).to_csv(os.path.join(DATA, "fig12_examples.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ text values
    V = dict(
        n_vl=str(n_vl), n_pg=str(n_pg), dur=f"{dur_min:.1f}", n_lab=str(n_lab), n_r1=str(n_r1), n_par=str(n_par),
        g_fo=n(med(fo_h, "AI")), g_fo_all=n(med(fo_all, "AI")), ci_lo=n(ci_h[0]), ci_hi=n(ci_h[1]), n_fo=str(len(fo_h)),
        g_gun=n(med(gun, "AI")), g_fo_eq=n(med(fo_h, "EQ")), g_gun_eq=n(med(gun, "EQ")), g_fo_eqm=n(med(fo_h, "EQm")), gap_gun=n(gap["AI"][NT[0]][0], sign=False), gap_rel=n(gap["AI"][NT[1]][0], sign=False),
        gap_ut=n(gap["AI"][NT[2]][0]), n_ut_gap=str(gap["AI"][NT[2]][1]), g_eqm=n(g_eqm), sei_ai=n(sei["AI"]), sei_eq=n(sei["EQ"]), sei_eqm=n(sei["EQm"]),
        mse_ai=n(mse(NT[0], "MSE_AI"), sign=False), mse_eq=n(mse(NT[0], "MSE_EQ"), sign=False),
        mse_ratio=f"{mse(NT[0], 'MSE_AI') / mse(NT[0], 'MSE_EQ') * 100:.0f}\\%",
        mse_frac=f"{mse(NT[0], 'MSE_EQ') / mse(NT[0], 'MSE_AI'):.0f}",
        mse_sfrac=f"{mse(NT[0], 'MSEshape_EQ') / mse(NT[0], 'MSEshape_AI'):.0f}", mse_fo_shape=n(mse(FOOT, "MSEshape_AI"), sign=False),
        fer=p(fer["AI"]), fer_g=p(fer_g["AI"]), fer_bg=p(fer_bg["AI"]), fer_eq=p(fer["EQ"]),
        n_boost_gun=str(c["n_boost_gun"]), gb_hi=n(c["gun_boost_lv"].max(), 0, False), gb_lo=n(c["gun_boost_lv"].min(), 0, False),
        auc=f"{auc['cv_ens'][0]:.3f}", auc_ast=f"{auc['cv_AST'][0]:.3f}", auc_clap=f"{auc['cv_CLAP'][0]:.3f}", auc_all=f"{auc['cv_ens'][1]:.3f}",
        k2=f"{rs['k2']:.2f}", k5=f"{rs['k5']:.2f}", agree=p(rs["agree"]), n_rv=str(rs["n"]), n_changed=str(rs["changed"]),
        prec_f=p(acc["prec_f"]), prec_n=p(acc["prec_n"]), n_pf=str(acc["n_f"]), n_pn=str(acc["n_n"]),
        r4_lo=p(c["r4_acc"].loc[0]), r4_hi=p(c["r4_acc"].loc[1]), n_amb=str(c["n_amb_now"]), amb_share=f"{c['n_amb_now'] / 1620 * 100:.1f}\\%",
        cover=p(1 - c["n_amb_now"] / 1620),
        d_ast=str(dims["AST"][0]), d_clap=str(dims["CLAP"][0]), k_ast=str(dims["AST"][1]), k_clap=str(dims["CLAP"][1]),
        p_hi=f"{P_HI}", p_lo=f"{1 - P_HI:.1f}",
        rob_n=str(rob["n"]), rob_base=n(rob["base"]), rob10=n(rob[0.1][0]), rob20=n(rob[0.2][0]), rob_worst=str(rob["worst"][0]),
        rob_worst_p=p(rob["worst"][1]), disagree=p(1 - rs["agree"]),
        g_pubg=n(gh("pubg")), g_val=n(gh("valorant")), gun_pubg=n(gh("pubg", NT[0])), gun_val=n(gh("valorant", NT[0])),
        gun_eq_pubg=n(gm("pubg", NT[0], "EQ")),
        ge3=p((fo_h.G_AI >= 3).mean()), in_claim=p(fo_h.G_AI.between(*CLAIM).mean()),
        g_gun_eqm=n(med(gun, "EQm")), g_bg_eqm=n(med(bg, "EQm")),
        mse_gs_ai=n(mse(NT[0], "MSEshape_AI"), sign=False), mse_gs_eq=n(mse(NT[0], "MSEshape_EQ"), sign=False),
        mse_fo_shape_eq=n(mse(FOOT, "MSEshape_EQ"), sign=False),
        n_eval=str(n_eval), n_ef=str(n_ef), n_en=str(n_en),
        b_dec=p(b0.decided), b_cor=p(b0.correct), b_rec=p(b0.rec_f), b_auc=f"{b0.auc:.2f}", b_auc3=f"{b0.auc:.3f}", cz_auc=f"{cz.auc:.3f}",
        a_dec=p(a4.decided), a_cor=p(a4.correct), a_rec=p(a4.rec_f), a_recn=p(a4.rec_n), a_auc=f"{a4.auc:.2f}",
        cov_b=p(cover_before), cov_a=p(cover_after), nb_amb=f"{cb_['a_m'] / n_all * 100:.1f}\\%", n_hx=str(ca_["a_h"]),
        n_amb_all=str(ca_["a_h"] + ca_["a_m"]), amb_all_share=f"{(ca_['a_h'] + ca_['a_m']) / n_all * 100:.1f}\\%",
        n_model=str(n_all - len(ev[ev.human_label != ""])), n_human=str(int((ev.human_label != "").sum())),
        lf_v=p(bs[("valorant", "footstep")][0]), lf_p=p(bs[("pubg", "footstep")][0]),
        n_hi=str(len(lvl_hi)), n_hiclip=str(len(hi_clip)), hi_low_txt=c["hi_low_txt"], hi_low_rng=esc(c["hi_low_rng"]), hi_norm_rng=esc(c["hi_norm_rng"]),
        lt3=p((fo_h.G_AI < 3).mean()), gt9=p((fo_h.G_AI > 9).mean()),
        n_seq=str(sp["n_seq"]), sp_first=n(sp["first"]), sp_later=n(sp["later"]), sp_diff=n(sp["diff"]),
        pr_n=str(prom["n_fo"]), pr_peak=n(prom["peak"]), pr_bg=n(prom["bg"]), pr_c=n(prom["c_fo"]), pr_lo=n(prom["c_fo_ci"][0]), pr_hi=n(prom["c_fo_ci"][1]),
        pr_nf=n(prom["c_nf"]), pr_eq=n(prom["c_eq"]), tv_ai=f"{c['tv_ai']:.2f}", tv_eq=f"{c['tv_eq']:.2f}", g_bg=n(med(bg, "AI")), pre_bg=n(c["pre_bg"]),
        n_bg=str(len(bg)), n_loud=str(int(is_loud.sum())), n_quiet=str(int((~is_loud).sum())), dmax_loud=n(dmax_loud.median()), l50_loud=n(d50_loud),
        dmax_quiet=n(dmax_quiet.median()), fps_max_quiet=n(fps_max_quiet, 0, False), raw_max_all=n(raw_max_all, 0, False), split=n(paper.LOUD_SPLIT, 0, False),
        dev=f"{max(eq['valorant']['dev'], eq['pubg']['dev']):.2f}", eq_q=str(EQ_Q), n_nt=str(len(nt_h)),
        n_gun_pubg=str(int(((d.group == NT[0]) & (d.game == "pubg")).sum())), n_clean=str(len(meta)),
        z_drop=n(abs((zf["recall"] - zr["recall"]) * 100), 1, False), z_thr=f"{zr['thr']:.4f}",
        ex_f=str(ex_f), ex_g=str(ex_g), exf_ai=n(ex_gains[ex_f][0]), exg_ai=n(ex_gains[ex_g][0]),
        exf_eqm=n(ex_gains[ex_f][2]), exg_eqm=n(ex_gains[ex_g][2]), match_db=f"{MATCH_DB:.0f}", match_min=str(MATCH_MIN), fer_thr=f"{FER_THR:.0f}",
        v_h1=v(h1), v_h2=v(h2), v_h3="支持" if h3a and h3b else ("部分支持" if h3a else "不支持"), v_h4=v(rob_ok and rob[0.2][0] > 0),
        rob_rng=esc(rob_rng), rob_wide_txt=esc(rob_wide_txt),
    )
    verdicts = (V["v_h1"], V["v_h2"], V["v_h3"], V["v_h4"])
    if verdicts != ("支持", "支持", "部分支持", "支持"):
        raise RuntimeError(f"hypothesis verdicts changed {verdicts}; update the summary text in TEMPLATE")
    body = TEMPLATE.replace("→", r"$\rightarrow$")
    for k, val in V.items():
        body = body.replace(f"[[{k}]]", val)
    left = re.findall(r"\[\[\w+\]\]", body)
    if left:
        raise ValueError(f"unfilled: {sorted(set(left))}")
    with open(os.path.join(OUTDIR, "main.tex"), "w", encoding="utf-8") as fh:
        fh.write(body)

    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUTDIR,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("xelatex failed")
    print("->", publish(os.path.join(OUTDIR, "main.pdf"), PDF_OUT))


def publish(src, dst):
    """Copy the PDF to dst; if dst is open in another program, fall back to dst（新）, dst（新2）, ..."""
    for k in range(10):
        out = dst if k == 0 else dst.replace(".pdf", f"（新{k if k > 1 else ''}）.pdf")
        try:
            shutil.copy(src, out)
            return out
        except PermissionError:
            continue
    raise PermissionError(dst)


TEMPLATE = r"""\documentclass[11pt,a4paper]{ctexart}
\usepackage[top=25mm,bottom=25mm,left=24mm,right=24mm]{geometry}
\usepackage{graphicx,booktabs,threeparttable,makecell,multirow,tabularx,array,enumitem,amsmath,amssymb,xcolor,float,fancyhdr,lastpage}
\usepackage[section]{placeins}
\usepackage{etoolbox}
\pretocmd{\subsection}{\FloatBarrier}{}{}
\usepackage[font=small,labelfont=bf,labelsep=quad,skip=4pt]{caption}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\ctexset{section={format=\Large\heiti\raggedright},subsection={format=\large\heiti\raggedright},
         subsubsection={format=\normalsize\heiti\raggedright}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\renewcommand{\arraystretch}{1.18}
\setlist{nosep,leftmargin=2em}
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}\setcounter{topnumber}{3}\setcounter{totalnumber}{5}
\setlength{\emergencystretch}{2em}
\pagestyle{fancy}\fancyhf{}
\lhead{\small C3 FPS AI 声卡客观评测技术报告（详细版）}\rhead{\small 第 \thepage\ 页 / 共 \pageref{LastPage} 页}
\renewcommand{\headrulewidth}{0.4pt}
\newcommand{\figslot}[4]{%
\begin{figure}[htbp]\centering
\IfFileExists{figures/#1.pdf}{\includegraphics[width=#2]{figures/#1.pdf}}{%
\IfFileExists{figures/#1.png}{\includegraphics[width=#2]{figures/#1.png}}{%
\fbox{\parbox[c][4.2cm][c]{#2}{\centering\small\color{gray}图位：\texttt{\detokenize{#1}}\\[2pt]按《图表需求说明》制作后放入 figures 目录}}}}
\caption{#3}\label{fig:#4}\end{figure}}
\newcommand{\keybox}[1]{\par\noindent\fcolorbox{black!45}{black!4}{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{#1}}\par}

\begin{document}
\begin{titlepage}
\centering\vspace*{35mm}
{\Huge\heiti C3 FPS AI 声卡脚步增强功能\\[10pt] 客观评测技术报告\par}
\vspace{8mm}
{\Large 技术验证版（详细版）· Technical Report\par}
\vspace{10mm}
{\large 基于 VALORANT 与 PUBG 真实游戏录音的同源三轨实录、人机协同标注与混杂受控实验\par}
\vspace{40mm}
\begin{tabular}{rl}
评测对象： & Fosi Audio C3 游戏声卡（FPS AI 模式）\\
评测范围： & 脚步增强功能的声学效果（不含硬件指标、7.1、通话降噪）\\
数据： & VALORANT [[n_vl]] 段、PUBG [[n_pg]] 段实录，共 [[dur]] min\\
编制： & [编制部门 / 人员]\\
版本： & v1.0\quad 2026 年 10 月\\
\end{tabular}
\vfill
{\small 本报告为事实与口径母版，对外简版（Executive Report）中的数字与结论均以本报告为准。\\
全部表格与数据由 \texttt{analysis/report\_tex.py} 从原始数据自动生成。\par}
\end{titlepage}

\tableofcontents
\clearpage

\section*{报告摘要}
\addcontentsline{toc}{section}{报告摘要}
本报告依据《C3 FPS AI 声卡客观评测方案》与 C3 宣传册，对 FPS AI 的脚步增强功能进行客观评测。评测使用本阶段计划评测的两款游戏 VALORANT（[[n_vl]] 段）与 PUBG（[[n_pg]] 段）的对局实录，
每段同时录制原始、FPS AI、EQ 三个版本，共检测出 1620 个声音事件。由于缺少厂商标注，本报告设计了“人工主导、模型辅助”的四轮标注流程（[[n_lab]] 条人工标注），
对照分两层：\textbf{产品主对照}为 C3 自带的实际 EQ（产品预设），回答“直接开 C3 的 EQ 与开 FPS AI 有何区别”；
\textbf{机制性辅助对照}为同电平比较与匹配静态 EQ，回答“观察到的选择性是否只是响度差异或固定频响造成的”。

\keybox{\textbf{核心结论}：在本次 VALORANT 与 PUBG 真实游戏录音测试中，C3 实际 EQ 未增强脚步（中位 [[g_fo_eq]] dB），而 C3 FPS AI 对人工确认脚步的中位增强为 [[g_fo]] dB，同时枪声中位变化仅 [[g_gun]] dB。
进一步将固定静态 EQ 调整至相同脚步增强水平后，枪声也会被提升约 [[g_gun_eqm]] dB，表明 FPS AI 的选择性无法由固定静态频响解释。
与此同时，少量与脚步特征相近的安静瞬态声仍可能被同步增强，本次非脚步事件中约 [[fer]] 的增益超过 3 dB。

\medskip\textbf{主要发现}
\begin{enumerate}
\item \textbf{产品主对比}：C3 实际 EQ 对脚步与枪声的中位增益分别为 [[g_fo_eq]] 与 [[g_gun_eq]] dB，FPS AI 为 [[g_fo]] 与 [[g_gun]] dB（表~\ref{tab:product}）。两种产品模式的作用逻辑完全不同：实际 EQ 不增强脚步，FPS AI 显著增强脚步而基本保持枪声。
\item \textbf{事件选择性（H2，机制性检验）}：在控制原始电平后，枪声相较同电平脚步少增强 [[gap_gun]] dB；采用与 FPS AI 相同脚步增益的匹配静态 EQ 时，该差异为 [[g_eqm]] dB，接近 0。
观察到的选择性不能由固定静态频响解释，表现出明显的事件选择性处理特征。
\item \textbf{脚步增强有效（H1）}：人工确认脚步的增益中位数为 [[g_fo]] dB（95\% CI [[ci_lo]]～[[ci_hi]] dB，$n=[[n_fo]]$），[[ge3]] 不低于 3 dB；
[[in_claim]] 落在宣称的 3--9 dB，[[gt9]] 超过 9 dB，中位数位于宣传区间上沿。
\item \textbf{非目标声音保持（H3）}：枪声中位增益 [[g_gun]] dB，频谱变化（MSE [[mse_ai]] dB$^2$）显著低于实际 EQ（[[mse_eq]] dB$^2$）；含响亮声音的片段最大电平变化 [[dmax_loud]] dB。
但约 [[fer]] 的非脚步瞬态（主要为远处安静枪声、换弹与道具声）被增强超过 3 dB。
\item \textbf{稳定性（H4）}：选择性趋势在 VALORANT 与 PUBG 两款计划评测游戏中均被观察到，并在不同纯净度、增益口径及 20\% 标签噪声扰动下方向一致。
\item \textbf{边界}：脚步附近背景同步抬高约 [[pr_bg]] dB；以脚步为主的安静场景整体响度随之上升；本报告只验证声学处理效果。
\end{enumerate}
综合判定：H1、H2、H4 \textbf{支持}，H3 \textbf{部分支持}；宣传册相关宣称的逐条核验见表~\ref{tab:claims}（支持 / 部分支持 / 未覆盖）。}

\section{评测概述}
\subsection{评测对象与产品宣称}
C3 是 Fosi Audio Spider S 系列的游戏声卡，内置自研 FPS AI 模型。宣传册称其可“实时识别游戏声音，智能强化关键脚步声”，脚步声细节增强 3--9 dB，
“不同于传统 EQ 调节，只强化必要频段，不干扰环境音或枪声”，并且“游戏整体音量未做改变”。模型针对 CS2、PUBG、三角洲行动分别训练（每款约 4000 小时、210 万～240 万条样本），
本阶段计划评测游戏为 VALORANT 与 PUBG。本报告只评测脚步增强功能的声学效果，不涉及 DAC、7.1 空间音频与通话降噪等硬件功能，也不推断设备内部的实现机制。

\subsection{评测目标与假设}
将宣传册中与脚步增强相关的宣称归纳为四个可检验假设（表~\ref{tab:hyp}），其中 H2 事件选择性为核心假设；H1 以增益中位数的置信区间下限判定，增益与 3--9 dB 区间的关系仅作描述。“游戏整体音量未做改变”另以整段响度分析检验（\ref{sec:loud} 节）。判定标准在查看结果之前确定。
各假设之前先给出产品层面的主对比（原始 vs 实际 EQ vs FPS AI，\ref{sec:product} 节）；匹配静态 EQ 只作为 H2 的机制性辅助对照（\ref{sec:ctrl} 节）。

\input{tables/hyp}

\subsection{评测流程}
评测流程如图~\ref{fig:pipeline} 所示，遵循四项原则：同源配对（三种条件来自同一段音频）、事件级测量、主分析以人工标签为真值（模型仅用于扩展分析，精度独立核验）、混杂受控。

\figslot{fig01_pipeline}{0.98\linewidth}{评测流程总览}{pipeline}

\section{数据与评测条件}
\subsection{数据}
评测数据由品牌方提供：VALORANT 16 段、PUBG 15 段真实对局录音，每段均含三轨文件。本报告使用了提供的全部 31 段，未依据 FPS AI 的处理效果筛选或剔除任何片段；
纳入前只检查与处理效果无关的条件，即三轨齐全、可正常解码、采样率一致且三轨可对齐，31 段均通过。
片段如何从原始对局中截取（例如是否偏向含脚步的场景）、来自多少局对局以及录制设置，由品牌方掌握，本报告无法独立核实；结论适用于这批片段所代表的场景（见第~\ref{sec:limit} 节）。

数据概况见表~\ref{tab:data}。录音为 48 kHz 立体声，每段有原始（AI 与 EQ 均关闭）、FPS AI（仅开 AI）、EQ（仅开 EQ，即产品预设，下称“实际 EQ”）三个版本。
VALORANT 与 PUBG 均为本阶段计划评测游戏。

\input{tables/data}

\subsection{处理条件与同源性验证}
EQ 条件为 10 段 RBJ 峰值滤波器级联（表~\ref{tab:eq}）。三轨经互相关对齐，偏移均为 0 样本。用原始轨与 EQ 轨的互谱估计实测频响，与理论频响在 50 Hz--16 kHz 内的最大偏差仅 [[dev]] dB（图~\ref{fig:eqresp}），
证明 EQ 轨为纯静态滤波，同时验证了对齐与测量链路的精度。

\input{tables/eq}
\figslot{fig05_eq_response}{0.92\linewidth}{EQ 频响验证：实测（粗线）与按参数计算（虚线）}{eqresp}

\section{标注流程}
\subsection{候选事件生成}
在原始轨上用频谱通量检测起点（STFT 窗长 1024、跳步 5 ms；60 Hz--16 kHz；阈值为 2 s 滑窗中位数加 3 倍中位绝对偏差；最小间隔 80 ms；峰值需比前方背景高 6 dB 以上），
共得到 1620 个候选事件，FPS AI 与 EQ 轨使用相同时间点。事件终点取电平回落至背景 +3 dB 以内或比峰值低 20 dB 处，最长 1.5 s，且不超过下一事件起点。

\subsection{标注规范}
每个事件两步判断：\textbf{类别}为脚步（f）、枪声（g）、其他非脚步（o）、不确定（x），可附子类备注；\textbf{纯净度}为纯净（区间内只有该声音）、
有持续背景（单一瞬态叠加平稳背景）、有重叠（区间内还有其他瞬态）。主分析只用前两档，保证增益测量对应单一声音。

\subsection{标注工具}
标注工具为浏览器端页面，每个事件显示前后共 1.5 s 的频谱图，叠加起点、估计终点框与邻近事件起点，提供只听事件、突出事件、完整原声、起点至终点四种播放方式，
全部操作由键盘完成并自动保存。标注者只接触原始轨，看不到 FPS AI / EQ 版本和模型输出，保证盲法。
\subsection{人机协同迭代}
标注按“人工标注 → 训练模型 → 推断全部事件 → 选取下一轮样本”迭代四轮（表~\ref{tab:rounds}）。第 1 轮分层随机，兼作无偏核对样本；第 2 轮按主动学习的不确定性准则选样；
第 3 轮定向补充响亮脚步与较弱枪声；第 4 轮对剩余不确定事件分层随机抽样，补充剩余难例。
少量事件经人工复听或模型判断仍无法确定类别，这在实录游戏音频中属正常情况：人工标为“不确定”的 [[n_hx]] 个与最终模型拿不准的 [[n_amb]] 个（[[amb_share]]），共 [[n_amb_all]] 个（[[amb_all_share]]）直接排除，不再追加标注。

\input{tables/rounds}

\subsection{质量控制}
标注质量从以下四个方面核对，指标汇总见表~\ref{tab:quality}。

\textbf{人工标注的一致性。}第 1 轮 [[n_rv]] 条在隐藏原标注的条件下由标注者重新盲听判断，两次判断一致的占 [[agree]]。
Cohen $\kappa$ 衡量扣除随机一致之后的一致程度，一般认为 $\kappa>0.8$ 为几乎完全一致：四类（脚步 / 枪声 / 其他 / 不确定）$\kappa=[[k5]]$，脚步与非脚步之间 $\kappa=[[k2]]$。
两次不一致的条目经复听裁定后作为最终标签，共改判 [[n_changed]] 条。

\textbf{标注模型的准确性。}模型精度只在第 1 轮分层随机样本上评估：第 2--4 轮是按模型不确定性挑出的难例，不代表全部事件，用它们评估会使精度估计产生偏差。
留片段交叉验证的 AUC 为 [[auc]]；按三区间判定时，判为脚步的 [[n_pf]] 个中 [[prec_f]] 正确，判为非脚步的 [[n_pn]] 个中 [[prec_n]] 正确。
判为非脚步的准确率相对较低，但模型标签只用于扩展分析，主分析只使用人工确认事件。

\textbf{无法确定的事件。}人工标为“不确定”的 [[n_hx]] 个与最终模型拿不准的 [[n_amb]] 个，共 [[n_amb_all]] 个（[[amb_all_share]]），不纳入任何分析。

\textbf{标注误差对结论的影响。}标注噪声模拟（\ref{sec:robust} 节）显示，随机翻转 10\% 与 20\% 的人工标注后，核心选择性效应仍为 [[rob10]] 与 [[rob20]] dB；
需刻意挑错 [[rob_worst]] 条（[[rob_worst_p]]）才能抹平效应，远超盲审反映的不一致水平（[[disagree]]）。因此，现有标注误差不足以改变本报告的结论。

\input{tables/quality}

\section{标注模型}
\subsection{设计约束}
\begin{itemize}
\item \textbf{少样本}：人工标注仅数百条，监督学习部分的参数必须很少；
\item \textbf{领域偏移}：AudioSet 预训练模型对游戏脚步的零样本识别不可用；
\item \textbf{电平不变}：后续要在同电平条件下比较脚步与非脚步，模型若利用响度会与分析形成循环，因此输入先做音量归一化；
\item \textbf{片段相关}：交叉验证一律按片段分组。
\end{itemize}

\subsection{模型结构}
模型结构如图~\ref{fig:model}。AST 与 CLAP（HTSAT）编码器全部冻结：AST 输入起点前 0.3 s 至后 0.7 s 的 16 kHz 片段，取最后一层 144 个图块的均值（[[d_ast]] 维）；
CLAP 分别处理短窗（$-0.05$～$0.35$ s）与长窗（$\pm0.5$ s），拼接为 [[d_clap]] 维。每个分支经标准化、PCA（保留 95\% 方差，得 [[k_ast]] / [[k_clap]] 维）与 L2 正则逻辑回归（$C=0.05$，类别平衡），
两分支概率取平均。监督分类器仅包含 [[n_par]] 个可训练参数（两个逻辑回归的权重与偏置，[[k_ast]] + 1 与 [[k_clap]] + 1）；AST/CLAP 主干全部冻结，StandardScaler 与 PCA 仅进行无监督拟合（在每折训练数据上拟合，不使用标签）。输出按三区间判定：$p\geq[[p_hi]]$ 为脚步，$p\leq[[p_lo]]$ 为非脚步，其余为不确定；人工标注一律优先。

\figslot{fig04_model}{0.98\linewidth}{标注模型结构}{model}

\subsection{模型效果}
表~\ref{tab:model_eff} 以第 1 轮随机样本的人工判断为准，比较各方案的区分能力。不使用人工标注时，现成的 AudioSet 分类器加规则与 CLAP 文本提示的 AUC 仅 [[b_auc3]] 与 [[cz_auc]]；
用人工标注训练线性探针后，AST 与 CLAP 单分支分别提高到 [[auc_ast]] 与 [[auc_clap]]；两分支误差部分互补，集成后进一步提高到 [[auc]]。
按三区间判定时，最终模型判为脚步的 [[n_pf]] 个中 [[prec_f]] 正确，判为非脚步的 [[n_pn]] 个中 [[prec_n]] 正确。主分析只使用人工确认事件，模型判定仅用于扩展分析，其误差影响由标注噪声模拟评估（\ref{sec:robust} 节）。

\input{tables/model_eff}

\subsection{标注前后的判别情况}
标注前后全部事件的判别情况见表~\ref{tab:label_ba} 与图~\ref{fig:labacc}。标注完成后，[[n_human]] 个事件由人工直接确定，其余由最终模型判定，仅 [[n_amb_all]] 个（[[amb_all_share]]）因拿不准而排除。
与标注前的自动标签相比，有明确标签的事件比例由 [[cov_b]] 提高到 [[cov_a]]，给出判断时的正确率由 [[b_cor]] 提高到 [[a_cor]]。

\input{tables/label_ba}
\figslot{figb_label_accuracy}{\linewidth}{标注前后的判别情况：(a) 全部事件的判别结果（标注后人工标注优先，其余由最终模型判定）；(b) 给出判断时的正确率与 AUC}{labacc}

\section{实验设计}
\subsection{核心指标定义}\label{sec:metrics}
本节集中给出全文使用的评价指标，后文各节只引用、不再重复定义。所有指标都在三条同源轨的同一时间区间上计算：区间为事件起点前 20 ms 至估计终点后 50 ms。

\textbf{基础量。}信号片段 $x$ 的均方根幅值与对数幅度谱定义为
\begin{equation}
\mathrm{RMS}(x)=\sqrt{\frac{1}{N}\sum_{n=1}^{N}x_n^2},\label{eq:rms}
\end{equation}
\begin{equation}
S(t,f)=20\log_{10}\bigl(\lvert\mathrm{STFT}(x)\rvert(t,f)+\epsilon\bigr).\label{eq:spec}
\end{equation}
其中 $x_n$ 为区间内的全部采样值（双声道一并计入）；STFT 帧长 1024 点、帧移 256 点，只保留 50 Hz--16 kHz；$\epsilon$ 取原始轨谱峰值以下 80 dB，避免静音时频点的对数发散。
在此基础上定义 5 个核心指标（表~\ref{tab:metrics}）。

\begin{table}[htbp]\centering\small
\caption{核心指标定义}\label{tab:metrics}
\renewcommand{\arraystretch}{1.5}
\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}p{2.3cm}>{\centering\arraybackslash}p{6.4cm}X}
\toprule
指标 & 公式 & 含义与判读 \\
\midrule
事件增益 $G$ & $\displaystyle G=20\log_{10}\frac{\mathrm{RMS}_{\mathrm{proc}}}{\mathrm{RMS}_{\mathrm{raw}}}$ &
单个事件被增强多少（dB）；$G>0$ 为增强，$G\approx 0$ 为基本不变，$+6$ dB 约为波形幅值翻倍 \\
选择性指数 SEI & $\displaystyle \mathrm{SEI}=\tilde G_{\text{脚步}}-\frac{1}{3}\sum_{c}\tilde G_{c}$ &
脚步相对非目标声音的差异化增强，越大越有选择性；汇总描述指标 \\
同电平增益差 $\Delta G_k$ & $\displaystyle \Delta G_k=\operatorname*{median}_{j:\,|L_j-L_k|\le [[match_db]]\,\mathrm{dB}} G_j-G_k$ &
非目标事件 $k$ 比原始电平相近的脚步少增强多少，排除原始响度差异；H2 主指标 \\
频谱 MSE & $\displaystyle \mathrm{MSE}=\frac{1}{TF}\sum_{t,f}\bigl[S_{\mathrm{proc}}(t,f)-S_{\mathrm{raw}}(t,f)\bigr]^2$ &
频谱结构变化（dB$^2$），越小越接近原声；分“总”与“形状”两种口径 \\
误增强率 FER & $\displaystyle \mathrm{FER}=\frac{N\bigl(G_{\text{非脚步}}>[[fer_thr]]\,\mathrm{dB}\bigr)}{N_{\text{非脚步}}}$ &
非脚步事件被误增强的比例；阈值即宣称的最低脚步增益 \\
\bottomrule
\end{tabularx}
\end{table}

符号与口径说明：
\begin{itemize}
\item $\mathrm{proc}$ 指 FPS AI、实际 EQ 或匹配静态 EQ 处理轨，$\mathrm{raw}$ 指原始轨；$\tilde G$ 为某类事件增益的中位数。
\item SEI 中 $c$ 取三类非目标声音（枪声、换弹、道具）。SEI 只作汇总描述；选择性的主要验证以同电平增益差和匹配静态 EQ 对照为准。
\item $\Delta G_k$ 中 $j$ 为参照脚步，$L$ 为原始轨事件电平（dBFS）；参照脚步不足 [[match_min]] 个的事件 $k$ 不计入，报告值为所有 $k$ 的中位数。
\item 频谱 MSE 的“总”口径即上式；“形状”口径先减去 $S_{\mathrm{proc}}-S_{\mathrm{raw}}$ 在全部时频点上的均值（即扣除整体电平变化）再计算，等于该差值的方差。
\item FER 另取 [[n_bg]] 段远离事件的背景段作环境负样本，按同一阈值计算。
\item 辅助指标“声学突出度”：峰值增益减去前方背景（起点前 0.6--0.05 s 的 20\% 分位电平）的增益。
\end{itemize}

\subsection{对照条件}\label{sec:ctrl}
对照分为两层（表~\ref{tab:ctrl}）。

\begin{table}[htbp]\centering\small
\caption{两层对照设计}\label{tab:ctrl}
\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{3.6cm}XX}
\toprule
层级 & 比较 & 回答的问题 & 性质 \\
\midrule
产品主对照 & 原始 vs 实际 EQ vs FPS AI & 直接开 C3 的 EQ 与开 FPS AI，表现有何区别？ & 两种均为用户实际可用的产品模式 \\
机制性辅助对照 & FPS AI vs 匹配静态 EQ；同电平比较 & 观察到的选择性是否只是固定频响或原始响度差异造成的？ & 为分析额外构造，并非 C3 自带模式 \\
\bottomrule
\end{tabularx}
\end{table}

\textbf{产品主对照：实际 EQ。}即 C3 自带的 EQ 预设（2.2 节），与 FPS AI 同为用户实际可选的模式，是回答产品层面问题的主对照。

\textbf{机制性辅助对照：匹配静态 EQ。}实际 EQ 对脚步几乎没有增益，无法满足评测方案“先将 EQ 调到与 AI 脚步增益相近”的要求，因此无法回答“固定 EQ 调得更激进能否复制 FPS AI 的效果”。
为此构造匹配静态 EQ：以 FPS AI 在人工确认脚步上的平均频响（图~\ref{fig:resp}，1/3 倍频程平滑）设计 2049 阶线性相位 FIR 滤波器，并平移使其脚步增益中位数与 FPS AI 相同，
作为对“固定频响解释”的强对照；它由同一批脚步拟合，对 EQ 一方偏有利，使检验更保守。匹配静态 EQ 只用于排除“固定频响即可解释结果”，不作为 FPS AI 的产品竞争对照。

\figslot{fig08_fps_response}{0.82\linewidth}{FPS AI 在脚步上的平均频响（即匹配静态 EQ 的设计曲线）}{resp}

\subsection{混杂控制与统计}\label{sec:design}
可能导致误判的因素及控制手段见表~\ref{tab:threat}。主分析只使用人工确认、纯净度为前两档的事件，同电平比较的参照脚步也只用人工确认脚步；扩展分析加入模型高置信度事件。置信区间一律采用按片段聚类的自助法。

\input{tables/threat}

\section{实验结果}
\subsection{样本}
\input{tables/sample}

\subsection{产品主对比：原始 vs 实际 EQ vs FPS AI}\label{sec:product}
本节回答产品层面的问题：用户直接开启 C3 的实际 EQ 与开启 FPS AI，表现有何区别（表~\ref{tab:product}，表~\ref{tab:gain_cls}，图~\ref{fig:bars}）。
实际 EQ 对脚步的中位增益为 [[g_fo_eq]] dB，对枪声为 [[g_gun_eq]] dB，即基本不增强任何类别；FPS AI 对脚步为 [[g_fo]] dB，对枪声仅 [[g_gun]] dB。
两种产品模式的作用逻辑完全不同：实际 EQ 是作用于全部声音的固定频响调整，FPS AI 则主要增强脚步。
在非目标声音上，FPS AI 的误增强率（[[fer]]）与实际 EQ（[[fer_eq]]）相近，枪声频谱变化则小于实际 EQ（\ref{sec:h3} 节）。

\input{tables/product}
\input{tables/gain_cls}
\figslot{fig07_gain_by_class}{0.95\linewidth}{各类事件在三种处理下的中位增益（误差线为四分位距；匹配静态 EQ 为机制性辅助对照）}{bars}

\subsection{H1 脚步增强有效性}
FPS AI 对人工确认脚步的增益中位数为 [[g_fo]] dB，95\% CI 下限 [[ci_lo]] dB 高于 3 dB（表~\ref{tab:gain_a}），增强效果明确成立。
从分布看，[[in_claim]] 的脚步落在宣称的 3--9 dB，[[gt9]] 超过 9 dB，[[lt3]] 低于 3 dB；实测中位数位于宣传区间上沿，但较多事件的增益高于 9 dB，因此不表述为“3--9 dB 宣称完全成立”。
实际 EQ 对脚步为 [[g_fo_eq]] dB：脚步约 [[lf_v]]（VALORANT）与 [[lf_p]]（PUBG）的能量在 300 Hz 以下，正好落在两套 EQ 削减的频段。
$-35$ dBFS 以上的脚步仅 [[n_hi]] 个、来自 [[n_hiclip]] 个片段，增益主要由片段决定（[[hi_low_txt]] 为 [[hi_low_rng]] dB，其余片段中位 [[hi_norm_rng]] dB），样本不足以下结论。

\input{tables/gain_a}
\figslot{fig06_gain_vs_level}{0.98\linewidth}{各事件增益与原始电平（左：FPS AI；右：实际 EQ）}{level}

\subsection{H2 事件选择性：机制性检验（核心发现）}
产品主对比显示 FPS AI 增强脚步而基本不增强枪声（[[g_gun]] dB）。本节用两项机制性辅助对照检验这一差异能否由原始响度差异或固定频响解释。同电平比较（表~\ref{tab:gap}）显示枪声比脚步少增强 [[gap_gun]] dB、换弹少增强 [[gap_rel]] dB，
显著削弱了“只是对响声增强少”这一简单解释（$\pm$[[match_db]] dB 匹配可控制一阶响度影响，但不能排除所有与电平相关的非线性行为）；匹配静态 EQ 的同电平差仅 [[g_eqm]] dB，即脚步增益相同的固定 EQ 会同步抬高枪声（[[g_gun_eqm]] dB），因此观察到的选择性无法由固定静态频响解释。
脚步 [[g_fo]]、枪声 [[g_gun]]、同电平差 [[gap_gun]]、匹配静态 EQ [[g_eqm]] dB 相互印证，是本报告技术层面最核心的发现。
作为汇总描述，SEI 为 [[sei_ai]] dB，实际 EQ 与匹配静态 EQ 分别为 [[sei_eq]] 与 [[sei_eqm]] dB。
例外是烟雾弹、吃药声与爆能器一类（[[gap_ut]] dB，仅 [[n_ut_gap]] 个可匹配），受到的增强不低于同电平脚步。

图~\ref{fig:wave} 与图~\ref{fig:spec} 以一个典型脚步与一个响亮枪声为例对比四种处理。
波形上，FPS AI 将脚步放大 [[exf_ai]] dB，枪声基本不变（[[exg_ai]] dB）；匹配静态 EQ 对脚步的增益相近（[[exf_eqm]] dB），却同时把枪声抬高 [[exg_eqm]] dB；实际 EQ 对两者均略有衰减。
差分声谱（处理后减原始，红色为增强、蓝色为衰减）更直观：实际 EQ 在两类事件上呈现同一条固定的频带图案，匹配静态 EQ 在两类事件上整体变红，而 FPS AI 只在脚步上变红、在枪声上几乎无变化。
FPS AI 对脚步的增强集中在起振段与 2 kHz 以下，后段高频略有衰减，与固定频响的“全时段同一图案”明显不同。

\input{tables/gap}
\figslot{figb_waveforms_full}{\linewidth}{典型事件在四种处理下的波形，同一行纵轴刻度相同（上：脚步 \#[[ex_f]]；下：枪声 \#[[ex_g]]；标签为相对原始轨的增益与波形幅值倍数；匹配静态 EQ 为机制性辅助对照）}{wave}
\figslot{fig12_spectrograms}{\linewidth}{典型事件的原始声谱与差分声谱（上：脚步 \#[[ex_f]]；下：枪声 \#[[ex_g]]）。差分声谱为处理后减原始，红色为增强、蓝色为衰减，灰色为原始能量低于峰值 60 dB 的区域；标签为事件增益}{spec}

\subsection{H3 非目标声音保持性}\label{sec:h3}
本节检验 FPS AI 能否在增强脚步的同时较好保持主要非目标声音。枪声的频谱总变化（MSE）为 [[mse_ai]] dB$^2$，约为实际 EQ（[[mse_eq]] dB$^2$）的 [[mse_ratio]]（约 1/[[mse_frac]]）；
扣除整体电平差后，枪声的频谱形状变化为 [[mse_gs_ai]] dB$^2$，也显著小于实际 EQ（[[mse_gs_eq]] dB$^2$，约 1/[[mse_sfrac]]）（表~\ref{tab:mse}）。
非脚步 FER 为 [[fer]]，枪声为 [[fer_g]]（表~\ref{tab:fer}，图~\ref{fig:fer}），高于 10\% 的判定线。该线是本报告设定的工程容忍阈值（约每 10 个非脚步事件允许 1 个被明显增强），并非行业标准；H3 判为部分支持主要源于这一点。
被误增强的事件电平低、与脚步相近：[[n_boost_gun]] 个被增强的枪声位于 [[gb_lo]]～[[gb_hi]] dBFS，均为远处枪声，近处响亮枪声无一被增强；且集中在少数片段（附录表~\ref{tab:boosted}）。
FPS AI 对脚步本身的频谱形状改变（[[mse_fo_shape]] dB$^2$）高于实际 EQ（[[mse_fo_shape_eq]] dB$^2$），这符合其主动增强目标，但是否影响材质、距离等感知需要主观实验。
结论为\textbf{部分支持}：对响亮枪声保持较好，但对部分安静瞬态、换弹和道具声存在误增强。

\input{tables/mse}
\input{tables/fer}
\figslot{fig09_fer_curve}{0.95\linewidth}{误增强率随判定阈值的变化（点线为本报告阈值 3 dB；匹配静态 EQ 为机制性辅助对照）}{fer}

\subsection{H4 稳定性}\label{sec:robust}
H4 检验在不同目标游戏、事件纯净度、增益计算口径及合理标签噪声扰动下，核心选择性效应的方向是否稳定。
在上述条件下，同电平 $\Delta G$ 的点估计均为正（[[rob_rng]] dB，表~\ref{tab:robust}，图~\ref{fig:forest}），效应方向不变；
其中[[rob_wide_txt]]因可匹配枪声过少，CI 跨 0，单独不足以下结论。标注噪声模拟中（[[rob_n]] 条明确标注），选择性效应为 [[rob_base]] dB，
随机翻转 10\% 与 20\% 标注后分别为 [[rob10]] 与 [[rob20]] dB；需刻意挑错 [[rob_worst]] 条（[[rob_worst_p]]）才能抹平效应，远超盲审反映的不一致水平（[[disagree]]）。
两款目标游戏的增强幅度不同（脚步：PUBG [[g_pubg]] dB，VALORANT [[g_val]] dB），但都呈现脚步明显增强、枪声中位变化较小（[[gun_pubg]] / [[gun_val]] dB）的模式（表~\ref{tab:game}）。
两款游戏的一致结果不构成对其他游戏的泛化结论。

\input{tables/robust}
\figslot{fig10_robustness_forest}{0.95\linewidth}{稳定性检验：各条件下枪声的同电平增益差及 95\% CI（菱形为主分析；空心点表示 CI 跨过 0。H4 只看效应方向是否保持为正，黑色实线 0 dB 为其参考线；灰色虚线为 H2 的 3 dB 效果参考线，不作为各子集的判定标准）}{forest}
\input{tables/game}

\subsection{整体响度}\label{sec:loud}
针对“游戏整体音量未做改变”，对每段计算整段 RMS 与 50 ms 短时电平的中位数、99\% 分位数和最大值（表~\ref{tab:loud}，图~\ref{fig:loud}）。在含响亮声音的 [[n_loud]] 个片段中，
最大电平变化中位 [[dmax_loud]] dB，最响的声音未被放大，未观察到统一的全局正增益（但中位电平 $L_{50}$ 仍抬高 [[l50_loud]] dB，反映片段内较弱声音被增强）；在以脚步为主的 [[n_quiet]] 个安静片段中，最大电平随脚步抬高 [[dmax_quiet]] dB，
但处理后至多 [[fps_max_quiet]] dBFS，仍低于响亮片段的原始最大电平（[[raw_max_all]] dBFS）。

\input{tables/loud}
\figslot{fig11_loudness}{0.8\linewidth}{各片段原始最大短时电平与处理后的变化（虚线为分界）}{loud}

\subsection{其他观察}
\begin{itemize}
\item \textbf{即时性}：[[n_seq]] 个至少 4 步的脚步串中，第一步增益 [[sp_first]] dB，第 2--4 步 [[sp_later]] dB，逐串差值 [[sp_diff]] dB，未观察到多步预热现象（事件级测量不能排除毫秒级的启动过程）。
\item \textbf{突出度}：[[pr_n]] 个人工确认脚步的峰值抬高 [[pr_peak]] dB，前方背景抬高 [[pr_bg]] dB，突出度净增 [[pr_c]] dB（95\% CI [[pr_lo]]～[[pr_hi]]）；非脚步 [[pr_nf]] dB，实际 EQ 对脚步 [[pr_eq]] dB。
\item \textbf{动态性}：FPS AI 的频带增益随时间变化（标准差中位 [[tv_ai]] dB），实际 EQ 几乎不变（[[tv_eq]] dB）。
\item \textbf{底噪}：背景段增益中位 [[g_bg]] dB，但 [[fer_bg]] 的背景段超过 3 dB，纯净脚步前紧邻背景抬高 [[pre_bg]] dB。
\end{itemize}

\section{宣称核验与适用边界}
\subsection{假设判定}
\input{tables/verdict}
在产品层面，C3 实际 EQ 对脚步与枪声分别为 [[g_fo_eq]] / [[g_gun_eq]] dB，FPS AI 为 [[g_fo]] / [[g_gun]] dB：用户可用的两种模式中，只有 FPS AI 显著增强脚步。
在技术层面，核心发现为 H2 事件选择性。最具区分力的证据是匹配静态 EQ 对照：当固定静态 EQ 对脚步取得与 FPS AI 相同的增益时，枪声与背景段也被同步抬高（[[g_gun_eqm]] / [[g_bg_eqm]] dB），
而 FPS AI 对枪声的中位增益为 [[g_gun]] dB。观察到的选择性无法由固定静态频响解释，表现出明显的事件选择性处理特征；本实验从输出行为描述这一差异，不直接推断设备内部的识别机制。
H1 中脚步增强效果明确，但增益分布与 3--9 dB 区间仅部分重合（描述性结果，不影响判定）。H3 为部分支持，其边界见 \ref{sec:bound} 节。

\subsection{宣传册宣称逐条核验}
\input{tables/claims}
宣传册称模型“精准提取脚步声频段”。实测 FPS AI 在脚步上的平均频响近似全频段抬升（图~\ref{fig:resp}），与按频段调节的 EQ 不同，表现为事件级的增强；
这一描述基于输出行为，不代表对设备内部实现的推断。

\subsection{适用边界}\label{sec:bound}
\begin{enumerate}
\item 误增强集中在电平与脚步相近的安静瞬态（远处枪声、换弹、烟雾弹、吃药声）；
\item 增强脚步时紧邻背景同步抬高约 [[pr_bg]] dB，有效突出度（[[pr_c]] dB）小于峰值增益（[[pr_peak]] dB）；
\item 脚步音色被改变，是否影响材质与距离判断需主观实验确认；
\item 安静场景整体响度随脚步上升（Lmax 中位 [[dmax_quiet]] dB）；
\item $-30$ dBFS 以上的近距离响亮脚步缺乏样本；
\item 结论限于 VALORANT 与 PUBG 两款目标游戏，且只验证声学处理效果，不等价于证明玩家定位能力提升。
\end{enumerate}

\section{局限与后续工作}\label{sec:limit}
\begin{itemize}
\item 数据由品牌方提供，片段的截取规则、对局数量与录制设置无法独立核实；若截取时偏向特定场景，结论只代表这类场景；
\item 候选事件由原始轨频谱通量自动检测，缺少逐帧完整真值，因此候选检测的召回率无法直接评估，部分极弱、起振不明显或被背景掩盖的事件可能未进入后续分析。
三轨使用相同的候选时间点，这不影响三种处理之间的比较，但限制了结果对“全部游戏脚步”的代表性；
\item 数据来自真实对局且无分轨；人工确认的非脚步事件仅 [[n_nt]] 个，PUBG 枪声仅 [[n_gun_pubg]] 个，缺少爆炸、装拆包、技能声等类别；
\item 仅一名标注者，盲审证明的是稳定性而非绝对准确性；事件终点为自动估计；
\item 宣传册中的处理延迟（40 ms）无法用离线录音检验；
\item 本报告只评价声学效果，增强能否提升玩家的脚步检测与定位表现需主观听辨实验验证。
\end{itemize}
后续建议：在训练场受控录制分类别、分距离素材以获得天然真值并补齐响亮脚步与爆炸声；引入第二名标注者；开展双盲主观听辨实验。

\appendix
\section{误增强事件清单}
\input{tables/boosted}

\section{机器可识别性的探索性分析}
\textbf{鉴于候选检测器在原始游戏音频上的识别能力不足，本报告不使用机器可检测性作为 FPS AI 效果证据。}本附录作为负结果保留，供方法审查参考。

使用未参与标注的 PANNs（Cnn14\_DecisionLevelMax）作检测器并限定六类游戏声音，另以人工标注微调 PANNs（按片段留出测试）。三个版本先按整段 RMS 对齐，阈值在原始轨上确定。
只看单一类别时 FPS AI 检出率下降 [[z_drop]] 个百分点，但该类别对游戏脚步的绝对概率极低（阈值 [[z_thr]]）；限定六类或微调后变化均不显著（表~\ref{tab:detect}）。
这些检测器原始轨 AUC 仅约 0.8，未达评测方案准入要求，本分析不作为可检测性结论。

\input{tables/detect}

\section{复现说明}
脚本位于 \texttt{analysis/}，按以下顺序运行：
\begin{enumerate}
\item \texttt{analyze.py}：事件检测与三轨增益计算；
\item \texttt{embed.py}、\texttt{embed\_clap.py}：提取 AST / CLAP 嵌入；
\item \texttt{train\_labeler.py}：训练标注模型并生成各轮候选；
\item \texttt{extract\_clean.py}：导出三轨纯净事件库；
\item \texttt{report\_ad.py}：同电平对比与匹配静态 EQ；
\item \texttt{detectability.py}、\texttt{finetune\_panns.py}：附录 B 的探索性分析；
\item \texttt{report\_tex.py}：生成本报告的全部表格、图表数据与草图并编译。
\end{enumerate}
图表数据位于 \texttt{docs/report\_tex/figures/data/}；三轨纯净事件库位于 \texttt{analysis/clean\_events/}（[[n_clean]] 个事件）；
人工标注与标注工具位于 \texttt{analysis/labeling/}。

\section{参考文献}
\begin{enumerate}[label={[\arabic*]},leftmargin=2.2em]\small
\item Gemmeke J F, et al. Audio Set: An ontology and human-labeled dataset for audio events. ICASSP, 2017.
\item Kong Q, et al. PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM TASLP, 2020.
\item Gong Y, Chung Y A, Glass J. AST: Audio Spectrogram Transformer. Interspeech, 2021.
\item Wu Y, et al. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. ICASSP, 2023.
\item Settles B. Active learning literature survey. University of Wisconsin--Madison, 2009.
\item Bello J P, et al. A tutorial on onset detection in music signals. IEEE TSAP, 2005.
\item Mesaros A, Heittola T, Virtanen T. Metrics for polyphonic sound event detection. Applied Sciences, 2016.
\item Bristow-Johnson R. Cookbook formulae for audio EQ biquad filter coefficients.
\item Cohen J. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 1960.
\item Field C A, Welsh A H. Bootstrapping clustered data. JRSS-B, 2007.
\item Fosi Audio. C3 游戏声卡宣传册.
\end{enumerate}
\end{document}
"""

if __name__ == "__main__":
    main()
