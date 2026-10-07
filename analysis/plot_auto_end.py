"""Illustrate the automatic event-end estimate on a few human-labelled events."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_triplet, SR
from analyze import rms_db, HOP
from window_sensitivity import auto_end, F2S

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ev = pd.read_csv("results/events_labeled_v2.csv", keep_default_na=False)
picks = [("f", "脚步"), ("g", "枪声"), ("f", "脚步（背景嘈杂）")]
rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, len(picks), figsize=(15, 4))
for ax, (code, title) in zip(axes, picks):
    g = ev[ev.human_label == code]
    g = g[g.raw_prom < 9] if "嘈杂" in title else g[g.raw_prom > 15]
    g = g[g.t > 0.7]
    durs = ev[["game", "clip_id", "t"]].groupby(["game", "clip_id"]).t.max()
    g = g[[durs[(r.game, r.clip_id)] - r.t > 1.2 for r in g.itertuples()]]
    r = g.iloc[int(rng.integers(len(g)))]
    a, _ = load_triplet(r.game, int(r.clip_id))
    lv = rms_db(a["raw"])
    k = int(r.t * F2S)
    lo, hi = k - int(0.6 * F2S), k + int(1.0 * F2S)
    tt = (np.arange(lo, hi) - k) / F2S * 1000
    bg = np.percentile(lv[lo: k - int(0.05 * F2S)], 20)
    seg = lv[k: k + int(0.08 * F2S)]
    peak = seg.max()
    end = auto_end(lv, k) * 1000
    ax.plot(tt, lv[lo:hi], c="k", lw=1)
    ax.axvspan(-600, -50, color="grey", alpha=0.15, label="背景窗口 (起点前 50–600 ms)")
    ax.axhline(bg, c="grey", ls="--", lw=1, label=f"背景 = {bg:.0f} dB (第 20 百分位)")
    ax.axhline(bg + 3, c="tab:blue", ls=":", lw=1.5, label="条件①：背景 + 3 dB")
    ax.axhline(peak - 20, c="tab:orange", ls=":", lw=1.5, label="条件②：峰值 − 20 dB")
    ax.plot((np.argmax(seg)) / F2S * 1000, peak, "o", c="tab:red", label=f"峰值 = {peak:.0f} dB (起点后 80 ms 内)")
    ax.axvline(0, c="cyan", lw=2, label="起点（人工标注）")
    ax.axvline(end, c="tab:green", lw=2, label=f"估计终点 = {end:.0f} ms")
    ax.axvspan(0, end, color="tab:green", alpha=0.12)
    ax.set_title(f"{title}：{r.game} test{int(r.clip_id)}  t={r.t:.2f}s")
    ax.set_xlabel("相对起点的时间 (ms)")
    ax.set_xlim(-600, 1000)
    ax.set_ylim(bg - 15, peak + 6)
    ax.legend(fontsize=7.5, loc="upper right")
axes[0].set_ylabel("20 ms RMS 电平 (dBFS)")
plt.tight_layout()
plt.savefig("results/fig_auto_end.png", dpi=100)
