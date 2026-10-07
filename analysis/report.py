"""Statistics + figures from results/events.csv and results/clips.csv."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
from analyze import BAND_NAMES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
rng = np.random.default_rng(0)
LEVEL_BINS = [-70, -50, -45, -40, -35, -30, -25, -20, 0]
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def label_events(ev, scheme):
    fb, fs, g = ev.p_foot_broad, ev.p_foot_strict, ev.p_gun
    nonfoot_score = ev[["p_gun", "p_knock", "p_vehicle"]].max(1)
    if scheme == "broad":
        foot = (fb >= 0.10) & (fb > 2 * nonfoot_score)
        non = (nonfoot_score >= 0.10) & (nonfoot_score > 2 * fb)
    else:  # strict: only AudioSet "Walk, footsteps"/"Run"/"Shuffle" and confident non-footstep
        foot = (fs >= 0.08) & (nonfoot_score < 0.03)
        non = (nonfoot_score >= 0.20) & (fb < 0.05)
    lab = np.where(foot, "footstep", np.where(non, "non-footstep", "ambiguous"))
    sub = np.where(non, ev[["p_gun", "p_knock", "p_vehicle"]].idxmax(1).str[2:], "")
    return lab, sub


def level_matched_diff(df, col):
    """Weighted mean over level bins of median(footstep) - median(non-footstep)."""
    df = df.assign(lvl=pd.cut(df.raw_peak, LEVEL_BINS))
    num = den = 0.0
    for _, g in df.groupby("lvl", observed=True):
        a = g[g.label == "footstep"][col]
        b = g[g.label == "non-footstep"][col]
        if len(a) >= 3 and len(b) >= 3:
            w = min(len(a), len(b))
            num += w * (a.median() - b.median())
            den += w
    return num / den if den else np.nan


def fe_regression(df, col):
    """OLS: col ~ is_footstep + level-bin FE + clip FE. Returns footstep coefficient."""
    d = df[df.label != "ambiguous"]
    X = pd.get_dummies(pd.cut(d.raw_peak, LEVEL_BINS).astype(str), drop_first=True, dtype=float)
    X = pd.concat([X, pd.get_dummies(d.game + d.clip_id.astype(str), drop_first=True, dtype=float)], axis=1)
    X.insert(0, "foot", (d.label == "footstep").astype(float).values)
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.values, d[col].values, rcond=None)
    return beta[1]


def cluster_boot(df, fn, n=1000):
    clips = df[["game", "clip_id"]].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in df.groupby(["game", "clip_id"])}
    vals = []
    for _ in range(n):
        pick = [tuple(clips[j]) for j in rng.integers(0, len(clips), len(clips))]
        bs = pd.concat([groups[k].assign(clip_id=f"{k[1]}_{m}") for m, k in enumerate(pick)])
        v = fn(bs)
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else np.array([np.nan, np.nan])


def section_stats(ev, scheme):
    out(f"\n==================== 标注方案: {scheme} ====================")
    out(ev.groupby(["label"]).size().to_string())
    out("非脚步子类: " + ev[ev.label == "non-footstep"]["subtype"].value_counts().to_dict().__repr__())
    cols = ["raw_peak", "raw_prom", "fps_gain", "eq_gain", "fps_contrast", "eq_contrast", "fps_bg_gain"]
    out("\n各类事件中位数 (dB):")
    out(ev.groupby("label")[cols].median().round(2).to_string())

    out("\n按原始峰值电平分箱的 FPS 突出度提升 (contrast, 中位数 dB / 事件数):")
    e = ev.assign(lvl=pd.cut(ev.raw_peak, LEVEL_BINS))
    t = e[e.label != "ambiguous"].pivot_table(index="lvl", columns="label", values="fps_contrast",
                                               aggfunc=["median", "count"], observed=False)
    out(t.round(2).to_string())
    t = e[e.label != "ambiguous"].pivot_table(index="lvl", columns="label", values="eq_contrast",
                                               aggfunc="median", observed=False)
    out("同上, EQ:")
    out(t.round(2).to_string())

    d = ev[ev.label != "ambiguous"]
    for proc in ["fps", "eq"]:
        for metric in ["contrast", "gain"]:
            col = f"{proc}_{metric}"
            print(f"  [{scheme}] bootstrap 置信区间: {col} ...", file=sys.stderr, flush=True)
            lm = level_matched_diff(d, col)
            lm_ci = cluster_boot(d, lambda x: level_matched_diff(x, col), n=500)
            b = fe_regression(d, col)
            b_ci = cluster_boot(d, lambda x: fe_regression(x, col), n=500)
            out(f"{proc.upper():3s} {metric:8s}: 电平匹配 脚步-非脚步 = {lm:+.2f} dB "
                f"[95%CI {lm_ci[0]:+.2f}, {lm_ci[1]:+.2f}] | 回归(电平+片段固定效应) = {b:+.2f} dB "
                f"[95%CI {b_ci[0]:+.2f}, {b_ci[1]:+.2f}]")


def fig_scatter(ev, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"footstep": "tab:red", "non-footstep": "tab:blue", "ambiguous": "lightgrey"}
    names = {"footstep": "脚步", "non-footstep": "非脚步(枪声/敲击/载具)", "ambiguous": "未能判定"}
    for k, proc in enumerate(["fps", "eq"]):
        for lab in ["ambiguous", "non-footstep", "footstep"]:
            g = ev[ev.label == lab]
            ax[k].scatter(g.raw_peak, g[f"{proc}_contrast"], s=8, alpha=0.5, c=colors[lab], label=names[lab])
        for lab in ["footstep", "non-footstep"]:
            g = ev[ev.label == lab].assign(lvl=pd.cut(ev.raw_peak, LEVEL_BINS))
            m = g.groupby("lvl", observed=True)[f"{proc}_contrast"].agg(["median", "count"])
            m = m[m["count"] >= 3]
            x = [iv.mid for iv in m.index]
            ax[k].plot(x, m["median"], "-o", c=colors[lab], lw=2.5, mec="k")
        ax[k].axhline(0, c="k", lw=0.8)
        ax[k].set_title(f"{'FPS StepSense' if proc == 'fps' else 'EQ'}：事件相对背景突出度的变化")
        ax[k].set_xlabel("原始音频中事件峰值电平 (dBFS, 20ms RMS)")
        ax[k].set_ylim(-15, 25)
    ax[0].set_ylabel("突出度提升 Δcontrast (dB)")
    ax[0].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def fig_bands(ev, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for k, proc in enumerate(["fps", "eq"]):
        for lab, c, nm in [("footstep", "tab:red", "脚步事件"), ("non-footstep", "tab:blue", "非脚步事件")]:
            g = ev[ev.label == lab]
            cols = [f"{proc}_band_{b}" for b in BAND_NAMES]
            med = g[cols].median().values
            q1, q3 = g[cols].quantile(0.25).values, g[cols].quantile(0.75).values
            ax[k].plot(BAND_NAMES, med, "-o", c=c, label=nm)
            ax[k].fill_between(BAND_NAMES, q1, q3, color=c, alpha=0.15)
        ax[k].axhline(0, c="k", lw=0.8)
        ax[k].set_title(f"{'FPS StepSense' if proc == 'fps' else 'EQ'}：事件窗口内各倍频程增益")
        ax[k].set_xlabel("倍频程中心频率 (Hz)")
    ax[0].set_ylabel("增益 (dB, 相对原始)")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def fig_clips(cl, ev, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(cl))
    names = [f"{'P' if g == 'pubg' else 'V'}{c}" for g, c in zip(cl.game, cl.clip_id)]
    ax[0].bar(x - 0.2, cl.fps_tv, 0.4, label="FPS", color="tab:orange")
    ax[0].bar(x + 0.2, cl.eq_tv, 0.4, label="EQ", color="tab:green")
    ax[0].set_xticks(x, names, rotation=90)
    ax[0].set_ylabel("频段增益随时间的标准差 (dB)")
    ax[0].set_title("时变性：EQ≈固定滤波器，FPS 随内容变化")
    ax[0].legend()
    s = ev[ev.label != "ambiguous"].groupby(["game", "clip_id", "label"])["fps_contrast"].median().unstack()
    s = s.reindex(pd.MultiIndex.from_frame(cl[["game", "clip_id"]]))
    ax[1].bar(x - 0.2, s.get("footstep"), 0.4, label="脚步事件", color="tab:red")
    ax[1].bar(x + 0.2, s.get("non-footstep"), 0.4, label="非脚步事件", color="tab:blue")
    ax[1].set_xticks(x, names, rotation=90)
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set_ylabel("FPS 突出度提升中位数 (dB)")
    ax[1].set_title("逐片段：FPS 对脚步 / 非脚步事件的突出度提升")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


if __name__ == "__main__":
    ev0 = pd.read_csv(os.path.join(RES, "events.csv"))
    cl = pd.read_csv(os.path.join(RES, "clips.csv"))

    out("==================== 片段级 ====================")
    out(cl[["game", "clip_id", "dur", "fps_gain", "eq_gain", "fps_tv", "eq_tv"]].round(2).to_string())
    out(f"\n时变性(频段增益标准差) 中位数: FPS={cl.fps_tv.median():.2f} dB, EQ={cl.eq_tv.median():.2f} dB, "
        f"Wilcoxon p={wilcoxon(cl.fps_tv, cl.eq_tv).pvalue:.2e}")

    v2 = os.path.join(RES, "events_labeled_v2.csv")
    if os.path.exists(v2):
        evv = pd.read_csv(v2, keep_default_na=False)
        sources = [("model", evv), ("human_only", evv[evv.human_label != ""])]
        tag = "_v2"
    else:
        sources = []
        for scheme in ["broad", "strict"]:
            e = ev0.copy()
            e["label"], e["subtype"] = label_events(e, scheme)
            sources.append((scheme, e))
        tag = ""

    for scheme, ev in sources:
        section_stats(ev, scheme)
        if scheme == "human_only":
            g = ev.assign(subtype=ev.subtype.replace("", "(无备注)"))
            t = g.groupby(["human_label", "subtype"]).agg(
                n=("fps_contrast", "size"), raw_peak=("raw_peak", "median"),
                fps_contrast=("fps_contrast", "median"), eq_contrast=("eq_contrast", "median"),
                fps_gain=("fps_gain", "median"))
            out("\n按人工标注和备注分组 (中位数 dB; f=脚步 g=枪声 o=其他 m=混合 x=不确定):")
            out(t.round(2).to_string())

        # clip-level paired test: footstep contrast FPS vs EQ
        fc = ev[ev.label == "footstep"].groupby(["game", "clip_id"]).filter(lambda g: len(g) >= 3)
        pc = fc.groupby(["game", "clip_id"])[["fps_contrast", "eq_contrast"]].median()
        if len(pc) < 5:
            out(f"\n含≥3个脚步事件的片段只有 {len(pc)} 个，跳过片段级配对检验")
            continue
        out(f"\n含≥3个脚步事件的片段数={len(pc)}; 脚步突出度提升中位数 FPS={pc.fps_contrast.median():+.2f} dB, "
            f"EQ={pc.eq_contrast.median():+.2f} dB, 配对 Wilcoxon p={wilcoxon(pc.fps_contrast, pc.eq_contrast).pvalue:.2e}")

        if scheme in ("broad", "model"):
            fig_scatter(ev, os.path.join(RES, f"fig_contrast_vs_level{tag}.png"))
            fig_bands(ev, os.path.join(RES, f"fig_band_gain{tag}.png"))
            fig_clips(cl, ev, os.path.join(RES, f"fig_per_clip{tag}.png"))
        if scheme == "broad":
            ev.to_csv(os.path.join(RES, "events_labeled.csv"), index=False)

    with open(os.path.join(RES, f"summary{tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES))
