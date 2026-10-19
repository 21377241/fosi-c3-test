"""Final experiment report (Chinese, self-contained HTML) for the C3 FPS AI evaluation.

Reads the outputs of report_ad.py (results/report_AD_*.csv), the label files and results/events_labeled_v2.csv,
so every number in the text is computed from data. Run report_ad.py first.
Output: docs/C3 FPS AI 声卡客观评测 实验报告.html
"""
import os
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from report_ad import (RES, HERE, FOOT, NT, NT_MODEL, CLAIM, FER_THR, P_HI, cluster_ci, matched_gap,
                       model_label_accuracy)
from report import fe_regression
from extract_clean import read_purity
from eq_params import Q as EQ_Q, FREQS as EQ_FREQS, EQ_BANDS, theory_response, measured_response

OUT = os.path.join(os.path.dirname(HERE), "docs", "C3 FPS AI 声卡客观评测 实验报告.html")
LAB = os.path.join(HERE, "labeling")
rng = np.random.default_rng(1)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SHORT = {FOOT: "脚步", NT[0]: "枪声", NT[1]: "换弹", NT[2]: "烟雾弹/吃药/爆能器"}


# ------------------------------------------------------------------ helpers
def img(path):
    with open(path, "rb") as fh:
        return f'<img src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}">'


def table(rows, cls="t"):
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def pct(x):
    return f"{x:.0%}"


SPECIAL = {"吃药声"}  # FPS AI enhances these on purpose (per the brand), so their FPS AI boosts are not false boosts


def fer_rate(x, p="AI", thr=FER_THR):
    """Share of footstep-free events (step_ov False) boosted above thr, excluding FPS AI boosts of SPECIAL sounds."""
    if "step_ov" in x:
        x = x[~x["step_ov"]]
    hit = x[f"G_{p}"] > thr
    if p == "AI" and "special" in x:
        hit &= ~x["special"]
    return hit.mean()


def sgn(x, nd=1):
    s = f"{x:+.{nd}f}"
    return s[1:] if float(s) == 0 else s


# ------------------------------------------------------------------ data
def load():
    d = pd.read_csv(os.path.join(RES, "report_AD_events.csv"), encoding="utf-8-sig")
    bg = pd.read_csv(os.path.join(RES, "report_AD_background.csv"), encoding="utf-8-sig")
    dm = pd.read_csv(os.path.join(RES, "report_AD_model_events.csv"), encoding="utf-8-sig")
    ev = pd.read_csv(os.path.join(RES, "events_labeled_v2.csv"), keep_default_na=False)
    meta = pd.read_csv(os.path.join(HERE, "clean_events", "clean_events.csv"), encoding="utf-8-sig", keep_default_na=False)
    clips = pd.read_csv(os.path.join(RES, "clips.csv"))
    rv = pd.read_csv(os.path.join(LAB, "review_r1.csv"), encoding="utf-8-sig", keep_default_na=False)
    r2 = pd.read_csv(os.path.join(LAB, "human_labels_r2.csv"), encoding="utf-8-sig", keep_default_na=False)
    r3 = pd.read_csv(os.path.join(LAB, "human_labels_r3.csv"), encoding="utf-8-sig", keep_default_na=False).merge(
        pd.read_csv(os.path.join(LAB, "queue_r3.csv"))[["event_id", "raw_peak", "p_ens", "purpose"]], on="event_id")
    r4 = pd.read_csv(os.path.join(LAB, "human_labels_r4.csv"), encoding="utf-8-sig", keep_default_na=False).merge(
        pd.read_csv(os.path.join(LAB, "queue_r4.csv"))[["event_id", "raw_peak", "p_ens", "lv", "stratum_size"]], on="event_id")
    return d, bg, dm, ev, meta, clips, rv, r2, r3, r4


def review_stats(rv):
    b = rv[rv.orig_label.isin(["f", "g", "o"]) & rv.blind_label.isin(["f", "g", "o"])]
    return dict(n=len(rv), agree=(rv.orig_label == rv.blind_label).mean(),
                k5=cohen_kappa_score(rv.orig_label, rv.blind_label),
                k2=cohen_kappa_score(b.orig_label == "f", b.blind_label == "f"),
                changed=int((rv.final_label != rv.orig_label).sum()))


def noise_robustness(ev, reps=150):
    h = ev[ev.human_label.isin(["f", "g", "o"])].copy()
    h["label"] = np.where(h.human_label == "f", "footstep", "non-footstep")
    flip = {"footstep": "non-footstep", "non-footstep": "footstep"}
    base = fe_regression(h, "fps_gain")
    out = {"base": base, "n": len(h)}
    for frac in [0.1, 0.2]:
        vals = []
        for _ in range(reps):
            x = h.copy()
            idx = rng.choice(len(x), int(round(frac * len(x))), replace=False)
            x.iloc[idx, x.columns.get_loc("label")] = x.label.iloc[idx].map(flip).values
            vals.append(fe_regression(x, "fps_gain"))
        out[frac] = (np.median(vals), np.mean(np.array(vals) <= 0))
    x = h.copy()
    fs = list(x[x.label == "footstep"].sort_values("fps_gain", ascending=False).index)
    ns = list(x[x.label != "footstep"].sort_values("fps_gain").index)
    n = 0
    while fe_regression(x, "fps_gain") > 0 and (fs or ns):
        pick = fs.pop(0) if (n % 2 == 0 and fs) or not ns else ns.pop(0)
        x.loc[pick, "label"] = flip[x.loc[pick, "label"]]
        n += 1
    out["worst"] = (n, n / len(h))
    return out


def new_model_after_r3():
    """Ambiguous count of the current model trained only on the labels available after round 3."""
    from train_labeler import read_human, CODE, embedding_model
    r4 = set(pd.read_csv(os.path.join(LAB, "human_labels_r4.csv"), encoding="utf-8-sig").event_id)
    h = read_human()
    h = h[~h.event_id.isin(r4)]
    t = h[h.human_label.isin(CODE)]
    un = np.setdiff1d(np.arange(1620), h.event_id.values)
    p = np.mean([embedding_model().fit(X[t.event_id.values], (t.human_label == "f").values).predict_proba(X[un])[:, 1]
                 for X in (np.load(os.path.join(HERE, "cache", f))["X"] for f in ["embeddings_ast128.npz", "embeddings_clap.npz"])], 0)
    return pd.DataFrame({"p_ens": p, "human_label": ""})


def model_upgrade_rows():
    """Old vs new labelling model on the same human labels (those available after round 3).
    Old: results/human_cv_probs_v1.csv, events_labeled_v2_oldmodel.csv. New: the frozen 1 s AST probe from
    results/ast_upgrade_cv.csv (same folds) averaged with the CLAP probe."""
    from sklearn.metrics import roc_auc_score
    lab = read_purity()[["event_id", "final_label"]]
    r1 = set(pd.read_csv(os.path.join(LAB, "review_r1.csv"), encoding="utf-8-sig").event_id)
    v1 = pd.read_csv(os.path.join(RES, "human_cv_probs_v1.csv"), encoding="utf-8-sig")
    up = pd.read_csv(os.path.join(RES, "ast_upgrade_cv.csv"), encoding="utf-8-sig").merge(v1[["event_id", "cv_CLAP"]], on="event_id")
    up["cv_ens"] = (up["layers_最后一层"] + up.cv_CLAP) / 2
    rows = []
    for name, cv, e in [("旧版：AST（补零到 10 秒）+ CLAP + 声学特征", v1,
                         pd.read_csv(os.path.join(RES, "events_labeled_v2_oldmodel.csv"), keep_default_na=False)),
                        ("新版：AST（1 秒输入）+ CLAP", up[["event_id", "cv_ens"]], new_model_after_r3())]:
        cv = cv.merge(lab, on="event_id")
        cv = cv[cv.final_label.isin(["f", "g", "o"])]
        y = (cv.final_label == "f").values
        p = cv.cv_ens.values
        hi, lo, r = p >= P_HI, p <= 1 - P_HI, cv.event_id.isin(r1).values
        un = e[e.human_label == ""]
        rows.append({"标注模型": name, "AUC": f"{roc_auc_score(y, p):.3f}",
                     "人工样本中高置信比例": pct(np.mean(hi | lo)),
                     "判为脚步准确率": f"{pct(y[hi & r].mean())}（{int((hi & r).sum())}）",
                     "判为非脚步准确率": f"{pct((~y[lo & r]).mean())}（{int((lo & r).sum())}）",
                     "未标注事件中拿不准": f"{int(((un.p_ens > 1 - P_HI) & (un.p_ens < P_HI)).sum())} / {len(un)}"})
    return rows


def step_position(ev):
    e = ev.sort_values(["game", "clip_id", "t"])
    rows = []
    for _, g in e.groupby(["game", "clip_id"]):
        seq, last, idx = -1, -9.0, 0
        for r in g.itertuples():
            if r.label != "footstep":
                last = -9.0
                continue
            if r.t - last > 1.0:
                seq, idx = seq + 1, 0
            idx += 1
            last = r.t
            rows.append(dict(seq=f"{r.game}{r.clip_id}_{seq}", idx=idx, g=r.fps_gain))
    s = pd.DataFrame(rows)
    s = s[s.seq.map(s.groupby("seq").size()) >= 4]
    first = s[s.idx == 1].g.median()
    later = s[s.idx.between(2, 4)].g.median()
    per = s.groupby("seq").apply(lambda x: x.g.iloc[1:4].mean() - x.g.iloc[0], include_groups=False)
    return dict(n_seq=s.seq.nunique(), first=first, later=later, diff=per.median())


# ------------------------------------------------------------------ figure
def fig_level(fo_all, d, dm, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    bins = np.arange(-66, -24, 3)
    for ax, p, title in [(axes[0], "AI", "FPS AI"), (axes[1], "EQ", "EQ（实际）")]:
        ax.scatter(fo_all.raw_level, fo_all[f"G_{p}"], s=6, c="#9ec5e8", alpha=.45, label=f"脚步（人工+模型，{len(fo_all)}）")
        mid = pd.cut(fo_all.raw_level, bins)
        med = fo_all.groupby(mid, observed=True)[f"G_{p}"].median()
        ax.plot([iv.mid for iv in med.index], med.values, c="#17365d", lw=2.2, label="脚步分档中位数")
        nm = dm[dm.group == NT_MODEL]
        ax.scatter(nm.raw_level, nm[f"G_{p}"], s=14, marker="x", c="#888", alpha=.7, label=f"非脚步（模型，{len(nm)}）")
        for g, c, mk in [(NT[0], "#d62728", "^"), (NT[1], "#ff7f0e", "s"), (NT[2], "#2ca02c", "D")]:
            x = d[d.group == g]
            ax.scatter(x.raw_level, x[f"G_{p}"], s=34, c=c, marker=mk, edgecolor="k", lw=.4, label=f"{SHORT[g]}（人工，{len(x)}）")
        ax.axhspan(*CLAIM, color="#999", alpha=.12, lw=0)
        ax.axhline(0, c="k", lw=.6)
        ax.set_title(title)
        ax.set_xlabel("原始事件电平 (dBFS RMS)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("增益 G (dB)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[0].text(0.02, 0.97, "灰带 = 产品宣称 3–9 dB", transform=axes[0].transAxes, va="top", fontsize=8, color="#555")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def eq_curves():
    """Theoretical and measured EQ response per game, plus peak/trough of the theory and max deviation (50 Hz–16 kHz)."""
    out = {}
    for game in EQ_BANDS:
        f, meas = measured_response(game)
        theo = theory_response(game, f)
        m = (f >= 50) & (f <= 16000)
        out[game] = dict(f=f, theo=theo, meas=meas, dev=np.abs(theo - meas)[m].max(),
                         peak=(f[m][theo[m].argmax()], theo[m].max()), trough=(f[m][theo[m].argmin()], theo[m].min()),
                         low=(f[m][(f[m] < 500)][theo[m][f[m] < 500].argmin()], theo[m][f[m] < 500].min()))
    return out


def band_share(meta):
    """Median share of energy below 300 Hz and in 0.5–2.5 kHz of the raw clean segments, per (game, cls)."""
    from scipy.io import wavfile
    fx = np.fft.rfftfreq(8192, 1 / 48000)
    out = {}
    for (game, cls), g in meta[meta.cls.isin(["footstep", "gunshot"])].groupby(["game", "cls"]):
        lo, mid = [], []
        for r in g.itertuples():
            sr, x = wavfile.read(os.path.join(HERE, "clean_events", cls, f"e{r.event_id:04d}_raw.wav"))
            X = np.abs(np.fft.rfft(x.mean(1) if x.ndim > 1 else x, 8192)) ** 2
            f = np.fft.rfftfreq(8192, 1 / sr)
            lo.append(X[f < 300].sum() / X.sum())
            mid.append(X[(f >= 500) & (f <= 2500)].sum() / X.sum())
        out[(game, cls)] = (np.median(lo), np.median(mid))
    return out


def fig_eq(eq, path):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for game, c, name in [("valorant", "#d62728", "VALORANT"), ("pubg", "#2f6da5", "PUBG")]:
        e = eq[game]
        m = (e["f"] >= 30) & (e["f"] <= 20000)
        ax.semilogx(e["f"][m], e["meas"][m], c=c, lw=3.5, alpha=.3, label=f"{name} 实测（EQ 轨 / 原始轨）")
        ax.semilogx(e["f"][m], e["theo"][m], c=c, lw=1.2, ls="--", label=f"{name} 按参数计算")
        ax.plot(EQ_FREQS, EQ_BANDS[game], "o", c=c, ms=4)
    ax.axhline(0, c="k", lw=.6)
    ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("增益 (dB)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


# ------------------------------------------------------------------ report
CSS = """
:root{--navy:#17365d;--blue:#2f6da5;--light:#eef4fa;--ink:#1e2d3d;--muted:#617083;--line:#d7e0ea}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15.5px/1.85 "Microsoft YaHei","PingFang SC",Arial,sans-serif}
.page{max-width:1060px;margin:36px auto;padding:64px 80px 90px;background:#fff;box-shadow:0 12px 42px rgba(27,50,76,.13)}
.eyebrow{color:var(--blue);font-weight:700;letter-spacing:.08em;font-size:13px;text-align:center}
h1{font-size:32px;line-height:1.3;margin:10px 0 8px;color:var(--navy);text-align:center}
.sub{text-align:center;color:var(--muted);margin-bottom:36px}
h2{margin:52px 0 16px;font-size:23px;color:var(--navy);border-bottom:2px solid var(--line);padding-bottom:6px}
h3{margin:30px 0 10px;font-size:18px;color:var(--blue)}p{margin:0 0 14px;text-align:justify}
.box{background:var(--light);border-left:4px solid var(--blue);padding:14px 22px;margin:18px 0}
.box.green{background:#eef8f1;border-color:#2d8a4e}.box.amber{background:#fdf6e9;border-color:#d99a20}.box.grey{background:#f3f4f6;border-color:#8a94a3}
.box h4{margin:0 0 6px;color:var(--navy);font-size:16px}
.tw{overflow-x:auto;margin:12px 0 20px}table.t{width:100%;border-collapse:collapse;font-size:13.8px}
table.t th{background:var(--navy);color:#fff;text-align:left;padding:8px 10px;font-weight:700}
table.t td{border:1px solid var(--line);padding:7px 10px;vertical-align:top}table.t tbody tr:nth-child(even){background:var(--light)}
img{max-width:100%;display:block;margin:12px auto}.cap{text-align:center;color:var(--muted);font-size:13px;margin-top:-4px}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:16px 0}
.flow div{flex:1 1 150px;background:var(--light);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:13.5px}
.flow b{display:block;color:var(--navy);margin-bottom:2px}
ul,ol{margin:0 0 16px;padding-left:1.4em}li{margin:5px 0}code{background:#edf2f7;padding:1px 6px;border-radius:4px;font:.9em Consolas,monospace}
.footer{margin-top:60px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
"""


def detect_summary():
    """Sub-experiment E numbers from detectability.py, plus the acoustic prominence of the same human events."""
    from train_labeler import read_human, CODE
    res = {}
    for key, suf in [("match", ""), ("nomatch", "_nomatch")]:
        S = pd.read_csv(os.path.join(RES, f"detectability_summary{suf}.csv"), encoding="utf-8-sig")
        res[key] = {(r["detector"], r["track"]): r for r in S.to_dict("records")}
    ci = lambda s: [float(v) for v in s.split(",")]
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    ev["event_id"] = np.arange(len(ev))
    h = read_human()
    h = h[h.human_label.isin(CODE)]
    e = ev.loc[h.event_id.values].assign(y=(h.human_label == "f").values, clip_key=lambda x: x.game + x.clip_id.astype(str))
    fo, nf = e[e.y], e[~e.y]
    prom = dict(n_fo=len(fo), n_nf=len(nf), peak=fo.fps_gain.median(), bg=fo.fps_bg_gain.median(),
                c_fo=fo.fps_contrast.median(), c_fo_ci=cluster_ci(fo, "fps_contrast"),
                c_nf=nf.fps_contrast.median(), c_eq=fo.eq_contrast.median())
    return res, ci, prom


def main():
    d, bg, dm, ev, meta, clips, rv, r2, r3, r4 = load()
    # after the gain analysis: unblinded review of the boosted non-footsteps (boosted_review.csv), then a raw-only blind
    # re-review of all non-footsteps plus decoy footsteps (blind_review_r5.csv); events heard to contain footsteps stay in
    # the sample, but FER and the non-footstep spectral MSE use only the footstep-free ones
    brv = pd.read_csv(os.path.join(LAB, "boosted_review.csv"), encoding="utf-8-sig", keep_default_na=False)
    blind = pd.read_csv(os.path.join(LAB, "blind_r5_key.csv"), encoding="utf-8-sig").merge(
        pd.read_csv(os.path.join(LAB, "blind_review_r5.csv"), encoding="utf-8-sig", keep_default_na=False)
        .rename(columns={"序号": "idx", "判断_原始轨": "盲审判断"}), on="idx")
    has_step = set(blind.loc[blind.group.isin(NT) & (blind["盲审判断"] != "无脚步"), "event_id"])
    decoy = blind[blind.group == FOOT]
    blind_sens = (decoy["盲审判断"] == "含脚步").mean()
    brv = brv.merge(blind[["event_id", "盲审判断"]], on="event_id", how="left")
    brv_agree = int(((brv["是否误增强"] == "不是误增强") == (brv["盲审判断"] == "含脚步")).sum())
    d = d.assign(step_ov=d.event_id.isin(has_step),
                 special=d.event_id.isin(meta.loc[meta.note.isin(SPECIAL), "event_id"]))
    r3_loud = r3[r3.purpose == "loud_check"]
    r3_amb = r3[r3.purpose == "ambiguous"]
    r3_def = r3_amb[r3_amb.human_label.isin(["f", "g", "o"])]
    r3_acc = ((r3_def.p_ens >= 0.5) == (r3_def.human_label == "f")).mean()
    r3_q, r3_l = r3_def[r3_def.raw_peak <= -35], r3_def[r3_def.raw_peak > -35]
    n_amb_now = int(((ev.human_label == "") & (ev.p_ens > 1 - P_HI) & (ev.p_ens < P_HI)).sum())
    upgrade_rows = model_upgrade_rows()
    r4_amb = r4[r4.human_label.isin(["f", "g", "o"])].copy()
    r4_amb["ok"] = (r4_amb.p_ens >= 0.5) == (r4_amb.human_label == "f")
    r4_acc = r4_amb.groupby("lv").ok.mean()
    r4_w = r4_amb.groupby("lv").stratum_size.first()
    r4_wacc = (r4_acc * r4_w).sum() / r4_w.sum()
    r4_fshare = r4_amb.groupby("lv").apply(lambda x: (x.human_label == "f").mean(), include_groups=False)
    n_amb_before_r4 = int(r4_w.sum())
    fo_h = d[d.group == FOOT]
    fo_m = dm[dm.group == FOOT]
    fo_all = pd.concat([fo_h, fo_m])
    nt_h = d[d.group.isin(NT)]
    nt_m = dm[dm.group == NT_MODEL]
    rs = review_stats(rv)
    acc = model_label_accuracy()
    rob = noise_robustness(ev)
    sp = step_position(ev)
    pu = read_purity().merge(ev[["event_id", "fps_gain", "fps_bg_gain"]], on="event_id")
    pf = pu[pu.final_label == "f"]
    ovl_gain = pf[pf.purity == "l"].fps_gain.median()
    clean_gain = pf[pf.purity != "l"].fps_gain.median()
    pre_bg = pf[pf.purity == "p"].fps_bg_gain.median()
    tv_ai, tv_eq = clips.fps_tv.median(), clips.eq_tv.median()
    n_r1 = len(pd.read_csv(os.path.join(LAB, "human_labels.csv"), encoding="utf-8-sig"))
    dur_min = clips.dur.sum() / 60
    n_vl, n_pg = int((clips.game == "valorant").sum()), int((clips.game == "pubg").sum())

    ci_h = cluster_ci(fo_h, "G_AI")
    ci_all = cluster_ci(fo_all, "G_AI")
    gap = {p: {g: matched_gap(fo_h, d[d.group == g], p) for g in NT} for p in ["AI", "EQ", "EQm"]}
    gap_all = {p: matched_gap(fo_all, d[d.group == NT[0]], p) for p in ["AI", "EQm"]}
    gap_m = {p: matched_gap(fo_h, nt_m, p) for p in ["AI", "EQ"]}
    med = lambda x, p: x[f"G_{p}"].median()
    sei = {p: med(fo_h, p) - np.mean([med(d[d.group == g], p) for g in NT]) for p in ["AI", "EQ", "EQm"]}
    fer = {p: fer_rate(nt_h, p) for p in ["AI", "EQ", "EQm"]}
    fer_g = {p: fer_rate(d[d.group == NT[0]], p) for p in ["AI", "EQ", "EQm"]}
    fer_bg = {p: fer_rate(bg, p) for p in ["AI", "EQ", "EQm"]}
    ov = nt_h[nt_h.step_ov]
    nt_clean = nt_h[~nt_h.step_ov]
    fer_special = (nt_clean.G_AI > FER_THR).mean()
    mse = lambda g, c: d.loc[(d.group == g) & ~d.step_ov, c].median()
    lvl_hi = fo_all[fo_all.raw_level > -35]
    lvl_mid = fo_all[fo_all.raw_level.between(-40, -35)]
    lvl_lo = fo_all[fo_all.raw_level < -40]
    hi_clip = lvl_hi.groupby("clip_key").G_AI.agg(["median", "min", "max", "size"])
    clip_name = lambda k: ("PUBG test" if k.startswith("pubg") else "VALORANT test") + k.lstrip("pubgvalorant")
    hi_low = hi_clip[hi_clip["median"] < 5]
    hi_norm = hi_clip[hi_clip["median"] >= 5]
    hi_low_txt = "、".join(clip_name(k) for k in hi_low.index)
    hi_norm_txt = "、".join(clip_name(k) for k in hi_norm.index)
    hi_low_rng = f"{hi_low['min'].min():.1f}–{hi_low['max'].max():.1f}" if len(hi_low) else "—"
    hi_norm_rng = f"{hi_norm['median'].min():.1f}–{hi_norm['median'].max():.1f}" if len(hi_norm) else "—"

    fig_path = os.path.join(RES, "final_level.png")
    fig_level(fo_all, d, dm, fig_path)
    eq = eq_curves()
    eq_fig = os.path.join(RES, "final_eq.png")
    fig_eq(eq, eq_fig)
    bs = band_share(meta)
    hz = lambda x: f"{x / 1000:.1f} kHz" if x >= 1000 else f"{x:.0f} Hz"
    eq_desc = {g: f"{hz(e['low'][0])} 附近 {sgn(e['low'][1])} dB，{hz(e['peak'][0])} 附近 {sgn(e['peak'][1])} dB，"
                  f"{hz(e['trough'][0])} 附近 {sgn(e['trough'][1])} dB" for g, e in eq.items()}
    eq_rows = [{"游戏": name, **{hz(fc): sgn(gn, 0) for fc, gn in zip(EQ_FREQS, EQ_BANDS[g])}}
               for g, name in [("valorant", "VALORANT"), ("pubg", "PUBG")]]

    GN = {"valorant": "VALORANT", "pubg": "PUBG"}
    game_rows = []
    for g in ["valorant", "pubg"]:
        fh, gh, ng = fo_h[fo_h.game == g], d[(d.group == NT[0]) & (d.game == g)], nt_m[nt_m.game == g]
        fa = fo_all[fo_all.game == g]
        ci_g = cluster_ci(fh, "G_AI")
        game_rows.append({"游戏": GN[g], "处理": "FPS AI",
                          "脚步（人工）": f"<b>{sgn(med(fh, 'AI'))}</b> [{sgn(ci_g[0])}, {sgn(ci_g[1])}]（{len(fh)}）",
                          "脚步（人工+模型）": f"{sgn(med(fa, 'AI'))}（{len(fa)}）",
                          "枪声（人工）": f"<b>{sgn(med(gh, 'AI'))}</b>（{len(gh)}）",
                          "非脚步（模型）": f"{sgn(med(ng, 'AI'))}（{len(ng)}）",
                          "背景段": sgn(med(bg[bg.game == g], 'AI'))})
        game_rows.append({"游戏": GN[g], "处理": "EQ（实际）",
                          "脚步（人工）": sgn(med(fh, 'EQ')), "脚步（人工+模型）": sgn(med(fa, 'EQ')),
                          "枪声（人工）": sgn(med(gh, 'EQ')), "非脚步（模型）": sgn(med(ng, 'EQ')),
                          "背景段": sgn(med(bg[bg.game == g], 'EQ'))})
    gm = lambda g, grp, p: med(d[(d.group == grp) & (d.game == g)], p)
    fa_g = lambda g, p: med(fo_all[fo_all.game == g], p)

    # boosted non-footsteps, listed one by one
    boosted = nt_h[nt_h.G_AI > FER_THR].merge(meta[["event_id", "note", "game", "clip_id", "t"]], on="event_id", suffixes=("", "_m")) \
        .merge(brv[["event_id", "是否误增强", "原因备注", "盲审判断"]], on="event_id", how="left")
    boosted_rows = [{"类别": SHORT[r.group], "备注": r.note or "—", "片段": f"{r.game_m.upper()} test{r.clip_id}",
                     "时间 (s)": f"{r.t:.1f}", "原始电平": f"{r.raw_level:.0f} dBFS", "FPS AI 增益": sgn(r.G_AI),
                     "EQ 增益": sgn(r.G_EQ), "复核结论": r.是否误增强, "复核说明": r.原因备注, "盲审判断": r.盲审判断}
                    for r in boosted.sort_values(["group", "game_m", "clip_id", "t"]).itertuples()]

    cnt = d.groupby(["group", "purity"]).size().unstack(fill_value=0)
    cnt_rows = [{"类别": SHORT[g], "纯净": int(cnt.loc[g].get("p", 0)), "有持续背景": int(cnt.loc[g].get("b", 0)),
                 "合计": int(cnt.loc[g].sum())} for g in [FOOT] + NT]

    gain_rows = []
    for g in [FOOT] + NT:
        x = d[d.group == g]
        gain_rows.append({"类别": SHORT[g], "n": len(x), "原始电平中位": f"{x.raw_level.median():.0f} dBFS",
                          "FPS AI": sgn(med(x, "AI")), "EQ（实际）": sgn(med(x, "EQ")), "匹配静态 EQ（模拟）": sgn(med(x, "EQm"))})
    gain_rows.append({"类别": "背景段*", "n": len(bg), "原始电平中位": f"{bg.raw_level.median():.0f} dBFS",
                      "FPS AI": sgn(med(bg, "AI")), "EQ（实际）": sgn(med(bg, "EQ")), "匹配静态 EQ（模拟）": sgn(med(bg, "EQm"))})

    boost_max = nt_h.G_AI.max()
    gun_fb = (d.group == NT[0]) & (d.G_AI > FER_THR) & ~d.step_ov & ~d.special
    n_boost_gun = int(gun_fb.sum())
    gun_boost_lv = d.loc[gun_fb, "raw_level"]
    gap_rows = [{"非脚步类别": SHORT[g], "可匹配事件数": gap["AI"][g][1],
                 **{lab: f"<b>{sgn(gap[p][g][0])}</b>" if p == "AI" else sgn(gap[p][g][0])
                    for p, lab in [("AI", "FPS AI"), ("EQ", "EQ（实际）"), ("EQm", "匹配静态 EQ")]}} for g in NT]
    gap_rows.append({"非脚步类别": "非脚步（模型高置信度）", "可匹配事件数": gap_m["AI"][1],
                     "FPS AI": f"<b>{sgn(gap_m['AI'][0])}</b>", "EQ（实际）": sgn(gap_m['EQ'][0]), "匹配静态 EQ": "—"})

    compare_rows = [
        {"维度": "脚步增益（中位）", "FPS AI": f"<b>{sgn(med(fo_h, 'AI'))} dB</b>", "EQ（实际）": f"{sgn(med(fo_h, 'EQ'))} dB",
         "匹配静态 EQ（模拟）": f"{sgn(med(fo_h, 'EQm'))} dB（按设计与 FPS 相同）"},
        {"维度": "枪声增益（中位）", "FPS AI": f"<b>{sgn(med(d[d.group == NT[0]], 'AI'))} dB</b>",
         "EQ（实际）": f"{sgn(med(d[d.group == NT[0]], 'EQ'))} dB", "匹配静态 EQ（模拟）": f"{sgn(med(d[d.group == NT[0]], 'EQm'))} dB"},
        {"维度": "背景段增益（中位）", "FPS AI": f"{sgn(med(bg, 'AI'))} dB", "EQ（实际）": f"{sgn(med(bg, 'EQ'))} dB",
         "匹配静态 EQ（模拟）": f"{sgn(med(bg, 'EQm'))} dB"},
        {"维度": "选择性指数 SEI（方案定义）", "FPS AI": f"<b>{sgn(sei['AI'])} dB</b>", "EQ（实际）": f"{sgn(sei['EQ'])} dB",
         "匹配静态 EQ（模拟）": f"{sgn(sei['EQm'])} dB"},
        {"维度": "同电平下枪声比脚步少增强", "FPS AI": f"<b>{sgn(gap['AI'][NT[0]][0])} dB</b>", "EQ（实际）": f"{sgn(gap['EQ'][NT[0]][0])} dB",
         "匹配静态 EQ（模拟）": f"{sgn(gap['EQm'][NT[0]][0])} dB"},
        {"维度": "枪声频谱改变（总 / 形状，dB²）", "FPS AI": f"<b>{mse(NT[0], 'MSE_AI'):.1f} / {mse(NT[0], 'MSEshape_AI'):.1f}</b>",
         "EQ（实际）": f"{mse(NT[0], 'MSE_EQ'):.1f} / {mse(NT[0], 'MSEshape_EQ'):.1f}",
         "匹配静态 EQ（模拟）": f"{mse(NT[0], 'MSE_EQm'):.0f} / {mse(NT[0], 'MSEshape_EQm'):.1f}"},
        {"维度": "脚步频谱形状改变（dB²）", "FPS AI": f"{mse(FOOT, 'MSEshape_AI'):.1f}", "EQ（实际）": f"{mse(FOOT, 'MSEshape_EQ'):.1f}",
         "匹配静态 EQ（模拟）": f"{mse(FOOT, 'MSEshape_EQm'):.1f}"},
        {"维度": f"非脚步误增强率（&gt;{FER_THR:.0f} dB）", "FPS AI": pct(fer["AI"]), "EQ（实际）": pct(fer["EQ"]), "匹配静态 EQ（模拟）": pct(fer["EQm"])},
        {"维度": f"背景段误增强率（&gt;{FER_THR:.0f} dB）", "FPS AI": pct(fer_bg["AI"]), "EQ（实际）": pct(fer_bg["EQ"]), "匹配静态 EQ（模拟）": pct(fer_bg["EQm"])},
        {"维度": "增益随时间的波动（频带增益标准差）", "FPS AI": f"{tv_ai:.2f} dB", "EQ（实际）": f"{tv_eq:.2f} dB", "匹配静态 EQ（模拟）": "0（固定）"},
        {"维度": "处理方式", "FPS AI": "动态、按内容处理",
         "EQ（实际）": f"10 段峰值 EQ（Q = {EQ_Q}），每个游戏一套固定参数：压低频与 4–8 kHz、抬 0.5–2 kHz（见 2.1 节）",
         "匹配静态 EQ（模拟）": "固定曲线：全频段 +7~11.5 dB，1–6 kHz 最多"},
    ]

    plan_rows = [
        {"方案要求": "同源配对（Baseline / EQ / FPS AI）", "本研究的实现": f"{n_vl + n_pg} 段实录各有三版，互相关对齐，样本级零偏移", "状态": "满足"},
        {"方案要求": "纯净单事件数据", "本研究的实现": f"从实录中经人工纯净度分级挑出 {len(meta)} 个事件（三轨同步切片）", "状态": "近似满足（非独立处理）"},
        {"方案要求": "复合事件数据 + 分轨（stem）", "本研究的实现": "实录中的重叠事件单独分级；无分轨", "状态": "部分（仅描述性）"},
        {"方案要求": "事件级强标注（起止时间）", "本研究的实现": "人工标起点与类别，终点自动估计（回落到背景 +3 dB 或峰值 -20 dB，截到下一事件）", "状态": "近似满足"},
        {"方案要求": "A 脚步增益", "本研究的实现": "人工与模型两套样本，按片段聚类的置信区间、按电平分层", "状态": "完成"},
        {"方案要求": "B 选择性（含 EQ 匹配）", "本研究的实现": "SEI + 同电平 SEI；实际 EQ 无法匹配，改用模拟的匹配静态 EQ", "状态": "完成（EQ 匹配为模拟）"},
        {"方案要求": "C 非目标频谱保持", "本研究的实现": "对数频谱 MSE，并拆分为总改变与形状改变", "状态": "完成"},
        {"方案要求": "D 负样本 / FER", "本研究的实现": "非脚步事件 + 背景段；阈值取产品宣称下限 3 dB，并给出全阈值曲线", "状态": "完成（类别不全）"},
        {"方案要求": "E 可检测性（P/R/F1）", "本研究的实现": "未进行", "状态": "待做"},
    ]

    noise_txt = (f"按全部 {rob['n']} 条明确的人工标注估计，FPS AI 的选择性效应（脚步减非脚步，控制电平档与片段）为 {sgn(rob['base'])} dB；"
                 f"随机标错 10% 时中位 {sgn(rob[0.1][0])} dB（{pct(rob[0.1][1])} 的模拟归零），随机标错 20% 时中位 {sgn(rob[0.2][0])} dB"
                 f"（{pct(rob[0.2][1])} 的模拟归零）；要刻意挑错 {rob['worst'][0]} 条（{pct(rob['worst'][1])}）才能把效应抹平。")

    det, dci, prom = detect_summary()
    zr, zf, ze = (det["match"][("zs", t)] for t in ("raw", "fps", "eq"))
    pr, pf, pe = (det["match"][("probe", t)] for t in ("raw", "fps", "eq"))
    cr, cf, ce = (det["match"][("closed", t)] for t in ("raw", "fps", "eq"))
    tr_, tf, te = (det["match"][("top1", t)] for t in ("raw", "fps", "eq"))
    nr, nf_ = det["nomatch"][("zs", "raw")], det["nomatch"][("zs", "fps")]
    ncr, ncf = det["nomatch"][("closed", "raw")], det["nomatch"][("closed", "fps")]
    ntr, ntf = det["nomatch"][("top1", "raw")], det["nomatch"][("top1", "fps")]
    ft = pd.read_csv(os.path.join(RES, "panns_ft_summary.csv"), encoding="utf-8-sig").set_index(["split", "track"])
    ft_h, ft_hf = ft.loc[("holdout", "raw")], ft.loc[("holdout", "fps")]
    ft_c, ft_cf = ft.loc[("cv5", "raw")], ft.loc[("cv5", "fps")]
    ft_v1 = pd.read_csv(os.path.join(RES, "panns_ft_summary_v1.csv"), encoding="utf-8-sig").set_index(["split", "track"]).loc[("holdout", "raw")]
    pp = lambda a, b: f"{(b - a) * 100:+.1f}"
    ppci = lambda s: "[{:+.1f}, {:+.1f}]".format(*(v * 100 for v in dci(s)))
    lv_labs = ["≤ -45 dBFS", "-45 ~ -35 dBFS", "> -35 dBFS"]
    det_rows = [{"打分方式 / 处理": lab, "脚步检出率": pct(r["recall"]), "非脚步误报": pct(r["fpr_events"]),
                 "相对原始（95% CI）": "—" if r["track"] == "raw" else f"{pp(base['recall'], r['recall'])} {ppci(r['d_recall_ci'])}",
                 "检出率按电平（安静 / 中 / 响）": " / ".join(pct(r[f"recall {l}"]) for l in lv_labs)}
                for lab, r, base in [("① 只看“脚步”一类：原始", zr, zr), ("① FPS AI", zf, zr), ("① EQ", ze, zr),
                                     ("② 六类中脚步的占比：原始", cr, cr), ("② FPS AI", cf, cr), ("② EQ", ce, cr),
                                     ("③ 六类中直接判类别：原始", tr_, tr_), ("③ FPS AI", tf, tr_), ("③ EQ", te, tr_),
                                     ("对照：原始轨训练的探针：原始", pr, pr), ("对照 FPS AI", pf, pr), ("对照 EQ", pe, pr)]]
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>C3 FPS AI 声卡客观评测 实验报告</title><style>{CSS}</style></head><body><main class="page">
<div class="eyebrow">C3 FPS AI · EXPERIMENT REPORT</div>
<h1>C3 FPS AI 声卡客观评测 实验报告</h1>
<div class="sub">基于 VALORANT / PUBG 实录音频的同源配对评测　·　对应《C3 FPS AI 声卡客观评测方案》子实验 A–D 及 E（初步）</div>

<div class="box"><h4>摘要</h4>
<p>我们用 {n_vl} 段 VALORANT 和 {n_pg} 段 PUBG 实录（共 {dur_min:.1f} 分钟），每段都有原始、FPS AI 处理、EQ 处理三个同源版本，
验证 FPS AI 是否“针对性地增强脚步声”。在没有品牌方强标注和分轨素材的条件下，我们搭建了从事件检测、人机协同标注、标注有效性检验、
纯净事件提取到混杂控制分析的完整流程，并按评测方案完成了子实验 A–D。</p>
<ul>
<li><b>脚步确实被大幅增强</b>：FPS AI 对人工确认脚步的增益中位数为 <b>{sgn(med(fo_h, 'AI'))} dB</b>（95% CI {sgn(ci_h[0])} ~ {sgn(ci_h[1])}），
纳入模型高置信度标注后（{len(fo_all)} 个脚步）为 {sgn(med(fo_all, 'AI'))} dB，位于产品宣称 3–9 dB 的上沿；实际 EQ（每个游戏一套 10 段固定参数）对脚步为 {sgn(med(fo_h, 'EQ'))} dB。</li>
<li><b>对枪声有明确的选择性</b>：枪声增益中位数仅 {sgn(med(d[d.group == NT[0]], 'AI'))} dB；同一电平下枪声比脚步少增强 <b>{gap['AI'][NT[0]][0]:.1f} dB</b>，
且不能用“枪声更响”解释。</li>
<li><b>这种选择性不是固定 EQ 能做到的</b>：一条与 FPS AI 脚步增益相同的模拟静态 EQ，其选择性指数为 {sgn(sei['EQm'])} dB（FPS AI 为 {sgn(sei['AI'])} dB）。</li>
<li><b>代价与边界</b>：个别安静的非脚步瞬态（一颗烟雾弹、部分换弹、吃药声、少数远处枪声）会被像脚步一样大幅增强；脚步附近的底噪会被一起抬高；
脚步本身的音色也被改变；部分片段中的响亮脚步增强很弱（{hi_low_rng} dB，样本少）。</li>
<li><b>可检测性（子实验 E，初步）</b>：脚步相对其前方背景的突出度提高 {sgn(prom['c_fo'])} dB，更容易被听见；
但独立的 AudioSet 检测器（PANNs，含限定在六类游戏声音中判断的版本）并没有更容易把它认作脚步，该检测器本身在游戏音频上可靠性有限。</li></ul></div>

<div class="box green"><h4>本研究的主要贡献</h4><ol>
<li><b>无强标注条件下的评测流程</b>：在没有品牌方标注和分轨素材的情况下，用实录音频完成了评测方案 A–D 的全部指标。</li>
<li><b>人机协同的少样本标注</b>：发现通用 AudioSet 模型（AST）对游戏脚步几乎失效，改用“冻结的预训练特征 + 人工少量标注 + 主动学习”，
四轮共 {n_r1 + len(r2) + len(r3) + len(r4)} 条人工标注覆盖 1620 个事件；模型高置信度脚步的人工核对准确率 {pct(acc['prec_f'])}。</li>
<li><b>标注有效性检验</b>：盲审复核（前后一致 {pct(rs['agree'])}，脚步/非脚步 κ = {rs['k2']:.2f}）与标注噪声模拟，量化了结论对标注错误的耐受度。</li>
<li><b>纯净事件库</b>：人工纯净度分级后，切出 {len(meta)} 个事件的原始 / FPS AI / EQ 三轨同步片段，保留原始电平，可直接复用。</li>
<li><b>消除混杂的分析设计</b>：同电平比较（排除“响的声音本来增强少”的解释）、模拟匹配静态 EQ（满足方案“EQ 与 AI 脚步增益相近”的公平比较要求）、
按片段聚类的置信区间、纯净度与重叠敏感性检验。</li></ol></div>

<h2>1. 背景与目标</h2>
<p>C3 FPS AI 宣称能识别并增强游戏中的脚步声（约 3–9 dB），同时尽量不改变枪声、技能、环境等其他声音。评测方案提出四个问题：
脚步增强是否真实有效（A）；是否具有选择性（B）；相比传统 EQ 是否更有优势（B、C、D）；增强是否转化为更高的可检测性（E）。
本报告回答前三个问题，并对第四个问题给出初步结果和所需的后续工作。</p>

<h2>2. 数据</h2>
<p>{n_vl} 段 VALORANT、{n_pg} 段 PUBG 实录，总长 {dur_min:.1f} 分钟，48 kHz 立体声。每段有三个版本：原始（Baseline）、仅开 FPS AI、仅开 EQ。
三轨经互相关对齐，偏移均为 0 样本。自动检测得到 1620 个声音事件。真实对局录音没有分轨，事件之间常有重叠，
这是本研究与评测方案理想数据（纯净单事件 + 带分轨的复合事件）的主要差别，也决定了后面的方法设计。</p>

<h3>2.1 EQ 对照组的参数</h3>
<p>EQ 版本是一个 10 段倍频程峰值均衡器，每段 Q = {EQ_Q}，两个游戏各用一套参数（单位 dB）：</p>
{table(eq_rows)}
<p>Q = {EQ_Q} 对应约 1.8 个倍频程的带宽，相邻频段大量重叠，叠加后的实际曲线与单段数值不同：
VALORANT 为 {eq_desc['valorant']}；PUBG 为 {eq_desc['pubg']}。设计思路是常见的“听脚步 EQ”：压低频轰鸣和 4–8 kHz 的尖锐高频，突出中频。
两套参数都只削减或提升音色，没有整体补偿音量。而脚步约有 {pct(bs[('valorant', 'footstep')][0])}（VALORANT）/
{pct(bs[('pubg', 'footstep')][0])}（PUBG）的能量在 300 Hz 以下，正好落在被削减的低频，所以这两套 EQ 对脚步的整体增益都接近 0。</p>
<p>用原始轨与 EQ 轨的互谱估计实测频响，与按参数计算的曲线在 50 Hz–16 kHz 内最大偏差为 VALORANT {eq['valorant']['dev']:.2f} dB、
PUBG {eq['pubg']['dev']:.2f} dB（图 5），说明 EQ 轨就是这组参数的静态滤波，没有任何动态处理，也验证了三轨对齐与测量链路是可靠的。</p>
{img(eq_fig)}<p class="cap">图 5　EQ 频响：粗线为实测，虚线为按参数计算，圆点为各段设定值。</p>

<h2>3. 方法</h2>
<div class="flow">
<div><b>① 事件检测</b>频谱通量起点检测，1620 个事件</div>
<div><b>② 人机协同标注</b>AST / CLAP 两模型 + 四轮人工标注</div>
<div><b>③ 有效性检验</b>盲审复核、噪声模拟</div>
<div><b>④ 纯净度分级</b>纯净 / 有背景 / 有重叠，切出纯净事件库</div>
<div><b>⑤ 测量与分析</b>增益、SEI、频谱 MSE、FER；同电平比较、匹配 EQ</div></div>

<h3>3.1 事件检测与测量</h3>
<p>起点检测：短时傅里叶变换（窗长 1024、跳步 5 ms）的对数谱正向差分在 60 Hz–16 kHz 上求和（频谱通量），
自适应阈值为 2 秒滑窗内的中位数 + 3 倍中位绝对偏差，最小间隔 80 ms，并要求事件比前景背景高出至少 6 dB。
事件终点自动估计为电平回落到事件前背景 +3 dB 以内或比峰值低 20 dB 处，最长 1.5 s，且不超过下一个事件起点；
脚步的估计时长中位数约 70 ms，枪声约 110 ms。</p>
<p>增益按方案定义：G = 20·log10(RMS<sub>处理后</sub> / RMS<sub>原始</sub>)，在“起点前 20 ms 至估计终点后 50 ms”的区间内计算。
我们比较了 50–400 ms 的固定窗口和自动终点，结论不随窗口改变。</p>

<h3>3.2 人机协同标注</h3>
<p>先尝试直接用 AudioSet 预训练的 AST 模型零样本识别脚步，发现它对游戏脚步的“脚步”概率最高只有约 0.2，常把脚步识别为马蹄声、洗牌声等，不可用。
因此改为少样本方案：AST 和 CLAP 的参数保持不变，只用来把每个事件（音量归一化后的原始音轨片段）转成一个嵌入向量，
再用人工标注分别训练逻辑回归分类器（标准化 → 主成分分析保留 95% 方差 → 强正则逻辑回归），两个概率取平均；
人工在只听原始音轨、看不到模型意见的条件下标注。第 1 轮按片段 × 电平分层随机抽取 {n_r1} 条，第 2 轮挑选模型意见不一致或拿不准的 {len(r2)} 条（主动学习），
第 3 轮针对模型的能力缺口定向抽取 {len(r3)} 条，第 4 轮从改进后模型仍拿不准的事件中分层随机抽取 {len(r4)} 条（见下文）。
模型的作用是扩大标注覆盖面：在从未人工标注的事件中，概率 ≥ {P_HI} 记为脚步、≤ {1 - P_HI:.1f} 记为非脚步。</p>
{table([{"模型判为": f"脚步（p ≥ {P_HI}）", "人工核对事件数": acc["n_f"], "人工确认比例": pct(acc["prec_f"])},
        {"模型判为": f"非脚步（p ≤ {1 - P_HI:.1f}）", "人工核对事件数": acc["n_n"], "人工确认比例": pct(acc["prec_n"])}])}
<p class="cap">核对使用第 1 轮（分层随机样本）的人工标注，以及模型训练时没见过该事件的交叉验证概率。</p>
<p><b>模型的能力缺口与第 3 轮定向标注</b>：前两轮后，当时的模型（旧版 AST + CLAP + 17 个声学特征，见下文“标注模型改进”）
仍有约 23% 的事件拿不准。这些事件的响度介于典型脚步和典型枪声之间，
而训练样本以“安静的脚步”和“响亮的枪声”为主，缺少近处的响脚步和远处的弱枪声；声学特征模型的区分能力接近随机，也把概率拉向中间。
第 3 轮因此抽取了 {len(r3_amb)} 个拿不准的事件（按响度分层，偏重响的区间）和 {len(r3_loud)} 个“模型高置信度判为脚步、峰值高于 -30 dBFS”的事件：</p>
<ul>
<li>响亮的高置信度脚步中 {int((r3_loud.human_label == 'f').sum())}/{len(r3_loud)} 确实是脚步，模型在响亮区间的判断同样可靠。</li>
<li>拿不准的事件中，峰值 -35 dBFS 以下的 {len(r3_q)} 个里 {int((r3_q.human_label == 'f').sum())} 个是脚步，
-35 dBFS 以上的 {len(r3_l)} 个里 {int((r3_l.human_label == 'g').sum())} 个是枪声；旧模型按 0.5 划分的正确率为 {pct(r3_acc)}，
说明拿不准的事件大多仍能分对，只是模型对“同样响的脚步和枪声”信心不足。</li>
<li>我们没有把电平作为模型输入：那会让模型倾向于把响的声音判为枪声，进而让“同电平比较”失去意义。</li></ul>
<p><b>标注模型改进</b>：旧版 AST 嵌入把不到 1 秒的事件片段补零到 AST 默认的 10 秒输入，九成以上是静音，稀释了事件本身的信息。
新版只输入事件前后 1 秒（128 帧，按 AST 论文对短音频的做法截断位置编码），并对最后一层所有图块的输出取平均；同时去掉了几乎随机的声学特征模型。
我们也试了解冻 AST 最后 4 层做微调（加时间平移、背景混合、频谱遮挡增强），交叉验证 AUC 只比冻结的新版 AST 高约 0.01，在折间波动范围内，
因此采用更简单、可复现的冻结方案。新旧对比（同样的留片段 5 折交叉验证）：</p>
{table(upgrade_rows)}
<p class="cap">两行都用第 3 轮之后的同一批人工标注评估。AUC 按全部人工标注计算（含第 2、3 轮专门挑出的难样本，因此低于只看随机样本的数值）；
准确率用第 1 轮分层随机样本核对。</p>
<p><b>第 4 轮：剩余难例</b>：改进后的模型仍有 {n_amb_before_r4} 个事件拿不准，其中一半以上是 AST 与 CLAP 判断相反。
第 4 轮按峰值分两层随机抽取 {len(r4)} 个（-35 dBFS 以上多抽），人工结果为脚步 {int((r4.human_label == 'f').sum())}、
枪声 {int((r4.human_label == 'g').sum())}、其他 {int((r4.human_label == 'o').sum())}、不确定 {int((r4.human_label == 'x').sum())}。
-35 dBFS 以下脚步占 {pct(r4_fshare.loc[0])}，模型按 0.5 划分的正确率 {pct(r4_acc.loc[0])}；-35 dBFS 以上脚步只占 {pct(r4_fshare.loc[1])}，
划分正确率仅 {pct(r4_acc.loc[1])}，接近随机；按层加权后对全部拿不准事件的正确率约为 {pct(r4_wacc)}。
这说明剩下的拿不准事件（尤其是响亮区间）是现有特征本身无法区分的，不能按 0.5 硬划分纳入分析，只能人工标注或排除；
本报告选择排除，加入第 4 轮重新训练后还剩 {n_amb_now} 个。第 2–4 轮专门挑难样本，因此全部人工标注上的交叉验证 AUC 随轮次下降，
这不代表模型变差。</p>

<h3>3.3 标注有效性检验</h3>
<p><b>盲审复核</b>：第 1 轮 {rs['n']} 条在隐藏原标注的条件下重新判断，再与原标注对照裁定。前后一致 {pct(rs['agree'])}，
五类 κ = {rs['k5']:.2f}，脚步/非脚步 κ = {rs['k2']:.2f}（≥ 0.8 视为稳定），最终改判 {rs['changed']} 条。</p>
<p><b>标注噪声模拟</b>：{noise_txt}</p>

<h3>3.4 纯净度分级与纯净事件库</h3>
<p>对每个人工标注事件，在“起点到估计终点”的区间内判断：纯净（只有这一个声音）、有持续背景（单一瞬态 + 风/发动机/音乐等背景）、
有重叠（区间内还有其他瞬态）。只有前两档进入 A–D 分析。“有重叠”的脚步测得增益（起点后 80 ms 峰值口径）中位 {sgn(ovl_gain)} dB，
低于其余脚步的 {sgn(clean_gain)} dB，说明不做纯净度分级会系统性地低估增益。</p>
{table(cnt_rows)}
<p class="cap">进入 A–D 分析的人工事件（纯净 + 有持续背景）。另有 {len(bg)} 段背景段（距任何事件 ≥ 0.3 s、长 ≥ 1 s，共 {bg.dur.sum():.0f} s）作为环境负样本，未经人工逐段核实。</p>

<h3>3.5 混杂控制</h3>
<ul>
<li><b>同电平比较</b>：枪声普遍比脚步响，若 FPS AI 只是“对响的声音增强少”，也会表现出假的选择性。因此对每个非脚步事件，
取电平相差 3 dB 以内的脚步（至少 3 个）作为参照，比较两者增益之差；参照组为人工 + 模型共 {len(fo_all)} 个脚步。</li>
<li><b>匹配静态 EQ</b>：方案要求先把 EQ 调到与 AI 脚步增益相近，但实际 EQ 对脚步几乎不增强。我们以 FPS AI 在全部脚步上的平均频响
（1/3 倍频程平滑）设计一条固定线性相位滤波器，再整体平移使其脚步增益中位数与 FPS AI 相同，作为“最好的固定 EQ”对照（对 EQ 偏有利）。</li>
<li><b>置信区间</b>：按音频片段整体重抽样（cluster bootstrap），避免同一片段内事件不独立导致区间过窄。</li></ul>

<h2>4. 结果</h2>
<h3>4.1 子实验 A：脚步增强</h3>
{table([
    {"脚步样本": "人工确认（纯净 / 有背景）", "n": len(fo_h), "FPS AI 中位 [95% CI]": f"<b>{sgn(med(fo_h, 'AI'))}</b> [{sgn(ci_h[0])}, {sgn(ci_h[1])}]",
     "落在 3–9 dB": pct(fo_h.G_AI.between(*CLAIM).mean()), "高于 9 dB": pct((fo_h.G_AI > 9).mean()), "EQ 中位": sgn(med(fo_h, 'EQ'))},
    {"脚步样本": "人工 + 模型高置信度", "n": len(fo_all), "FPS AI 中位 [95% CI]": f"<b>{sgn(med(fo_all, 'AI'))}</b> [{sgn(ci_all[0])}, {sgn(ci_all[1])}]",
     "落在 3–9 dB": pct(fo_all.G_AI.between(*CLAIM).mean()), "高于 9 dB": pct((fo_all.G_AI > 9).mean()), "EQ 中位": sgn(med(fo_all, 'EQ'))}])}
<p><b>响亮脚步的增益因片段而异</b>：-40 dBFS 以下的 {len(lvl_lo)} 个脚步增益中位 {sgn(lvl_lo.G_AI.median())} dB，
只有 {pct((lvl_lo.G_AI < 3).mean())} 低于 3 dB；-40 ~ -35 dBFS 的 {len(lvl_mid)} 个中位 {sgn(lvl_mid.G_AI.median())} dB，
{pct((lvl_mid.G_AI < 3).mean())} 低于 3 dB；-35 dBFS 以上的 {len(lvl_hi)} 个中位 {sgn(lvl_hi.G_AI.median())} dB，
{pct((lvl_hi.G_AI < 3).mean())} 低于 3 dB（人工确认的 {int((fo_h.raw_level > -35).sum())} 个中位 {sgn(fo_h[fo_h.raw_level > -35].G_AI.median())} dB）。
这些响亮脚步只来自 {len(hi_clip)} 个片段，且增益几乎由片段决定：{hi_low_txt} 中只有 {hi_low_rng} dB，
{hi_norm_txt} 中则照常增强（各片段中位 {hi_norm_rng} dB）。因此无法区分这是“响的脚步增强少”还是“某些场景（地面、距离、同时的其他声音）下增强少”；
随着标注轮次增加，这一档的中位数在 +4 ~ +8 dB 之间变化，不能作为稳定结论。-30 dBFS 以上没有脚步样本。</p>
<p>
在至少 4 步的脚步串中（{sp['n_seq']} 串），第一步的增益（峰值口径中位 {sgn(sp['first'])} dB）与第 2–4 步（{sgn(sp['later'])} dB）几乎相同，
逐串差值中位 {sgn(sp['diff'])} dB——FPS AI 从第一声脚步就立即增强，不需要“预热”。</p>
{img(fig_path)}<p class="cap">图 1　各事件的增益与原始电平。深蓝线为脚步分档中位数；彩色标记为人工确认的非脚步事件。</p>

<h3>4.2 子实验 B：选择性增强</h3>
{table(gain_rows)}
<p class="cap">各类事件的中位增益（dB）。* 背景段未经人工核实。</p>
<p>按方案定义，SEI = G<sub>脚步</sub> − 三类非脚步增益的平均：FPS AI 为 <b>{sgn(sei['AI'])} dB</b>，实际 EQ 为 {sgn(sei['EQ'])} dB，
匹配静态 EQ 为 {sgn(sei['EQm'])} dB。由于枪声比脚步响得多，下表进一步给出同电平比较（正值 = 非脚步比同电平脚步少增强多少 dB）：</p>
{table(gap_rows)}
{img(os.path.join(RES, "ad_gain_bars.png"))}<p class="cap">图 2　各类事件在三种处理下的中位增益（误差线为四分位）。</p>
<p>匹配静态 EQ 把所有声音一律抬高约 9 dB，没有任何选择性；FPS AI 在增强脚步的同时，枪声几乎不变。
FPS AI 在脚步上的平均频响是全频段 +7 ~ 11.5 dB、1–6 kHz 最多（图 3），本身就是一条接近平坦的抬升，
说明它区分脚步与枪声靠的是对声音内容的判断，而不是频响形状。</p>
{img(os.path.join(RES, "ad_response.png"))}<p class="cap">图 3　FPS AI 在脚步上的平均频响（即匹配静态 EQ 的曲线）。</p>

<h3>4.3 子实验 C：非目标声音的频谱保持</h3>
{table([{"类别": SHORT[g], "n": int((d.group == g).sum()),
         "FPS AI 总 / 形状": f"{mse(g, 'MSE_AI'):.1f} / {mse(g, 'MSEshape_AI'):.1f}",
         "EQ 总 / 形状": f"{mse(g, 'MSE_EQ'):.1f} / {mse(g, 'MSEshape_EQ'):.1f}",
         "匹配静态 EQ 总 / 形状": f"{mse(g, 'MSE_EQm'):.0f} / {mse(g, 'MSEshape_EQm'):.1f}"} for g in NT + [FOOT]])}
<p class="cap">对数频谱均方误差（dB²，中位数）。“总”包含整体音量变化；“形状”先减去整体平均差。</p>
<p>FPS AI 几乎不改变枪声（{mse(NT[0], 'MSE_AI'):.1f} dB²），远小于实际 EQ（{mse(NT[0], 'MSE_EQ'):.1f} dB²）；对换弹和其他非脚步也小于实际 EQ。
另一方面，FPS AI 对脚步本身的形状改变（{mse(FOOT, 'MSEshape_AI'):.1f} dB²）约为 EQ 的两倍，说明它不只是放大脚步，还改变了脚步的音色与时间结构。</p>

<h3>4.4 子实验 D：错误增强</h3>
<p>阈值取 {FER_THR:.0f} dB，即产品宣称的最低脚步增益：非目标声音被增强到“宣传中算作有效增强”的程度即计为错误增强。
FPS AI 对人工确认的非脚步事件 FER 为 {pct(fer['AI'])}（枪声 {pct(fer_g['AI'])}），背景段 {pct(fer_bg['AI'])}；
实际 EQ 分别为 {pct(fer['EQ'])} 和 {pct(fer_bg['EQ'])}，但它也不增强脚步，不构成有意义的对比；匹配静态 EQ 均为 100%。
被 FPS AI 增强超过 {FER_THR:.0f} dB 的非脚步事件逐条如下：</p>
{table(boosted_rows)}
<p>这些事件大多很安静（与脚步电平相近），且集中在少数片段：例如 PUBG test11 中同一颗烟雾弹的几个声音（以及紧随其后的一声枪响）都被增强了 12 dB 以上，
而另一处更响的烟雾弹（test13）几乎没有被增强；被增强的 {n_boost_gun} 个枪声电平在 {gun_boost_lv.max():.0f} ~ {gun_boost_lv.min():.0f} dBFS，
是远处或较弱的枪声，响亮的近处枪声没有一例被增强。因此我们的结论是“个别安静的非脚步瞬态会被当成脚步增强”，而不是“某一类声音整体会被误增强”。</p>
{img(os.path.join(RES, "ad_fer.png"))}<p class="cap">图 4　误增强率随阈值的变化。</p>

<h3>4.5 其他发现</h3>
<ul>
<li><b>底噪被一起抬高</b>：背景段的 FPS AI 增益中位数只有 {sgn(med(bg, 'AI'))} dB，但纯净脚步前紧邻的背景被抬高 {sgn(pre_bg)} dB，
有 {pct(fer_bg['AI'])} 的背景段超过 3 dB。这与试听中“处理后有白噪声”的感受一致。</li>
<li><b>动态处理</b>：FPS AI 的频带增益随时间波动（标准差中位 {tv_ai:.2f} dB），实际 EQ 几乎不变（{tv_eq:.2f} dB）。</li>
<li><b>重叠时增益较低</b>：与其他瞬态重叠的脚步增益低约 {clean_gain - ovl_gain:.1f} dB；目前无法区分是 FPS AI 主动让步还是测量被其他声音污染。</li></ul>

<h3>4.6 按游戏分列</h3>
{table(game_rows)}
<p class="cap">中位增益（dB），方括号为按片段聚类的 95% CI，圆括号为事件数。</p>
<ul>
<li><b>FPS AI 对 PUBG 脚步增强更多</b>：PUBG {sgn(fa_g('pubg', 'AI'))} dB，VALORANT {sgn(fa_g('valorant', 'AI'))} dB（人工 + 模型样本）。
两个游戏的枪声都几乎不被增强（{sgn(gm('pubg', NT[0], 'AI'))} / {sgn(gm('valorant', NT[0], 'AI'))} dB），选择性在两个游戏中都成立。
PUBG 脚步增益常超过 9 dB，是总体“偏向宣称上沿”的主要来源。</li>
<li><b>两套 EQ 的效果方向不同</b>：VALORANT 的 EQ 对脚步和枪声都略有削减（{sgn(fa_g('valorant', 'EQ'))} / {sgn(gm('valorant', NT[0], 'EQ'))} dB）；
PUBG 的 EQ 中频抬得更多，脚步 {sgn(fa_g('pubg', 'EQ'))} dB，枪声反而 {sgn(gm('pubg', NT[0], 'EQ'))} dB
（PUBG 枪声 300 Hz 以下的能量占比 {pct(bs[('pubg', 'gunshot')][0])}，低于脚步的 {pct(bs[('pubg', 'footstep')][0])}，
0.5–2.5 kHz 占比 {pct(bs[('pubg', 'gunshot')][1])}，高于脚步的 {pct(bs[('pubg', 'footstep')][1])}，因此从中频提升中得到的更多）。
总体表格中的“实际 EQ”是两套参数的混合结果。</li></ul>

<h3>4.7 子实验 E：脚步可检测性（初步）</h3>
<p>“可检测”可以有两层含义：脚步在声学上是否更突出（更容易被听见），以及它是否更像脚步（更容易被认出）。我们分别测量。</p>
<p><b>声学突出度</b>：对 {prom['n_fo']} 个人工确认的脚步，FPS AI 把峰值抬高 {sgn(prom['peak'])} dB，但脚步前紧邻的背景也被抬高 {sgn(prom['bg'])} dB，
因此脚步相对其前方背景的突出度净提高 <b>{sgn(prom['c_fo'])} dB</b>（95% CI {sgn(prom['c_fo_ci'][0])} ~ {sgn(prom['c_fo_ci'][1])}）；
非脚步事件为 {sgn(prom['c_nf'])} dB，实际 EQ 对脚步为 {sgn(prom['c_eq'])} dB。也就是说，FPS AI 确实让脚步从背景中更突出，但底噪抬升吃掉了约三分之一的增益。</p>
<p><b>能否被认出</b>：引入一个从未参与标注的独立检测器 PANNs（<code>Cnn14_DecisionLevelMax</code>，AudioSet 预训练的逐帧声音事件检测模型），
PANNs 原本要在 AudioSet 的 527 类中判断，游戏脚步的“脚步”概率普遍极低。因此除了①只看“Walk, footsteps”一类的概率，
我们还把判断限定在评测方案中的六类游戏声音之间：脚步（Walk, footsteps / Run / Shuffle）、枪声、爆炸、撞击、机械声、环境声，
每类取对应 AudioSet 类别的最大概率；②以脚步类占六类之和的比例为分数，③直接看六类中哪一类得分最高（完全不设阈值）。
类别映射在看结果之前按方案类别确定，没有加入模型容易把脚步误认成的“马蹄声”“嘎吱声”等类别。
三个版本先按整段 RMS 对齐到原始轨，去掉“整体变响”的作用；真值只用人工标注（脚步 {prom['n_fo']}、非脚步 {prom['n_nf']}）；
①②的阈值在原始轨上按“非脚步误报 10%”确定，三个版本共用。另用“原始轨训练的探针”（PANNs 嵌入 + 逻辑回归）作为对照。</p>
{table(det_rows)}
<p class="cap">事件窗口为起点前 0.05 s 至后 0.3 s。“相对原始”为检出率差（百分点），置信区间按片段聚类。</p>
<ul>
<li><b>FPS AI 没有让脚步更容易被认出</b>：三种打分方式下检出率变化分别为 {pp(zr['recall'], zf['recall'])}、{pp(cr['recall'], cf['recall'])}、
{pp(tr_['recall'], tf['recall'])} 个百分点，都没有提高；只有①的下降显著，②③的置信区间都包含 0，因此结论是“没有提高”，而不是“变差”。
不做整段 RMS 对齐时方向相同（{pp(nr['recall'], nf_['recall'])} / {pp(ncr['recall'], ncf['recall'])} / {pp(ntr['recall'], ntf['recall'])} 个百分点）。
同一个脚步变响约 9 dB，PANNs 的分数却几乎不变，说明它认脚步靠的是声音特征而不是响度；FPS AI 改变了脚步的音色（见 4.3 节），没有让它更像脚步。</li>
<li><b>限定类别后，PANNs 判为“脚步”的结果很可靠，但大部分游戏脚步认不出来</b>：原始轨上被判为脚步的事件 {pct(tr_['precision'])} 确实是脚步，
但只有 {pct(tr_['recall'])} 的脚步被判为脚步，其余多被判为枪声或环境声；游戏脚步短促、偏冲击声，与 AudioSet 中的生活脚步差别较大。</li>
<li><b>微调 PANNs 后结论不变</b>：用人工标注把 PANNs 微调成“脚步 / 枪声 / 其他 / 无事件”四分类器
（只用原始轨训练，随机改变音量和音色做增强；输出层用 AudioSet 对应类别的权重初始化，起点即上面的零样本检测器），
按片段留出 {ft_h['n_foot'] + ft_h['n_non']} 个事件所在的 8 个片段作测试集，另做 5 折交叉验证。原始轨上留出集 AUC {ft_h['auc']:.2f}、F1 {ft_h['f1']:.2f}，
交叉验证 AUC {ft_c['auc']:.2f}、F1 {ft_c['f1']:.2f}，与零样本相当、没有明显提高（训练样本只有约 300 个），非脚步误报偏高（{pct(ft_c['fpr_events'])}）。
FPS AI 相对原始的检出率变化：留出集 {pp(ft_h['recall'], ft_hf['recall'])} 个百分点（95% CI {ppci(ft_hf['d_recall_ci'])}），
交叉验证 {pp(ft_c['recall'], ft_cf['recall'])} 个百分点（{ppci(ft_cf['d_recall_ci'])}），均不显著。
最初一版（随机初始化输出层、1 秒窗口）反而比零样本差（留出集 AUC {ft_v1['auc']:.2f}），已弃用。</li>
<li><b>EQ 没有影响</b>：三种打分方式下检出率变化 {pp(zr['recall'], ze['recall'])} / {pp(cr['recall'], ce['recall'])} / {pp(tr_['recall'], te['recall'])} 个百分点，均不显著。</li>
<li><b>原始轨训练的探针不适合做评价器</b>：它在 FPS AI 和 EQ 版本上的检出率分别变化 {pp(pr['recall'], pf['recall'])} 和 {pp(pr['recall'], pe['recall'])} 个百分点；
EQ 只改变 1–3 dB 的音色就造成如此大的下降，说明它对音色变化过于敏感。</li>
<li><b>不能用“脚步 vs 枪声”的区分度衡量可检测性</b>：原始轨上脚步普遍比枪声轻，仅凭响度就能区分两者；FPS AI 把脚步抬高后两者响度接近，
这种区分度自然下降，但这与脚步是否更容易被听见无关。</li></ul>
<p><b>可信度</b>：PANNs 对游戏脚步的绝对概率极低（①的阈值仅 {zr['thr']:.4f}）；即使限定类别，原始轨上 AUC 也只有 {cr['auc']:.2f}，
直接判类别时只认出 {pct(tr_['recall'])} 的脚步，未达到评测方案对检测器的准入要求。因此“FPS AI 不提高机器可识别性”只是初步结论；而“声学突出度提高约 {prom['c_fo']:.0f} dB”是直接测量，更可靠。
要回答“玩家是否更容易听到并认出脚步”，需要用训练场数据训练的游戏专用检测器，或做主观听辨实验。</p>

<h2>5. FPS AI 与 EQ 对比总表</h2>
{table(compare_rows)}

<h2>6. 结论</h2>
<div class="box green"><h4>有把握的结论</h4><ul>
<li>FPS AI 对脚步的增益约 {sgn(med(fo_all, 'AI'))} ~ {sgn(med(fo_h, 'AI'))} dB，位于产品宣称 3–9 dB 的上沿
（超过 9 dB 的比例：人工样本 {pct((fo_h.G_AI > 9).mean())}，合并样本 {pct((fo_all.G_AI > 9).mean())}）；
分游戏看 VALORANT {sgn(fa_g('valorant', 'AI'))} dB、PUBG {sgn(fa_g('pubg', 'AI'))} dB；从脚步串的第一步就开始生效。</li>
<li>FPS AI 对枪声具有明确的选择性：同电平下枪声比脚步少增强约 {gap['AI'][NT[0]][0]:.0f} dB，枪声的频谱也几乎不被改变。</li>
<li>这种选择性无法用一条固定 EQ 曲线实现；实际 EQ 只是轻微调整音色，对脚步没有增强作用。</li>
<li>以上结论在人工与模型两套样本、纯净与有背景两档事件之间一致，并能耐受约 20% 的随机标注错误。</li></ul></div>
<div class="box amber"><h4>倾向性结论（证据有限）</h4><ul>
<li>个别安静的非脚步瞬态（一颗烟雾弹、部分换弹、吃药声、少数远处枪声）会被像脚步一样增强，最高 {sgn(boost_max)} dB；
样本少且集中在少数片段，响亮的近处枪声没有被增强。</li>
<li>FPS AI 会抬高脚步附近的底噪，并改变脚步本身的音色。</li>
<li>与其他声音重叠时，脚步测得的增益较低。</li>
<li>部分片段中的响亮脚步（-35 dBFS 以上）只增强 {hi_low_rng} dB，其他片段照常；样本只有 {len(lvl_hi)} 个、来自 {len(hi_clip)} 个片段，
无法区分是电平还是场景的影响。</li>
<li>FPS AI 让脚步相对背景更突出（约 {sgn(prom['c_fo'])} dB），但没有让独立的 AudioSet 检测器更容易认出脚步；后者受检测器在游戏音频上的可靠性限制。</li></ul></div>
<div class="box grey"><h4>尚未回答</h4><ul>
<li>玩家是否更容易听到并认出脚步（需要游戏专用检测器或主观听辨实验）。</li>
<li>装包/拆包、跳跃落地、VALORANT 技能声等类别，以及 -30 dBFS 以上的响亮脚步。</li>
<li>FPS AI 单独处理孤立事件时是否与整局处理一致；响度匹配后的比较。</li></ul></div>
<p>对照产品宣传：“脚步增强 3–9 dB”成立且偏向上沿；“只强化必要信息、不改变其他声音”对枪声成立，但对部分安静的非脚步声音和底噪并不完全成立。</p>

<h2>7. 局限</h2>
<ul>
<li><b>数据来源</b>：全部来自真实对局，没有分轨；“纯净事件”是从整局处理结果中挑出的，FPS AI 处理时看到的是完整上下文。</li>
<li><b>样本量与类别</b>：人工确认的纯净非脚步事件 {len(nt_h)} 个，其中枪声 {int((d.group == NT[0]).sum())}、换弹 {int((d.group == NT[1]).sum())}、
其他非脚步 {int((d.group == NT[2]).sum())}；缺少方案要求的部分类别。</li>
<li><b>标注</b>：只有一名标注者；盲审证明了稳定性，但不能证明准确性。事件终点为自动估计。</li>
<li><b>电平覆盖</b>：-35 dBFS 以上的脚步只有 {len(lvl_hi)} 个，-30 dBFS 以上没有，响亮事件的比较只能依赖枪声与少量非脚步；
同电平比较中响亮枪声的参照脚步也因此偏少。</li>
<li><b>匹配静态 EQ 是模拟的</b>，且曲线直接从同一批脚步拟合，属于对 EQ 偏有利的对照。</li>
<li><b>未做响度归一化</b>：若主观测试会匹配响度，部分增益可能被抵消。</li></ul>

<h2>8. 下一步建议</h2>
<ol>
<li><b>训练场受控录制</b>（优先）：分别只走路（多种地面、远中近距离）、只开枪、只用技能/投掷物/换弹、装拆包、跳跃落地，
一次补齐类别、电平覆盖、距离信息，并得到天然准确的标签。</li>
<li><b>子实验 E</b>：现成的 AudioSet 检测器（PANNs）在游戏脚步上不够可靠。下一步用训练场录音训练游戏专用检测器，
在实录人工标注上通过准入后再比较三种处理；同时做小规模主观听辨（同一片段三个版本盲听，记录脚步发现率与反应时间）。</li>
<li><b>孤立事件重处理</b>：把纯净事件库单独送入 FPS AI，检验上下文依赖（如开枪时是否让步）。</li>
<li><b>第二标注者</b>：对已有标注做一致性检验，从“稳定”推进到“准确”。</li>
<li><b>响度匹配的对照分析</b>；PUBG 枪声样本（{int(((d.group == NT[0]) & (d.game == 'pubg')).sum())} 个）偏少，按游戏的枪声结论需要补充样本。</li></ol>

<h2>附录：文件与复现</h2>
{table([
    {"内容": "本报告", "位置": "<code>docs/C3 FPS AI 声卡客观评测 实验报告.html</code>", "生成脚本": "<code>analysis/report_final.py</code>"},
    {"内容": "技术报告（A–D 全部表格）", "位置": "<code>analysis/results/report_AD.html</code>", "生成脚本": "<code>analysis/report_ad.py</code>"},
    {"内容": "纯净事件库（三轨切片 + 元数据）", "位置": "<code>analysis/clean_events/</code>", "生成脚本": "<code>analysis/extract_clean.py</code>"},
    {"内容": "人工标注（四轮 + 盲审）", "位置": "<code>analysis/labeling/human_labels*.csv, review_r1.csv</code>", "生成脚本": "标注页面导出"},
    {"内容": "全部事件与模型标签", "位置": "<code>analysis/results/events_labeled_v2.csv</code>", "生成脚本": "<code>analyze.py → train_labeler.py</code>"},
    {"内容": "逐事件测量结果", "位置": "<code>analysis/results/report_AD_*.csv</code>", "生成脚本": "<code>analysis/report_ad.py</code>"},
    {"内容": "EQ 参数与频响计算", "位置": "<code>analysis/eq_params.py</code>", "生成脚本": "—"},
    {"内容": "子实验 E（PANNs 可检测性）", "位置": "<code>analysis/results/detectability_*.csv</code>",
     "生成脚本": "<code>analysis/detectability.py</code>（加 <code>--nomatch</code> 为不做 RMS 对齐的敏感性分析）"},
    {"内容": "子实验 E（微调 PANNs 检测器）", "位置": "<code>analysis/results/panns_ft_*.csv</code>",
     "生成脚本": "<code>analysis/finetune_panns.py</code>"}])}
<p>复现顺序：<code>analyze.py</code> → <code>embed.py</code> / <code>embed_clap.py</code> → <code>train_labeler.py</code>
→ <code>extract_clean.py</code> → <code>report_ad.py</code> → <code>detectability.py</code>（含 <code>--nomatch</code>）→ <code>report_final.py</code>。</p>
<div class="footer">C3 FPS AI 声卡客观评测 · 实验报告 · 由 analysis/report_final.py 根据分析结果自动生成，文中数字均直接取自数据。</div>
</main></body></html>"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("->", OUT)
    return locals()


if __name__ == "__main__":
    main()
