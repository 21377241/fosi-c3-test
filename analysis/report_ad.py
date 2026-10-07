"""Preliminary report for sub-experiments A-D of the C3 FPS AI evaluation plan, on the reviewed
clean-event library (clean_events/clean_events.csv) plus guarded background stretches.

Gain   G = 20 log10(RMS_processed / RMS_baseline) over the event segment (plan section 4A).
Tracks AI = FPS StepSense, EQ = the supplied EQ track, EQm = simulated static EQ whose response is the
       average FPS response on footsteps, offset so its median footstep gain equals the AI's (plan 4B).
Output results/report_AD.html + figures results/ad_*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft, firwin2, fftconvolve
from common import load_triplet, SR
from analyze import rms_db
from window_sensitivity import auto_end, F2S
from coverage import union

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PROCS = ["AI", "EQ", "EQm"]
CLAIM = (3.0, 9.0)
FER_THR = 3.0
BG_GUARD, BG_MIN = 0.3, 1.0
P_HI = 0.8
NT_MODEL = "非脚步（模型）"
NFFT = 16384
rng = np.random.default_rng(0)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GROUP = {"换弹": "换弹 Reload", "烟雾弹": "技能/道具 Utility", "爆能器": "技能/道具 Utility", "吃药声": "技能/道具 Utility"}


def group_of(r):
    if r.cls == "unsure":
        return None
    if r.cls == "footstep":
        return "脚步 Footstep"
    if r.cls == "gunshot":
        return "枪声 Gunshot"
    return GROUP.get(r.note, "其他 Other")


def gain_db(proc, raw):
    return 10 * np.log10((proc ** 2).mean() / (raw ** 2).mean())


def logspec(x, eps):
    _, _, Z = stft(x.mean(1), SR, nperseg=1024, noverlap=768)
    return 20 * np.log10(np.abs(Z) + eps)


def spec_mse(proc, raw, f_keep):
    _, _, Zr = stft(raw.mean(1), SR, nperseg=1024, noverlap=768)
    eps = np.abs(Zr).max() * 10 ** (-80 / 20)
    d = (logspec(proc, eps) - logspec(raw, eps))[f_keep]
    return float((d ** 2).mean()), float(d.var())


def smooth_third_octave(f, r):
    out = np.empty_like(r)
    for i, fi in enumerate(f):
        lo, hi = fi / 2 ** (1 / 6), fi * 2 ** (1 / 6)
        out[i] = r[(f >= lo) & (f <= hi)].mean() if fi > 0 else r[1]
    return out


def cluster_ci(df, col, stat=np.median, n=2000):
    clips = df.clip_key.unique()
    if len(clips) < 3:
        return np.nan, np.nan
    groups = {c: df.loc[df.clip_key == c, col].values for c in clips}
    vals = []
    for _ in range(n):
        pick = rng.choice(clips, len(clips))
        vals.append(stat(np.concatenate([groups[c] for c in pick])))
    return np.percentile(vals, [2.5, 97.5])


def load_segments():
    meta = pd.read_csv(os.path.join(HERE, "clean_events", "clean_events.csv"), encoding="utf-8-sig", keep_default_na=False)
    meta["group"] = [group_of(r) for r in meta.itertuples()]
    meta = meta[meta.group.notna()]
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    clips = {(g, c) for g, c in zip(meta.game, meta.clip_id)} | {(g, c) for g, c in zip(ev.game, ev.clip_id)}
    return meta, ev, sorted(clips)


def model_pool(ev):
    """Events never labelled by a human whose ensemble probability is confident, with a predicted
    overlap probability from an AST-embedding classifier trained on the human purity grades."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from extract_clean import read_purity
    lab = pd.read_csv(os.path.join(RES, "events_labeled_v2.csv"), keep_default_na=False)
    pu = read_purity()
    X = np.load(os.path.join(HERE, "cache", "embeddings.npz"))["X"]
    clf = make_pipeline(StandardScaler(), PCA(0.95, svd_solver="full"),
                        LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))
    clf.fit(X[pu.event_id.values], (pu.purity == "l").values)
    m = lab[~lab.event_id.isin(pu.event_id) & ((lab.p_ens >= P_HI) | (lab.p_ens <= 1 - P_HI))].copy()
    m["group"] = np.where(m.p_ens >= P_HI, FOOT, NT_MODEL)
    m["p_overlap"] = clf.predict_proba(X[m.event_id.values])[:, 1]
    return m[["event_id", "game", "clip_id", "t", "group", "p_ens", "p_overlap"]]


def main():
    meta, ev, clips = load_segments()
    pool = model_pool(ev)
    segs, mrows = {}, []
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    P = {"raw": np.zeros(len(f)), "fps": np.zeros(len(f))}
    rows, bg_rows, cache = [], [], {}

    # pass 1: AI / EQ metrics, footstep spectra for the matched EQ, background stretches
    for game, clip in clips:
        a, _ = load_triplet(game, clip)
        cache[(game, clip)] = a
        g = meta[(meta.game == game) & (meta.clip_id == clip)]
        for r in g.itertuples():
            s = int(round(r.seg_start * SR)); e = s + int(r.seg_dur_ms / 1000 * SR)
            raw, fps, eq = a["raw"][s:e], a["fps"][s:e], a["eq"][s:e]
            segs[r.event_id] = (game, clip, s, e)
            if r.cls == "footstep":
                w = np.hanning(len(raw))[:, None]
                for k, x in [("raw", raw), ("fps", fps)]:
                    P[k] += (np.abs(np.fft.rfft(x * w, NFFT, axis=0)) ** 2).sum(1)
            rows.append(dict(event_id=r.event_id, group=r.group, purity=r.purity, game=game, clip_key=f"{game}{clip}",
                             raw_level=20 * np.log10(np.sqrt((raw ** 2).mean()) + 1e-12),
                             G_AI=gain_db(fps, raw), G_EQ=gain_db(eq, raw)))
        e_t = ev[(ev.game == game) & (ev.clip_id == clip)].t.sort_values().values
        lv = rms_db(a["raw"])
        for r in pool[(pool.game == game) & (pool.clip_id == clip)].itertuples():
            nxt = e_t[e_t > r.t + 0.03]
            end = max(auto_end(lv, int(r.t * F2S)), 0.03)
            if len(nxt):
                end = min(end, nxt[0] - r.t)
            s, e = int((r.t - 0.02) * SR), min(len(a["raw"]), int((r.t + end + 0.05) * SR))
            raw = a["raw"][s:e]
            segs[r.event_id] = (game, clip, s, e)
            mrows.append(dict(event_id=r.event_id, group=r.group, p_ens=r.p_ens, p_overlap=r.p_overlap, game=game,
                              clip_key=f"{game}{clip}", raw_level=20 * np.log10(np.sqrt((raw ** 2).mean()) + 1e-12),
                              G_AI=gain_db(a["fps"][s:e], raw), G_EQ=gain_db(a["eq"][s:e], raw)))
        T = len(lv) / F2S
        occ = union([(t - BG_GUARD, t + max(auto_end(lv, int(t * F2S)), 0.03) + BG_GUARD) for t in e_t])
        edges = [0.0] + [x for iv in occ for x in iv] + [T]
        for lo, hi in zip(edges[::2], edges[1::2]):
            lo, hi = max(lo, 0.3), min(hi, T - 0.1)
            if hi - lo >= BG_MIN:
                s, e = int(lo * SR), int(hi * SR)
                raw = a["raw"][s:e]
                bg_rows.append(dict(game=game, clip=clip, clip_key=f"{game}{clip}", start=lo, dur=hi - lo, s=s, e=e,
                                    raw_level=20 * np.log10(np.sqrt((raw ** 2).mean()) + 1e-12),
                                    G_AI=gain_db(a["fps"][s:e], raw), G_EQ=gain_db(a["eq"][s:e], raw)))
    d = pd.DataFrame(rows)
    bg = pd.DataFrame(bg_rows)
    dm = pd.DataFrame(mrows)

    # matched static EQ: average FPS response on footsteps, 1/3-octave smoothed, as linear-phase FIR
    resp = 10 * np.log10(smooth_third_octave(f, P["fps"]) / smooth_third_octave(f, P["raw"]))
    band = (f >= 60) & (f <= 16000)
    resp = np.interp(f, f[band], resp[band])
    h = firwin2(2049, f / (SR / 2), 10 ** (resp / 20))
    filt = {}
    for (game, clip), a in cache.items():
        filt[(game, clip)] = np.stack([fftconvolve(a["raw"][:, c], h, mode="same")
                                       for c in range(a["raw"].shape[1])], 1).astype(np.float32)

    def eqm_seg(game, clip, s, e):
        return filt[(game, clip)][s:e]

    def raw_seg(game, clip, s, e):
        return cache[(game, clip)]["raw"][s:e]
    d["G_EQm"] = [gain_db(eqm_seg(*segs[i]), raw_seg(*segs[i])) for i in d.event_id]
    off = d.loc[d.group == "脚步 Footstep", "G_AI"].median() - d.loc[d.group == "脚步 Footstep", "G_EQm"].median()
    d["G_EQm"] += off
    resp_eqm = resp + off
    bg["G_EQm"] = [gain_db(eqm_seg(r.game, r.clip, r.s, r.e), cache[(r.game, r.clip)]["raw"][r.s:r.e]) + off
                   for r in bg.itertuples()]
    dm["G_EQm"] = [gain_db(eqm_seg(*segs[i]), raw_seg(*segs[i])) + off for i in dm.event_id]

    # spectral MSE (plan 4C), on the segments
    fz = np.fft.rfftfreq(1024, 1 / SR)
    keep = (fz >= 50) & (fz <= 16000)
    gscale = 10 ** (off / 20)
    for k, i in enumerate(d.event_id):
        game, clip, s, e = segs[i]
        a = cache[(game, clip)]
        raw = a["raw"][s:e]
        for p, x in [("AI", a["fps"][s:e]), ("EQ", a["eq"][s:e]), ("EQm", eqm_seg(game, clip, s, e) * gscale)]:
            mse, shape = spec_mse(x, raw, keep)
            d.loc[d.index[k], f"MSE_{p}"] = mse
            d.loc[d.index[k], f"MSEshape_{p}"] = shape

    d.to_csv(os.path.join(RES, "report_AD_events.csv"), index=False, encoding="utf-8-sig")
    bg.drop(columns=["s", "e"]).to_csv(os.path.join(RES, "report_AD_background.csv"), index=False, encoding="utf-8-sig")
    dm.to_csv(os.path.join(RES, "report_AD_model_events.csv"), index=False, encoding="utf-8-sig")
    html = build_report(d, bg, f, resp_eqm, off, dm)
    with open(os.path.join(RES, "report_AD.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    print("-> results/report_AD.html")


# ---------------------------------------------------------------- report
FOOT = "脚步 Footstep"
NT = ["枪声 Gunshot", "换弹 Reload", "技能/道具 Utility"]


def fmt_ci(med, ci):
    return f"{med:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]"


def table(df):
    return df.to_html(index=False, border=0, escape=False, classes="t")


MATCH_DB, MATCH_MIN = 3.0, 3


def matched_gap(fo, nt, p):
    """For each non-target event: median gain of footsteps within ±MATCH_DB of its level, minus its own gain.
    Events with fewer than MATCH_MIN such footsteps are skipped. Returns (median difference, n used)."""
    diffs = []
    for r in nt.itertuples():
        near = fo.loc[(fo.raw_level - r.raw_level).abs() <= MATCH_DB, f"G_{p}"]
        if len(near) >= MATCH_MIN:
            diffs.append(near.median() - getattr(r, f"G_{p}"))
    return (float(np.median(diffs)) if diffs else np.nan, len(diffs))


def level_matched_sei(d, p):
    fo = d[d.group == FOOT]
    return {g: matched_gap(fo, d[d.group == g], p) for g in NT}, None


def model_label_accuracy():
    """Precision of confident ensemble labels on human events, from out-of-fold probabilities.
    Round 1 is a stratified random sample, so it is used as the representative estimate."""
    from extract_clean import read_purity
    cv = pd.read_csv(os.path.join(RES, "human_cv_probs.csv"), encoding="utf-8-sig")
    h = read_purity().merge(cv, on="event_id")
    r1 = set(pd.read_csv(os.path.join(HERE, "labeling", "review_r1.csv"), encoding="utf-8-sig").event_id)
    h = h[h.event_id.isin(r1) & h.final_label.isin(["f", "g", "o"])]
    hf, hn = h[h.cv_ens >= P_HI], h[h.cv_ens <= 1 - P_HI]
    return dict(n_f=len(hf), prec_f=(hf.final_label == "f").mean(), n_n=len(hn), prec_n=(hn.final_label != "f").mean(),
                comp_n=hn.final_label.map({"f": "脚步", "g": "枪声", "o": "其他"}).value_counts().to_dict())


def model_section(d, dm):
    acc = model_label_accuracy()
    fo_h, fo_m = d[d.group == FOOT], dm[dm.group == FOOT]
    nt_h, nt_m = d[d.group.isin(NT)], dm[dm.group == NT_MODEL]
    fo_all = pd.concat([fo_h, fo_m])
    cut = dm.p_overlap.quantile(2 / 3)
    dm_lo = dm[dm.p_overlap <= cut]

    def a_row(name, x):
        lo, hi = cluster_ci(x, "G_AI")
        return {"脚步样本": name, "n": len(x), "FPS AI 中位 [95% CI]": f"{x.G_AI.median():+.1f} [{lo:+.1f}, {hi:+.1f}]",
                "EQ 中位": f"{x.G_EQ.median():+.1f}", "落在 3–9 dB": f"{x.G_AI.between(*CLAIM).mean():.0%}",
                "高于 9 dB": f"{(x.G_AI > CLAIM[1]).mean():.0%}", "原始电平中位": f"{x.raw_level.median():.0f}"}
    rows_a = [a_row("人工（纯净/有背景）", fo_h), a_row("模型高置信度", fo_m),
              a_row("模型高置信度，去掉最可能重叠的 1/3", dm_lo[dm_lo.group == FOOT]), a_row("人工 + 模型", fo_all)]
    bins = pd.cut(fo_all.raw_level, [-80, -50, -40, -30, 0], labels=["低于 -50", "-50 ~ -40", "-40 ~ -30", "高于 -30"])
    lvl = fo_all.groupby(bins, observed=True).agg(n=("G_AI", "size"), AI=("G_AI", "median"), EQ=("G_EQ", "median")).round(1)
    lvl = lvl.reset_index().rename(columns={"raw_level": "原始电平 (dBFS RMS)", "AI": "FPS AI 中位增益", "EQ": "EQ 中位增益"})

    def b_row(name, nt, fo_ref):
        g, n = matched_gap(fo_ref, nt, "AI")
        ge, _ = matched_gap(fo_ref, nt, "EQ")
        return {"非脚步样本": name, "n": len(nt), "原始电平中位": f"{nt.raw_level.median():.0f}",
                "FPS AI 中位": f"{nt.G_AI.median():+.1f}", "EQ 中位": f"{nt.G_EQ.median():+.1f}",
                "同电平 SEI：AI": f"<b>{g:+.1f}</b> (n={n})", "同电平 SEI：EQ": f"{ge:+.1f}",
                f"AI FER（&gt;{FER_THR:.0f} dB）": f"{(nt.G_AI > FER_THR).mean():.0%}"}
    rows_b = [b_row("人工：枪声+换弹+道具", nt_h, fo_all), b_row("人工：仅枪声", d[d.group == NT[0]], fo_all),
              b_row("模型高置信度非脚步", nt_m, fo_all),
              b_row("模型高置信度非脚步，去掉最可能重叠的 1/3", dm_lo[dm_lo.group == NT_MODEL], fo_all)]
    gm, nm = matched_gap(fo_all, nt_m, "AI")
    gh, nh = matched_gap(fo_all, d[d.group == NT[0]], "AI")
    g_lo, _ = matched_gap(fo_all, dm_lo[dm_lo.group == NT_MODEL], "AI")
    f_lo = dm_lo.loc[dm_lo.group == FOOT, "G_AI"].median()
    stable = abs(g_lo - gm) < 1.5 and abs(f_lo - fo_m.G_AI.median()) < 1.5
    sens = (f"去掉最可能重叠的三分之一后，模型脚步增益为 {f_lo:+.1f} dB、非脚步同电平差距为 {g_lo:.1f} dB，"
            + ("与未筛选时相差不到 1.5 dB，结论不依赖于是否含重叠事件。" if stable else "与未筛选时差异较大，重叠事件对结果有明显影响，需谨慎解读。"))
    comp = "、".join(f"{k} {v}" for k, v in acc["comp_n"].items())
    html = f"""
<p>模型用于扩大标注覆盖面：AST 与 CLAP 两模型集成给出“是脚步”的概率，≥ {P_HI} 记为脚步、≤ {1 - P_HI:.1f} 记为非脚步，
只取从未人工标注过的事件。用第 1 轮人工标注（分层随机抽样，代表整体）核对模型在“训练时没见过该事件”时的判断：</p>
{table(pd.DataFrame([
        {"模型判为": f"脚步（p ≥ {P_HI}）", "核对事件数": acc["n_f"], "人工确认比例": f"{acc['prec_f']:.0%}"},
        {"模型判为": f"非脚步（p ≤ {1 - P_HI:.1f}）", "核对事件数": acc["n_n"], "人工确认比例": f"{acc['prec_n']:.0%}（人工结果：{comp}）"}]))}
<p>由此纳入 {len(fo_m)} 个模型脚步和 {len(nt_m)} 个模型非脚步。模型不判断纯净度：按人工样本的比例，约两成事件含重叠。
作为敏感性检验，另用 AST 特征训练的“是否重叠”分类器（交叉验证 AUC 0.75）去掉最可能重叠的三分之一后重算。
模型非脚步不细分类别，从核对结果看主要是枪声。</p>
<h3>脚步增益（对应子实验 A）</h3>{table(pd.DataFrame(rows_a))}
<p>人工 + 模型合并后按电平分层（-40 ~ -30 dBFS 及以上的脚步样本因此增加，同电平比较更稳）：</p>{table(lvl)}
<h3>非脚步增益、同电平选择性与误增强（对应子实验 B、D）</h3>
<p>同电平 SEI 的参照组统一用“人工 + 模型”的全部脚步。参照组扩大后，人工枪声的同电平差距与第 2 节（只用人工脚步作参照）不同：
第 2 节在枪声所在的 -40 ~ -30 dBFS 只有十来个参照脚步，估计不稳；这里该电平段有 {int(fo_all.raw_level.between(-40, -30).sum())} 个，应以这里为准。</p>
{table(pd.DataFrame(rows_b))}
<p>注意：模型高置信度非脚步的 FER 明显低于人工样本，这是选择偏差——模型最有把握判为非脚步的，正是最不像脚步的声音，
FPS AI 也最不会增强它们。误增强率应以人工样本为准，模型样本只用于验证同电平选择性。</p>
<p><b>结论：</b>模型标注的 {len(fo_m)} 个脚步得到的 FPS AI 增益（中位 {fo_m.G_AI.median():+.1f} dB）与人工样本（{fo_h.G_AI.median():+.1f} dB）
{"一致" if abs(fo_m.G_AI.median() - fo_h.G_AI.median()) < 1.5 else "存在差异"}；
模型非脚步与同电平脚步相比少增强 {gm:.1f} dB（n={nm}），人工枪声为 {gh:.1f} dB（n={nh}）。
{sens}
模型样本无法回答道具类的问题（它不区分非脚步的细类，且道具在非脚步中占比很小）。</p>"""
    per_class = {g.split()[0]: matched_gap(fo_all, d[d.group == g], "AI") for g in NT}
    return html, dict(n_f=len(fo_m), n_n=len(nt_m), g_f=fo_m.G_AI.median(), gap=gm, n_gap=nm, prec_f=acc["prec_f"],
                      n_ref=len(fo_all), per_class=per_class)


def build_report(d, bg, f, resp_eqm, off, dm):
    fo = d[d.group == FOOT]
    sec = []

    # ---------------- data
    cnt = d.groupby(["group", "purity"]).size().unstack(fill_value=0).rename(columns={"p": "纯净", "b": "有持续背景"})
    cnt["合计"] = cnt.sum(1)
    cnt = cnt.reset_index().rename(columns={"group": "类别"})
    sec.append(("数据", f"""
<p>来源：VALORANT 16 段 + PUBG 15 段真实对局录音，每段有原始（Baseline）、FPS AI、EQ 三个同源版本，已按样本对齐。
事件由人工标注（第 1 轮 153 条经盲审复核：前后一致率 94%，脚步/非脚步 κ = 0.96；第 2 轮 80 条为模型最拿不准的事件，
标注时同时判断纯净度），纯净度由人工分级，
只取“纯净”和“有持续背景”两档，“有重叠”的事件不进入本报告。事件区间 = 起点前 20 ms 至自动估计终点后 50 ms。</p>
{table(cnt)}
<p>另取 {len(bg)} 段“背景段”作为环境类负样本：距离任何检测到的事件都 ≥ {BG_GUARD} s、长度 ≥ {BG_MIN} s 的区间，
共 {bg.dur.sum():.0f} s。<b>背景段未经人工逐段核实</b>，可能含检测器漏掉的弱声音。</p>"""))

    # ---------------- A
    rowsA = []
    for p, name in [("AI", "FPS AI"), ("EQ", "EQ（实际）")]:
        x = fo[f"G_{p}"]
        rowsA.append({"处理": name, "中位数 [95% CI]": fmt_ci(x.median(), cluster_ci(fo, f"G_{p}")),
                      "均值": f"{x.mean():+.1f}", "四分位": f"{x.quantile(.25):+.1f} ~ {x.quantile(.75):+.1f}",
                      f"落在 {CLAIM[0]:.0f}–{CLAIM[1]:.0f} dB": f"{x.between(*CLAIM).mean():.0%}",
                      "高于 9 dB": f"{(x > CLAIM[1]).mean():.0%}", "低于 3 dB": f"{(x < CLAIM[0]).mean():.0%}"})
    bins = pd.cut(fo.raw_level, [-80, -50, -40, -30, 0], labels=["低于 -50", "-50 ~ -40", "-40 ~ -30", "高于 -30"])
    lvl = fo.groupby(bins, observed=True).agg(n=("G_AI", "size"), AI=("G_AI", "median"), EQ=("G_EQ", "median")).round(1)
    lvl = lvl.reset_index().rename(columns={"raw_level": "原始电平 (dBFS RMS)", "AI": "FPS AI 中位增益", "EQ": "EQ 中位增益"})
    pur = fo.groupby("purity").agg(n=("G_AI", "size"), AI=("G_AI", "median"), EQ=("G_EQ", "median")).round(1)
    pur.index = pur.index.map({"p": "纯净", "b": "有持续背景"})
    pur = pur.reset_index().rename(columns={"purity": "纯净度", "AI": "FPS AI 中位增益", "EQ": "EQ 中位增益"})
    fig_scatter(d)
    sec.append(("子实验 A：脚步增强能力", f"""
<p>G<sub>footstep</sub> = 20·log10(RMS<sub>处理后</sub> / RMS<sub>原始</sub>)，{len(fo)} 个脚步事件，单位 dB；
置信区间为按音频片段聚类的 bootstrap。</p>{table(pd.DataFrame(rowsA))}
<p><b>结论：</b>FPS AI 对脚步的中位增益为 {fo.G_AI.median():+.1f} dB，处在产品资料 3–9 dB 的上沿，
{(fo.G_AI > 9).mean():.0%} 的脚步超过 9 dB，只有 {(fo.G_AI < 3).mean():.0%} 低于 3 dB；EQ 对脚步几乎不增强甚至略有衰减。</p>
<p>按原始电平分层：在脚步实际出现的电平范围内，FPS AI 的增益只略随电平升高而下降；
更响的电平（-30 dBFS 以上）没有脚步样本，无法判断增益是否会进一步下降，所以跨类别比较时需要同电平对照（见子实验 B）。</p>
{table(lvl)}<p>纯净与有持续背景两档的脚步增益接近：</p>{table(pur)}
<img src="ad_gain_vs_level.png">"""))

    # ---------------- B
    rowsB = []
    for gname in [FOOT] + NT:
        x = d[d.group == gname]
        rowsB.append({"类别": gname, "n": len(x), "原始电平中位": f"{x.raw_level.median():.0f}",
                      **{lab: f"{x[f'G_{p}'].median():+.1f}" for p, lab in
                         [("AI", "FPS AI"), ("EQ", "EQ（实际）"), ("EQm", "匹配静态 EQ")]}})
    xb = bg
    rowsB.append({"类别": "背景段 Environment*", "n": len(xb), "原始电平中位": f"{xb.raw_level.median():.0f}",
                  "FPS AI": f"{xb.G_AI.median():+.1f}", "EQ（实际）": f"{xb.G_EQ.median():+.1f}", "匹配静态 EQ": f"{xb.G_EQm.median():+.1f}"})
    sei = []
    lm_ai, _ = level_matched_sei(d, "AI")
    gap_txt = "、".join(f"{g.split()[0]}少约 {v:.1f} dB（n={n}）" for g, (v, n) in list(lm_ai.items())[:2])
    for p, lab in [("AI", "FPS AI"), ("EQ", "EQ（实际）"), ("EQm", "匹配静态 EQ")]:
        gf = fo[f"G_{p}"].median()
        gnt = np.mean([d.loc[d.group == g, f"G_{p}"].median() for g in NT])
        lm, _ = level_matched_sei(d, p)
        lm_mean = np.nanmean([v for v, _ in lm.values()])
        sei.append({"处理": lab, "G_footstep": f"{gf:+.1f}", "G_non-target（3 类平均）": f"{gnt:+.1f}",
                    "SEI（方案定义）": f"<b>{gf - gnt:+.1f}</b>",
                    "SEI（同电平比较）": f"<b>{lm_mean:+.1f}</b>",
                    "其中 枪声 / 换弹 / 道具": " / ".join(f"{v:+.1f} (n={n})" for v, n in lm.values())})
    fig_bars(d, bg)
    fig_response(f, resp_eqm)
    sec.append(("子实验 B：选择性增强（SEI）", f"""
<p>各类事件的中位增益（dB）：</p>{table(pd.DataFrame(rowsB))}
<p>* 背景段未经人工核实。“匹配静态 EQ”是模拟的：以 FPS AI 在全部脚步上的平均频响（1/3 倍频程平滑）设计一个固定线性相位滤波器，
再整体平移 {off:+.1f} dB 使其脚步中位增益与 FPS AI 相同。它代表“用一条固定 EQ 曲线达到同样脚步增益”时的最好情况
（曲线直接从这批脚步拟合，属于偏乐观的对照）。实际 EQ 轨的脚步增益约为 0，无法直接满足方案中“先把 EQ 调到与 AI 相近”的要求，
而整体平移并不改变 SEI，所以 SEI 比较以匹配静态 EQ 为主。</p>
{table(pd.DataFrame(sei))}
<p>“SEI（方案定义）”= G<sub>footstep</sub> − 三类非目标增益的平均。枪声的原始电平中位数比脚步高约
{d.loc[d.group == NT[0], 'raw_level'].median() - fo.raw_level.median():.0f} dB，而 FPS AI 对响的声音增强较少，
所以这个数字可能混入“电平效应”（-30 dBFS 以上没有脚步样本，无法判断更响时 FPS AI 是否也会减弱对脚步的增强；
第 5 节合并模型样本后，-35 ~ -30 dBFS 的脚步仍有约 8 dB 增益）。“SEI（同电平比较）”对每个非目标事件，
取电平相差 {MATCH_DB:.0f} dB 以内的脚步（至少 {MATCH_MIN} 个）的中位增益，减去该事件自身的增益，再取中位数；
找不到足够同电平脚步的事件不计入（括号内为计入的事件数）。</p>
<p><b>结论：</b>FPS AI 的 SEI 为正，而达到同样脚步增益的固定 EQ 的 SEI 接近 0——FPS AI 的选择性不是一条频响曲线能做到的。
分类别看差异很大：与同电平的脚步相比，{gap_txt}，<b>技能/道具类（烟雾弹、爆能器、吃药）却和脚步一样被增强</b>
（同电平 SEI 为 {lm_ai[NT[2]][0]:+.1f} dB，样本只有 {lm_ai[NT[2]][1]} 个，需要更多数据确认）。所以 FPS AI 的选择性是“偏向脚步类声音”，
对与脚步声学特征相近的道具声并不能区分。</p>
<img src="ad_gain_bars.png">
<p>下图是 FPS AI 在脚步上的平均频响（也就是匹配静态 EQ 的曲线）：全频段都抬高了 7–11.5 dB，1–6 kHz 最多。
也就是说 FPS AI 对脚步做的主要是“整体抬升 + 中高频略多”，而它区分脚步与枪声的能力来自对声音的识别/动态判断，而非频响形状。</p>
<img src="ad_response.png">"""))

    # ---------------- C
    rowsC = []
    for gname in NT + [FOOT]:
        x = d[d.group == gname]
        rowsC.append({"类别": gname, "n": len(x),
                      **{f"{lab} 总 MSE": f"{x[f'MSE_{p}'].median():.1f}" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]},
                      **{f"{lab} 形状 MSE": f"{x[f'MSEshape_{p}'].median():.1f}" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]}})
    sec.append(("子实验 C：非目标声音的频谱保持", f"""
<p>MSE<sub>spec</sub> = 时频点上 [S<sub>处理后</sub> − S<sub>原始</sub>]² 的平均，S = 20·log10(|STFT| + ε)，
STFT 窗长 1024、跳步 256，频率 50 Hz–16 kHz，ε 取原始片段最大幅度以下 80 dB。单位 dB²，表中为各类中位数。
“总 MSE”包含整体音量变化；“形状 MSE”先减去整体平均差，只反映频谱形状和时间上的变化。</p>
{table(pd.DataFrame(rowsC))}
<p><b>解读：</b>总 MSE 受整体增益主导（整体增益 3 dB 本身就贡献约 9 dB²），因此要和形状 MSE 一起看。
FPS AI 对枪声几乎不做改变（总 MSE 中位 {d.loc[d.group == NT[0], 'MSE_AI'].median():.1f} dB²），而实际 EQ 作为一条固定曲线，对所有声音都改变约 8–9 dB²；
对换弹和道具，FPS AI 的总 MSE 与形状 MSE 也都低于实际 EQ。匹配静态 EQ 的形状改变最小，但总 MSE 约 {d[d.group.isin(NT)].MSE_EQm.median():.0f} dB²，
几乎全部来自它对所有声音一律施加的约 9 dB 增益。注意 FPS AI 对脚步本身的形状 MSE 明显更高，说明它不只是放大脚步，
还改变了脚步的频谱/时间结构。</p>"""))

    # ---------------- D
    neg = d[d.group.isin(NT)]
    thr = np.arange(0, 10.5, 0.5)
    fer = {p: [(neg[f"G_{p}"] > t).mean() for t in thr] for p in PROCS}
    fer_bg = {p: [(bg[f"G_{p}"] > t).mean() for t in thr] for p in PROCS}
    fig_fer(thr, fer, fer_bg)
    rowsD = []
    for gname in NT:
        x = d[d.group == gname]
        rowsD.append({"负样本类别": gname, "n": len(x),
                      **{f"{lab} G_negative": f"{x[f'G_{p}'].median():+.1f}" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]},
                      **{f"{lab} FER": f"{(x[f'G_{p}'] > FER_THR).mean():.0%}" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]}})
    rowsD.append({"负样本类别": "全部非脚步事件", "n": len(neg),
                  **{f"{lab} G_negative": f"{neg[f'G_{p}'].median():+.1f}" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]},
                  **{f"{lab} FER": f"<b>{(neg[f'G_{p}'] > FER_THR).mean():.0%}</b>" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]}})
    rowsD.append({"负样本类别": "背景段*", "n": len(bg),
                  **{f"{lab} G_negative": f"{bg[f'G_{p}'].median():+.1f}" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]},
                  **{f"{lab} FER": f"{(bg[f'G_{p}'] > FER_THR).mean():.0%}" for p, lab in [("AI", "AI"), ("EQ", "EQ"), ("EQm", "匹配EQ")]}})
    sec.append(("子实验 D：负样本 / 错误增强（FER）", f"""
<p>FER = 增益超过阈值的非脚步事件数 / 全部非脚步事件数。方案要求阈值由数据确定而非随意指定，这里取
<b>{FER_THR:.0f} dB = 产品资料宣称的最低脚步增益</b>：非目标声音被增强到“宣传中算作有效增强”的程度即计为错误增强。
同时给出阈值从 0 到 10 dB 的完整曲线，便于换用其他阈值。</p>
{table(pd.DataFrame(rowsD))}<img src="ad_fer.png">
<p><b>结论：</b>按 3 dB 阈值，FPS AI 的总体 FER 为 {(neg.G_AI > FER_THR).mean():.0%}：
枪声 {(d.loc[d.group == NT[0], 'G_AI'] > FER_THR).mean():.0%}、换弹 {(d.loc[d.group == NT[1], 'G_AI'] > FER_THR).mean():.0%}、
道具 {(d.loc[d.group == NT[2], 'G_AI'] > FER_THR).mean():.0%}，背景段 {(bg.G_AI > FER_THR).mean():.0%}。
匹配静态 EQ 对所有声音一律增强，FER 为 100%；实际 EQ 几乎不增强任何声音，FER 很低，但它也没有增强脚步，不构成有意义的对比。</p>"""))

    msec, ms = model_section(d, dm)
    sec.append(("扩大样本：纳入高置信度模型标注", msec))

    # ---------------- E + limits
    sec.append(("子实验 E：脚步可检测性（未做）", """
<p>需要一个对电平敏感、且在游戏数据上验证过的帧级脚步检测器，以及带起止时间的强标注作为 Ground Truth。
现有的少样本分类器在提取特征前做了音量归一化，对增益不敏感，不能直接用于此项。前置工作：区间（脚步串）标注 → 帧级检测器 → 在 Baseline 上验证 P/R/F1 → 再比较三种处理。</p>"""))
    sec.append(("局限与下一步", f"""<ul>
<li><b>样本量</b>：非脚步纯净事件只有 {len(neg)} 个（枪声 {int((d.group == NT[0]).sum())}、换弹 {int((d.group == NT[1]).sum())}、道具 {int((d.group == NT[2]).sum())}），
换弹和道具的结论只能算方向性；缺装包/拆包、跳跃落地等方案要求的类别。第 2 轮按“模型拿不准”挑选，补到的主要是枪声，
道具类没有增加；要补道具/换弹，需要按类别定向抽样或训练场受控录制。</li>
<li><b>标注效度</b>：盲审证明的是同一标注者的稳定性，不是准确性；仍需第二标注者或训练场受控录制。</li>
<li><b>“纯净”是从真实对局中挑出来的</b>：FPS AI 处理时看到的是整局上下文，而不是孤立事件；若要严格符合方案的“纯净单事件”，需要把片段单独送入 FPS AI 重新处理。</li>
<li><b>电平混杂</b>：枪声的原始电平中位数比脚步高约 {d.loc[d.group == NT[0], 'raw_level'].median() - fo.raw_level.median():.0f} dB，
能找到同电平脚步的枪声 {level_matched_sei(d, 'AI')[0][NT[0]][1]} 个、换弹 {level_matched_sei(d, 'AI')[0][NT[1]][1]} 个、道具 {level_matched_sei(d, 'AI')[0][NT[2]][1]} 个；-30 dBFS 以上没有脚步样本。理想数据应让各类事件覆盖相同的电平范围（可用受控录制或按电平分层抽样）。</li>
<li><b>输出响度</b>：本报告没有做响度归一化。背景段的 FPS AI 中位增益为 {bg.G_AI.median():+.1f} dB，
但有 {(bg.G_AI > FER_THR).mean():.0%} 的背景段超过 {FER_THR:.0f} dB；此前还测到脚步前紧邻的背景被抬高 2–4.6 dB，
说明增强在脚步附近会一并抬高底噪。</li>
<li><b>匹配静态 EQ 是模拟</b>，并且曲线直接从同一批脚步拟合，属于对 EQ 偏有利的对照。</li></ul>"""))

    body = "".join(f"<h2>{i}. {t}</h2>{c}" if i else f"<h2>{t}</h2>{c}" for i, (t, c) in enumerate(sec))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>C3 FPS AI 初步评测报告（A–D）</title>
<style>body{{margin:0;background:#f3f6fa;color:#1e2d3d;font:15px/1.8 "Microsoft YaHei",sans-serif}}
.page{{max-width:1040px;margin:30px auto;padding:50px 70px;background:#fff;box-shadow:0 8px 30px rgba(27,50,76,.12)}}
h1{{color:#17365d;text-align:center}}h2{{color:#17365d;margin-top:40px;border-bottom:2px solid #d7e0ea;padding-bottom:4px}}
table.t{{border-collapse:collapse;margin:12px 0 18px;font-size:13.5px;width:100%}}table.t th{{background:#17365d;color:#fff;padding:7px 9px;text-align:left}}
table.t td{{border:1px solid #d7e0ea;padding:6px 9px}}table.t tr:nth-child(even){{background:#eef4fa}}img{{max-width:100%;display:block;margin:10px auto}}
.sum{{background:#eef4fa;border-left:4px solid #2f6da5;padding:12px 18px;margin:18px 0}}</style></head>
<body><main class="page"><h1>C3 FPS AI 初步评测报告（子实验 A–D）</h1>
<div class="sum"><b>摘要</b><ul>
<li>A 脚步增强：FPS AI 中位 {fo.G_AI.median():+.1f} dB（实际 EQ {fo.G_EQ.median():+.1f} dB），处在产品宣称 3–9 dB 的上沿，一半脚步超过 9 dB。</li>
<li>B 选择性：FPS AI 的 SEI 为正，而达到同样脚步增益的固定 EQ 的 SEI 接近 0。以 {ms['n_ref']} 个脚步为同电平参照，
人工标注的枪声、换弹、道具分别比脚步少增强 {' / '.join(f"{v:.1f}" for v, _ in ms['per_class'].values())} dB
（n = {' / '.join(str(n) for _, n in ms['per_class'].values())}）——道具类被增强得和脚步一样多，但样本很少。</li>
<li>C 频谱保持：FPS AI 几乎不改变枪声（{d.loc[d.group == NT[0], 'MSE_AI'].median():.1f} dB²，实际 EQ {d.loc[d.group == NT[0], 'MSE_EQ'].median():.1f} dB²），对换弹/道具的改变也小于实际 EQ。</li>
<li>D 错误增强：3 dB 阈值下 FPS AI 总体 FER {(d[d.group.isin(NT)].G_AI > FER_THR).mean():.0%}，主要来自道具类（{(d.loc[d.group == NT[2], 'G_AI'] > FER_THR).mean():.0%}）和部分背景段。</li>
<li>扩大样本：纳入 {ms['n_f']} 个模型高置信度脚步（人工核对准确率 {ms['prec_f']:.0%}）和 {ms['n_n']} 个非脚步，
脚步增益中位 {ms['g_f']:+.1f} dB，非脚步比同电平脚步少增强 {ms['gap']:.1f} dB（n={ms['n_gap']}），与人工样本结论一致。</li>
<li>E 可检测性：尚未进行。样本量与标注效度见“局限”。</li></ul></div>
{body}<p style="color:#888;font-size:12px;margin-top:40px">由 analysis/report_ad.py 生成；逐事件数据见 report_AD_events.csv、report_AD_background.csv。</p>
</main></body></html>"""


def fig_scatter(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {FOOT: "#2f6da5", "枪声 Gunshot": "#d62728", "换弹 Reload": "#ff7f0e", "技能/道具 Utility": "#2ca02c"}
    for ax, p, title in [(axes[0], "AI", "FPS AI"), (axes[1], "EQ", "EQ（实际）")]:
        for gname, c in colors.items():
            x = d[d.group == gname]
            ax.scatter(x.raw_level, x[f"G_{p}"], s=18 if gname == FOOT else 34, c=c, alpha=.7, label=gname,
                       marker="o" if gname == FOOT else "^")
        ax.axhspan(*CLAIM, color="#999", alpha=.12, lw=0)
        ax.axhline(0, c="k", lw=.6)
        ax.set_title(title); ax.set_xlabel("原始事件电平 (dBFS RMS)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("增益 G (dB)")
    axes[0].legend(fontsize=8)
    axes[0].text(0.02, 0.97, "灰带 = 产品宣称 3–9 dB", transform=axes[0].transAxes, va="top", fontsize=8, color="#555")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "ad_gain_vs_level.png"), dpi=110); plt.close()


def fig_bars(d, bg):
    groups = [FOOT] + NT
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.26
    for j, (p, lab, c) in enumerate([("AI", "FPS AI", "#2f6da5"), ("EQ", "EQ（实际）", "#999"), ("EQm", "匹配静态 EQ", "#e6a23c")]):
        med = [d.loc[d.group == g, f"G_{p}"].median() for g in groups] + [bg[f"G_{p}"].median()]
        q1 = [d.loc[d.group == g, f"G_{p}"].quantile(.25) for g in groups] + [bg[f"G_{p}"].quantile(.25)]
        q3 = [d.loc[d.group == g, f"G_{p}"].quantile(.75) for g in groups] + [bg[f"G_{p}"].quantile(.75)]
        xs = np.arange(len(med)) + (j - 1) * w
        ax.bar(xs, med, w, color=c, label=lab)
        ax.errorbar(xs, med, [np.subtract(med, q1), np.subtract(q3, med)], fmt="none", c="k", lw=.8, capsize=2)
    ax.set_xticks(range(len(groups) + 1), groups + ["背景段*"])
    ax.axhline(0, c="k", lw=.6); ax.set_ylabel("中位增益 (dB)，误差线=四分位"); ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "ad_gain_bars.png"), dpi=110); plt.close()


def fig_response(f, resp):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    m = (f >= 60) & (f <= 16000)
    pd.DataFrame({"freq_hz": f[m], "gain_db": resp[m]}).to_csv(os.path.join(RES, "ad_response.csv"), index=False)
    ax.semilogx(f[m], resp[m], c="#e6a23c", label="匹配静态 EQ 频响（= FPS AI 在脚步上的平均频响）")
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("增益 (dB)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "ad_response.png"), dpi=110); plt.close()


def fig_fer(thr, fer, fer_bg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, data, title in [(axes[0], fer, "非脚步事件（枪声/换弹/道具）"), (axes[1], fer_bg, "背景段*")]:
        for p, lab, c in [("AI", "FPS AI", "#2f6da5"), ("EQ", "EQ（实际）", "#999"), ("EQm", "匹配静态 EQ", "#e6a23c")]:
            ax.plot(thr, np.array(data[p]) * 100, c=c, label=lab, marker=".")
        ax.axvline(FER_THR, c="k", ls=":", lw=.8)
        ax.set_title(title); ax.set_xlabel("错误增强阈值 (dB)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("FER (%)"); axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "ad_fer.png"), dpi=110); plt.close()


if __name__ == "__main__":
    main()
