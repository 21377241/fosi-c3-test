"""Independent PUBG comparison: old FPS AI vs new-model FPS AI on the same events.

Uses the existing clean-event segments, background stretches and blind-review
footstep-overlap flags. Does not modify the main report pipeline.

Output:
  results/pubg_newmodel_events.csv
  results/pubg_newmodel_background.csv
  results/pubg_newmodel_compare.png
  docs/PUBG 新旧模型对比.html
"""
import os
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

from common import ROOT, SR, GAMES, read, find_lag
from report_ad import (HERE, RES, FOOT, NT, NT_MODEL, FER_THR, CLAIM, BG_GUARD, BG_MIN,
                       gain_db, group_of, load_segments, model_pool, matched_gap, cluster_ci)
from analyze import rms_db
from window_sensitivity import auto_end, F2S
from coverage import union
from report_final import SPECIAL, fer_rate, sgn, pct

LAB = os.path.join(HERE, "labeling")
OUT_HTML = os.path.join(os.path.dirname(HERE), "docs", "PUBG 新旧模型对比.html")
NEW_PAT = "pubg/pubg_output_newmodel/{n}.wav"
SHORT = {FOOT: "脚步", NT[0]: "枪声", NT[1]: "换弹", NT[2]: "烟雾弹/吃药/爆能器", NT_MODEL: "非脚步（模型）"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_pubg_quad(idx):
    """Aligned (raw, old fps, new fps, eq) for one PUBG clip."""
    cfg = GAMES["pubg"]
    name = f"test{idx}"
    x = {
        "raw": read(os.path.join(ROOT, cfg["raw"].format(n=name))),
        "old": read(os.path.join(ROOT, cfg["fps"].format(n=name))),
        "new": read(os.path.join(ROOT, NEW_PAT.format(n=name))),
        "eq": read(os.path.join(ROOT, cfg["eq"].format(n=name))),
    }
    lags = {"raw": 0}
    for t in ["old", "new", "eq"]:
        lags[t] = find_lag(x["raw"], x[t])
    k0 = max(0, -min(lags.values()))
    aligned = {t: x[t][k0 + lags[t]:] for t in x}
    L = min(len(v) for v in aligned.values())
    return {t: v[:L] for t, v in aligned.items()}, lags


def spec_mse(proc, raw, f_keep):
    _, _, Zr = stft(raw.mean(1), SR, nperseg=1024, noverlap=768)
    eps = np.abs(Zr).max() * 10 ** (-80 / 20)
    _, _, Zp = stft(proc.mean(1), SR, nperseg=1024, noverlap=768)
    d = (20 * np.log10(np.abs(Zp) + eps) - 20 * np.log10(np.abs(Zr) + eps))[f_keep]
    return float((d ** 2).mean()), float(d.var())


def measure():
    meta, ev, _ = load_segments()
    meta = meta[meta.game == "pubg"].copy()
    meta["group"] = [group_of(r) for r in meta.itertuples()]
    meta = meta[meta.group.notna()]
    pool = model_pool(ev)
    pool = pool[pool.game == "pubg"]

    fz = np.fft.rfftfreq(1024, 1 / SR)
    keep = (fz >= 50) & (fz <= 16000)
    rows, mrows, bg_rows, lags_all = [], [], [], []

    for clip in sorted(meta.clip_id.unique()):
        a, lags = load_pubg_quad(int(clip))
        lags_all.append({"clip": clip, **lags})
        g = meta[meta.clip_id == clip]
        for r in g.itertuples():
            s = int(round(r.seg_start * SR))
            e = s + int(r.seg_dur_ms / 1000 * SR)
            raw = a["raw"][s:e]
            old, new, eq = a["old"][s:e], a["new"][s:e], a["eq"][s:e]
            mse_o, sh_o = spec_mse(old, raw, keep)
            mse_n, sh_n = spec_mse(new, raw, keep)
            mse_e, sh_e = spec_mse(eq, raw, keep)
            rows.append(dict(
                event_id=r.event_id, group=r.group, note=r.note, purity=r.purity,
                game="pubg", clip_key=f"pubg{clip}", clip_id=clip,
                raw_level=20 * np.log10(np.sqrt((raw ** 2).mean()) + 1e-12),
                G_old=gain_db(old, raw), G_new=gain_db(new, raw), G_EQ=gain_db(eq, raw),
                MSE_old=mse_o, MSE_new=mse_n, MSE_EQ=mse_e,
                MSEshape_old=sh_o, MSEshape_new=sh_n, MSEshape_EQ=sh_e,
                dG=gain_db(new, raw) - gain_db(old, raw),
            ))
        e_t = ev[(ev.game == "pubg") & (ev.clip_id == clip)].t.sort_values().values
        lv = rms_db(a["raw"])
        for r in pool[pool.clip_id == clip].itertuples():
            nxt = e_t[e_t > r.t + 0.03]
            end = max(auto_end(lv, int(r.t * F2S)), 0.03)
            if len(nxt):
                end = min(end, nxt[0] - r.t)
            s, e = int((r.t - 0.02) * SR), min(len(a["raw"]), int((r.t + end + 0.05) * SR))
            raw = a["raw"][s:e]
            mrows.append(dict(
                event_id=r.event_id, group=r.group, game="pubg", clip_key=f"pubg{clip}",
                raw_level=20 * np.log10(np.sqrt((raw ** 2).mean()) + 1e-12),
                G_old=gain_db(a["old"][s:e], raw), G_new=gain_db(a["new"][s:e], raw),
                G_EQ=gain_db(a["eq"][s:e], raw),
                dG=gain_db(a["new"][s:e], raw) - gain_db(a["old"][s:e], raw),
            ))
        T = len(lv) / F2S
        occ = union([(t - BG_GUARD, t + max(auto_end(lv, int(t * F2S)), 0.03) + BG_GUARD) for t in e_t])
        edges = [0.0] + [x for iv in occ for x in iv] + [T]
        for lo, hi in zip(edges[::2], edges[1::2]):
            lo, hi = max(lo, 0.3), min(hi, T - 0.1)
            if hi - lo >= BG_MIN:
                s, e = int(lo * SR), int(hi * SR)
                raw = a["raw"][s:e]
                bg_rows.append(dict(
                    game="pubg", clip=clip, clip_key=f"pubg{clip}", start=lo, dur=hi - lo,
                    raw_level=20 * np.log10(np.sqrt((raw ** 2).mean()) + 1e-12),
                    G_old=gain_db(a["old"][s:e], raw), G_new=gain_db(a["new"][s:e], raw),
                    G_EQ=gain_db(a["eq"][s:e], raw),
                ))

    d = pd.DataFrame(rows)
    dm = pd.DataFrame(mrows)
    bg = pd.DataFrame(bg_rows)

    blind = pd.read_csv(os.path.join(LAB, "blind_r5_key.csv"), encoding="utf-8-sig").merge(
        pd.read_csv(os.path.join(LAB, "blind_review_r5.csv"), encoding="utf-8-sig", keep_default_na=False)
        .rename(columns={"序号": "idx", "判断_原始轨": "盲审判断"}), on="idx")
    has_step = set(blind.loc[blind.group.isin(NT) & (blind["盲审判断"] != "无脚步"), "event_id"])
    d = d.assign(step_ov=d.event_id.isin(has_step),
                 special=d.note.isin(SPECIAL))
    return d, dm, bg, pd.DataFrame(lags_all)


def summarize(d, dm, bg):
    fo = d[d.group == FOOT]
    fo_all = pd.concat([fo, dm[dm.group == FOOT]], ignore_index=True)
    nt = d[d.group.isin(NT)]
    nt_clean = nt[~nt.step_ov]
    ov = nt[nt.step_ov]
    gun = d[(d.group == NT[0]) & ~d.step_ov]

    def med(x, col):
        return float(x[col].median()) if len(x) else float("nan")

    def rate_ge3(x, col):
        return float((x[col] >= 3).mean()) if len(x) else float("nan")

    # fer_rate expects G_AI / G_EQ column names; wrap per-model views
    def as_ai(x, col):
        y = x.copy()
        y["G_AI"] = y[col]
        return y

    fer = {k: fer_rate(as_ai(nt, f"G_{k}"), "AI") for k in ["old", "new", "EQ"]}
    fer_g = {k: fer_rate(as_ai(d[d.group == NT[0]], f"G_{k}"), "AI") for k in ["old", "new", "EQ"]}
    fer_bg = {k: fer_rate(as_ai(bg.assign(step_ov=False, special=False), f"G_{k}"), "AI") for k in ["old", "new", "EQ"]}

    gap = {}
    for k in ["old", "new", "EQ"]:
        fo_k = fo.assign(**{f"G_{k}": fo[f"G_{k}"]})
        gun_k = d[d.group == NT[0]].assign(**{f"G_{k}": d.loc[d.group == NT[0], f"G_{k}"]})
        # matched_gap looks up G_{p}; temporarily alias
        fo_tmp = fo.rename(columns={f"G_{k}": "G_tmp"})
        gun_tmp = d[d.group == NT[0]].rename(columns={f"G_{k}": "G_tmp"})
        fo_tmp["G_AI"] = fo_tmp["G_tmp"]
        gun_tmp["G_AI"] = gun_tmp["G_tmp"]
        gap[k] = matched_gap(fo_tmp, gun_tmp, "AI")

    mse = {k: med(gun, f"MSE_{k}") for k in ["old", "new", "EQ"]}
    mse_sh = {k: med(gun, f"MSEshape_{k}") for k in ["old", "new", "EQ"]}

    by_grp = []
    for g in [FOOT] + NT:
        x = d[d.group == g]
        by_grp.append({
            "类别": SHORT[g], "n": len(x),
            "旧模型中位": sgn(med(x, "G_old")), "新模型中位": sgn(med(x, "G_new")),
            "Δ（新−旧）": sgn(med(x, "G_new") - med(x, "G_old")),
            "实际 EQ 中位": sgn(med(x, "G_EQ")),
            "旧 ≥3 dB": pct(rate_ge3(x, "G_old")), "新 ≥3 dB": pct(rate_ge3(x, "G_new")),
        })
    by_grp.append({
        "类别": "不含脚步的非脚步（合计）", "n": len(nt_clean),
        "旧模型中位": sgn(med(nt_clean, "G_old")), "新模型中位": sgn(med(nt_clean, "G_new")),
        "Δ（新−旧）": sgn(med(nt_clean, "G_new") - med(nt_clean, "G_old")),
        "实际 EQ 中位": sgn(med(nt_clean, "G_EQ")),
        "旧 ≥3 dB": pct(rate_ge3(nt_clean, "G_old")), "新 ≥3 dB": pct(rate_ge3(nt_clean, "G_new")),
    })
    by_grp.append({
        "类别": "与脚步重叠的非脚步", "n": len(ov),
        "旧模型中位": sgn(med(ov, "G_old")), "新模型中位": sgn(med(ov, "G_new")),
        "Δ（新−旧）": sgn(med(ov, "G_new") - med(ov, "G_old")),
        "实际 EQ 中位": sgn(med(ov, "G_EQ")),
        "旧 ≥3 dB": pct(rate_ge3(ov, "G_old")), "新 ≥3 dB": pct(rate_ge3(ov, "G_new")),
    })
    by_grp.append({
        "类别": "背景段", "n": len(bg),
        "旧模型中位": sgn(med(bg, "G_old")), "新模型中位": sgn(med(bg, "G_new")),
        "Δ（新−旧）": sgn(med(bg, "G_new") - med(bg, "G_old")),
        "实际 EQ 中位": sgn(med(bg, "G_EQ")),
        "旧 ≥3 dB": pct(rate_ge3(bg, "G_old")), "新 ≥3 dB": pct(rate_ge3(bg, "G_new")),
    })

    # false-boost detail on clean non-footsteps
    fb_old = nt_clean[(nt_clean.G_old > FER_THR) & ~nt_clean.special]
    fb_new = nt_clean[(nt_clean.G_new > FER_THR) & ~nt_clean.special]
    sp = nt_clean[nt_clean.special]

    ci_old = cluster_ci(fo, "G_old")
    ci_new = cluster_ci(fo, "G_new")

    return dict(
        fo=fo, fo_all=fo_all, nt=nt, nt_clean=nt_clean, ov=ov, gun=gun, bg=bg,
        by_grp=by_grp, fer=fer, fer_g=fer_g, fer_bg=fer_bg, gap=gap,
        mse=mse, mse_sh=mse_sh, fb_old=fb_old, fb_new=fb_new, sp=sp,
        ci_old=ci_old, ci_new=ci_new,
        g_old=med(fo, "G_old"), g_new=med(fo, "G_new"),
        g_old_all=med(fo_all, "G_old"), g_new_all=med(fo_all, "G_new"),
        gun_old=med(gun, "G_old"), gun_new=med(gun, "G_new"),
        ge3_old=rate_ge3(fo, "G_old"), ge3_new=rate_ge3(fo, "G_new"),
    )


def fig_scatter(s, path):
    fo, nt_clean, ov = s["fo"], s["nt_clean"], s["ov"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, col, title in zip(axes, ["G_old", "G_new"], ["旧模型（当前报告）", "新模型"]):
        ax.axhline(0, color="#999", lw=.8)
        ax.axhline(3, color="#bbb", ls="--", lw=.8)
        ax.scatter(fo.raw_level, fo[col], s=18, c="#d62728", alpha=.55, lw=0, label=f"脚步（{len(fo)}）")
        ax.scatter(nt_clean.raw_level, nt_clean[col], s=22, c="#555", alpha=.7, lw=0,
                   label=f"不含脚步的非脚步（{len(nt_clean)}）")
        if len(ov):
            ax.scatter(ov.raw_level, ov[col], s=48, marker="D", c="#ff7f0e", ec="k", lw=.5,
                       label=f"与脚步重叠（{len(ov)}）", zorder=3)
        ax.set_xlabel("原始电平（dBFS）")
        ax.set_title(title)
        ax.grid(True, alpha=.25)
    axes[0].set_ylabel("增益（dB）")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=.9)
    fig.suptitle("PUBG：同一批人工确认事件上的增益分布", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_delta(s, path):
    fo, nt_clean = s["fo"], s["nt_clean"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, color="#999", lw=.8)
    ax.scatter(fo.raw_level, fo.dG, s=18, c="#d62728", alpha=.55, lw=0, label="脚步")
    ax.scatter(nt_clean.raw_level, nt_clean.dG, s=22, c="#555", alpha=.7, lw=0, label="不含脚步的非脚步")
    ax.set_xlabel("原始电平（dBFS）")
    ax.set_ylabel("ΔG = 新 − 旧（dB）")
    ax.set_title("新模型相对旧模型的增益变化（PUBG）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def table(rows):
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def img(path):
    with open(path, "rb") as fh:
        return f'<img src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}">'


def build_html(s, fig1, fig2):
    fo, gun = s["fo"], s["gun"]
    dg_fo = s["g_new"] - s["g_old"]
    dg_gun = s["gun_new"] - s["gun_old"]
    dfer = s["fer"]["new"] - s["fer"]["old"]
    dmse = s["mse"]["new"] - s["mse"]["old"]

    # improvement bullets (rule-based, same directions as the main report)
    bullets = []
    if s["g_new"] > s["g_old"] + 0.3:
        bullets.append(f"脚步中位增益提高 {sgn(dg_fo)} dB（{sgn(s['g_old'])} → {sgn(s['g_new'])}）。")
    elif s["g_new"] < s["g_old"] - 0.3:
        bullets.append(f"脚步中位增益下降 {sgn(dg_fo)} dB（{sgn(s['g_old'])} → {sgn(s['g_new'])}）。")
    else:
        bullets.append(f"脚步中位增益基本持平（{sgn(s['g_old'])} → {sgn(s['g_new'])}，差 {sgn(dg_fo)} dB）。")

    if abs(s["gun_new"]) < abs(s["gun_old"]) - 0.2:
        bullets.append(f"不含脚步的枪声更接近 0：{sgn(s['gun_old'])} → {sgn(s['gun_new'])} dB。")
    elif abs(s["gun_new"]) > abs(s["gun_old"]) + 0.2:
        bullets.append(f"不含脚步的枪声偏离 0 增大：{sgn(s['gun_old'])} → {sgn(s['gun_new'])} dB。")
    else:
        bullets.append(f"不含脚步的枪声中位变化不大（{sgn(s['gun_old'])} → {sgn(s['gun_new'])} dB）。")

    if s["fer"]["new"] < s["fer"]["old"] - 1e-9:
        bullets.append(f"不含脚步的非脚步误增强率 FER 下降：{pct(s['fer']['old'])} → {pct(s['fer']['new'])}。")
    elif s["fer"]["new"] > s["fer"]["old"] + 1e-9:
        bullets.append(f"不含脚步的非脚步误增强率 FER 上升：{pct(s['fer']['old'])} → {pct(s['fer']['new'])}。")
    else:
        bullets.append(f"FER 不变：{pct(s['fer']['old'])}。")

    if s["mse"]["new"] < s["mse"]["old"] - 0.2:
        bullets.append(f"纯净枪声频谱总变化 MSE 下降：{s['mse']['old']:.1f} → {s['mse']['new']:.1f} dB²。")
    elif s["mse"]["new"] > s["mse"]["old"] + 0.2:
        bullets.append(f"纯净枪声频谱总变化 MSE 上升：{s['mse']['old']:.1f} → {s['mse']['new']:.1f} dB²。")
    else:
        bullets.append(f"纯净枪声 MSE 基本持平（{s['mse']['old']:.1f} → {s['mse']['new']:.1f} dB²）。")

    product = [
        {"指标": "脚步中位增益（dB，人工）", "旧模型": sgn(s["g_old"]), "新模型": sgn(s["g_new"]),
         "Δ（新−旧）": sgn(dg_fo), "实际 EQ": sgn(fo.G_EQ.median())},
        {"指标": "脚步 ≥ 3 dB 比例", "旧模型": pct(s["ge3_old"]), "新模型": pct(s["ge3_new"]),
         "Δ（新−旧）": pct(s["ge3_new"] - s["ge3_old"]), "实际 EQ": pct((fo.G_EQ >= 3).mean())},
        {"指标": "脚步（人工+模型）中位增益", "旧模型": sgn(s["g_old_all"]), "新模型": sgn(s["g_new_all"]),
         "Δ（新−旧）": sgn(s["g_new_all"] - s["g_old_all"]), "实际 EQ": "—"},
        {"指标": "不含脚步的枪声中位增益（dB）", "旧模型": sgn(s["gun_old"]), "新模型": sgn(s["gun_new"]),
         "Δ（新−旧）": sgn(dg_gun), "实际 EQ": sgn(gun.G_EQ.median()) if len(gun) else "—"},
        {"指标": "同电平 ΔG（脚步−枪声，dB）",
         "旧模型": f"{sgn(s['gap']['old'][0])}（n={s['gap']['old'][1]}）",
         "新模型": f"{sgn(s['gap']['new'][0])}（n={s['gap']['new'][1]}）",
         "Δ（新−旧）": sgn(s["gap"]["new"][0] - s["gap"]["old"][0]),
         "实际 EQ": f"{sgn(s['gap']['EQ'][0])}（n={s['gap']['EQ'][1]}）"},
        {"指标": "不含脚步的非脚步 FER", "旧模型": pct(s["fer"]["old"]), "新模型": pct(s["fer"]["new"]),
         "Δ（新−旧）": pct(dfer), "实际 EQ": pct(s["fer"]["EQ"])},
        {"指标": "枪声误增强率（不含脚步）", "旧模型": pct(s["fer_g"]["old"]), "新模型": pct(s["fer_g"]["new"]),
         "Δ（新−旧）": pct(s["fer_g"]["new"] - s["fer_g"]["old"]), "实际 EQ": pct(s["fer_g"]["EQ"])},
        {"指标": "背景段误增强率", "旧模型": pct(s["fer_bg"]["old"]), "新模型": pct(s["fer_bg"]["new"]),
         "Δ（新−旧）": pct(s["fer_bg"]["new"] - s["fer_bg"]["old"]), "实际 EQ": pct(s["fer_bg"]["EQ"])},
        {"指标": "纯净枪声频谱总变化 MSE（dB²）", "旧模型": f"{s['mse']['old']:.1f}", "新模型": f"{s['mse']['new']:.1f}",
         "Δ（新−旧）": f"{dmse:+.1f}", "实际 EQ": f"{s['mse']['EQ']:.1f}"},
        {"指标": "纯净枪声频谱形状变化（dB²）", "旧模型": f"{s['mse_sh']['old']:.1f}", "新模型": f"{s['mse_sh']['new']:.1f}",
         "Δ（新−旧）": f"{s['mse_sh']['new'] - s['mse_sh']['old']:+.1f}", "实际 EQ": f"{s['mse_sh']['EQ']:.1f}"},
    ]

    fb_rows = []
    for tag, x in [("旧模型误增强", s["fb_old"]), ("新模型误增强", s["fb_new"]), ("吃药声（特化，不计 FER）", s["sp"])]:
        for r in x.itertuples():
            fb_rows.append({
                "集合": tag, "event_id": r.event_id, "类别": SHORT.get(r.group, r.group),
                "备注": r.note or "—", "原始电平": f"{r.raw_level:.0f}",
                "旧增益": sgn(r.G_old), "新增益": sgn(r.G_new), "Δ": sgn(r.dG),
            })
    if not fb_rows:
        fb_rows = [{"集合": "—", "event_id": "—", "类别": "无", "备注": "—", "原始电平": "—",
                    "旧增益": "—", "新增益": "—", "Δ": "—"}]

    ci_o = s["ci_old"]
    ci_n = s["ci_new"]
    ci_o_txt = f"[{sgn(ci_o[0])}, {sgn(ci_o[1])}]" if np.isfinite(ci_o[0]) else "—"
    ci_n_txt = f"[{sgn(ci_n[0])}, {sgn(ci_n[1])}]" if np.isfinite(ci_n[0]) else "—"

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>PUBG 新旧模型对比</title>
<style>
body{{font-family:"Microsoft YaHei","Segoe UI",sans-serif;max-width:920px;margin:32px auto;padding:0 20px;line-height:1.55;color:#222}}
h1{{font-size:1.45rem;margin-bottom:.2em}} h2{{font-size:1.15rem;margin-top:1.6em;border-bottom:1px solid #ddd;padding-bottom:.2em}}
.box{{background:#f4f6f8;border-left:4px solid #1f4e79;padding:12px 16px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}} th,td{{border:1px solid #ccc;padding:5px 8px;text-align:center}}
th{{background:#eef2f6}} td:first-child,th:first-child{{text-align:left}}
.tw{{overflow-x:auto;margin:10px 0}} img{{max-width:100%;height:auto}}
.note{{color:#555;font-size:.88rem}} ul{{padding-left:1.2em}}
</style></head><body>
<h1>PUBG：更新模型相对旧模型的独立对比</h1>
<p class="note">范围仅限 PUBG 15 段测试素材。事件边界、人工标注与盲审（是否含脚步）沿用主报告；
增益定义与主报告相同（事件窗内 RMS 比，dB）。吃药声按品牌方说明不计为误增强。本页不改动主报告数字。</p>

<div class="box">
<b>一句话：</b>
脚步（人工，n={len(fo)}）中位增益 {sgn(s['g_old'])} → {sgn(s['g_new'])} dB（{ci_o_txt} → {ci_n_txt}）；
不含脚步的枪声 {sgn(s['gun_old'])} → {sgn(s['gun_new'])} dB；
FER {pct(s['fer']['old'])} → {pct(s['fer']['new'])}；
纯净枪声 MSE {s['mse']['old']:.1f} → {s['mse']['new']:.1f} dB²。
</div>

<h2>1. 相对旧模型的主要变化</h2>
<ul>{''.join(f'<li>{b}</li>' for b in bullets)}</ul>

<h2>2. 产品向主对比（PUBG）</h2>
{table(product)}
<p class="note">FER 分母为盲审确认不含脚步的非脚步事件（n={len(s['nt_clean'])}）；
与脚步重叠的非脚步（n={len(s['ov'])}）仍留在样本中，但不计入 FER。
同电平 ΔG：脚步相对枪声的选择性优势，越大越好。</p>

<h2>3. 按类别的中位增益</h2>
{table(s['by_grp'])}

<h2>4. 增益分布与新旧差</h2>
{img(fig1)}
{img(fig2)}

<h2>5. 误增强明细（不含脚步）</h2>
<p class="note">旧模型误增强 {len(s['fb_old'])} 个，新模型误增强 {len(s['fb_new'])} 个
（阈值 &gt; {FER_THR:.0f} dB，已排除吃药声）。</p>
{table(fb_rows)}

<h2>6. 方法说明</h2>
<ol>
<li>新模型音频：<code>fps_test/pubg/pubg_output_newmodel/test1–15.wav</code>，与旧模型
<code>pubg_output_onlyfps</code> 文件长度一致，按与原始轨的互相关对齐后测量。</li>
<li>人工事件来自 <code>clean_events.csv</code> 中 PUBG 子集；模型高置信度脚步用于“人工+模型”口径。</li>
<li>是否含脚步沿用主报告盲审 <code>blind_review_r5</code>；特化增强类别为 {sorted(SPECIAL)}。</li>
<li>本分析独立于详细版 / 简版 LaTeX 报告，不会覆盖主报告的 CSV。</li>
</ol>
</body></html>"""


def main():
    print("measuring PUBG old vs new model …")
    d, dm, bg, lags = measure()
    d.to_csv(os.path.join(RES, "pubg_newmodel_events.csv"), index=False, encoding="utf-8-sig")
    if len(dm):
        dm.to_csv(os.path.join(RES, "pubg_newmodel_model_events.csv"), index=False, encoding="utf-8-sig")
    bg.to_csv(os.path.join(RES, "pubg_newmodel_background.csv"), index=False, encoding="utf-8-sig")
    lags.to_csv(os.path.join(RES, "pubg_newmodel_lags.csv"), index=False)

    s = summarize(d, dm, bg)
    fig1 = os.path.join(RES, "pubg_newmodel_scatter.png")
    fig2 = os.path.join(RES, "pubg_newmodel_delta.png")
    fig_scatter(s, fig1)
    fig_delta(s, fig2)

    html = build_html(s, fig1, fig2)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"PUBG footsteps n={len(s['fo'])}: old {s['g_old']:+.1f} -> new {s['g_new']:+.1f} dB")
    print(f"clean gun n={len(s['gun'])}: old {s['gun_old']:+.1f} -> new {s['gun_new']:+.1f} dB")
    print(f"FER: old {s['fer']['old']:.0%} -> new {s['fer']['new']:.0%}")
    print(f"gun MSE: old {s['mse']['old']:.1f} -> new {s['mse']['new']:.1f}")
    print(f"-> {OUT_HTML}")


if __name__ == "__main__":
    main()
