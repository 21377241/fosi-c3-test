"""Sub-experiment E: is a footstep easier to detect after FPS AI / EQ processing?

Detector: PANNs Cnn14_DecisionLevelMax (AudioSet SED, 32 kHz), never used in labelling.
Ground truth: human labels only (f = footstep vs g/o = non-footstep), events detected on the raw track,
same time stamps on all three tracks. Each processed track is scaled to the raw track's clip RMS first,
so a global level change cannot help; what remains is the event's prominence and timbre.

  zs     : zero-shot, max framewise P("Walk, footsteps") in [t-0.05, t+0.3] s (whole clip as context)
  closed : closed set of game event categories (GROUPS): footstep share of the summed category maxima
  closed2: the same with footstep vs gunshot only
  top1   : whether footstep is the highest of the GROUPS categories (no threshold at all)
  probe  : PANNs fc1 embedding of [t-0.3, t+0.7] s -> scaler/PCA/logistic regression trained on the RAW
           track (leave-clips-out 5-fold), the same fold model then scores raw / fps / eq of the held-out clips

The decision threshold is fixed on the raw track (FPR_TARGET of the human non-footstep events) and reused
for fps / eq. Also reported: paired per-event score change (logit) and false alarms in background windows
(no detected event within BG_GUARD before / BG_TAIL after).
Outputs: cache/panns_scores.npz, results/detectability_events.csv, results/detectability_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from common import GAMES, TRACKS, load_triplet
from train_labeler import read_human, CODE

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "cache", "panns_hf", "models", "pretrained", "Cnn14_DecisionLevelMax_mAP=0.385.pth")
SR32, FPS_FRAMES = 32000, 100
ZS_WIN, EMB_WIN = (-0.05, 0.3), (-0.3, 0.7)
FOOT = "Walk, footsteps"
N_BOOT = 2000
FPR_TARGET = 0.10
BG_GUARD, BG_TAIL = 0.3, 1.0
NO_MATCH = "--nomatch" in sys.argv
SUFFIX = "_nomatch" if NO_MATCH else ""
LEVELS = [(-90, -45, "≤ -45 dBFS"), (-45, -35, "-45 ~ -35 dBFS"), (-35, 0, "> -35 dBFS")]
# Closed set: the game event categories of the test plan mapped to AudioSet classes, fixed before looking at
# results. Footstep first. Classes the models were seen to confuse footsteps with (hooves, crunch) are left out.
GROUPS = {
    "footstep": ["Walk, footsteps", "Run", "Shuffle"],
    "gunshot": ["Gunshot, gunfire", "Machine gun", "Cap gun", "Artillery fire", "Fusillade"],
    "explosion": ["Explosion", "Burst, pop", "Boom"],
    "impact": ["Thump, thud", "Knock", "Bang", "Slam", "Clang", "Thunk"],
    "mechanical": ["Mechanisms", "Clicking", "Ratchet, pawl", "Tick", "Beep, bleep"],
    "environment": ["Wind", "Vehicle", "Engine", "Music", "Speech", "Noise", "Rustle", "Whoosh, swoosh, swish"],
}


def load_model():
    # analysis/coverage.py shadows the `coverage` package that numba (via librosa) imports
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
    sys.modules.pop("coverage", None)
    import librosa.filters  # noqa: F401
    sys.path[:] = saved
    from panns_inference.models import Cnn14_DecisionLevelMax
    from panns_inference.config import labels
    m = Cnn14_DecisionLevelMax(sample_rate=SR32, window_size=1024, hop_size=320, mel_bins=64, fmin=50,
                               fmax=14000, classes_num=527, interpolate_mode="nearest")
    m.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    m = m.cuda().eval()
    feat = {}
    m.fc1.register_forward_hook(lambda mod, i, o: feat.__setitem__("fc1", o))
    return m, feat, labels.index(FOOT)


@torch.no_grad()
def scores(model, feat, ifoot, ev):
    zs = np.zeros((len(ev), len(TRACKS)), np.float32)
    emb = np.zeros((len(ev), len(TRACKS), 2048), np.float32)
    curves = {}
    for (game, clip), g in ev.groupby(["game", "clip_id"]):
        a, _ = load_triplet(game, clip)
        mono = {t: a[t].mean(1) for t in TRACKS}
        ref = np.sqrt(np.mean(mono["raw"] ** 2))
        for k, t in enumerate(TRACKS):
            gain = 1.0 if NO_MATCH else ref / (np.sqrt(np.mean(mono[t] ** 2)) + 1e-12)
            x = resample_poly(mono[t] * gain, 2, 3).astype(np.float32)
            fw = model(torch.from_numpy(x)[None].cuda())["framewise_output"][0, :, ifoot].cpu().numpy()
            curves[f"fw_{game}_{clip}_{t}"] = fw
            for j, tt in zip(g.index, g.t):
                lo, hi = int((tt + ZS_WIN[0]) * FPS_FRAMES), int((tt + ZS_WIN[1]) * FPS_FRAMES)
                zs[j, k] = fw[max(0, lo):max(lo + 1, hi)].max()
            segs = []
            for tt in g.t:
                s = int((tt + EMB_WIN[0]) * SR32)
                e = s + int((EMB_WIN[1] - EMB_WIN[0]) * SR32)
                seg = np.zeros(e - s, np.float32)
                lo, hi = max(0, s), min(len(x), e)
                seg[lo - s:hi - s] = x[lo:hi]
                segs.append(seg)
            for b in range(0, len(segs), 64):
                model(torch.from_numpy(np.stack(segs[b:b + 64])).cuda())
                emb[g.index[b:b + 64], k] = torch.relu(feat["fc1"]).mean(1).cpu().numpy()
        print(f"{game} test{clip}: {len(g)} events", flush=True)
    return zs, emb, curves


@torch.no_grad()
def group_curves(ev):
    """Framewise max probability within each GROUPS category: {gc_{game}_{clip}_{track}: (frames, n_groups)}."""
    from panns_inference.config import labels
    model, _, _ = load_model()
    gidx = [[labels.index(c) for c in cls] for cls in GROUPS.values()]
    out = {}
    for game, clip in ev[["game", "clip_id"]].drop_duplicates().itertuples(index=False):
        a, _ = load_triplet(game, clip)
        mono = {t: a[t].mean(1) for t in TRACKS}
        ref = np.sqrt(np.mean(mono["raw"] ** 2))
        for t in TRACKS:
            gain = 1.0 if NO_MATCH else ref / (np.sqrt(np.mean(mono[t] ** 2)) + 1e-12)
            x = resample_poly(mono[t] * gain, 2, 3).astype(np.float32)
            fw = model(torch.from_numpy(x)[None].cuda())["framewise_output"][0].cpu().numpy()
            out[f"gc_{game}_{clip}_{t}"] = np.stack([fw[:, i].max(1) for i in gidx], 1).astype(np.float32)
    return out


def event_window_max(ev, curves, prefix):
    """Max of each curve over [t+ZS_WIN] for every event: (n_events, n_tracks[, n_groups])."""
    first = next(iter(curves.values()))
    out = np.zeros((len(ev), len(TRACKS)) + first.shape[1:], np.float32)
    for (game, clip), g in ev.groupby(["game", "clip_id"]):
        for k, t in enumerate(TRACKS):
            c = curves[f"{prefix}_{game}_{clip}_{t}"]
            for j, tt in zip(g.index, g.t):
                lo, hi = int((tt + ZS_WIN[0]) * FPS_FRAMES), int((tt + ZS_WIN[1]) * FPS_FRAMES)
                out[j, k] = c[max(0, lo):max(lo + 1, hi)].max(0)
    return out


def closed_share(g, n=None):
    """Footstep share of the summed category maxima (first n categories)."""
    g = g[..., :n] if n else g
    return g[..., 0] / (g.sum(-1) + 1e-12)


def background_scores(ev, curves, prefix="fw"):
    """Max curve value in non-overlapping windows of the zero-shot window length that lie at least
    BG_GUARD before and BG_TAIL after every detected event. Returns {track: scores}, clip group per window."""
    out, grp = {t: [] for t in TRACKS}, []
    w = int((ZS_WIN[1] - ZS_WIN[0]) * FPS_FRAMES)
    for (game, clip), g in ev.groupby(["game", "clip_id"]):
        n = len(curves[f"{prefix}_{game}_{clip}_raw"])
        busy = np.zeros(n, bool)
        for tt in g.t:
            busy[max(0, int((tt - BG_GUARD) * FPS_FRAMES)):int((tt + BG_TAIL) * FPS_FRAMES)] = True
        for s in range(0, n - w, w):
            if not busy[s:s + w].any():
                for t in TRACKS:
                    out[t].append(curves[f"{prefix}_{game}_{clip}_{t}"][s:s + w].max(0))
                grp.append(f"{game}{clip}")
    return {t: np.array(v) for t, v in out.items()}, np.array(grp)


def probe():
    return make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
                         LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))


def cv_probe(Xtr, Xte_list, y, groups):
    out = [np.zeros(len(y)) for _ in Xte_list]
    for tr, te in GroupKFold(5).split(Xtr, y, groups):
        m = probe().fit(Xtr[tr], y[tr])
        for o, X in zip(out, Xte_list):
            o[te] = m.predict_proba(X[te])[:, 1]
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cluster_ci(fn, groups, rng):
    """95% CI of fn(index array) under a bootstrap that resamples whole clips."""
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = [fn(np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])) for _ in range(N_BOOT)]
    return np.nanpercentile(vals, [2.5, 97.5])


def fmt_ci(v, ci, pct=False):
    if pct:
        return f"{v * 100:+.1f} [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]"
    return f"{v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def main():
    ev = pd.read_csv(os.path.join(HERE, "results", "events.csv"))
    ev["event_id"] = np.arange(len(ev))
    cache = os.path.join(HERE, "cache", f"panns_scores{SUFFIX}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        zs, emb = d["zs"], d["emb"]
        curves = {k: d[k] for k in d.files if k.startswith("fw_")}
    else:
        model, feat, ifoot = load_model()
        zs, emb, curves = scores(model, feat, ifoot, ev)
        np.savez(cache, zs=zs, emb=emb, **curves)
    bg, bg_grp = background_scores(ev, curves)
    gcache = os.path.join(HERE, "cache", f"panns_groups{SUFFIX}.npz")
    if os.path.exists(gcache):
        gc = dict(np.load(gcache))
    else:
        gc = group_curves(ev)
        np.savez(gcache, **gc)
    gev = event_window_max(ev, gc, "gc")
    gbg, _ = background_scores(ev, gc, "gc")
    bgd = {"zs": bg,
           "closed": {t: closed_share(v) for t, v in gbg.items()},
           "closed2": {t: closed_share(v, 2) for t, v in gbg.items()}}

    h = read_human()
    h = h[h.human_label.isin(CODE)]
    idx = h.event_id.values
    y = (h.human_label == "f").astype(int).values
    groups = (ev.game + ev.clip_id.astype(str)).values[idx]
    peak = ev.raw_peak.values[idx]
    r1 = set(pd.read_csv(os.path.join(HERE, "labeling", "human_labels.csv")).event_id)
    clean = set(pd.read_csv(os.path.join(HERE, "clean_events", "clean_events.csv"), encoding="utf-8-sig").event_id)

    det = {"zs": [zs[idx, k] for k in range(3)],
           "closed": [closed_share(gev[idx, k]) for k in range(3)],
           "closed2": [closed_share(gev[idx, k], 2) for k in range(3)],
           "probe": cv_probe(emb[idx, 0], [emb[idx, k] for k in range(3)], y, groups)}
    top = pd.Series(np.array(list(GROUPS))[gev[idx, 0].argmax(1)])
    print("原始轨上六类中得分最高的类别：", {lab: top[y == v].value_counts().to_dict() for lab, v in [("脚步", 1), ("非脚步", 0)]})
    rows_ev = pd.DataFrame({"event_id": idx, "y": y, "group": groups, "raw_peak": peak,
                            "r1": [i in r1 for i in idx], "clean": [i in clean for i in idx]})
    for name, ss in det.items():
        for t, s in zip(TRACKS, ss):
            rows_ev[f"{name}_{t}"] = s
    rows_ev.to_csv(os.path.join(HERE, "results", f"detectability_events{SUFFIX}.csv"), index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(0)
    fo, nf = y == 1, y == 0
    r1m = rows_ev.r1.values
    print(f"人工标注事件 {len(y)}（脚步 {fo.sum()}，非脚步 {nf.sum()}），背景窗 {len(bg['raw'])} 个")
    lv_fps = (ev.raw_peak + ev.fps_gain).values[idx]
    print(f"响度本身区分脚步/非脚步的 AUC：原始 {roc_auc_score(y, -peak):.3f}，FPS AI 后 {roc_auc_score(y, -lv_fps):.3f}")

    summ = []
    for name, ss in det.items():
        raw = ss[0]
        thr = np.quantile(raw[nf], 1 - FPR_TARGET)
        print(f"\n== {name}  阈值 {thr:.3g}（原始轨非脚步误报率 {FPR_TARGET:.0%}）")
        base = None
        for t, s in zip(TRACKS, ss):
            row = dict(detector=name, track=t, thr=thr, auc=roc_auc_score(y, s),
                       auc_r1=roc_auc_score(y[r1m], s[r1m]),
                       recall=(s[fo] >= thr).mean(), fpr_events=(s[nf] >= thr).mean())
            if name in bgd:
                row["fpr_bg"] = (bgd[name][t] >= thr).mean()
            for lo_, hi_, lab in LEVELS:
                m = fo & (peak > lo_) & (peak <= hi_)
                row[f"recall {lab}"] = (s[m] >= thr).mean()
                row[f"n {lab}"] = int(m.sum())
            if t == "raw":
                base = row
            else:
                dl = logit(s) - logit(raw)
                row["dlogit_foot"] = np.median(dl[fo])
                row["dlogit_foot_ci"] = cluster_ci(lambda i: np.median(dl[i][y[i] == 1]), groups, rng)
                row["dlogit_non"] = np.median(dl[nf])
                row["dlogit_non_ci"] = cluster_ci(lambda i: np.median(dl[i][y[i] == 0]), groups, rng)
                hit_r, hit_t = raw >= thr, s >= thr
                row["d_recall_ci"] = cluster_ci(lambda i: hit_t[i][y[i] == 1].mean() - hit_r[i][y[i] == 1].mean(),
                                                groups, rng)
                row["d_fpr_ci"] = cluster_ci(lambda i: hit_t[i][y[i] == 0].mean() - hit_r[i][y[i] == 0].mean(),
                                             groups, rng)
                if name in bgd:
                    bg = bgd[name]
                    db = logit(bg[t]) - logit(bg["raw"])
                    row["dlogit_bg"] = np.median(db)
                    row["dlogit_bg_ci"] = cluster_ci(lambda i: np.median(db[i]), bg_grp, rng)
                    hb_r, hb_t = bg["raw"] >= thr, bg[t] >= thr
                    row["d_fpr_bg_ci"] = cluster_ci(lambda i: hb_t[i].mean() - hb_r[i].mean(), bg_grp, rng)
            summ.append(row)
            msg = (f"  {t:4s} AUC={row['auc']:.3f}（第1轮 {row['auc_r1']:.3f}）  脚步检出率={row['recall']:.1%}  "
                   f"非脚步误报={row['fpr_events']:.1%}")
            if name in bgd:
                msg += f"  背景误报={row['fpr_bg']:.1%}"
            msg += "  按电平检出率: " + " / ".join(f"{row[f'recall {lab}']:.0%}({row[f'n {lab}']})" for *_, lab in LEVELS)
            print(msg)
            if t != "raw":
                msg = (f"       相对原始: 检出率 {fmt_ci(row['recall'] - base['recall'], row['d_recall_ci'], True)} 个百分点，"
                       f"非脚步误报 {fmt_ci(row['fpr_events'] - base['fpr_events'], row['d_fpr_ci'], True)}；"
                       f"脚步分数 Δlogit {fmt_ci(row['dlogit_foot'], row['dlogit_foot_ci'])}，"
                       f"非脚步 {fmt_ci(row['dlogit_non'], row['dlogit_non_ci'])}")
                if name in bgd:
                    msg += (f"，背景 {fmt_ci(row['dlogit_bg'], row['dlogit_bg_ci'])}；背景误报 "
                            f"{fmt_ci(row['fpr_bg'] - base['fpr_bg'], row['d_fpr_bg_ci'], True)}")
                print(msg)
    top = (gev[idx].argmax(-1) == 0).astype(float)
    print("\n== top1  六类中得分最高的是否为脚步（无阈值）")
    for k, t in enumerate(TRACKS):
        s = top[:, k]
        row = dict(detector="top1", track=t, recall=s[fo].mean(), fpr_events=s[nf].mean(),
                   precision=(s * y).sum() / max(s.sum(), 1))
        for lo_, hi_, lab in LEVELS:
            m = fo & (peak > lo_) & (peak <= hi_)
            row[f"recall {lab}"] = s[m].mean()
            row[f"n {lab}"] = int(m.sum())
        if k:
            row["d_recall_ci"] = cluster_ci(lambda i: top[i, k][y[i] == 1].mean() - top[i, 0][y[i] == 1].mean(),
                                            groups, rng)
        summ.append(row)
        print(f"  {t:4s} 脚步判为脚步={row['recall']:.1%}  非脚步判为脚步={row['fpr_events']:.1%}  "
              f"判为脚步的准确率={row['precision']:.0%}" +
              (f"  相对原始 {fmt_ci(row['recall'] - summ[-1 - k]['recall'], row['d_recall_ci'], True)}" if k else ""))
    S = pd.DataFrame(summ)
    for c in [c for c in S.columns if c.endswith("_ci")]:
        S[c] = S[c].apply(lambda v: f"{v[0]:.4f},{v[1]:.4f}" if isinstance(v, np.ndarray) else "")
    S.to_csv(os.path.join(HERE, "results", f"detectability_summary{SUFFIX}.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()


