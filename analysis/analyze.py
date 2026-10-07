"""Event-level measurements for raw / FPS / EQ.

For every transient event detected in the raw track we measure, for FPS and EQ:
  gain      = event peak level (processed) - event peak level (raw)          [dB]
  contrast  = (peak - local background) processed - (peak - background) raw  [dB]
  band_k    = per-octave-band gain inside the event window                   [dB]
and store AST (AudioSet) family scores of the raw track around the event for labelling.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import stft, find_peaks
from common import GAMES, TRACKS, load_triplet, SR

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUTDIR = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)

FAMILIES = {
    "foot_strict": ["Walk, footsteps", "Run", "Shuffle"],
    # AST tends to call game footsteps hooves (VALORANT) or crunch (gravel, PUBG)
    "foot_broad": ["Walk, footsteps", "Run", "Shuffle", "Clip-clop", "Horse", "Crunch"],
    "gun": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun",
            "Explosion", "Burst, pop", "Firecracker"],
    "knock": ["Knock", "Door", "Slam", "Bang", "Thump, thud", "Clang"],
    "vehicle": ["Vehicle", "Car", "Engine", "Accelerating, revving, vroom"],
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking"],
}

HOP = 240                    # 5 ms
RMS_WIN = 960                # 20 ms
BAND_EDGES = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
BAND_NAMES = ["63", "125", "250", "500", "1k", "2k", "4k", "8k"]


def rms_db(x):
    p = (x ** 2).sum(1)
    e = np.convolve(p, np.ones(RMS_WIN) / RMS_WIN, "same")[::HOP]
    return 10 * np.log10(e + 1e-12)


def band_power(x):
    f, _, Z = stft(x.mean(1), SR, nperseg=1024, noverlap=1024 - HOP, boundary=None)
    P = np.abs(Z) ** 2
    return np.stack([P[(f >= lo) & (f < lo * 2)].sum(0) for lo in BAND_EDGES[:-1]])


def onsets(x):
    f, t, Z = stft(x.mean(1), SR, nperseg=1024, noverlap=1024 - HOP, boundary=None)
    L = np.log10(np.abs(Z) + 1e-6)
    band = (f > 60) & (f < 16000)
    flux = np.concatenate([[0], np.maximum(np.diff(L[band], axis=1), 0).sum(0)])
    w = int(2.0 * SR / HOP) | 1
    win = np.lib.stride_tricks.sliding_window_view(np.pad(flux, w // 2, mode="reflect"), w)
    med = np.median(win, 1)
    mad = np.median(np.abs(win - med[:, None]), 1)
    peaks, _ = find_peaks(flux, height=med + 3 * mad + 1e-9, distance=int(0.08 * SR / HOP))
    return (peaks * HOP + 512) / SR


def family_scores(P, classes):
    out = {}
    for name, fam in FAMILIES.items():
        idx = [int(np.where(classes == c)[0][0]) for c in fam]
        out[name] = P[:, idx].astype(np.float32).max(1)
    return out


def band_gain_std(a, t):
    """Mean over octave bands of the std over time of processed/raw band gain (active frames).
    A static (LTI) EQ gives ~0; content-adaptive processing gives large values."""
    f, _, R = stft(a["raw"].mean(1), SR, nperseg=2048, noverlap=1024)
    _, _, Y = stft(a[t].mean(1), SR, nperseg=2048, noverlap=1024)
    R, Y = np.abs(R) ** 2, np.abs(Y) ** 2
    stds = []
    for lo in BAND_EDGES[:-1]:
        m = (f >= lo) & (f < lo * 2)
        rd = 10 * np.log10(R[m].sum(0) + 1e-12)
        act = rd > rd.max() - 50
        stds.append(np.std(10 * np.log10(Y[m].sum(0)[act] + 1e-12) - rd[act]))
    return float(np.mean(stds))


def analyze_clip(game, i):
    a, _ = load_triplet(game, i)
    d = np.load(os.path.join(CACHE, f"probs_{game}_{i}.npz"))
    classes, centers = d["classes"], d["centers"]
    lv = {t: rms_db(a[t]) for t in TRACKS}
    bp = {t: band_power(a[t]) for t in TRACKS}
    fam = family_scores(d["raw"], classes)
    fam_p = {t: family_scores(d[t], classes) for t in ["fps", "eq"]}
    n = min(len(lv["raw"]), bp["raw"].shape[1])
    f2s = SR / HOP

    rows = []
    for on in onsets(a["raw"]):
        k = int(on * f2s)
        pk = slice(k, k + int(0.08 * f2s))
        bg = slice(k - int(0.6 * f2s), k - int(0.05 * f2s))
        if bg.start < 0 or pk.stop >= n:
            continue
        peak = {t: lv[t][pk].max() for t in TRACKS}
        back = {t: np.percentile(lv[t][bg], 20) for t in TRACKS}
        prom = peak["raw"] - back["raw"]
        if prom < 6 or peak["raw"] < -85:
            continue
        w = int(np.argmin(np.abs(centers - (on + 0.1))))
        r = dict(game=game, clip_id=i, t=round(on, 3), raw_peak=peak["raw"], raw_prom=prom)
        for name in FAMILIES:
            r[f"p_{name}"] = float(fam[name][w])
        for t in ["fps", "eq"]:
            r[f"{t}_gain"] = peak[t] - peak["raw"]
            r[f"{t}_contrast"] = (peak[t] - back[t]) - prom
            r[f"{t}_bg_gain"] = back[t] - back["raw"]
            for name in ["foot_broad", "gun"]:
                r[f"{t}_p_{name}"] = float(fam_p[t][name][w])
            ev_bands = 10 * np.log10(bp[t][:, pk].sum(1) + 1e-12) - 10 * np.log10(bp["raw"][:, pk].sum(1) + 1e-12)
            for b, nm in enumerate(BAND_NAMES):
                r[f"{t}_band_{nm}"] = ev_bands[b]
        rows.append(r)

    clip = dict(game=game, clip_id=i, dur=len(a["raw"]) / SR,
                fps_gain=10 * np.log10(np.mean(a["fps"] ** 2) / np.mean(a["raw"] ** 2)),
                eq_gain=10 * np.log10(np.mean(a["eq"] ** 2) / np.mean(a["raw"] ** 2)),
                fps_tv=band_gain_std(a, "fps"), eq_tv=band_gain_std(a, "eq"))
    for name in FAMILIES:
        clip[f"p90_{name}"] = float(np.percentile(fam[name], 90))
    return rows, clip


if __name__ == "__main__":
    ev, cl = [], []
    for game, cfg in GAMES.items():
        for i in range(1, cfg["n"] + 1):
            rows, clip = analyze_clip(game, i)
            ev += rows
            cl.append(clip)
            print(f"{game} test{i}: {len(rows)} events", flush=True)
    pd.DataFrame(ev).to_csv(os.path.join(OUTDIR, "events.csv"), index=False)
    pd.DataFrame(cl).to_csv(os.path.join(OUTDIR, "clips.csv"), index=False)
