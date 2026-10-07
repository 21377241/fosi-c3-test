"""Hand-crafted, level-invariant acoustic features per event (RAW track).

Output: cache/features_acoustic.npz  (X aligned with results/events.csv rows, names)
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import stft
from common import GAMES, ROOT, read, SR
from analyze import rms_db, HOP, BAND_EDGES

HERE = os.path.dirname(os.path.abspath(__file__))
F2S = SR / HOP


def event_features(x, lv, f, P, t, neighbours):
    k = int(t * F2S)
    pk = slice(k, k + int(0.08 * F2S))
    spec = P[:, pk].sum(1) + 1e-20
    bands = np.array([spec[(f >= lo) & (f < lo * 2)].sum() for lo in BAND_EDGES[:-1]])
    band_rel = 10 * np.log10(bands / bands.sum())
    m = (f > 50) & (f < 20000)
    centroid = np.log2((f[m] * spec[m]).sum() / spec[m].sum())
    flat = np.exp(np.mean(np.log(spec[m]))) / np.mean(spec[m])
    cum = np.cumsum(spec[m]) / spec[m].sum()
    rolloff = np.log2(f[m][np.searchsorted(cum, 0.85)] + 1)

    seg = lv[k: k + int(0.6 * F2S)]
    ipk = int(np.argmax(seg[: int(0.08 * F2S)]))
    after = seg[ipk:] - seg[ipk]
    d10 = (np.argmax(after < -10) if (after < -10).any() else len(after)) / F2S
    d20 = (np.argmax(after < -20) if (after < -20).any() else len(after)) / F2S
    pre = lv[max(0, k - int(0.03 * F2S)): k + ipk + 1]
    attack = (len(pre) - np.argmax(pre > seg[ipk] - 10)) / F2S if len(pre) else 0

    near = neighbours[(np.abs(neighbours - t) <= 1.5) & (neighbours != t)]
    ioi = np.diff(np.sort(np.append(near, t)))
    n_near = len(near)
    med_ioi = np.median(ioi) if len(ioi) else 1.5
    cv_ioi = (np.std(ioi) / (np.mean(ioi) + 1e-6)) if len(ioi) > 1 else 1.0
    return np.concatenate([band_rel, [centroid, flat, rolloff, d10, d20, attack, n_near, med_ioi, cv_ioi]])


NAMES = [f"band_{lo}" for lo in BAND_EDGES[:-1]] + \
        ["log_centroid", "flatness", "log_rolloff", "decay10", "decay20", "attack", "n_near", "median_ioi", "cv_ioi"]


if __name__ == "__main__":
    ev = pd.read_csv(os.path.join(HERE, "results", "events.csv"))
    X = np.zeros((len(ev), len(NAMES)), np.float32)
    for (game, clip), g in ev.groupby(["game", "clip_id"]):
        x = read(os.path.join(ROOT, GAMES[game]["raw"].format(n=f"test{clip}")))
        lv = rms_db(x)
        f, _, Z = stft(x.mean(1), SR, nperseg=1024, noverlap=1024 - HOP, boundary=None)
        P = np.abs(Z) ** 2
        ts = g.t.values
        for idx, t in zip(g.index, ts):
            X[idx] = event_features(x, lv, f, P, t, ts)
        print(f"{game} test{clip}: {len(g)} events", flush=True)
    np.savez(os.path.join(HERE, "cache", "features_acoustic.npz"), X=X, names=np.array(NAMES))
