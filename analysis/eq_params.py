"""Parameters of the EQ condition (10-band peaking EQ, one preset per game) and its frequency response,
theoretical (RBJ peaking biquads) and measured from the aligned raw / EQ tracks (cross-spectral estimate)."""
import os
import numpy as np
import pandas as pd
from scipy.signal import csd, welch, sosfreqz
from common import load_triplet, SR

Q = 0.75
FREQS = [63, 125, 250, 500, 1000, 2000, 4000, 6000, 8000, 16000]
EQ_BANDS = {
    "valorant": [-3, -2, -2, 1, 1, 3, -2, -2, -2, -1],
    "pubg": [-2, -2, 0, 2, 3, 3, -1, -2, -2, -1],
}
HERE = os.path.dirname(os.path.abspath(__file__))


def peaking(f0, g, q, fs):
    A = 10 ** (g / 40)
    w = 2 * np.pi * f0 / fs
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return np.r_[b, a] / a[0]


def theory_response(game, f):
    sos = np.array([peaking(fc, g, Q, SR) for fc, g in zip(FREQS, EQ_BANDS[game])])
    _, h = sosfreqz(sos, worN=f, fs=SR)
    return 20 * np.log10(np.abs(h))


def measured_response(game, nperseg=4096):
    """Median over clips of |P_raw,eq / P_raw,raw| in dB, on the mono mix."""
    clips = pd.read_csv(os.path.join(HERE, "results", "clips.csv"))
    out = []
    for c in clips[clips.game == game].clip_id.unique():
        a, _ = load_triplet(game, c)
        x, y = a["raw"].mean(1), a["eq"].mean(1)
        f, pxy = csd(x, y, fs=SR, nperseg=nperseg)
        _, pxx = welch(x, fs=SR, nperseg=nperseg)
        out.append(20 * np.log10(np.abs(pxy) / pxx + 1e-12))
    return f, np.median(out, 0)
