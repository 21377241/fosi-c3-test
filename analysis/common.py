import os
import warnings
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fps_test")
SR = 48000

GAMES = {
    "pubg": {
        "raw": "pubg/pubg_test_raw/{n}.wav",
        "fps": "pubg/pubg_output_onlyfps/{n}.wav",
        "eq": "pubg/pubg_output_onlyeq/{n}.wav_eq.wav",
        "n": 15,
    },
    "valorant": {
        "raw": "VALORANT/valorant_test_raw/{n}.wav",
        "fps": "VALORANT/valorant_output_onlyfps/{n}.wav",
        "eq": "VALORANT/valorant_output_onlyeq/{n}.wav_eq.wav",
        "n": 16,
    },
}
TRACKS = ["raw", "fps", "eq"]


def read(path):
    sr, x = wavfile.read(path)
    assert sr == SR, (path, sr)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    x = x.astype(np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return x


def find_lag(ref, sig, max_lag=4800):
    """Lag (samples) such that sig[n + lag] ~ ref[n]. Uses envelope-free mono xcorr on the first 20 s."""
    a = ref.mean(1)[: SR * 20]
    b = sig.mean(1)[: SR * 20]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    c = correlate(b, a, mode="full", method="fft")
    mid = n - 1
    c = c[mid - max_lag: mid + max_lag + 1]
    lag = int(np.argmax(np.abs(c))) - max_lag
    return lag


def load_triplet(game, idx):
    """Return aligned (raw, fps, eq) float arrays of identical length, plus lags."""
    cfg = GAMES[game]
    name = f"test{idx}"
    x = {t: read(os.path.join(ROOT, cfg[t].format(n=name))) for t in TRACKS}
    lags = {"raw": 0}
    for t in ["fps", "eq"]:
        lags[t] = find_lag(x["raw"], x[t])
    # raw sample k  <->  track sample k + lag
    k0 = max(0, -min(lags.values()))
    aligned = {t: x[t][k0 + lags[t]:] for t in TRACKS}
    L = min(len(v) for v in aligned.values())
    aligned = {t: v[:L] for t, v in aligned.items()}
    return aligned, lags
