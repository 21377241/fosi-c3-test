import numpy as np
from common import GAMES, load_triplet


def db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


for game, cfg in GAMES.items():
    for i in range(1, cfg["n"] + 1):
        a, lags = load_triplet(game, i)
        r = a["raw"]
        out = [f"{game:8s} test{i:<3d} len={len(r)/48000:6.2f}s lag_fps={lags['fps']:5d} lag_eq={lags['eq']:5d}"]
        for t in ["fps", "eq"]:
            y = a[t]
            g = db(y) - db(r)
            # residual after best scalar gain -> how "linear/static" the processing is
            s = np.sum(y * r) / np.sum(r * r)
            res = db(y - s * r) - db(y)
            clip = np.mean(np.abs(y) > 0.999) * 100
            out.append(f"{t}: gain={g:+6.2f}dB corr_res={res:+6.1f}dB clip={clip:.2f}%")
        print("  ".join(out))
