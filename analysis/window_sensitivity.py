"""How much does the FPS / EQ gain of human-labelled events depend on the measurement window,
given only onset labels? Compares fixed windows with an automatically estimated event end."""
import numpy as np
import pandas as pd
from common import load_triplet, SR
from analyze import rms_db, HOP

F2S = SR / HOP
WINS = [0.05, 0.1, 0.2, 0.4]
CAP = 1.5


def auto_end(lv, k):
    """Event end: level back within 3 dB of the pre-onset background, or 20 dB below the peak."""
    bg = np.percentile(lv[max(0, k - int(0.6 * F2S)): k - int(0.05 * F2S)], 20)
    seg = lv[k: k + int(CAP * F2S)]
    ip = int(np.argmax(seg[: int(0.08 * F2S)]))
    after = seg[ip:]
    done = (after < bg + 3) | (after < seg[ip] - 20)
    return (ip + (int(np.argmax(done)) if done.any() else len(after))) / F2S


def main():
    ev = pd.read_csv("results/events_labeled_v2.csv", keep_default_na=False)
    h["cls"] = np.where(h.human_label == "f", "footstep",
                        np.where(h.human_label == "g", "gunshot", np.where(h.subtype == "换弹", "reload", "other")))

    rows = []
    for (game, clip), g in h.groupby(["game", "clip_id"]):
        a, _ = load_triplet(game, clip)
        e = {t: (a[t] ** 2).sum(1) for t in a}
        lv = rms_db(a["raw"])
        for r in g.itertuples():
            s, k = int(r.t * SR), int(r.t * F2S)
            dur = max(auto_end(lv, k), 0.03)
            row = dict(cls=r.cls, dur_ms=dur * 1000)
            for proc in ["fps", "eq"]:
                for w in WINS + ["auto"]:
                    n = int((dur if w == "auto" else w) * SR)
                    name = "auto" if w == "auto" else f"{int(w * 1000)}ms"
                    row[f"{proc}_{name}"] = 10 * np.log10(e[proc][s:s + n].mean() / e["raw"][s:s + n].mean())
            rows.append(row)

    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("counts:", d.cls.value_counts().to_dict())
    print("\nestimated event duration (ms), median / IQR:")
    print(d.groupby("cls").dur_ms.describe(percentiles=[.25, .5, .75])[["25%", "50%", "75%"]].round(0).to_string())
    for proc in ["fps", "eq"]:
        cols = [c for c in d.columns if c.startswith(proc + "_")]
        print(f"\n{proc.upper()} gain (dB, median) by window:")
        print(d.groupby("cls")[cols].median().round(1).rename(columns=lambda c: c[len(proc) + 1:]).to_string())


if __name__ == "__main__":
    main()
