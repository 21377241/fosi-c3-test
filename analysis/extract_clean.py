"""Cut reviewed single events (purity p = clean, b = clean event over steady background) from the
aligned raw / FPS / EQ tracks into clean_events/, keeping original levels (no normalisation).
Segment = onset - PAD_PRE .. estimated end + PAD_POST, with short fades."""
import os
import glob
import shutil
import numpy as np
import pandas as pd
from scipy.io import wavfile
from common import load_triplet, SR, TRACKS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "clean_events")
PAD_PRE, PAD_POST, FADE = 0.02, 0.05, 0.005
CLASS = {"f": "footstep", "g": "gunshot", "o": "other", "x": "unsure"}
KEEP = {"p", "b"}


def read_purity():
    """Final label + purity per event: round-1 review plus later rounds labelled with purity."""
    lab = os.path.join(HERE, "labeling")
    parts = [pd.read_csv(os.path.join(lab, "review_r1.csv"), encoding="utf-8-sig", keep_default_na=False)]
    for fn in sorted(glob.glob(os.path.join(lab, "human_labels_r*.csv"))):
        d = pd.read_csv(fn, encoding="utf-8-sig", keep_default_na=False)
        if "purity" in d:
            parts.append(d.rename(columns={"human_label": "final_label", "note": "final_note"}))
    cols = ["event_id", "game", "clip_id", "t", "final_label", "final_note", "purity", "est_dur_ms"]
    return pd.concat([p[cols] for p in parts]).drop_duplicates("event_id", keep="last")


def main():
    rv = read_purity()
    ev = pd.read_csv(os.path.join(HERE, "results", "events_labeled_v2.csv"), keep_default_na=False)
    shutil.rmtree(OUT, ignore_errors=True)
    rv = rv[rv.purity.isin(KEEP)].merge(ev[["event_id", "raw_peak", "raw_prom", "fps_gain", "eq_gain"]], on="event_id")
    meta = []
    for (game, clip), g in rv.groupby(["game", "clip_id"]):
        a, _ = load_triplet(game, clip)
        for r in g.itertuples():
            s = max(0, int((r.t - PAD_PRE) * SR))
            e = min(len(a["raw"]), int((r.t + r.est_dur_ms / 1000 + PAD_POST) * SR))
            n, nf = e - s, int(FADE * SR)
            env = np.ones(n, np.float32)
            env[:nf], env[-nf:] = np.linspace(0, 1, nf), np.linspace(1, 0, nf)
            cls = CLASS[r.final_label]
            d = os.path.join(OUT, cls)
            os.makedirs(d, exist_ok=True)
            for tr in TRACKS:
                wavfile.write(os.path.join(d, f"e{r.event_id:04d}_{tr}.wav"), SR, (a[tr][s:e] * env[:, None]).astype(np.float32))
            meta.append(dict(event_id=r.event_id, cls=cls, note=r.final_note, purity=r.purity, game=game, clip_id=clip,
                             t=r.t, seg_start=s / SR, seg_dur_ms=round(n / SR * 1000), est_dur_ms=r.est_dur_ms,
                             raw_peak=round(r.raw_peak, 1), raw_prom=round(r.raw_prom, 1),
                             fps_gain=round(r.fps_gain, 2), eq_gain=round(r.eq_gain, 2)))
    m = pd.DataFrame(meta).sort_values(["cls", "purity", "event_id"])
    m.to_csv(os.path.join(OUT, "clean_events.csv"), index=False, encoding="utf-8-sig")
    print(f"{len(m)} events x {len(TRACKS)} tracks -> clean_events/")
    print(pd.crosstab(m.cls, m.purity.map({"p": "clean", "b": "with background"})).to_string())


if __name__ == "__main__":
    main()
