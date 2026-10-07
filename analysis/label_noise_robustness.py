"""How robust is the selectivity conclusion to errors in the human labels?

Uses only human-labelled events (f vs g/o). Effect = FPS gain difference footstep - non-footstep,
from the regression with level-bin and clip fixed effects (report.fe_regression).
  random:     flip a random fraction of labels, repeat
  worst case: flip the labels that hurt the conclusion most, until the effect reaches 0
"""
import numpy as np
import pandas as pd
from report import fe_regression

rng = np.random.default_rng(0)
ev = pd.read_csv("results/events_labeled_v2.csv", keep_default_na=False)
h = ev[ev.human_label.isin(["f", "g", "o"])].copy()
h["label"] = np.where(h.human_label == "f", "footstep", "non-footstep")
flip = {"footstep": "non-footstep", "non-footstep": "footstep"}

for col in ["fps_gain", "fps_contrast", "eq_gain"]:
    base = fe_regression(h, col)
    print(f"\n{col}: effect with human labels = {base:+.2f} dB  (n={len(h)}, footsteps={int((h.label == 'footstep').sum())})")
    for frac in [0.05, 0.10, 0.20, 0.30]:
        vals = []
        for _ in range(300):
            d = h.copy()
            idx = rng.choice(len(d), int(round(frac * len(d))), replace=False)
            d.iloc[idx, d.columns.get_loc("label")] = d.label.iloc[idx].map(flip).values
            vals.append(fe_regression(d, col))
        vals = np.array(vals)
        print(f"  random {frac:4.0%} flipped: median {np.median(vals):+.2f}, 5th pct {np.percentile(vals, 5):+.2f}, "
              f"effect <= 0 in {np.mean(vals <= 0):.0%} of runs")
    if col.startswith("fps"):
        d = h.copy()
        order = list(d[d.label == "footstep"].sort_values(col, ascending=False).index) + \
                list(d[d.label == "non-footstep"].sort_values(col).index)
        # greedy: alternately relabel the most favourable footstep / non-footstep
        fs = list(d[d.label == "footstep"].sort_values(col, ascending=False).index)
        ns = list(d[d.label == "non-footstep"].sort_values(col).index)
        n = 0
        while fe_regression(d, col) > 0 and (fs or ns):
            pick = fs.pop(0) if (n % 2 == 0 and fs) or not ns else ns.pop(0)
            d.loc[pick, "label"] = flip[d.loc[pick, "label"]]
            n += 1
        print(f"  worst case: {n} targeted flips ({n / len(h):.0%} of labels) needed to bring the effect to 0")
