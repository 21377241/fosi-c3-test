"""FPS / EQ gain of reviewed events split by purity grade (review_r1.csv), and the selectivity
effect (footstep - non-footstep, level-bin + clip fixed effects) on progressively cleaner subsets."""
import numpy as np
import pandas as pd
from report import fe_regression

NAMES = {"f": "脚步", "g": "枪声", "o": "其他", "x": "不确定"}
PURE = {"p": "纯净", "b": "有持续背景", "l": "有重叠"}

ev = pd.read_csv("results/events_labeled_v2.csv", keep_default_na=False)
rv = pd.read_csv("labeling/review_r1.csv", encoding="utf-8-sig", keep_default_na=False)
d = rv[["event_id", "final_label", "purity"]].merge(ev, on="event_id")
d = d[d.final_label.isin(["f", "g", "o"])].copy()
d["label"] = np.where(d.final_label == "f", "footstep", "non-footstep")

pd.set_option("display.width", 200)
print("median by class x purity:")
print(d.groupby([d.final_label.map(NAMES), d.purity.map(PURE)]).agg(
    n=("fps_gain", "size"), raw_peak=("raw_peak", "median"), fps_gain=("fps_gain", "median"),
    fps_contrast=("fps_contrast", "median"), fps_bg_gain=("fps_bg_gain", "median"),
    eq_gain=("eq_gain", "median")).round(1).to_string())

print("\nselectivity effect (footstep - non-footstep), level-bin + clip fixed effects:")
for name, sub in [("all reviewed", d), ("pure + background", d[d.purity != "l"]), ("pure only", d[d.purity == "p"])]:
    nf, nn = (sub.label == "footstep").sum(), (sub.label != "footstep").sum()
    print(f"  {name:18s} n={len(sub):3d} (footstep {nf}, non {nn}):  "
          + "  ".join(f"{c}={fe_regression(sub, c):+.2f}" for c in ["fps_gain", "fps_contrast", "eq_gain"]))
