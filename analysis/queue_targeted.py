"""Targeted labelling queues for ambiguous events (0.2 < p_ens < 0.8, never labelled), stratified by peak level.

r3: weighted toward the loud range where footsteps and non-footsteps overlap, plus a spot check of loud
    confident footsteps (old labelling model).
r4: stratified random sample of the ambiguous events of the current model (oversampling the loud stratum),
    so the accuracy of a 0.5 split can be estimated for the whole ambiguous zone with stratum weights.
Output: labeling/queue_{tag}.csv. Then: prepare_labeling.build_page -> build_review.py --label {tag}."""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "labeling")
CONFIG = {
    "r3": dict(bins=[-90, -45, -35, -25, 0], quota=[8, 14, 16, 12], per_clip=4, n_loud_check=10, seed=3),
    "r4": dict(bins=[-90, -35, 0], quota=[20, 35], per_clip=4, n_loud_check=0, seed=4),
}
LOUD = -30


def sample(ev, bins, quota, per_clip, n_loud_check, seed):
    rng = np.random.RandomState(seed)
    un = ev[ev.human_label == ""].copy()
    amb = un[(un.p_ens > 0.2) & (un.p_ens < 0.8)].copy()
    amb["lv"] = pd.cut(amb.raw_peak, bins, labels=False)
    picked, n_clip = [], {}
    for b, q in enumerate(quota):
        n = 0
        for r in amb[amb.lv == b].sample(frac=1, random_state=rng).itertuples():
            k = (r.game, r.clip_id)
            if n_clip.get(k, 0) >= per_clip:
                continue
            picked.append(r.event_id); n_clip[k] = n_clip.get(k, 0) + 1; n += 1
            if n == q:
                break
    out = [amb[amb.event_id.isin(picked)].assign(purpose="ambiguous")]
    if n_loud_check:
        out.append(un[(un.p_ens >= 0.8) & (un.raw_peak > LOUD)].sample(n_loud_check, random_state=rng).assign(purpose="loud_check"))
    out = pd.concat(out).sample(frac=1, random_state=rng)
    out["stratum_size"] = out.lv.map(amb.lv.value_counts()) if "lv" in out else np.nan
    cols = ["event_id", "game", "clip_id", "t", "raw_peak", "p_ens", "models_disagree", "purpose", "lv", "stratum_size"]
    cols += [c for c in ["p_AST", "p_CLAP"] if c in out]
    return out[cols]


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "r4"
    ev = pd.read_csv(os.path.join(HERE, "results", "events_labeled_v2.csv"), keep_default_na=False)
    sel = sample(ev, **CONFIG[tag])
    path = os.path.join(LAB, f"queue_{tag}.csv")
    if os.path.exists(path) and "--force" not in sys.argv:
        sys.exit(f"{path} exists; pass --force to overwrite")
    sel.to_csv(path, index=False)
    print(f"{tag}: {len(sel)} events from {sel.groupby(['game', 'clip_id']).ngroups} clips")
    print(sel.groupby(["purpose", "lv"]).size().to_string())
    print("models disagree:", int(sel.models_disagree.sum()))
