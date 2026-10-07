"""How much of each clip would be covered by labelled intervals, and how much stays blank?
Event extent = onset -> auto_end (capped at the next onset). Footstep sequences = model-labelled
footsteps closer than 1 s merged into one interval."""
import os
import numpy as np
import pandas as pd
from common import GAMES, ROOT, read
from analyze import rms_db
from window_sensitivity import auto_end, F2S

SEQ_GAP = 1.0


def union(iv):
    out = []
    for a, b in sorted(map(tuple, iv)):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def length(iv):
    return sum(b - a for a, b in iv)


def main():
    ev = pd.read_csv("results/events_labeled_v2.csv", keep_default_na=False).sort_values(["game", "clip_id", "t"])
    rows, blanks = [], []
    for (game, clip), d in ev.groupby(["game", "clip_id"]):
        lv = rms_db(read(os.path.join(ROOT, GAMES[game]["raw"].format(n=f"test{clip}"))))
        T = len(lv) / F2S
        t = d.t.values
        ends = [min(auto_end(lv, int(x * F2S)), (t[i + 1] - x) if i + 1 < len(t) else 9) for i, x in enumerate(t)]
        evt = union([(x, x + max(e, 0.03)) for x, e in zip(t, ends)])
        foot = d[d.label == "footstep"]
        seq = union([(x, x + max(e, 0.03) + SEQ_GAP) for x, e, l in zip(t, ends, d.label) if l == "footstep"])
        seq = [(a, b - SEQ_GAP) for a, b in seq]
        other = union([(x, x + max(e, 0.03)) for x, e, l in zip(t, ends, d.label) if l != "footstep"])
        covered = union(seq + other)
        gaps = [b - a for a, b in zip([0] + [b for _, b in covered], [a for a, _ in covered] + [T]) if b - a > 0]
        blanks += gaps
        rows.append(dict(game=game, clip=clip, dur=T, evt=length(evt), seq=length(seq), other=length(other),
                         covered=length(covered), n_seq=len(seq), n_foot=len(foot)))
    r = pd.DataFrame(rows)
    tot = r.dur.sum()
    print(f"clips {len(r)}, total {tot / 60:.1f} min")
    print(f"  events themselves (onset->end, all events):     {r.evt.sum() / tot:6.1%}")
    print(f"  footstep sequences (incl. gaps between steps):  {r.seq.sum() / tot:6.1%}  ({r.n_seq.sum()} sequences)")
    print(f"  non-footstep / ambiguous events:                {r.other.sum() / tot:6.1%}")
    print(f"  labelled intervals in total:                    {r.covered.sum() / tot:6.1%}")
    print(f"  blank (no detected event):                      {1 - r.covered.sum() / tot:6.1%}")
    b = np.array(blanks)
    print(f"\nblank stretches: {len(b)}, median {np.median(b):.2f} s, "
          f"<0.5 s: {(b < 0.5).sum()}, 0.5-2 s: {((b >= 0.5) & (b < 2)).sum()}, >=2 s: {(b >= 2).sum()} "
          f"(these hold {b[b >= 2].sum() / b.sum():.0%} of blank time)")
    print("\nper game blank share:")
    g = r.groupby("game")[["dur", "covered"]].sum()
    print((1 - g.covered / g.dur).map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main()
