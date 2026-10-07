"""Fine-tune PANNs (Cnn14_DecisionLevelMax) into a small game-sound classifier and use it as the detector of
sub-experiment E, with test clips that are never seen in training or model selection.

Classes: footstep / gunshot / other non-footstep (human labels f / g / o) and background (windows inside 1 s
stretches with no detected event). Inputs are 0.4 s windows ([t-0.05, t+0.35] s, as the zero-shot detector) of
the RAW track only, 32 kHz mono. Augmentation: time shift, random gain and random EQ, so the model can rely
neither on loudness nor on one particular timbre. Blocks 1-4 stay frozen; blocks 5-6, fc1 and a 4-way head
initialised from the AudioSet output layer (HEAD_INIT) are trained; early stopping on a validation split taken
from the training clips, with the untrained (zero-shot) state as epoch 0.
v1 (randomly initialised head, 1 s window) did worse than zero-shot; its outputs are kept as *_v1.csv.

Splits (fixed before looking at any result):
  holdout : 4 VALORANT + 4 PUBG clips drawn with seed 2026 are the test set
  cv5     : 5-fold leave-clips-out, every clip is a test clip exactly once
Test: the same model scores raw / fps / eq of the test clips (processed tracks scaled to the raw clip RMS).
Outputs: results/panns_ft_predictions.csv, results/panns_ft_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from common import TRACKS, load_triplet
from train_labeler import read_human
from detectability import load_model, cluster_ci, LEVELS, GROUPS

HERE = os.path.dirname(os.path.abspath(__file__))
SR32 = 32000
WIN = (-0.05, 0.35)
CLASSES = ["footstep", "gunshot", "other", "background"]
# Head initialised from the AudioSet output layer (mean weight of the mapped classes), so training starts from
# the zero-shot closed-set detector; epoch 0 also competes in early stopping.
HEAD_INIT = {"footstep": ["footstep"], "gunshot": ["gunshot"], "other": ["explosion", "impact", "mechanical"],
             "background": ["environment"]}
CODE = {"f": 0, "g": 1, "o": 2}
BG_GUARD, BG_TAIL = 0.3, 1.0
EPOCHS, PATIENCE, BS = 40, 8, 32
LR_BODY, LR_HEAD = 1e-4, 1e-3
SHIFT, GAIN_DB, EQ_DB = 0.05, 10.0, 4.0
N_BOOT = 2000
DEV = "cuda"


def load_audio(clips):
    audio = {}
    for game, clip in clips:
        a, _ = load_triplet(game, clip)
        mono = {t: a[t].mean(1) for t in TRACKS}
        ref = np.sqrt(np.mean(mono["raw"] ** 2))
        for t in TRACKS:
            x = mono[t] * ref / (np.sqrt(np.mean(mono[t] ** 2)) + 1e-12)
            audio[(f"{game}{clip}", t)] = resample_poly(x, 2, 3).astype(np.float32)
    return audio


def items_table(ev, audio):
    """Human-labelled events plus background windows: columns key, start (s), y, kind, event_id, raw_peak."""
    h = read_human()
    h = h[h.human_label.isin(CODE)]
    e = ev.loc[h.event_id.values]
    rows = [dict(key=f"{r.game}{r.clip_id}", start=r.t + WIN[0], y=CODE[lab], kind="event", event_id=r.event_id,
                 raw_peak=r.raw_peak) for r, lab in zip(e.itertuples(), h.human_label)]
    for (game, clip), g in ev.groupby(["game", "clip_id"]):
        n = int(len(audio[(f"{game}{clip}", "raw")]) / SR32 * 100)
        busy = np.zeros(n, bool)
        for t in g.t:
            busy[max(0, int((t - BG_GUARD) * 100)):int((t + BG_TAIL) * 100)] = True
        s = 0
        while s + 100 <= n:
            if not busy[s:s + 100].any():
                rows.append(dict(key=f"{game}{clip}", start=s / 100, y=3, kind="background", event_id=-1, raw_peak=np.nan))
                s += 100
            else:
                s += 10
    return pd.DataFrame(rows)


def cut(x, start, shift=0.0):
    s = int((start + shift) * SR32)
    n = int((WIN[1] - WIN[0]) * SR32)
    seg = np.zeros(n, np.float32)
    lo, hi = max(0, s), min(len(x), s + n)
    if hi > lo:
        seg[lo - s:hi - s] = x[lo:hi]
    return seg


def random_eq(seg, rng):
    """Smooth random gain curve (±EQ_DB at 10 log-spaced points) applied in the frequency domain."""
    X = np.fft.rfft(seg)
    f = np.fft.rfftfreq(len(seg), 1 / SR32)
    pts = np.geomspace(50, 16000, 10)
    g = np.interp(np.log(np.maximum(f, 20)), np.log(pts), rng.uniform(-EQ_DB, EQ_DB, len(pts)))
    return np.fft.irfft(X * 10 ** (g / 20), len(seg)).astype(np.float32)


def batch(items, audio, track, rng=None):
    segs = []
    for r in items.itertuples():
        x = audio[(r.key, track)]
        if rng is None:
            segs.append(cut(x, r.start))
        else:
            seg = cut(x, r.start, rng.uniform(-SHIFT, SHIFT))
            seg = random_eq(seg, rng) * 10 ** (rng.uniform(-GAIN_DB, GAIN_DB) / 20)
            segs.append(seg)
    return torch.from_numpy(np.stack(segs)).to(DEV)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        from panns_inference.config import labels
        self.m, _, _ = load_model()
        self.head = nn.Linear(2048, len(CLASSES))
        W, b = self.m.fc_audioset.weight.data, self.m.fc_audioset.bias.data
        with torch.no_grad():
            for i, c in enumerate(CLASSES):
                ids = [labels.index(n) for gname in HEAD_INIT[c] for n in GROUPS[gname]]
                self.head.weight[i] = W[ids].mean(0)
                self.head.bias[i] = b[ids].mean()
        self.frozen = [self.m.bn0, self.m.conv_block1, self.m.conv_block2, self.m.conv_block3, self.m.conv_block4]
        for mod in self.frozen:
            for p in mod.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for mod in self.frozen:
            mod.eval()
        return self

    def forward(self, wav):
        m = self.m
        x = m.logmel_extractor(m.spectrogram_extractor(wav))
        x = m.bn0(x.transpose(1, 3)).transpose(1, 3)
        for blk, pool in [(m.conv_block1, 2), (m.conv_block2, 2), (m.conv_block3, 2), (m.conv_block4, 2),
                          (m.conv_block5, 2), (m.conv_block6, 1)]:
            x = F.dropout(blk(x, pool_size=(pool, pool), pool_type="avg"), p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x = F.max_pool1d(x, 3, 1, 1) + F.avg_pool1d(x, 3, 1, 1)
        x = F.dropout(x, p=0.5, training=self.training).transpose(1, 2)
        x = F.relu_(self.m.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.head(x.max(1)[0])


def predict(net, items, audio, track):
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(items), 64):
            out.append(torch.softmax(net(batch(items.iloc[b:b + 64], audio, track)), 1).cpu().numpy())
    return np.concatenate(out)


def train(items, audio, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(items, groups=items.key))
    tr, va = items.iloc[tr_idx], items.iloc[va_idx]
    cnt = np.bincount(tr.y, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    net = Net().to(DEV)
    body = [p for n, p in net.m.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": LR_BODY}, {"params": net.head.parameters(), "lr": LR_HEAD}],
                            weight_decay=1e-2)
    yv = torch.tensor(va.y.values, device=DEV)
    pv = torch.from_numpy(predict(net, va, audio, "raw")).to(DEV)
    best = F.nll_loss(torch.log(pv + 1e-9), yv, weight=w).item()
    best_state, bad, best_ep = {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, 0
    for ep in range(EPOCHS):
        net.train()
        order = rng.permutation(len(tr))
        for b in range(0, len(order), BS):
            part = tr.iloc[order[b:b + BS]]
            loss = F.cross_entropy(net(batch(part, audio, "raw", rng)), torch.tensor(part.y.values, device=DEV), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = torch.from_numpy(predict(net, va, audio, "raw")).to(DEV)
        vloss = F.nll_loss(torch.log(pv + 1e-9), yv, weight=w).item()
        if vloss < best - 1e-4:
            best, bad, best_ep = vloss, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net, best_ep


def metrics(df):
    ev = df[df.kind == "event"]
    bg = df[df.kind == "background"]
    fo = ev.y == 0
    pred = ev.pred == 0
    rec = pred[fo].mean()
    prec = (pred & fo).sum() / max(pred.sum(), 1)
    out = dict(n_foot=int(fo.sum()), n_non=int((~fo).sum()), n_bg=len(bg), auc=roc_auc_score(fo, ev.p_foot),
               recall=rec, precision=prec, f1=2 * rec * prec / max(rec + prec, 1e-12),
               fpr_events=pred[~fo].mean(), fpr_bg=(bg.pred == 0).mean(), acc4=(df.pred == df.y).mean())
    for lo, hi, lab in LEVELS:
        m = fo & (ev.raw_peak > lo) & (ev.raw_peak <= hi)
        out[f"recall {lab}"] = pred[m].mean()
    return out


def main():
    ev = pd.read_csv(os.path.join(HERE, "results", "events.csv"))
    ev["event_id"] = np.arange(len(ev))
    clips = list(ev[["game", "clip_id"]].drop_duplicates().itertuples(index=False, name=None))
    audio = load_audio(clips)
    items = items_table(ev, audio)
    print("样本:", items.groupby("y").size().rename(dict(enumerate(CLASSES))).to_dict())

    rng = np.random.default_rng(2026)
    test_keys = set()
    for game, k in [("valorant", 4), ("pubg", 4)]:
        keys = sorted({f"{g}{c}" for g, c in clips if g == game})
        test_keys |= set(rng.choice(keys, k, replace=False))
    splits = [("holdout", 0, items.key.isin(test_keys).values)]
    for f, (_, te) in enumerate(GroupKFold(5).split(items, groups=items.key)):
        m = np.zeros(len(items), bool)
        m[te] = True
        splits.append(("cv5", f, m))
    print("留出测试片段:", sorted(test_keys))

    preds = []
    for split, fold, te in splits:
        net, n_ep = train(items[~te].reset_index(drop=True), audio, seed=fold)
        test = items[te].reset_index(drop=True)
        for t in TRACKS:
            p = predict(net, test, audio, t)
            preds.append(test.assign(split=split, fold=fold, track=t, p_foot=p[:, 0], pred=p.argmax(1),
                                     **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        print(f"{split} fold {fold}: 训练 {int((~te).sum())} 测试 {int(te.sum())}，最佳轮次 {n_ep}（0 = 未微调）", flush=True)
        del net
        torch.cuda.empty_cache()
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(os.path.join(HERE, "results", "panns_ft_predictions.csv"), index=False, encoding="utf-8-sig")

    boot = np.random.default_rng(0)
    rows = []
    for split in ["holdout", "cv5"]:
        S = P[P.split == split]
        print(f"\n== {split}")
        base = None
        for t in TRACKS:
            m = metrics(S[S.track == t])
            row = dict(split=split, track=t, **m)
            if t == "raw":
                base = S[S.track == "raw"].reset_index(drop=True)
            else:
                cur = S[S.track == t].reset_index(drop=True)
                fo = (base.kind == "event") & (base.y == 0)
                nf = (base.kind == "event") & (base.y != 0)
                hb, hc = (base.pred == 0).values, (cur.pred == 0).values
                g = base.key.values
                row["d_recall_ci"] = cluster_ci(lambda i: hc[i][fo.values[i]].mean() - hb[i][fo.values[i]].mean(), g, boot)
                row["d_fpr_ci"] = cluster_ci(lambda i: hc[i][nf.values[i]].mean() - hb[i][nf.values[i]].mean(), g, boot)
                bgm = (base.kind == "background").values
                row["d_fpr_bg_ci"] = cluster_ci(lambda i: hc[i][bgm[i]].mean() - hb[i][bgm[i]].mean()
                                                if bgm[i].any() else np.nan, g, boot)
                dl = np.log(cur.p_foot.values + 1e-6) - np.log(base.p_foot.values + 1e-6)
                row["dlogp_foot"] = np.median(dl[fo.values])
                row["dlogp_foot_ci"] = cluster_ci(lambda i: np.median(dl[i][fo.values[i]]), g, boot)
            rows.append(row)
            msg = (f"  {t:4s} 四类准确率={m['acc4']:.1%}  脚步: AUC={m['auc']:.3f} 检出率={m['recall']:.1%} "
                   f"精确率={m['precision']:.1%} F1={m['f1']:.3f}  非脚步误报={m['fpr_events']:.1%}  "
                   f"背景误报={m['fpr_bg']:.1%}  按电平检出率 " +
                   " / ".join(f"{m[f'recall {lab}']:.0%}" for *_, lab in LEVELS))
            print(msg)
            if t != "raw":
                c = row["d_recall_ci"]
                print(f"       相对原始: 检出率 {(m['recall'] - rows[-1 - TRACKS.index(t)]['recall']) * 100:+.1f} "
                      f"[{c[0] * 100:+.1f}, {c[1] * 100:+.1f}] 个百分点；非脚步误报 "
                      f"{(m['fpr_events'] - rows[-1 - TRACKS.index(t)]['fpr_events']) * 100:+.1f} "
                      f"[{row['d_fpr_ci'][0] * 100:+.1f}, {row['d_fpr_ci'][1] * 100:+.1f}]；背景误报 "
                      f"{(m['fpr_bg'] - rows[-1 - TRACKS.index(t)]['fpr_bg']) * 100:+.1f} "
                      f"[{row['d_fpr_bg_ci'][0] * 100:+.1f}, {row['d_fpr_bg_ci'][1] * 100:+.1f}]；"
                      f"脚步概率 Δlog 中位 {row['dlogp_foot']:+.2f} "
                      f"[{row['dlogp_foot_ci'][0]:+.2f}, {row['dlogp_foot_ci'][1]:+.2f}]")
    R = pd.DataFrame(rows)
    for c in [c for c in R.columns if c.endswith("_ci")]:
        R[c] = R[c].apply(lambda v: f"{v[0]:.4f},{v[1]:.4f}" if isinstance(v, np.ndarray) else "")
    R.to_csv(os.path.join(HERE, "results", "panns_ft_summary.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
