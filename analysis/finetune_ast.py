"""Footstep classifier from AST beyond the frozen linear probe, evaluated with the same leave-clips-out
5-fold CV as train_labeler.py so the numbers are directly comparable.

  layers : frozen AST, mean-pooled hidden states of every layer -> logistic regression per layer group
  ft     : partial fine-tune (last N_UNFREEZE blocks + head) with augmentation (time shift,
           background mixing from the same clip, SpecAugment)

Inputs are 1 s windows of the RAW track around the onset, level-normalised as in embed.py, fed as
128 frames (position embeddings truncated in time, as in the AST paper for short clips).
Usage: python finetune_ast.py [layers] [ft] [--predict]
  --predict  fine-tune on all human labels and write cache/ft_probs.npz (p for every event)
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTConfig
from common import GAMES, ROOT, read
from train_labeler import read_human, CODE

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
WIN = (-0.3, 0.7)
FRAMES = 128
N_UNFREEZE, EPOCHS, BS = 4, 12, 16
LR_BODY, LR_HEAD = 3e-5, 1e-3
DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)
fe = ASTFeatureExtractor.from_pretrained(MODEL, max_length=FRAMES)


def load_ast():
    full = ASTForAudioClassification.from_pretrained(MODEL).audio_spectrogram_transformer
    cfg = ASTConfig.from_pretrained(MODEL, max_length=FRAMES)
    from transformers import ASTModel
    m = ASTModel(cfg)
    sd = full.state_dict()
    pos = sd["embeddings.position_embeddings"]
    f_out = (cfg.num_mel_bins - 16) // cfg.frequency_stride + 1
    t_full = (1024 - 16) // cfg.time_stride + 1
    t_out = (FRAMES - 16) // cfg.time_stride + 1
    grid = pos[:, 2:].reshape(1, f_out, t_full, -1)[:, :, :t_out].reshape(1, f_out * t_out, -1)
    sd["embeddings.position_embeddings"] = torch.cat([pos[:, :2], grid], 1)
    m.load_state_dict(sd)
    return m


def norm(seg):
    return seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9) * 0.05


class Audio:
    def __init__(self, ev):
        self.x = {}
        for game, clip in ev[["game", "clip_id"]].drop_duplicates().itertuples(index=False):
            x = read(os.path.join(ROOT, GAMES[game]["raw"].format(n=f"test{clip}")))
            self.x[(game, clip)] = resample_poly(x.mean(1), 1, 3).astype(np.float32)

    def cut(self, game, clip, t, shift=0.0):
        x = self.x[(game, clip)]
        s, e = int((t + WIN[0] + shift) * 16000), int((t + WIN[1] + shift) * 16000)
        seg = np.zeros(e - s, np.float32)
        lo, hi = max(0, s), min(len(x), e)
        seg[lo - s: hi - s] = x[lo:hi]
        return seg

    def augment(self, game, clip, t, rng):
        seg = self.cut(game, clip, t, rng.uniform(-0.05, 0.05))
        if rng.random() < 0.5:
            x = self.x[(game, clip)]
            n = len(seg)
            b = x[(o := rng.integers(0, max(1, len(x) - n))): o + n]
            if len(b) == n:
                seg = seg + norm(b) * np.sqrt(np.mean(seg ** 2) + 1e-12) / 0.05 * 10 ** (-rng.uniform(10, 25) / 20)
        return norm(seg)


def fbank(segs):
    return fe(list(segs), sampling_rate=16000, return_tensors="pt")["input_values"]


def spec_augment(x, rng):
    x = x.clone()
    for i in range(len(x)):
        f0 = rng.integers(0, 128 - 16); x[i, :, f0:f0 + rng.integers(0, 16)] = 0
        t0 = rng.integers(0, FRAMES - 12); x[i, t0:t0 + rng.integers(0, 12), :] = 0
    return x


# ------------------------------------------------------------------ frozen, every layer
@torch.no_grad()
def layer_embeddings(au, ev):
    m = load_ast().to(DEV).eval().half()
    out = []
    rows = list(ev[["game", "clip_id", "t"]].itertuples(index=False))
    for j in range(0, len(rows), 64):
        x = fbank([norm(au.cut(*r)) for r in rows[j:j + 64]]).to(DEV).half()
        hs = m(x, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, 2:].mean(1) for h in hs], 1).float().cpu().numpy())
    return np.concatenate(out)


def run_layers(au, ev, idx, y, groups):
    H = layer_embeddings(au, ev)
    np.savez(os.path.join(HERE, "cache", "ast_layers.npz"), H=H)
    mk = lambda: make_pipeline(StandardScaler(), PCA(0.95, svd_solver="full"),
                               LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))
    res = {}
    for name, sl in [("最后一层", [12]), ("第 4–8 层平均", list(range(4, 9))), ("第 1–12 层平均", list(range(1, 13)))] + \
                    [(f"第 {k} 层", [k]) for k in (2, 4, 6, 8, 10)]:
        X = H[:, sl].mean(1)
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(idx, y, groups):
            p[te] = mk().fit(X[idx][tr], y[tr]).predict_proba(X[idx][te])[:, 1]
        res[name] = p
        print(f"  冻结 AST {name:10s} AUC={roc_auc_score(y, p):.3f}", flush=True)
    return res


# ------------------------------------------------------------------ partial fine-tune
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = load_ast()
        for p in self.ast.parameters():
            p.requires_grad = False
        for blk in self.ast.encoder.layer[-N_UNFREEZE:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.ast.layernorm.parameters():
            p.requires_grad = True
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 1))

    def forward(self, x):
        return self.head(self.ast(x).pooler_output).squeeze(1)


def train(au, ev, tr_ids, y_tr, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Net().to(DEV)
    body = [p for p in net.ast.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": LR_BODY}, {"params": net.head.parameters(), "lr": LR_HEAD}],
                            weight_decay=0.01)
    pos_w = torch.tensor((1 - y_tr).sum() / max(1, y_tr.sum()), device=DEV, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.amp.GradScaler()
    meta = ev.loc[tr_ids, ["game", "clip_id", "t"]].values
    steps = EPOCHS * int(np.ceil(len(tr_ids) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BODY, LR_HEAD], total_steps=steps, pct_start=0.15)
    net.train()
    for _ in range(EPOCHS):
        order = rng.permutation(len(tr_ids))
        for j in range(0, len(order), BS):
            b = order[j:j + BS]
            x = spec_augment(fbank([au.augment(*meta[k], rng) for k in b]), rng).to(DEV)
            yt = torch.tensor(y_tr[b], dtype=torch.float32, device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(net(x), yt)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return net


@torch.no_grad()
def predict(net, au, ev, ids):
    net.eval()
    rows = ev.loc[ids, ["game", "clip_id", "t"]].values
    out = []
    for j in range(0, len(rows), 64):
        x = fbank([norm(au.cut(*r)) for r in rows[j:j + 64]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(net(x)).float().cpu().numpy())
    return np.concatenate(out)


def run_ft(au, ev, idx, y, groups):
    p = np.zeros(len(y))
    for k, (tr, te) in enumerate(GroupKFold(5).split(idx, y, groups)):
        net = train(au, ev, idx[tr], y[tr], seed=k)
        p[te] = predict(net, au, ev, idx[te])
        print(f"  第 {k + 1} 折 AUC={roc_auc_score(y[te], p[te]):.3f}", flush=True)
        del net; torch.cuda.empty_cache()
    print(f"  微调 AST（解冻最后 {N_UNFREEZE} 层）AUC={roc_auc_score(y, p):.3f}", flush=True)
    return p


if __name__ == "__main__":
    ev = pd.read_csv(os.path.join(HERE, "results", "events.csv"))
    ev["event_id"] = np.arange(len(ev))
    h = read_human()
    h = h[h.human_label.isin(CODE)]
    idx = h.event_id.values
    y = (h.human_label == "f").astype(int).values
    groups = (ev.game + ev.clip_id.astype(str)).values[idx]
    au = Audio(ev)
    print(f"人工标注 {len(y)} 条（脚步 {y.sum()}，非脚步 {len(y) - y.sum()}），留片段 5 折交叉验证")
    out = {"event_id": idx, "y": y}
    if "layers" in sys.argv:
        for k, v in run_layers(au, ev, idx, y, groups).items():
            out[f"layers_{k}"] = v
    if "ft" in sys.argv:
        out["ft"] = run_ft(au, ev, idx, y, groups)
    if len(out) > 2:
        pd.DataFrame(out).to_csv(os.path.join(HERE, "results", "ast_upgrade_cv.csv"), index=False, encoding="utf-8-sig")
    if "--predict" in sys.argv:
        nets = [train(au, ev, idx, y, seed=s) for s in range(3)]
        p = np.mean([predict(n, au, ev, ev.index.values) for n in nets], 0)
        np.savez(os.path.join(HERE, "cache", "ft_probs.npz"), p=p)
        print("-> cache/ft_probs.npz")
