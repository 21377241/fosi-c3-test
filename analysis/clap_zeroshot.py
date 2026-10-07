"""Zero-shot footstep detection with CLAP text prompts (no human labels used for training), compared
with the linear probe on the same human labels.

p_foot = softmax over all prompts of (logit_scale * cosine(audio, text)), summed over footstep prompts;
averaged over the two audio windows stored in cache/embeddings_clap.npz.
Output: results/clap_zeroshot.csv (p_zs per event)
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import ClapModel, ClapProcessor
from train_labeler import read_human, CODE

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "cache", "clap_model")
MODEL = LOCAL if os.path.exists(os.path.join(LOCAL, "model.safetensors")) else "laion/clap-htsat-unfused"

PROMPT_SETS = {
    "简单两类": (["footsteps"], ["a gunshot"]),
    "多描述": (["footsteps", "the sound of footsteps", "a person walking", "footsteps on a hard floor",
              "running footsteps", "footsteps in a video game"],
             ["a gunshot", "gunfire", "a gun being reloaded", "an explosion", "a grenade",
              "background noise", "a car engine", "a person speaking", "metal clanking"]),
}


@torch.no_grad()
def text_features(model, proc, texts):
    inp = proc(text=texts, return_tensors="pt", padding=True)
    return model.get_text_features(**{k: v.cuda() for k, v in inp.items()}).float().cpu().numpy()


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


if __name__ == "__main__":
    proc = ClapProcessor.from_pretrained(MODEL)
    model = ClapModel.from_pretrained(MODEL).cuda().eval()
    scale = float(model.logit_scale_a.exp()) if hasattr(model, "logit_scale_a") else 100.0
    A = np.load(os.path.join(HERE, "cache", "embeddings_clap.npz"))["X"]
    wins = [unit(A[:, :512]), unit(A[:, 512:])]

    h = read_human()
    h = h[h.human_label.isin(CODE)]
    idx, y = h.event_id.values, (h.human_label == "f").values
    r1 = set(pd.read_csv(os.path.join(HERE, "labeling", "review_r1.csv"), encoding="utf-8-sig").event_id)
    is_r1 = np.isin(idx, list(r1))
    out = {}
    print(f"logit_scale = {scale:.1f}")
    for name, (pos, neg) in PROMPT_SETS.items():
        T = unit(text_features(model, proc, pos + neg))
        ps = []
        for W in wins:
            z = scale * W @ T.T
            z = np.exp(z - z.max(1, keepdims=True))
            ps.append(z[:, :len(pos)].sum(1) / z.sum(1))
        p = np.mean(ps, 0)
        out[name] = p
        hi, lo = p[idx] >= .8, p[idx] <= .2
        print(f"零样本 {name:6s} AUC 全部人工={roc_auc_score(y, p[idx]):.3f}  第1轮={roc_auc_score(y[is_r1], p[idx][is_r1]):.3f}  "
              f"判为脚步 {hi.sum()} 条准确 {y[hi].mean():.0%}  判为非脚步 {lo.sum()} 条准确 {(~y[lo]).mean():.0%}")
        for k, W in enumerate(wins):
            sims = W[idx] @ T.T
            print(f"    窗口{k + 1}: 脚步事件与“{pos[0]}”平均相似度 {sims[y, 0].mean():.3f}，非脚步 {sims[~y, 0].mean():.3f}")
    cv = pd.read_csv(os.path.join(HERE, "results", "human_cv_probs.csv"), encoding="utf-8-sig").set_index("event_id").loc[idx]
    print(f"对照：线性探针 CLAP AUC={roc_auc_score(y, cv.cv_CLAP):.3f}，新AST+CLAP AUC={roc_auc_score(y, cv.cv_ens):.3f}"
          f"（第1轮 {roc_auc_score(y[is_r1], cv.cv_ens.values[is_r1]):.3f}）")
    pd.DataFrame({"event_id": np.arange(len(A)), **{f"p_zs_{k}": v for k, v in out.items()}}).to_csv(
        os.path.join(HERE, "results", "clap_zeroshot.csv"), index=False, encoding="utf-8-sig")
