"""Frame-level sound event probabilities with AST (AudioSet) on raw / fps / eq tracks.

Each processed track is RMS-matched to raw before inference so that the classifier
sees content changes rather than overall loudness changes.
Output: cache/probs_{game}_{idx}.npz with probs[track] of shape (n_windows, n_classes_of_interest).
"""
import os
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from common import GAMES, TRACKS, load_triplet

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
WIN = 1.0
HOP = 0.1
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)

fe = ASTFeatureExtractor.from_pretrained(MODEL)
model = ASTForAudioClassification.from_pretrained(MODEL).cuda().eval().half()
CLASSES = [model.config.id2label[i] for i in range(model.config.num_labels)]
cls_ids = list(range(len(CLASSES)))


@torch.no_grad()
def infer(x16):
    w, h = int(WIN * 16000), int(HOP * 16000)
    starts = list(range(0, max(1, len(x16) - w + 1), h))
    out = []
    for b in range(0, len(starts), 64):
        batch = [x16[s:s + w] for s in starts[b:b + 64]]
        feats = fe(batch, sampling_rate=16000, return_tensors="pt")["input_values"].cuda().half()
        logits = model(feats).logits.float()
        out.append(torch.sigmoid(logits[:, cls_ids]).cpu().numpy().astype(np.float16))
    centers = np.array(starts) / 16000 + WIN / 2
    return np.concatenate(out), centers


if __name__ == "__main__":
    for game, cfg in GAMES.items():
        for i in range(1, cfg["n"] + 1):
            path = os.path.join(CACHE, f"probs_{game}_{i}.npz")
            if os.path.exists(path):
                continue
            a, _ = load_triplet(game, i)
            ref_rms = np.sqrt(np.mean(a["raw"] ** 2))
            res = {}
            for t in TRACKS:
                m = a[t].mean(1)
                m = m * ref_rms / (np.sqrt(np.mean(a[t] ** 2)) + 1e-12)
                x16 = resample_poly(m, 1, 3).astype(np.float32)
                p, centers = infer(x16)
                res[t] = p
            np.savez(path, centers=centers, classes=np.array(CLASSES), **res)
            print(f"{game} test{i}: windows={len(centers)}", flush=True)
