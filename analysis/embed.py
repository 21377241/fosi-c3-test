"""Per-event AST embeddings from the RAW track (level-normalised, so the classifier cannot use loudness).

Default: a 1 s window around the onset fed as 128 frames (position embeddings truncated in time, see
finetune_ast.load_ast), mean of the patch tokens of the last layer -> cache/embeddings_ast128.npz (n x 768).
--legacy: the earlier version, two windows each zero-padded to AST's 10 s input, pooler output
-> cache/embeddings.npz (n x 1536); still used by the overlap classifier in report_ad.py.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from common import GAMES, ROOT, read

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
WINDOWS = [(-0.05, 0.35), (-0.5, 0.5)]


def cut(x16, t, a, b):
    s, e = int((t + a) * 16000), int((t + b) * 16000)
    seg = np.zeros(e - s, np.float32)
    lo, hi = max(0, s), min(len(x16), e)
    seg[lo - s: hi - s] = x16[lo:hi]
    return seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9) * 0.05


def legacy(ev):
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    ast = ASTForAudioClassification.from_pretrained(MODEL).audio_spectrogram_transformer.cuda().eval().half()

    @torch.no_grad()
    def embed(segs):
        feats = fe(segs, sampling_rate=16000, return_tensors="pt")["input_values"].cuda().half()
        return ast(feats).pooler_output.float().cpu().numpy()

    X = np.zeros((len(ev), 768 * len(WINDOWS)), np.float32)
    for (game, clip), g in ev.groupby(["game", "clip_id"]):
        x = read(os.path.join(ROOT, GAMES[game]["raw"].format(n=f"test{clip}")))
        x16 = resample_poly(x.mean(1), 1, 3).astype(np.float32)
        for k, (a, b) in enumerate(WINDOWS):
            segs = [cut(x16, t, a, b) for t in g.t]
            out = np.concatenate([embed(segs[j:j + 64]) for j in range(0, len(segs), 64)])
            X[g.index, k * 768:(k + 1) * 768] = out
        print(f"{game} test{clip}: {len(g)} events", flush=True)
    return X


if __name__ == "__main__":
    ev = pd.read_csv(os.path.join(HERE, "results", "events.csv"))
    if "--legacy" in sys.argv:
        np.savez(os.path.join(HERE, "cache", "embeddings.npz"), X=legacy(ev))
    else:
        from finetune_ast import Audio, layer_embeddings
        X = layer_embeddings(Audio(ev), ev)[:, -1]
        np.savez(os.path.join(HERE, "cache", "embeddings_ast128.npz"), X=X)
        print(f"-> cache/embeddings_ast128.npz {X.shape}")
