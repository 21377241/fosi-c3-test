"""Per-event CLAP audio embeddings from the RAW track (level-normalised, 48 kHz native).

Two windows per event (tight + 1 s context), like embed.py.
Output: cache/embeddings_clap.npz  (X: n_events x 1024, aligned with results/events.csv rows)
"""
import os
import numpy as np
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor
from common import GAMES, ROOT, read, SR

HERE = os.path.dirname(os.path.abspath(__file__))
# laion/clap-htsat-unfused only ships pytorch_model.bin, which transformers refuses to load with torch<2.6;
# cache/clap_model holds the same weights converted to safetensors.
LOCAL = os.path.join(HERE, "cache", "clap_model")
MODEL = LOCAL if os.path.exists(os.path.join(LOCAL, "model.safetensors")) else "laion/clap-htsat-unfused"
WINDOWS = [(-0.05, 0.35), (-0.5, 0.5)]

proc = ClapProcessor.from_pretrained(MODEL)
model = ClapModel.from_pretrained(MODEL).cuda().eval()


@torch.no_grad()
def embed(segs):
    inp = proc(audios=segs, sampling_rate=SR, return_tensors="pt")
    feats = model.get_audio_features(input_features=inp["input_features"].cuda())
    return feats.float().cpu().numpy()


def cut(x, t, a, b):
    s, e = int((t + a) * SR), int((t + b) * SR)
    seg = np.zeros(e - s, np.float32)
    lo, hi = max(0, s), min(len(x), e)
    seg[lo - s: hi - s] = x[lo:hi]
    return seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9) * 0.05


if __name__ == "__main__":
    ev = pd.read_csv(os.path.join(HERE, "results", "events.csv"))
    dim = None
    X = None
    for (game, clip), g in ev.groupby(["game", "clip_id"]):
        x = read(os.path.join(ROOT, GAMES[game]["raw"].format(n=f"test{clip}"))).mean(1)
        for k, (a, b) in enumerate(WINDOWS):
            segs = [cut(x, t, a, b) for t in g.t]
            out = np.concatenate([embed(segs[j:j + 32]) for j in range(0, len(segs), 32)])
            if X is None:
                dim = out.shape[1]
                X = np.zeros((len(ev), dim * len(WINDOWS)), np.float32)
            X[g.index, k * dim:(k + 1) * dim] = out
        print(f"{game} test{clip}: {len(g)} events", flush=True)
    np.savez(os.path.join(HERE, "cache", "embeddings_clap.npz"), X=X)
