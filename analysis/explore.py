import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, stft
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from common import GAMES, load_triplet, SR

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
fe = ASTFeatureExtractor.from_pretrained(MODEL)
model = ASTForAudioClassification.from_pretrained(MODEL).cuda().eval()
id2label = model.config.id2label


@torch.no_grad()
def top_labels(x):
    x16 = resample_poly(x.mean(1), 1, 3).astype(np.float32)
    w = 16000 * 10
    segs = [x16[s:s + w] for s in range(0, max(1, len(x16) - w // 2), w)]
    feats = fe(segs, sampling_rate=16000, return_tensors="pt")["input_values"].cuda()
    p = torch.sigmoid(model(feats).logits).max(0).values.cpu().numpy()
    idx = np.argsort(-p)[:6]
    return ", ".join(f"{id2label[i]}={p[i]:.2f}" for i in idx)


def spec_db(x):
    f, t, Z = stft(x.mean(1), SR, nperseg=2048, noverlap=1536)
    return f, t, 20 * np.log10(np.abs(Z) + 1e-7)


def plot(game, i, a):
    f, t, R = spec_db(a["raw"])
    _, _, F = spec_db(a["fps"])
    _, _, E = spec_db(a["eq"])
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    vmax = R.max()
    ax[0].pcolormesh(t, f, R, vmin=vmax - 90, vmax=vmax, shading="auto", cmap="magma")
    ax[0].set_title(f"{game} test{i} raw spectrogram")
    act = R > vmax - 70
    for k, (Y, name) in enumerate([(F, "FPS"), (E, "EQ")]):
        G = Y - R
        G = G - np.median(G[act])
        G[~act] = np.nan
        m = ax[k + 1].pcolormesh(t, f, G, vmin=-15, vmax=15, shading="auto", cmap="RdBu_r")
        ax[k + 1].set_title(f"{name} gain vs raw (dB, clip-median removed)")
        plt.colorbar(m, ax=ax[k + 1], pad=0.01)
    for k in range(3):
        ax[k].set_yscale("symlog", linthresh=500)
        ax[k].set_ylim(50, 24000)
    env = lambda x: 10 * np.log10(np.convolve(x.mean(1) ** 2, np.ones(2400) / 2400, "same")[::480] + 1e-10)
    tt = np.arange(len(a["raw"]))[::480] / SR
    for x, name in [(a["raw"], "raw"), (a["fps"], "fps"), (a["eq"], "eq")]:
        ax[3].plot(tt, env(x), lw=0.8, label=name)
    ax[3].legend()
    ax[3].set_ylabel("50ms RMS dBFS")
    ax[3].set_ylim(-90, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"explore_{game}_{i}.png"), dpi=80)
    plt.close()


if __name__ == "__main__":
    for game, cfg in GAMES.items():
        for i in range(1, cfg["n"] + 1):
            a, _ = load_triplet(game, i)
            print(f"{game} test{i}: {top_labels(a['raw'])}")
            plot(game, i, a)
