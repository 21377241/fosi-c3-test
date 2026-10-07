"""Footstep recognition accuracy before annotation (zero-shot) and after each labelling round.

Evaluation set: the round-1 stratified random sample (clear labels only), the only unbiased sample.
For each stage the labeller is trained on the cumulative human labels of rounds 1..k and scored with
leave-clips-out 5-fold CV (same procedure as train_labeler.py), so no event is scored by a model that saw it.
Output: results/label_curve.csv (one row per stage), results/label_curve_probs.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from train_labeler import read_human, embedding_model, CODE, FEATURE_SETS, CACHE, LAB, P_FOOT, P_NON

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROUNDS = ["human_labels.csv", "human_labels_r2.csv", "human_labels_r3.csv", "human_labels_r4.csv"]


def metrics(y, p, thr=0.5):
    pred = p >= thr
    conf = (p >= P_FOOT) | (p <= P_NON)
    return dict(auc=roc_auc_score(y, p), acc=(pred == y).mean(),
                rec_f=pred[y == 1].mean(), rec_n=(~pred[y == 0]).mean(),
                conf_share=conf.mean(), conf_acc=(pred[conf] == y[conf]).mean() if conf.any() else np.nan)


def main():
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    ev["event_id"] = np.arange(len(ev))
    groups_all = (ev.game + ev.clip_id.astype(str)).values
    h = read_human()
    h = h[h.human_label.isin(CODE)].set_index("event_id")
    rnd = {}
    for k, fn in enumerate(ROUNDS, 1):
        for e in pd.read_csv(os.path.join(LAB, fn), encoding="utf-8-sig").event_id:
            rnd.setdefault(int(e), k)
    h["round"] = [rnd.get(e, 0) for e in h.index]
    r1 = h[h["round"] == 1].index.values
    y1 = (h.loc[r1, "human_label"] == "f").values
    X = {n: np.load(os.path.join(CACHE, f))["X"] for n, f, _ in FEATURE_SETS}

    rows, probs = [], {"event_id": r1, "y": y1.astype(int)}
    zs = pd.read_csv(os.path.join(RES, "clap_zeroshot.csv")).set_index("event_id")
    base = {"AudioSet 分类器（零样本）": ev.set_index("event_id").loc[r1, "p_foot_broad"].values,
            "CLAP 文本提示（零样本）": zs.loc[r1, "p_zs_多描述"].values}
    for name, p in base.items():
        # zero-shot scores are not calibrated: report accuracy at the best threshold on this very sample
        # (optimistic for the baselines), plus the natural 0.5 threshold
        thr = max(np.unique(p), key=lambda t: ((p >= t) == y1).mean())
        m = metrics(y1, p)
        m_best = metrics(y1, p, thr)
        rows.append(dict(stage=name, n_train=0, **m, acc_best=m_best["acc"], thr_best=thr, unlabeled_conf=np.nan))
        probs[name] = p

    for k in range(1, 5):
        lab = h[h["round"].between(1, k)]
        idx = lab.index.values
        y = (lab.human_label == "f").values.astype(int)
        g = groups_all[idx]
        oof = {}
        for n in X:
            p = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(idx, y, g):
                p[te] = embedding_model().fit(X[n][idx][tr], y[tr]).predict_proba(X[n][idx][te])[:, 1]
            oof[n] = p
        p_ens = pd.Series(np.mean(list(oof.values()), 0), index=idx).loc[r1].values
        full = np.mean([embedding_model().fit(X[n][idx], y).predict_proba(X[n])[:, 1] for n in X], 0)
        unl = np.setdiff1d(ev.event_id.values, h.index.values)
        unl_conf = ((full[unl] >= P_FOOT) | (full[unl] <= P_NON)).mean()
        m = metrics(y1, p_ens)
        rows.append(dict(stage=f"第 1–{k} 轮标注后" if k > 1 else "第 1 轮标注后", n_train=len(idx), **m,
                         acc_best=np.nan, thr_best=0.5, unlabeled_conf=unl_conf))
        probs[f"round{k}"] = p_ens
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "label_curve.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(probs).to_csv(os.path.join(RES, "label_curve_probs.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(f"评估集：第 1 轮随机样本 {len(r1)} 条（脚步 {y1.sum()}）")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
