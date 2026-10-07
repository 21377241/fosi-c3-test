"""Train footstep / non-footstep classifiers on several feature sets from human labels,
evaluate with leave-clips-out cross-validation, label every event with the ensemble, and
queue the events the models disagree on (or are unsure about) for the next round of human labelling.

Human labels (codes f=footstep, g=gunfire, o=other non-footstep, m=mixed, x=unsure) are read from
  labeling/human_labels*.csv           (exported by the label tool, one file per round)
  labeling/audacity_done/*.txt         (Audacity label export, label text "<event_id> <code> [note]")
  labeling/review_*.csv                (exported by the review tool; final_label overrides the above)
Outputs: results/events_labeled_v2.csv, labeling/queue_next.csv
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "labeling")
CACHE = os.path.join(HERE, "cache")
P_FOOT, P_NON = 0.8, 0.2
N_QUEUE, QUEUE_PER_CLIP = 80, 6
CODE = {"f": "footstep", "g": "non-footstep", "o": "non-footstep"}
SUBTYPE = {"g": "gun", "o": "other"}


def embedding_model():
    return make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
                         LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))


def acoustic_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))


FEATURE_SETS = [
    ("AST", "embeddings_ast128.npz", embedding_model),
    ("CLAP", "embeddings_clap.npz", embedding_model),
]


def read_human():
    parts = []
    for fn in sorted(glob.glob(os.path.join(LAB, "human_labels*.csv"))):
        d = pd.read_csv(fn, encoding="utf-8-sig", keep_default_na=False)
        if "note" not in d:
            d["note"] = ""
        parts.append(d[["event_id", "human_label", "note"]])
    rows = []
    for fn in glob.glob(os.path.join(LAB, "audacity_done", "*.txt")):
        for line in open(fn, encoding="utf-8-sig"):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            tok = cols[2].replace(",", " ").split()
            if len(tok) >= 2 and tok[0].isdigit() and len(tok[1]) == 1 and tok[1].lower() in "fgomx":
                rows.append((int(tok[0]), tok[1].lower(), " ".join(tok[2:])))
    if rows:
        parts.append(pd.DataFrame(rows, columns=["event_id", "human_label", "note"]))
    for fn in sorted(glob.glob(os.path.join(LAB, "review_*.csv"))):
        d = pd.read_csv(fn, encoding="utf-8-sig", keep_default_na=False)
        d = d[d.final_label != ""].rename(columns={"final_label": "human_label", "final_note": "note"})
        parts.append(d[["event_id", "human_label", "note"]])
    if not parts:
        sys.exit("no human labels found: export human_labels.csv from the label tool into labeling/, "
                 "or put Audacity label exports into labeling/audacity_done/")
    h = pd.concat(parts).drop_duplicates("event_id", keep="last")
    h["event_id"] = h.event_id.astype(int)
    h["human_label"] = h.human_label.astype(str).str.strip().str.lower()
    h["note"] = h.note.astype(str).str.strip()
    return h


def report_cv(name, y, prob):
    pred = (prob >= 0.5).astype(int)
    conf = (prob >= P_FOOT) | (prob <= P_NON)
    ba_conf = balanced_accuracy_score(y[conf], pred[conf]) if len(np.unique(y[conf])) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"  {name:8s} 平衡准确率={balanced_accuracy_score(y, pred):.3f}  AUC={roc_auc_score(y, prob):.3f}  "
          f"非脚步召回={tn}/{tn + fp}  脚步召回={tp}/{tp + fn}  "
          f"高置信覆盖={conf.mean():.0%} (其中平衡准确率={ba_conf:.3f})")


def main():
    ev = pd.read_csv(os.path.join(HERE, "results", "events.csv"))
    ev["event_id"] = np.arange(len(ev))
    groups_all = (ev.game + ev.clip_id.astype(str)).values
    sets = [(n, np.load(os.path.join(CACHE, f))["X"], mk) for n, f, mk in FEATURE_SETS
            if os.path.exists(os.path.join(CACHE, f))]
    print("使用的特征:", [n for n, _, _ in sets])

    h_all = read_human()
    print("人工标注:", h_all.human_label.value_counts().to_dict())
    for code, name in [("f", "脚步地面/背景"), ("o", "其他声音类型"), ("m", "混合声音组合")]:
        c = h_all[h_all.human_label == code].note.replace("", "(未填写)").value_counts()
        if len(c):
            print(f"{name}:", c.to_dict())
    h = h_all[h_all.human_label.isin(CODE)]
    idx = h.event_id.values
    y = (h.human_label == "f").astype(int).values
    groups = groups_all[idx]
    if y.sum() < 5 or (1 - y).sum() < 5:
        sys.exit("need at least 5 footstep and 5 non-footstep human labels")

    n_splits = min(5, len(np.unique(groups)))
    print(f"\n留片段交叉验证 ({n_splits} 折, 同一片段的事件不会同时出现在训练和测试中):")
    cv = {}
    for name, X, mk in sets:
        prob = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits).split(X[idx], y, groups):
            prob[te] = mk().fit(X[idx][tr], y[tr]).predict_proba(X[idx][te])[:, 1]
        cv[name] = prob
        report_cv(name, y, prob)
    ens_cv = np.mean(list(cv.values()), axis=0)
    report_cv("集成", y, ens_cv)
    pd.DataFrame({"event_id": idx, **{f"cv_{n}": p for n, p in cv.items()}, "cv_ens": ens_cv}).to_csv(
        os.path.join(HERE, "results", "human_cv_probs.csv"), index=False, encoding="utf-8-sig")
    if len(cv) > 1:
        hard = np.array([p >= 0.5 for p in cv.values()])
        agree = hard.all(0) | (~hard).all(0)
        ok = (hard[0] == y.astype(bool))
        print(f"  模型意见一致的人工事件占 {agree.mean():.0%}，其中判断正确率 {ok[agree].mean():.0%}；"
              f"意见不一致时（{(~agree).sum()} 个）集成判断正确率 "
              f"{((ens_cv[~agree] >= 0.5) == y[~agree].astype(bool)).mean():.0%}")

    P = {name: mk().fit(X[idx], y).predict_proba(X)[:, 1] for name, X, mk in sets}
    p_ens = np.mean(list(P.values()), axis=0)
    for name, p in P.items():
        ev[f"p_{name}"] = p
    ev["p_ens"] = p_ens
    hard = np.array([p >= 0.5 for p in P.values()])
    ev["models_disagree"] = ~(hard.all(0) | (~hard).all(0))
    ev["label"] = np.where(p_ens >= P_FOOT, "footstep", np.where(p_ens <= P_NON, "non-footstep", "ambiguous"))
    ev["subtype"] = ""
    ev["human_label"] = ""

    hh = h_all.set_index("event_id")
    hl = hh.human_label
    ev.loc[hl.index, "human_label"] = hl.values
    ev.loc[hl.index, "label"] = hl.map(CODE).fillna("ambiguous").values
    sub = hl.map(SUBTYPE).fillna("")
    has_note = hl.isin(["f", "o", "m"]) & (hh.note != "")
    sub[has_note] = hh.note[has_note]
    sub[(hl == "m") & ~has_note] = "mixed"
    ev.loc[hl.index, "subtype"] = sub.values
    ev.to_csv(os.path.join(HERE, "results", "events_labeled_v2.csv"), index=False)
    print("\n全部事件新标注:", ev.label.value_counts().to_dict())

    un = ev[ev.human_label == ""].copy()
    spread = np.std(np.array([un[f"p_{n}"] for n in P]), axis=0) if len(P) > 1 else 0
    un["priority"] = spread + (0.5 - np.abs(un.p_ens - 0.5))
    cand = un[un.models_disagree | ((un.p_ens > P_NON) & (un.p_ens < P_FOOT))]
    cand = cand.sort_values("priority", ascending=False).groupby(["game", "clip_id"]).head(QUEUE_PER_CLIP)
    cand = cand.head(N_QUEUE).sort_values(["game", "clip_id", "t"])
    cand[["event_id", "game", "clip_id", "t", "raw_peak", "p_ens", "models_disagree"]].to_csv(
        os.path.join(LAB, "queue_next.csv"), index=False)
    print(f"未标注事件中: 模型意见不一致 {int(un.models_disagree.sum())} 个, 集成拿不准 "
          f"{int(((un.p_ens > P_NON) & (un.p_ens < P_FOOT)).sum())} 个; "
          f"已挑出 {len(cand)} 个 (每段最多 {QUEUE_PER_CLIP} 个) 写入 labeling/queue_next.csv 供下一轮人工判断")


if __name__ == "__main__":
    main()
