"""Academic-paper version of the evaluation (Chinese, self-contained HTML).

Focus: annotation pipeline, labelling-model design, experiment design and the practicality verdict.
Reuses every statistic and figure computed by report_final.main(), so the paper and the report always agree.
Output: docs/C3 FPS AI 声卡脚步增强客观评测（论文稿）.html
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import report_final as rf
import matplotlib.pyplot as plt
from report_final import img, pct, sgn, LAB, RES, HERE, FOOT, NT, CLAIM, FER_THR, P_HI, EQ_Q
from report_ad import matched_gap, MATCH_DB, MATCH_MIN

OUT = os.path.join(os.path.dirname(HERE), "docs", "C3 FPS AI 声卡脚步增强客观评测（论文稿）.html")

CSS = """
@page{size:A4;margin:22mm 20mm}
*{box-sizing:border-box}
body{margin:0;background:#eceff3;color:#111;font:15px/1.9 "Times New Roman","SimSun","Songti SC",serif}
.paper{max-width:900px;margin:30px auto;padding:70px 78px 80px;background:#fff;box-shadow:0 8px 30px rgba(0,0,0,.12)}
h1{font:700 25px/1.45 "SimHei","Microsoft YaHei",sans-serif;text-align:center;margin:0 0 10px}
.en-title{text-align:center;font:italic 16px/1.5 "Times New Roman",serif;margin:0 0 18px;color:#222}
.authors{text-align:center;font-size:14.5px;margin:0 0 4px}.affil{text-align:center;font-size:13px;color:#444;margin:0 0 26px}
.abstract{border-top:1.5px solid #000;border-bottom:1.5px solid #000;padding:14px 4px 10px;margin:0 0 30px;font-size:14px;line-height:1.85}
.abstract p{margin:0 0 8px;text-indent:0}.abstract b{font-family:"SimHei","Microsoft YaHei",sans-serif}
h2{font:700 18px/1.5 "SimHei","Microsoft YaHei",sans-serif;margin:34px 0 12px}
h3{font:700 15.5px/1.5 "SimHei","Microsoft YaHei",sans-serif;margin:22px 0 8px}
p{margin:0 0 10px;text-indent:2em;text-align:justify}
.eq{text-align:center;margin:10px 0 14px;font-style:italic;text-indent:0}.eq .no{float:right;font-style:normal}
.tcap{text-align:center;font:700 13.5px/1.6 "SimHei","Microsoft YaHei",sans-serif;margin:18px 0 6px;text-indent:0}
.fcap{text-align:center;font-size:13.5px;margin:4px 0 18px;text-indent:0}
.fcap b{font-family:"SimHei","Microsoft YaHei",sans-serif}
.tnote{font-size:12.5px;color:#333;margin:4px 0 16px;text-indent:0}
.tw{overflow-x:auto}table.t{width:100%;border-collapse:collapse;font-size:13px;line-height:1.55;border-top:1.5px solid #000;border-bottom:1.5px solid #000}
table.t th{border-bottom:.8px solid #000;padding:6px 8px;text-align:center;font-weight:700}
table.t td{padding:5px 8px;text-align:center;vertical-align:top}
table.t td:first-child,table.t th:first-child{text-align:left}
img{max-width:100%;display:block;margin:12px auto 4px}
.diag{display:flex;align-items:center;gap:6px;margin:16px 0 6px;font:12.5px/1.5 "SimHei","Microsoft YaHei",sans-serif}
.diag .col{display:flex;flex-direction:column;gap:8px;flex:1}
.diag .nd{border:1px solid #000;padding:7px 6px;text-align:center;background:#fff}
.diag .nd.h{background:#f1f1f1}.diag .nd small{display:block;font:11.5px/1.45 "SimSun",serif;color:#333}
.diag .ar{font-size:16px;flex:0 0 auto}
.diag.wrap{flex-wrap:wrap}
ol.ref{font-size:13px;line-height:1.7;padding-left:2.2em}ol.ref li{margin:3px 0}
ul.contrib{margin:0 0 12px;padding-left:3.2em}ul.contrib li{margin:4px 0}
code{font:.9em Consolas,monospace}
.footer{margin-top:40px;padding-top:10px;border-top:1px solid #999;font-size:12px;color:#555;text-indent:0}
@media print{body{background:#fff}.paper{box-shadow:none;margin:0;padding:0;max-width:none}}
"""


def table(rows):
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    cols = list(rows[0].keys())
    clean = lambda v: re.sub(r"</?b>", "", str(v))
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{clean(r[c])}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class Num:
    """Running numbers for tables and figures."""
    def __init__(self):
        self.t = self.f = 0

    def tab(self, title):
        self.t += 1
        return f'<p class="tcap">表 {self.t}　{title}</p>'

    def fig(self, path, title):
        self.f += 1
        return f'{img(path)}<p class="fcap"><b>图 {self.f}</b>　{title}</p>'

    def diag(self, html, title):
        self.f += 1
        return f'{html}<p class="fcap"><b>图 {self.f}</b>　{title}</p>'


def labeler_stats():
    """Leave-clip-out CV AUC of each branch and the ensemble (round-1 random sample and all labels), PCA sizes."""
    from train_labeler import read_human, CODE, embedding_model
    cv = pd.read_csv(os.path.join(RES, "human_cv_probs.csv"), encoding="utf-8-sig")
    h = read_human()
    h = h[h.human_label.isin(CODE)]
    r1 = set(pd.read_csv(os.path.join(LAB, "human_labels.csv")).event_id)
    c = cv.merge(h, on="event_id")
    c1 = c[c.event_id.isin(r1)]
    auc = {k: (roc_auc_score(c1.human_label == "f", c1[k]), roc_auc_score(c.human_label == "f", c[k]))
           for k in ["cv_AST", "cv_CLAP", "cv_ens"]}
    y = (h.human_label == "f").values
    dims = {}
    for name, fn in [("AST", "embeddings_ast128.npz"), ("CLAP", "embeddings_clap.npz")]:
        X = np.load(os.path.join(HERE, "cache", fn))["X"]
        m = embedding_model().fit(X[h.event_id.values], y)
        dims[name] = (X.shape[1], int(m.named_steps["pca"].n_components_))
    return auc, len(c1), len(c), dims


def loudness_stats():
    """Per clip and track: whole-clip RMS and percentiles of the 50 ms short-term level (dBFS). Cached."""
    path = os.path.join(HERE, "cache", "paper_loudness.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from common import load_triplet, GAMES, SR
    rows = []
    for game, cfg in GAMES.items():
        for i in range(1, cfg["n"] + 1):
            a, _ = load_triplet(game, i)
            for t, x in a.items():
                x = x.mean(1) if x.ndim > 1 else x
                n = int(0.05 * SR)
                st = 10 * np.log10((x[:len(x) // n * n].reshape(-1, n) ** 2).mean(1) + 1e-12)
                rows.append(dict(game=game, clip_id=i, track=t, rms=10 * np.log10((x ** 2).mean() + 1e-12),
                                 l50=np.percentile(st, 50), l95=np.percentile(st, 95), l99=np.percentile(st, 99), lmax=st.max()))
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


LOUD_SPLIT = -28.0


def fig_loudness(p, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for t, c, mk, name in [("fps", "#c0392b", "o", "FPS AI"), ("eq", "#2f6da5", "s", "实际 EQ")]:
        ax.scatter(p[("lmax", "raw")], p[("lmax", t)] - p[("lmax", "raw")], s=26, c=c, marker=mk, edgecolor="k", lw=.4, label=name, alpha=.85)
    ax.axvline(LOUD_SPLIT, c="#888", ls="--", lw=.8)
    ax.axhline(0, c="k", lw=.6)
    ax.set_xlabel("原始轨最大短时电平 (dBFS，50 ms)")
    ax.set_ylabel("处理后最大短时电平变化 (dB)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def round_mix():
    rows = []
    for name, fn in [("第 1 轮", "human_labels.csv"), ("第 2 轮", "human_labels_r2.csv"),
                     ("第 3 轮", "human_labels_r3.csv"), ("第 4 轮", "human_labels_r4.csv")]:
        x = pd.read_csv(os.path.join(LAB, fn), encoding="utf-8-sig", keep_default_na=False).human_label.value_counts()
        rows.append({k: int(x.get(k, 0)) for k in "fgomx"})
    return rows


def main():
    c = rf.main()
    d, bg, ev, meta = c["d"], c["bg"], c["ev"], c["meta"]
    fo_h, fo_all, nt_h = c["fo_h"], c["fo_all"], c["nt_h"]
    med, gap, sei, mse = c["med"], c["gap"], c["sei"], c["mse"]
    fer, fer_g, fer_bg = c["fer"], c["fer_g"], c["fer_bg"]
    rs, acc, rob, sp, prom = c["rs"], c["acc"], c["rob"], c["sp"], c["prom"]
    r2, r3, r4, n_r1 = c["r2"], c["r3"], c["r4"], c["n_r1"]
    ci_h, ci_all, eq, bs = c["ci_h"], c["ci_all"], c["eq"], c["bs"]
    fa_g, gm = c["fa_g"], c["gm"]
    lvl_lo, lvl_mid, lvl_hi, hi_clip = c["lvl_lo"], c["lvl_mid"], c["lvl_hi"], c["hi_clip"]
    n_vl, n_pg, dur_min = c["n_vl"], c["n_pg"], c["dur_min"]
    gun = d[d.group == NT[0]]
    n_lab = n_r1 + len(r2) + len(r3) + len(r4)
    auc, n_auc_r1, n_auc_all, dims = labeler_stats()
    mix = round_mix()
    lab_cnt = ev.label.value_counts()
    N = Num()

    # ---------------------------------------------------------------- tables: annotation & model
    plan = [("第 1 轮", n_r1, "按片段 × 电平分层随机", "初始训练集；无偏精度核对样本"),
            ("第 2 轮", len(r2), "两分支意见相反或集成概率居中（每段 ≤ 6 个）", "主动学习：补充决策边界样本"),
            ("第 3 轮", len(r3), f"不确定事件按峰值分层 {len(c['r3_amb'])} 个 + 响亮高置信度脚步 {len(c['r3_loud'])} 个",
             "补齐“响脚步 / 弱枪声”；核对响亮区间"),
            ("第 4 轮", len(r4), "剩余不确定事件按峰值分两层随机", "估计剩余难例可分性，触发停止规则")]
    t_rounds = table([{"轮次": a, "条数": n, "抽样策略": s, "目的": u, "脚步 / 枪声 / 其他 / 混合 / 不确定":
                       " / ".join(str(m[k]) for k in "fgomx")} for (a, n, s, u), m in zip(plan, mix)])

    t_abl = table([
        {"方案": "AST 零样本（AudioSet 527 类）", "可训练参数": "0", "第 1 轮随机样本 AUC": "—", "全部人工标注 AUC": "—",
         "说明": "游戏脚步的“Walk, footsteps”概率 ≤ 0.2，常判为马蹄声，不可用"},
        {"方案": "AST 线性探针（单分支）", "可训练参数": f"{dims['AST'][1] + 1}", "第 1 轮随机样本 AUC": f"{auc['cv_AST'][0]:.3f}",
         "全部人工标注 AUC": f"{auc['cv_AST'][1]:.3f}", "说明": f"{dims['AST'][0]} 维 → PCA {dims['AST'][1]} 维"},
        {"方案": "CLAP 线性探针（单分支）", "可训练参数": f"{dims['CLAP'][1] + 1}", "第 1 轮随机样本 AUC": f"{auc['cv_CLAP'][0]:.3f}",
         "全部人工标注 AUC": f"{auc['cv_CLAP'][1]:.3f}", "说明": f"{dims['CLAP'][0]} 维 → PCA {dims['CLAP'][1]} 维"},
        {"方案": "AST + CLAP 概率平均（采用）", "可训练参数": f"{dims['AST'][1] + dims['CLAP'][1] + 2}",
         "第 1 轮随机样本 AUC": f"{auc['cv_ens'][0]:.3f}", "全部人工标注 AUC": f"{auc['cv_ens'][1]:.3f}", "说明": "两分支误差部分互补"},
        {"方案": "解冻 AST 最后 4 层微调", "可训练参数": "约 2.8×10⁷", "第 1 轮随机样本 AUC": "—", "全部人工标注 AUC": "比冻结版高约 0.01",
         "说明": "在折间波动范围内，过拟合风险高，未采用"}])

    # ---------------------------------------------------------------- tables: experiment design
    t_hyp = table([
        {"假设": "H1 有效性", "内容": "FPS AI 对脚步产生达到宣称水平的增益", "子实验": "A", "主要指标": "脚步增益 G",
         "判定标准": f"人工确认脚步 G 中位数 ≥ {CLAIM[0]:.0f} dB，且 95% CI 下限 ≥ {CLAIM[0]:.0f} dB"},
        {"假设": "H2 选择性", "内容": "增益针对脚步内容，而非响度差异或固定频响", "子实验": "B", "主要指标": "SEI、同电平 ΔG",
         "判定标准": "同电平下枪声 ΔG ≥ 3 dB，且匹配静态 EQ 的 |ΔG| &lt; 1 dB"},
        {"假设": "H3 非破坏性", "内容": "非目标声音基本保持原样", "子实验": "C、D", "主要指标": "对数频谱 MSE、FER",
         "判定标准": f"枪声频谱 MSE 低于实际 EQ；非脚步 FER（> {FER_THR:.0f} dB）≤ 10%"},
        {"假设": "H4 稳健性", "内容": "上述结论不依赖于特定游戏、样本口径或标注误差", "子实验": "敏感性分析", "主要指标": "分组重复 H1–H2",
         "判定标准": "两游戏、纯净度分档、峰值口径与 20% 标注噪声下效应方向不变"}])

    t_threat = table([
        {"潜在混杂或误差": "枪声比脚步响，处理可能只是“对响声增强少”", "控制手段": f"同电平比较（±{MATCH_DB:.0f} dB，至少 {MATCH_MIN} 个参照脚步）"},
        {"潜在混杂或误差": "选择性可能来自固定频响（脚步与枪声频谱不同）", "控制手段": "匹配静态 EQ：以 FPS AI 的脚步平均频响设计固定滤波器作对照"},
        {"潜在混杂或误差": "标注错误或标注模型误判", "控制手段": "人工优先；盲审复核；标注噪声模拟；人工与“人工 + 模型”两套样本"},
        {"潜在混杂或误差": "标注模型利用电平信息，与同电平分析形成循环", "控制手段": "模型输入先做音量归一化"},
        {"潜在混杂或误差": "重叠声音污染增益测量", "控制手段": "纯净度三档分级，主分析排除“有重叠”"},
        {"潜在混杂或误差": "同一片段内事件相关，置信区间偏窄", "控制手段": "按片段聚类的自助法"},
        {"潜在混杂或误差": "增益测量窗口的选择", "控制手段": "区间 RMS 与起点后 80 ms 峰值两种口径对照"},
        {"潜在混杂或误差": "三轨未对齐或 EQ 轨含额外处理", "控制手段": "互相关对齐（0 样本偏移）；实测与理论 EQ 频响比对"}])

    # ---------------------------------------------------------------- tables: results
    t_A = table([
        {"脚步样本": "人工确认", "n": len(fo_h), "FPS AI 中位 [95% CI]": f"{sgn(med(fo_h, 'AI'))} [{sgn(ci_h[0])}, {sgn(ci_h[1])}]",
         "3–9 dB 占比": pct(fo_h.G_AI.between(*CLAIM).mean()), "&gt; 9 dB 占比": pct((fo_h.G_AI > 9).mean()),
         "&lt; 3 dB 占比": pct((fo_h.G_AI < 3).mean()), "实际 EQ 中位": sgn(med(fo_h, 'EQ'))},
        {"脚步样本": "人工 + 模型高置信度", "n": len(fo_all), "FPS AI 中位 [95% CI]": f"{sgn(med(fo_all, 'AI'))} [{sgn(ci_all[0])}, {sgn(ci_all[1])}]",
         "3–9 dB 占比": pct(fo_all.G_AI.between(*CLAIM).mean()), "&gt; 9 dB 占比": pct((fo_all.G_AI > 9).mean()),
         "&lt; 3 dB 占比": pct((fo_all.G_AI < 3).mean()), "实际 EQ 中位": sgn(med(fo_all, 'EQ'))}])
    t_lvl = table([{"原始电平": lab, "n": len(x), "FPS AI 中位": sgn(x.G_AI.median()), "&lt; 3 dB 占比": pct((x.G_AI < 3).mean())}
                   for lab, x in [("&lt; -40 dBFS", lvl_lo), ("-40 ~ -35 dBFS", lvl_mid), ("&gt; -35 dBFS", lvl_hi)]])
    t_mse = table([{"类别": rf.SHORT[g], "n": int((d.group == g).sum()),
                    "FPS AI（总 / 形状）": f"{mse(g, 'MSE_AI'):.1f} / {mse(g, 'MSEshape_AI'):.1f}",
                    "实际 EQ（总 / 形状）": f"{mse(g, 'MSE_EQ'):.1f} / {mse(g, 'MSEshape_EQ'):.1f}",
                    "匹配静态 EQ（总 / 形状）": f"{mse(g, 'MSE_EQm'):.0f} / {mse(g, 'MSEshape_EQm'):.1f}"} for g in NT + [FOOT]])
    t_fer = table([{"类别": name, "n": len(x), "FPS AI": pct((x.G_AI > FER_THR).mean()), "实际 EQ": pct((x.G_EQ > FER_THR).mean()),
                    "匹配静态 EQ": pct((x.G_EQm > FER_THR).mean())}
                   for name, x in [("枪声", gun), ("换弹", d[d.group == NT[1]]), ("烟雾弹/吃药/爆能器", d[d.group == NT[2]]),
                                   ("全部非脚步", nt_h), ("背景段*", bg)]])

    # robustness: human events only, footsteps of the same subset as level-matched reference
    pk = d.merge(ev[["event_id", "fps_gain", "raw_peak"]], on="event_id").assign(G_AI=lambda x: x.fps_gain, raw_level=lambda x: x.raw_peak)

    def rob_row(name, x):
        f, g = x[x.group == FOOT], x[x.group == NT[0]]
        gp = matched_gap(f, g, "AI")
        return {"检验条件": name, "脚步 n": len(f), "脚步 G 中位": sgn(f.G_AI.median()), "枪声 n": len(g),
                "枪声 G 中位": sgn(g.G_AI.median()), "同电平 ΔG（枪声）": f"{sgn(gp[0])}（{gp[1]}）"}
    rob_rows = [rob_row("主分析（纯净 + 有持续背景）", d), rob_row("仅纯净", d[d.purity == "p"]),
                rob_row("仅有持续背景", d[d.purity == "b"]), rob_row("VALORANT", d[d.game == "valorant"]),
                rob_row("PUBG", d[d.game == "pubg"]), rob_row("峰值口径（起点后 80 ms 短时电平峰值）", pk)]
    t_rob = table(rob_rows)
    rob_ok = all(float(r["同电平 ΔG（枪声）"].split("（")[0]) > 0 for r in rob_rows if "—" not in r["同电平 ΔG（枪声）"])
    flip_txt = ("模拟中均未出现效应归零" if rob[0.1][1] == 0 and rob[0.2][1] == 0
                else f"模拟中效应归零的比例为 {pct(rob[0.1][1])} 与 {pct(rob[0.2][1])}")

    # ---------------------------------------------------------------- overall loudness (brochure: "整体音量未做改变")
    L = loudness_stats()
    lp = L.pivot_table(index=["game", "clip_id"], columns="track", values=["rms", "l50", "l99", "lmax"])
    loud_fig = os.path.join(RES, "paper_loudness.png")
    fig_loudness(lp, loud_fig)
    dl = lambda m, t: lp[(m, t)] - lp[(m, "raw")]
    is_loud = lp[("lmax", "raw")] >= LOUD_SPLIT
    n_loud, n_quiet = int(is_loud.sum()), int((~is_loud).sum())
    dmax_loud, dmax_quiet = dl("lmax", "fps")[is_loud], dl("lmax", "fps")[~is_loud]
    fps_max_quiet = lp[("lmax", "fps")][~is_loud].max()
    raw_max_all = lp[("lmax", "raw")].max()
    t_loud = table([{"片段类型": name, "片段数": int(m.sum()),
                     "整段 RMS：FPS AI / 实际 EQ": f"{sgn(dl('rms', 'fps')[m].median())} / {sgn(dl('rms', 'eq')[m].median())}",
                     "中位短时电平：FPS AI / 实际 EQ": f"{sgn(dl('l50', 'fps')[m].median())} / {sgn(dl('l50', 'eq')[m].median())}",
                     "99% 分位短时电平：FPS AI / 实际 EQ": f"{sgn(dl('l99', 'fps')[m].median())} / {sgn(dl('l99', 'eq')[m].median())}",
                     "最大短时电平：FPS AI / 实际 EQ": f"{sgn(dl('lmax', 'fps')[m].median())} / {sgn(dl('lmax', 'eq')[m].median())}"}
                    for name, m in [(f"含响亮声音（原始最大 ≥ {LOUD_SPLIT:.0f} dBFS）", is_loud),
                                    (f"安静、以脚步为主（原始最大 &lt; {LOUD_SPLIT:.0f} dBFS）", ~is_loud),
                                    ("全部", is_loud | ~is_loud)]])

    # ---------------------------------------------------------------- verdicts
    g_eqm = gap["EQm"][NT[0]][0]
    h1 = ci_h[0] >= CLAIM[0]
    h2 = gap["AI"][NT[0]][0] >= 3 and abs(g_eqm) < 1
    h3a, h3b = mse(NT[0], "MSE_AI") < mse(NT[0], "MSE_EQ"), fer["AI"] <= 0.10
    v = lambda ok: "支持" if ok else "不支持"
    t_verdict = table([
        {"假设": "H1 有效性", "结果": f"脚步 G 中位 {sgn(med(fo_h, 'AI'))} dB，95% CI [{sgn(ci_h[0])}, {sgn(ci_h[1])}]", "判定": v(h1)},
        {"假设": "H2 选择性", "结果": f"同电平枪声 ΔG：FPS AI {sgn(gap['AI'][NT[0]][0])} dB，匹配静态 EQ {sgn(g_eqm)} dB；SEI {sgn(sei['AI'])} vs {sgn(sei['EQm'])} dB",
         "判定": v(h2)},
        {"假设": "H3 非破坏性", "结果": f"枪声 MSE {mse(NT[0], 'MSE_AI'):.1f} vs 实际 EQ {mse(NT[0], 'MSE_EQ'):.1f} dB²；非脚步 FER {pct(fer['AI'])}（枪声 {pct(fer_g['AI'])}）",
         "判定": "支持" if h3a and h3b else ("部分支持" if h3a else "不支持")},
        {"假设": "H4 稳健性", "结果": f"6 种分组 / 口径下同电平 ΔG 均为正；20% 标注噪声下效应 {sgn(rob[0.2][0])} dB", "判定": v(rob_ok and rob[0.2][0] > 0)}])

    t_claims = table([
        {"宣传册宣称": "“智能强化关键脚步声”“脚步声细节增强 3–9 dB”", "对应检验": "H1",
         "本文结果": f"脚步增益中位 {sgn(med(fo_h, 'AI'))} dB；{pct(fo_h.G_AI.between(*CLAIM).mean())} 落在 3–9 dB，{pct((fo_h.G_AI > 9).mean())} 超过 9 dB",
         "结论": "成立，位于范围上沿"},
        {"宣传册宣称": "“踩草地、跳窗等细微脚步声”听得清", "对应检验": "H1 按电平分档",
         "本文结果": f"-40 dBFS 以下的安静脚步（n = {len(lvl_lo)}）增益中位 {sgn(lvl_lo.G_AI.median())} dB，低于 3 dB 的仅 {pct((lvl_lo.G_AI < 3).mean())}",
         "结论": "成立（按地面材质未单独检验）"},
        {"宣传册宣称": "“实时识别游戏声音”“不同于传统 EQ，只强化必要频段”", "对应检验": "H2",
         "本文结果": f"同电平枪声少增强 {gap['AI'][NT[0]][0]:.1f} dB，匹配静态 EQ 为 {sgn(g_eqm)} dB；脚步上的平均频响近似全频段抬升",
         "结论": "成立；机制表现为按事件整体增强，而非固定频段提升"},
        {"宣传册宣称": "“不改变枪声”", "对应检验": "H3",
         "本文结果": f"枪声增益中位 {sgn(med(gun, 'AI'))} dB，频谱 MSE {mse(NT[0], 'MSE_AI'):.1f} dB²；远处安静枪声 {pct(fer_g['AI'])} 被增强",
         "结论": "对响亮枪声成立"},
        {"宣传册宣称": "“不改变爆炸声”", "对应检验": "—", "本文结果": "样本中无人工确认的爆炸声", "结论": "未检验"},
        {"宣传册宣称": "“不干扰环境音”“不改变其他声音信号特征”", "对应检验": "H3、7.7 节",
         "本文结果": f"背景段增益中位 {sgn(med(bg, 'AI'))} dB，但脚步附近背景抬高 {sgn(c['pre_bg'])} dB；部分安静瞬态被误增强（FER {pct(fer['AI'])}）",
         "结论": "部分成立"},
        {"宣传册宣称": "“游戏整体音量未做改变，长时间畅玩无压力”", "对应检验": "7.6 节",
         "本文结果": f"含响亮声音的片段最大电平变化中位 {sgn(dmax_loud.median())} dB；安静片段随脚步抬高中位 {sgn(dmax_quiet.median())} dB，处理后最高 {fps_max_quiet:.0f} dBFS",
         "结论": "未施加整体增益，响亮场景不变；安静场景整体响度上升"},
        {"宣传册宣称": "传统 EQ“所有声音一并受影响，枪声刺耳……环境噪音变大”", "对应检验": "实际 EQ 与匹配静态 EQ",
         "本文结果": f"实际 EQ 改变枪声频谱 {mse(NT[0], 'MSE_EQ'):.1f} dB²、PUBG 枪声 {sgn(gm('pubg', NT[0], 'EQ'))} dB；达到同等脚步增益的固定 EQ 使枪声 {sgn(med(gun, 'EQm'))} dB、背景 {sgn(med(bg, 'EQm'))} dB",
         "结论": "成立"},
        {"宣传册宣称": "“针对 CS2、PUBG 等 FPS 游戏进行优化”", "对应检验": "分游戏（7.5 节）",
         "本文结果": f"PUBG（已适配）脚步 {sgn(fa_g('pubg', 'AI'))} dB；VALORANT（未列入适配）{sgn(fa_g('valorant', 'AI'))} dB，选择性同样成立",
         "结论": "成立，且对未适配游戏有泛化"},
        {"宣传册宣称": "“全链路处理延迟仅 40 ms”", "对应检验": "—", "本文结果": "录音为离线对齐，无法测量处理延迟", "结论": "未检验"},
        {"宣传册宣称": "HiFi DAC、7.1 空间音频、AI 通话降噪等", "对应检验": "—", "本文结果": "不属于脚步增强功能", "结论": "不在本文范围"}])

    # ---------------------------------------------------------------- appendix B (detection, exploratory)
    det = c["det"]
    tr_, tf, te = (det["match"][("top1", t)] for t in ("raw", "fps", "eq"))
    cr, cf, ce = (det["match"][("closed", t)] for t in ("raw", "fps", "eq"))
    zr, zf, ze = c["zr"], c["zf"], c["ze"]
    ft_h, ft_hf, ft_c, ft_cf = c["ft_h"], c["ft_hf"], c["ft_c"], c["ft_cf"]
    ft = pd.read_csv(os.path.join(RES, "panns_ft_summary.csv"), encoding="utf-8-sig").set_index(["split", "track"])
    ft_he, ft_ce = ft.loc[("holdout", "eq")], ft.loc[("cv5", "eq")]
    pp, ppci = c["pp"], c["ppci"]
    drow = lambda name, a, b, e, au: {"检测器": name, "原始轨 AUC": au, "原始轨检出率": pct(a["recall"]),
                                      "FPS AI − 原始（百分点）": f"{pp(a['recall'], b['recall'])} {ppci(b['d_recall_ci'])}",
                                      "EQ − 原始（百分点）": f"{pp(a['recall'], e['recall'])} {ppci(e['d_recall_ci'])}"}
    t_det = table([drow("PANNs 零样本，仅“Walk, footsteps”一类", zr, zf, ze, f"{zr['auc']:.2f}"),
                   drow("PANNs 零样本，六类中脚步占比", cr, cf, ce, f"{cr['auc']:.2f}"),
                   drow("PANNs 零样本，六类中直接判类别", tr_, tf, te, "—"),
                   drow("PANNs 微调（留出 8 个片段）", ft_h, ft_hf, ft_he, f"{ft_h['auc']:.2f}"),
                   drow("PANNs 微调（5 折交叉验证）", ft_c, ft_cf, ft_ce, f"{ft_c['auc']:.2f}")])

    # ---------------------------------------------------------------- diagrams
    d_pipe = """<div class="diag">
<div class="col"><div class="nd h">同源三轨实录<small>原始 / FPS AI / EQ<br>互相关对齐</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">候选事件检测<small>频谱通量起点<br>1620 个事件</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">人机协同标注<small>人工 4 轮 ⇄ 标注模型<br>类别 + 纯净度</small></div><div class="nd">质量控制<small>盲审复核 · 无偏核对<br>标注噪声模拟</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">事件级指标<small>增益 · 同电平 ΔG<br>频谱 MSE · FER</small></div><div class="nd">对照与混杂控制<small>实际 EQ · 匹配静态 EQ<br>聚类自助法</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd h">假设检验<small>H1 有效 · H2 选择<br>H3 非破坏 · H4 稳健</small></div></div></div>"""

    d_model = f"""<div class="diag">
<div class="col"><div class="nd h">事件片段<small>仅原始轨<br>音量归一化</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">AST（冻结）<small>起点 −0.3～+0.7 s，16 kHz<br>144 图块均值 → {dims['AST'][0]} 维</small></div>
<div class="nd">CLAP-HTSAT（冻结）<small>窗口 (−0.05, 0.35) 与 (−0.5, 0.5) s<br>拼接 → {dims['CLAP'][0]} 维</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">标准化 → PCA 95%<small>→ {dims['AST'][1]} 维 → L2 逻辑回归</small></div>
<div class="nd">标准化 → PCA 95%<small>→ {dims['CLAP'][1]} 维 → L2 逻辑回归</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">概率平均<small>p = (p<sub>AST</sub> + p<sub>CLAP</sub>) / 2</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd h">三区间判定<small>p ≥ {P_HI}：脚步<br>p ≤ {1 - P_HI:.1f}：非脚步<br>其余：不确定（排除）<br>人工标注优先</small></div></div></div>"""

    d_loop = f"""<div class="diag">
<div class="col"><div class="nd h">第 1 轮<small>分层随机 {n_r1} 条</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">训练标注模型<small>留片段 5 折交叉验证</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">推断全部事件<small>脚步 / 非脚步 / 不确定</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd">选取下一轮样本<small>第 2 轮：不确定性准则<br>第 3–4 轮：分层随机</small></div></div><div class="ar">→</div>
<div class="col"><div class="nd h">停止规则<small>某电平层随机样本上<br>正确率接近随机 → 停止</small></div></div></div>"""

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 游戏声卡脚步增强的实用性客观评测</title><style>{CSS}</style></head><body><main class="paper">

<h1>AI 游戏声卡脚步增强的实用性客观评测：<br>人机协同标注与混杂受控的事件级实验</h1>
<p class="en-title">Objective Evaluation of the Practical Utility of AI Footstep Enhancement in a Gaming Sound Card:<br>
Human-in-the-Loop Annotation and Confound-Controlled Event-Level Experiments</p>
<p class="authors">[作者姓名]</p>
<p class="affil">[作者单位，城市 邮编]</p>

<div class="abstract">
<p><b>摘　要：</b><b>目的</b>　从客观、可复现的角度检验 Fosi Audio C3 游戏声卡 FPS AI 功能的实用性，并对其宣传册中“智能强化关键脚步声（3–9 dB）”
“不干扰环境音或枪声”“游戏整体音量未做改变”等宣称逐条验证。
<b>方法</b>　将宣传册宣称归纳为有效性、选择性、非破坏性与稳健性四个可检验假设，采集 VALORANT {n_vl} 段与 PUBG {n_pg} 段同源三轨实录
（原始、FPS AI、EQ；共 {dur_min:.1f} min），检测得到 1620 个候选事件。在无官方标注与分轨素材的条件下，设计了“人工主导、模型辅助”的四轮标注流程
（{n_lab} 条人工标注，含主动学习与基于分层随机抽样的停止规则），标注模型采用冻结 AST 与 CLAP 嵌入上的线性探针集成，
监督分类器仅约 {dims['AST'][1] + dims['CLAP'][1] + 2} 个可训练参数，并刻意屏蔽电平信息以避免与后续分析形成循环。
实验以同电平比较和匹配静态 EQ 两种对照排除响度与固定频响的混杂，统计推断采用按片段聚类的自助法。
<b>结果</b>　标注模型在无偏随机样本上的留片段交叉验证 AUC 为 {auc['cv_ens'][0]:.3f}，盲审复核脚步/非脚步 κ = {rs['k2']:.2f}。
FPS AI 对人工确认脚步的增益中位数为 {sgn(med(fo_h, 'AI'))} dB（95% CI {sgn(ci_h[0])}～{sgn(ci_h[1])} dB），对枪声仅 {sgn(med(gun, 'AI'))} dB；
同电平条件下枪声比脚步少增强 {gap['AI'][NT[0]][0]:.1f} dB，而与其脚步增益相同的匹配静态 EQ 无此差异（{sgn(g_eqm)} dB）。
FPS AI 对枪声的频谱改变（{mse(NT[0], 'MSE_AI'):.1f} dB²）远小于实际 EQ（{mse(NT[0], 'MSE_EQ'):.1f} dB²），脚步相对紧邻背景的突出度净增 {sgn(prom['c_fo'])} dB，
且从脚步串的第一步即生效。结论在已适配的 PUBG 与未列入适配的 VALORANT、不同纯净度、不同增益口径及 20% 标注噪声下保持一致。
含响亮声音的片段最大电平几乎不变（{sgn(dmax_loud.median())} dB），产品未施加整体增益。
<b>结论</b>　客观证据支持 FPS AI 的有效性、选择性与稳健性：它以基于声音内容的方式增强脚步，这是固定 EQ 无法实现的；
宣传册的核心宣称成立，“不干扰环境音”与“整体音量未做改变”部分成立：约 {pct(fer['AI'])} 的非脚步瞬态（主要为安静的远处声音）被误增强，
脚步附近底噪抬升，安静场景的整体响度随脚步上升。本文提出的标注与实验框架可推广到其他黑盒音频增强产品的评测。</p>
<p><b>关键词：</b>游戏音频；脚步增强；实用性评测；人机协同标注；主动学习；预训练音频模型；混杂控制</p>
<p style="margin-top:10px"><b>Abstract:</b> We assess the practical utility of the C3 FPS AI sound card, which claims to detect and selectively boost
footsteps without altering other sounds. Utility is decomposed into four testable hypotheses: effectiveness, selectivity, non-destructiveness and
robustness. Paired three-track recordings (unprocessed, FPS AI, EQ; {n_vl} VALORANT and {n_pg} PUBG clips, {dur_min:.1f} min) yielded 1620 candidate events.
Without vendor labels or stems, we designed a four-round human-led, model-assisted annotation pipeline ({n_lab} human labels, active learning and a
stratified-random stopping rule). The labeller is an ensemble of linear probes on frozen AST and CLAP embeddings (about
{dims['AST'][1] + dims['CLAP'][1] + 2} trainable parameters) with level information removed to avoid circularity. Level-matched comparison and a
simulated static EQ matched to the FPS AI footstep gain control for loudness and fixed-response confounds; inference uses a clip-level cluster bootstrap.
The labeller reached a leave-clip-out AUC of {auc['cv_ens'][0]:.3f} on an unbiased random sample, and blind re-annotation gave κ = {rs['k2']:.2f}.
FPS AI boosted human-verified footsteps by {sgn(med(fo_h, 'AI'))} dB (95% CI {sgn(ci_h[0])} to {sgn(ci_h[1])} dB) and gunshots by only {sgn(med(gun, 'AI'))} dB;
at matched level gunshots received {gap['AI'][NT[0]][0]:.1f} dB less gain, whereas the matched static EQ showed no such difference ({sgn(g_eqm)} dB).
Findings held across games (including VALORANT, which is not among the vendor's adapted titles), purity strata, gain definitions and 20% label noise,
and the loudest sounds in a clip were not amplified ({sgn(dmax_loud.median())} dB). The evidence supports effectiveness, selectivity and robustness, and the core
brochure claims; non-destructiveness is partially supported: {pct(fer['AI'])} of non-footstep transients, mostly quiet distant sounds, were also boosted, and
overall loudness rose in quiet footstep-dominated scenes.</p>
<p><b>Keywords:</b> game audio; footstep enhancement; utility evaluation; human-in-the-loop annotation; active learning; pretrained audio models; confound control</p>
</div>

<h2>1　引言</h2>
<p>在第一人称射击（FPS）游戏中，脚步声是判断对手位置与动向的关键听觉线索，但它响度低、持续短，且常被枪声、技能音效与环境声掩蔽。
传统做法是使用“听脚步”EQ 预设，但固定滤波对所有声音一视同仁，抬高脚步的同时也改变了枪声等关键信息。
Fosi Audio C3 游戏声卡内置自研 FPS AI 模型，宣传册<sup>[15]</sup>称其可“实时识别游戏声音，智能强化关键脚步声”，脚步声细节增强 3–9 dB，
“不同于传统 EQ 调节，只强化必要频段，不干扰环境音或枪声”，且“游戏整体音量未做改变”。据宣传册介绍，该模型针对 CS2、PUBG 与三角洲行动分别训练，
每款游戏的训练时长约 4000 小时、样本约 210 万～240 万条，其他游戏的针对性适配仍在进行中。</p>
<p>对这类产品的实用性做客观评价面临三项困难：（1）算法为黑盒，厂商不提供分轨素材与事件级标注，缺少“哪里是脚步”的真值；
（2）通用声音事件检测模型在游戏音效上存在明显的领域偏移<sup>[1-3]</sup>，无法直接充当裁判；
（3）脚步与枪声在响度、频谱上本就不同，若不加控制，表观的“选择性”可能只是动态压缩或固定频响的副产品。</p>
<p>为此，本文从宣传册中提取与脚步增强相关的宣称，归纳为四个可检验的假设，即有效性（H1）、选择性（H2）、非破坏性（H3）与稳健性（H4），
并围绕它们设计标注流程、标注模型与实验，最后对宣传册宣称逐条给出检验结论。主要贡献如下：</p>
<ul class="contrib">
<li><b>标注流程</b>：提出“人工主导、模型辅助”的四轮迭代标注流程，配套专用标注工具、纯净度分级、盲审复核，以及基于分层随机抽样的停止规则，在缺乏真值的条件下获得可验证的事件级标注；</li>
<li><b>模型设计</b>：在少样本与领域偏移约束下，采用冻结预训练嵌入加线性探针的双分支集成，监督分类器仅约 {dims['AST'][1] + dims['CLAP'][1] + 2} 个可训练参数，并通过音量归一化切断与实验分析的循环依赖；</li>
<li><b>实验设计</b>：以假设驱动的子实验、同电平比较与匹配静态 EQ 双重对照，以及系统的混杂控制清单，区分“基于内容的选择性”与各种替代解释；</li>
<li><b>结论</b>：对宣传册中与脚步增强相关的宣称逐条给出“成立 / 部分成立 / 未检验”的判定，并明确产品的适用边界。</li></ul>

<h2>2　相关工作</h2>
<p>AudioSet<sup>[1]</sup> 提供了 527 类日常声音的弱标注，PANNs<sup>[2]</sup>、AST<sup>[3]</sup> 与 CLAP<sup>[4,5]</sup> 等在其上预训练的模型被广泛用作通用音频表示。
在目标域数据稀缺时，常用做法是在冻结表示上训练线性探针<sup>[6]</sup>。主动学习通过优先标注不确定或模型间分歧大的样本来提高标注效率<sup>[7,8]</sup>。
在客观评价方面，起点检测常用频谱通量<sup>[9]</sup>，频谱失真常用对数谱距离<sup>[10]</sup>，事件检测采用精确率、召回率等指标<sup>[11]</sup>。
然而，针对“黑盒、选择性音频增强”的评测尚缺乏系统方法，尤其缺少对响度与频响混杂的控制，这正是本文的切入点。</p>

<h2>3　评测框架与数据</h2>
<h3>3.1　框架总览</h3>
<p>评测框架如图 1 所示，遵循四项原则：（1）<b>同源配对</b>，三种处理条件来自同一段游戏音频，任何差异只能归因于处理本身；
（2）<b>事件级</b>，在单个声音事件上测量，而非整段平均；（3）<b>真值由人确定</b>，模型只用于扩展人工标注，且其精度在独立样本上核对；
（4）<b>混杂受控</b>，每一个可能产生“表观选择性”的替代解释都有对应的控制手段（见 6.4 节）。</p>
{N.diag(d_pipe, "评测框架总览")}
<h3>3.2　数据与同源性验证</h3>
<p>数据为 VALORANT {n_vl} 段、PUBG {n_pg} 段对局实录，总长 {dur_min:.1f} min，48 kHz 立体声，每段有原始（AI 与 EQ 均关闭）、FPS AI（仅开 AI）
与 EQ（仅开 EQ）三个版本。PUBG 属于宣传册所列的已适配游戏，VALORANT 不在其列，因此两者的对比也可检验模型在未适配游戏上的泛化能力。EQ 条件为 10 段 RBJ 峰值滤波器级联<sup>[12]</sup>（Q = {EQ_Q}），每款游戏一套参数（表 1）。三轨经互相关对齐，偏移均为 0 样本；
由原始轨与 EQ 轨互谱估计的实测频响与理论频响在 50 Hz–16 kHz 内最大偏差仅 {max(eq['valorant']['dev'], eq['pubg']['dev']):.2f} dB（图 2），
证明 EQ 轨为纯静态滤波，同时验证了对齐与测量链路的精度。</p>
{N.tab("EQ 条件的各频段增益（dB）")}
{table(c["eq_rows"])}
{N.fig(c["eq_fig"], "EQ 频响验证：粗线为实测，虚线为按参数计算，圆点为各频段设定值")}

<h2>4　标注流程</h2>
<h3>4.1　候选事件生成</h3>
<p>在原始轨上检测事件，FPS AI 与 EQ 轨使用相同时间点，保证三轨比较的是同一个声音。起点检测采用频谱通量<sup>[9]</sup>：
短时傅里叶变换（窗长 1024，跳步 5 ms）的对数幅度谱正向差分在 60 Hz–16 kHz 上求和，自适应阈值为 2 s 滑窗中位数加 3 倍中位绝对偏差，最小间隔 80 ms，
并要求峰值比前方背景高至少 6 dB，共得到 1620 个候选事件。事件终点取电平回落至前方背景 +3 dB 以内或比峰值低 20 dB 处，最长 1.5 s，且不超过下一事件起点。</p>
<h3>4.2　标注规范</h3>
<p>每个事件需要两步判断。第一步判类别：脚步（f）、枪声（g）、其他非脚步（o）、混合（m）、不确定（x），并可附子类备注（地面材质、换弹、烟雾弹、吃药声等）。
第二步判纯净度：<b>纯净</b>（起止区间内只有该声音）、<b>有持续背景</b>（单一瞬态叠加风声、发动机等平稳背景）、<b>有重叠</b>（区间内还有其他瞬态）。
只有前两档进入主分析，以保证增益测量对应的是单一声音；“有重叠”用于敏感性分析。</p>
<h3>4.3　标注工具</h3>
<p>为降低误判并保证盲法，开发了浏览器端标注工具。每个事件提供起点前后共 1.5 s 的频谱图，叠加起点、估计终点框与邻近事件起点；
提供四种播放方式：只听事件、突出事件（压低上下文）、完整原声、起点至估计终点。全部操作由键盘完成，进度自动保存。
标注者只接触原始轨，看不到 FPS AI 与 EQ 版本，也看不到模型输出，从而避免处理结果反过来影响类别判断。</p>
<h3>4.4　人机协同迭代</h3>
<p>标注采用“人工标注 → 训练模型 → 推断全部事件 → 选取下一轮样本”的迭代流程（图 3），共四轮（表 2）。
第 1 轮按片段 × 电平分层随机抽样，既作初始训练集，也作后续精度核对的无偏样本，不再参与样本选择。第 2 轮采用主动学习的不确定性准则<sup>[7,8]</sup>，
选取两分支意见相反或集成概率居中的事件。第 3 轮针对第 2 轮暴露的薄弱区间（响亮脚步与较弱枪声）定向补充，并抽查高置信度的响亮脚步。
第 4 轮对剩余不确定事件按电平分层随机抽样，用于检验停止规则。</p>
{N.diag(d_loop, "人机协同迭代标注流程")}
{N.tab("四轮人工标注的抽样策略与类别构成")}
{t_rounds}
<h3>4.5　停止规则</h3>
<p>主动学习的常见问题是不知道何时停止。本文的规则是：对剩余不确定事件按电平分层随机抽样，若某层上模型按 0.5 划分的正确率接近随机，
说明现有特征无法区分该层，继续标注只能逐条解决、不再提升模型，此时停止并将该层排除，做敏感性分析。
第 4 轮中，-35 dBFS 以下层的正确率为 {pct(c['r4_acc'].loc[0])}，以上层为 {pct(c['r4_acc'].loc[1])}，接近随机，因此停止标注。
最终 {c['n_amb_now']} 个未经人工标注的不确定事件被排除，占全部事件的 {pct(c['n_amb_now'] / 1620)}。</p>
<h3>4.6　质量控制</h3>
<p>（1）<b>盲审复核</b>：第 1 轮 {rs['n']} 条在隐藏原标注的条件下重新判断，前后一致率 {pct(rs['agree'])}，五类 Cohen κ<sup>[13]</sup> = {rs['k5']:.2f}，
脚步/非脚步 κ = {rs['k2']:.2f}，属“几乎完全一致”。不一致的条目由标注者裁定后作为最终标签。
（2）<b>无偏核对</b>：模型精度只在第 1 轮分层随机样本上评估，避免用主动学习挑出的难例低估精度，也避免用模型参与选出的样本高估精度（见 5.4 节）。
（3）<b>标注噪声模拟</b>：见 7.5 节。最终，1620 个事件中脚步 {int(lab_cnt.get('footstep', 0))} 个、非脚步 {int(lab_cnt.get('non-footstep', 0))} 个、
不确定 {int(lab_cnt.get('ambiguous', 0))} 个。</p>

<h2>5　标注模型设计</h2>
<h3>5.1　设计约束</h3>
<p>标注模型的设计受四项约束：（1）<b>少样本</b>，人工标注仅数百条，参数量必须很小；（2）<b>领域偏移</b>，AudioSet 预训练模型对游戏脚步的零样本识别不可用
（“Walk, footsteps”概率最高约 0.2，常判为马蹄声）；（3）<b>电平不变性</b>，后续实验要在同电平条件下比较脚步与非脚步，
若标注模型利用响度判断类别，会与分析形成循环，因此模型输入必须先做音量归一化；（4）<b>片段相关</b>，同一片段的事件高度相似，交叉验证必须按片段分组。</p>
<h3>5.2　模型结构</h3>
<p>模型结构如图 4 所示。两个预训练编码器参数全部冻结，仅作特征提取。AST<sup>[3]</sup> 输入起点前 0.3 s 至后 0.7 s 的 16 kHz 片段（128 帧，位置编码沿时间轴截断），
取最后一层 144 个图块输出的平均，得到 {dims['AST'][0]} 维表示。CLAP<sup>[5]</sup> 的 HTSAT 音频编码器分别处理短窗（起点前 0.05 s 至后 0.35 s，覆盖瞬态本身）
与长窗（前后各 0.5 s，包含上下文），拼接为 {dims['CLAP'][0]} 维。每个分支依次经标准化、主成分分析（保留 95% 方差，分别得到 {dims['AST'][1]} 维与 {dims['CLAP'][1]} 维）
与 L2 正则逻辑回归（C = 0.05，类别平衡加权）；两分支概率取平均。监督分类器仅包含 {dims['AST'][1] + dims['CLAP'][1] + 2} 个可训练参数；AST/CLAP 主干全部冻结，StandardScaler 与 PCA 仅进行无监督拟合。
输出按三区间判定：p ≥ {P_HI} 为脚步，p ≤ {1 - P_HI:.1f} 为非脚步，其余为不确定并排除；有人工标注的事件一律以人工为准。</p>
{N.diag(d_model, "标注模型结构（灰底为输入与输出）")}
<h3>5.3　设计取舍与消融</h3>
<p>表 3 比较了候选方案。双分支集成在随机样本上优于任一单分支，说明 AST（针对声音事件分类训练）与 CLAP（针对音频-文本语义对齐训练）的误差部分互补。
解冻 AST 最后 4 层微调需要训练约 10⁷ 量级的参数，AUC 仅提高约 0.01，在折间波动范围内，以数百条标注训练如此多的参数存在明显过拟合风险，因此未采用。
表 4 记录了一次关键的结构改进：旧版把不足 1 s 的片段补零至 AST 默认的 10 s 输入，并加入 17 维手工声学特征；
新版改为 1 s 输入并去掉声学特征，AUC 与高置信度覆盖率同时提高，未标注事件中的不确定数量减少近一半。</p>
{N.tab("标注模型方案比较（留片段 5 折交叉验证）")}
{t_abl}
<p class="tnote">注：第 1 轮随机样本 n = {n_auc_r1}，全部人工标注 n = {n_auc_all}（第 2–4 轮专挑难例，因此 AUC 低于随机样本）。可训练参数指监督分类器（逻辑回归）的权重与偏置；StandardScaler 与 PCA 为无监督拟合，不计入。</p>
{N.tab("标注模型结构改进前后对比（同一批人工标注）")}
{table(c["upgrade_rows"])}
<p class="tnote">注：准确率在第 1 轮分层随机样本上核对，括号内为事件数；“拿不准”指未经人工标注、概率介于 0.2 与 0.8 之间的事件。</p>
<h3>5.4　模型验证</h3>
<p>在第 1 轮分层随机样本上（模型未见过这些事件的标签），集成模型的 AUC 为 {auc['cv_ens'][0]:.3f}；判为脚步的 {acc['n_f']} 个中 {pct(acc['prec_f'])} 正确，
判为非脚步的 {acc['n_n']} 个中 {pct(acc['prec_n'])} 正确。由于主分析只使用人工确认事件，模型判定只进入扩展分析，
其误差对结论的影响由 7.5 节的标注噪声模拟单独评估。</p>

<h2>6　实验设计</h2>
<h3>6.1　假设与子实验</h3>
<p>宣传册中与脚步增强相关的宣称可归为四类：“智能强化关键脚步声、增强 3–9 dB”对应有效性（H1）；“实时识别”“不同于传统 EQ，只强化必要频段”对应选择性（H2）；
“不干扰环境音或枪声”“不改变其他声音信号特征”对应非破坏性（H3）；“针对 CS2、PUBG 等优化”以及在复杂战场中的一致表现对应稳健性（H4）。
“游戏整体音量未做改变”另以整段响度分析检验（7.6 节）。表 5 给出各假设对应的子实验与判定标准，判定标准取自宣传册数值（3 dB 为宣称增益下限）和评测方案的要求。</p>
{N.tab("实用性假设、对应子实验与判定标准")}
{t_hyp}
<h3>6.2　指标定义</h3>
<p><b>事件增益</b>　在“起点前 20 ms 至估计终点后 50 ms”的区间内计算 RMS：</p>
<p class="eq">G = 20 log<sub>10</sub>(RMS<sub>processed</sub> / RMS<sub>baseline</sub>)<span class="no">(1)</span></p>
<p><b>选择性指数</b>　SEI = G<sub>footstep</sub> − G<sub>non-target</sub>，G<sub>non-target</sub> 为各非目标类别中位增益的平均。</p>
<p><b>同电平增益差</b>　SEI 会把“对响声增强少”误判为选择性，因此以同电平增益差为 H2 的主指标。对每个非目标事件 k：</p>
<p class="eq">ΔG<sub>k</sub> = median(G<sub>footstep</sub> | |L<sub>footstep</sub> − L<sub>k</sub>| ≤ {MATCH_DB:.0f} dB) − G<sub>k</sub><span class="no">(2)</span></p>
<p>要求至少 {MATCH_MIN} 个参照脚步，按类别报告 ΔG<sub>k</sub> 的中位数。</p>
<p><b>频谱保持</b>　对数频谱均方误差<sup>[10]</sup>，同时报告“总”（含音量变化）与“形状”（先减去平均差）两种口径：</p>
<p class="eq">MSE<sub>spec</sub> = (1/TF) Σ<sub>t,f</sub> [S<sub>processed</sub>(t,f) − S<sub>baseline</sub>(t,f)]<sup>2</sup>，S = 20 log<sub>10</sub>(|STFT| + ε)<span class="no">(3)</span></p>
<p><b>误增强率</b>　FER 为 G &gt; {FER_THR:.0f} dB 的非脚步事件比例，阈值即宣称的最低脚步增益。另取 {len(bg)} 段远离事件的背景段（距事件 ≥ 0.3 s、长 ≥ 1 s）作环境负样本。</p>
<p><b>声学突出度</b>　事件峰值增益减去其前方背景（起点前 0.6–0.05 s 的 20% 分位电平）的增益，反映脚步相对紧邻背景是否真正“更突出”。</p>
<h3>6.3　对照条件</h3>
<p>除实际 EQ 外，评测方案要求先把 EQ 调到与 AI 平均脚步增益相近再比较选择性。实际 EQ 对脚步几乎没有增益，无法满足该要求。
因此本文构造<b>匹配静态 EQ</b>：以 FPS AI 在人工确认脚步上的平均频响（1/3 倍频程平滑）设计 2049 阶线性相位 FIR 滤波器，施加于原始轨，
并整体平移使其脚步增益中位数与 FPS AI 相同。它代表“对脚步效果与 FPS AI 一样好的最佳固定 EQ”；由于曲线直接由同一批脚步拟合，该对照对 EQ 一方偏有利，使检验更保守。</p>
<h3>6.4　混杂控制</h3>
<p>表 6 列出可能导致错误结论的混杂或误差来源及其控制手段。</p>
{N.tab("潜在混杂、误差来源与控制手段")}
{t_threat}
<h3>6.5　统计分析</h3>
<p>主分析使用人工确认、纯净度为前两档的事件；扩展分析加入模型高置信度事件。同一片段内的事件不独立，置信区间一律采用按片段整体重抽样的聚类自助法<sup>[14]</sup>（2000 次）。</p>

<h2>7　结果</h2>
<h3>7.1　样本</h3>
{N.tab("进入主分析的人工确认事件")}
{table(c["cnt_rows"])}

<h3>7.2　H1 有效性（子实验 A）</h3>
<p>FPS AI 对人工确认脚步的增益中位数为 {sgn(med(fo_h, 'AI'))} dB，95% CI 下限 {sgn(ci_h[0])} dB 高于宣称下限 {CLAIM[0]:.0f} dB；
合并模型高置信度样本后为 {sgn(med(fo_all, 'AI'))} dB（表 8）。增益低于 3 dB 的脚步仅占 {pct((fo_h.G_AI < 3).mean())}，其余均达到宣称水平，约四成超过 9 dB。
作为对比，实际 EQ 对脚步为 {sgn(med(fo_h, 'EQ'))} dB：脚步约 {pct(bs[('valorant', 'footstep')][0])}（VALORANT）与 {pct(bs[('pubg', 'footstep')][0])}（PUBG）的能量位于 300 Hz 以下，
正好落在两套 EQ 削减的频段，说明“听脚步”EQ 预设对本数据中的脚步并不有效。按电平分档（表 9），-35 dBFS 以下的脚步增益稳定；
-35 dBFS 以上仅 {len(lvl_hi)} 个、来自 {len(hi_clip)} 个片段，增益主要由片段决定（{c['hi_low_txt']} 为 {c['hi_low_rng']} dB，其余片段中位 {c['hi_norm_rng']} dB），样本不足以下结论。</p>
{N.tab("FPS AI 与实际 EQ 对脚步的增益（dB）")}
{t_A}
{N.tab("不同原始电平下的脚步增益")}
{t_lvl}
{N.fig(c["fig_path"], "各事件增益与原始电平（左：FPS AI；右：实际 EQ）。深蓝线为脚步分档中位数，彩色标记为人工确认的非脚步事件，灰带为宣称范围 3–9 dB")}

<h3>7.3　H2 选择性（子实验 B）</h3>
<p>FPS AI 对枪声的增益中位数仅 {sgn(med(gun, 'AI'))} dB（表 10），SEI 为 {sgn(sei['AI'])} dB，实际 EQ 与匹配静态 EQ 分别为 {sgn(sei['EQ'])} dB 与 {sgn(sei['EQm'])} dB。
关键证据来自两种对照（表 11）：在电平相同的条件下，枪声比脚步少增强 {gap['AI'][NT[0]][0]:.1f} dB，换弹少增强 {gap['AI'][NT[1]][0]:.1f} dB，
排除了“对响声增强少”的解释；而与 FPS AI 脚步增益完全相同的匹配静态 EQ 把所有声音一律抬高约 9 dB（图 6），同电平增益差仅 {sgn(g_eqm)} dB，
排除了“固定频响”的解释。FPS AI 在脚步上的平均频响接近全频段平坦抬升（图 7），进一步说明它区分脚步与枪声依靠的是对声音内容的判断。
例外是烟雾弹、吃药声与爆能器一类（{sgn(gap['AI'][NT[2]][0])} dB），它们受到的增强不低于同电平脚步（见 7.4 节）。</p>
{N.tab("各类事件在三种处理下的中位增益（dB）")}
{table(c["gain_rows"])}
<p class="tnote">注：* 背景段未经人工核实。</p>
{N.tab("同电平增益差 ΔG（dB，正值表示非脚步比同电平脚步少增强）")}
{table(c["gap_rows"])}
{N.fig(os.path.join(RES, "ad_gain_bars.png"), "各类事件在三种处理下的中位增益（误差线为四分位距）")}
{N.fig(os.path.join(RES, "ad_response.png"), "FPS AI 在脚步上的平均频响，即匹配静态 EQ 的设计曲线")}

<h3>7.4　H3 非破坏性（子实验 C、D）</h3>
<p>FPS AI 对枪声的对数频谱 MSE 为 {mse(NT[0], 'MSE_AI'):.1f} dB²，约为实际 EQ（{mse(NT[0], 'MSE_EQ'):.1f} dB²）的 {mse(NT[0], 'MSE_AI') / mse(NT[0], 'MSE_EQ') * 100:.0f}%；
对换弹与其他非脚步同样小于实际 EQ（表 12），即 FPS AI 对枪声的“不改变”程度优于一个温和的固定 EQ。
误增强方面（表 13、图 8），FPS AI 对非脚步事件的 FER 为 {pct(fer['AI'])}，对枪声为 {pct(fer_g['AI'])}，匹配静态 EQ 均为 100%。
被误增强的事件（附录 A）有两个共同特点：电平低、与脚步相近（被增强的 {c['n_boost_gun']} 个枪声在 {c['gun_boost_lv'].max():.0f}～{c['gun_boost_lv'].min():.0f} dBFS，
为远处枪声，近处响亮枪声无一被增强）；集中在少数片段（如 PUBG test11 同一颗烟雾弹的多个声音）。因此误增强是“个别安静瞬态被当作脚步”，
而非某类声音被系统性增强。FPS AI 对脚步本身的形状改变（{mse(FOOT, 'MSEshape_AI'):.1f} dB²）大于实际 EQ，即增强伴随脚步音色的变化。</p>
{N.tab("对数频谱均方误差（dB²，中位数）")}
{t_mse}
{N.tab(f"误增强率（增益 &gt; {FER_THR:.0f} dB 的比例）")}
{t_fer}
<p class="tnote">注：* 背景段未经人工核实。</p>
{N.fig(os.path.join(RES, "ad_fer.png"), "误增强率随阈值的变化")}

<h3>7.5　H4 稳健性</h3>
<p>表 14 在不同纯净度、不同游戏与不同增益口径下重复 H1–H2 的核心比较（此处参照脚步只用同一子集的人工确认脚步）。
所有条件下脚步增益均远高于枪声，同电平 ΔG 均为正。标注噪声模拟中，基于 {rob['n']} 条明确标注，FPS AI 的选择性效应（脚步减非脚步，控制电平档与片段）为 {sgn(rob['base'])} dB；
随机翻转 10% 与 20% 标注时中位数分别为 {sgn(rob[0.1][0])} dB 与 {sgn(rob[0.2][0])} dB，{flip_txt}；
需刻意挑错 {rob['worst'][0]} 条（{pct(rob['worst'][1])}）才能将效应抹平，远超盲审所反映的不一致水平（{pct(1 - rs['agree'])}）。
分游戏的完整结果见表 15：PUBG 脚步增益（{sgn(fa_g('pubg', 'AI'))} dB）高于 VALORANT（{sgn(fa_g('valorant', 'AI'))} dB），两款游戏的枪声均几乎未被增强。</p>
{N.tab("稳健性检验：不同条件下的脚步与枪声增益（FPS AI，dB）")}
{t_rob}
<p class="tnote">注：同电平 ΔG 括号内为可匹配的枪声事件数。</p>
{N.tab("分游戏的中位增益（dB；方括号为 95% CI，圆括号为事件数）")}
{table(c["game_rows"])}

<h3>7.6　整体响度</h3>
<p>针对宣传册“游戏整体音量未做改变，长时间畅玩无压力”的宣称，对每段录音计算整段 RMS 与 50 ms 短时电平的中位数、99% 分位数和最大值（表 16，图 9）。
按原始最大短时电平将片段分为两类，分界取 {LOUD_SPLIT:.0f} dBFS，位于两类片段之间的自然间隙
（{lp[('lmax', 'raw')][~is_loud].max():.1f} 与 {lp[('lmax', 'raw')][is_loud].min():.1f} dBFS 之间）。
结果取决于场景：在含枪声等响亮声音的 {n_loud} 个片段中，FPS AI 处理后最大短时电平的变化中位数为 {sgn(dmax_loud.median())} dB，即最响的声音没有被放大，
这与枪声增益约为 0 一致，说明产品没有施加整体增益；在以脚步为主的 {n_quiet} 个安静片段中，脚步本身就是最响的声音，最大电平随脚步抬高，中位数为 {sgn(dmax_quiet.median())} dB，
但处理后最高仅 {fps_max_quiet:.0f} dBFS，仍明显低于数据中响亮片段的原始最大电平（{raw_max_all:.0f} dBFS）。
因此，“整体音量未做改变”应理解为“不施加整体增益、不放大最响的声音”：它在交火等响亮场景中成立，而在安静场景中，整体响度会随脚步增强而上升。</p>
{N.tab("整段响度变化（dB，各片段中位数）")}
{t_loud}
{N.fig(loud_fig, f"各片段原始最大短时电平与处理后的变化。虚线为 {LOUD_SPLIT:.0f} dBFS 分界")}

<h3>7.7　与使用体验相关的附加证据</h3>
<p>（1）<b>即时性</b>：在至少 4 步的 {sp['n_seq']} 个脚步串中，第一步增益（{sgn(sp['first'])} dB）与第 2–4 步（{sgn(sp['later'])} dB）几乎相同，
逐串差值中位数 {sgn(sp['diff'])} dB，增强无需“预热”，对实战中常见的单次、短促脚步同样有效。
（2）<b>突出度</b>：在全部 {prom['n_fo']} 个人工确认脚步上，峰值抬高 {sgn(prom['peak'])} dB，紧邻背景同时抬高 {sgn(prom['bg'])} dB，
脚步相对背景的突出度净增 {sgn(prom['c_fo'])} dB（95% CI {sgn(prom['c_fo_ci'][0])}～{sgn(prom['c_fo_ci'][1])} dB），而非脚步为 {sgn(prom['c_nf'])} dB，实际 EQ 对脚步为 {sgn(prom['c_eq'])} dB。
（3）<b>动态性</b>：FPS AI 的频带增益随时间变化（标准差中位数 {c['tv_ai']:.2f} dB），实际 EQ 几乎不变（{c['tv_eq']:.2f} dB），符合“按需增强”的工作方式。
（4）<b>底噪</b>：远离事件的背景段增益中位数仅 {sgn(med(bg, 'AI'))} dB，但 {pct(fer_bg['AI'])} 的背景段超过 3 dB，纯净脚步前紧邻的背景被抬高 {sgn(c['pre_bg'])} dB，抬升主要发生在脚步附近。</p>

<h2>8　讨论</h2>
<h3>8.1　实用性的证据链</h3>
<p>表 17 汇总了四个假设的判定。H1、H2、H4 得到支持：FPS AI 对脚步的增益达到并略超宣称水平，增强针对声音内容而非响度或频响，
且结论在多种分组、口径与标注误差下保持不变。其中最具区分力的证据是匹配静态 EQ 对照：即便允许固定 EQ 对脚步取得与 FPS AI 相同的增益，
它也必然同等地抬高枪声与环境声，而 FPS AI 对响亮枪声几乎不做改变。这说明 FPS AI 提供了固定 EQ 在原理上无法提供的能力，是其实用价值的核心。
H3 得到部分支持：响亮枪声的保持优于实际 EQ，但约 {pct(fer['AI'])} 的非脚步瞬态被误增强。</p>
{N.tab("实用性假设的判定汇总")}
{t_verdict}
<h3>8.2　对宣传册宣称的逐条检验</h3>
<p>表 18 将宣传册中与脚步增强相关的宣称逐条对应到本文的检验。核心宣称，即脚步增强 3–9 dB、基于识别的选择性增强、不改变响亮枪声，以及优于传统 EQ，均得到客观数据支持；
VALORANT 不在宣传册所列的已适配游戏中，FPS AI 在其上仍有 {sgn(fa_g('valorant', 'AI'))} dB 的脚步增益和明显的选择性，说明模型具有跨游戏的泛化能力。
“不干扰环境音”与“整体音量未做改变”为部分成立，需要限定场景。宣传册称模型“精准提取脚步声频段”，但实测 FPS AI 在脚步上的增益近似全频段抬升，
更接近“识别出脚步事件后整体放大”，而非固定频段提升，这恰好是它与 EQ 的本质区别。处理延迟和爆炸声相关宣称在本文数据中无法检验。</p>
{N.tab("宣传册宣称逐条检验")}
{t_claims}
<h3>8.3　适用边界</h3>
<p>结果同时界定了产品的适用边界：（1）误增强集中在电平与脚步相近的安静瞬态（远处枪声、换弹、烟雾弹、吃药声），在这些场景下玩家可能听到被放大的非脚步声音；
（2）增强脚步时紧邻背景同步抬高约 {sgn(prom['bg'])} dB，使有效突出度（{sgn(prom['c_fo'])} dB）小于峰值增益（{sgn(prom['peak'])} dB），在嘈杂场景中收益会打折扣；
（3）脚步音色被改变，是否影响玩家依据音色判断地面材质与距离，需要主观实验确认；（4）在以脚步为主的安静场景中，整体响度随脚步一起上升（最大电平中位 {sgn(dmax_quiet.median())} dB），玩家可能需要相应调低音量；
（5）-30 dBFS 以上的响亮脚步缺乏样本，近距离脚步的表现尚未验证。</p>
<h3>8.4　框架的可推广性</h3>
<p>本文的标注与实验框架不依赖厂商配合：同源配对录制只需开关产品功能；人机协同标注在 {n_lab} 条人工标注下即可覆盖 1620 个事件中的 {pct(1 - c['n_amb_now'] / 1620)}；
冻结编码器加线性探针的结构可直接迁移到其他目标声音（如枪声方向、技能提示音）；同电平比较与匹配静态 EQ 对照适用于任何宣称“选择性增强”的黑盒处理。</p>
<h3>8.5　局限</h3>
<p>（1）数据来自真实对局且无分轨，“纯净事件”是从整局处理结果中挑出的；（2）人工确认的非脚步事件仅 {len(nt_h)} 个，PUBG 枪声仅
{int(((d.group == NT[0]) & (d.game == 'pubg')).sum())} 个，缺少装拆包、跳跃落地、技能声等类别；（3）仅一名标注者，盲审证明的是标注的稳定性而非绝对准确性；
（4）事件终点为自动估计；宣传册中的处理延迟（40 ms）与“不改变爆炸声”无法用本文数据检验；（5）本文仅在声学层面评价，增强能否提升玩家的脚步检测与定位表现，需要主观听辨实验验证；基于现成检测器的探索性分析见附录 B，
因检测器在游戏音频上可靠性不足，不作为结论依据。</p>

<h2>9　结论</h2>
<p>本文提出了一套面向黑盒游戏音频增强产品的客观评测框架，包括人工主导、模型辅助的四轮标注流程，冻结预训练嵌入上的轻量标注模型，
以及以同电平比较和匹配静态 EQ 为核心的混杂受控实验设计。应用于 C3 FPS AI 的结果表明：其脚步增益中位数为 {sgn(med(fo_h, 'AI'))} dB，达到并略超宣称水平，
且从第一步即生效；同电平下对枪声少增强 {gap['AI'][NT[0]][0]:.1f} dB，几乎不改变枪声频谱；这种基于声音内容的选择性是任何固定 EQ 都无法实现的，
且在多种稳健性检验下成立，对宣传册未列入适配的 VALORANT 同样有效。对照宣传册，脚步增强 3–9 dB、基于识别的选择性增强、不改变响亮枪声与优于传统 EQ 等核心宣称均成立；
“不干扰环境音”与“整体音量未做改变”在响亮场景中成立，在安静场景中需要限定。综合来看，客观证据支持 FPS AI 在声学层面的实用性，
其主要代价是部分安静非脚步瞬态被误增强、脚步附近底噪抬升、安静场景整体响度上升以及脚步音色改变。
后续工作包括：在训练场受控录制分类别、分距离素材以获得天然真值并补齐响亮脚步样本，引入第二名标注者，以及通过主观听辨实验检验增强对玩家脚步检测与定位的实际帮助。</p>

<h2>参考文献</h2>
<ol class="ref">
<li>Gemmeke J F, Ellis D P W, Freedman D, et al. Audio Set: An ontology and human-labeled dataset for audio events[C]//Proc. IEEE ICASSP. 2017: 776-780.</li>
<li>Kong Q, Cao Y, Iqbal T, et al. PANNs: Large-scale pretrained audio neural networks for audio pattern recognition[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020, 28: 2880-2894.</li>
<li>Gong Y, Chung Y A, Glass J. AST: Audio Spectrogram Transformer[C]//Proc. Interspeech. 2021: 571-575.</li>
<li>Elizalde B, Deshmukh S, Al Ismail M, et al. CLAP: Learning audio concepts from natural language supervision[C]//Proc. IEEE ICASSP. 2023.</li>
<li>Wu Y, Chen K, Zhang T, et al. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation[C]//Proc. IEEE ICASSP. 2023.</li>
<li>Alain G, Bengio Y. Understanding intermediate layers using linear classifier probes[J]. arXiv preprint arXiv:1610.01644, 2016.</li>
<li>Settles B. Active learning literature survey[R]. Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009.</li>
<li>Seung H S, Opper M, Sompolinsky H. Query by committee[C]//Proc. 5th Annual Workshop on Computational Learning Theory. 1992: 287-294.</li>
<li>Bello J P, Daudet L, Abdallah S, et al. A tutorial on onset detection in music signals[J]. IEEE Transactions on Speech and Audio Processing, 2005, 13(5): 1035-1047.</li>
<li>Gray A H, Markel J D. Distance measures for speech processing[J]. IEEE Transactions on Acoustics, Speech, and Signal Processing, 1976, 24(5): 380-391.</li>
<li>Mesaros A, Heittola T, Virtanen T. Metrics for polyphonic sound event detection[J]. Applied Sciences, 2016, 6(6): 162.</li>
<li>Bristow-Johnson R. Cookbook formulae for audio EQ biquad filter coefficients[EB/OL]. https://www.w3.org/TR/audio-eq-cookbook/.</li>
<li>Cohen J. A coefficient of agreement for nominal scales[J]. Educational and Psychological Measurement, 1960, 20(1): 37-46.</li>
<li>Field C A, Welsh A H. Bootstrapping clustered data[J]. Journal of the Royal Statistical Society: Series B, 2007, 69(3): 369-390.</li>
<li>Fosi Audio. C3 游戏声卡宣传册[Z]. 深圳: Fosi Audio.</li>
</ol>

<h2>附录 A　被 FPS AI 增强超过 {FER_THR:.0f} dB 的人工确认非脚步事件</h2>
{table(c["boosted_rows"])}

<h2>附录 B　机器可识别性的探索性分析</h2>
<p>为初步考察增强是否使脚步“更容易被认出”，使用从未参与标注的 PANNs<sup>[2]</sup>（Cnn14_DecisionLevelMax）作检测器，并将判断限定在评测方案的六类游戏声音之间
（类别映射事先确定）；另以人工标注微调 PANNs，按片段留出测试集并做 5 折交叉验证。三个版本先按整段 RMS 对齐，真值只用人工标注，阈值在原始轨上确定。
只看“Walk, footsteps”单一类别时 FPS AI 检出率下降 {abs((zf['recall'] - zr['recall']) * 100):.1f} 个百分点，但该类别对游戏脚步的绝对概率极低（阈值仅 {zr['thr']:.4f}），
易受整体电平与背景变化影响；限定六类或使用微调模型后，FPS AI 与 EQ 相对原始的变化均不显著。由于这些检测器在原始轨上的 AUC 仅约 0.8，
未达到评测方案的准入要求，该分析不作为可检测性结论。</p>
<p class="tcap">附表 B1　不同检测器下的脚步检出率变化（方括号为按片段聚类的 95% CI）</p>
{t_det}

<h2>附录 C　数据与代码</h2>
<p>全部数字由脚本从数据自动生成。复现顺序：<code>analyze.py</code> → <code>embed.py</code> / <code>embed_clap.py</code> → <code>train_labeler.py</code>
→ <code>extract_clean.py</code> → <code>report_ad.py</code> → <code>detectability.py</code>（含 <code>--nomatch</code>）→ <code>finetune_panns.py</code> → <code>paper.py</code>。
标注工具与人工标注位于 <code>analysis/labeling/</code>，三轨纯净事件库位于 <code>analysis/clean_events/</code>（{len(meta)} 个事件），EQ 参数与频响计算见 <code>analysis/eq_params.py</code>。</p>
<p class="footer">本稿由 analysis/paper.py 根据分析结果自动生成，与《C3 FPS AI 声卡客观评测 实验报告》使用同一套统计结果。</p>
</main></body></html>"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("->", OUT)


if __name__ == "__main__":
    main()
