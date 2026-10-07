"""External brief (Executive Report, LaTeX): readable in 1-3 minutes by an ordinary player.

Every number and wording here must already appear in the detailed Technical Report (report_tex.py); the brief only
compresses it. Method details (labelling model, AUC / kappa, SEI, PANNs) stay in the detailed report.
Shares docs/report_tex/ with the detailed report: tables are written as tables/b_*.tex, figures go to figures/.
Output: docs/report_tex/brief.tex and docs/C3 FPS AI 声卡客观评测 报告（简版）.pdf
"""
import os
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import report_final as rf
import report_tex as rt
from report_tex import n, p, tex_table, save, C_AI, C_EQ, C_EQM, C_GRP, OUTDIR, TAB, FIG, DATA, DOCS
from report_final import pct, sgn, FOOT, NT, CLAIM, FER_THR, SHORT
import paper

PDF_OUT = os.path.join(DOCS, "C3 FPS AI 声卡客观评测 报告（简版）.pdf")


def fig_compare(panels, name):
    """panels = [(title, (footstep gain, gunshot gain), title colour)], one subplot each, shared y axis."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.1), sharey=True)
    for ax, (title, vals, col) in zip(axes, panels):
        bars = ax.bar(["脚步", "枪声"], vals, color=[C_GRP[FOOT], C_GRP[NT[0]]], width=.5)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, max(v, 0) + 0.3, f"{v:+.1f} dB", ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=11, color=col, fontweight="bold")
        ax.axhline(0, c="k", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)
    axes[0].set_ylabel("中位增益（dB，相对原始）")
    axes[0].set_ylim(-2, 11.5)
    plt.tight_layout()
    save(fig, name)


def fig_gain_bars(rows, name):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(rows))
    w = 0.26
    for k, (key, lab, col) in enumerate([("EQ", "C3 实际 EQ", C_EQ), ("AI", "C3 FPS AI", C_AI),
                                         ("EQm", "匹配静态 EQ（辅助对照）", C_EQM)]):
        vals = [r[key] for r in rows]
        bb = ax.bar(x + (k - 1) * w, vals, w, color=col, label=lab)
        for rect, v in zip(bb, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, c="k", lw=.7)
    ax.axhspan(CLAIM[0], CLAIM[1], color="#4C9BE8", alpha=.08, zorder=0, label="宣传 3–9 dB")
    ax.set_xticks(x, [f"{r['name']}\n（n={r['n']}）" for r in rows], fontsize=8.5)
    ax.set_ylabel("中位增益（dB，相对原始）")
    ax.set_ylim(-3, 12.5)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    plt.tight_layout()
    save(fig, name)


def main():
    for p_ in [OUTDIR, TAB, FIG, DATA]:
        os.makedirs(p_, exist_ok=True)
    c = rf.main()
    d, bg, ev, clips = c["d"], c["bg"], c["ev"], c["clips"]
    fo_h, nt_h, med, gap, mse = c["fo_h"], c["nt_h"], c["med"], c["gap"], c["mse"]
    gun = d[d.group == NT[0]]
    n_lab = c["n_r1"] + len(c["r2"]) + len(c["r3"]) + len(c["r4"])
    fer = lambda x, k: (x[f"G_{k}"] > FER_THR).mean()

    core = {k: (med(fo_h, k), med(gun, k)) for k in ("EQ", "AI", "EQm")}
    fig_compare([("原始 → C3 实际 EQ", core["EQ"], C_EQ), ("原始 → C3 FPS AI", core["AI"], C_AI)], "figb_core_compare")
    pd.DataFrame({"track": ["C3 实际 EQ", "C3 FPS AI", "匹配静态 EQ（辅助对照）"], "footstep": [v[0] for v in core.values()],
                  "gunshot": [v[1] for v in core.values()]}).to_csv(os.path.join(DATA, "figb_core_compare.csv"), index=False, encoding="utf-8-sig")

    rows_g = []
    for g in [FOOT] + NT:
        x = d[d.group == g]
        rows_g.append(dict(name=SHORT[g].replace("烟雾弹/吃药/爆能器", "道具"), n=len(x), AI=med(x, "AI"), EQ=med(x, "EQ"), EQm=med(x, "EQm")))
    rows_g.append(dict(name="背景段", n=len(bg), AI=med(bg, "AI"), EQ=med(bg, "EQ"), EQm=med(bg, "EQm")))
    fig_gain_bars(rows_g, "figb_gain_bars")
    pd.DataFrame(rows_g).to_csv(os.path.join(DATA, "figb_gain_bars.csv"), index=False, encoding="utf-8-sig")

    sub = lambda g, grp: d[(d.game == g) & (d.group == grp)]
    tex_table("b_game", [{"游戏": nm, "脚步事件数": len(sub(g, FOOT)), "脚步 EQ": sgn(sub(g, FOOT).G_EQ.median()),
                          "脚步 AI": sgn(sub(g, FOOT).G_AI.median()), "枪声事件数": len(sub(g, NT[0])),
                          "枪声 EQ": sgn(sub(g, NT[0]).G_EQ.median()), "枪声 AI": sgn(sub(g, NT[0]).G_AI.median())}
                         for g, nm in [("valorant", "VALORANT"), ("pubg", "PUBG")]],
              "两款目标游戏中的中位增益（dB，人工确认事件）", spec="lcccccc",
              heads=["游戏", r"\makecell{脚步\\事件数}", r"\makecell{脚步\\实际 EQ}", r"\makecell{\textbf{脚步}\\\textbf{FPS AI}}",
                     r"\makecell{枪声\\事件数}", r"\makecell{枪声\\实际 EQ}", r"\makecell{\textbf{枪声}\\\textbf{FPS AI}}"])

    L = paper.loudness_stats()
    lp = L.pivot_table(index=["game", "clip_id"], columns="track", values="lmax")
    loud = lp["raw"] >= paper.LOUD_SPLIT
    d_loud = (lp["fps"] - lp["raw"])[loud].median()
    d_quiet = (lp["fps"] - lp["raw"])[~loud].median()

    pure = d[d.purity == "p"]
    fs = pure[pure.group == FOOT]
    ex_f = int(fs.iloc[(fs.G_AI - fs.G_AI.median()).abs().argsort().iloc[0]].event_id)
    ex_g = int(pure[pure.group == NT[0]].sort_values("raw_level").iloc[-1].event_id)
    di = d.set_index("event_id")
    ex_gains = {e: tuple(di.loc[e, ["G_AI", "G_EQ"]]) for e in (ex_f, ex_g)}
    rt.fig_waveforms([("脚步", "footstep", ex_f), ("枪声", "gunshot", ex_g)], ex_gains, "figb_waveforms")

    V = dict(
        n_vl=str(c["n_vl"]), n_pg=str(c["n_pg"]), n_clip=str(len(clips)), dur=f"{c['dur_min']:.1f}", n_ev=str(len(ev)), n_lab=str(n_lab),
        n_fo=str(len(fo_h)), n_gun=str(len(gun)),
        g_fo=n(med(fo_h, "AI")), g_fo_u=n(med(fo_h, "AI"), sign=False), g_fo_eqm=n(med(fo_h, "EQm")),
        g_gun=n(med(gun, "AI")), g_gun_eqm=n(med(gun, "EQm")), g_bg_eqm=n(med(bg, "EQm")),
        g_val_eq=n(sub("valorant", FOOT).G_EQ.median()), g_pubg_eq=n(sub("pubg", FOOT).G_EQ.median()),
        g_fo_eq=n(med(fo_h, "EQ")), g_gun_eq=n(med(gun, "EQ")), g_ex_f_eq=n(ex_gains[ex_f][1]), g_ex_g_eq=n(ex_gains[ex_g][1]),
        ge3=p((fo_h.G_AI >= 3).mean()),
        gap_gun=n(gap["AI"][NT[0]][0], sign=False), gap_eqm=n(gap["EQm"][NT[0]][0]),
        mse_ai=n(mse(NT[0], "MSE_AI"), sign=False), mse_eq=n(mse(NT[0], "MSE_EQ"), sign=False),
        mse_frac=f"{mse(NT[0], 'MSE_EQ') / mse(NT[0], 'MSE_AI'):.0f}",
        mse_gs_ai=n(mse(NT[0], "MSEshape_AI"), sign=False), mse_gs_eq=n(mse(NT[0], "MSEshape_EQ"), sign=False),
        mse_sfrac=f"{mse(NT[0], 'MSEshape_EQ') / mse(NT[0], 'MSEshape_AI'):.0f}",
        fer_ai=p(fer(nt_h, "AI")), loud_ai=n(d_loud), quiet_ai=n(d_quiet),
        g_ex_f=n(ex_gains[ex_f][0]), amp_ex_f=f"{10 ** (ex_gains[ex_f][0] / 20):.1f}", g_ex_g=n(ex_gains[ex_g][0]),
        ex_f=str(ex_f), ex_g=str(ex_g), n_ut=str(int((d.group == NT[2]).sum())),
    )
    body = TEMPLATE.replace("→", r"$\rightarrow$")
    for k, val in V.items():
        body = body.replace(f"[[{k}]]", val)
    left = re.findall(r"\[\[\w+\]\]", body)
    if left:
        raise ValueError(f"unfilled: {sorted(set(left))}")
    with open(os.path.join(OUTDIR, "brief.tex"), "w", encoding="utf-8") as fh:
        fh.write(body)
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "brief.tex"], cwd=OUTDIR,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("xelatex failed")
    print("->", rt.publish(os.path.join(OUTDIR, "brief.pdf"), PDF_OUT))


PREAMBLE = rt.TEMPLATE.split(r"\begin{document}")[0].replace(
    r"\lhead{\small C3 FPS AI 声卡客观评测技术报告（详细版）}", r"\lhead{\small C3 FPS AI 声卡脚步增强功能 · 客观评测报告}")

TEMPLATE = PREAMBLE + r"""
\renewcommand{\FloatBarrier}{}
\definecolor{ai}{HTML}{1F4E79}\definecolor{gun}{HTML}{D62728}
\newcommand{\tagbox}[1]{\colorbox{ai!10}{\strut\heiti\color{ai}\ #1\ }}
\newcommand{\greybox}[1]{\par\noindent\colorbox{black!7}{\parbox{\dimexpr\linewidth-2\fboxsep}{\small #1}}\par}
\newcommand{\flowbox}[1]{\fcolorbox{ai}{ai!5}{\parbox[c][1.6cm][c]{2.8cm}{\centering\small #1}}}
\newcommand{\bignum}[2]{\parbox[t]{0.235\linewidth}{\centering{\fontsize{22}{26}\selectfont\heiti\color{ai}#1}\\[3pt]{\small #2}}}
\newcommand{\bigval}[1]{{\fontsize{24}{28}\selectfont #1}}

\begin{document}
\begin{titlepage}
\centering\vspace*{45mm}
{\Huge\heiti C3 FPS AI 声卡脚步增强功能\par}\vspace{8pt}
{\huge\heiti 客观评测报告\par}
\vspace{10mm}
{\large 基于 VALORANT 与 PUBG 真实游戏录音的同源三轨对比测试\par}
\vspace{5mm}
{\normalsize 对外简版 · Executive Report\par}
\vspace{35mm}
\begin{tabular}{rl}
评测对象： & Fosi Audio C3 游戏声卡（FPS AI 模式）\\
目标游戏： & VALORANT、PUBG（本阶段计划评测游戏）\\
数据： & [[n_clip]] 段真实对局录音，共 [[dur]] min\\
编制： & [编制部门 / 人员]\\
版本： & v1.0\quad 2026 年 10 月\\
\end{tabular}
\vfill
{\small 本报告为《客观评测技术报告（详细版）》的压缩版本，所有数字与结论均以详细版为准。\par}
\end{titlepage}

\section{结论速览}
\keybox{\textbf{一句话结论：}C3 实际 EQ 在本次测试中并未增强脚步，而 FPS AI 对人工确认脚步的中位增强达到 [[g_fo]] dB，同时枪声中位变化仅 [[g_gun]] dB。}

\begin{center}
\renewcommand{\arraystretch}{1.6}
\begin{tabular}{lcc}
\toprule
模式 & 脚步中位增益 & 枪声中位增益 \\
\midrule
原始 & \bigval{\color{black!45}0 dB} & \bigval{\color{black!45}0 dB} \\
C3 实际 EQ & \bigval{[[g_fo_eq]] dB} & \bigval{[[g_gun_eq]] dB} \\
\textbf{\color{ai}C3 FPS AI} & \bigval{\color{ai}\textbf{[[g_fo]] dB}} & \bigval{\color{ai}[[g_gun]] dB} \\
\bottomrule
\end{tabular}

\vspace{4mm}
\tagbox{脚步更突出}\quad\tagbox{枪声基本保持}\quad\tagbox{固定 EQ 难以复制这种选择性}
\end{center}

两种都是用户在 C3 上实际可用的模式。脚步增强幅度：中位 [[g_fo]] dB，[[ge3]] $\geq$ 3 dB，整体位于宣传区间（3--9 dB）上沿，部分样本超过 9 dB。

为进一步检验“普通固定 EQ 是否只要调大一点也能做到类似效果”，技术报告额外构造了一个匹配静态 EQ（并非 C3 自带模式），使其脚步增益达到与 FPS AI 相同的 [[g_fo_eqm]] dB；
此时枪声也被同步提升约 [[g_gun_eqm]] dB（见第~\ref{sec:core} 节）。

\vspace{3mm}
\greybox{\textbf{边界说明：}约 [[fer_ai]] 的非脚步瞬态声增益也超过 3 dB，主要是远处的安静枪声、换弹和部分道具声；本测试只验证声学处理效果，不直接证明定位能力提升。}

\clearpage
\section{实验为什么公平}
同一段游戏声音同时输出为三条轨，三条轨内容完全相同，差别只来自处理方式。因此可以对\textbf{同一个声音、同一时间点}直接比较“处理后比原始响了多少”。

\begin{center}
\flowbox{\textbf{同一段游戏音频}\\[2pt]VALORANT / PUBG}
→
\flowbox{\textbf{三轨同步录制}\\[2pt]Raw / FPS AI / EQ}
→
\flowbox{\textbf{同一个声音}\\[2pt]同一时间点\\比较音量}
→
\flowbox{\textbf{人工盲听}\\[2pt]确认事件类别}
\end{center}

\begin{itemize}
\item \textbf{Raw}：原始声音，不做处理；
\item \textbf{FPS AI}：开启 C3 的 FPS AI 脚步增强；
\item \textbf{EQ}：C3 产品自带的 EQ 预设（下文称“实际 EQ”）。
\end{itemize}
\vspace{2mm}
\keybox{\textbf{增益怎么算？}对同一个声音事件，分别测原始轨和处理轨在同一时间段内的 RMS（均方根幅值）
\[
\mathrm{RMS}(x)=\sqrt{\frac{1}{N}\sum_{n=1}^{N}x_n^2},\qquad
G=20\log_{10}\frac{\mathrm{RMS}_{\text{处理}}}{\mathrm{RMS}_{\text{原始}}}\ \ (\mathrm{dB})
\]
$G>0$ 表示增强，$G=0$ 表示基本不变；$+6$ dB 约对应波形幅值翻倍。报告中的“脚步 [[g_fo]] dB”即人工确认脚步的 $G$ 的中位数。}

\vspace{5mm}
\begin{center}
\bignum{[[n_clip]] 段}{真实对局录音}%
\bignum{[[dur]] min}{总时长}%
\bignum{[[n_ev]]}{自动检出的\\候选声音事件}%
\bignum{[[n_lab]]}{人工盲听标注}
\end{center}
\vspace{3mm}

录音来自品牌方提供的 VALORANT [[n_vl]] 段、PUBG [[n_pg]] 段真实对局，全部使用，未按处理效果筛选。主结果仅使用人工确认且无瞬态重叠的事件，包括 [[n_fo]] 个脚步和 [[n_gun]] 个枪声\footnote{标注采用盲听复核，人机协同模型仅辅助扩展；详细质量控制见技术报告。}。

\clearpage
\section{核心对比：开 C3 实际 EQ 与开 FPS AI，有什么区别？}\label{sec:core}
消费者最直接的问题是：“我直接开 C3 的 EQ，和开 FPS AI，到底有什么区别？”下图比较这两种实际可用的产品模式。

\figslot{figb_core_compare}{0.9\linewidth}{C3 两种实际可用模式对脚步与枪声的影响（人工确认的 [[n_fo]] 个脚步、[[n_gun]] 个枪声）}{core}

\begin{itemize}
\item \textbf{左：C3 实际 EQ}——脚步 [[g_fo_eq]] dB，枪声 [[g_gun_eq]] dB：没有增强脚步；
\item \textbf{右：C3 FPS AI}——脚步 [[g_fo]] dB，枪声仅 [[g_gun]] dB：脚步明显更突出，枪声基本保持。
\end{itemize}
两种模式的作用逻辑完全不同：实际 EQ 是作用于全部声音的固定频响调整，FPS AI 则主要增强脚步。

\figslot{figb_waveforms}{\linewidth}{直观示例：同一个脚步（上）与响亮枪声（下）在三条轨上的波形，同一行纵轴刻度相同}{wave}
FPS AI 轨上脚步波形明显变大（[[g_ex_f]] dB，波形幅值约 ×[[amp_ex_f]]），枪声几乎不变（[[g_ex_g]] dB）；实际 EQ 轨上两者都没有明显变化（[[g_ex_f_eq]] / [[g_ex_g_eq]] dB）。

\keybox{\textbf{“那如果把 EQ 调得更激进呢？”}\\[3pt]
为回答这个技术追问，技术报告额外构造了一个\textbf{匹配静态 EQ}：它不是 C3 自带模式，而是按 FPS AI 的脚步增强曲线构造、并调到与 FPS AI 相同脚步增益的固定 EQ
（这一设定对 EQ 一方偏有利）。
\begin{center}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{lcc}
\toprule
 & 脚步 & 枪声 \\
\midrule
C3 FPS AI & [[g_fo]] dB & [[g_gun]] dB \\
匹配静态 EQ & [[g_fo_eqm]] dB & \textbf{\color{gun}[[g_gun_eqm]] dB} \\
\bottomrule
\end{tabular}
\end{center}
即使固定 EQ 强行做到同等脚步增强，枪声也会被一起抬高（背景声 [[g_bg_eqm]] dB）。为控制“枪声本来比脚步更响”的影响，实验还将每个枪声与原始音量相近（$\pm$3 dB）的脚步进行对照：
FPS AI 下枪声仍比脚步少增强 [[gap_gun]] dB，而匹配静态 EQ 下这一差值仅 [[gap_eqm]] dB。这种差异无法由固定静态频响解释，表现出明显的事件选择性处理特征。}

\section{真实游戏结果：两款目标游戏中均观察到选择性增强}
\input{tables/b_game}
两款游戏中实际 EQ 均未明显增强脚步（[[g_val_eq]] / [[g_pubg_eq]] dB）；FPS AI 的增强幅度不同，但都呈现“脚步明显增强、枪声中位变化较小”的总体模式。

\section{更多证据与已知边界}
\subsection{不仅是音量：FPS AI 对枪声音色结构的改变也明显低于实际 EQ}
\begin{center}
{\fontsize{22}{26}\selectfont\heiti\color{ai}枪声频谱变化 $\approx$ 实际 EQ 的 1/[[mse_frac]]}\\[4pt]
{\small 频谱总变化：FPS AI [[mse_ai]] dB$^2$ vs 实际 EQ [[mse_eq]] dB$^2$\\
扣除整体音量、只看频谱形状：约 1/[[mse_sfrac]]（[[mse_gs_ai]] vs [[mse_gs_eq]] dB$^2$）}
\end{center}
在含响亮枪声的片段中，整段最大音量中位只变化 [[loud_ai]] dB：最响的声音没有被一起推高。

\figslot{figb_gain_bars}{0.92\linewidth}{各类声音在三种处理下的中位增益（浅蓝带为宣传的 3--9 dB）}{bars}
FPS AI 主要增强脚步，对响亮枪声影响较小；道具类声音（烟雾弹、吃药、爆能器，仅 [[n_ut]] 个）也会被增强。

\subsection{已知边界}
\greybox{\begin{itemize}
\item 约 [[fer_ai]] 的非脚步瞬态声增益超过 3 dB，主要是远处的安静枪声、换弹和部分道具声，它们与脚步一样安静、短促；
\item 脚步附近的背景声会随脚步同步抬升；以脚步为主的安静片段，整段最大音量中位上升 [[quiet_ai]] dB；
\item 本测试只验证声学处理效果，不直接证明玩家定位能力提升；
\item 结论限于 VALORANT 与 PUBG 两款目标游戏。
\end{itemize}}

\section{结论}
\keybox{本次测试表明，在 VALORANT 与 PUBG 两款目标游戏中，C3 实际 EQ 未增强脚步，而 C3 FPS AI 表现出明显的脚步选择性增强：脚步获得显著增益，响亮枪声整体保持稳定；
进一步的技术对照表明，这一差异无法由具有相同脚步增益的固定静态 EQ 复现。}
\vspace{3mm}

实验支持的是声学处理效果。玩家是否因此更早发现敌人、提升定位准确率，仍需进一步主观听辨实验验证。

\vspace{6mm}
{\small 方法细节、标注质量控制、统计检验与完整数据见《C3 FPS AI 声卡脚步增强功能 客观评测技术报告（详细版）》。\par}
\end{document}
"""


if __name__ == "__main__":
    main()
