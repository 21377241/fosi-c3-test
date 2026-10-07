# C3 FPS AI 声卡客观评测

对 C3 游戏声卡 FPS AI 脚步增强功能的客观评测代码与报告源文件。评测基于 VALORANT 16 段、PUBG 15 段真实对局录音，每段有原始、FPS AI、实际 EQ 三条同源音轨；
在事件级别测量增益，以人工确认事件为主分析，比较 FPS AI 与产品实际 EQ、匹配静态 EQ 的差异，检验脚步增强（H1）、事件选择性（H2）、非目标声音保持性（H3）与稳定性（H4）。

## 目录

| 路径 | 内容 |
|---|---|
| `analysis/` | 全部分析脚本 |
| `analysis/labeling/` | 四轮人工标注结果、盲审复核结果与标注工具（HTML） |
| `analysis/results/` | 中间分析结果（CSV） |
| `analysis/clean_events/clean_events.csv` | 三轨纯净事件库的元数据 |
| `docs/report_tex/` | 详细版（`main.tex`）与简版（`brief.tex`）报告的 LaTeX 源文件、表格、图表与图表数据 |
| `figures/` | 评测流程图与标注模型结构图原图 |

## 未包含的数据

- `fps_test/`：品牌方提供的原始三轨音频，不公开。
- `analysis/cache/`：AST / CLAP 嵌入缓存与模型权重，可由脚本重新生成。
- `analysis/clean_events/**/*.wav`、`analysis/labeling/snippets/`：由原始音频切出的事件片段。
- 生成的报告 PDF / HTML 与产品宣传册。

因此仓库内的 CSV 与标注可以直接复核统计结果与图表；从音频重新计算需要自备 `fps_test/`，目录结构见 `analysis/common.py`。

## 运行

Python 3.12，依赖见 `requirements.txt`。在 `analysis/` 下按顺序运行：

1. `analyze.py`：事件检测与三轨增益计算
2. `embed.py`、`embed_clap.py`：提取 AST / CLAP 嵌入
3. `train_labeler.py`：训练标注模型并生成各轮候选
4. `extract_clean.py`：导出三轨纯净事件库
5. `report_ad.py`：同电平对比与匹配静态 EQ
6. `detectability.py`、`finetune_panns.py`：附录 B 的探索性分析
7. `report_tex.py`：生成详细版报告的全部表格、图表并编译（需要 xelatex 与 ctex）
8. `report_brief.py`：生成简版报告
