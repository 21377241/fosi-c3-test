"""Summarise labeling/review_r1.csv: test-retest agreement between the original and the blind
re-label, what changed, and the purity grades of the final labels."""
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"f": "脚步", "g": "枪声", "o": "其他", "m": "混合", "x": "不确定"}
PURE = {"p": "纯净", "b": "有持续背景", "l": "有重叠"}

r = pd.read_csv(os.path.join(HERE, "labeling", "review_r1.csv"), encoding="utf-8-sig", keep_default_na=False)
r = r[r.final_label != ""]
print(f"已审阅 {len(r)} 条\n")

same = (r.orig_label == r.blind_label)
print(f"盲判与原标注一致: {same.sum()}/{len(r)} = {same.mean():.0%},  κ(5 类) = "
      f"{cohen_kappa_score(r.orig_label, r.blind_label):.2f}")
b = r[r.orig_label.isin(["f", "g", "o"]) & r.blind_label.isin(["f", "g", "o"])]
print(f"脚步 vs 非脚步（双方都给了明确类别的 {len(b)} 条）: 一致 {((b.orig_label == 'f') == (b.blind_label == 'f')).mean():.0%},  "
      f"κ = {cohen_kappa_score(b.orig_label == 'f', b.blind_label == 'f'):.2f}")
print("（κ ≥ 0.8 可认为标注稳定，0.6–0.8 尚可，< 0.6 需要修订标注规则）\n")

print("原标注（行） × 盲判（列）:")
print(pd.crosstab(r.orig_label.map(NAMES), r.blind_label.map(NAMES)).to_string(), "\n")

ch = r[r.final_label != r.orig_label]
print(f"最终改判 {len(ch)} 条:")
for x in ch.itertuples():
    print(f"  #{x.event_id} {x.game} test{x.clip_id} t={x.t:.2f}: {NAMES[x.orig_label]}({x.orig_note}) -> "
          f"{NAMES[x.final_label]}({x.final_note})")

print("\n最终标签 × 纯净度:")
print(pd.crosstab(r.final_label.map(NAMES), r.purity.map(PURE), margins=True, margins_name="合计").to_string())
