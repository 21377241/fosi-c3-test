"""Raw-only blind re-review of every human-confirmed non-footstep event in the gain analysis (before the
boosted-event review), mixed with randomly drawn human footsteps as decoys, in a fixed shuffled order.

The page shows the raw track (event segment + ±CTX s context) and an opaque index; category, gains and
event ids stay out of the HTML. The index -> event key is written to labeling/blind_r5_key.csv.
The FPS AI version of a row can be opened only after the raw-only verdict is given, which then locks; a second
verdict after hearing FPS AI is recorded separately. The raw-only verdict is the primary (blind) result.

  python blind_review.py                  -> 原始轨盲审.html + labeling/blind_r5_key.csv
  python blind_review.py analyze <csv>    -> statistics from the exported review CSV
"""
import os
import sys
import html
import numpy as np
import pandas as pd
import soundfile as sf
from common import load_triplet, SR
from report_final import RES, HERE, LAB, FOOT, NT, FER_THR
from listen_boosted import mp3_uri

OUT = os.path.join(HERE, "原始轨盲审.html")
KEY = os.path.join(LAB, "blind_r5_key.csv")
CTX = 1.5
N_DECOY = 25
SEED = 20261010
VERDICTS = ["未审核", "无脚步", "含脚步", "无法判断"]
STORE = "fosi_blind_r5_v1"


def build():
    d = pd.read_csv(os.path.join(RES, "report_AD_events.csv"), encoding="utf-8-sig")
    meta = pd.read_csv(os.path.join(HERE, "clean_events", "clean_events.csv"), encoding="utf-8-sig", keep_default_na=False)
    rng = np.random.default_rng(SEED)
    nt = d[d.group.isin(NT)]
    fo = d[d.group == FOOT]
    decoy = fo.iloc[rng.choice(len(fo), N_DECOY, replace=False)]
    ev = pd.concat([nt, decoy]).drop(columns="game").merge(meta[["event_id", "cls", "game", "clip_id", "t"]], on="event_id")
    ev = ev.iloc[rng.permutation(len(ev))].reset_index(drop=True)
    ev["idx"] = [f"B{k + 1:03d}" for k in range(len(ev))]
    ev[["idx", "event_id", "group"]].to_csv(KEY, index=False, encoding="utf-8-sig")

    clips, rows = {}, []
    opts = "".join(f"<option>{o}</option>" for o in VERDICTS)
    player = lambda src: f'<audio controls preload="none" src="{src}"></audio>'
    for r in ev.itertuples():
        key = (r.game, r.clip_id)
        if key not in clips:
            clips[key] = load_triplet(*key)[0]
        a = clips[key]
        s, e = max(0, int((r.t - CTX) * SR)), min(len(a["raw"]), int((r.t + CTX) * SR))
        ctx = {t: mp3_uri(a[t][s:e]) for t in ("raw", "fps")}
        seg = {t: mp3_uri(*sf.read(os.path.join(HERE, "clean_events", r.cls, f"e{r.event_id:04d}_{t}.wav"))) for t in ("raw", "fps")}
        rows.append(f'<tr data-idx="{r.idx}"><td>{r.idx}</td>'
                    f'<td><div class="lab">上下文</div>{player(ctx["raw"])}<div class="lab">事件</div>{player(seg["raw"])}</td>'
                    f'<td class="rv"><select class="v">{opts}</select></td>'
                    f'<td class="fp"><button class="open" disabled>先完成左侧判断</button>'
                    f'<div class="fps" hidden><div class="lab">上下文</div>{player(ctx["fps"])}<div class="lab">事件</div>{player(seg["fps"])}</div></td>'
                    f'<td class="rv"><select class="v2" disabled>{opts}</select>'
                    f'<textarea class="c" rows="2" placeholder="备注（可选）"></textarea></td></tr>')

    page = f"""<!doctype html><meta charset="utf-8"><title>原始轨盲审</title>
<style>body{{font:14px "Microsoft YaHei",sans-serif;margin:20px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:4px 8px;vertical-align:top}}th{{background:#f3f3f3}}audio{{width:240px;height:30px;display:block}}
.lab{{font-size:12px;color:#888;margin-top:2px}}td.rv{{min-width:170px}}td.rv select{{width:100%;margin-bottom:3px}}
td.rv textarea{{width:100%;box-sizing:border-box;font:13px inherit}}td.fp{{min-width:250px;background:#fafafa}}
tr.done td:first-child{{background:#d9f2dc}}tr.done td.rv:nth-child(3){{background:#f1faf1}}
#bar{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd;z-index:1}}#bar button{{margin-right:10px}}</style>
<h2>原始轨盲审（共 {len(ev)} 个声音事件）</h2>
<p>每行是一个声音事件。“上下文”为事件前后各 {CTX} s 的完整录音（事件位于正中），“事件”为分析使用的片段。顺序已随机打乱，不提供类别信息。</p>
<p>请判断<b>这个事件里有没有脚步声</b>：<b>无脚步</b>＝只有非脚步声音（枪声、换弹、道具、技能等）；<b>含脚步</b>＝事件本身是脚步，或脚步与其他声音重叠；<b>无法判断</b>＝听不清或拿不准。</p>
<p><b>步骤：</b>① 先只听<b>原始</b>录音并给出判断；② 判断后才能打开该行的 <b>FPS AI 版本</b>，打开后第一次判断即锁定、不能修改；③ 听完 FPS AI 版本后可另填“听 FPS 后的判断”（可选）。
本审核以第一次判断为主结果，请不要为了“确认”而先打开 FPS AI。</p>
<p>建议戴耳机、固定音量；请独立判断，不要参考其他页面或报告。内容自动保存在本浏览器中，完成后点“导出 CSV”并发回。</p>
<div id="bar"><button id="exp">导出 CSV</button><span id="prog"></span></div>
<table><tr><th>序号</th><th>① 原始</th><th>① 判断（原始）</th><th>② FPS AI 版本</th><th>③ 听 FPS 后的判断 / 备注</th></tr>{"".join(rows)}</table>
<script>
const KEY = "{STORE}";
const load = () => JSON.parse(localStorage.getItem(KEY) || "{{}}");
const put = (id, patch) => {{ const st = load(); st[id] = {{...(st[id] || {{}}), ...patch}}; localStorage.setItem(KEY, JSON.stringify(st)); }};
const rows = [...document.querySelectorAll("tr[data-idx]")];
function refresh() {{
  const st = load(); let n = 0, k = 0;
  rows.forEach(r => {{ const s = st[r.dataset.idx], done = !!s && !!s.v && s.v !== "未审核"; r.classList.toggle("done", done); n += done; k += !!(s && s.open); }});
  document.getElementById("prog").textContent = `已完成 ${{n}} / ${{rows.length}}（其中打开过 FPS AI 的 ${{k}} 个）`;
}}
rows.forEach(r => {{
  const id = r.dataset.idx, v = r.querySelector(".v"), v2 = r.querySelector(".v2"), c = r.querySelector(".c");
  const btn = r.querySelector(".open"), box = r.querySelector(".fps"), s = load()[id] || {{}};
  const reveal = () => {{ v.disabled = true; btn.hidden = true; box.hidden = false; v2.disabled = false; }};
  v.value = s.v || "未审核"; v2.value = s.v2 || "未审核"; c.value = s.c || "";
  const arm = () => {{ const ok = v.value !== "未审核"; btn.disabled = !ok; btn.textContent = ok ? "打开 FPS AI（锁定左侧判断）" : "先完成左侧判断"; }};
  arm();
  if (s.open) reveal();
  v.addEventListener("change", () => {{ put(id, {{v: v.value}}); arm(); refresh(); }});
  btn.addEventListener("click", () => {{ if (v.value === "未审核") return; put(id, {{v: v.value, open: true}}); reveal(); refresh(); }});
  v2.addEventListener("change", () => put(id, {{v2: v2.value}}));
  c.addEventListener("input", () => put(id, {{c: c.value}}));
}});
document.getElementById("exp").onclick = () => {{
  const q = x => '"' + String(x).replace(/"/g, '""') + '"';
  const lines = [["序号", "判断_原始轨", "是否打开FPS", "判断_听FPS后", "备注"].join(",")];
  rows.forEach(r => {{ const s = load()[r.dataset.idx] || {{}};
    lines.push([r.dataset.idx, q(r.querySelector(".v").value), s.open ? "是" : "否", q(s.open ? r.querySelector(".v2").value : ""), q(r.querySelector(".c").value)].join(",")); }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\\ufeff" + lines.join("\\n")], {{type: "text/csv"}}));
  a.download = "blind_review.csv"; a.click();
}};
refresh();
</script>"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"{len(ev)} events ({len(nt)} non-footstep + {N_DECOY} decoy footsteps) -> {OUT} ({os.path.getsize(OUT) / 2**20:.1f} MB)")


def analyze(path):
    d = pd.read_csv(os.path.join(RES, "report_AD_events.csv"), encoding="utf-8-sig")
    key = pd.read_csv(KEY, encoding="utf-8-sig")
    rv = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False).rename(columns={"序号": "idx", "判断_原始轨": "判断"})
    x = key.merge(rv, on="idx").merge(d[["event_id", "G_AI", "G_EQ", "G_EQm"]], on="event_id")
    if "判断_听FPS后" not in x:
        x["是否打开FPS"], x["判断_听FPS后"] = "否", ""
    x["判断_最终"] = np.where((x["是否打开FPS"] == "是") & ~x["判断_听FPS后"].isin(["", "未审核"]), x["判断_听FPS后"], x["判断"])
    todo = int((x["判断"] == "未审核").sum())
    if todo:
        print(f"warning: {todo} events not reviewed")
    nt, fo = x[x.group.isin(NT)], x[x.group == FOOT]
    print("non-footstep verdicts:", nt["判断"].value_counts().to_dict())
    print("decoy footstep verdicts:", fo["判断"].value_counts().to_dict(),
          f"-> sensitivity {(fo['判断'] == '含脚步').mean():.0%}")
    for col, lab in [("判断", "raw-only (primary, blind)"), ("判断_最终", "after hearing FPS AI where opened (secondary)")]:
        clean = nt[nt[col] == "无脚步"]
        print(f"\nFER, {lab}:")
        for p in ["AI", "EQ", "EQm"]:
            k = int((clean[f"G_{p}"] > FER_THR).sum())
            print(f"  {p}: {k}/{len(clean)} = {k / len(clean):.1%}  (before review: {(nt[f'G_{p}'] > FER_THR).mean():.1%})")
    op = x[x["是否打开FPS"] == "是"]
    print(f"\nFPS AI opened for {len(op)} / {len(x)} events "
          f"({int((op.G_AI > FER_THR).sum())} of them boosted > {FER_THR:.0f} dB); verdict change after hearing FPS AI:")
    if len(op):
        print(pd.crosstab(op["判断"], op["判断_最终"]).to_string())
    brv = pd.read_csv(os.path.join(LAB, "boosted_review.csv"), encoding="utf-8-sig", keep_default_na=False)
    cmp_ = brv.merge(x[["event_id", "判断"]], on="event_id")
    print("\nboosted-event review vs blind review:")
    print(pd.crosstab(cmp_["是否误增强"], cmp_["判断"]).to_string())


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "analyze":
        analyze(sys.argv[2])
    else:
        build()
