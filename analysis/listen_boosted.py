"""Listening page for the falsely boosted events (human-confirmed non-footsteps with FPS AI gain > FER_THR).

For every event: the event segment (clean_events/) and a ±CTX s context cut from the aligned full clips,
each as raw / FPS AI / actual EQ. A typical footstep is added as reference.
Output: a single self-contained 误增强事件试听.html (audio embedded as 320 kbps MP3 data URIs). Needs fps_test/.
"""
import io
import os
import html
import json
import base64
import numpy as np
import pandas as pd
import soundfile as sf
from common import load_triplet, SR
from report_final import RES, HERE, FOOT, NT, FER_THR, SHORT

OUT = os.path.join(HERE, "误增强事件试听.html")
CTX = 1.5
TRACKS = [("raw", "原始"), ("fps", "FPS AI"), ("eq", "实际 EQ")]
VERDICTS = ["未审核", "是误增强", "不是误增强", "无法判断"]
STORE = "fosi_boosted_review_v2"
OLD_STORE = "fosi_boosted_review_v1"
OLD_MAP = {1: "是误增强", 2: "不是误增强", 3: "不是误增强", 4: "无法判断"}


def mp3_uri(x, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), sr, format="MP3", bitrate_mode="CONSTANT", compression_level=0)
    return "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    d = pd.read_csv(os.path.join(RES, "report_AD_events.csv"), encoding="utf-8-sig")
    meta = pd.read_csv(os.path.join(HERE, "clean_events", "clean_events.csv"), encoding="utf-8-sig", keep_default_na=False)
    ev = d[d.group.isin(NT) & (d.G_AI > FER_THR)].merge(meta, on="event_id", suffixes=("", "_m"))
    ev = ev.sort_values(["group", "game_m", "clip_id", "t"])
    fs = d[(d.group == FOOT) & (d.purity == "p")]
    ref = fs.iloc[(fs.G_AI - fs.G_AI.median()).abs().argsort().iloc[0]]
    ev = pd.concat([ev, d[d.event_id == ref.event_id].merge(meta, on="event_id", suffixes=("", "_m")).assign(ref=True)])

    brv = pd.read_csv(os.path.join(HERE, "labeling", "boosted_review.csv"), encoding="utf-8-sig",
                      keep_default_na=False).set_index("event_id")
    clips, rows = {}, []
    for r in ev.itertuples():
        key = (r.game_m, r.clip_id)
        if key not in clips:
            clips[key] = load_triplet(*key)[0]
        a = clips[key]
        s, e = max(0, int((r.t - CTX) * SR)), min(len(a["raw"]), int((r.t + CTX) * SR))
        ctx = {t: mp3_uri(a[t][s:e]) for t, _ in TRACKS}
        seg = {t: mp3_uri(*sf.read(os.path.join(HERE, "clean_events", r.cls, f"e{r.event_id:04d}_{t}.wav")))
               for t, _ in TRACKS}
        is_ref = getattr(r, "ref", False) is True
        name = "对照：典型脚步" if is_ref else SHORT[r.group]
        player = lambda src: f'<audio controls preload="none" src="{src}"></audio>'
        review = "" if is_ref else (
            f'<select class="v">{"".join(f"<option>{o}</option>" for o in VERDICTS)}</select>'
            '<textarea class="c" rows="2" placeholder="原因（如：吃药声；脚步与爆能器重叠）"></textarea>')
        prior = brv.loc[r.event_id] if r.event_id in brv.index else None
        prior_v, prior_c = ("", "") if prior is None else (prior["是否误增强"], prior["原因备注"])
        prior_td = ("<td>—</td>" if prior is None else
                    f'<td class="pv {"yes" if prior_v == "是误增强" else "no"}"><b>{html.escape(prior_v)}</b><br>{html.escape(prior_c)}</td>')
        rows.append(
            f'<tr class="{"ref" if is_ref else ""}" data-id="{r.event_id}" data-clip="{r.game_m.upper()} test{r.clip_id}" '
            f'data-t="{r.t:.2f}" data-cls="{html.escape(name)}" data-g="{r.G_AI:.1f}" '
            f'data-pv="{html.escape(prior_v)}" data-pc="{html.escape(prior_c)}">'
            f"<td>{r.event_id}</td><td>{html.escape(name)}</td><td>{html.escape(r.note or '—')}</td>"
            f"<td>{r.game_m.upper()} test{r.clip_id}</td><td>{r.t:.2f}</td><td>{r.raw_level:.0f}</td>"
            f'<td class="g">{r.G_AI:+.1f}</td><td>{r.G_EQ:+.1f}</td>'
            + "".join(f"<td>{player(ctx[t])}</td>" for t, _ in TRACKS)
            + "".join(f"<td>{player(seg[t])}</td>" for t, _ in TRACKS) + prior_td + f'<td class="rv">{review}</td></tr>')

    head = "".join(f"<th>{lab}</th>" for _, lab in TRACKS)
    page = f"""<!doctype html><meta charset="utf-8"><title>误增强事件试听</title>
<style>body{{font:14px "Microsoft YaHei",sans-serif;margin:20px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:4px 6px;white-space:nowrap}}th{{background:#f3f3f3}}audio{{width:170px;height:30px}}
td.g{{color:#1F4E79;font-weight:bold}}tr.ref td{{background:#eef5ff}}
td.rv{{white-space:normal;min-width:230px}}td.rv select{{width:100%;margin-bottom:3px}}td.rv textarea{{width:100%;box-sizing:border-box;font:13px inherit}}
td.pv{{white-space:normal;min-width:170px;max-width:220px;font-size:13px}}td.pv.yes b{{color:#C0392B}}td.pv.no b{{color:#1E7B34}}
tr.done td.rv{{background:#f1faf1}}#bar{{margin:8px 0}}#bar button{{margin-right:8px}}</style>
<h2>误增强事件试听（人工确认非脚步，FPS AI 增益 &gt; {FER_THR:.0f} dB，共 {len(ev) - 1} 个）</h2>
<p>“上下文”为事件前后各 {CTX} s 的完整录音（事件位于正中）；“事件”为分析使用的事件片段。最后一行为典型脚步对照。建议戴耳机、固定音量对比三轨。</p>
<p>“已有审核意见”为报告采用的审核结果（{(brv["是否误增强"] == "是误增强").sum()} 个是误增强，{(brv["是否误增强"] == "不是误增强").sum()} 个不是），只读。
如需独立复核，可在最后一列填写自己的意见：结论只区分是否为误增强，原因写在备注中；内容自动保存在本浏览器中，完成后点“导出 CSV”（会同时附上已有意见便于对照）。</p>
<div id="bar"><button id="exp">导出 CSV</button><span id="prog"></span></div>
<table><tr><th rowspan=2>事件</th><th rowspan=2>类别</th><th rowspan=2>备注</th><th rowspan=2>片段</th><th rowspan=2>时间 (s)</th><th rowspan=2>原始电平 (dBFS)</th>
<th rowspan=2>FPS AI 增益 (dB)</th><th rowspan=2>EQ 增益 (dB)</th><th colspan=3>上下文 ±{CTX} s</th><th colspan=3>事件</th><th rowspan=2>已有审核意见</th><th rowspan=2>你的审核意见</th></tr>
<tr>{head}{head}</tr>{"".join(rows)}</table>
<script>
const KEY = "{STORE}", OLD = "{OLD_STORE}", OLD_MAP = {json.dumps({str(k): v for k, v in OLD_MAP.items()}, ensure_ascii=False)};
if (!localStorage.getItem(KEY) && localStorage.getItem(OLD)) {{
  const old = JSON.parse(localStorage.getItem(OLD)), st = {{}};
  for (const [id, s] of Object.entries(old)) st[id] = {{v: OLD_MAP[s.v] || "未审核", c: s.c || ""}};
  localStorage.setItem(KEY, JSON.stringify(st));
}}
const load = () => JSON.parse(localStorage.getItem(KEY) || "{{}}");
const rows = [...document.querySelectorAll("tr[data-id]")].filter(r => r.querySelector(".v"));
function refresh() {{
  const st = load(); let n = 0;
  rows.forEach(r => {{ const s = st[r.dataset.id], done = !!s && s.v !== "未审核"; r.classList.toggle("done", done); n += done; }});
  document.getElementById("prog").textContent = `已审核 ${{n}} / ${{rows.length}}`;
}}
rows.forEach(r => {{
  const id = r.dataset.id, v = r.querySelector(".v"), c = r.querySelector(".c"), s = load()[id] || {{}};
  v.value = s.v || "未审核"; c.value = s.c || "";
  const save = () => {{ const st = load(); st[id] = {{v: v.value, c: c.value}}; localStorage.setItem(KEY, JSON.stringify(st)); refresh(); }};
  v.addEventListener("change", save); c.addEventListener("input", save);
}});
document.getElementById("exp").onclick = () => {{
  const q = x => '"' + String(x).replace(/"/g, '""') + '"';
  const lines = [["event_id", "类别", "片段", "时间_s", "FPS_AI_增益_dB", "是否误增强", "原因备注", "已有_是否误增强", "已有_原因备注"].join(",")];
  rows.forEach(r => {{ const d = r.dataset;
    lines.push([d.id, q(d.cls), q(d.clip), d.t, d.g, q(r.querySelector(".v").value), q(r.querySelector(".c").value), q(d.pv), q(d.pc)].join(",")); }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\\ufeff" + lines.join("\\n")], {{type: "text/csv"}}));
  a.download = "boosted_review.csv"; a.click();
}};
refresh();
</script>"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(len(ev) - 1, "events ->", OUT, f"({os.path.getsize(OUT) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
