"""Sample events for manual labelling and export Audacity label tracks + an HTML labelling page.

Labelling is done on the RAW track only, blind to AST guesses and to the FPS output.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
from common import GAMES, ROOT, read, SR

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "labeling")
N_TARGET = 150
PRE, POST = 0.6, 0.9          # snippet context (s)
MIN_GAP = 0.3                 # min spacing between sampled events in one clip (s)
rng = np.random.default_rng(42)


def sample_events(ev):
    ev = ev.copy()
    ev["event_id"] = np.arange(len(ev))
    ev["lvl"] = pd.cut(ev.raw_peak, [-90, -45, -35, -25, 0], labels=False)
    sizes = ev.groupby(["game", "clip_id"]).size()
    quota = np.maximum(3, np.round(N_TARGET * np.sqrt(sizes) / np.sqrt(sizes).sum())).astype(int)
    picked = []
    for key, g in ev.groupby(["game", "clip_id"]):
        strata = [s.sample(frac=1, random_state=int(rng.integers(1 << 31))) for _, s in g.groupby(["label", "lvl"])]
        chosen = []
        while len(chosen) < quota[key] and any(len(s) for s in strata):
            for j, s in enumerate(strata):
                if not len(s) or len(chosen) >= quota[key]:
                    continue
                row, strata[j] = s.iloc[0], s.iloc[1:]
                if all(abs(row.t - c.t) >= MIN_GAP for c in chosen):
                    chosen.append(row)
        picked += chosen
    return pd.DataFrame(picked).sort_values(["game", "clip_id", "t"])


def write_audacity(sel):
    os.makedirs(os.path.join(LAB, "audacity"), exist_ok=True)
    for (game, clip), g in sel.groupby(["game", "clip_id"]):
        with open(os.path.join(LAB, "audacity", f"{game}_test{clip}.txt"), "w", encoding="utf-8") as fh:
            for r in g.itertuples():
                fh.write(f"{r.t - 0.02:.3f}\t{r.t + 0.15:.3f}\t{r.event_id}\n")


EVT_PRE, EVT_POST = 0.03, 0.30   # "event only" playback window around the onset (s)
FOCUS_DB = -18                   # attenuation outside the event window in "focus" playback
PEAK_WIN, BG_WIN = 0.08, (0.6, 0.05)   # measurement windows used in analyze.py (s)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_wav(path, y):
    wavfile.write(path, SR, (y / (np.abs(y).max() + 1e-9) * 0.9 * 32767).astype(np.int16))


def ramp_env(n, a, b, ramp):
    """1 inside [a, b) samples, 0 outside, with linear ramps of `ramp` samples."""
    t = np.arange(n)
    up = np.clip((t - (a - ramp)) / ramp, 0, 1)
    down = np.clip(((b + ramp) - t) / ramp, 0, 1)
    return np.minimum(up, down)


def write_snippets(sel):
    os.makedirs(os.path.join(LAB, "snippets"), exist_ok=True)
    items = []
    cache = {}
    for r in sel.itertuples():
        key = (r.game, r.clip_id)
        if key not in cache:
            cache = {key: read(os.path.join(ROOT, GAMES[r.game]["raw"].format(n=f"test{r.clip_id}")))}
        x = cache[key]
        a, b = max(0, int((r.t - PRE) * SR)), min(len(x), int((r.t + POST) * SR))
        seg = x[a:b]
        on = r.t - a / SR
        dur = len(seg) / SR
        base = os.path.join(LAB, "snippets", f"e{r.event_id:04d}")

        ea, eb = max(0, int((on - EVT_PRE) * SR)), min(len(seg), int((on + EVT_POST) * SR))
        env = ramp_env(len(seg), ea, eb, int(0.005 * SR))[:, None]
        save_wav(base + ".wav", seg)
        save_wav(base + "_evt.wav", (seg * env)[ea:eb])
        focus = ramp_env(len(seg), ea, eb, int(0.015 * SR))
        focus = 10 ** (FOCUS_DB / 20) + (1 - 10 ** (FOCUS_DB / 20)) * focus
        save_wav(base + "_foc.wav", seg * focus[:, None])

        f, tt, Z = stft(seg.mean(1), SR, nperseg=1024, noverlap=768)
        S = 20 * np.log10(np.abs(Z) + 1e-7)
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        m = ax.pcolormesh(tt, f, S - S.max(), vmin=-80, vmax=0, shading="auto", cmap="magma")
        ax.axvspan(max(0, on - BG_WIN[0]), on - BG_WIN[1], color="white", alpha=0.13, lw=0)
        ax.axvspan(on, on + PEAK_WIN, color="cyan", alpha=0.30, lw=0)
        ax.axvline(on, c="cyan", lw=1.5)
        ax.set_yscale("symlog", linthresh=500)
        ax.set_ylim(50, 20000)
        ax.set_yticks([100, 500, 1000, 2000, 5000, 10000, 20000], ["100", "500", "1k", "2k", "5k", "10k", "20k"])
        ax.set_xlim(0, dur)
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("频率 (Hz)")
        cb = plt.colorbar(m, ax=ax, pad=0.01)
        cb.set_label("相对能量 (dB)")
        plt.tight_layout()
        pos = ax.get_position()
        plt.savefig(base + ".png", dpi=80)
        plt.close()
        items.append(dict(id=int(r.event_id), game=r.game, clip=int(r.clip_id), t=float(r.t),
                          onset=round(on, 4), dur=round(dur, 4), evt0=round(ea / SR, 4),
                          x0=round(pos.x0, 4), x1=round(pos.x1, 4), y0=round(pos.y0, 4), y1=round(pos.y1, 4)))
    return items


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>脚步标注</title>
<style>
body{font-family:"Microsoft YaHei",sans-serif;margin:20px;background:#1e1e1e;color:#ddd}
#card{max-width:760px}img{width:100%;display:block}
#imgwrap{position:relative;border:1px solid #444}
#ph{position:absolute;width:2px;background:#fff;box-shadow:0 0 3px #000;display:none;pointer-events:none}
button.mode{font-size:14px;padding:6px 10px}button.mode.on{background:#2a5d8f;border-color:#6af}
.legend{color:#aaa;font-size:13px;line-height:1.6;margin:6px 0}
.sw{display:inline-block;width:14px;height:10px;vertical-align:middle;margin-right:3px}
button{font-size:16px;margin:4px;padding:8px 14px;cursor:pointer;border-radius:6px;border:1px solid #666;background:#333;color:#eee}
button.sel{background:#2d7d46;border-color:#4c4}.meta{color:#999;margin:6px 0}
#bar{height:6px;background:#333;margin:10px 0}#fill{height:6px;background:#4c4;width:0}
kbd{background:#444;padding:1px 5px;border-radius:3px}
</style></head><body>
<h3>脚步标注 __ROUND__（只听原始音频）</h3>
<div>快捷键：<kbd>F</kbd> 脚步　<kbd>Shift+F</kbd> 脚步并注明地面/背景　<kbd>G</kbd> 枪声　<kbd>O</kbd> 其他非脚步（可填类型）　<kbd>M</kbd> 混合（可填组合）　<kbd>X</kbd> 不确定　
<kbd>1</kbd> 只听事件　<kbd>2</kbd> 突出事件　<kbd>3</kbd> 完整原声　<kbd>空格</kbd> 重播　<kbd>←</kbd>/<kbd>→</kbd> 上/下一个。进度会自动保存在浏览器里。</div>
<div style="color:#aaa;margin-top:6px">只判断<b>青色区域</b>那一下声音。听不准时先按 <b>1</b> 只听这一下，再按 <b>2</b>/<b>3</b> 结合上下文判断。
脚步 + 持续背景（风/发动机/雨/音乐）→ <b>F</b>，背景明显时用 <b>Shift+F</b> 注明（如"背景发动机"）；
特殊地面的脚步（水/草/金属/木板/碎石）→ <b>Shift+F</b> 注明；脚步与另一个瞬态同时出现（枪声/换弹/技能）→ <b>M</b>；
竖线处是非脚步、别处有脚步 → 按竖线处标；跳跃落地、爬梯等移动声 → <b>O</b> 并注明。</div>
<div id="bar"><div id="fill"></div></div>
<div id="card">
<div class="meta" id="meta"></div>
<div id="imgwrap"><img id="img"><div id="ph"></div></div>
<div class="legend">
频谱图：横轴=时间，纵轴=频率（下低音、上高音），颜色=能量（<span class="sw" style="background:#fcfdbf"></span>亮黄=响，<span class="sw" style="background:#000"></span>黑=静）。
<span class="sw" style="background:#0ff"></span>青色竖线/区域 = 待标注事件的起点及测量窗口（80 ms）；
<span class="sw" style="background:#666"></span>浅灰区域 = 用来估计背景的窗口；白色移动线 = 当前播放位置。<br>
经验：脚步通常是短促的竖条，能量集中在 2–4 kHz 以下；枪声是又亮又宽、直通 10 kHz 以上、带拖尾的竖条；
持续背景（发动机/风/雨）是横向铺开的纹理或横线；人声是一组平行的弯曲横线。
</div>
<div>播放：<button class="mode" data-m="evt">1 只听事件 (0.33 s)</button><button class="mode" data-m="foc">2 突出事件 (背景压低 18 dB)</button>
<button class="mode" data-m="ctx">3 完整原声 (1.5 s)</button></div>
<audio id="aud" controls style="width:100%"></audio><br>
<button data-l="f">F 脚步</button><button id="fnote">Shift+F 脚步+备注</button><button data-l="g">G 枪声</button>
<button data-l="o">O 其他非脚步</button><button data-l="m">M 混合</button><button data-l="x">X 不确定</button>
<div id="otherbox" style="display:none;margin:6px 4px">
<span id="notetitle">其他声音类型：</span><input id="note" list="notes" placeholder="如：换弹 / 开门 / 技能 / 人声，回车确认" style="font-size:15px;width:320px;padding:4px">
<datalist id="notes"></datalist>
<button id="noteok">确认</button><span style="color:#999">（可留空，Esc 取消）</span></div>
<div style="margin-top:14px"><button id="prev">← 上一个</button><button id="next">下一个 →</button>
<button id="nextun">跳到下一个未标注</button><button id="exp">导出 CSV</button></div>
</div>
<script>
const ITEMS = __ITEMS__;
const KEY = "fosi_labels_v1";
const NAMES = {f:"脚步", g:"枪声", o:"其他", m:"混合", x:"不确定"};
const NOTE_UI = {f:["脚步的地面 / 背景：","如：水 / 草地 / 金属 / 木板 / 背景发动机 / 水+背景雨，回车确认"],
                 o:["其他声音类型：","如：换弹 / 开门 / 技能 / 人声 / 落地，回车确认"],
                 m:["混合了哪些声音：","如：脚步+枪声 / 脚步+换弹，回车确认"]};
let noteCode = "o";
let labels = JSON.parse(localStorage.getItem(KEY) || "{}");
for (const k in labels) if (typeof labels[k] === "string") labels[k] = {c: labels[k], note: ""};
let i = 0;
const $ = s => document.querySelector(s);
const typing = () => document.activeElement === $("#note");
function refreshNotes(){
  const s = new Set(Object.values(labels).filter(v=>v.c===noteCode).map(v=>v.note).filter(Boolean));
  $("#notes").innerHTML = [...s].map(n=>`<option value="${n.replace(/"/g,"&quot;")}">`).join("");
}
let mode = localStorage.getItem("fosi_mode") || "foc";
const SUFFIX = {evt: "_evt", foc: "_foc", ctx: ""};
const stem = it => `snippets/e${String(it.id).padStart(4,"0")}`;
function play(m){
  mode = m; localStorage.setItem("fosi_mode", m);
  document.querySelectorAll("button.mode").forEach(b=>b.classList.toggle("on", b.dataset.m===m));
  const a = $("#aud"); a.src = stem(ITEMS[i]) + SUFFIX[m] + ".wav"; a.play().catch(()=>{});
}
document.querySelectorAll("button.mode").forEach(b=>b.onclick=()=>play(b.dataset.m));
function tick(){
  const it = ITEMS[i], a = $("#aud"), ph = $("#ph");
  if (it && it.dur && !a.paused) {
    const t = (mode==="evt" ? it.evt0 : 0) + a.currentTime;
    ph.style.left = (100*(it.x0 + (t/it.dur)*(it.x1-it.x0))) + "%";
    ph.style.top = (100*(1-it.y1)) + "%"; ph.style.height = (100*(it.y1-it.y0)) + "%";
    ph.style.display = "block";
  } else ph.style.display = "none";
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
function show(){
  const it = ITEMS[i], cur = labels[it.id];
  const txt = cur ? NAMES[cur.c] + (cur.note ? `（${cur.note}）` : "") : "—";
  $("#meta").textContent = `${i+1}/${ITEMS.length}  事件 #${it.id}  ${it.game} test${it.clip}  t=${it.t.toFixed(2)}s  当前标注: ${txt}`;
  $("#img").src = stem(it) + ".png";
  play(mode);
  document.querySelectorAll("button[data-l]").forEach(b=>b.classList.toggle("sel", cur && b.dataset.l===cur.c));
  $("#otherbox").style.display = "none";
  const n = ITEMS.filter(x=>labels[x.id]).length; $("#fill").style.width = (100*n/ITEMS.length)+"%";
}
function setLabel(l, note=""){
  labels[ITEMS[i].id] = {c: l, note: note.trim()};
  localStorage.setItem(KEY, JSON.stringify(labels)); refreshNotes();
  if(i<ITEMS.length-1) i++; show();
}
function askNote(code){
  noteCode = code;
  const cur = labels[ITEMS[i].id];
  $("#notetitle").textContent = NOTE_UI[code][0];
  $("#note").placeholder = NOTE_UI[code][1];
  refreshNotes();
  $("#otherbox").style.display = "block";
  $("#note").value = cur && cur.c===code ? cur.note : "";
  $("#note").focus();
}
function confirmOther(){ $("#note").blur(); setLabel(noteCode, $("#note").value); }
document.querySelectorAll("button[data-l]").forEach(b=>b.onclick=()=>
  (NOTE_UI[b.dataset.l] && b.dataset.l!=="f") ? askNote(b.dataset.l) : setLabel(b.dataset.l));
$("#fnote").onclick = ()=>askNote("f");
$("#noteok").onclick = confirmOther;
$("#note").addEventListener("keydown", e=>{
  if(e.key==="Enter"){ e.preventDefault(); confirmOther(); }
  else if(e.key==="Escape"){ $("#note").blur(); $("#otherbox").style.display="none"; }
  e.stopPropagation();
});
refreshNotes();
$("#prev").onclick=()=>{ if(i>0) i--; show(); };
$("#next").onclick=()=>{ if(i<ITEMS.length-1) i++; show(); };
$("#nextun").onclick=()=>{ const j=ITEMS.findIndex(it=>!labels[it.id]); if(j>=0){i=j; show();} };
$("#exp").onclick=()=>{
  let csv="event_id,game,clip_id,t,human_label,note\n";
  ITEMS.forEach(it=>{ const v=labels[it.id];
    if(v) csv+=`${it.id},${it.game},${it.clip},${it.t},${v.c},"${(v.note||"").replace(/"/g,'""')}"\n`; });
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8"}));
  a.download="__EXPORT__"; a.click();
};
document.addEventListener("keydown", e=>{
  if(typing()) return;
  const k=e.key.toLowerCase();
  if(k==="f" && e.shiftKey){ e.preventDefault(); askNote("f"); }
  else if(k==="o" || k==="m"){ e.preventDefault(); askNote(k); }
  else if("fgx".includes(k) && k.length===1) setLabel(k);
  else if(k==="1") play("evt");
  else if(k==="2") play("foc");
  else if(k==="3") play("ctx");
  else if(e.key===" "){ e.preventDefault(); const a=$("#aud"); a.currentTime=0; a.play(); }
  else if(e.key==="ArrowLeft") $("#prev").click();
  else if(e.key==="ArrowRight") $("#next").click();
});
show();
</script></body></html>"""


def build_page(sel, tag, html_only):
    """tag '' = round 1 (label_tool.html, human_labels.csv); e.g. 'r2' -> label_tool_r2.html, human_labels_r2.csv."""
    sfx = f"_{tag}" if tag else ""
    items_path = os.path.join(LAB, f"items{sfx}.json")
    if html_only:
        with open(items_path, encoding="utf-8") as fh:
            items = json.load(fh)
        assert [it["id"] for it in items] == sel.event_id.astype(int).tolist(), "sample changed; rerun without --html-only"
    else:
        items = write_snippets(sel)
        with open(items_path, "w", encoding="utf-8") as fh:
            json.dump(items, fh)
    page = (HTML.replace("__ITEMS__", json.dumps(items))
                .replace("__EXPORT__", f"human_labels{sfx}.csv")
                .replace("__ROUND__", f"第 {tag[1:]} 轮" if tag else "第 1 轮"))
    with open(os.path.join(LAB, f"label_tool{sfx}.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    return items


if __name__ == "__main__":
    os.makedirs(LAB, exist_ok=True)
    html_only = "--html-only" in sys.argv
    if "--queue" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "r2"
        sel = pd.read_csv(os.path.join(LAB, "queue_next.csv"))
        sel.to_csv(os.path.join(LAB, f"queue_{tag}.csv"), index=False)
        build_page(sel, tag, html_only)
        print(f"{tag}: {len(sel)} queued events from {sel.groupby(['game', 'clip_id']).ngroups} clips "
              f"-> label_tool_{tag}.html")
    else:
        ev = pd.read_csv(os.path.join(HERE, "results", "events_labeled.csv"))
        sel = sample_events(ev)
        sel[["event_id", "game", "clip_id", "t", "raw_peak"]].to_csv(os.path.join(LAB, "sampled_events.csv"), index=False)
        write_audacity(sel)
        build_page(sel, "", html_only)
        print(f"sampled {len(sel)} events from {sel.groupby(['game', 'clip_id']).ngroups} clips")
        print(sel.groupby("label").size().to_string())
        print(pd.cut(sel.raw_peak, [-90, -45, -35, -25, 0]).value_counts().sort_index().to_string())
