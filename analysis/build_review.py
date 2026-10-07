"""Build a review page for the round-1 human labels (labeling/review_tool.html).

Per event the reviewer first judges blind (class + purity, original label hidden), then the original
label and the out-of-fold model opinion are revealed; disagreements are resolved on the spot.
This yields test-retest agreement, a corrected final label and a purity grade for clean-event extraction.
Export: labeling/review_r1.csv (read by train_labeler.py, final_label overrides the original label).

  python build_review.py --label r2   new-round page (class + purity, no original label) from queue_r2.csv
                                      -> labeling/label_tool_r2.html, export human_labels_r2.csv
"""
import os
import json
import numpy as np
import pandas as pd
from common import GAMES, ROOT, read
from analyze import rms_db
from window_sensitivity import auto_end, F2S
from prepare_labeling import LAB, PRE, POST

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["AST", "CLAP", "声学特征"]


def with_extent(sel, items, ev, extra=None):
    """Add estimated end (capped at the next onset) and nearby onsets to each item; extra(row) -> dict."""
    out = []
    for (game, clip), g in sel.groupby(["game", "clip_id"], sort=False):
        lv = rms_db(read(os.path.join(ROOT, GAMES[game]["raw"].format(n=f"test{clip}"))))
        others = ev[(ev.game == game) & (ev.clip_id == clip)].t.values
        for r in g.itertuples():
            it = dict(items[r.event_id])
            nxt = others[others > r.t + 0.03]
            end = max(auto_end(lv, int(r.t * F2S)), 0.03)
            capped = bool(len(nxt) and nxt[0] - r.t < end)
            if capped:
                end = nxt[0] - r.t
            near = others[(others > r.t - PRE) & (others < r.t + POST) & (np.abs(others - r.t) > 1e-3)]
            it.update(end=round(it["onset"] + end, 4), capped=capped,
                      near=[round(it["onset"] + (o - r.t), 4) for o in near], **(extra(r) if extra else {}))
            out.append(it)
    out.sort(key=lambda it: (it["game"], it["clip"], it["t"]))
    return out


def write_page(out, path, mode, key, export, title, flow):
    page = (HTML.replace("__ITEMS__", json.dumps(out, ensure_ascii=False)).replace("__MODE__", mode)
            .replace("__KEY__", key).replace("__EXPORT__", export).replace("__TITLE__", title).replace("__FLOW__", flow))
    with open(os.path.join(LAB, path), "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"{len(out)} events -> labeling/{path}")


def build_review():
    items = {it["id"]: it for it in json.load(open(os.path.join(LAB, "items.json"), encoding="utf-8"))}
    h = pd.read_csv(os.path.join(LAB, "human_labels.csv"), encoding="utf-8-sig", keep_default_na=False)
    ev = pd.read_csv(os.path.join(HERE, "results", "events_labeled_v2.csv"), keep_default_na=False)
    cv = pd.read_csv(os.path.join(HERE, "results", "human_cv_probs.csv"), encoding="utf-8-sig").set_index("event_id")

    def extra(r):
        if r.event_id in cv.index:
            p = {m: float(cv.loc[r.event_id, f"cv_{m}"]) for m in MODELS}
            p["ens"] = float(cv.loc[r.event_id, "cv_ens"])
        else:
            row = ev.loc[r.event_id]
            p = {m: float(row[f"p_{m}"]) for m in MODELS}
            p["ens"] = float(row.p_ens)
        return dict(orig=r.human_label, orig_note=r.note, p={k: round(v, 2) for k, v in p.items()})
    write_page(with_extent(h, items, ev, extra), "review_tool.html", "review", "fosi_review_v1", "review_r1.csv",
               "第 1 轮标注审阅（先盲判，再对照原标注）",
               "之后显示原标注和模型意见：一致自动跳下一个；不一致时 <kbd>Enter</kbd> 采用本次判断，"
               "<kbd>Backspace</kbd> 恢复原标注，也可以直接按类别键重新判。")


def build_label(tag):
    """New-round labelling page with the same two-step flow (class + purity) from queue_{tag}.csv."""
    items = {it["id"]: it for it in json.load(open(os.path.join(LAB, f"items_{tag}.json"), encoding="utf-8"))}
    sel = pd.read_csv(os.path.join(LAB, f"queue_{tag}.csv"))
    ev = pd.read_csv(os.path.join(HERE, "results", "events_labeled_v2.csv"), keep_default_na=False)
    write_page(with_extent(sel, items, ev), f"label_tool_{tag}.html", "label", f"fosi_{tag}_v2",
               f"human_labels_{tag}.csv", f"第 {tag[1:]} 轮标注（类别 + 纯净度）",
               "两步都完成后自动跳到下一个；按类别键或 P/B/L 可随时修改。")


def main():
    import sys
    if "--label" in sys.argv:
        build_label(sys.argv[sys.argv.index("--label") + 1])
    else:
        build_review()


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{font-family:"Microsoft YaHei",sans-serif;margin:20px;background:#1e1e1e;color:#ddd}
#card{max-width:780px}img{width:100%;display:block}
#imgwrap{position:relative;border:1px solid #444}
.ov{position:absolute;pointer-events:none}
#ph{width:2px;background:#fff;box-shadow:0 0 3px #000;display:none}
#endbox{border:2px dashed #ffd400;box-sizing:border-box}
.tick{width:0;border-left:2px dotted #ff5a5a}
button{font-size:15px;margin:3px;padding:7px 12px;cursor:pointer;border-radius:6px;border:1px solid #666;background:#333;color:#eee}
button.mode.on{background:#2a5d8f;border-color:#6af}button.sel{background:#2d7d46;border-color:#4c4}
.legend{color:#aaa;font-size:13px;line-height:1.6;margin:6px 0}
.sw{display:inline-block;width:14px;height:10px;vertical-align:middle;margin-right:3px}
.meta{color:#999;margin:6px 0}#bar{height:6px;background:#333;margin:10px 0}#fill{height:6px;background:#4c4;width:0}
kbd{background:#444;padding:1px 5px;border-radius:3px}
#step{font-size:17px;margin:8px 0;padding:8px;border-radius:6px;background:#2b2b2b}
#reveal{display:none;margin:8px 0;padding:10px;border-radius:6px;background:#26323d;line-height:1.8}
#reveal.conflict{background:#4a2b2b}
</style></head><body>
<h3>__TITLE__</h3>
<div>每个事件两步：① 判断类别 <kbd>F</kbd> 脚步 <kbd>Shift+F</kbd> 脚步+备注 <kbd>G</kbd> 枪声 <kbd>O</kbd> 其他 <kbd>M</kbd> 混合 <kbd>X</kbd> 不确定；
② 判断纯净度 <kbd>P</kbd> 纯净 <kbd>B</kbd> 有持续背景 <kbd>L</kbd> 有重叠。__FLOW__<br>
播放：<kbd>1</kbd> 只听事件 <kbd>2</kbd> 突出事件 <kbd>3</kbd> 完整原声 <kbd>4</kbd> 起点到估计终点　<kbd>空格</kbd> 重播　<kbd>←</kbd>/<kbd>→</kbd> 上/下一个。进度自动保存在浏览器里。</div>
<div class="legend" style="margin-top:6px">纯净度看<b style="color:#ffd400">黄色虚线框</b>（起点到估计终点）：
<b>P 纯净</b> = 框内只有这一个声音，背景几乎听不到；<b>B 有持续背景</b> = 只有这一个瞬态，但能听到风/发动机/雨/音乐/环境噪声；
<b>L 有重叠</b> = 框内还有别的瞬态（另一下脚步、枪声、换弹等都算）。<span style="color:#ff5a5a">红色点线</span> = 检测到的其他事件起点，供参考。</div>
<div id="bar"><div id="fill"></div></div>
<div id="card">
<div class="meta" id="meta"></div>
<div id="step"></div>
<div id="imgwrap"><img id="img"><div id="endbox" class="ov"></div><div id="ticks"></div><div id="ph" class="ov"></div></div>
<div class="legend">频谱图：横轴=时间，纵轴=频率，颜色=能量（亮=响）。青色竖线/区域 = 事件起点及 80 ms 测量窗口；浅灰 = 背景窗口；白色移动线 = 播放位置。</div>
<div><button class="mode" data-m="evt">1 只听事件</button><button class="mode" data-m="foc">2 突出事件</button>
<button class="mode" data-m="ctx">3 完整原声</button><button class="mode" data-m="span">4 起点→估计终点</button></div>
<audio id="aud" controls style="width:100%"></audio>
<div><button data-l="f">F 脚步</button><button id="fnote">Shift+F 脚步+备注</button><button data-l="g">G 枪声</button>
<button data-l="o">O 其他</button><button data-l="m">M 混合</button><button data-l="x">X 不确定</button></div>
<div><button data-p="p">P 纯净</button><button data-p="b">B 有持续背景</button><button data-p="l">L 有重叠</button></div>
<div id="otherbox" style="display:none;margin:6px 4px">
<span id="notetitle"></span><input id="note" list="notes" style="font-size:15px;width:320px;padding:4px">
<datalist id="notes"></datalist><button id="noteok">确认</button><span style="color:#999">（可留空，Esc 取消）</span></div>
<div id="reveal"></div>
<div style="margin-top:12px"><button id="prev">← 上一个</button><button id="next">下一个 →</button>
<button id="nextun">跳到下一个未完成</button><button id="exp">导出 __EXPORT__</button></div>
</div>
<script>
const ITEMS = __ITEMS__;
const MODE = "__MODE__";
const KEY = "__KEY__";
const NAMES = {f:"脚步", g:"枪声", o:"其他", m:"混合", x:"不确定"};
const PURE = {p:"纯净", b:"有持续背景", l:"有重叠"};
const NOTE_UI = {f:["脚步的地面 / 背景：","如：水 / 草地 / 金属 / 背景发动机，回车确认"],
                 o:["其他声音类型：","如：换弹 / 开门 / 技能 / 人声 / 落地，回车确认"],
                 m:["混合了哪些声音：","如：脚步+枪声 / 脚步+换弹，回车确认"]};
let rev = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0, noteCode = "o", advTimer = null, stopAt = null;
let mode = localStorage.getItem("fosi_review_mode") || "foc";
const $ = s => document.querySelector(s);
const typing = () => document.activeElement === $("#note");
const R = it => rev[it.id] || {};
const save = () => localStorage.setItem(KEY, JSON.stringify(rev));
const fmt = (c, n) => NAMES[c] + (n ? `（${n}）` : "");
const stem = it => `snippets/e${String(it.id).padStart(4,"0")}`;
const SUFFIX = {evt:"_evt", foc:"_foc", ctx:"", span:""};
function stage(r){ return !r.c ? "label" : !r.pure ? "pure" : !r.final ? "conflict" : "done"; }
const X = (it, t) => 100*(it.x0 + Math.min(Math.max(t,0),it.dur)/it.dur*(it.x1-it.x0));

function play(m){
  mode = m; localStorage.setItem("fosi_review_mode", m);
  document.querySelectorAll("button.mode").forEach(b=>b.classList.toggle("on", b.dataset.m===m));
  const it = ITEMS[i], a = $("#aud");
  a.src = stem(it) + SUFFIX[m] + ".wav";
  stopAt = null;
  if (m === "span") {
    a.onloadedmetadata = () => { a.currentTime = Math.max(0, it.onset - 0.03); stopAt = it.end + 0.05; a.play().catch(()=>{}); };
  } else { a.onloadedmetadata = null; a.play().catch(()=>{}); }
}
document.querySelectorAll("button.mode").forEach(b=>b.onclick=()=>play(b.dataset.m));
function tick(){
  const it = ITEMS[i], a = $("#aud"), ph = $("#ph");
  if (stopAt !== null && a.currentTime >= stopAt) a.pause();
  if (it && !a.paused) {
    const t = (mode==="evt" ? it.evt0 : 0) + a.currentTime;
    ph.style.left = X(it, t) + "%"; ph.style.display = "block";
  } else ph.style.display = "none";
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

function overlays(it){
  const top = (100*(1-it.y1)) + "%", hgt = (100*(it.y1-it.y0)) + "%";
  const eb = $("#endbox");
  eb.style.left = X(it, it.onset) + "%"; eb.style.width = (X(it, it.end) - X(it, it.onset)) + "%";
  eb.style.top = top; eb.style.height = hgt;
  $("#ph").style.top = top; $("#ph").style.height = hgt;
  $("#ticks").innerHTML = it.near.map(t => `<div class="ov tick" style="left:${X(it,t)}%;top:${top};height:${hgt}"></div>`).join("");
}

function revealHtml(it, r){
  if (MODE === "label") return `本次标注：<b>${fmt(r.final, r.fnote)}</b>　｜　纯净度：<b>${PURE[r.pure]}</b>　（按类别键或 P/B/L 可修改）`;
  const p = it.p, pf = p.ens >= 0.5;
  const model = `模型（训练时没见过这条标注）：AST ${p.AST.toFixed(2)}　CLAP ${p.CLAP.toFixed(2)}　声学 ${p["声学特征"].toFixed(2)}　→ 集成 ${p.ens.toFixed(2)}（${pf ? "像脚步" : "像非脚步"}）`;
  let s = `原标注：<b>${fmt(it.orig, it.orig_note)}</b>　｜　本次盲判：<b>${fmt(r.c, r.note)}</b>　｜　纯净度：<b>${PURE[r.pure]}</b><br>${model}`;
  if (stage(r) === "conflict") s += `<br><b style="color:#ff9">两次判断不一致</b>：<kbd>Enter</kbd> 采用本次（${NAMES[r.c]}）　<kbd>Backspace</kbd> 恢复原标注（${NAMES[it.orig]}）　或按类别键重新判`;
  else s += `<br>最终标签：<b>${fmt(r.final, r.fnote)}</b>` + (r.c === it.orig ? "（与原标注一致）" : "（已改判）");
  return s;
}

function show(){
  clearTimeout(advTimer);
  const it = ITEMS[i], r = R(it), st = stage(r);
  $("#meta").textContent = `${i+1}/${ITEMS.length}  事件 #${it.id}  ${it.game} test${it.clip}  t=${it.t.toFixed(2)}s  估计时长 ${Math.round((it.end-it.onset)*1000)} ms${it.capped ? "（截到下一个事件起点）" : ""}`;
  $("#step").innerHTML = st === "label" ? (MODE === "label" ? "① 判断类别" : "① 判断类别（原标注已隐藏）") :
                         st === "pure" ? `② 判断纯净度：本次判为 <b>${fmt(r.c, r.note)}</b>，按 P / B / L` :
                         st === "conflict" ? "③ 与原标注不一致，请裁定" : "✓ 已完成";
  $("#img").src = stem(it) + ".png";
  $("#img").onload = () => overlays(it);
  if ($("#img").complete) overlays(it);
  document.querySelectorAll("button[data-l]").forEach(b=>b.classList.toggle("sel", b.dataset.l===r.c));
  document.querySelectorAll("button[data-p]").forEach(b=>b.classList.toggle("sel", b.dataset.p===r.pure));
  const rv = $("#reveal");
  if (st === "conflict" || st === "done") { rv.style.display = "block"; rv.className = st === "conflict" ? "conflict" : ""; rv.innerHTML = revealHtml(it, r); }
  else rv.style.display = "none";
  $("#otherbox").style.display = "none";
  const done = ITEMS.filter(x=>R(x).final).length;
  const same = ITEMS.filter(x=>R(x).final && R(x).c===x.orig).length;
  $("#fill").style.width = (100*done/ITEMS.length) + "%";
  $("#meta").textContent += `　｜　已完成 ${done}/${ITEMS.length}` + (MODE === "review" ? `，盲判与原标注一致 ${same}/${done}` : "");
  play(mode);
}
function go(j){ if (j>=0 && j<ITEMS.length){ i=j; show(); } }
function compare(it, r){
  if (MODE === "label" || r.c === it.orig) { r.final = r.c; r.fnote = r.note || it.orig_note || ""; }
  else { delete r.final; delete r.fnote; }
}
function setLabel(code, note=""){
  const it = ITEMS[i], r = rev[it.id] = R(it);
  r.c = code; r.note = note.trim();
  if (r.pure) compare(it, r);
  save(); show();
  if (stage(r) === "done") advTimer = setTimeout(()=>go(i+1), 900);
}
function setPure(k){
  const it = ITEMS[i], r = R(it);
  if (!r.c) { $("#step").innerHTML = "请先判断类别（F/G/O/M/X），再判纯净度"; return; }
  const first = !r.pure;
  r.pure = k; rev[it.id] = r;
  if (first || !r.final) compare(it, r);
  save(); show();
  if (first && stage(r) === "done") advTimer = setTimeout(()=>go(i+1), 900);
}
function resolve(useNew){
  const it = ITEMS[i], r = R(it);
  if (stage(r) !== "conflict") return;
  if (useNew) { r.final = r.c; r.fnote = r.note; } else { r.final = it.orig; r.fnote = it.orig_note; }
  save(); go(i+1);
}
function refreshNotes(){
  const s = new Set(Object.values(rev).filter(v=>v.c===noteCode).map(v=>v.note).filter(Boolean));
  ITEMS.forEach(it=>{ if (it.orig===noteCode && it.orig_note) s.add(it.orig_note); });
  $("#notes").innerHTML = [...s].map(n=>`<option value="${n.replace(/"/g,"&quot;")}">`).join("");
}
function askNote(code){
  noteCode = code;
  const r = R(ITEMS[i]);
  $("#notetitle").textContent = NOTE_UI[code][0];
  $("#note").placeholder = NOTE_UI[code][1];
  refreshNotes();
  $("#otherbox").style.display = "block";
  $("#note").value = r.c===code ? (r.note||"") : "";
  $("#note").focus();
}
function confirmNote(){ $("#note").blur(); setLabel(noteCode, $("#note").value); }
document.querySelectorAll("button[data-l]").forEach(b=>b.onclick=()=>
  (NOTE_UI[b.dataset.l] && b.dataset.l!=="f") ? askNote(b.dataset.l) : setLabel(b.dataset.l));
document.querySelectorAll("button[data-p]").forEach(b=>b.onclick=()=>setPure(b.dataset.p));
$("#fnote").onclick = ()=>askNote("f");
$("#noteok").onclick = confirmNote;
$("#note").addEventListener("keydown", e=>{
  if (e.key==="Enter"){ e.preventDefault(); confirmNote(); }
  else if (e.key==="Escape"){ $("#note").blur(); $("#otherbox").style.display="none"; }
  e.stopPropagation();
});
$("#prev").onclick = ()=>go(i-1);
$("#next").onclick = ()=>go(i+1);
$("#nextun").onclick = ()=>{ const j = ITEMS.findIndex(it=>!R(it).final); if (j>=0) go(j); };
$("#exp").onclick = ()=>{
  const q = s => `"${(s||"").replace(/"/g,'""')}"`;
  const dur = it => Math.round((it.end-it.onset)*1000);
  let csv;
  if (MODE === "label") {
    csv = "event_id,game,clip_id,t,human_label,note,purity,est_dur_ms\n";
    ITEMS.forEach(it=>{ const r = R(it); if (!r.final) return;
      csv += [it.id, it.game, it.clip, it.t, r.final, q(r.fnote), r.pure, dur(it)].join(",") + "\n"; });
  } else {
    csv = "event_id,game,clip_id,t,orig_label,orig_note,blind_label,blind_note,purity,final_label,final_note,est_dur_ms\n";
    ITEMS.forEach(it=>{ const r = R(it); if (!r.c) return;
      csv += [it.id, it.game, it.clip, it.t, it.orig, q(it.orig_note), r.c, q(r.note), r.pure||"", r.final||"", q(r.fnote),
              dur(it)].join(",") + "\n"; });
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\ufeff"+csv], {type:"text/csv;charset=utf-8"}));
  a.download = "__EXPORT__"; a.click();
};
document.addEventListener("keydown", e=>{
  if (typing()) return;
  const k = e.key.toLowerCase();
  if (k==="f" && e.shiftKey){ e.preventDefault(); askNote("f"); }
  else if (k==="o" || k==="m"){ e.preventDefault(); askNote(k); }
  else if ("fgx".includes(k) && k.length===1) setLabel(k);
  else if ("pbl".includes(k) && k.length===1) setPure(k);
  else if (e.key==="Enter"){ e.preventDefault(); const r = R(ITEMS[i]); stage(r)==="conflict" ? resolve(true) : stage(r)==="done" && go(i+1); }
  else if (e.key==="Backspace"){ e.preventDefault(); resolve(false); }
  else if (k==="1") play("evt");
  else if (k==="2") play("foc");
  else if (k==="3") play("ctx");
  else if (k==="4") play("span");
  else if (e.key===" "){ e.preventDefault(); const a=$("#aud"); a.currentTime = mode==="span" ? Math.max(0, ITEMS[i].onset-0.03) : 0; a.play(); }
  else if (e.key==="ArrowLeft") go(i-1);
  else if (e.key==="ArrowRight") go(i+1);
});
show();
</script></body></html>"""


if __name__ == "__main__":
    main()
