/* FinWise AI — role-aware UI: modes, confidence+evidence, insights, history, chat */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const money = (v) => (v === null || v === undefined || v === "") ? "—"
  : (typeof v === "number" ? v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : esc(v));
const ROLE = () => (window.ROLE || "user");
const admin = () => ROLE() === "admin";

let state = { src:"upload", file:null, imgUrl:null, sessionCost:0, last:null, chat:[], ctx:"{}" };

/* ---------- theme ---------- */
let THEME = localStorage.getItem("theme") || "light";
function applyTheme(){ document.documentElement.dataset.theme = THEME; const b=$("themeBtn"); if(b) b.textContent = THEME==="dark"?"☀":"☾"; }
function toggleTheme(){ THEME = THEME==="dark"?"light":"dark"; localStorage.setItem("theme",THEME); applyTheme(); }

/* re-render on language switch */
window.onLangChange = () => { if(state.last) render(state.last); };

/* ---------- key modal (admin only) ---------- */
function initKey(){
  const modal = $("keyModal"); if(!modal || !$("keyBtn")) return;
  $("keyBtn").onclick = () => { modal.hidden=false; $("keyInput").focus(); };
  $("keyCancel").onclick = () => modal.hidden=true;
  $("keySave").onclick = async () => {
    const fd = new FormData(); fd.append("key", $("keyInput").value);
    const r = await fetch("/api/key",{method:"POST",body:fd}).then(r=>r.json());
    const pill = $("keyStatus"); if(pill){ pill.textContent = r.have_key ? t("key.set") : t("key.free"); pill.classList.toggle("on", r.have_key); }
    window.HAVE_KEY = r.have_key;
    if (r.methods && $("model")){ $("model").innerHTML = r.methods.map(m=>`<option value="${m.id}" ${m.id===r.recommended?"selected":""}>${esc(m.label)}${m.id===r.recommended?" · recommended":""}</option>`).join(""); }
    modal.hidden=true;
    if(state.last) render(state.last);
  };
}

/* ---------- source picker (analyze page) ---------- */
function initSource(){
  document.querySelectorAll(".seg-btn").forEach(b => b.onclick = () => {
    document.querySelectorAll(".seg-btn").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); state.src=b.dataset.src;
    $("uploadPanel").classList.toggle("hidden", state.src!=="upload");
    $("samplePanel").classList.toggle("hidden", state.src!=="sample");
  });
  const dz=$("dropzone"), fi=$("fileInput");
  // dz is a <label> wrapping #fileInput — its native label→input activation opens the
  // dialog once. Do NOT also call fi.click() here, or the dialog opens twice and the
  // second request cancels the first ("needs two clicks" bug).
  fi.onchange=()=>setFile(fi.files[0]);
  ["dragover","dragenter"].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.add("drag")}));
  ["dragleave","drop"].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.remove("drag")}));
  dz.addEventListener("drop",ev=>setFile(ev.dataTransfer.files[0]));
  $("sampleType").onchange=loadSamples; loadSamples();
  // reveal the optional financial inputs only while the Freedom Plan is opted in
  const fp=$("freedomPlan"), fpi=$("freedomInputs");
  if(fp && fpi){ const sync=()=>fpi.classList.toggle("hidden", !fp.checked); fp.addEventListener("change",sync); sync(); }
}
function setFile(f){ if(!f) return; state.file=f; $("fileName").textContent=f.name; }
async function loadSamples(){
  const type=$("sampleType").value;
  const r=await fetch(`/api/samples?type=${encodeURIComponent(type)}`).then(r=>r.json());
  $("sampleDoc").innerHTML=r.samples.map(s=>`<option>${esc(s)}</option>`).join("");
}

/* ---------- staged loader ---------- */
const LOAD_STEPS = ["load.upload","load.ocr","load.extract","load.verify","load.insights"];
let _loadTimer=null;
function showLoading(){
  let i=0;
  const paint=()=>{ $("results").innerHTML = `<div class="rcard loader"><div class="loader-steps">`+
    LOAD_STEPS.map((k,idx)=>`<div class="lstep ${idx<i?'done':idx===i?'active':''}"><span class="ldot"></span>${t(k)}</div>`).join("")+
    `</div></div>`; };
  paint(); _loadTimer=setInterval(()=>{ if(i<LOAD_STEPS.length-1){ i++; paint(); } }, 850);
}
function stopLoading(){ if(_loadTimer){ clearInterval(_loadTimer); _loadTimer=null; } }

/* ---------- analyze ---------- */
async function analyze(overrideType){
  const btn=$("analyzeBtn"); const fd=new FormData();
  if(state.src==="sample"){
    const type=$("sampleType").value, stem=$("sampleDoc").value; if(!stem){ return; }
    fd.append("sample",stem); fd.append("doc_type",type);
    state.imgUrl=`/api/sample_image/${encodeURIComponent(type)}/${encodeURIComponent(stem)}`;
  } else {
    if(!state.file){ $("results").innerHTML = stateCard("err","🗂",t("err.nofile.t"),t("err.nofile.d")); return; }
    fd.append("file",state.file);
    // users have no type picker -> send blank (auto-detect); a results-panel override can force a type
    fd.append("doc_type", (typeof overrideType==="string" && overrideType) || ($("docType") ? $("docType").value : ""));
    state.imgUrl = state.file.type.startsWith("image/") ? URL.createObjectURL(state.file) : null;
  }
  if(admin()){
    fd.append("model",$("model").value);
    if($("llmCat")?.checked) fd.append("use_llm_cat","1");
    if($("thinking")?.checked) fd.append("thinking","1");
  }
  if($("freedomPlan")?.checked){                            // opt-in, all roles
    fd.append("freedom","1");
    const put=(id,name)=>{ const v=($(id)?.value||"").trim(); if(v!=="") fd.append(name,v); };
    put("finIncome","income_monthly"); put("finSavings","starting_assets"); put("finFixed","fixed_costs");
  }
  fd.append("lang",LANG);
  btn.disabled=true; btn.innerHTML=`<span class="spinner"></span> ${esc(t("analyzing"))}`;
  showLoading();
  try{
    const res=await fetch("/api/analyze",{method:"POST",body:fd}).then(r=>r.json());
    stopLoading();
    state.chat=[]; state.ctx=JSON.stringify(res.data||{});
    render(res);
    if(admin() && $("sessionCost")){
      state.sessionCost += (res.cost_usd||0) + ((res.advice&&res.advice.cost_usd)||0)
        + ((res.freedom&&res.freedom.story&&res.freedom.story.cost_usd)||0);
      $("sessionCost").textContent="$"+state.sessionCost.toFixed(4);
    }
  }catch(e){ stopLoading(); $("results").innerHTML = stateCard("err","⚠",t("err.title"),esc(e)); }
  finally{ btn.disabled=false; btn.textContent=t("btn.analyze"); }
}

/* ---------- render ---------- */
const chip = (c)=>`<span class="chip c-${esc(c||"Other")}">${esc(c||"Other")}</span>`;
const CAT_COLOR = {Income:"#0f9d58",Payment:"#3b5bfd",Transfer:"#3b5bfd",Groceries:"#2e7d32",Dining:"#4caf50",
  Transport:"#1c6dd0",Utilities:"#7a8699",Shopping:"#7b3fbf",Entertainment:"#c2418a",Health:"#c0392b",
  Cash:"#b8770a",Fees:"#e5484d",Other:"#98a4b6"};
const NUMERIC_COLS = ["debit","amount","balance","credit","unit_price","line_total","quantity","qty"];

function stateCard(kind,icon,title,desc){
  return `<div class="rcard state-card ${kind}"><div class="es-icon">${icon}</div><h3>${esc(title)}</h3><p class="muted">${desc}</p></div>`;
}
function confBadge(level){
  const lab = t("conf."+level);
  return `<span class="conf ${level}" title="${esc(lab)}"><span class="conf-dot"></span>${esc(lab)}</span>`;
}
function modeLabel(res){ return admin() ? esc(res.model_label||res.model) : t("mode."+(res.mode||"balanced")); }
/* wrap a group of cards into a titled board; hidden when it has no content */
function board(titleKey, icon, inner){
  return (inner && inner.trim())
    ? `<section class="board"><div class="board-head"><h2>${icon} ${t(titleKey)}</h2></div>${inner}</section>` : "";
}

function render(res){
  state.last=res;
  if(res.status==="unconfigured"){
    $("results").innerHTML=setupCard();
    const b=$("setupKeyBtn"); if(b && $("keyBtn")) b.onclick=()=>$("keyBtn").click();
    return;
  }
  if(res.status==="needs_type"){ $("results").innerHTML=needTypeCard(res); wireNeedType(); return; }
  const a=res.analysis||{}; const isLedger=["bank_statement","credit_card_statement"].includes(res.doc_type);
  const rec = a.reconcile&&a.reconcile.reconciles;

  // tiles (role-aware)
  let tiles = tile(t("tile.doc"), esc(res.doc_type));
  if(admin()){
    tiles += tile(t("tile.model"), modeLabel(res), "s");
    tiles += tile(t("tile.cost"), "$"+(res.cost_usd||0).toFixed(4));
    tiles += tile(t("tile.latency"), (res.latency_s||0)+"s");
  } else {
    tiles += tile(t("tile.status"), t("tile.processed").replace("{s}", res.latency_s||0));
    if(res.confidence!=null) tiles += tile(t("tile.confidence"), Math.round(res.confidence*100)+"%");
  }
  if(isLedger) tiles += tile(t("tile.reconcile"), rec?`<span class="badge ok">${t("badge.balanced")}</span>`:`<span class="badge bad">${t("badge.off")}</span>`);

  // ---------- board 1: document analysis ----------
  let aHtml=`<div class="tiles">${tiles}</div>`;
  if(res.reference) aHtml+=`<div class="rcard ref-badge">📄 ${t("badge.reference")}</div>`;
  aHtml+=detectFixBar(res);
  aHtml+=(res.notice||[]).map(n=>`<div class="rcard notice-banner">ℹ️ ${esc(n)}</div>`).join("");

  if(res.error){ aHtml+=stateCard("err","⚠",t("err.extract.t"),errorMsg(res.error)); $("results").innerHTML=board("board.analysis","🗂",aHtml); return; }

  aHtml+=`<div class="export-bar"><button class="btn ghost sm" id="expJson">⬇ ${t("export.json")}</button><button class="btn ghost sm" id="expCsv">⬇ ${t("export.csv")}</button></div>`;
  if(res.needs_review>0) aHtml+=`<div class="rcard review-banner"><span class="badge warn">⚑ ${res.needs_review}</span> ${t("review.banner")}${(res.review||[]).length?`: <b>${(res.review||[]).map(esc).join(", ")}</b>`:""}</div>`;
  aHtml+=fieldsCard(res, a);
  aHtml+=insightCards(res);

  const rows=a.rows||[];
  if(rows.length) aHtml+=`<div class="rcard"><h3>${listTitle(res.doc_type)} · ${rows.length}</h3>${rowsTable(rows)}</div>`;

  if(isLedger && a.cashflow && Object.keys(a.cashflow).length){
    aHtml+=`<div class="rcard"><h3>${t("sec.cashflow")}</h3><div class="cf-donut"><div>${cashflow(a.cashflow)}${catBreak(a.category_totals||{})}</div>${donut(a.category_totals||{})}</div></div>`;
  } else if(Object.keys(a.category_totals||{}).length){
    aHtml+=`<div class="rcard"><h3>${t("sec.categories")}</h3><div class="cf-donut"><div>${catBreak(a.category_totals)}</div>${donut(a.category_totals)}</div></div>`;
  }

  if(admin()){
    if(res.thinking) aHtml+=`<div class="rcard"><details class="think" open><summary>${t("thinking.summary")}</summary><pre class="think">${esc(res.thinking)}</pre></details></div>`;
    aHtml+=`<div class="rcard"><details class="raw"><summary>${t("raw.json")}</summary><pre class="json">${esc(JSON.stringify(res.data,null,2))}</pre></details></div>`;
  }

  // ---------- board 2: financial planning ----------
  const pHtml=adviceCard(res)+freedomCard(res);

  $("results").innerHTML = board("board.analysis","🗂",aHtml) + board("board.planning","🚀",pHtml) + chatPanel();
  wireChat(); wireExport(res);
}

/* setup / needs-type / export */
function setupCard(){
  // BYO key: any role can add their own key to enable extraction
  const btn = `<button class="btn primary" id="setupKeyBtn">${t("btn.setupkey")}</button>`;
  return `<div class="rcard setup-card"><div class="es-icon">🔑</div><h3>${t("state.unconfigured.t")}</h3><p class="muted">${t("state.unconfigured.d")}</p>${btn}</div>`;
}
function needTypeCard(res){
  const opts=(window.DOC_TYPES||[]).map(x=>`<option value="${esc(x)}"${x===res.guess?" selected":""}>${esc(x)}</option>`).join("");
  return `<div class="rcard needtype-card"><div class="es-icon">❓</div><h3>${t("needtype.t")}</h3><p class="muted">${t("needtype.d")}</p>
    <div class="nt-row"><select id="ntType">${opts}</select><button class="btn primary" id="ntGo">${t("needtype.pick")}</button></div></div>`;
}
function wireNeedType(){ const go=$("ntGo"); if(go) go.onclick=()=>analyze($("ntType").value); }
function detectFixBar(res){
  if(state.src!=="upload") return "";   // samples already carry a known type
  const opts=(window.DOC_TYPES||[]).map(x=>`<option value="${esc(x)}"${x===res.doc_type?" selected":""}>${esc(x)}</option>`).join("");
  return `<div class="rcard detect-bar"><span class="df-label">🔎 ${t("detect.as")} <b>${esc(res.doc_type)}</b></span>
    <span class="df-controls"><span class="muted">${t("detect.wrong")}</span><select id="dtFix">${opts}</select><button class="btn ghost sm" id="dtGo">${t("detect.reanalyze")}</button></span></div>`;
}
function download(name,text,type){ const b=new Blob([text],{type}); const u=URL.createObjectURL(b); const a=document.createElement("a"); a.href=u; a.download=name; a.click(); URL.revokeObjectURL(u); }
function toCsv(res){
  const q=v=>`"${String(v??"").replace(/"/g,'""')}"`;
  const lines=["section,field,value,confidence"];
  (res.fields||[]).forEach(f=>lines.push(["field",q(LANG==="zh"?f.label_zh:f.label_en),q(f.value),f.level].join(",")));
  const rows=(res.analysis&&res.analysis.rows)||[];
  if(rows.length){ const cols=Object.keys(rows[0]); lines.push(""); lines.push("row,"+cols.join(","));
    rows.forEach((r,i)=>lines.push([i+1,...cols.map(c=>q(r[c]))].join(","))); }
  return lines.join("\n");
}
function wireExport(res){
  const base=`finwise_${res.doc_type}_${res.id||"result"}`;
  const j=$("expJson"), cv=$("expCsv");
  if(j) j.onclick=()=>download(base+".json", JSON.stringify(res.data||{},null,2), "application/json");
  if(cv) cv.onclick=()=>download(base+".csv", toCsv(res), "text/csv");
}
function tile(label,val,size){ return `<div class="tile"><div class="t-label">${label}</div><div class="t-value"${size==="s"?' style="font-size:15px"':""}>${val}</div></div>`; }
function errorMsg(err){
  const e=String(err||"");
  if(/output_truncated|truncat|max_tokens/i.test(e)) return t("err.toolong");
  if(/output_unparseable|json parse/i.test(e)) return t("err.parse");
  if(/too complex|union types|invalid_request/i.test(e)) return t("err.schema");
  if(/overloaded|timeout/i.test(e)) return t("err.model");
  return esc(e);
}

/* extracted fields with business labels, confidence, click-to-evidence */
function fieldsCard(res, a){
  const fs=res.fields||[];
  const img = state.imgUrl?`<img class="doc-img" src="${state.imgUrl}">`:`<div class="empty-state" style="padding:26px">${t("pdf.uploaded")}</div>`;
  let table;
  if(!fs.length){ table=`<p class="muted">${t("no.fields")}</p>`; }
  else {
    table=`<table class="data-table fields"><tbody>`+fs.map(f=>{
      const lab = LANG==="zh"? f.label_zh : f.label_en;
      const val = (f.value==null||f.value==="") ? `<span class="muted">—</span>` : (f.kind==="num"?`<b>${money(f.value)}</b>`:esc(f.value));
      const ev = f.source ? ` data-ev="${esc(f.source)}" data-match="${esc(f.match||"")}"` : "";
      return `<tr class="fld ${f.source?'has-ev':''}"${ev}><td class="fld-label">${esc(lab)}</td><td class="fld-val">${val}</td><td class="fld-conf">${confBadge(f.level)}</td></tr>`;
    }).join("")+`</tbody></table><div class="evidence" id="evidenceBox"><span class="muted">${t("evidence.hint")}</span></div>`;
  }
  return `<div class="rcard"><h3>${t("sec.extracted")}</h3><div class="split"><div>${img}</div><div>${table}</div></div></div>`;
}

/* insight cards */
const INS_ICON = {duplicate_charge:"⛔",large_bill:"🧾",recurring_payment:"🔁",anomaly_spend:"📈",missing_info:"❓"};
function insightCards(res){
  const cs=res.insights||[]; if(!cs.length) return "";
  return `<div class="rcard"><h3>${t("sec.insights")}</h3><div class="insights">`+cs.map(c=>`
    <div class="insight sev-${c.severity}">
      <div class="ins-head"><span class="ins-type">${INS_ICON[c.type]||"•"} ${t("ins."+c.type)}</span><span class="ins-impact">${esc(c.impact)}</span></div>
      <div class="ins-title">${esc(c.title)}</div>
      ${(c.evidence&&c.evidence.length)?`<details class="ins-ev"><summary>${t("ins.evidence")} (${c.evidence.length})</summary><ul>${c.evidence.map(e=>`<li>${esc(e)}</li>`).join("")}</ul></details>`:""}
      <div class="ins-action">→ ${esc(c.action)}</div>
    </div>`).join("")+`</div></div>`;
}

function adviceCard(res){
  const adv=res.advice; if(!adv||!adv.text) return "";
  return `<div class="rcard advice"><h3>💡 ${t("sec.advice")}</h3><p class="advice-text">${esc(adv.text)}</p></div>`;
}

/* ---------- Financial Freedom Plan (static before/after · no sliders) ---------- */
function freedomCard(res){
  const p=res.freedom; if(!p) return "";
  if(p.available===false){
    return `<div class="rcard freedom-card"><h3>🚀 ${t("sec.freedom")}</h3>
      <div class="empty-state" style="padding:22px"><div class="es-icon">🏦</div>
      <p class="muted">${t("freedom.needledger")}</p></div></div>`;
  }
  if(p.insufficient){
    return `<div class="rcard freedom-card"><h3>🚀 ${t("sec.freedom")}</h3><p class="muted">${esc(p.note||"")}</p></div>`;
  }
  let h=`<div class="rcard freedom-card"><h3>🚀 ${t("sec.freedom")}</h3>`;
  const asm=p.assumptions||{};
  if(asm.multi_month) h+=`<div class="freedom-multimonth">📅 ${esc(t("freedom.multimonth").replace("{n}", Math.round(asm.months||0)))}</div>`;
  h+=`<div class="freedom-hero"><span class="fh-badge">${esc(p.headline)}</span></div>`;
  if(p.story&&p.story.text) h+=`<p class="freedom-story">${esc(p.story.text)}</p>`;
  h+=freedomBaseline(p.baseline);
  h+=`<div class="freedom-note muted">↔ ${t("freedom.optnote")}</div>`;
  h+=freedomBars(p.comparison);
  h+=fiCurve(p);
  h+=freedomOpps(p.opportunities);
  h+=`<details class="freedom-assume"><summary>${t("freedom.assumptions")}</summary>
    <ul>${(p.assumptions.notes||[]).map(n=>`<li>${esc(n)}</li>`).join("")}</ul></details>`;
  h+=`<p class="freedom-disclaimer muted">⚠ ${esc(p.disclaimer)}</p>`;
  return h+`</div>`;
}
function freedomBaseline(b){
  let tiles="";
  if(b.income_monthly!=null) tiles+=tile(t("freedom.income"), "$"+money(b.income_monthly));
  tiles+=tile(t("freedom.expenses"), "$"+money(b.expenses_monthly));
  if(b.net_monthly!=null) tiles+=tile(t("freedom.net"), "$"+money(b.net_monthly));
  if(b.savings_rate!=null) tiles+=tile(t("freedom.savingsrate"), Math.round(b.savings_rate*100)+"%");
  let out=`<div class="tiles freedom-tiles">${tiles}</div>`;
  if(b.fixed_costs>0) out+=`<div class="freedom-fixed-note muted">${esc(t("freedom.incl_fixed").replace("{v}", "$"+money(b.fixed_costs)))}</div>`;
  return out;
}
function freedomBars(cmp){
  const cur=cmp.current, opt=cmp.optimized;
  if(cur.savings_rate!=null && opt.savings_rate!=null){
    const bar=(label,rate,cls)=>{ const pct=Math.max(0,Math.min(100,rate*100));
      return `<div class="cf-row"><span class="muted">${label}</span>
        <div class="cf-track"><div class="cf-fill ${cls}" style="width:${pct}%"></div></div>
        <span class="cf-val">${Math.round(rate*100)}%</span></div>`; };
    return `<div class="rcard-sub"><h4>${t("freedom.comparison")}</h4><div class="cf freedom-cf">
      ${bar(t("freedom.current"),cur.savings_rate,"cur")}
      ${bar(t("freedom.optimized"),opt.savings_rate,"in")}</div>
      <div class="freedom-surplus-row"><span class="muted">${t("freedom.surplus")}</span>
        <b>$${money(cur.monthly_surplus)}</b> → <b class="pos">$${money(opt.monthly_surplus)}</b>
        <span class="muted">(+$${money(cmp.extra_monthly_savings)}/mo)</span></div></div>`;
  }
  return `<div class="rcard-sub"><h4>${t("freedom.comparison")}</h4>
    <div class="freedom-surplus-row"><span class="muted">${t("freedom.opp.savings")}</span>
      <b class="pos">+$${money(cmp.extra_monthly_savings)}</b> <span class="muted">/mo</span></div></div>`;
}
function fiCurve(p){
  const proj=p.projection;
  // no income timeline (credit card / unreachable) -> compare freedom numbers as two bars
  if(proj.years_now==null){
    const mx=Math.max(proj.fi_number_now,proj.fi_number_opt,1);
    const bar=(label,v,cls)=>`<div class="cf-row"><span class="muted">${label}</span>
      <div class="cf-track"><div class="cf-fill ${cls}" style="width:${Math.max(4,v/mx*100)}%"></div></div>
      <span class="cf-val">$${money(v)}</span></div>`;
    return `<div class="rcard-sub"><h4>${t("freedom.finumber")}</h4><div class="cf">
      ${bar(t("freedom.current"),proj.fi_number_now,"out")}
      ${bar(t("freedom.optimized"),proj.fi_number_opt,"in")}</div></div>`;
  }
  const r=p.assumptions.real_return, S0=p.assumptions.starting_assets||0;
  const Pnow=(p.baseline.net_monthly||0)*12, Popt=(p.comparison.optimized.monthly_surplus||0)*12;
  const fv=(P,tt)=> r>0 ? S0*Math.pow(1+r,tt)+P*(Math.pow(1+r,tt)-1)/r : S0+P*tt;
  const W=320,H=160,padL=8,padR=8,padT=10,padB=18;
  const xmax=Math.max(proj.years_now,proj.years_opt,1);
  const ymax=Math.max(proj.fi_number_now, fv(Popt,proj.years_opt), 1);
  const X=tt=>padL+(tt/xmax)*(W-padL-padR), Y=v=>H-padB-(Math.min(v,ymax)/ymax)*(H-padT-padB);
  const line=(P,end,cls)=>{ let pts=[]; const N=40;
    for(let i=0;i<=N;i++){ const tt=end*i/N; pts.push(X(tt).toFixed(1)+","+Y(fv(P,tt)).toFixed(1)); }
    return `<polyline class="${cls}" vector-effect="non-scaling-stroke" fill="none" points="${pts.join(" ")}"/>`; };
  const mark=(tv,cls)=>`<line class="fi-mark ${cls}" vector-effect="non-scaling-stroke" x1="${X(tv).toFixed(1)}" y1="${padT}" x2="${X(tv).toFixed(1)}" y2="${H-padB}"/>`;
  const fiY=Y(proj.fi_number_now).toFixed(1);
  return `<div class="rcard-sub"><h4>${t("freedom.projection")}</h4>
    <svg class="fi-curve" viewBox="0 0 ${W} ${H}" width="100%">
      <line class="fi-target" vector-effect="non-scaling-stroke" x1="${padL}" y1="${fiY}" x2="${W-padR}" y2="${fiY}"/>
      ${line(Pnow,proj.years_now,"fi-now")}${line(Popt,proj.years_opt,"fi-opt")}
      ${mark(proj.years_now,"now")}${mark(proj.years_opt,"opt")}</svg>
    <div class="fi-legend">
      <span class="fi-leg now">${t("freedom.yearsnow")}: <b>${proj.years_now} ${t("freedom.years")}</b></span>
      <span class="fi-leg opt">${t("freedom.yearsopt")}: <b>${proj.years_opt} ${t("freedom.years")}</b></span>
      <span class="fi-leg tgt">${t("freedom.finumber")}: <b>$${money(proj.fi_number_now)}</b></span></div></div>`;
}
function freedomOpps(opps){
  if(!opps||!opps.length) return "";
  const counted=opps.filter(o=>o.counted), also=opps.filter(o=>!o.counted);
  const row=(o)=>`<tr>
    <td>${chip(o.category)}</td>
    <td class="num">$${money(o.current_monthly)}</td>
    <td class="num">${o.trim_pct!=null?Math.round(o.trim_pct*100)+"%":"—"}</td>
    <td class="num pos">$${money(o.monthly_savings)}</td>
    <td class="opp-ev">${(o.evidence&&o.evidence.length)?`<details class="ins-ev"><summary>${t("freedom.opp.evidence")}</summary><ul>${o.evidence.map(e=>`<li>${esc(e)}</li>`).join("")}</ul></details>`:""}</td></tr>`;
  let body=counted.map(row).join("");
  if(also.length){ body+=`<tr class="opp-subhead"><td colspan="5">${t("freedom.recurring")}</td></tr>`+also.map(row).join(""); }
  return `<div class="rcard-sub"><h4>${t("freedom.opportunities")}</h4>
    <div class="table-wrap"><table class="data-table freedom-opps">
    <thead><tr><th>${t("freedom.opp.category")}</th><th class="num">${t("freedom.opp.current")}</th>
    <th class="num">${t("freedom.opp.trim")}</th><th class="num">${t("freedom.opp.savings")}</th><th></th></tr></thead>
    <tbody>${body}</tbody></table></div></div>`;
}
function listTitle(dt){ const m={bank_statement:["Transactions","交易明细"],credit_card_statement:["Transactions","交易明细"],invoice:["Line Items","行项目"],receipt:["Line Items","行项目"]}[dt]; return m?(LANG==="zh"?m[1]:m[0]):(state.last?.analysis?.list_field||"rows"); }

function rowsTable(rows){
  let cols=Object.keys(rows[0]).filter(c=>c!=="category"); cols.push("category");
  const head=cols.map(c=>`<th>${esc(c)}</th>`).join("");
  const body=rows.map(r=>`<tr>${cols.map(c=>{
    if(c==="category") return `<td>${chip(r.category)}</td>`;
    if(NUMERIC_COLS.includes(c)){ const cls=c==="debit"?"amt-neg":(c==="credit"?"amt-pos":""); return `<td class="num ${cls}">${money(r[c])}</td>`; }
    return `<td>${esc(r[c])}</td>`; }).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function cashflow(cf){ const mx=Math.max(Math.abs(cf.inflow||0),Math.abs(cf.outflow||0),Math.abs(cf.net||0),1);
  const bar=(l,v,c)=>`<div class="cf-row"><span class="muted">${l}</span><div class="cf-track"><div class="cf-fill ${c}" style="width:${Math.min(100,Math.abs(v)/mx*100)}%"></div></div><span class="cf-val">${money(v)}</span></div>`;
  return `<div class="cf">${bar(t("cf.inflow"),cf.inflow,"in")}${bar(t("cf.outflow"),cf.outflow,"out")}${bar(t("cf.net"),cf.net,"net")}</div>`; }
function catBreak(totals){ const it=Object.entries(totals); if(!it.length) return "";
  return `<div class="cat-break">${it.map(([c,v])=>`<span class="cb">${chip(c)} <b>${money(v)}</b></span>`).join("")}</div>`; }
function donut(totals){
  const it=Object.entries(totals); const sum=it.reduce((s,[,v])=>s+v,0); if(sum<=0) return "";
  const R=52,C=2*Math.PI*R; let off=0;
  const segs=it.map(([cat,v])=>{ const len=v/sum*C; const s=`<circle r="${R}" cx="60" cy="60" fill="none" stroke="${CAT_COLOR[cat]||'#98a4b6'}" stroke-width="16" stroke-dasharray="${len} ${C-len}" stroke-dashoffset="${-off}" transform="rotate(-90 60 60)"></circle>`; off+=len; return s; }).join("");
  return `<div class="donut"><svg width="120" height="120" viewBox="0 0 120 120">${segs}<circle r="36" cx="60" cy="60" fill="var(--surface)"></circle></svg></div>`;
}

/* evidence: click a field row -> show its source snippet with the match highlighted */
function onResultsClick(e){
  const dg=e.target.closest && e.target.closest("#dtGo");   // re-analyze as a corrected type
  if(dg){ const sel=$("dtFix"); if(sel) analyze(sel.value); return; }
  const tr=e.target.closest && e.target.closest(".fld.has-ev");
  if(!tr) return;
  const box=$("evidenceBox"); if(!box) return;
  document.querySelectorAll(".fld.active").forEach(x=>x.classList.remove("active"));
  tr.classList.add("active");
  const snip=tr.getAttribute("data-ev")||"", m=tr.getAttribute("data-match")||"";
  let html=esc(snip);
  if(m){ const i=snip.toLowerCase().indexOf(m.toLowerCase()); if(i>=0) html=esc(snip.slice(0,i))+`<mark>${esc(snip.slice(i,i+m.length))}</mark>`+esc(snip.slice(i+m.length)); }
  box.innerHTML=`<span class="ev-label">${t("evidence.source")}</span> <span class="ev-text">…${html}…</span>`;
}

/* ---------- chat ---------- */
function chatPanel(){
  if(!window.HAVE_KEY) return `<div class="rcard"><h3>${t("chat.title")}</h3><p class="chat-hint">${t("chat.needkey")}</p></div>`;
  const msgs=state.chat.map(m=>msgHtml(m)).join("");
  return `<div class="rcard"><h3>${t("chat.title")}</h3>
    <div class="chat-msgs" id="chatMsgs">${msgs}</div>
    <div class="chat-input"><textarea id="chatInput" data-i18n-ph="chat.ph" placeholder="${t("chat.ph")}"></textarea>
    <button class="btn primary" id="chatSend">${t("chat.send")}</button></div></div>`;
}
function msgHtml(m){
  const meta = (admin() && m.role==="assistant" && (m.cost!==undefined)) ? `<span class="meta">$${(m.cost||0).toFixed(4)} · ${m.latency}s</span>` : "";
  const think = (admin() && m.thinking) ? `<details class="think"><summary>${t("thinking.summary")}</summary><pre class="think">${esc(m.thinking)}</pre></details>` : "";
  return `<div class="msg ${m.role==="user"?"user":"bot"}">${esc(m.content)}${think}${meta}</div>`;
}
function wireChat(){
  const send=$("chatSend"); if(!send) return;
  const doSend=async ()=>{
    const inp=$("chatInput"); const text=inp.value.trim(); if(!text) return;
    state.chat.push({role:"user",content:text}); inp.value="";
    $("chatMsgs").insertAdjacentHTML("beforeend", msgHtml(state.chat[state.chat.length-1]));
    send.disabled=true; send.innerHTML=`<span class="spinner"></span>`;
    const fd=new FormData();
    fd.append("doc_type", state.last?.doc_type||""); fd.append("context", state.ctx);
    fd.append("messages", JSON.stringify(state.chat.map(m=>({role:m.role,content:m.content}))));
    if(admin() && $("thinking")?.checked) fd.append("thinking","1");
    try{
      const r=await fetch("/api/chat",{method:"POST",body:fd}).then(r=>r.json());
      if(r.error){ state.chat.push({role:"assistant",content:r.error}); }
      else { state.chat.push({role:"assistant",content:r.answer,thinking:r.thinking,cost:r.cost_usd,latency:r.latency_s});
        if(admin() && $("sessionCost")){ state.sessionCost += (r.cost_usd||0); $("sessionCost").textContent="$"+state.sessionCost.toFixed(4); } }
      $("chatMsgs").insertAdjacentHTML("beforeend", msgHtml(state.chat[state.chat.length-1]));
      $("chatMsgs").scrollTop=$("chatMsgs").scrollHeight;
    }finally{ send.disabled=false; send.textContent=t("chat.send"); }
  };
  send.onclick=doSend;
  $("chatInput").addEventListener("keydown",e=>{ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); doSend(); }});
}

/* ---------- history ---------- */
function initHistory(){
  const load=async ()=>{
    const dt=$("histType").value, st=$("histStatus").value;
    const r=await fetch(`/api/history?doc_type=${encodeURIComponent(dt)}&status=${encodeURIComponent(st)}`).then(r=>r.json());
    renderHist(r.records||[]);
  };
  $("histRefresh").onclick=load; $("histType").onchange=load; $("histStatus").onchange=load;
  load();
}
function statusBadge(s){ const cls=s==="ok"?"ok":s==="review"?"warn":"bad"; return `<span class="badge ${cls}">${t("hist.st."+s)||s}</span>`; }
function renderHist(recs){
  const list=$("histList");
  if(!recs.length){ list.innerHTML=`<div class="empty-state"><div class="es-icon">🗂</div><h3>${t("hist.empty.t")}</h3><p class="muted">${t("hist.empty.d")}</p></div>`; return; }
  const rows=recs.map(r=>`<tr>
    <td class="muted">${esc((r.ts||"").replace("T"," "))}</td>
    <td><b>${esc(r.filename||"—")}</b></td>
    <td>${esc(r.doc_type||"—")}</td>
    <td>${admin()?esc(r.model||"—"):t("mode."+(r.mode||"balanced"))}</td>
    <td>${statusBadge(r.status)}</td>
    <td class="num">${r.confidence!=null?Math.round(r.confidence*100)+"%":"—"}</td>
    ${admin()?`<td class="num">$${(r.cost_usd||0).toFixed(4)}</td>`:""}
    <td class="hist-act">
      <button class="btn ghost xs" data-open="${r.id}" data-sample="${esc(r.sample||"")}" data-type="${esc(r.doc_type||"")}">${t("hist.open")}</button>
      ${r.source==="sample"?`<button class="btn ghost xs" data-rerun="${r.id}" data-sample="${esc(r.sample||"")}" data-type="${esc(r.doc_type||"")}" data-mode="${esc(r.mode||"balanced")}" data-model="${esc(r.model||"")}">${t("hist.rerun")}</button>`:""}
      <button class="btn ghost xs danger" data-del="${r.id}">${t("hist.delete")}</button>
    </td></tr>`).join("");
  const costHead = admin()?`<th>${t("tile.cost")}</th>`:"";
  list.innerHTML=`<div class="table-wrap"><table class="data-table hist"><thead><tr>
    <th>${t("hist.time")}</th><th>${t("hist.file")}</th><th>${t("field.type")}</th>
    <th>${admin()?t("tile.model"):t("field.mode")}</th><th>${t("hist.status")}</th><th>${t("tile.confidence")}</th>${costHead}<th></th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}
async function histDelete(id){ await fetch(`/api/history/${id}`,{method:"DELETE"}); $("histRefresh").click(); }
async function histRerun(btn){
  const fd=new FormData();
  fd.append("sample",btn.dataset.sample); fd.append("doc_type",btn.dataset.type); fd.append("lang",LANG);
  if(admin()){ fd.append("model",btn.dataset.model); } else { fd.append("mode",btn.dataset.mode); }
  state.imgUrl=`/api/sample_image/${encodeURIComponent(btn.dataset.type)}/${encodeURIComponent(btn.dataset.sample)}`;
  const res=await fetch("/api/analyze",{method:"POST",body:fd}).then(r=>r.json());
  render(res); $("histRefresh").click(); window.scrollTo({top:$("results").offsetTop-20,behavior:"smooth"});
}
function onHistClick(e){
  const o=e.target.closest("[data-open]"), d=e.target.closest("[data-del]"), rr=e.target.closest("[data-rerun]");
  if(o){
    // sample records can re-resolve their page image; uploads weren't stored
    state.imgUrl = o.dataset.sample ? `/api/sample_image/${encodeURIComponent(o.dataset.type)}/${encodeURIComponent(o.dataset.sample)}` : null;
    fetch(`/api/history/${o.dataset.open}`).then(r=>r.json()).then(res=>{
      state.chat=[]; state.ctx=JSON.stringify(res.data||{});
      render(res); window.scrollTo({top:$("results").offsetTop-20,behavior:"smooth"});
    });
  }
  else if(rr){ histRerun(rr); }
  else if(d){ histDelete(d.dataset.del); }
}

/* ---------- init ---------- */
window.addEventListener("DOMContentLoaded", () => {
  applyTheme(); applyI18n();
  $("themeBtn") && ($("themeBtn").onclick = toggleTheme);
  $("langBtn") && ($("langBtn").onclick = toggleLang);
  initKey();
  document.addEventListener("click", onResultsClick);
  if ($("dropzone")) { initSource(); $("analyzeBtn").onclick = analyze; }
  if ($("histList")) { initHistory(); $("histList").addEventListener("click", onHistClick); }
});
