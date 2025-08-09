import React, { useMemo, useRef, useState } from "react";
import { Upload, Search, Filter, Download, RefreshCw, Trash2 } from "lucide-react";
import Papa from "papaparse";

const TODAY = new Date();
const fmtDate = (d) => (d ? new Date(d).toISOString().slice(0, 10) : "");
const toNum = (v, def = 0) => (v === null || v === undefined || v === "" || Number.isNaN(Number(v)) ? def : Number(v));
function parseCSV(file, { onData, onDone, onError }) {
  Papa.parse(file, { header: true, skipEmptyLines: true, encoding: "UTF-8",
    complete: (res) => { const rows = res.data || []; onData?.(rows); onDone?.(); },
    error: (err) => onError?.(err) });
}
function downloadCSV(rows, filename) {
  const csv = Papa.unparse(rows);
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob); const a = document.createElement("a");
  a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
}
function daysBetween(d1, d2) { try { const x = new Date(d1); const y = new Date(d2);
  return Math.round((y - x) / (1000 * 60 * 60 * 24)); } catch { return undefined; } }
function classNames(...xs) { return xs.filter(Boolean).join(" "); }

const tabs = [{ key: "overdue", label: "逾期清单" }, { key: "reports", label: "日报中心" }, { key: "console", label: "AI 搜索台" }, { key: "config", label: "配置" }];

export default function App() {
  const [active, setActive] = useState("overdue");
  const [overdueRows, setOverdueRows] = useState([]);
  const [followupRows, setFollowupRows] = useState([]);
  const [fullRows, setFullRows] = useState([]);
  const [threshold, setThreshold] = useState(30);
  const [region, setRegion] = useState("全部");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  // —— 设置（本地存储，仅前端，生产建议用服务端管理密钥）
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [apiBase, setApiBase] = useState(() => localStorage.getItem('apiBase') || '/api');
  const [devSendKey, setDevSendKey] = useState(() => localStorage.getItem('devSendKey') === '1');
  const [openaiKey, setOpenaiKey] = useState(() => localStorage.getItem('openaiKey') || '');
  const [chatModel, setChatModel] = useState(() => localStorage.getItem('chatModel') || 'gpt-4o-mini');
  const [embedModel, setEmbedModel] = useState(() => localStorage.getItem('embedModel') || 'text-embedding-3-small');

  function saveSettings() {
    localStorage.setItem('apiBase', apiBase || '');
    localStorage.setItem('devSendKey', devSendKey ? '1' : '0');
    if (devSendKey) localStorage.setItem('openaiKey', openaiKey || '');
    localStorage.setItem('chatModel', chatModel || '');
    localStorage.setItem('embedModel', embedModel || '');
    setSettingsOpen(false);
  }

  async function fetchJSON(path, opts={}) {
    const headers = opts.headers || {};
    if (devSendKey && openaiKey) headers['x-openai-key'] = openaiKey;
    if (chatModel) headers['x-chat-model'] = chatModel;
    const res = await fetch((apiBase || '') + path, { ...opts, headers });
    return res.json();
  }
  // 自动从 API 拉取数据（可替代手动上传）
  React.useEffect(() => {
    (async () => {
      try {
        const res = await fetchJSON('/overdue');
        const j = await res.json();
        if (j.ok) setOverdueRows(j.rows || []);
        const rep = await fetchJSON('/reports');
        const jr = await rep.json();
        // 报表数据直接用接口返回，不强制本地合成（前端仍有兜底逻辑）
      } catch (e) { /* ignore for MVP */ }
    })();
  }, []);
  const [selected, setSelected] = useState({});

  function keyOf(r) { return `${r["医院ID"] ?? ""}__${r["标准医院名称"] ?? r["医院名称"] ?? ""}`; }
  const regions = useMemo(() => { const set = new Set(); (overdueRows||[]).forEach(r=>{ if(r["地区"]) set.add(String(r["地区"]).trim()) }); return ["全部", ...Array.from(set).sort()]; }, [overdueRows]);
  const visibleOverdue = useMemo(() => {
    const rows = overdueRows.map(r=>{ const last = r["最近拜访日期"] || r["最近拜访时间"] || r["拜访日期"];
      const days = toNum(r["距离今天天数"], daysBetween(last, TODAY)); const score = toNum(r["最近关系评分"] ?? r["关系评分"], "");
      return { ...r, __days: days, __score: score, __last: last }; });
    return rows.filter(r => region==="全部" ? true : String(r["地区"]).trim()===region)
      .filter(r => threshold ? (r.__days ?? 0) >= threshold : true)
      .sort((a,b)=>(b.__days??0)-(a.__days??0));
  }, [overdueRows, region, threshold]);
  const kpis = useMemo(() => {
    const hospitals = new Set(overdueRows.map(r=>keyOf(r))).size;
    const overdue30 = overdueRows.filter(r=>toNum(r["距离今天天数"])>=30).length;
    const maxDays = overdueRows.reduce((m,r)=>Math.max(m,toNum(r["距离今天天数"],0)),0);
    return [{label:"覆盖医院数", value:hospitals},{label:">=30天逾期", value:overdue30},{label:"最大逾期天数", value:maxDays}];
  }, [overdueRows]);

  const fileInputs = { overdue: useRef(null), followup: useRef(null), full: useRef(null) };
  function onPick(which){ fileInputs[which]?.current?.click(); }
  function onFile(which, e){
    const f = e.target.files?.[0]; if(!f) return; setLoading(true);
    parseCSV(f, { onData: rows => { if(which==="overdue") setOverdueRows(rows); if(which==="followup") setFollowupRows(rows); if(which==="full") setFullRows(rows); },
      onDone: ()=>setLoading(false), onError: ()=>setLoading(false) });
  }
  function clearData(which){ if(which==="overdue") setOverdueRows([]); if(which==="followup") setFollowupRows([]); if(which==="full") setFullRows([]); }

  function genFollowupSuggestion(r){
    const hospital = r["标准医院名称"]||r["医院名称"]||"目标医院";
    const days = r.__days ?? toNum(r["距离今天天数"],0);
    const last = r.__last || r["最近拜访日期"] || r["拜访日期"] || "—";
    const score = r["__score"] || r["最近关系评分"] || r["关系评分"] || "—";
    const area = r["地区"] || "该地区";
    return `针对${hospital}（${area}），已${days}天未回访（上次：${fmtDate(last)}，关系评分：${score}）。建议：1）电话确认本周合适时间；2）携带最新临床资料与价格政策；3）跟进上次沟通未决事项；4）预约拜访：${fmtDate(new Date(Date.now()+3*24*3600*1000))} 前。`;
  }
  function toggleSelectAll(){ const ids = visibleOverdue.map(r=>keyOf(r)); const next={...selected}; const target = !(ids.length>0 && ids.every(id=>selected[id])); ids.forEach(id=>next[id]=target); setSelected(next); }
  function toggleRow(r){ const id = keyOf(r); setSelected(s=>({...s,[id]:!s[id]})); }
  function applySuggestionToSelected(){ const next = overdueRows.map(r=>{ const id=keyOf(r); if(!selected[id]) return r; return { ...r, 后续行动: genFollowupSuggestion(r) }; }); setOverdueRows(next); }
  function exportVisible(){ downloadCSV(visibleOverdue, `逾期清单_${threshold}天_${fmtDate(TODAY)}.csv`); }
  function exportWithSuggestions(){ downloadCSV(overdueRows.map(r=>({...r})), `逾期清单_含建议_${fmtDate(TODAY)}.csv`); }

  const reportStats = useMemo(()=>{
    const src = fullRows.length?fullRows:followupRows.length?followupRows:overdueRows; const rows=src||[];
    const byRep={}, hospitals=new Set(), doctors=new Set(); let minDate, maxDate;
    rows.forEach(r=>{ const rep=(r["拜访员工"]||"未填").trim(); byRep[rep]=(byRep[rep]||0)+1;
      if(r["医院ID"]||r["标准医院名称"]||r["医院名称"]) hospitals.add(`${r["医院ID"]??""}__${r["标准医院名称"]??r["医院名称"]??""}`);
      if(r["医生姓名"]) doctors.add(String(r["医生姓名"]).trim());
      const d=r["拜访日期"]||r["最近拜访日期"]; if(d){ const t=new Date(d); if(!isNaN(+t)){ if(!minDate||t<minDate) minDate=t; if(!maxDate||t>maxDate) maxDate=t; } }
    });
    const repArr = Object.entries(byRep).map(([k,v])=>({rep:k, 次数:v})).sort((a,b)=>b.次数-a.次数).slice(0,10);
    const missingFollowups = rows.filter(r=>!String(r["后续行动"]||"").trim() || String(r["后续行动"]).trim().length<4).length;
    return { 总记录数: rows.length, 覆盖医院数: hospitals.size, 覆盖医生数: doctors.size,
      时间范围: `${minDate?fmtDate(minDate):"—"} → ${maxDate?fmtDate(maxDate):"—"}`, 缺失后续行动: missingFollowups, 员工Top10: repArr };
  },[overdueRows, followupRows, fullRows]);

  const searchResults = useMemo(()=>{
    if(!query.trim()) return []; const src = fullRows.length?fullRows:followupRows.length?followupRows:overdueRows; const q=query.trim();
    return (src||[]).filter(r=>{ const text = `${r["标准医院名称"]||r["医院名称"]||""} ${r["医生姓名"]||""} ${r["沟通内容"]||""} ${r["后续行动"]||""}`; return text.includes(q); }).slice(0,30);
  },[query, overdueRows, followupRows, fullRows]);
  function naiveSummarize(rows,q){ if(!rows.length) return "未检索到相关记录。"; const hospitals=new Set(rows.map(r=>r["标准医院名称"]||r["医院名称"]));
    const lastDate = rows.reduce((m,r)=>{ const d=r["拜访日期"]||r["最近拜访日期"]; const t=d?new Date(d):null; return !t||isNaN(+t)?m:!m||t>m?t:m; }, null);
    const tips = `共匹配 ${rows.length} 条，涉及 ${hospitals.size} 家医院；最近一次：${lastDate?fmtDate(lastDate):"—"}。`;
    const next = `建议：1）围绕“${q}”的近期问题做复盘；2）联系最近未回访的目标；3）补全缺失的后续行动；4）准备相关材料（产品证据/价格政策）。`; return `${tips}\n${next}`; }

  const needUpload = overdueRows.length===0 && followupRows.length===0 && fullRows.length===0;

  return (<div className="min-h-screen bg-gray-50 text-gray-900">
    <header className="sticky top-0 z-20 bg-white/80 backdrop-blur border-b border-gray-200">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-4">
        <div className="text-xl font-bold tracking-tight">AI 销售/医代助理 · Web 原型</div>
        <nav className="flex-1"><ul className="flex gap-2">{tabs.map(t=>(
          <li key={t.key}><button className={classNames("px-3 py-1.5 rounded-full text-sm", active===t.key?"bg-gray-900 text-white":"hover:bg-gray-100")} onClick={()=>setActive(t.key)}>{t.label}</button></li>
        ))}</ul></nav>
        <div className="flex items-center gap-2"><button className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-900 text-white text-sm" onClick={exportVisible} title="导出当前视图"><Download size={16}/> 导出</button></div>
      </div>
    </header>

    <main className="max-w-6xl mx-auto px-4 py-4">
      <section className="mb-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Uploader title="逾期清单 (建议重点回访清单_医院级.csv)" hint="字段示例：医院ID, 标准医院名称, 地区, 最近拜访日期, 距离今天天数, 最近关系评分" onPick={()=>onPick("overdue")} onClear={()=>clearData("overdue")} hasData={overdueRows.length>0}/>
          <Uploader title="待补全后续行动 (补全后续行动清单.csv)" hint="字段示例：医院ID, 标准医院名称, 拜访日期, 拜访员工, 医生姓名, 沟通内容, 后续行动" onPick={()=>onPick("followup")} onClear={()=>clearData("followup")} hasData={followupRows.length>0}/>
          <Uploader title="完整日报（可选，用于更全面统计）" hint="上传 hospital_data.csv，可解锁更多指标" onPick={()=>onPick("full")} onClear={()=>clearData("full")} hasData={fullRows.length>0}/>
        </div>
        <input ref={fileInputs.overdue} type="file" accept=".csv" className="hidden" onChange={(e)=>onFile("overdue", e)}/>
        <input ref={fileInputs.followup} type="file" accept=".csv" className="hidden" onChange={(e)=>onFile("followup", e)}/>
        <input ref={fileInputs.full} type="file" accept=".csv" className="hidden" onChange={(e)=>onFile("full", e)}/>
      </section>

      {needUpload && (<div className="rounded-2xl border border-dashed border-gray-300 p-8 text-center bg-white">
        <div className="mx-auto w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-3"><Upload/></div>
        <div className="text-lg font-medium">请先上传 CSV</div>
        <div className="text-gray-500 text-sm">先上传“逾期清单”和“待补全后续行动”两份 CSV，或直接上传完整日报 hospital_data.csv</div>
      </div>)}

      {active==="overdue" && (<section className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {kpis.map(k=> <div key={k.label} className="rounded-xl border border-gray-200 bg-white p-4"><div className="text-sm text-gray-500">{k.label}</div><div className="text-2xl font-semibold mt-1">{k.value}</div></div>)}
        </div>

        <div className="flex flex-wrap items-center gap-3 bg-white p-3 rounded-xl border border-gray-200">
          <div className="flex items-center gap-2"><Filter size={18} className="text-gray-500"/><span className="text-sm text-gray-600">逾期 ≥</span>
            <select className="px-2 py-1.5 rounded-lg border" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}>{[0,7,14,21,30,45,60,90].map(d=><option key={d} value={d}>{d} 天</option>)}</select>
          </div>
          <div className="flex items-center gap-2"><span className="text-sm text-gray-600">地区</span>
            <select className="px-2 py-1.5 rounded-lg border" value={region} onChange={e=>setRegion(e.target.value)}>{regions.map(r=><option key={r} value={r}>{r}</option>)}</select>
          </div>
          <div className="flex-1"/>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 rounded-lg border" onClick={toggleSelectAll}>全选可见</button>
            <button className="px-3 py-1.5 rounded-lg border" onClick={applySuggestionToSelected}>批量生成“后续行动”</button>
            <button className="px-3 py-1.5 rounded-lg border" onClick={exportWithSuggestions}>导出含建议</button>
          </div>
        </div>

        <div className="overflow-x-auto bg-white rounded-xl border border-gray-200">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50">
              <tr className="text-left">
                <th className="p-3 w-10"></th><th className="p-3">医院</th><th className="p-3">地区</th><th className="p-3">最近拜访</th><th className="p-3">距今天数</th><th className="p-3">最近关系评分</th><th className="p-3">后续行动（本地）</th>
              </tr>
            </thead>
            <tbody>
              {visibleOverdue.map((r,i)=>(
                <tr key={keyOf(r)} className={(i%2 ? "bg-white":"bg-gray-50/40")}>
                  <td className="p-3 align-top"><input type="checkbox" onChange={()=>toggleRow(r)}/></td>
                  <td className="p-3 align-top"><div className="font-medium">{r["标准医院名称"]||r["医院名称"]}</div><div className="text-xs text-gray-500">ID: {r["医院ID"]}</div></td>
                  <td className="p-3 align-top">{r["地区"]||"—"}</td>
                  <td className="p-3 align-top">{fmtDate(r["最近拜访日期"]||r["拜访日期"])}</td>
                  <td className="p-3 align-top font-semibold">{r["__days"]??r["距离今天天数"]??"—"}</td>
                  <td className="p-3 align-top">{r["__score"]??r["最近关系评分"]??r["关系评分"]??"—"}</td>
                  <td className="p-3 align-top">
                    <textarea className="w-full min-w-[24rem] h-20 p-2 border rounded-lg" placeholder="点击批量按钮自动生成，或手动填写…"
                      value={r["后续行动"]||""} onChange={(e)=>{ const next=[...overdueRows]; const idx=next.findIndex(x=>keyOf(x)===keyOf(r)); next[idx]={...next[idx], 后续行动:e.target.value}; setOverdueRows(next); }}/>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>)}

      {active==="reports" && (<section className="space-y-4">
        <div className="bg-white rounded-xl border p-4"><div className="font-medium mb-2">提示</div><div className="text-sm text-gray-500">上传完整日报 hospital_data.csv 可展示更全面的指标与 Top10。</div></div>
      </section>)}

      {active==="console" && (<section className="space-y-4">
        <div className="bg-white rounded-xl border p-4"><div className="flex items-center gap-2"><Search size={18}/>
          <input className="flex-1 border rounded-lg px-3 py-2" placeholder="搜索 医院/医生/关键词（先上传 CSV）" value={query} onChange={(e)=>setQuery(e.target.value)}/></div>
          <div className="mt-3 text-sm text-gray-500">演示检索，后续将接入 RAG + LLM。</div></div>
      </section>)}

      {active==="config" && (<section className="space-y-4">
        <div className="bg-white rounded-xl border p-4"><div className="font-medium mb-2">示例参数（本地生效）</div>
          <div className="flex items-center gap-3 flex-wrap"><div className="flex items-center gap-2"><span className="text-sm text-gray-600">默认逾期阈值</span>
            <select className="px-2 py-1.5 rounded-lg border" value={threshold} onChange={(e)=>setThreshold(Number(e.target.value))}>{[7,14,21,30,45,60,90].map(d=><option key={d} value={d}>{d} 天</option>)}</select>
          </div><div className="text-sm text-gray-500">后续上线将改由后端配置 + 角色权限控制。</div></div></div>
      </section>)}
    </main>

    {loading && (<div className="fixed bottom-4 right-4 bg-gray-900 text-white px-3 py-2 rounded-lg shadow-lg text-sm flex items-center gap-2"><RefreshCw size={16} className="animate-spin"/> 解析 CSV…</div>)}
    <footer className="py-8"/>
  </div>);
}

function Uploader({ title, hint, onPick, onClear, hasData }) {
  return (<div className="rounded-xl border border-gray-200 bg-white p-4 flex items-start gap-3">
    <div className="mt-0.5"><Upload size={18}/></div>
    <div className="flex-1"><div className="font-medium">{title}</div><div className="text-xs text-gray-500 mt-1">{hint}</div>
      <div className="mt-3 flex items-center gap-2"><button className="px-3 py-1.5 rounded-lg border" onClick={onPick}>选择 CSV</button>
        {hasData && (<button className="px-3 py-1.5 rounded-lg border text-red-600" onClick={onClear}>清除</button>)}
      </div></div></div>);
}
