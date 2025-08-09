
import OpenAI from 'openai';
import { getSql } from '../api/_db.js';

function systemPrompt() {
  return `你是一名面向医药代表团队的“销售拜访后续行动”助理。目标：根据最近一次拜访记录，给出可执行、具体且合规的中文行动建议。
规范：
- 风格：简洁专业，直给动作，不要空话；避免夸大/承诺疗效；遵守合规（不涉及回扣/违规采购）。
- 结构：输出 3-4 条编号要点，每条≤28字；如信息不足，第一条写“先补信息”，说明要补什么。
- 约束：尽量含时间（如“3天内/下周三前”）、对象（科室/医生/器械科）、材料（资料/证据/政策）。`;
}

function userPrompt(r) {
  const last = r.last_visit || '未知';
  const rel = r.last_relation ?? '未知';
  const content = r.last_content || '（无记录）';
  return `医院：${r.hospital}（${r.region || '未知地区'}）
最近拜访：${last}；最近关系评分：${rel}
最近沟通摘要：${content}
请生成“后续行动”建议。若信息不足（如缺联系人/时间/采购流程），请第一条先提示补充信息。`;
}

export default async function handler(req, res) {
  try {
    const { hospital_ids = [] } = (req.method === 'POST') ? (req.body || {}) : req.query;
    const ids = Array.isArray(hospital_ids) ? hospital_ids : String(hospital_ids).split(',').filter(Boolean);
    if (!ids.length) return res.status(400).json({ ok: false, error: 'hospital_ids required' });

    const sql = getSql();
    const rows = await sql(`
      SELECT v.hospital_id, COALESCE(h.std_name, h.name) AS hospital, h.region,
             MAX(v.visit_date) AS last_visit,
             (ARRAY_REMOVE(ARRAY_AGG(v.relation_score ORDER BY v.visit_date DESC), NULL))[1] AS last_relation,
             (ARRAY_REMOVE(ARRAY_AGG(v.content ORDER BY v.visit_date DESC), NULL))[1] AS last_content
      FROM visits v
      JOIN hospitals h ON h.hospital_id = v.hospital_id
      WHERE v.hospital_id = ANY($1::bigint[])
      GROUP BY v.hospital_id, h.std_name, h.name, h.region
    `, ids);

    // Allow per-request override via headers (dev only; prefer server env vars in production)
    const headerKey = req.headers['x-openai-key'];
    const headerModel = req.headers['x-chat-model'];
    const apiKey = headerKey || process.env.OPENAI_API_KEY;
    const model = headerModel || process.env.CHAT_MODEL || 'gpt-4o-mini';

    if (!apiKey) {
      const demo = rows.map(r => ({
        hospital_id: r.hospital_id,
        suggestion: `【演示】${r.hospital}：1) 电话确认时间；2) 发送最新资料；3) 跟进未决事项；4) 3天内回访。`
      }));
      return res.status(200).json({ ok: true, model: 'demo', suggestions: demo });
    }

    const client = new OpenAI({ apiKey });
    const out = [];
    for (const r of rows) {
      const messages = [
        { role: 'system', content: systemPrompt() },
        { role: 'user', content: userPrompt(r) }
      ];
      const resp = await client.chat.completions.create({
        model,
        messages,
        temperature: 0.2
      });
      const text = resp.choices?.[0]?.message?.content?.trim() || '';
      out.push({ hospital_id: r.hospital_id, suggestion: text });
    }
    return res.status(200).json({ ok: true, model, suggestions: out });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
}
