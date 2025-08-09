
import OpenAI from 'openai';
import { getSql } from '../api/_db.js';

export default async function handler(req, res) {
  try {
    const body = req.method === 'POST' ? req.body : null;
    const q = body?.q || req.query.q || '';
    if (!q) return res.status(400).json({ ok: false, error: 'q required' });

    const sql = getSql();
    const rows = await sql(`
      SELECT visit_id, hospital_id, COALESCE(std_name, '') AS hospital, doctor_name, visit_date, content, followup, region
      FROM visits
      WHERE (content ILIKE '%' || $1 || '%' OR followup ILIKE '%' || $1 || '%' OR COALESCE(std_name,'') ILIKE '%' || $1 || '%' OR COALESCE(doctor_name,'') ILIKE '%' || $1 || '%')
      ORDER BY visit_date DESC NULLS LAST
      LIMIT 30;
    `, q);

    const headerKey = req.headers['x-openai-key'];
    const headerModel = req.headers['x-chat-model'];
    const apiKey = headerKey || process.env.OPENAI_API_KEY;
    const model = headerModel || process.env.CHAT_MODEL || 'gpt-4o-mini';

    if (!apiKey) {
      return res.status(200).json({ ok: true, model: 'demo', rows, summary: '（演示）未配置 OPENAI_API_KEY，返回原始检索结果。' });
    }

    const client = new OpenAI({ apiKey });
    const context = rows.map(r => `【${r.hospital}｜${r.region||'—'}｜${r.visit_date||'—'}】医生：${r.doctor_name||'—'}；沟通：${(r.content||'').slice(0,120)}；后续：${(r.followup||'').slice(0,120)}`).join('\n');
    const prompt = `你是医药代表团队的复盘助理。请基于以下上下文，产出：
1) 100字内摘要
2) 3条下一步行动（≤28字/条，含时间/对象/材料）
查询：${q}
上下文：\n${context}`;

    const chat = await client.chat.completions.create({
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.2
    });
    const summary = chat.choices?.[0]?.message?.content?.trim() || '';
    return res.status(200).json({ ok: true, model, rows, summary });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
}
