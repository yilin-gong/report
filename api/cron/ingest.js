
import { getSql } from '../api/_db.js';
import Papa from 'papaparse';

async function fetchCSV(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch CSV: ${r.status}`);
  const text = await r.text();
  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
  return parsed.data;
}

export default async function handler(req, res) {
  try {
    const url = process.env.CSV_SOURCE_URL;
    if (!url) return res.status(400).json({ ok: false, error: 'CSV_SOURCE_URL not set' });
    const rows = await fetchCSV(url);
    const sql = getSql();

    // Simple upsert in batches
    const BATCH = 500;
    let inserted = 0;

    // Upsert hospitals first
    for (let i = 0; i < rows.length; i += BATCH) {
      const chunk = rows.slice(i, i + BATCH);
      const values = chunk.map(r => ({
        hospital_id: Number(r['医院ID'] || r['hospital_id']),
        name: r['医院名称'] || null,
        std_name: r['标准医院名称'] || r['标准名'] || r['std_name'] || null,
        region: r['地区'] || r['region'] || null,
        match_score: r['匹配分数'] ? Number(r['匹配分数']) : null,
      })).filter(v => v.hospital_id);

      if (values.length === 0) continue;
      const insert = `
        INSERT INTO hospitals (hospital_id, name, std_name, region, match_score)
        SELECT * FROM json_populate_recordset(NULL::hospitals, $1::json)
        ON CONFLICT (hospital_id) DO UPDATE SET
          name = EXCLUDED.name,
          std_name = EXCLUDED.std_name,
          region = EXCLUDED.region,
          match_score = EXCLUDED.match_score;
      `;
      await sql(insert, JSON.stringify(values));
    }

    // Upsert visits
    for (let i = 0; i < rows.length; i += BATCH) {
      const chunk = rows.slice(i, i + BATCH);
      const values = chunk.map(r => ({
        hospital_id: Number(r['医院ID'] || r['hospital_id']),
        std_name: r['标准医院名称'] || r['std_name'] || null,
        region: r['地区'] || null,
        visit_date: r['拜访日期'] || r['visit_date'] || null,
        rep_name: r['拜访员工'] || r['rep'] || null,
        doctor_name: r['医生姓名'] || r['doctor'] || null,
        department: r['科室'] || null,
        title: r['职称'] || null,
        content: r['沟通内容'] || null,
        followup: r['后续行动'] || null,
        relation_score: r['关系评分'] ? Number(r['关系评分']) : null,
      })).filter(v => v.hospital_id);

      if (values.length === 0) continue;
      const insert = `
        INSERT INTO visits (hospital_id, std_name, region, visit_date, rep_name, doctor_name, department, title, content, followup, relation_score)
        SELECT hospital_id, std_name, region, to_date(visit_date,'YYYY-MM-DD'), rep_name, doctor_name, department, title, content, followup, relation_score
        FROM json_populate_recordset(NULL::visits, $1::json);
      `;
      await sql(insert, JSON.stringify(values));
      inserted += values.length;
    }

    // Refresh hospital_last_visit
    await sql(`
      DELETE FROM hospital_last_visit;
      INSERT INTO hospital_last_visit (hospital_id, std_name, region, last_visit, days_since, last_relation)
      SELECT
        v.hospital_id,
        COALESCE(MAX(v.std_name) FILTER (WHERE v.visit_date IS NOT NULL), h.std_name) AS std_name,
        COALESCE(MAX(v.region) FILTER (WHERE v.visit_date IS NOT NULL), h.region) AS region,
        MAX(v.visit_date) AS last_visit,
        CASE WHEN MAX(v.visit_date) IS NULL THEN NULL ELSE (CURRENT_DATE - MAX(v.visit_date))::int END AS days_since,
        (ARRAY_REMOVE(ARRAY_AGG(v.relation_score ORDER BY v.visit_date DESC), NULL))[1] AS last_relation
      FROM visits v
      JOIN hospitals h ON h.hospital_id = v.hospital_id
      GROUP BY v.hospital_id;
    `);

    return res.status(200).json({ ok: true, inserted });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
}
