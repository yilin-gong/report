
import { getSql } from '../api/_db.js';

export default async function handler(req, res) {
  try {
    const sql = getSql();
    const stats = await sql(`
      WITH t AS (
        SELECT COUNT(*) AS total,
               COUNT(DISTINCT hospital_id) AS hospitals,
               COUNT(DISTINCT doctor_name) AS doctors,
               MIN(visit_date) AS min_date,
               MAX(visit_date) AS max_date,
               COUNT(*) FILTER (WHERE COALESCE(NULLIF(TRIM(followup), ''), NULL) IS NULL) AS missing_followup
        FROM visits
      ),
      rep AS (
        SELECT rep_name, COUNT(*) AS cnt
        FROM visits
        GROUP BY rep_name ORDER BY cnt DESC LIMIT 10
      )
      SELECT row_to_json(t.*) AS t, json_agg(rep.*) AS rep
      FROM t, rep;
    `);
    const t = stats[0]?.t || {};
    const rep = stats[0]?.rep || [];
    return res.status(200).json({ ok: true, stats: {
      总记录数: t.total || 0,
      覆盖医院数: t.hospitals || 0,
      覆盖医生数: t.doctors || 0,
      时间范围: `${t.min_date || '—'} → ${t.max_date || '—'}`,
      缺失后续行动: t.missing_followup || 0,
      员工Top10: rep.map(r => ({ rep: r.rep_name || '未填', 次数: r.cnt }))
    }});
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
}
