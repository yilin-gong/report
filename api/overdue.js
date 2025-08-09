
import { getSql } from '../_db.js';

export default async function handler(req, res) {
  try {
    const { days = '30', region } = req.query;
    const sql = getSql();
    let q = `
      SELECT h.hospital_id, h.std_name, h.name, COALESCE(h.std_name, h.name) AS hospital,
             hv.region, hv.last_visit, hv.days_since, hv.last_relation
      FROM hospital_last_visit hv
      JOIN hospitals h ON h.hospital_id = hv.hospital_id
      WHERE hv.days_since >= $1::int
    `;
    const args = [days];
    if (region) { q += ` AND hv.region = $2`; args.push(region); }
    q += ` ORDER BY hv.days_since DESC NULLS LAST LIMIT 2000`;
    const rows = await sql(q, ...args);
    return res.status(200).json({ ok: true, rows });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
}
