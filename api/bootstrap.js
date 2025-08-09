
import { getSql } from '../api/_db.js';
import fs from 'node:fs/promises';
import path from 'node:path';

export default async function handler(req, res) {
  try {
    const sql = getSql();
    const schemaPath = path.join(process.cwd(), 'schema.sql');
    const schema = await fs.readFile(schemaPath, 'utf-8');
    await sql(schema);
    return res.status(200).json({ ok: true, message: 'Schema applied' });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
}
