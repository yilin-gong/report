
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Hospitals
CREATE TABLE IF NOT EXISTS hospitals (
  hospital_id BIGINT PRIMARY KEY,
  name TEXT,
  std_name TEXT,
  region TEXT,
  match_score REAL
);

-- Doctors
CREATE TABLE IF NOT EXISTS doctors (
  doctor_id BIGSERIAL PRIMARY KEY,
  name TEXT,
  title TEXT,
  department TEXT,
  hospital_id BIGINT REFERENCES hospitals(hospital_id)
);

-- Reps (employees)
CREATE TABLE IF NOT EXISTS reps (
  rep_id BIGSERIAL PRIMARY KEY,
  name TEXT,
  area TEXT
);

-- Visits (source rows)
CREATE TABLE IF NOT EXISTS visits (
  visit_id BIGSERIAL PRIMARY KEY,
  hospital_id BIGINT REFERENCES hospitals(hospital_id),
  std_name TEXT,
  region TEXT,
  visit_date DATE,
  rep_name TEXT,
  doctor_name TEXT,
  department TEXT,
  title TEXT,
  content TEXT,
  followup TEXT,
  relation_score INT
);

-- Materialized-ish views (as tables we can refresh)
CREATE TABLE IF NOT EXISTS hospital_last_visit (
  hospital_id BIGINT PRIMARY KEY,
  std_name TEXT,
  region TEXT,
  last_visit DATE,
  days_since INT,
  last_relation INT
);

-- Embeddings for RAG (vector dim 1536 for text-embedding-3-small)
CREATE TABLE IF NOT EXISTS note_embeddings (
  id BIGSERIAL PRIMARY KEY,
  hospital_id BIGINT,
  visit_id BIGINT,
  chunk TEXT,
  embedding vector(1536)
);
