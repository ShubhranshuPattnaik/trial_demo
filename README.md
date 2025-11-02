# TrialMatcher RAG — End-to-End README.md

## 1 What this system does

**Data → Indexes → APIs → Scores → Final Rank**

1. Pull trials from ClinicalTrials.gov v2 → **Postgres** (`trials`, `trial_locations`, `eligibility_text`, `eligibility_atoms`).
2. Build **BM25** index in Elasticsearch/OpenSearch (`trials_bm25`) on:
   - `title`, `brief_summary`, `conditions`, `eligibility_text`, `intervention_name`
3. Build **Vector** index in Elasticsearch/OpenSearch (`trials_vector`) on the **same fields** as BM25.
4. Serve APIs (FastAPI):
   - `POST /search/bm25` — lexical (BM25).
   - `POST /search/vector` — dense (semantic).
   - `POST /search/hybrid` — **RRF** fusion of BM25 + Vector.
   - `POST /search` — **Unified final score** = blend of **hybrid** + **practicality** (+ eligibility placeholder).
5. **Practicality** = distance to nearest site + status/phase features.

---

## 2 Prerequisites

- Python **3.10+**
- Elasticsearch/OpenSearch **8.x** reachable at `http://localhost:{port}` eg. 9200
- Postgres url
- macOS/Linux shell; `curl`, `jq` recommended

---

## 3 Environment variables (copy/paste)

```bash
# ── Elasticsearch / OpenSearch ────────────────────────────────────────────
export ES_URL="http://localhost:9200"
export ES_USER="admin"
export ES_PASS="Str0ng!Passw0rd"
export ES_BM25_INDEX="trials_bm25"
export ES_VECTOR_INDEX="trials_vector"
export ES_VECTOR_FIELD="embedding"     # MUST match vector mapping name in ES


# ── Embedder ─────────────────────────────────────────────────────────────
export EMBEDDER_TYPE="pubmedbert"      # or "sentence-transformer" if installed

# ── Unified Final-score weights (optional) ───────────────────────────────
export FINAL_ALPHA="0.0"               # v0: eligibility = hybrid_norm → alpha can be 0
export FINAL_BETA="0.7"
export FINAL_GAMMA="0.3"
```

---

## 4 Install (single shot)

```bash
pip install -r requirements.txt
```

---

## 5 Database: create & verify tables

The pipeline creates tables for you, but you can verify anytime:

```bash
# list tables
psql "$DATABASE_URL" -c "\\dt"

# describe the sites table (trial locations)
psql "$DATABASE_URL" -c "\\d+ trial_locations"

# sample rows
psql "$DATABASE_URL" -c "SELECT id, nct_id, facility, city, state, country, recruiting, latitude, longitude FROM trial_locations LIMIT 10;"
```

Tables (per Implementation Guide):
- `trials`
- `trial_locations`  ← **sites table**
- `eligibility_text`
- `eligibility_atoms`

---

## 6 Build the Data Pipeline

> Run from repo root with your venv active.

### 6.1 Initialize DB schema

```bash
python -m data_preparation.load init-db
```

### 6.2 Ingest Trials (ct.gov v2)

```bash
python -m data_preparation.load ingest-trials --limit 5000
```

### 6.3 Backfill Eligibility Text

```bash
python -m data_preparation.load backfill-eligibility
```

### 6.4 (Optional) Atomize Eligibility

```bash
python -m data_preparation.load atomize
```

### 6.5 Index BM25 → ES

```bash
python -m data_preparation.load index-bm25
```

### 6.6 Backfill Vectors (BM25 → Vector Index)

```bash
python scripts/backfill_vectors.py
```

**Sanity check (ES):**
```bash
# BM25 index should have docs
curl -s 'http://localhost:9200/_cat/count/trials_bm25?v'

# Vector index doc + embedding length 
curl -s 'http://localhost:9200/trials_vector/_search?size=1&_source=embedding,title,nct_id' \
| jq '.hits.hits[0]._source | {nct_id, title, vec_len: (.embedding|length)}'
```

---

## 7 Run the API

```bash
# choose embedder at startup
EMBEDDER_TYPE=pubmedbert uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Health probe:**

```bash
curl -s http://localhost:8000/health | jq .
# { "ok": true, "bm25_index": true, "vector_index": true }
```

---

## 8 Endpoints & How to Interpret Results

Set once:
```bash
BASE=http://localhost:8000
```

### 8.1 `POST /search/bm25` — Lexical

**Request**
```bash
curl -s -X POST "$BASE/search/bm25" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text":"stage IV melanoma; prior PD-1 inhibitor",
    "top_k":10,
    "filters":{"overall_status":["Recruiting","Active, not recruiting"],"phase":["Phase 2","Phase 3"]}
  }' | jq .
```

**Response (per hit)**
```json
{
  "trial_id": "NCT01234567",
  "title": "…",
  "status": "RECRUITING",
  "phase": "PHASE3",
  "conditions": "melanoma …",
  "score": 21.2558
}
```

**Meaning**: BM25 score over boosted fields  
`title^3, conditions^2, brief_summary^1.5, eligibility_text, intervention_name`.

---

### 8.2 `POST /search/vector` — Dense

**Request**
```bash
curl -s -X POST "$BASE/search/vector" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text":"postoperative pain; ketamine vs morphine randomized trial",
    "top_k":10,
    "min_score":0.0,
    "filters":{"overall_status":["Recruiting","Active, not recruiting"]}
  }' | jq .
```

**Response (per hit)**
```json
{
  "trial_id": "NCT0…",
  "title": "…",
  "status": "RECRUITING",
  "phase": "PHASE2",
  "conditions": "…",
  "score": 1.46
}
```

**Meaning**: dense similarity (cosineSimilarity + 1.0). Scores typically ~1.0–1.8 for strong matches.

---

### 8.3 `POST /search/hybrid` — RRF Fusion (BM25 + Vector)

**Request**
```bash
curl -s -X POST "$BASE/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text":"ER+ HER2- metastatic breast cancer after CDK4/6 inhibitor",
    "top_k":10,
    "filters":{"phase":["Phase 2","Phase 3"]}
  }' | jq .
```

**Response (per hit)**
```json
{
  "trial_id": "NCT…",
  "title": "…",
  "status": "RECRUITING",
  "phase": "PHASE2",
  "conditions": "…",
  "scores": { "bm25": 0.0159, "vector": 0.0147, "rrf": 0.0306 }
}
```

**Meaning**: `bm25 = 1/(k + rank_bm25)`, `vector = 1/(k + rank_vec)`, `rrf = bm25 + vector` with `k=60`.

---

### 8.4 `POST /search` — Unified Final Rank (adds Practicality)

**What is Practicality?**
- **Distance** to the closest site (km) → mapped to [0,1] via `1/(1 + d/1000)`.
- **Status score**: Recruiting=1, Not recruiting=0.2, Withdrawn=0.
- **Phase score**: 3=1, 2=0.7, 1=0.4, N/A=0.3.
- Practicality = average of distance_score, status_score, phase_score.

**Final score**
```
final = α*eligibility + β*hybrid_norm + γ*practicality
# v0: eligibility = hybrid_norm, so final ≈ (α+β)*hybrid_norm + γ*practicality
# defaults: α=0.0, β=0.7, γ=0.3
```

**Request**
```bash
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text": "ER+ HER2- metastatic breast cancer after CDK4/6 inhibitor",
    "patient_location": { "lat": 34.0522, "lon": -118.2437 },
    "filters": { "phase": ["Phase 2","Phase 3"] },
    "top_k": 10
  }' | jq .
```

**Response (per hit)**
```json
{
  "trial_id": "NCT…",
  "title": "…",
  "status": "RECRUITING",
  "phase": "PHASE2",
  "conditions": "…",
  "scores": { "hybrid": 0.0325, "practicality": 0.5691, "eligibility": 1.0, "final": 0.8707 },
  "top_site": { "city": "scottsdale", "distance_km": 589.4, "recruiting": false }
}
```

**Interpretation**
- **hybrid**: rank-fusion relevance (higher is better).
- **practicality**: nearer sites + recruiting + later phase → larger.
- **eligibility**: v0 = normalized hybrid.
- **final**: what you should sort by to recommend trials.

---


## Appendix: Example curls (copy/paste)

```bash
# Health
curl -s $BASE/health | jq .

# BM25
curl -s -X POST "$BASE/search/bm25" -H "Content-Type: application/json" -d '{
  "patient_text":"stage IV melanoma; prior PD-1 inhibitor",
  "top_k":10,
  "filters":{"overall_status":["Recruiting","Active, not recruiting"],"phase":["Phase 2","Phase 3"]}
}' | jq .

# Vector
curl -s -X POST "$BASE/search/vector" -H "Content-Type: application/json" -d '{
  "patient_text":"postoperative pain; ketamine vs morphine randomized trial",
  "top_k":10,
  "min_score":0.0,
  "filters":{"overall_status":["Recruiting","Active, not recruiting"]}
}' | jq .

# Hybrid
curl -s -X POST "$BASE/search/hybrid" -H "Content-Type: application/json" -d '{
  "patient_text":"ER+ HER2- metastatic breast cancer after CDK4/6 inhibitor",
  "top_k":10,
  "filters":{"phase":["Phase 2","Phase 3"]}
}' | jq .

# Unified Final
curl -s -X POST "$BASE/search" -H "Content-Type: application/json" -d '{
  "patient_text":"ER+ HER2- metastatic breast cancer after CDK4/6 inhibitor",
  "patient_location": { "lat": 34.0522, "lon": -118.2437 },
  "filters": { "phase": ["Phase 2","Phase 3"] },
  "top_k": 10
}' | jq .
```
