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

docker run --name es-trials --rm -p 9200:9200 -p 9300:9300 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0

# ── Embedder ─────────────────────────────────────────────────────────────
export EMBEDDER_TYPE="pubmedbert"      # or "sentence-transformer" if installed

# ── Unified Final-score weights (optional) ───────────────────────────────  # v0: eligibility = hybrid_norm → alpha can be 0
export FINAL_ALPHA="0.0"              
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


## Eligibility Atom Extraction & LLM Integration
1. Run the API 
```bash
EMBEDDER_TYPE=pubmedbert uvicorn api.main:app --host 0.0.0.0 --port 8000
```

2. API Endpoint: /extract/eligibility-atoms

- Request structure

```bash
POST /extract/eligibility-atoms
Content-Type: application/json
{
  "eligibility_text": "<raw ct.gov eligibility section>"
}
```

3. Example Curl:
```bash
curl -s -X POST http://localhost:8000/extract/eligibility-atoms \
  -H "Content-Type: application/json" \
  -d @- <<'JSON' | jq .
{
  "eligibility_text": "Inclusion Criteria:\n- Age 18 to 75 years\n- ECOG <= 1\n- ANC >= 1.5 x 10^9/L\n- Platelets >= 100000 /uL\n- Hemoglobin >= 9 g/dL\n- Bilirubin <= 1.5 mg/dL\n- ER+ / HER2- metastatic breast cancer\n- Prior CDK4/6 inhibitor allowed\n- At least 4 weeks washout\n\nExclusion Criteria:\n- Active autoimmune disease\n- Prior pembrolizumab not allowed\n- Uncontrolled brain metastases"
}
JSON
```

## Elasticsearch Snapshots (share BM25 + Vector indexes)
https://drive.google.com/drive/folders/1uWvSfNnuMGLEMNceEYa_B0yIcPzhR7kQ?usp=sharing 
# 1 Unpack snapshot into a host folder
```bash
mkdir -p "$HOME/docker/essnapshots"
tar -xzf ~/trials_es_snapshots.tgz -C "$HOME/docker/essnapshots"
```

# 2 Start ES with the same repo path
```bash
docker run -d --name es \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  -e "path.repo=/snapshots" \
  -v "$HOME/docker/essnapshots:/snapshots" \
  docker.elastic.co/elasticsearch/elasticsearch:8.14.0
```

# 3 Register the repository name
```bash
curl -s -X PUT 'http://localhost:9200/_snapshot/local_backup' \
  -H 'Content-Type: application/json' \
  -d '{"type":"fs","settings":{"location":"/snapshots","compress":true}}'
```

# 4 Inspect available snapshots
```bash
curl -s 'http://localhost:9200/_snapshot/local_backup/_all' | jq '.snapshots[].snapshot'
```

# 5 Restore (delete existing indices first or restore with rename)
# Delete if they already exist:
```bash
curl -X DELETE 'http://localhost:9200/trials_bm25'
curl -X DELETE 'http://localhost:9200/trials_vector'
```

# Restore under original names:
```bash
curl -s -X POST "http://localhost:9200/_snapshot/local_backup/$SNAP/_restore" \
  -H 'Content-Type: application/json' \
  -d '{"indices":"trials_bm25,trials_vector","include_global_state":false}'
```

# 6 Verify
```bash
curl -s 'http://localhost:9200/_cat/indices?v'
curl -s 'http://localhost:9200/_cat/count/trials_bm25?v'
curl -s 'http://localhost:9200/_cat/count/trials_vector?v'
```


### API: /search/hybrid_v2 - Integration data is passed to LLM
```bash
curl -s -X POST "http://localhost:8000/search/hybrid_v2" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text": "62-year-old with metastatic melanoma previously treated with PD-1 inhibitor; looking for immunotherapy trials.",
    "top_k": 5,
    "include_evidence": true,
    "include_eligibility_text": false
  }' | jq .
  ```

  Reponse:
  ```bash
  {
  "schema": { ... JSON Schema for TrialAtoms ... },
  "patient_text": "…",
  "results": [
    {
      "trial_id": "NCT01234567",
      "title": "…",
      "scores": { "bm25": 0.015, "vector": 0.014, "rrf": 0.029 },
      "trial_atoms": { "... full extracted atoms ..." },
      "evidence": {
        "inclusion_items": ["Age ≥ 18 years", "ECOG ≤ 1", "..."],
        "exclusion_items": ["Active autoimmune disease", "..."]
      },
      "eligibility_text": null
    }
  ],
  "integration_data": {
    "patient_text": "…",
    "trials": [
      {
        "trial_id": "NCT01234567",
        "title": "…",
        "atoms_subset": {
          "age": { "min_years": 18, "max_years": null },
          "sex": "all",
          "performance": { "ecog_max": 1 },
          "biomarkers": { "require": ["HER2-"], "exclude": [] },
          "prior_therapies": { "allow": ["PD-1 inhibitor"], "disallow": [] },
          "comorbid_exclusions": ["active autoimmune disease"],
          "lab_thresholds": { "hemoglobin_g_dL_min": 9.0, "...": null }
        },
        "quotes": { "performance": "ECOG 0–1 …", "lab_thresholds": "Hgb ≥ 9 g/dL …" },
        "evidence": { "inclusion_items": ["…"], "exclusion_items": ["…"] },
        "retrieval": { "bm25": 0.015, "vector": 0.014, "rrf": 0.029 }
      }
    ]
  }
}

## LLM Overview 

- feat(llm): add /llm/rank endpoint + local Ollama integration

- New LLM ranking flow:
  - /search/hybrid_v2 now returns integration_data
  - /llm/rank accepts either {integration_data} or the full hybrid_v2 payload
  - Produces {verdict, eligibility_score, unmet_criteria, notes} per trial
- Added health probe /llm/health (reports provider/model)
- Ranker prefers Ollama when configured; falls back to heuristic if unreachable
- README: added LLM Quickstart (Ollama/OpenAI), end-to-end test commands
- requirements.txt: updated with FastAPI/UVicorn, elasticsearch, sentence-transformers,
  psycopg[binary], and optional openai/ollama clients

Notes:
- Ensure ES indices exist (trials_bm25, trials_vector)
- For local LLM testing: `ollama serve` + `ollama pull llama3:8b`, then
  `LLM_PROVIDER=OLLAMA LLM_MODEL=llama3:8b uvicorn api.main:app`


#Start — Run LLM Ranking Locally

```bash
# --- 1. Install & start Ollama (macOS) ---
brew install ollama
ollama serve          # keep running in separate terminal
ollama pull llama3:8b # one-time model download

# --- 2. From project root ---
export ES_URL=http://localhost:9200
export ES_BM25_INDEX=trials_bm25
export ES_VECTOR_INDEX=trials_vector

# --- 3. Select LLM provider ---
# Local (Ollama)
export LLM_PROVIDER=OLLAMA
export LLM_MODEL="llama3:8b"

# (or) OpenAI cloud
# export LLM_PROVIDER=OPENAI
# export LLM_MODEL="gpt-4o-mini"
# export OPENAI_API_KEY=sk-...

# --- 4. Start the API ---
python3 -m uvicorn api.main:app --reload --port 8000

# --- 5. Verify ---
# API health:  http://127.0.0.1:8000/health
# LLM health:  http://127.0.0.1:8000/llm/health
```

Then test end-to-end:

1. `POST /search/hybrid_v2` → copy `integration_data`
2. `POST /llm/rank` → paste the copied object under `integration_data`

---

## LLM Eligibility Ranking (Ollama / OpenAI)

### Setup

**Choose your provider:**

**Local (Ollama)**

```bash
brew install ollama
ollama serve
ollama pull llama3:8b
```

**OpenAI**

```bash
export OPENAI_API_KEY=sk-...
```

### Run API with LLM Enabled

```bash
export ES_URL=http://localhost:9200
export ES_BM25_INDEX=trials_bm25
export ES_VECTOR_INDEX=trials_vector

#For Ollama
export LLM_PROVIDER=OLLAMA
export LLM_MODEL="llama3:8b"

# Or for OpenAI
# export LLM_PROVIDER=OPENAI
# export LLM_MODEL="gpt-4o-mini"

python3 -m uvicorn api.main:app --reload --port 8000
```

**Health checks**

* `http://127.0.0.1:8000/health` → check ES indices
* `http://127.0.0.1:8000/llm/health` → verify provider & model

### End-to-End Test

**Step 1 — Retrieve candidate trials**

```bash
curl -s -X POST "http://127.0.0.1:8000/search/hybrid_v2" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text": "62-year-old with metastatic melanoma previously treated with a PD-1 inhibitor; ECOG 1; seeking immunotherapy.",
    "top_k": 5,
    "include_evidence": true
  }' | jq .
```

**Step 2 — LLM rank**

```bash
curl -s -X POST "http://127.0.0.1:8000/llm/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "integration_data": {
      "patient_text": "62-year-old with metastatic melanoma previously treated with a PD-1 inhibitor; ECOG 1; seeking immunotherapy.",
      "trials": [
        {"trial_id": "LOCAL_0", "retrieval": {"bm25": 1e-9, "vector": 0.01639, "rrf": 0.01639}},
        {"trial_id": "LOCAL_1", "retrieval": {"bm25": 1e-9, "vector": 0.01613, "rrf": 0.01613}}
      ]
    }
  }' | jq .
```

**Expected Response:**

```json
{
  "trial_id": "...",
  "verdict": "include | exclude | unsure",
  "eligibility_score": 0.0-1.0,
  "unmet_criteria": [],
  "notes": "short rationale"
}
```

**Troubleshooting:**

* If you see `heuristic fallback`, verify that `LLM_PROVIDER=OLLAMA` is exported and `ollama serve` is running.
* To confirm inference: check logs for `provider=OLLAMA model=llama3:8b time_sec=...`


```