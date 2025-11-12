# api/main.py
import os
import math
import time
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import requests
from fastapi import FastAPI, HTTPException, Body, APIRouter
from fastapi.responses import JSONResponse,  RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from eligibility_atoms import extract_atoms_from_eligibility

from llm_ranking.ranker import llm_rank_trials

# Optional DB imports for distance-to-site
import psycopg
from decimal import Decimal

# ==============================
# Config (env)
# ==============================
ES_URL   = os.getenv("ES_URL",   "http://localhost:9200").rstrip("/")
ES_USER  = os.getenv("ES_USER",  "admin")
ES_PASS  = os.getenv("ES_PASS",  "Str0ng!Passw0rd")

BM25_INDEX   = os.getenv("ES_BM25_INDEX",   "trials_bm25")
VECTOR_INDEX = os.getenv("ES_VECTOR_INDEX", "trials_vector")
VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "embedding")

# Embedder type for text→vector fallback (if you call /search/vector or hybrid)
EMBEDDER_TYPE = os.getenv("EMBEDDER_TYPE", "pubmedbert")

# Final-score weights (v0: eligibility = hybrid_norm)
FINAL_ALPHA = float(os.getenv("FINAL_ALPHA", "0.0"))  # pre-ML-eligibility, keep 0
FINAL_BETA  = float(os.getenv("FINAL_BETA",  "0.7"))
FINAL_GAMMA = float(os.getenv("FINAL_GAMMA", "0.3"))

# ==============================
# Database URL (auto add sslmode=require for hosted PG like Supabase)
# ==============================
def _add_sslmode_if_needed(db_url: str) -> str:
    try:
        u = urlparse(db_url)
        host = (u.hostname or "").lower()
        is_local = (host in ("", "localhost", "127.0.0.1")) or host.endswith(".local")
        if is_local:
            return db_url  # local PG usually no SSL
        q = parse_qs(u.query)
        if "sslmode" not in q:
            q["sslmode"] = ["require"]
            new_query = urlencode({k: v[0] for k, v in q.items()})
            u = u._replace(query=new_query)
            return urlunparse(u)
        return db_url
    except Exception:
        return db_url

DATABASE_URL_RAW = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres.jgvhlbpjohghmdtatldc:CSCI544@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
)
DATABASE_URL = _add_sslmode_if_needed(DATABASE_URL_RAW)

# ==============================
# Sessions / Clients
# ==============================
# ES HTTP session
session = requests.Session()
session.auth = (ES_USER, ES_PASS)
session.verify = False
session.headers.update({"Content-Type": "application/json"})

# Lazy Postgres connection (psycopg3) with keepalive + retry
_pg_conn = None
def pg_conn():
    """Return a live psycopg connection; reconnects if needed."""
    global _pg_conn
    if _pg_conn is not None:
        try:
            with _pg_conn.cursor() as c:
                c.execute("SELECT 1")
            return _pg_conn
        except Exception:
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None

    last_err = None
    for _ in range(3):
        try:
            _pg_conn = psycopg.connect(DATABASE_URL)
            with _pg_conn.cursor() as c:
                c.execute("SELECT 1")
            return _pg_conn
        except Exception as e:
            last_err = e
            time.sleep(0.4)
    raise RuntimeError(f"Could not connect to Postgres: {last_err}")

# ==============================
# Models
# ==============================
class Filters(BaseModel):
    overall_status: Optional[List[str]] = Field(default=None)
    phase: Optional[List[str]] = Field(default=None)
    country: Optional[List[str]] = None
    state: Optional[List[str]] = None

class BM25Request(BaseModel):
    patient_text: str
    top_k: int = 10
    filters: Optional[Filters] = None

class VectorRequest(BaseModel):
    patient_text: str
    top_k: int = 10
    min_score: float = 0.0
    filters: Optional[Filters] = None

class TrialHit(BaseModel):
    trial_id: str
    title: Optional[str] = None
    status: Optional[str] = None
    phase: Optional[str] = None
    conditions: Optional[str] = None
    score: float

class BM25Response(BaseModel):
    results: List[TrialHit]

class VectorResponse(BaseModel):
    results: List[TrialHit]

class HybridScore(BaseModel):
    bm25: Optional[float] = None
    vector: Optional[float] = None
    rrf: float

class HybridHit(BaseModel):
    trial_id: str
    title: Optional[str] = None
    status: Optional[str] = None
    phase: Optional[str] = None
    conditions: Optional[str] = None
    scores: HybridScore

class HybridResponse(BaseModel):
    results: List[HybridHit]

class PatientLocation(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    zip: Optional[str] = None  # placeholder (no ZIP→lat/lon lookup in v0)

class UnifiedSearchRequest(BaseModel):
    patient_text: str
    patient_location: Optional[PatientLocation] = None
    filters: Optional[Filters] = None
    top_k: int = 10

class UnifiedHit(BaseModel):
    trial_id: str
    title: Optional[str] = None
    status: Optional[str] = None
    phase: Optional[str] = None
    conditions: Optional[str] = None
    scores: Dict[str, float]
    top_site: Optional[Dict[str, Any]] = None  # {city, distance_km, recruiting}

class UnifiedResponse(BaseModel):
    results: List[UnifiedHit]

# ==============================
# NEW: Eligibility extraction I/O model
# ==============================
class EligibilityTextIn(BaseModel):
    eligibility_text: str = Field(..., description="Raw clinical-trial eligibility text")

# ==============================
# Normalization helpers
# ==============================
def normalize_status(values: List[str]) -> List[str]:
    mapping = {
        "recruiting": "RECRUITING",
        "active, not recruiting": "ACTIVE_NOT_RECRUITING",
        "active_not_recruiting": "ACTIVE_NOT_RECRUITING",
        "not yet recruiting": "NOT_YET_RECRUITING",
        "completed": "COMPLETED",
        "terminated": "TERMINATED",
        "withdrawn": "WITHDRAWN",
        "suspended": "SUSPENDED",
        "enrolling by invitation": "ENROLLING_BY_INVITATION",
        "available": "AVAILABLE",
    }
    out = []
    for v in values:
        key = str(v).strip().lower()
        out.append(mapping.get(key, v if (isinstance(v, str) and v.isupper()) else key.upper().replace(" ", "_")))
    return out

def normalize_phase(values: List[str]) -> List[str]:
    mapping = {
        "phase 1": "PHASE1", "phase i": "PHASE1",
        "phase 2": "PHASE2", "phase ii": "PHASE2",
        "phase 3": "PHASE3", "phase iii": "PHASE3",
        "phase 4": "PHASE4", "phase iv": "PHASE4",
        "na": "NA", "n/a": "NA", "none": "NA",
    }
    out = []
    for v in values:
        key = str(v).strip().lower()
        out.append(mapping.get(key, v if (isinstance(v, str) and v.isupper()) else key.replace(" ", "").upper()))
    return out

def term_or_keyword(field: str, values: List[str]) -> Dict[str, Any]:
    return {
        "bool": {
            "should": [
                {"terms": {f"{field}.keyword": values}},
                {"terms": {f"{field}.raw": values}},
                {"terms": {field: values}},
                *[{"match_phrase": {field: v}} for v in values],
            ],
            "minimum_should_match": 1,
        }
    }

# ==============================
# ES: BM25
# ==============================
def es_bm25(patient_text: str, top_k: int, filters: Optional[Filters]) -> List[Dict[str, Any]]:
    must = [{
        "multi_match": {
            "query": patient_text,
            "type": "best_fields",
            "fields": [
                "title^3",
                "conditions^2",
                "brief_summary^1.5",
                "eligibility_text",
                "intervention_name"
            ],
            "operator": "or",
        }
    }]
    f: List[Dict[str, Any]] = []
    if filters:
        if filters.overall_status:
            f.append(term_or_keyword("overall_status", normalize_status(filters.overall_status)))
        if filters.phase:
            f.append(term_or_keyword("phase", normalize_phase(filters.phase)))
        if filters.country:
            f.append(term_or_keyword("sites.country", filters.country))
        if filters.state:
            f.append(term_or_keyword("sites.state", filters.state))

    payload = {
        "size": top_k,
        "query": {"bool": {"must": must, "filter": f}},
        "_source": ["nct_id", "title", "overall_status", "phase", "conditions"]
    }
    r = session.post(f"{ES_URL}/{BM25_INDEX}/_search", json=payload, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "trial_id": src.get("nct_id", h.get("_id", "")),
            "title": src.get("title"),
            "status": src.get("overall_status"),
            "phase": src.get("phase"),
            "conditions": src.get("conditions"),
            "score": float(h.get("_score", 0.0)),
        })
    return out

# ==============================
# ES: Vector (kNN) and embed
# ==============================
def es_vector_knn(query_vector: List[float], top_k: int, filters: Optional[Filters]) -> List[Dict[str, Any]]:
    f: List[Dict[str, Any]] = []
    if filters:
        if filters.overall_status:
            f.append({"terms": {"overall_status": normalize_status(filters.overall_status)}})
        if filters.phase:
            f.append({"terms": {"phase": normalize_phase(filters.phase)}})
        if filters.country:
            f.append({"terms": {"sites.country": filters.country}})
        if filters.state:
            f.append({"terms": {"sites.state": filters.state}})
    knn = {
        "field": VECTOR_FIELD,
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": max(100, top_k * 10),
    }
    if f:
        knn["filter"] = {"bool": {"filter": f}}

    payload = {
        "size": top_k,
        "knn": knn,
        "_source": ["nct_id", "title", "overall_status", "phase", "conditions"]
    }
    r = session.post(f"{ES_URL}/{VECTOR_INDEX}/_search", json=payload, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "trial_id": src.get("nct_id", h.get("_id", "")),
            "title": src.get("title"),
            "status": src.get("overall_status"),
            "phase": src.get("phase"),
            "conditions": src.get("conditions"),
            "score": float(h.get("_score", 0.0)),
        })
    return out

# Text→vector embedder via your util factory
_embedder = None
def _load_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from utils.util import create_embedder
        _embedder = create_embedder(embedder_type=EMBEDDER_TYPE)
        return _embedder
    except Exception as e:
        raise RuntimeError(f"Could not initialize embedder ({EMBEDDER_TYPE}): {e}")

def embed_text(text: str) -> List[float]:
    emb = _load_embedder().generate_embedding(text)
    return emb.tolist() if hasattr(emb, "tolist") else list(emb)

# ==============================
# RRF fusion utilities
# ==============================
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def fuse_rrf(bm25_hits: List[Dict[str, Any]],
             vec_hits: List[Dict[str, Any]],
             top_k: int) -> List[Tuple[str, float, float, float, Dict[str, Any]]]:
    bm25_rank = {h["trial_id"]: i + 1 for i, h in enumerate(bm25_hits)}
    vec_rank  = {h["trial_id"]: i + 1 for i, h in enumerate(vec_hits)}
    meta: Dict[str, Dict[str, Any]] = {}
    for h in bm25_hits + vec_hits:
        tid = h["trial_id"]
        if tid not in meta:
            meta[tid] = {
                "trial_id": tid,
                "title": h.get("title"),
                "status": h.get("status"),
                "phase":  h.get("phase"),
                "conditions": h.get("conditions"),
            }
    fused: List[Tuple[str, float, float, float, Dict[str, Any]]] = []
    keys = set(bm25_rank.keys()) | set(vec_rank.keys())
    for tid in keys:
        sb = rrf_score(bm25_rank.get(tid, 10**9))
        sv = rrf_score(vec_rank.get(tid, 10**9))
        fused.append((tid, sb + sv, sb, sv, meta[tid]))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]

# ==============================
# Practicality pieces
# ==============================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    try:
        R = 6371.0088
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
        return 2*R*math.asin(math.sqrt(a))
    except Exception:
        return None

def status_score(status: Optional[str]) -> float:
    if not status: return 0.2
    s = status.upper().replace(" ", "_")
    if s == "RECRUITING": return 1.0
    if s in ("TERMINATED", "WITHDRAWN"): return 0.0
    return 0.2  # Not recruiting, etc.

def phase_score(phase: Optional[str]) -> float:
    if not phase: return 0.3
    p = phase.upper().replace(" ", "")
    if p == "PHASE3": return 1.0
    if p == "PHASE2": return 0.7
    if p == "PHASE1": return 0.4
    return 0.3  # N/A or other

def distance_score_km(d_km: Optional[float]) -> float:
    if d_km is None: return 0.5  # neutral if unknown
    return math.exp(-d_km / 120.0)  # smooth decay (tunable)

def _to_float(x):
    if x is None: return None
    if isinstance(x, Decimal): return float(x)
    return float(x)

def fetch_trial_sites_from_db(nct_id: str) -> List[Dict[str, Any]]:
    """
    Pull sites for one trial from Postgres trial_locations.
    Prefer rows with coords. recruiting may be NULL; treat as False when missing.
    """
    sql = """
      SELECT facility, city, state, country,
             latitude, longitude, COALESCE(recruiting,false) AS recruiting
      FROM public.trial_locations
      WHERE nct_id = %s
    """
    with pg_conn().cursor() as cur:
        cur.execute(sql, (nct_id,))
        rows = cur.fetchall()
    out = []
    for facility, city, state, country, lat, lon, recruiting in rows:
        out.append({
            "facility": facility,
            "city": city,
            "state": state,
            "country": country,
            "lat": _to_float(lat) if lat is not None else None,
            "lon": _to_float(lon) if lon is not None else None,
            "recruiting": bool(recruiting),
        })
    return out

def min_site_distance_and_top_site_db(nct_id: str,
                                      patient_lat: Optional[float],
                                      patient_lon: Optional[float]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    if patient_lat is None or patient_lon is None:
        return None, None
    sites = fetch_trial_sites_from_db(nct_id)
    if not sites:
        return None, None
    # prefer recruiting sites if available
    pool = [s for s in sites if s.get("recruiting")] or sites
    best = None
    for s in pool:
        lat, lon = s.get("lat"), s.get("lon")
        if lat is None or lon is None:
            continue
        d = haversine_km(patient_lat, patient_lon, lat, lon)
        if d is None:
            continue
        if best is None or d < best[0]:
            best = (d, s)
    if best is None:
        return None, None
    d_km, site = best
    top_site = {
        "city": site.get("city") or site.get("facility"),
        "distance_km": round(d_km, 1),
        "recruiting": bool(site.get("recruiting", False))
    }
    return d_km, top_site

def minmax_norm(values: List[float]) -> List[float]:
    if not values: return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

# ==============================
# Helpers: compact per-trial data subset for external model integration
# ==============================
_DEFAULT_ATOM_KEYS = [
    "age", "sex", "diagnoses", "stage", "performance",
    "biomarkers", "prior_therapies", "lab_thresholds",
    "comorbid_exclusions", "pregnancy_required", "geo"
]

def _safe_clip_list(x, max_len=8):
    if isinstance(x, list) and len(x) > max_len:
        return x[:max_len]
    return x

# Map your extractor keys -> compact integration keys expected by the LLM
def _normalize_atoms(trial_atoms: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(trial_atoms, dict):
        return {}

    out: Dict[str, Any] = {}

    # age
    min_age = trial_atoms.get("min_age_years")
    max_age = trial_atoms.get("max_age_years")
    if min_age is not None or max_age is not None:
        out["age"] = {"min_years": min_age, "max_years": max_age}

    # sex
    sex = trial_atoms.get("sex_allowed")
    if sex:
        out["sex"] = sex  # "male" | "female" | "all" | "unknown"

    # diagnoses / stage
    if trial_atoms.get("required_cancer_type"):
        out["diagnoses"] = [trial_atoms["required_cancer_type"]]
    if trial_atoms.get("required_stage"):
        out["stage"] = trial_atoms["required_stage"]

    # performance (from ECOG)
    if trial_atoms.get("ecog_max") is not None:
        out["performance"] = {"ecog_max": trial_atoms["ecog_max"]}

    # biomarkers (merge required/excluded with polarity tags)
    req_bm = trial_atoms.get("required_biomarkers") or []
    exc_bm = trial_atoms.get("excluded_biomarkers") or []
    if req_bm or exc_bm:
        out["biomarkers"] = {
            "require": req_bm[:10],
            "exclude": exc_bm[:10],
        }

    # prior_therapies (normalize names)
    req_tx = trial_atoms.get("required_or_allowed_therapies") or []
    dis_tx = trial_atoms.get("disallowed_therapies") or []
    if req_tx or dis_tx:
        out["prior_therapies"] = {
            "allow": req_tx[:10],
            "disallow": dis_tx[:10],
        }

    # comorbid_exclusions
    exc_cond = trial_atoms.get("excluded_conditions") or []
    if exc_cond:
        out["comorbid_exclusions"] = exc_cond[:10]

    # labs (already matches)
    labs = trial_atoms.get("lab_thresholds")
    if isinstance(labs, dict) and labs:
        # keep only a handful to bound tokens
        keep_order = [
            "anc_per_uL_min", "platelets_per_uL_min", "hemoglobin_g_dL_min",
            "creatinine_mg_dL_max", "ast_uL_max", "alt_uL_max", "bilirubin_mg_dL_max",
        ]
        out["lab_thresholds"] = {k: labs.get(k) for k in keep_order if k in labs}

    # pregnancy (derive if present in excluded_conditions in future, placeholder)
    preg = trial_atoms.get("pregnancy_required")
    if preg is not None:
        out["pregnancy_required"] = preg

    # geo placeholder (if you ever add site distance filters)
    if trial_atoms.get("geo"):
        out["geo"] = trial_atoms["geo"]

    return out

_DEFAULT_ATOM_KEYS = [
    # keep as documentation; not used directly anymore
    "age", "sex", "diagnoses", "stage", "performance",
    "biomarkers", "prior_therapies", "lab_thresholds",
    "comorbid_exclusions", "pregnancy_required", "geo"
]

def _safe_clip_list(x, max_len=8):
    if isinstance(x, list) and len(x) > max_len:
        return x[:max_len]
    return x

def build_compact_atoms(trial_atoms: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """
    Normalize extractor schema to compact atoms, clip long lists, and keep short quotes.
    """
    norm = _normalize_atoms(trial_atoms)
    # clip a few potentially long lists
    if "biomarkers" in norm:
        for k in ("require", "exclude"):
            if isinstance(norm["biomarkers"].get(k), list):
                norm["biomarkers"][k] = _safe_clip_list(norm["biomarkers"][k], 10)
    if "prior_therapies" in norm:
        for k in ("allow", "disallow"):
            if isinstance(norm["prior_therapies"].get(k), list):
                norm["prior_therapies"][k] = _safe_clip_list(norm["prior_therapies"][k], 10)
    if "comorbid_exclusions" in norm:
        norm["comorbid_exclusions"] = _safe_clip_list(norm["comorbid_exclusions"], 10)
    if "lab_thresholds" in norm and isinstance(norm["lab_thresholds"], list):
        norm["lab_thresholds"] = norm["lab_thresholds"][:6]

    # short quotes for provenance (if your extractor provides them under free_text_spans)
    quotes = None
    spans = trial_atoms.get("free_text_spans") or {}
    if isinstance(spans, dict) and spans:
        keep = {}
        for k in ("biomarkers", "washout", "performance", "lab_thresholds"):
            if k in spans and spans[k]:
                s = str(spans[k]).strip()
                keep[k] = s[:220]
        quotes = keep or None

    return norm, quotes


# ==============================
# FastAPI
# ==============================
app = FastAPI(title="TrialMatcher Search API (BM25 + Vector + Hybrid + Practicality)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      #fine for our local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_db_probe():
    try:
        _ = pg_conn()
    except Exception:
        pass

@app.get("/health")
def health():
    try:
        r = session.get(f"{ES_URL}/_cluster/health", timeout=5); r.raise_for_status()
        bm25_ok = session.get(f"{ES_URL}/{BM25_INDEX}", timeout=5).status_code in (200, 201)
        vec_ok = session.get(f"{ES_URL}/{VECTOR_INDEX}", timeout=5).status_code in (200, 201)
        try:
            with pg_conn().cursor() as c:
                c.execute("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False
        return {"ok": True, "bm25_index": bm25_ok, "vector_index": vec_ok, "db": db_ok}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# --------- Existing endpoints ---------
@app.post("/search/bm25", response_model=BM25Response)
def search_bm25(body: BM25Request):
    try:
        hits = es_bm25(body.patient_text, body.top_k, body.filters)
        return BM25Response(results=[TrialHit(**h) for h in hits])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BM25 error: {e}")

@app.post("/search/vector", response_model=VectorResponse)
def search_vector(body: VectorRequest):
    try:
        qv = embed_text(body.patient_text)
        hits = es_vector_knn(qv, body.top_k, body.filters)
        return VectorResponse(results=[TrialHit(**h) for h in hits])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector error: {e}")

@app.post("/search/hybrid", response_model=HybridResponse)
def search_hybrid(body: BM25Request):
    try:
        bm25_hits = es_bm25(body.patient_text, max(body.top_k, 50), body.filters)
        qv = embed_text(body.patient_text)
        vec_hits  = es_vector_knn(qv, max(body.top_k, 50), body.filters)
        fused = fuse_rrf(bm25_hits, vec_hits, max(body.top_k, 50))

        out: List[HybridHit] = []
        for tid, s_rrf, s_b, s_v, meta in fused[:body.top_k]:
            out.append(HybridHit(
                trial_id=tid,
                title=meta.get("title"),
                status=meta.get("status"),
                phase=meta.get("phase"),
                conditions=meta.get("conditions"),
                scores=HybridScore(bm25=s_b, vector=s_v, rrf=s_rrf)
            ))
        return HybridResponse(results=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid error: {e}")

# --------- Unified endpoint with Practicality + Final ---------
@app.post("/search", response_model=UnifiedResponse)
def unified_search(body: UnifiedSearchRequest):
    try:
        # 1) Get candidate set via RRF hybrid
        bm25_hits = es_bm25(body.patient_text, max(body.top_k, 50), body.filters)
        qv = embed_text(body.patient_text)
        vec_hits  = es_vector_knn(qv, max(body.top_k, 50), body.filters)
        fused = fuse_rrf(bm25_hits, vec_hits, max(body.top_k, 50))

        # 2) Normalize hybrid to [0,1]
        hybrid_vals = [s_rrf for _, s_rrf, _, _, _ in fused]
        hybrid_norms = minmax_norm(hybrid_vals)
        tid_to_hn = {tid: hn for (tid, _, _, _, _), hn in zip(fused, hybrid_norms)}

        # 3) Practicality (distance + status + phase)
        plat = body.patient_location
        plat_lat = plat.lat if plat and (plat.lat is not None) else None
        plat_lon = plat.lon if plat and (plat.lon is not None) else None

        results: List[UnifiedHit] = []
        for tid, s_rrf, s_b, s_v, meta in fused:
            d_km, top_site = min_site_distance_and_top_site_db(tid, plat_lat, plat_lon)
            s_status = status_score(meta.get("status"))
            s_phase  = phase_score(meta.get("phase"))
            s_dist   = distance_score_km(d_km)
            practicality = (s_status + s_phase + s_dist) / 3.0

            hybrid_norm = tid_to_hn[tid]
            eligibility = hybrid_norm  # v0 placeholder

            final = FINAL_ALPHA*eligibility + FINAL_BETA*hybrid_norm + FINAL_GAMMA*practicality

            results.append(UnifiedHit(
                trial_id=tid,
                title=meta.get("title"),
                status=meta.get("status"),
                phase=meta.get("phase"),
                conditions=meta.get("conditions"),
                scores={
                    "hybrid": round(s_rrf, 12),
                    "practicality": round(practicality, 6),
                    "eligibility": round(eligibility, 6),
                    "final": round(final, 6),
                },
                top_site=top_site
            ))

        results.sort(key=lambda h: h.scores["final"], reverse=True)
        return UnifiedResponse(results=results[:body.top_k])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified search error: {e}")

# ==============================
# NEW: Eligibility atoms extraction endpoint
# ==============================
@app.post("/extract/eligibility-atoms", tags=["eligibility"])
def extract_eligibility_atoms(body: EligibilityTextIn) -> Dict[str, Any]:
    """
    Normalize + split the raw eligibility text and prefill rule-based atoms.
    Returns an integration-ready payload with:
      - schema (JSON Schema for TrialAtoms)
      - context: normalized_text, inclusion/exclusion items,
                 rule_extraction_atoms (prefill), provenance (line-level sources)
    A downstream model can use `schema` for function-calling and complete/correct the atoms.
    """
    try:
        return extract_atoms_from_eligibility(body.eligibility_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eligibility extraction error: {e}")

# -- helper: fetch raw eligibility text from DB (tolerant) --
def _fetch_elig_text_db(nct_id: str) -> Optional[str]:
    sql = """
      SELECT COALESCE(inclusion_text,'') || ' ' || COALESCE(exclusion_text,'') AS t
      FROM public.eligibility_text
      WHERE nct_id = %s
    """
    try:
        with pg_conn().cursor() as cur:
            cur.execute(sql, (nct_id,))
            row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None

# -- v2 request/response models --
class HybridV2Request(BaseModel):
    patient_text: str
    top_k: int = 10
    filters: Optional[Filters] = None
    include_evidence: bool = True            # inclusion/exclusion bullets
    include_eligibility_text: bool = False   # raw trial text (bigger payload)

class HybridV2Hit(BaseModel):
    trial_id: str
    title: Optional[str] = None
    status: Optional[str] = None
    phase: Optional[str] = None
    conditions: Optional[str] = None
    scores: HybridScore
    trial_atoms: Dict[str, Any]              # extracted rule-based atoms JSON (full)
    evidence: Optional[Dict[str, Any]] = None
    eligibility_text: Optional[str] = None

# --------- Data-only handoff models (generic, production-friendly names) ---------
class IntegrationTrialInput(BaseModel):
    trial_id: str
    title: Optional[str] = None
    atoms_subset: Dict[str, Any]
    quotes: Optional[Dict[str, str]] = None
    evidence: Optional[Dict[str, Any]] = None     # short inclusion/exclusion lists
    retrieval: Optional[HybridScore] = None       # optional transparency

class IntegrationPayload(BaseModel):
    patient_text: str
    trials: List[IntegrationTrialInput]

class HybridV2Response(BaseModel):
    schema: Dict[str, Any]                   # JSON Schema for TrialAtoms (for reference)
    patient_text: str
    results: List[HybridV2Hit]               # full envelope for UI/debugging
    integration_data: IntegrationPayload     # compact data-only blob for model integration

# -- v2: Search + atoms + data-only handoff --
@app.post("/search/hybrid_v2", response_model=HybridV2Response, tags=["search"])
def search_hybrid_v2(body: HybridV2Request):
    """
    ONE-CALL integration handoff:
      0) Warm DB connection (non-fatal).
      1) Hybrid retrieval (BM25 + Vector -> RRF).
      2) For each trial: fetch eligibility_text from DB (if available), extract TrialAtoms in-process.
      3) Return:
         - schema            : JSON Schema for TrialAtoms (reference only)
         - patient_text      : raw patient text
         - results           : per-trial envelope (full atoms/evidence for UI)
         - integration_data  : compact, bounded JSON for downstream model integration
    """
    try:
        # 0) Establish DB connection up front (non-fatal)
        db_ok = True
        try:
            _ = pg_conn()
        except Exception:
            db_ok = False

        # 1) Retrieval
        pool_k = max(body.top_k, 50)
        bm25_hits = es_bm25(body.patient_text, pool_k, body.filters)
        qv = embed_text(body.patient_text)
        vec_hits = es_vector_knn(qv, pool_k, body.filters)
        fused = fuse_rrf(bm25_hits, vec_hits, pool_k)

        results: List[HybridV2Hit] = []
        schema: Dict[str, Any] = {}

        # Build compact integration payload
        integration_trials: List[IntegrationTrialInput] = []

        # 2) Per-trial extraction
        for tid, s_rrf, s_b, s_v, meta in fused[:body.top_k]:
            elig_text: Optional[str] = None
            trial_atoms: Dict[str, Any] = {}
            evidence: Optional[Dict[str, Any]] = None

            if db_ok:
                elig_text = _fetch_elig_text_db(tid)
                if elig_text:
                    payload = extract_atoms_from_eligibility(elig_text)
                    if not schema:
                        schema = payload.get("schema", {}) or {}
                    ctx = payload.get("context", {}) or {}
                    trial_atoms = ctx.get("rule_extraction_atoms", {}) or {}

                    if body.include_evidence:
                        evidence = {
                            "inclusion_items": (ctx.get("inclusion_items", []) or [])[:10],
                            "exclusion_items": (ctx.get("exclusion_items", []) or [])[:10]
                        }

            # Compact atoms + short quotes for downstream model input
            subset, quotes = build_compact_atoms(trial_atoms)

            # Full envelope (for UI/debug)
            results.append(HybridV2Hit(
                trial_id=tid,
                title=meta.get("title"),
                status=meta.get("status"),
                phase=meta.get("phase"),
                conditions=meta.get("conditions"),
                scores=HybridScore(bm25=s_b, vector=s_v, rrf=s_rrf),
                trial_atoms=trial_atoms,
                evidence=evidence,
                eligibility_text=(elig_text if (db_ok and body.include_eligibility_text) else None),
            ))

            # Data-only item
            integration_trials.append(IntegrationTrialInput(
                trial_id=tid,
                title=meta.get("title"),
                atoms_subset=subset,
                quotes=quotes,
                evidence=evidence,
                retrieval=HybridScore(bm25=s_b, vector=s_v, rrf=s_rrf)
            ))

        # 3) Ensure we return a schema even if no trial had text
        if not schema:
            try:
                dummy = extract_atoms_from_eligibility("Inclusion:\n- Age 18+")
                schema = dummy.get("schema", {}) or {}
            except Exception:
                schema = {}

        # 4) Assemble data-only payload
        integration_data = IntegrationPayload(
            patient_text=body.patient_text,
            trials=integration_trials
        )

        return HybridV2Response(
            schema=schema,
            patient_text=body.patient_text,
            results=results,
            integration_data=integration_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid_v2 error: {e}")

# ==============================
# Also New: LLM Ranking Endpoint
# ==============================

@app.post("/llm/rank", tags=["llm"])
def rank_with_llm(payload: dict = Body(...)):
    """
    Accepts either direct integration_data or the full /search/hybrid_v2 response.
    """
    try:
        integration_data = payload.get("integration_data") or payload
        if not isinstance(integration_data, dict) or "trials" not in integration_data:
            raise HTTPException(status_code=400, detail="integration_data.trials required")

        return llm_rank_trials(integration_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM ranking error: {e}")

# ==============================
# Also New: LLM Health Check Endpoint
# ==============================

@app.get("/llm/health", tags=["llm"])
def llm_health():
    """
    Health check for LLM ranking subsystem.
    Returns current provider, model, and status info.
    """
    provider = os.getenv("LLM_PROVIDER", "LOCAL_HEURISTIC").upper()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini" if provider == "OPENAI" else "llama3")
    api_key = os.getenv("OPENAI_API_KEY")

    status = "ok"
    details = {"provider": provider, "model": model}

    if provider == "OPENAI":
        details["key_loaded"] = bool(api_key)
        if not api_key:
            status = "warning"
    elif provider == "OLLAMA":
        details["endpoint"] = "http://localhost:11434"
    else:
        details["note"] = "Using local heuristic mode (no external LLM)."

    return JSONResponse(
        content={"status": status, "details": details},
        status_code=200
    )

# ==============================
# Also New: Root redirect or simple ping
# ==============================

@app.get("/", include_in_schema=False)
def root():
    #Option 1: redirect users to Swagger docs
    return RedirectResponse(url="/docs")

    #Option 2 (comment out redirect above if we prefer a JSON message instead)
    #return JSONResponse({"status": "ok", "message": "TrialMatcher API running"})
