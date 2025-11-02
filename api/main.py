# api/main.py
import os
import math
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests

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
FINAL_ALPHA = float(os.getenv("FINAL_ALPHA", "0.0"))  # pre-LLM, keep 0
FINAL_BETA  = float(os.getenv("FINAL_BETA",  "0.7"))
FINAL_GAMMA = float(os.getenv("FINAL_GAMMA", "0.3"))

# Database (for trial_locations)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres.jgvhlbpjohghmdtatldc:CSCI544@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
)

# ==============================
# Sessions / Clients
# ==============================
# ES HTTP session
session = requests.Session()
session.auth = (ES_USER, ES_PASS)
session.verify = False
session.headers.update({"Content-Type": "application/json"})

# Lazy Postgres connection (psycopg3)
_pg_conn = None
def pg_conn():
    global _pg_conn
    if _pg_conn is None:
        _pg_conn = psycopg.connect(DATABASE_URL)
    return _pg_conn

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
    # Use OpenSearch/ES 8.x kNN with optional filter
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
# FastAPI
# ==============================
app = FastAPI(title="TrialMatcher Search API (BM25 + Vector + Hybrid + Practicality)", version="2.0.0")

@app.get("/health")
def health():
    try:
        r = session.get(f"{ES_URL}/_cluster/health", timeout=5); r.raise_for_status()
        bm25_ok = session.get(f"{ES_URL}/{BM25_INDEX}", timeout=5).status_code in (200, 201)
        vec_ok = session.get(f"{ES_URL}/{VECTOR_INDEX}", timeout=5).status_code in (200, 201)
        # DB check (shallow)
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
            # distance + top site (from Postgres trial_locations)
            d_km, top_site = min_site_distance_and_top_site_db(tid, plat_lat, plat_lon)
            s_status = status_score(meta.get("status"))
            s_phase  = phase_score(meta.get("phase"))
            s_dist   = distance_score_km(d_km)
            practicality = (s_status + s_phase + s_dist) / 3.0

            # v0 eligibility
            hybrid_norm = tid_to_hn[tid]
            eligibility = hybrid_norm

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

        # 4) Sort by final and return top_k
        results.sort(key=lambda h: h.scores["final"], reverse=True)
        return UnifiedResponse(results=results[:body.top_k])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified search error: {e}")
